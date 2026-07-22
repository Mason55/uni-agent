# Gateway 如何处理 Harness 的上下文压缩与 Subagent

## 核心结论

Gateway **不实现**上下文压缩算法，也**不调度** subagent。这两类行为完全由外部 harness（如 Claude Code、Mini-SWE-Agent）自己决定。Gateway 的职责是：当 harness 发来的请求历史与已存轨迹的 prefix 不再匹配时，**把它识别为一条新的 trajectory chain**，从而保证训练数据的 token/mask 仍然正确。

换句话说：

- 压缩 = harness 改写了 history → prefix hash 变了 → Gateway 开新 chain。
- Subagent = harness 换了 system prompt / 切到子任务 history → prefix hash 变了 → Gateway 开新 chain。

两条路径走的是**同一套机制**，没有任何特判分支。

---

## 机制总览：Chain + Prefix Hash

### 数据结构

一个 `GatewaySession` 内部持有两类 chain（见 [trajectory_session.py](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/trajectory_session.py)）：

- `active_chains: list[ChainState]` —— 还在继续生成的链。
- `materialized_chains: list[MaterializedChain]` —— 已闭合、等待 finalize 写出的链。

每个 `ChainState` 关键字段：

| 字段 | 作用 |
| --- | --- |
| `chain_id` | 单调递增的唯一 id。 |
| `message_history` | 这条链已 commit 的归一化消息序列。 |
| `message_tip_hash` | 历史 prefix 的 SHA256 tip，用于 O(1) 前缀匹配。 |
| `active_tool_schemas` | 这条链生效的 tools schema，变更会强制开新链。 |
| `buffer: TrajectoryBuffer` | 真正的 `prompt_ids` / `response_ids` / `response_mask` / `response_logprobs` / `routed_experts`。 |

### Prefix Hash 计算

见 [trajectory_session.py#_extend_message_prefix_hashes](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/trajectory_session.py)，链式 SHA256：

```
prefix_hash[0] = sha256("uni-agent-prefix-v1\0" + EMPTY_PREFIX + "\0" + message_hash[0])
prefix_hash[i] = sha256("uni-agent-prefix-v1\0" + prefix_hash[i-1] + "\0" + message_hash[i])
```

其中 `message_hash` 由 [codec.py#canonicalize_message_for_prefix_comparison](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/codec.py) 归一化后 SHA256 得到。归一化会：

- 去掉 `tool_call_id`（不稳定）。
- 去掉 tool_call 的 `id`（uuid，不稳定）。
- 把 `function.arguments` 统一成 JSON 解析后的对象做比较（字符串/字典等价）。

**这就是 harness 上下文压缩能被正确识别的关键**：只要 harness 在压缩后重写的消息文本和原消息逐字相同，prefix 仍然匹配；只要 harness 真的改写了内容（如 summary 替换原文），hash 立刻不匹配，触发新 chain。

### Chain 选择规则

每次请求进来，[session.py#_select_chain](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py) 会从 `active_chains` 里筛候选：

1. 没被 reserve（避免并发同链竞争）。
2. `active_tool_schemas == 请求 tools`（tools 变了强制开新链）。
3. `_is_chain_prefix_hash_match` —— 历史 hash 是请求 prefix hash 的前缀。

命中则选 `max((len(history), updated_seq, chain_id))`，即**最长匹配、最近更新**的那条；不命中返回 `None`，进入“开新 chain”分支。

---

## 路径 A：Harness 上下文压缩

### 触发场景

Claude Code 等 harness 在 token 数累积到阈值时，会自己把早期对话替换成一段 summary，例如：

```python
# 第一次请求
[HELPFUL_SYS, {"role":"user","content":"produce a detailed answer"}]
# 压缩后第二次请求（system 已被替换为 summary）
[{"role":"system","content":"Summary so far: the detailed answer was compacted."},
 {"role":"user","content":"continue from the summary"}]
```

### Gateway 的处理

1. 第二次请求进来，`_extend_message_prefix_hashes` 算出新的 prefix hash 序列。
2. `_select_chain` 遍历 `active_chains`：
   - 旧 chain 的 `message_tip_hash` 是基于 `HELPFUL_SYS + "produce a detailed answer"` 算出的。
   - 新请求第一条消息变成了 summary 文本，`message_hash` 不同 → prefix hash 不同 → 不匹配。
3. `selected_chain is None`，走 [session.py#_prepare_generation_inputs](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py) 的“新 chain”分支：
   - `encode_full` 把整段新 history 重新编码成 `prompt_ids`。
   - `buffer = TrajectoryBuffer(prompt_ids=prompt_ids)`，`chain_id = None`。
4. 生成完成后 `_commit_generation_to_chain` 分配新 `chain_id`，append 到 `active_chains`。
5. Finalize 时两条 chain 各自物化成独立的 `Trajectory`。

### 训练语义

参考测试 [test_multiple_chains_context_compaction_starts_new_chain](file:///data1/lmy/JiuWen/uni-agent/tests/uni_agent/gateway/test_session_multiple_chains_on_cpu.py)：

- 两条 trajectory 完全独立：`prompt_ids` 各自是当次请求的完整编码。
- `response_mask` 全为 `1`（每条 chain 自己的 response），不存在跨 chain 的 `0` mask。
- 训练侧看到的是“两个独立的样本”，而不是一条被截断的长轨迹。

> 注意：这意味着压缩前的 token 不会和压缩后的 token 拼在同一个 `response_ids` 里。压缩把一条长轨迹切成了多条短轨迹，训练时各自计 loss。这是设计选择，不是 bug。

---

## 路径 B：Subagent

### 触发场景

Claude Code 在主对话过程中会切到 subagent 上下文（不同 system prompt、不同任务），完成后再切回主链。环境变量 `CLAUDE_CODE_SUBAGENT_MODEL` 把 subagent 的模型 slot 也指向 gateway（见 [claude_code/agent.py](file:///data1/lmy/JiuWen/uni-agent/uni_agent/agents/claude_code/agent.py)），所以 subagent 请求也会打到同一个 session URL。

### 典型时序

参考测试 [test_multiple_chains_subagent_system_split_returns_to_main_chain](file:///data1/lmy/JiuWen/uni-agent/tests/uni_agent/gateway/test_session_multiple_chains_on_cpu.py)：

```python
main_first      = [HELPFUL_SYS,    {"role":"user","content":"name a fruit"}]
subagent        = [SUBAGENT_SYS,   {"role":"user","content":"name a color"}]
main_continuation = [
    HELPFUL_SYS, {"role":"user","content":"name a fruit"},
    {"role":"assistant","content":"Mango"},
    {"role":"user","content":"name another fruit"},
]
await _run(session, backend, main_first)        # chain 1
await _run(session, backend, subagent)          # chain 2 (新 system → 新 chain)
await _run(session, backend, main_continuation) # 续 chain 1
```

### Gateway 的处理

1. **subagent 请求**：`SUBAGENT_SYS` 的 `message_hash` 与 `HELPFUL_SYS` 不同 → 第一条消息的 prefix hash 就不匹配 → 开新 chain（chain 2）。`buffer.prompt_ids` 是 subagent 完整 history 的编码。
2. **主链续写**：`main_continuation` 的前 3 条消息和 chain 1 的 `message_history` 完全一致（包括归一化后的 assistant 消息）→ prefix hash 匹配 → 选中 chain 1。
3. `_prepare_generation_inputs` 走“续 chain”分支：
   - 拷贝 chain 1 的 `TrajectoryBuffer`。
   - `incremental_messages = messages[len(selected_chain.message_history):]`，只编码新增的 `{"role":"user","content":"name another fruit"}`。
   - `encode_incremental` 会去掉 system prompt 前缀（见 [codec.py#encode_incremental](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/codec.py)），只追加增量 token。
   - 增量 token 以 `response_mask=0` 写入 buffer（表示不是模型生成的）。
4. 生成完成，新 response token 以 `response_mask=1` 追加，更新 chain 1 的 `message_tip_hash`。

### 训练语义

Finalize 后得到 2 条 trajectory：

- chain 2（subagent）：`response_ids` 只有 subagent 自己的回答，`response_mask` 全 1。
- chain 1（主链）：`response_ids = ["Mango"] + [增量 user 消息 token] + ["Apple"]`，其中 `"Mango"` 和 `"Apple"` 是 `mask=1`，增量 user 消息是 `mask=0`。

**关键点**：subagent 的 token 不会污染主链 trajectory，主链的 loss 只算在模型真正生成的 token 上。

---

## 并发与 Chain Reservation

### 为什么需要 reservation

同一个 session 内可能有并发请求（harness 并行发起多个 generation）。如果两个请求都选中了同一条 chain，会发生：

- 请求 A 拷贝 buffer → 后端生成 → commit 写回 chain。
- 请求 B 拷贝 buffer（此时还没 A 的 response）→ 后端生成 → commit 覆盖 chain。

后到的 commit 会丢掉先到的 response。Gateway 的处理见 [session.py#run_generation](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py)：

1. `_prepare_generation_inputs` 选定 `chain_id` 后，立刻调 `_reserve_chain(chain_id)` 把 id 加进 `_reserved_chain_ids`。
2. 后续并发的 `_select_chain` 看到 `_is_chain_reserved(chain_id) == True`，跳过这条 chain。
   - 如果它的 prefix 还能匹配别的 chain，就续别的。
   - 如果都不匹配，就开新 chain（变成 sibling）。
3. 生成完成后 commit，调 `_discard_chain_reservation` 释放。
4. `finally` 里用 `asyncio.shield(_release_chain_reservation(...))` 保证即便任务被 cancel 也能释放。

### Subagent 并发的典型例子

参考 [test_multiple_chains_parallel_different_chains_commit_in_place](file:///data1/lmy/JiuWen/uni-agent/tests/uni_agent/gateway/test_session_multiple_chains_on_cpu.py)：主链和 subagent 同时续写，各自命中 chain 1 和 chain 2，commit 时 in-place 更新各自 chain，互不干扰。

---

## Finalize 时的物化

`finalize` 见 [trajectory_session.py#finalize](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/trajectory_session.py)：

1. `_materialize_active_chains` 把所有还活着的 chain 转成 `MaterializedChain`，`order_seq` 取 chain 的 `updated_seq`。
2. 按 `order_seq` 排序，依次生成 `Trajectory` 列表。
3. 把 session 级 `reward_info` 浅拷到每条 trajectory 上。

所以压缩/subagent 产生的多条 chain，最终是**一个 session 输出多条 trajectory**，下游 TransferQueue 会写成多条记录（`{uid}_{session_index}_{trajectory_index}`），见 [gateway-and-trajectories.md](file:///data1/lmy/JiuWen/uni-agent/docs/source/concepts/gateway-and-trajectories.md)。

---

## 早期返回：length-exhausted

如果某条 chain 的 `response_mask` 已经达到 `response_length` 上限，`_prepare_generation_inputs` 会直接构造 `length_exhausted_trajectory` 并把 chain 移进 `materialized_chains`，不再调后端。这条 chain 的 `extra_fields.materialization_reason = "max_response_length"`，finish_reason 返回 `"length"`。压缩/subagent 场景里如果某条子链超长，会单独提前闭合，不影响其他 chain。

---

## 一张图

```
harness (Claude Code / Mini-SWE / ...)
   │  POST /sessions/{sid}/v1/messages
   │  payload.messages = 当前 history（可能被压缩 / 可能切到 subagent）
   ▼
_GatewayActor._handle_anthropic_messages
   │  anthropic_to_internal → InternalGenerationRequest{messages, tools, sampling_params}
   ▼
GatewaySession.run_generation
   │  async with request_lock:
   │    _prepare_generation_inputs(request)
   │      ├─ _extend_message_prefix_hashes(messages)
   │      ├─ _select_chain(tools, prefix_hashes)
   │      │    └─ 过滤 reserved / tools 不一致 / prefix 不匹配
   │      └─ selected_chain is None?
   │           ├─ Yes → encode_full → 新 buffer, chain_id=None
   │           └─ No  → copy buffer + encode_incremental → 续 chain_id
   │    _reserve_chain(chain_id)  # 防并发竞争
   ▼
backend.generate(prompt_ids, sampling_params, image_data, video_data)  # 锁外
   ▼
async with request_lock:
   codec.decode_response(response_ids, tools, stop_reason)
   _commit_generation_to_chain(encoded, assistant_msg)
     ├─ chain_id is None → append 新 ChainState
     └─ chain_id 存在    → in-place 替换该 chain
   _discard_chain_reservation(chain_id)
   ▼
GenerationOutcome{assistant_msg, finish_reason, prompt_tokens, completion_tokens}
   │
   ▼
adapter 序列化成 Anthropic / OpenAI 响应回给 harness
```

---

## 轨迹采集：Gateway 还做了什么

除了被动的 chain 切分，Gateway 在轨迹采集层面还承担了 token 级数据落盘的全部工作。下面按采集路径、字段、生命周期、契约四个维度梳理。

### 双采集路径

Gateway 同时支持两种采集入口，底层共用 `TrajectorySession` 状态机：

**路径 1：HTTP 同步路径**（[gateway.py](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/gateway.py) + [session.py#run_generation](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py)）

harness 通过 `/v1/chat/completions` 或 `/v1/messages` 打进来：

1. `_GatewayActor` 把 wire payload 经 adapter 降成 `InternalGenerationRequest{messages, tools, sampling_params}`。
2. `GatewaySession.run_generation` 在 `request_lock` 内调 `_prepare_generation_inputs` 生成 `EncodedData`。
3. **锁外**调 `backend.generate(prompt_ids, sampling_params, image_data, video_data)`。
4. 再回 `request_lock` 内调 `codec.decode_response` + `_commit_generation_to_chain`。
5. 返回 `GenerationOutcome{assistant_msg, finish_reason, prompt_tokens, completion_tokens}`，由 adapter 序列化成 OpenAI/Anthropic 响应。

**路径 2：Passive Capture 路径**（[trajectory_session.py#capture](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/trajectory_session.py)）

外部 rollout owner（非 gateway 自家 backend）调 `TrajectorySession.capture(request)` 拿一个 `_CaptureTransaction`：

```python
async with await session.capture(request) as tx:           # prepare 在锁内
    output = await external_backend.generate(tx.context_ids, tx.sampling_params, ...)
    receipt = await tx.commit(CapturedGeneration(...))     # commit 在锁内
```

`_CaptureTransaction` 的语义：

| 阶段 | 锁 | 行为 |
| --- | --- | --- |
| `capture()` | request_lock | prepare：选 chain、编码、reserve、返回 tx |
| `async with` / `.context_ids` 等属性 | 无锁 | 外部读取不可变上下文去生成 |
| `commit(CapturedGeneration)` | request_lock | 把外部生成的 token/mask/logprobs 写回 chain |
| `rollback()` / `__aexit__` | request_lock | 释放 chain reservation，不写任何状态 |

两条路径的差异：HTTP 路径由 `GatewaySession` 自己驱动 backend；Passive 路径把生成权完全交给外部，session 只管状态。`_commit_capture` 和 `_commit_generation_to_chain` 写回 chain 的逻辑等价。

### Token 级采集字段

`TrajectoryBuffer` 是真正的采集载体（见 [trajectory_session.py#TrajectoryBuffer](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/trajectory_session.py)）：

| 字段 | 采集规则 |
| --- | --- |
| `prompt_ids` | 首轮由 `encode_full` 生成；续轮从 selected_chain 拷贝，不变。Passive 路径在 commit 时若 `chain_id is None`，会用 `generation.prompt_ids` 覆盖（允许外部替换实际 prefilled 的 prompt）。 |
| `response_ids` | 模型生成 token 直接 extend；续轮的增量 user/tool 消息由 `encode_incremental` 编码后 extend，这些 token 是**上下文而非生成**。 |
| `response_mask` | 模型生成 token 写 `1`；增量 context token 写 `0`。最终 loss 只算在 `1` 上。 |
| `response_logprobs` | 仅当 `sampling_params["logprobs"] == True` 时采集。模型生成 token 写实际 logprob；增量 context token 写 `0.0` 占位以保证长度对齐。 |
| `routed_experts` | MoE 路由信息。后端返回的是**全 context 的路由**（prompt + response so far + new tokens），所以代码里是 `buffer.routed_experts = routed_experts` **替换**而不是 extend（见 [session.py#L226-L228](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py)）。框架侧在写 TransferQueue 时对齐到 input_ids。 |

物化时（`_build_materialized_trajectory`）：
- `response_logprobs` 只在长度等于 `response_ids` 时才写入 trajectory，否则置 `None`。
- `multi_modal_data` 由 chain 上累积的 `image_data` / `video_data` 重建。
- `num_turns` 按 message_history 里 user/assistant 消息数 +1 计算。
- `extra_fields` 保留物化原因（目前只有 `max_response_length`）。

### 采样参数合并与预算 clamp

采集前的采样参数经过三层合并（见 [gateway.py#_handle_openai_chat_completions](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/gateway.py) 与 [session.py#_prepare_generation_inputs](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py)）：

```
base_sampling_params (actor 配置, 可信)
  ← merge ← session.sampling_params (create_session 时传入, 可信)
  ← merge ← request.sampling_params (adapter 已按白名单过滤)
```

- **白名单**：`DEFAULT_ALLOWED_REQUEST_SAMPLING_KEYS = {"temperature", "top_p", "top_k", "max_tokens", "stop"}`，其余 key 会被 adapter 丢弃，防止 harness 注入训练不关心的字段。
- **max_tokens 校验**：`_validate_sampling_params` 强制 `max_tokens` 为正整数。
- **response budget clamp**：`remaining_response_budget = response_length - len(buffer.response_mask)`，`max_tokens = min(请求 max_tokens, remaining_response_budget)`。budget 用尽直接走 length-exhausted 分支，不调 backend。

### 多模态采集

[codec.py#extract_multi_modal_data](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/codec.py) 负责检测+抽取：

1. 扫描 messages 里是否有 `type in {image, image_url, video, video_url}` 的 content block，没有则返回 `(None, None)`，跳过多模态路径。
2. 有则调 `_vision_info_extractor`（默认 `qwen_vl_utils.process_vision_info`）抽出 `image_data` / `video_data`。
3. `encode_full` 把 text + image + video 一起喂给 `processor`，输出统一 `input_ids`。
4. 续轮的增量 multimodal 通过 `encode_incremental` 编码，并 extend 到 chain 的 `image_data` / `video_data` 列表。
5. Finalize 时 `_build_multi_modal_trajectory_data` 组装成 `{"images": [...], "videos": [...]}` 写入 `Trajectory.multi_modal_data`。

chain 切换时 `image_data` / `video_data` 通过 `_copy_media_list` 深拷贝，避免兄弟 chain 共享引用。

### Reward 采集

- HTTP endpoint `POST /sessions/{sid}/reward_info` → `set_reward_info`，在 session ACTIVE 时写入 `self.reward_info`（见 [trajectory_session.py#set_reward_info](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/trajectory_session.py)）。
- reward_info 是 **session 级**而非 chain 级：finalize 时用 `replace(trajectory, reward_info=dict(self.reward_info))` 浅拷到每条 trajectory。
- 下游训练侧（框架）负责把这个标量 reward 转成 sparse `rm_scores` 张量，放在最后一个 token 上。Gateway 自己不生成 rm_scores。
- 没报 reward 也不阻塞 finalize，下游会落 0 并打 warning。

### 生命周期阶段保护

`SessionPhase` 三态机（见 [trajectory_session.py](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/trajectory_session.py)）：

```
ACTIVE ──finalize──▶ FINALIZED
   │                    
   └──abort──▶ ABORTED (不可 finalize)
```

- `capture` / `_commit_capture` / `set_reward_info` / `run_generation` 入口都校验 `phase == ACTIVE`，否则抛 `SessionLifecycleError`（HTTP 路径转成 409）。
- `abort` 把 `active_chains` / `materialized_chains` / `_reserved_chain_ids` 全清空，已采集的轨迹**直接丢弃**，不会输出。
- `finalize` 幂等性：ABORTED 不能 finalize（抛错），FINALIZED 重复调也抛错。
- `_materialize_active_chains` 把所有未闭合的 active chain 强制物化（用于 session 自然结束时还没显式关闭的 chain）。

### 不可变契约（防外部篡改 session 状态）

`CapturedGeneration` 和 `CaptureReceipt` 都是 `@dataclass(frozen=True)`（见 [types.py](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/types.py)），`__post_init__` 里：

- `Mapping` → `MappingProxyType`（只读视图）。
- `list` / `tuple` → `tuple`。
- `set` / `frozenset` → `frozenset`。
- 其余 `deepcopy`。

`mutable_capture_value` 是反操作：commit 时把外部传入的 frozen 容器再转回 mutable dict/list，让 session 内部可以修改。

这层契约保证了 passive capture 路径里外部 rollout owner 拿到的 receipt 无法回写 session 状态。

### Token 对齐校验

Gateway 在采集过程中多处做长度对齐校验，失败直接 RuntimeError：

- `output.log_probs` 长度必须等于 `token_ids` 长度（[session.py#L215-L219](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py)）。
- `response_logprobs` 和 `response_ids` 长度对齐时才物化进 trajectory，否则 trajectory 的 `response_logprobs = None`。
- `_extend_message_prefix_hashes` 后 `assert len(message_prefix_hashes) == len(message_history)`。
- `_commit_generation_to_chain` 续 chain 时 `chain_index` 必须能在 `active_chains` 里找到，否则 `RuntimeError("active chain {id} not found")`。

### turn_id 计算

用于 CaptureReceipt 里回报给外部的轮次编号：

- 新 chain：`turn_id = 1`。
- 续 chain：`turn_id = sum(1 for m in chain.message_history if m["role"] == "assistant") + 1`。

注意计算的是 **assistant 消息数 +1**，不是总消息数。即一个 user→assistant→user 的 chain 续写时 turn_id=2。

### 后端错误转换

`run_generation` 里 backend 异常的转换规则（[session.py#L186-L190](file:///data1/lmy/JiuWen/uni-agent/uni_agent/gateway/session/session.py)）：

| 异常类型 | HTTP 状态码 | 说明 |
| --- | --- | --- |
| `ValueError` | 400 | backend 拒绝请求（如 prompt 过长、sampling 非法） |
| 其他 `Exception` | 500 | 包装成 `"{Class}: {msg}"` |
| phase != ACTIVE | 409 | session 已 finalize/abort |

### 采集流程全景

```
┌─ HTTP 路径 ──────────────────────────────────────────────────────┐
│ harness                                                         │
│   │ POST /v1/messages                                           │
│   ▼                                                             │
│ adapter: anthropic_to_internal / openai_to_internal             │
│   │ InternalGenerationRequest                                   │
│   ▼                                                             │
│ GatewaySession.run_generation                                   │
│   ├─ [lock] _prepare_generation_inputs                          │
│   │    ├─ prefix hash 计算                                      │
│   │    ├─ _select_chain → 选/不选                               │
│   │    ├─ encode_full / encode_incremental + multimodal         │
│   │    ├─ max_tokens budget clamp                               │
│   │    └─ _reserve_chain (防并发)                               │
│   ├─ backend.generate(prompt_ids, sampling, image, video)       │
│   ├─ [lock] decode_response + _commit_generation_to_chain       │
│   │    ├─ buffer.response_ids.extend(token_ids)                 │
│   │    ├─ buffer.response_mask.extend([1]*n)                    │
│   │    ├─ buffer.response_logprobs.extend(logprobs) (可选)      │
│   │    ├─ buffer.routed_experts = output.routed_experts (替换)  │
│   │    └─ _discard_chain_reservation                            │
│   └─ return GenerationOutcome                                   │
└─────────────────────────────────────────────────────────────────┘

┌─ Passive 路径 ───────────────────────────────────────────────────┐
│ external owner                                                  │
│   │ tx = await session.capture(request)                         │
│   │   └─ [lock] prepare → _PreparedCapture + _reserve_chain    │
│   ▼                                                             │
│ async with tx:                                                  │
│   output = await external_backend.generate(tx.context_ids, ...) │
│   receipt = await tx.commit(CapturedGeneration(                 │
│       assistant_message, prompt_ids, completion_ids,            │
│       completion_logprobs, stop_reason,                         │
│       routed_experts, routing_metadata))                        │
│   │   └─ [lock] _commit_capture → 写回 chain buffer             │
│   ▼                                                             │
│ CaptureReceipt (frozen)                                         │
└─────────────────────────────────────────────────────────────────┘

┌─ Finalize ───────────────────────────────────────────────────────┐
│ finalize()                                                      │
│   ├─ [lock] phase check (ACTIVE only)                           │
│   ├─ _materialize_active_chains (强制闭合未关 chain)            │
│   ├─ sort materialized_chains by order_seq                      │
│   ├─ replace(t, reward_info=...) 把 reward 浅拷到每条 trajectory│
│   └─ phase = FINALIZED                                          │
│ → list[Trajectory]                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 设计要点小结

1. **Gateway 对压缩/subagent 是无感的**：它只看到“请求的 prefix hash 是否匹配某条 active chain”。
2. **不匹配就开新 chain**，新 chain 有自己独立的 `prompt_ids`，不会和旧 chain 的 token 互相污染。
3. **匹配就续 chain**，只编码增量消息，增量 token 标 `mask=0`，模型生成的标 `mask=1`。
4. **并发同链通过 `_reserved_chain_ids` 串行化**，避免 commit 覆盖。
5. **Finalize 输出多条 trajectory**，每条带 session 级 reward_info，下游训练侧独立消费。
6. **tools schema 变更也走同一机制**：`active_tool_schemas != 请求 tools` 直接判为不匹配，强制开新 chain。

如果想加真正的“主动压缩”（比如 gateway 自己 summarize 后塞回 chain），目前代码里**没有这个 hook**，需要新增一层 chain-rewrite 逻辑并保证 `message_tip_hash` 同步更新。
