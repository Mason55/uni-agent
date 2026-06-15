import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_anthropic_payload_to_openai_basic():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "model": "default",
            "system": "Be concise.",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hello"},
    ]
    assert converted["model"] == "default"
    assert converted["max_tokens"] == 16


def test_anthropic_payload_accepts_system_role_message():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "hello"},
            ],
        }
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "hello"},
    ]


def test_anthropic_payload_moves_all_system_content_to_front():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "system": "Top system.",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": [{"type": "text", "text": "Late system."}]},
                {"role": "assistant", "content": "ok"},
            ],
        }
    )
    assert converted["messages"] == [
        {"role": "system", "content": "Top system.\nLate system."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ok"},
    ]


def test_anthropic_tools_are_normalized_for_qwen_tool_parser():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    converted = module.anthropic_payload_to_openai(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "TaskUpdate",
                    "description": "Update task state",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "description": "New status",
                                "anyOf": [
                                    {"type": "string", "enum": ["pending", "completed"]},
                                    {"type": "string", "const": "deleted"},
                                ],
                            },
                            "metadata": {
                                "description": "Free-form metadata",
                                "type": "object",
                                "additionalProperties": {},
                            },
                        },
                        "required": ["status"],
                    },
                }
            ],
        }
    )
    tool = converted["tools"][0]
    assert tool == {
        "type": "function",
        "function": {
            "name": "TaskUpdate",
            "description": "Update task state",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "New status",
                        "enum": ["pending", "completed", "deleted"],
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Free-form metadata",
                    },
                },
                "required": ["status"],
            },
        },
    }


def test_openai_completion_to_anthropic_tool_use():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    result = module.openai_completion_to_anthropic_message(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "edit", "arguments": '{"path": "a.py"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7},
        },
        request_model="default",
    )
    assert result["stop_reason"] == "tool_use"
    assert result["content"] == [{"type": "tool_use", "id": "call_1", "name": "edit", "input": {"path": "a.py"}}]
    assert result["usage"] == {"input_tokens": 5, "output_tokens": 7}


def test_anthropic_stream_events_for_text():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "default",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    events = module.anthropic_stream_events(message)
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0]["message"]["content"] == []
    assert events[2]["delta"] == {"type": "text_delta", "text": "hello"}
    encoded = module.encode_anthropic_sse_event(events[2])
    assert encoded.startswith("event: content_block_delta\n")
    assert encoded.endswith("\n\n")


def test_anthropic_stream_events_for_tool_use():
    module = _load_module("anthropic_compat_test", "uni_agent/trainer/gateway/anthropic_compat.py")

    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "default",
        "content": [{"type": "tool_use", "id": "call_1", "name": "edit", "input": {"path": "a.py"}}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    events = module.anthropic_stream_events(message)
    assert events[1]["content_block"] == {"type": "tool_use", "id": "call_1", "name": "edit", "input": {}}
    assert events[2]["delta"]["type"] == "input_json_delta"
    assert events[2]["delta"]["partial_json"] == '{"path": "a.py"}'


def test_claude_command_uses_anthropic_base_url_and_no_proxy():
    module = _load_module("claude_code_runner_test", "examples/swe_agent_blackbox/claude_code_runner.py")

    cmd = module.build_claude_command(
        task="fix bug",
        base_url="http://127.0.0.1:38197/sessions/s",
        max_turns=3,
    )
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:38197/sessions/s" in cmd
    assert "ANTHROPIC_API_KEY=not-needed" in cmd
    assert "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy" in cmd
    assert "cd /testbed" in cmd
    assert "CONDA_DEFAULT_ENV=testbed" in cmd
    assert "CONDA_PREFIX=/opt/miniconda3/envs/testbed" in cmd
    assert "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH" in cmd
    assert "/opt/claude-code/bin/claude -p 'fix bug'" in cmd
    assert "--max-turns 3" in cmd
    assert "--permission-mode bypassPermissions" in cmd
    assert "--settings" in cmd
    assert "autoCompactWindow" in cmd
    assert "--disable-slash-commands" in cmd
    assert "--disallowedTools WebFetch WebSearch" in cmd


def test_claude_task_rewrites_swe_prompt_for_claude_code():
    module = _load_module("claude_code_runner_test", "examples/swe_agent_blackbox/claude_code_runner.py")

    prompt = (
        "<issue_description>\n"
        "Fix nested separability.\n"
        "</issue_description>\n\n"
        "Follow these steps to resolve the issue:\n"
        "Create additional test cases in /testbed/edge_case_tests.py.\n"
        "Submit your solution using the submit tool."
    )
    task = module.build_claude_task(
        [{"role": "user", "content": prompt}],
        {
            "reward": {
                "metadata": {
                    "FAIL_TO_PASS": '["astropy/modeling/tests/test_separable.py::test_case"]',
                }
            }
        },
    )
    assert "Fix nested separability." in task
    assert "astropy/modeling/tests/test_separable.py::test_case" in task
    assert "There is no submit tool" in task
    assert "print a one-line summary and exit immediately" in task
    assert "Do not run additional ad-hoc verification" in task
    assert "Do not run `pytest --collect-only`, `git log`, or any other command" in task
    assert "Do not analyze unrelated `is_separable` behavior" in task
    assert "Create additional test cases" not in task


def test_rewrite_gateway_url_removes_v1_for_anthropic_sdk_base():
    module = _load_module("claude_sandbox_test", "examples/swe_agent_blackbox/sandbox/yr_sandbox.py")

    assert (
        module.rewrite_gateway_url("http://8.8.8.8:1234/sessions/abc/v1", strip_v1=True)
        == "http://127.0.0.1:38197/sessions/abc"
    )


def test_rewrite_gateway_url_keeps_v1_by_default_for_openai_gateway():
    module = _load_module("claude_sandbox_test", "examples/swe_agent_blackbox/sandbox/yr_sandbox.py")

    assert (
        module.rewrite_gateway_url("http://8.8.8.8:1234/sessions/abc/v1")
        == "http://127.0.0.1:38197/sessions/abc/v1"
    )
