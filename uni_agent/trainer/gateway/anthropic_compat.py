"""Anthropic Messages API <-> internal OpenAI-format conversion.

The gateway tracks sessions in the OpenAI Chat Completions message format
(the format consumed by chat templates and the prefix-comparison logic in
``gateway.py``).  This module converts an Anthropic ``/v1/messages`` request
into that internal format, and converts the gateway's chat-completion result
back into an Anthropic Message response.

Round-trip stability matters: when an Anthropic-API agent echoes a previous
assistant turn back (text + tool_use blocks) the conversion must reproduce
exactly the normalized OpenAI message the gateway stored in
``session.message_history``, otherwise the prefix check fails and the gateway
re-encodes the conversation as a new trajectory.  Both directions are kept
deterministic for that reason (text blocks joined with "\\n", tool arguments
serialized/parsed as JSON, which ``_canonicalize_tool_arguments_for_comparison``
treats as equivalent).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from uni_agent.trainer.gateway.types import MalformedRequestError

logger = logging.getLogger(__name__)

# Anthropic request fields copied through for _build_sampling_params.
# "stop_sequences" is renamed to OpenAI's "stop" (only honored when the
# gateway operator adds "stop" to allowed_request_sampling_param_keys).
_PASSTHROUGH_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens")

# OpenAI finish_reason -> Anthropic stop_reason.
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}

# FastAPI HTTP status -> Anthropic error.type.
ANTHROPIC_ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "invalid_request_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}


def _join_text_blocks(blocks: list[Any], *, context: str) -> str:
    texts = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            raise MalformedRequestError(f"{context} only supports text blocks, got: {block!r}")
        texts.append(str(block.get("text", "")))
    return "\n".join(texts)


def _convert_system(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return _join_text_blocks(system, context="system")
    raise MalformedRequestError("system must be a string or a list of text blocks")


def _image_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise MalformedRequestError("image block requires a source object")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        url = f"data:{media_type};base64,{source.get('data', '')}"
    elif source_type == "url":
        url = source.get("url", "")
    else:
        raise MalformedRequestError(f"unsupported image source type: {source_type!r}")
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_content_to_openai(content: Any) -> str | list[dict[str, Any]]:
    """Convert tool_result content (string or text/image blocks) to OpenAI tool content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise MalformedRequestError("tool_result content must be a string or a list of blocks")

    parts: list[dict[str, Any]] = []
    all_text = True
    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("tool_result content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif block_type == "image":
            parts.append(_image_block_to_openai_part(block))
            all_text = False
        else:
            raise MalformedRequestError(f"unsupported tool_result block type: {block_type!r}")
    if all_text:
        return "\n".join(part["text"] for part in parts)
    return parts


def _convert_user_message(content: Any) -> list[dict[str, Any]]:
    """Convert one Anthropic user message into OpenAI messages.

    tool_result blocks become individual ``role: "tool"`` messages (emitted
    first, directly after the preceding assistant tool_calls turn); the
    remaining text/image blocks form a trailing ``role: "user"`` message.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise MalformedRequestError("user message content must be a string or a list of blocks")

    tool_messages: list[dict[str, Any]] = []
    user_parts: list[dict[str, Any]] = []
    has_image = False
    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("message content blocks must be objects")
        block_type = block.get("type")
        if block_type == "tool_result":
            tool_use_id = block.get("tool_use_id")
            if not tool_use_id:
                raise MalformedRequestError("tool_result block requires tool_use_id")
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_use_id),
                    "content": _tool_result_content_to_openai(block.get("content")),
                }
            )
        elif block_type == "text":
            user_parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif block_type == "image":
            user_parts.append(_image_block_to_openai_part(block))
            has_image = True
        else:
            raise MalformedRequestError(f"unsupported user content block type: {block_type!r}")

    messages = list(tool_messages)
    if user_parts:
        if has_image:
            messages.append({"role": "user", "content": user_parts})
        else:
            messages.append({"role": "user", "content": "\n".join(part["text"] for part in user_parts)})
    return messages


def _convert_assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise MalformedRequestError("assistant message content must be a string or a list of blocks")

    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise MalformedRequestError("message content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            texts.append(str(block.get("text", "")))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"call_{uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        elif block_type in ("thinking", "redacted_thinking"):
            # Thinking blocks have no chat-template representation; the model's
            # reasoning tokens are already captured in the token trajectory.
            continue
        else:
            raise MalformedRequestError(f"unsupported assistant content block type: {block_type!r}")

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _convert_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise MalformedRequestError("tools must be a list")
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise MalformedRequestError("tools entries must be objects")
        tool_type = tool.get("type")
        if tool_type not in (None, "custom"):
            raise MalformedRequestError(
                f"unsupported tool type: {tool_type!r} (only client tools with input_schema are supported)"
            )
        if not tool.get("name"):
            raise MalformedRequestError("tool requires a name")
        function: dict[str, Any] = {
            "name": tool["name"],
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted


def anthropic_payload_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``/v1/messages`` request payload to the OpenAI
    Chat Completions payload shape consumed by the gateway."""
    if not isinstance(payload, dict):
        raise MalformedRequestError("request body must be a JSON object")
    if payload.get("stream"):
        raise MalformedRequestError("streaming is not supported by the gateway; set stream=False")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise MalformedRequestError("messages must be non-empty")

    messages: list[dict[str, Any]] = []
    if payload.get("system") is not None:
        messages.append({"role": "system", "content": _convert_system(payload["system"])})

    for message in raw_messages:
        if not isinstance(message, dict):
            raise MalformedRequestError("messages entries must be objects")
        role = message.get("role")
        if role == "user":
            messages.extend(_convert_user_message(message.get("content")))
        elif role == "assistant":
            messages.append(_convert_assistant_message(message.get("content")))
        else:
            raise MalformedRequestError(f"unsupported message role: {role!r}")

    openai_payload: dict[str, Any] = {"messages": messages}
    tools = _convert_tools(payload.get("tools"))
    if tools is not None:
        openai_payload["tools"] = tools
    for key in _PASSTHROUGH_SAMPLING_KEYS:
        if key in payload:
            openai_payload[key] = payload[key]
    if "stop_sequences" in payload:
        openai_payload["stop"] = payload["stop_sequences"]
    if "model" in payload:
        openai_payload["model"] = payload["model"]
    return openai_payload


def _tool_call_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            logger.warning("tool call arguments are not valid JSON; returning empty input: %.200s", arguments)
            return {}
        if isinstance(parsed, dict):
            return parsed
    logger.warning("tool call arguments are not a JSON object; returning empty input: %.200s", arguments)
    return {}


def openai_completion_to_anthropic_message(
    completion: dict[str, Any],
    *,
    request_model: str | None = None,
) -> dict[str, Any]:
    """Convert the gateway's chat-completion response dict to an Anthropic Message."""
    choice = completion["choices"][0]
    assistant_message = choice["message"]
    finish_reason = choice.get("finish_reason")

    content: list[dict[str, Any]] = []
    text = assistant_message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for tool_call in assistant_message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"call_{uuid4().hex[:8]}",
                "name": function.get("name", ""),
                "input": _tool_call_input(function.get("arguments")),
            }
        )

    usage = completion.get("usage") or {}
    return {
        "id": f"msg_{uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": request_model or completion.get("model") or "default",
        "content": content,
        "stop_reason": _STOP_REASON_MAP.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
    }


def anthropic_error_body(status_code: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": ANTHROPIC_ERROR_TYPE_BY_STATUS.get(status_code, "api_error"),
            "message": message,
        },
    }
