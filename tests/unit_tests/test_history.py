"""对话历史渲染工具的离线单元测试."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.utils.history import render_messages


def test_render_empty() -> None:
    """无历史消息时渲染为占位符。"""
    assert render_messages([], limit=6) == "（无）"


def test_render_roles_and_strip() -> None:
    """human/ai 消息渲染为「用户/助手」文本并去除首尾空白。"""
    text = render_messages(
        [HumanMessage(content="  StateGraph 是什么 "), AIMessage(content="是状态图")],
        limit=6,
    )
    assert text == "用户：StateGraph 是什么\n助手：是状态图"


def test_render_keeps_only_last_k() -> None:
    """历史超过上限时只保留最近 k 条。"""
    messages = [HumanMessage(content=f"问题{i}") for i in range(10)]
    text = render_messages(messages, limit=4)

    assert "问题0" not in text
    assert "问题9" in text
    assert text.count("用户：") == 4


def test_render_system_and_tool_have_own_labels() -> None:
    """system/tool 消息使用各自角色标签，不冒充助手。"""
    messages = [
        SystemMessage(content="遵守规则"),
        ToolMessage(content="工具返回", tool_call_id="call-1"),
    ]

    text = render_messages(messages, limit=6)

    assert text == "系统：遵守规则\n工具：工具返回"


def test_render_nonpositive_limit_returns_placeholder() -> None:
    """limit 为 0 或负数时直接返回占位符，不渲染任何消息。"""
    messages = [HumanMessage(content="问题")]

    assert render_messages(messages, limit=0) == "（无）"
    assert render_messages(messages, limit=-1) == "（无）"


def test_render_skips_non_message_and_non_text() -> None:
    """非 BaseMessage 对象与空文本消息被跳过。"""
    messages: list[Any] = [
        HumanMessage(content="   "),
        HumanMessage(content="有效问题"),
        "不是消息对象",
    ]

    assert render_messages(messages, limit=6) == "用户：有效问题"
