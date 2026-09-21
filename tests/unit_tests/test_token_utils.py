"""消息 token 估算工具的离线单元测试."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent.utils.token_utils import count_text_tokens, estimate_messages_tokens


def test_count_empty_text_is_zero() -> None:
    """空字符串计 0 token。"""
    assert count_text_tokens("") == 0


def test_count_text_positive_for_cjk_and_latin() -> None:
    """中英文文本都能离线编码出正数 token。"""
    assert count_text_tokens("StateGraph 条件边") > 0
    assert count_text_tokens("hello world") >= 2


def test_estimate_includes_per_message_overhead() -> None:
    """总 token = 正文 token + 每条消息固定开销。"""
    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好，有什么可以帮你"),
    ]

    expected = (
        count_text_tokens("你好") + count_text_tokens("你好，有什么可以帮你") + 4 * 2
    )
    assert estimate_messages_tokens(messages) == expected


def test_estimate_skips_non_message_objects() -> None:
    """非 BaseMessage 对象不计入。"""
    messages: list[Any] = ["字符串", HumanMessage(content="问题")]

    assert estimate_messages_tokens(messages) == count_text_tokens("问题") + 4
