"""对话历史压缩节点的离线单元测试.

用 RunnableLambda 伪造摘要模型，验证条数/token 双触发条件、旧消息
RemoveMessage 删除（经真实 add_messages reducer 验证）、摘要参数注入、
以及「摘要失败不动历史」的安全保护，全程不联网。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langgraph.graph import add_messages

from agent.config import Config
from agent.nodes import summarize
from agent.nodes.summarize import node_summarize, should_summarize


@pytest.fixture(autouse=True)
def _isolate_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, Any]]:
    """固定压缩相关 Config 并注入假摘要模型，测试前后清缓存。"""
    monkeypatch.setattr(Config, "SUMMARY_TRIGGER_MESSAGES", 10)
    monkeypatch.setattr(Config, "SUMMARY_TRIGGER_TOKENS", 4000)
    monkeypatch.setattr(Config, "SUMMARY_COMPRESS_RATIO", 0.5)
    monkeypatch.setattr(Config, "SUMMARY_MAX_CHARS", 300)
    monkeypatch.setattr(Config, "SUMMARY_MAX_ITEMS", 8)

    seen: list[dict[str, Any]] = []
    state: dict[str, Any] = {"output": "合并后的新摘要"}

    def _fake_invoke(messages: list[Any]) -> AIMessage:
        seen.append({"messages": messages})
        return AIMessage(content=state["output"])

    monkeypatch.setattr(
        summarize,
        "create_chat_model",
        lambda: RunnableLambda(_fake_invoke),
    )
    summarize._get_summary_chain.cache_clear()
    bundle: dict[str, Any] = {"seen": seen, "state": state}
    yield bundle
    summarize._get_summary_chain.cache_clear()


def _messages(n: int) -> list[Any]:
    """构造 n 条带稳定 id 的 Human/AI 交替消息。"""
    messages: list[Any] = []
    for i in range(n):
        messages.append(HumanMessage(content=f"问题{i}", id=f"h{i}"))
        messages.append(AIMessage(content=f"回答{i}", id=f"a{i}"))
    return messages


# ---------------- should_summarize ----------------


def test_should_summarize_false_when_under_limits() -> None:
    """条数与 token 都未超阈值时不压缩。"""
    state: dict[str, Any] = {"messages": _messages(4)}  # 8 条 < 10

    assert should_summarize(state) is False  # type: ignore[arg-type]


def test_should_summarize_true_when_message_count_exceeds_trigger() -> None:
    """消息条数超过 SUMMARY_TRIGGER_MESSAGES 时触发。"""
    state: dict[str, Any] = {"messages": _messages(6)}  # 12 条 > 10

    assert should_summarize(state) is True  # type: ignore[arg-type]


def test_should_summarize_true_when_token_threshold_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """条数未超但 token 超阈值时同样触发（OR 条件）。"""
    monkeypatch.setattr(Config, "SUMMARY_TRIGGER_TOKENS", 1)
    state: dict[str, Any] = {"messages": _messages(2)}  # 4 条 < 10

    assert should_summarize(state) is True  # type: ignore[arg-type]


# ---------------- node_summarize ----------------


def test_summarize_removes_older_half_and_writes_summary(
    _isolate_summary: dict[str, Any],
) -> None:
    """12 条消息（6 对）对半切：最旧 6 条删除，较新 6 条保留。"""
    messages = _messages(6)
    state: dict[str, Any] = {"messages": messages, "summary": None}

    result = node_summarize(state)  # type: ignore[arg-type]

    assert result["summary"] == "合并后的新摘要"
    removals = cast(list[RemoveMessage], result["messages"])
    assert len(removals) == 6
    assert all(isinstance(m, RemoveMessage) for m in removals)
    assert [m.id for m in removals] == [
        "h0",
        "a0",
        "h1",
        "a1",
        "h2",
        "a2",
    ]

    # 经真实 add_messages reducer 后：旧的一半消失，新的一半原样保留
    remaining = add_messages(messages, removals)
    assert [m.id for m in remaining] == [m.id for m in messages[6:]]


def test_summarize_prompt_receives_dialogue_limits_and_old_summary(
    _isolate_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧摘要、待压缩对话文本、长度上限都进入摘要 prompt。"""
    monkeypatch.setattr(Config, "SUMMARY_MAX_CHARS", 200)
    monkeypatch.setattr(Config, "SUMMARY_MAX_ITEMS", 5)
    state: dict[str, Any] = {"messages": _messages(6), "summary": "旧摘要要点"}

    node_summarize(state)  # type: ignore[arg-type]

    human_content = _isolate_summary["seen"][0]["messages"].messages[1].content
    system_content = str(_isolate_summary["seen"][0]["messages"].messages[0].content)
    assert "旧摘要要点" in human_content
    # 进入摘要的只有旧的一半（前 3 对），新的一半不进摘要
    assert "问题0" in human_content and "回答0" in human_content
    assert "问题2" in human_content
    assert "问题3" not in human_content
    assert "200 字以内" in system_content
    assert "不超过 5 条" in system_content


def test_summarize_token_triggered_also_splits_half(
    _isolate_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """token 触发（条数未超阈值）时同样按半压半留：3 对压 1 对留 2 对。"""
    monkeypatch.setattr(Config, "SUMMARY_TRIGGER_TOKENS", 1)
    messages = _messages(3)  # 6 条 < 触发条数 10
    state: dict[str, Any] = {"messages": messages}

    result = node_summarize(state)  # type: ignore[arg-type]

    removals = cast(list[RemoveMessage], result["messages"])
    assert [m.id for m in removals] == ["h0", "a0"]
    remaining = add_messages(messages, removals)
    assert [m.id for m in remaining] == ["h1", "a1", "h2", "a2"]


def test_summarize_ratio_configurable(
    _isolate_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """压缩比例可通过 Config 覆盖：0.34 时 6 对压掉 2 对、保留 4 对。"""
    monkeypatch.setattr(Config, "SUMMARY_COMPRESS_RATIO", 0.34)
    messages = _messages(6)

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    removals = cast(list[RemoveMessage], result["messages"])
    assert [m.id for m in removals] == ["h0", "a0", "h1", "a1"]
    remaining = add_messages(messages, removals)
    assert len(remaining) == 8


def test_summarize_invalid_ratio_falls_back_to_half(
    _isolate_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ratio 非法（<=0 或 >=1）时按 0.5 兜底。"""
    monkeypatch.setattr(Config, "SUMMARY_COMPRESS_RATIO", 0.0)
    messages = _messages(4)  # 4 对

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    removals = cast(list[RemoveMessage], result["messages"])
    assert len(removals) == 4  # 对半：压 2 对
    remaining = add_messages(messages, removals)
    assert len(remaining) == 4


def test_summarize_odd_tail_message_is_kept(
    _isolate_summary: dict[str, Any],
) -> None:
    """奇数条（尾部多出的半对）不参与切分，始终留在近期侧。"""
    messages = _messages(2) + [HumanMessage(content="在途追问", id="hx")]

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    removals = cast(list[RemoveMessage], result["messages"])
    assert [m.id for m in removals] == ["h0", "a0"]
    remaining = add_messages(messages, removals)
    assert [m.id for m in remaining] == ["h1", "a1", "hx"]


def test_summarize_llm_failure_keeps_history_intact(
    _isolate_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """摘要 LLM 抛异常时返回空更新，不删除任何历史消息。"""

    def _boom(_messages: list[Any]) -> AIMessage:
        raise RuntimeError("LLM 500")

    monkeypatch.setattr(
        summarize,
        "create_chat_model",
        lambda: RunnableLambda(_boom),
    )
    summarize._get_summary_chain.cache_clear()
    messages = _messages(6)

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    assert result == {}


def test_summarize_blank_output_keeps_history_intact(
    _isolate_summary: dict[str, Any],
) -> None:
    """摘要返回空白时同样放弃压缩，历史原样保留。"""
    _isolate_summary["state"]["output"] = "   \n "
    messages = _messages(6)

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    assert result == {}


def test_summarize_too_few_pairs_skips(
    _isolate_summary: dict[str, Any],
) -> None:
    """不足两轮（无法各留一轮）时直接跳过且不调模型。"""
    result = node_summarize({"messages": _messages(1)})  # type: ignore[arg-type]

    assert result == {}
    assert _isolate_summary["seen"] == []


# ---------------- SystemMessage / ToolMessage 保护 ----------------


def test_leading_system_message_never_compressed(
    _isolate_summary: dict[str, Any],
) -> None:
    """前导 SystemMessage 不计入轮次、不被删除、不进入摘要文本。"""
    system = SystemMessage(content="持久系统指令", id="sys")
    messages = [system, *_messages(3)]  # sys + 3 个问答轮

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    removals = cast(list[RemoveMessage], result["messages"])
    assert "sys" not in [m.id for m in removals]
    assert [m.id for m in removals] == ["h0", "a0"]

    # 经 reducer 后系统消息仍在最前，近期轮次原样保留
    remaining = add_messages(messages, removals)
    assert remaining[0].id == "sys"
    assert [m.id for m in remaining[1:]] == ["h1", "a1", "h2", "a2"]

    # 系统指令不进入摘要 LLM 的对话文本
    human_content = _isolate_summary["seen"][0]["messages"].messages[1].content
    assert "持久系统指令" not in human_content


def test_system_message_inside_old_turn_is_rescued(
    _isolate_summary: dict[str, Any],
) -> None:
    """混在旧轮区间内的 SystemMessage 同样被救出，绝不删除。"""
    messages = [
        *_messages(1),  # h0 a0（第 1 轮）
        SystemMessage(content="中途插入的指令", id="sys-mid"),
        HumanMessage(content="问题1", id="h1"),
        AIMessage(content="回答1", id="a1"),
        HumanMessage(content="问题2", id="h2"),
        AIMessage(content="回答2", id="a2"),
    ]

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    removals = cast(list[RemoveMessage], result["messages"])
    assert "sys-mid" not in [m.id for m in removals]
    assert [m.id for m in removals] == ["h0", "a0"]
    remaining = add_messages(messages, removals)
    assert "sys-mid" in [m.id for m in remaining]


def test_tool_call_turn_compressed_as_atomic_block(
    _isolate_summary: dict[str, Any],
) -> None:
    """工具调用轮 [Human, AI(tool_calls), Tool, AI] 整体压缩，不被劈半。"""
    tool_call = {
        "name": "search",
        "args": {"q": "天气"},
        "id": "call-1",
        "type": "tool_call",
    }
    messages = [
        HumanMessage(content="查天气", id="h0"),
        AIMessage(content="", tool_calls=[tool_call], id="a0"),
        ToolMessage(content="晴天", tool_call_id="call-1", id="t0"),
        AIMessage(content="今天晴天", id="a1"),
        HumanMessage(content="再问一个", id="h1"),
        AIMessage(content="回答二", id="a2"),
    ]

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    removals = cast(list[RemoveMessage], result["messages"])
    # 旧的一整轮（4 条，含工具调用组）一起删除，新轮完整保留
    assert [m.id for m in removals] == ["h0", "a0", "t0", "a1"]
    remaining = add_messages(messages, removals)
    assert [m.id for m in remaining] == ["h1", "a2"]


def test_no_human_message_skips_compression(
    _isolate_summary: dict[str, Any],
) -> None:
    """没有任何用户消息（如纯系统/工具消息）时不压缩、不调模型。"""
    messages = [SystemMessage(content="只有系统消息", id="sys")]

    result = node_summarize({"messages": messages})  # type: ignore[arg-type]

    assert result == {}
    assert _isolate_summary["seen"] == []
