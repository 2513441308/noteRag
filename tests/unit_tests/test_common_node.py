"""通用问答节点的离线单元测试.

用 RunnableLambda 伪造聊天模型，验证摘要/原生历史注入与成对
Human/AI 消息的输出契约，全程不联网。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from agent.nodes import common
from agent.nodes.common import node_common


@pytest.fixture(autouse=True)
def _isolate_common_chain(monkeypatch: pytest.MonkeyPatch) -> Any:
    """每个测试前后清空通用问答链缓存，保证假模型即时生效。"""
    common._get_common_chain.cache_clear()
    yield
    common._get_common_chain.cache_clear()


def _patch_fake_model(
    monkeypatch: pytest.MonkeyPatch, output: str = "你好呀"
) -> list[dict[str, Any]]:
    """注入假模型并记录它收到的渲染后消息列表。"""
    seen: list[dict[str, Any]] = []

    def _fake_invoke(messages: list[Any]) -> AIMessage:
        seen.append({"messages": messages})
        return AIMessage(content=output)

    monkeypatch.setattr(
        common,
        "create_chat_model",
        lambda: RunnableLambda(_fake_invoke),
    )
    return seen


def test_node_writes_answer_and_appends_qa_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """节点输出 answer，并向 messages 成对追加本轮 Human + AI 消息。"""
    _patch_fake_model(monkeypatch, "当然可以帮你")

    result = node_common({"question": "你好"})  # type: ignore[arg-type]

    assert result["answer"] == "当然可以帮你"
    assert result["sources"] == []
    new_messages = result["messages"]
    assert isinstance(new_messages, list)
    assert len(new_messages) == 2
    assert isinstance(new_messages[0], HumanMessage)
    assert new_messages[0].content == "你好"
    assert isinstance(new_messages[1], AIMessage)
    assert new_messages[1].content == "当然可以帮你"


def test_node_summary_history_and_question_enter_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """摘要与历史以原生消息注入，本轮问题作为末尾 human 消息。"""
    seen = _patch_fake_model(monkeypatch)
    history = [
        HumanMessage(content="上一轮寒暄"),
        AIMessage(content="上一轮回复"),
    ]
    state: dict[str, Any] = {
        "question": "今天天气如何",
        "summary": "用户喜欢简短回答",
        "messages": history,
    }

    node_common(state)  # type: ignore[arg-type]

    messages = seen[0]["messages"].messages
    # 规则 system + 摘要 system + 历史两条 + 本轮 human
    assert "用户喜欢简短回答" in str(messages[1].content)
    assert messages[2] is history[0]
    assert messages[3] is history[1]
    assert isinstance(messages[-1], HumanMessage)
    assert "今天天气如何" in messages[-1].content


def test_node_first_turn_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    """首轮无 summary/messages 时渲染占位符且不产生历史消息。"""
    seen = _patch_fake_model(monkeypatch)

    node_common({"question": "嗨"})  # type: ignore[arg-type]

    messages = seen[0]["messages"].messages
    assert len(messages) == 3  # 规则 system + 摘要 system + 本轮 human
    assert "（无）" in str(messages[1].content)


def test_common_chain_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型只构建一次，后续调用复用缓存链。"""
    build_count = {"n": 0}

    def _factory() -> RunnableLambda[Any, AIMessage]:
        build_count["n"] += 1
        return RunnableLambda(lambda _messages: AIMessage(content="答"))

    monkeypatch.setattr(common, "create_chat_model", _factory)

    for _ in range(2):
        node_common({"question": "q"})  # type: ignore[arg-type]

    assert build_count["n"] == 1
