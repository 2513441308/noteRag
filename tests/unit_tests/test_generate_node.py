"""最终回答节点的离线单元测试.

用 RunnableLambda 伪造聊天模型，验证来源级编号的 context 渲染、
摘要/空文档兜底、原生消息历史全量注入、原始问题注入，以及 answer 与
成对 Human/AI 消息的输出契约，全程不联网。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from agent.nodes import generate
from agent.nodes.generate import _build_context, node_generate


@pytest.fixture(autouse=True)
def _isolate_generate_chain(monkeypatch: pytest.MonkeyPatch) -> Any:
    """每个测试前后清空生成链缓存，保证假模型即时生效。"""
    generate._get_generate_chain.cache_clear()
    yield
    generate._get_generate_chain.cache_clear()


def _patch_fake_model(
    monkeypatch: pytest.MonkeyPatch, output: str = "这是最终回答"
) -> list[dict[str, Any]]:
    """注入假模型并记录它收到的渲染后消息列表。"""
    seen: list[dict[str, Any]] = []

    def _fake_invoke(messages: list[Any]) -> AIMessage:
        seen.append({"messages": messages})
        return AIMessage(content=output)

    monkeypatch.setattr(
        generate,
        "create_chat_model",
        lambda: RunnableLambda(_fake_invoke),
    )
    return seen


def _ranked_docs() -> list[dict[str, Any]]:
    """同一来源(a.md)两个 chunk 故意排在 b.md 前后，验证按来源聚合。"""
    return [
        {"id": "d1", "page_content": "片段一", "metadata": {"source": "/docs/a.md"}},
        {"id": "d3", "page_content": "片段三", "metadata": {"source": "/docs/a.md"}},
        {"id": "d2", "page_content": "片段二", "metadata": {"source": "/docs/b.md"}},
    ]


# ---------------- _build_context ----------------


def test_build_context_groups_chunks_by_source() -> None:
    """同来源多 chunk 共用一个编号，编号按首次出现顺序，与 sources 对齐。"""
    context = _build_context(_ranked_docs())

    assert context.startswith("[1]\n片段一\n---\n片段三\n\n[2]\n片段二")
    # 末尾给出确定性的合法编号范围
    assert context.endswith(
        "（本次共 2 个可用来源，合法引用编号仅为 [1]~[2]，严禁使用其他编号）"
    )


def test_build_context_restores_chunk_order_within_source() -> None:
    """同源片段按 chunk_index 恢复文档原序，无视传入的分数序。"""
    docs = [
        {
            "id": "d5",
            "page_content": "第五片",
            "metadata": {"source": "/a.md", "chunk_index": 5},
        },
        {
            "id": "d4",
            "page_content": "第四片",
            "metadata": {"source": "/a.md", "chunk_index": 4},
        },
        {
            "id": "d6",
            "page_content": "第六片",
            "metadata": {"source": "/a.md", "chunk_index": 6},
        },
    ]

    context = _build_context(docs)

    assert context.startswith("[1]\n第四片\n---\n第五片\n---\n第六片")


def test_build_context_unindexed_chunks_keep_passed_order() -> None:
    """缺 chunk_index 的旧块不参与排序，按传入顺序追加在有序块之后。"""
    docs = [
        {
            "id": "d2",
            "page_content": "有序二",
            "metadata": {"source": "/a.md", "chunk_index": 2},
        },
        {"id": "x1", "page_content": "无序号甲", "metadata": {"source": "/a.md"}},
        {
            "id": "d1",
            "page_content": "有序一",
            "metadata": {"source": "/a.md", "chunk_index": 1},
        },
        {"id": "x2", "page_content": "无序号乙", "metadata": {"source": "/a.md"}},
    ]

    context = _build_context(docs)

    assert context.startswith("[1]\n有序一\n---\n有序二\n---\n无序号甲\n---\n无序号乙")


def test_build_context_single_source_range_is_one() -> None:
    """只有一个来源时明确告知只能用 [1]，防止模型标 [2]。"""
    docs = [{"id": "d1", "page_content": "唯一片段", "metadata": {"source": "/a.md"}}]

    context = _build_context(docs)

    assert context == (
        "[1]\n唯一片段\n\n"
        "（本次共 1 个可用来源，合法引用编号仅为 [1]~[1]，严禁使用其他编号）"
    )


def test_build_context_sourceless_docs_go_to_dedicated_section() -> None:
    """缺 source 的文档不丢，归入「无来源」分组，且该分组不分配编号。"""
    docs = _ranked_docs() + [{"id": "d4", "page_content": "野生片段", "metadata": {}}]

    context = _build_context(docs)

    assert "\n\n[无来源]\n野生片段\n\n" in context
    assert context.startswith("[1]\n")  # 有来源的编号不受影响
    # 编号范围提示始终在最后
    assert context.endswith(
        "（本次共 2 个可用来源，合法引用编号仅为 [1]~[2]，严禁使用其他编号）"
    )


def test_build_context_only_sourceless_has_no_number_range() -> None:
    """全部文档都无来源时不出现编号范围提示，只有无来源分组。"""
    docs = [{"id": "d4", "page_content": "野生片段", "metadata": {}}]

    context = _build_context(docs)

    assert context == "[无来源]\n野生片段"


def test_build_context_empty_or_blank_returns_placeholder() -> None:
    """无文档或正文全空白时返回占位符。"""
    assert _build_context([]) == "（无）"
    assert _build_context([{"page_content": "   ", "metadata": {}}]) == "（无）"


# ---------------- node_generate ----------------


def test_node_writes_answer_and_appends_qa_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """节点输出 answer，并向 messages 成对追加本轮 Human + AI 消息。"""
    _patch_fake_model(monkeypatch, "带来源的回答[1]")
    state: dict[str, Any] = {"question": "q", "ranked_docs": _ranked_docs()}

    result = node_generate(state)  # type: ignore[arg-type]

    assert result["answer"] == "带来源的回答[1]"
    new_messages = result["messages"]
    assert isinstance(new_messages, list)
    assert len(new_messages) == 2
    assert isinstance(new_messages[0], HumanMessage)
    assert new_messages[0].content == "q"
    assert isinstance(new_messages[1], AIMessage)
    assert new_messages[1].content == "带来源的回答[1]"  # answer 即消息 content


def test_node_context_and_question_enter_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编号 context 进 system 消息，原始问题作为末尾 human 消息（非改写词）。"""
    seen = _patch_fake_model(monkeypatch)
    state: dict[str, Any] = {
        "question": "它怎么加条件边？",
        "rewritten_query": "StateGraph 如何添加条件边",
        "ranked_docs": _ranked_docs(),
    }

    node_generate(state)  # type: ignore[arg-type]

    messages = seen[0]["messages"].messages
    artifact_system = messages[1].content  # 第二条 system：摘要+检索文档
    assert "[1]" in artifact_system
    assert "片段一" in artifact_system and "片段三" in artifact_system
    assert "[2]" in artifact_system and "片段二" in artifact_system
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == "它怎么加条件边？"  # 用原始表述
    joined = "\n".join(str(m.content) for m in messages)
    assert "StateGraph 如何添加条件边" not in joined  # 改写词不进生成 prompt


def test_node_empty_summary_and_context_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """summary 与上下文为空时渲染占位符；首轮无历史则不多出消息。"""
    seen = _patch_fake_model(monkeypatch)

    node_generate({"question": "q", "ranked_docs": []})  # type: ignore[arg-type]

    messages = seen[0]["messages"].messages
    # 规则 system + 摘要/文档 system + 本轮 human，共 3 条，无历史消息
    assert len(messages) == 3
    assert messages[1].content.count("（无）") == 2  # 摘要、上下文两处


def test_node_history_injected_as_native_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """历史以原生 Human/AI 消息对象直接注入 prompt，角色结构保留。"""
    seen = _patch_fake_model(monkeypatch)
    history = [
        HumanMessage(content="StateGraph 是什么"),
        AIMessage(content="是状态图"),
    ]
    state: dict[str, Any] = {
        "question": "追问",
        "ranked_docs": [],
        "messages": history,
    }

    node_generate(state)  # type: ignore[arg-type]

    messages = seen[0]["messages"].messages
    # 两条 system 之后紧跟历史原对象，末尾是本轮问题
    assert messages[2] is history[0]
    assert messages[3] is history[1]
    assert messages[-1].content == "追问"


def test_node_history_passed_through_in_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """历史全量注入 prompt，节点不截断（规模由 summarize 节点统一控制）。"""
    seen = _patch_fake_model(monkeypatch)
    history = [HumanMessage(content=f"第{i}轮") for i in range(10)]
    state: dict[str, Any] = {
        "question": "q",
        "ranked_docs": [],
        "messages": history,
    }

    node_generate(state)  # type: ignore[arg-type]

    history_messages = seen[0]["messages"].messages[2:-1]
    assert len(history_messages) == 10
    assert history_messages[0].content == "第0轮"
    assert history_messages[-1].content == "第9轮"


def test_generate_chain_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型只构建一次，后续调用复用缓存链。"""
    build_count = {"n": 0}

    def _factory() -> RunnableLambda[Any, AIMessage]:
        build_count["n"] += 1
        return RunnableLambda(lambda _messages: AIMessage(content="答"))

    monkeypatch.setattr(generate, "create_chat_model", _factory)

    for _ in range(2):
        node_generate({"question": "q", "ranked_docs": []})  # type: ignore[arg-type]

    assert build_count["n"] == 1
