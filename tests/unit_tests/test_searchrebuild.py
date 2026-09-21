"""检索词重构节点的离线单元测试.

用 RunnableLambda 伪造聊天模型，验证历史渲染、术语表注入与改写回退逻辑，
全程不联网。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from agent.config import Config
from agent.nodes import rag, searchrebuild
from agent.nodes.searchrebuild import node_search_rebuild

# 真实聚合函数（autouse fixture 会把模块属性替换成桩，这里先留住未装饰引用）
_raw_glossary = searchrebuild._get_kb_glossary.__wrapped__


@pytest.fixture(autouse=True)
def _isolate_rewrite_chain(monkeypatch: pytest.MonkeyPatch) -> Any:
    """每个测试前清空链与术语表缓存；默认用桩替换术语表，不碰向量库。

    teardown 只清未被打桩的改写链缓存；术语表属性由 monkeypatch 自动还原
    为真实的 lru_cache 包装器（其缓存在测试中从未被填充：节点读桩、聚合
    测试走 ``_raw_glossary`` 旁路），无需再手动清。
    """
    searchrebuild._get_rewrite_chain.cache_clear()
    searchrebuild._get_kb_glossary.cache_clear()
    monkeypatch.setattr(searchrebuild, "_get_kb_glossary", lambda: "（无）")
    yield
    searchrebuild._get_rewrite_chain.cache_clear()


def _patch_fake_model(
    monkeypatch: pytest.MonkeyPatch, output: str | AIMessage
) -> list[dict[str, Any]]:
    """注入假模型并记录它收到的渲染后消息列表。"""
    seen: list[dict[str, Any]] = []

    def _fake_invoke(messages: list[Any]) -> AIMessage:
        seen.append({"messages": messages})
        if isinstance(output, AIMessage):
            return output
        return AIMessage(content=output)

    monkeypatch.setattr(
        searchrebuild,
        "create_chat_model",
        lambda: RunnableLambda(_fake_invoke),
    )
    return seen


def test_node_rewrites_with_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """节点把历史与问题送入链，返回改写后的检索词。"""
    seen = _patch_fake_model(monkeypatch, "StateGraph 如何添加条件边")
    state: dict[str, Any] = {
        "question": "它怎么添加条件边？",
        "messages": [
            HumanMessage(content="StateGraph 是什么"),
            AIMessage(content="是 LangGraph 的状态图结构"),
        ],
    }

    result = node_search_rebuild(state)  # type: ignore[arg-type]

    assert result == {"rewritten_query": "StateGraph 如何添加条件边"}
    human_content = seen[0]["messages"].messages[1].content
    assert "StateGraph 是什么" in human_content  # 历史确实进入了 prompt
    assert "它怎么添加条件边？" in human_content  # 最新问题同样进入


def test_glossary_enters_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """术语表随改写请求注入 system 消息，供模型对齐文档用词。"""
    seen = _patch_fake_model(monkeypatch, "条件边怎么写")
    monkeypatch.setattr(
        searchrebuild, "_get_kb_glossary", lambda: "- 核心概念 > 条件边"
    )
    state: dict[str, Any] = {
        "question": "分支判断怎么做？",
        "messages": [
            HumanMessage(content="StateGraph 是什么"),
            AIMessage(content="是状态图"),
        ],
    }

    node_search_rebuild(state)  # type: ignore[arg-type]

    system_content = str(seen[0]["messages"].messages[0].content)
    assert "核心概念 > 条件边" in system_content


def test_node_without_messages_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首轮对话无历史时直接回退原问题，不调用改写模型（省一次 LLM 请求）。"""
    seen = _patch_fake_model(monkeypatch, "独立检索词")
    result = node_search_rebuild({"question": "你好"})  # type: ignore[arg-type]

    assert result == {"rewritten_query": "你好"}
    assert seen == []  # 模型一次都不应被调用


def test_node_empty_content_history_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """历史全是空内容消息（渲染结果为「（无）」）时同样短路，不调用模型。"""
    seen = _patch_fake_model(monkeypatch, "独立检索词")
    state: dict[str, Any] = {
        "question": "你好",
        "messages": [HumanMessage(content="   ")],
    }

    result = node_search_rebuild(state)  # type: ignore[arg-type]

    assert result == {"rewritten_query": "你好"}
    assert seen == []


def test_node_falls_back_to_original_when_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有历史时模型只返回空白，回退为原始问题，避免空查询进入检索。"""
    _patch_fake_model(monkeypatch, "   \n  ")
    state: dict[str, Any] = {
        "question": "原始问题",
        "messages": [
            HumanMessage(content="上一轮问过 StateGraph"),
            AIMessage(content="StateGraph 是状态图结构"),
        ],
    }
    result = node_search_rebuild(state)  # type: ignore[arg-type]

    assert result == {"rewritten_query": "原始问题"}


# ---------------- 术语表聚合 ----------------


class _FakeHeadersCollection:
    """只支持 metadata 导出的假 Chroma 集合。"""

    def __init__(self, metadatas: list[dict[str, Any] | None]) -> None:
        self._metadatas = metadatas

    def get(self, include: list[str]) -> dict[str, Any]:
        assert include == ["metadatas"]
        return {"metadatas": self._metadatas}


class _FakeHeadersStore:
    def __init__(self, collection: _FakeHeadersCollection) -> None:
        self._collection = collection


def test_get_kb_glossary_renders_deduped_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """标题路径去重保序、每行一个条目；fixture 的桩被撤掉后走真实聚合函数。"""
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "chroma")
    store = _FakeHeadersStore(
        _FakeHeadersCollection(
            [
                {"headers": "核心概念 > StateGraph"},
                {"headers": "核心概念 > StateGraph"},  # 重复，只保留一次
                {"headers": "核心概念 > 条件边"},
                None,  # 游离 chunk
                {"other": "无标题字段"},
            ]
        )
    )
    monkeypatch.setattr(rag, "_get_vectorstore_from_config", lambda: store)

    glossary = _raw_glossary()

    assert glossary == "- 核心概念 > StateGraph\n- 核心概念 > 条件边"


def test_get_kb_glossary_empty_collection_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """空集合（没有任何标题）返回占位符，不返回空串。"""
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "chroma")
    store = _FakeHeadersStore(_FakeHeadersCollection([]))
    monkeypatch.setattr(rag, "_get_vectorstore_from_config", lambda: store)

    assert _raw_glossary() == "（无）"


def test_get_kb_glossary_postgres_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非 Chroma 后端直接返回占位符，不尝试访问集合。"""
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "postgres")

    def _unexpected() -> Any:
        raise AssertionError("postgres 不应触发术语表聚合")

    monkeypatch.setattr(rag, "_get_vectorstore_from_config", _unexpected)

    assert _raw_glossary() == "（无）"


def test_get_kb_glossary_failure_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """聚合抛任何异常都降级为占位符，绝不阻断改写节点。"""
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "chroma")

    def _boom() -> Any:
        raise RuntimeError("集合不可用")

    monkeypatch.setattr(rag, "_get_vectorstore_from_config", _boom)

    assert _raw_glossary() == "（无）"
