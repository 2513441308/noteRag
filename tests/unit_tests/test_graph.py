"""工作流全链路的离线单元测试.

所有 LLM / 检索 / 重排组件全部打桩（不加载模型、不联网、不读向量库），
真实编译并运行 LangGraph，验证：路由分叉、RAG 四节点串联、检索词传递、
OutputState 字段收窄、通用分支、以及回答后的压缩条件边。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from agent.config import Config
from agent.nodes import common, generate, rag, rerank, route, searchrebuild, summarize
from agent.prompts.route import RouteResponse

# 注意：agent.__init__ 把编译后的图实例也命名为 graph，会遮蔽子模块属性，
# 必须按模块名从 sys.modules 取真正的模块对象，以便 monkeypatch 其全局函数。
graph_module = import_module("agent.graph")


class _FakeStructuredModel:
    """假的结构化输出模型：with_structured_output 返回固定对象。"""

    def __init__(self, response: RouteResponse) -> None:
        self.response = response

    def with_structured_output(self, schema: type[Any]) -> RunnableLambda[Any, Any]:
        del schema
        return RunnableLambda(lambda _payload: self.response)


def _text_model(text: str) -> RunnableLambda[Any, AIMessage]:
    """假文本模型：忽略输入，固定返回 AIMessage。"""
    return RunnableLambda(lambda _messages: AIMessage(content=text))


@pytest.fixture
def _rag_pipeline_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """打桩 RAG 链路全部外部依赖，返回调用记录器。"""
    calls: dict[str, Any] = {
        "retrieval_queries": [],
        "rerank_query": None,
        "common_model_built": 0,
        "split_called": 0,
    }

    # route：判定为领域相关
    monkeypatch.setattr(
        route,
        "create_chat_model",
        lambda: _FakeStructuredModel(RouteResponse(relevant=True)),
    )

    # searchrebuild：返回固定改写词；术语表打桩，不碰向量库
    monkeypatch.setattr(
        searchrebuild, "create_chat_model", lambda: _text_model("改写检索词")
    )
    monkeypatch.setattr(searchrebuild, "_get_kb_glossary", lambda: "（无）")

    # rag：双路各返回一篇不同来源文档；强制走 chroma 双路分支
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "chroma")

    def _fake_vector_search(query: str) -> list[Document]:
        calls["retrieval_queries"].append(query)
        return [
            Document(id="d1", page_content="向量片段", metadata={"source": "/a.md"})
        ]

    def _fake_bm25_search(query: str) -> list[Document]:
        calls["retrieval_queries"].append(query)
        return [
            Document(id="d2", page_content="BM25片段", metadata={"source": "/b.md"})
        ]

    monkeypatch.setattr(rag, "_vectorstore_search", _fake_vector_search)
    monkeypatch.setattr(rag, "_bm25_search", _fake_bm25_search)

    # rerank：不加载模型，保序返回（top_k 截断交给真实函数语义外的桩）
    monkeypatch.setattr(Config, "RERANK_ENABLED", True)
    monkeypatch.setattr(rerank, "build_reranker", lambda *a, **k: object())

    def _fake_rerank(
        _scorer: Any, query: str, docs: list[Document], top_k: int
    ) -> list[Document]:
        calls["rerank_query"] = query
        # 与真实 rerank_documents 一样盖名次与分数，否则 .env 配置了
        # RERANK_SCORE_THRESHOLD 时缺分按 0.0 处理会触发熔断清空候选
        kept = docs[:top_k]
        for rank, doc in enumerate(kept, start=1):
            doc.metadata["rerank_score"] = 1.0
            doc.metadata["rerank_rank"] = rank
        return kept

    monkeypatch.setattr(rerank, "rerank_documents", _fake_rerank)

    # generate：固定最终回答
    monkeypatch.setattr(
        generate, "create_chat_model", lambda: _text_model("RAG最终回答[1]")
    )

    # common：一旦被调用即失败，用于证明 RAG 分支不会误入通用分支
    def _common_boom() -> RunnableLambda[Any, AIMessage]:
        calls["common_model_built"] += 1
        raise AssertionError("RAG 分支不应调用 common 模型")

    monkeypatch.setattr(common, "create_chat_model", _common_boom)

    # summarize：记录节点是否真正执行（2 条消息=1 对，切分后无可压内容）
    real_split = summarize._split_stale_and_kept

    def _spy_split(messages: list[Any]) -> tuple[list[Any], list[Any]]:
        calls["split_called"] += 1
        return real_split(messages)

    monkeypatch.setattr(summarize, "_split_stale_and_kept", _spy_split)

    for module in (
        route,
        searchrebuild,
        rerank,
        generate,
        common,
    ):
        for name in dir(module):
            attr = getattr(module, name, None)
            if name.startswith("_get_") and callable(attr):
                cache_clear = getattr(attr, "cache_clear", None)
                if cache_clear is not None:
                    cache_clear()

    return calls


def test_rag_branch_full_flow(_rag_pipeline_stubs: dict[str, Any]) -> None:
    """相关追问走完整 RAG 链路，输出收窄为 answer+sources。"""
    config = {"configurable": {"thread_id": "test-rag-full-flow"}}
    # 首轮无历史时 searchrebuild 会短路，故先向线程注入一对历史，
    # 使本轮「它…」追问真正触发改写链，验证改写词流入双路召回
    # as_node 必须指定：默认按 InputState 过滤会丢弃 messages 字段；
    # 用末节点 summarize 身份写入使其边指向 END，下一轮 invoke 仍从 route 开始
    graph_module.graph.update_state(
        config,
        {
            "messages": [
                HumanMessage(content="StateGraph 是什么"),
                AIMessage(content="是 LangGraph 的状态图结构"),
            ]
        },
        as_node="summarize",
    )

    result = graph_module.graph.invoke(
        {"question": "它怎么加条件边？"},
        config=config,
    )

    assert set(result.keys()) == {"answer", "sources"}
    assert result["answer"] == "RAG最终回答[1]"
    # 改写词与原问题各走一遍双路召回（向量→BM25），改写漂移时裸问题兜底
    assert _rag_pipeline_stubs["retrieval_queries"] == [
        "改写检索词",
        "改写检索词",
        "它怎么加条件边？",
        "它怎么加条件边？",
    ]
    assert _rag_pipeline_stubs["rerank_query"] == "改写检索词"
    assert result["sources"] == ["/a.md", "/b.md"]
    assert _rag_pipeline_stubs["common_model_built"] == 0
    # 两轮共 4 条消息，低于压缩阈值（SUMMARY_TRIGGER_MESSAGES=10），未触发压缩
    assert _rag_pipeline_stubs["split_called"] == 0


def test_common_branch_when_irrelevant(
    monkeypatch: pytest.MonkeyPatch, _rag_pipeline_stubs: dict[str, Any]
) -> None:
    """无关问题直接走 common，不触发检索/重排，sources 为空。"""
    monkeypatch.setattr(
        route,
        "create_chat_model",
        lambda: _FakeStructuredModel(RouteResponse(relevant=False)),
    )
    monkeypatch.setattr(common, "create_chat_model", lambda: _text_model("通用回答"))
    route._get_route_chain.cache_clear()
    common._get_common_chain.cache_clear()

    result = graph_module.graph.invoke(
        {"question": "今天天气怎么样"},
        config={"configurable": {"thread_id": "test-common-branch"}},
    )

    assert result["answer"] == "通用回答"
    assert result["sources"] == []
    assert _rag_pipeline_stubs["retrieval_queries"] == []
    assert _rag_pipeline_stubs["rerank_query"] is None


def test_summarize_branch_entered_when_threshold_hit(
    monkeypatch: pytest.MonkeyPatch, _rag_pipeline_stubs: dict[str, Any]
) -> None:
    """条件边判定需要压缩时，summarize 节点确实执行（本轮无旧消息则空操作）。"""
    monkeypatch.setattr(graph_module, "should_summarize", lambda _state: True)

    result = graph_module.graph.invoke(
        {"question": "长对话后的问题"},
        config={"configurable": {"thread_id": "test-summarize-entered"}},
    )

    assert result["answer"] == "RAG最终回答[1]"
    assert _rag_pipeline_stubs["split_called"] == 1


def test_summarize_branch_skipped_when_under_threshold(
    _rag_pipeline_stubs: dict[str, Any],
) -> None:
    """未超阈值时不进入 summarize 节点。"""
    graph_module.graph.invoke(
        {"question": "短对话"},
        config={"configurable": {"thread_id": "test-summarize-skipped"}},
    )

    assert _rag_pipeline_stubs["split_called"] == 0
