"""Rerank 精排节点的离线单元测试.

用假 scorer 替换重排模型（不加载权重、不联网），但保留真实的
``rerank_documents`` 排序逻辑，验证节点的 Config 注入、排序、sources
去重保序、开关直通、空召回兜底、绝对分熔断与邻域扩展。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.documents import Document

from agent.config import Config
from agent.nodes import rag, rerank
from agent.nodes.rerank import extract_sources, node_rerank


class _FakeScorer:
    """按正文查分的假重排模型，同时记录收到的 query。"""

    def __init__(self, score_map: dict[str, float]) -> None:
        self.score_map = score_map
        self.seen_pairs: list[Sequence[tuple[str, str]]] = []

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[float]:
        del batch_size, show_progress_bar
        self.seen_pairs.append(pairs)
        return [self.score_map.get(passage, 0.0) for _, passage in pairs]


def _docs() -> list[dict[str, Any]]:
    """4 篇召回：a.md 两篇、b.md 一篇、无来源一篇（故意乱序给分）。"""
    return [
        {"id": "d1", "page_content": "c1", "metadata": {"source": "/docs/a.md"}},
        {"id": "d2", "page_content": "c2", "metadata": {"source": "/docs/b.md"}},
        {"id": "d3", "page_content": "c3", "metadata": {"source": "/docs/a.md"}},
        {"id": "d4", "page_content": "c4", "metadata": {}},
    ]


@pytest.fixture(autouse=True)
def _isolate_rerank(monkeypatch: pytest.MonkeyPatch) -> Any:
    """固定 RERANK_* 配置并注入假 build_reranker，测试前后清空模型缓存。"""
    scorer = _FakeScorer({"c1": 0.1, "c2": 0.9, "c3": 0.5, "c4": 0.01})
    build_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def _fake_build(*args: Any, **kwargs: Any) -> _FakeScorer:
        build_calls.append((args, kwargs))
        return scorer

    monkeypatch.setattr(rerank, "build_reranker", _fake_build)

    monkeypatch.setattr(Config, "RERANK_ENABLED", True)
    monkeypatch.setattr(Config, "RERANK_PROVIDER", "local")
    monkeypatch.setattr(Config, "RERANK_MODEL", "fake-reranker")
    monkeypatch.setattr(Config, "RERANK_DEVICE", "cpu")
    monkeypatch.setattr(Config, "RERANK_API_KEY", None)
    monkeypatch.setattr(Config, "RERANK_BASE_URL", None)
    monkeypatch.setattr(Config, "RERANK_API_TIMEOUT", 30.0)
    monkeypatch.setattr(Config, "RERANK_MAX_RETRIES", 2)
    monkeypatch.setattr(Config, "RERANK_TOP_K", 2)
    # 默认关闭熔断与邻域扩展，二者由专门用例显式开启
    monkeypatch.setattr(Config, "RERANK_SCORE_THRESHOLD", None)
    monkeypatch.setattr(Config, "RERANK_CONTEXT_WINDOW", 0)
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "chroma")

    rerank._get_reranker.cache_clear()
    bundle: dict[str, Any] = {"scorer": scorer, "build_calls": build_calls}
    yield bundle
    rerank._get_reranker.cache_clear()


# ---------------- extract_sources ----------------


def test_extract_sources_dedup_preserve_order() -> None:
    """sources 按出现顺序去重，缺失/非法 source 跳过。"""
    docs = [
        {"metadata": {"source": "/a.md"}},
        {"metadata": {"source": "/a.md"}},
        {"metadata": {"source": "/b.md"}},
        {"metadata": {}},
        {"metadata": "broken"},
        {},
    ]

    assert extract_sources(docs) == ["/a.md", "/b.md"]


# ---------------- 精排主流程 ----------------


def test_reranker_built_from_config(_isolate_rerank: dict[str, Any]) -> None:
    """模型按 Config 注入参数构建且只构建一次。"""
    state = {"question": "q", "rewritten_query": "改写q", "retrieved_docs": _docs()}

    node_rerank(state)  # type: ignore[arg-type]
    node_rerank(state)  # type: ignore[arg-type]

    assert len(_isolate_rerank["build_calls"]) == 1
    args, kwargs = _isolate_rerank["build_calls"][0]
    assert args == ("local", "fake-reranker")
    assert kwargs["device"] == "cpu"
    assert kwargs["timeout"] == 30.0
    assert kwargs["max_retries"] == 2


def test_enabled_reranks_and_writes_outputs(_isolate_rerank: dict[str, Any]) -> None:
    """开启时按分数降序取 top_k，写 ranked_docs 与去重后的 sources。"""
    state = {"question": "q", "rewritten_query": "改写q", "retrieved_docs": _docs()}

    result = node_rerank(state)  # type: ignore[arg-type]

    ranked = result["ranked_docs"]
    # c2(0.9) > c3(0.5) 入选；顺序即分数降序
    assert [doc["id"] for doc in ranked] == ["d2", "d3"]  # type: ignore[index]
    assert result["sources"] == ["/docs/b.md", "/docs/a.md"]
    # rerank_documents 追加的名次/分数元数据透传到 state
    assert ranked[0]["metadata"]["rerank_rank"] == 1  # type: ignore[index]
    assert ranked[0]["metadata"]["rerank_score"] == 0.9  # type: ignore[index]


def test_scoring_uses_rewritten_query_then_fallback(
    _isolate_rerank: dict[str, Any],
) -> None:
    """打分 query 优先用 rewritten_query，缺失时回退原始 question。"""
    scorer = _isolate_rerank["scorer"]

    rewritten_state: dict[str, Any] = {
        "question": "原始",
        "rewritten_query": "改写",
        "retrieved_docs": _docs(),
    }
    node_rerank(rewritten_state)  # type: ignore[arg-type]
    assert scorer.seen_pairs[0][0][0] == "改写"

    fallback_state: dict[str, Any] = {"question": "原始", "retrieved_docs": _docs()}
    node_rerank(fallback_state)  # type: ignore[arg-type]
    assert scorer.seen_pairs[1][0][0] == "原始"


def test_disabled_passthrough_keeps_order_and_truncates(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """关闭开关：不构建模型，保留召回原序，截断到 top_k，sources 照常产出。"""
    monkeypatch.setattr(Config, "RERANK_ENABLED", False)

    passthrough_state: dict[str, Any] = {"question": "q", "retrieved_docs": _docs()}
    result = node_rerank(passthrough_state)  # type: ignore[arg-type]

    ranked: list[dict[str, Any]] = result["ranked_docs"]  # type: ignore[assignment]
    assert [doc["id"] for doc in ranked] == ["d1", "d2"]
    assert result["sources"] == ["/docs/a.md", "/docs/b.md"]
    assert _isolate_rerank["build_calls"] == []
    assert _isolate_rerank["scorer"].seen_pairs == []


def test_empty_docs_returns_empty_without_building_model(
    _isolate_rerank: dict[str, Any],
) -> None:
    """无召回（或字段缺失）时返回空列表，且不触发重模型加载。"""
    result = node_rerank({"question": "q", "retrieved_docs": []})  # type: ignore[arg-type]

    assert result["ranked_docs"] == []
    assert result["sources"] == []
    assert _isolate_rerank["build_calls"] == []


# ---------------- 绝对分熔断 ----------------


def test_score_threshold_fuses_when_below(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """最高分仍低于阈值时熔断：清空候选与来源，不硬答。"""
    monkeypatch.setattr(Config, "RERANK_SCORE_THRESHOLD", 1.0)

    result = node_rerank({"question": "q", "retrieved_docs": _docs()})  # type: ignore[arg-type]

    assert result["ranked_docs"] == []
    assert result["sources"] == []


def test_score_threshold_keeps_when_top_meets_it(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """最高分达到阈值（含等于）时正常保留；阈值是严格小于才熔断。"""
    monkeypatch.setattr(Config, "RERANK_SCORE_THRESHOLD", 0.9)

    result = node_rerank({"question": "q", "retrieved_docs": _docs()})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["ranked_docs"]] == ["d2", "d3"]  # type: ignore[index]


def test_threshold_none_means_disabled(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """阈值为 None（默认）时即使分数极低也不熔断。"""
    monkeypatch.setattr(Config, "RERANK_SCORE_THRESHOLD", None)

    result = node_rerank({"question": "q", "retrieved_docs": _docs()})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["ranked_docs"]] == ["d2", "d3"]  # type: ignore[index]


def test_threshold_not_applied_when_rerank_disabled(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """关闭重排走直通时没有分数，阈值不应误杀候选。"""
    monkeypatch.setattr(Config, "RERANK_ENABLED", False)
    monkeypatch.setattr(Config, "RERANK_SCORE_THRESHOLD", 100.0)

    result = node_rerank({"question": "q", "retrieved_docs": _docs()})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["ranked_docs"]] == ["d1", "d2"]  # type: ignore[index]


# ---------------- 邻域扩展 ----------------


class _FakeCollection:
    """按 source + chunk_index 区间过滤的假 Chroma 集合。"""

    def __init__(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
        # rows: (id, content, metadata)，metadata 含 source 与 chunk_index
        self.rows = rows
        self.seen_wheres: list[dict[str, Any]] = []

    def get(
        self,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        del include
        assert where is not None
        self.seen_wheres.append(where)
        # $and 中含两个 chunk_index 子句（$gte / $lte），需分别提取
        source: str | None = None
        lo = hi = 0
        for clause in where["$and"]:
            if "source" in clause:
                source = clause["source"]
            index_clause = clause.get("chunk_index")
            if isinstance(index_clause, dict):
                if "$gte" in index_clause:
                    lo = index_clause["$gte"]
                if "$lte" in index_clause:
                    hi = index_clause["$lte"]
        picked = [
            row
            for row in self.rows
            if row[2]["source"] == source and lo <= row[2]["chunk_index"] <= hi
        ]
        return {
            "ids": [row[0] for row in picked],
            "documents": [row[1] for row in picked],
            "metadatas": [row[2] for row in picked],
        }


class _FakeStore:
    """暴露 ``_collection`` 的假 Chroma 向量库。"""

    def __init__(self, collection: _FakeCollection) -> None:
        self._collection = collection


def _indexed_docs() -> list[dict[str, Any]]:
    """d2=b.md#5（0.9 分）、d3=a.md#2（0.5 分）入选 top_k；带索引元数据。"""
    return [
        {
            "id": "d2",
            "page_content": "c2",
            "metadata": {"source": "/docs/b.md", "chunk_index": 5},
        },
        {
            "id": "d3",
            "page_content": "c3",
            "metadata": {"source": "/docs/a.md", "chunk_index": 2},
        },
    ]


def _patch_fake_store(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeCollection:
    """把 rag 节点的向量库工厂替换为含兄弟块的假库，返回假集合供断言。"""
    rows: list[tuple[str, str, dict[str, Any]]] = [
        ("a1", "a前文", {"source": "/docs/a.md", "chunk_index": 1}),
        ("a3", "a后文", {"source": "/docs/a.md", "chunk_index": 3}),
        ("b4", "b前文", {"source": "/docs/b.md", "chunk_index": 4}),
        ("b6", "b后文", {"source": "/docs/b.md", "chunk_index": 6}),
    ]
    collection = _FakeCollection(rows)

    def _fake_getter() -> _FakeStore:
        return _FakeStore(collection)

    monkeypatch.setattr(rag, "_get_vectorstore_from_config", _fake_getter)
    return collection


def test_neighbor_expansion_fills_siblings_and_restores_order(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """开启窗口后补回前后兄弟块：按来源分组、组内 chunk_index 升序。"""
    monkeypatch.setattr(Config, "RERANK_CONTEXT_WINDOW", 1)
    collection = _patch_fake_store(monkeypatch)

    result = node_rerank({"question": "q", "retrieved_docs": _indexed_docs()})  # type: ignore[arg-type]

    ranked = result["ranked_docs"]
    # 来源顺序跟随命中首现：b.md（rank1）在前、a.md 在后；组内恢复文档原序
    assert [doc["id"] for doc in ranked] == ["b4", "d2", "b6", "a1", "d3", "a3"]  # type: ignore[index]
    assert result["sources"] == ["/docs/b.md", "/docs/a.md"]
    # 命中块对象原样保留，重排分数元数据不丢
    seed_b = next(doc for doc in ranked if doc["id"] == "d2")  # type: ignore[index]
    assert seed_b["metadata"]["rerank_score"] == 0.9
    # 每个来源只发一次区间查询（窗口合并）
    assert len(collection.seen_wheres) == 2


def test_neighbor_expansion_window_zero_skips_store(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """窗口为 0 时不取邻域，向量库工厂一次都不碰。"""
    monkeypatch.setattr(Config, "RERANK_CONTEXT_WINDOW", 0)

    def _unexpected() -> Any:
        raise AssertionError("窗口关闭时不应访问向量库")

    monkeypatch.setattr(rag, "_get_vectorstore_from_config", _unexpected)

    result = node_rerank({"question": "q", "retrieved_docs": _indexed_docs()})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["ranked_docs"]] == ["d2", "d3"]  # type: ignore[index]


def test_neighbor_expansion_skipped_without_index_metadata(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """旧库/非 Markdown chunk 缺 chunk_index 时无法扩展，原样返回、不碰库。"""
    monkeypatch.setattr(Config, "RERANK_CONTEXT_WINDOW", 1)

    def _unexpected() -> Any:
        raise AssertionError("无索引元数据时不应访问向量库")

    monkeypatch.setattr(rag, "_get_vectorstore_from_config", _unexpected)

    result = node_rerank({"question": "q", "retrieved_docs": _docs()})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["ranked_docs"]] == ["d2", "d3"]  # type: ignore[index]


def test_neighbor_expansion_skipped_on_postgres(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """非 Chroma 后端不支持邻域扩展，静默跳过、不报错。"""
    monkeypatch.setattr(Config, "RERANK_CONTEXT_WINDOW", 1)
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "postgres")

    def _unexpected() -> Any:
        raise AssertionError("postgres 后端不应触发邻域取片")

    monkeypatch.setattr(rag, "_get_vectorstore_from_config", _unexpected)

    result = node_rerank({"question": "q", "retrieved_docs": _indexed_docs()})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["ranked_docs"]] == ["d2", "d3"]  # type: ignore[index]


def test_neighbor_expansion_failure_degrades_gracefully(
    _isolate_rerank: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """取邻域抛异常时降级为仅命中块，主链路不中断。"""
    monkeypatch.setattr(Config, "RERANK_CONTEXT_WINDOW", 1)

    def _boom() -> Any:
        raise RuntimeError("集合不可用")

    monkeypatch.setattr(rag, "_get_vectorstore_from_config", _boom)

    result = node_rerank({"question": "q", "retrieved_docs": _indexed_docs()})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["ranked_docs"]] == ["d2", "d3"]  # type: ignore[index]


def test_expand_helper_window_zero_is_identity(
    _isolate_rerank: dict[str, Any],
) -> None:
    """window=0 时扩展函数原样返回，不访问任何外部组件。"""
    seeded = [
        Document(
            id="x",
            page_content="x",
            metadata={"source": "/s.md", "chunk_index": 9},
        ),
    ]

    assert rerank._expand_with_neighbors(seeded) is seeded
