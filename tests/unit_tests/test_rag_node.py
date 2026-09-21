"""RAG 双路召回节点的离线单元测试.

monkeypatch 掉全部组件工厂，验证：
- 向量库/向量检索器/BM25 检索器的参数全部来自 Config 且只构建一次；
- BM25 复用同一个向量库实例，k 读 BM25_K；
- node_rag 双路合路、按 id 去重、输出 dict 形态文档；
- postgres 下 BM25 自动跳过，仅走向量路。
全程不加载真实模型、不碰磁盘。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from agent.config import ROOT_DIR, Config
from agent.nodes import rag


class _FakeRetriever:
    """记录查询并返回固定文档的假检索器。"""

    def __init__(self, name: str, docs: list[Document], log: dict[str, Any]) -> None:
        self.name = name
        self.docs = docs
        self.log = log
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        self.log["invokes"][self.name] = self.log["invokes"].get(self.name, 0) + 1
        return list(self.docs)


@pytest.fixture(autouse=True)
def _isolate_rag(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """注入假工厂并固定配置，每个测试前后清空三个组件缓存。"""
    calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
        "embedding": [],
        "vectorstore": [],
        "vector_retriever": [],
        "bm25": [],
    }
    log: dict[str, Any] = {"invokes": {}}
    retrievers: dict[str, _FakeRetriever] = {}

    docs_by_name = {
        "vector": [
            Document(id="v1", page_content="向量文档1", metadata={"src": "v"}),
            Document(id="shared", page_content="重复文档", metadata={"src": "v"}),
        ],
        "bm25": [
            Document(id="shared", page_content="重复文档", metadata={"src": "b"}),
            Document(id="b1", page_content="BM25文档1", metadata={"src": "b"}),
        ],
    }

    monkeypatch.setattr(
        rag,
        "create_embedding_model",
        lambda *a, **k: calls["embedding"].append((a, k)) or "fake-embedding",
    )

    def _fake_vectorstore(*args: Any, **kwargs: Any) -> str:
        calls["vectorstore"].append((args, kwargs))
        return "fake-store"

    def _fake_vector_retriever(*args: Any, **kwargs: Any) -> _FakeRetriever:
        calls["vector_retriever"].append((args, kwargs))
        retriever = _FakeRetriever("vector", docs_by_name["vector"], log)
        retrievers["vector"] = retriever
        return retriever

    def _fake_bm25(*args: Any, **kwargs: Any) -> _FakeRetriever:
        calls["bm25"].append((args, kwargs))
        retriever = _FakeRetriever("bm25", docs_by_name["bm25"], log)
        retrievers["bm25"] = retriever
        return retriever

    monkeypatch.setattr(rag, "get_vectorstore", _fake_vectorstore)
    monkeypatch.setattr(rag, "get_retriever", _fake_vector_retriever)
    monkeypatch.setattr(rag, "build_bm25_from_store", _fake_bm25)

    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "chroma")
    monkeypatch.setattr(Config, "VECTORSTORE_COLLECTION_NAME", "test_collection")
    monkeypatch.setattr(Config, "VECTORSTORE_PATH", "./test_chroma")
    monkeypatch.setattr(Config, "RETRIEVAL_K", 3)
    monkeypatch.setattr(Config, "BM25_K", 5)

    for cached in (
        rag._get_vectorstore_from_config,
        rag._get_vector_retriever,
        rag._get_bm25_retriever,
    ):
        cached.cache_clear()

    bundle: dict[str, Any] = {
        "calls": calls,
        "log": log,
        "docs": docs_by_name,
        "retrievers": retrievers,
    }
    yield bundle

    for cached in (
        rag._get_vectorstore_from_config,
        rag._get_vector_retriever,
        rag._get_bm25_retriever,
    ):
        cached.cache_clear()


# ---------------- 向量库组装 ----------------


def test_chroma_settings_all_from_config(_isolate_rag: dict[str, Any]) -> None:
    """chroma：集合名、provider、相对路径 ROOT_DIR 解析全部来自 Config。"""
    calls = _isolate_rag["calls"]

    rag._get_vectorstore_from_config()

    args, kwargs = calls["vectorstore"][0]
    assert args[0] == "test_collection"
    assert args[1] == "fake-embedding"
    assert kwargs["provider"] == "chroma"
    assert kwargs["persist_directory"] == ROOT_DIR / "test_chroma"
    assert "connection" not in kwargs


def test_absolute_persist_path_kept(
    _isolate_rag: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """绝对路径原样透传，不再拼接 ROOT_DIR。"""
    calls = _isolate_rag["calls"]
    monkeypatch.setattr(Config, "VECTORSTORE_PATH", str(tmp_path / "abs_db"))

    rag._get_vectorstore_from_config()

    _, kwargs = calls["vectorstore"][0]
    assert kwargs["persist_directory"] == Path(str(tmp_path / "abs_db"))


def test_postgres_uses_dsn_from_config(
    _isolate_rag: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """postgres：用 Config 分项参数拼 DSN，不传持久化目录。"""
    calls = _isolate_rag["calls"]
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "postgres")
    monkeypatch.setattr(Config, "VECTORSTORE_POSTGRES_HOST", "pg.example")
    monkeypatch.setattr(Config, "VECTORSTORE_POSTGRES_PORT", 6543)
    monkeypatch.setattr(Config, "VECTORSTORE_POSTGRES_USER", "u1")
    monkeypatch.setattr(Config, "VECTORSTORE_POSTGRES_PASSWORD", "p1")
    monkeypatch.setattr(Config, "VECTORSTORE_POSTGRES_DB", "db1")

    rag._get_vectorstore_from_config()

    _, kwargs = calls["vectorstore"][0]
    assert kwargs["provider"] == "postgres"
    assert kwargs["connection"] == "postgresql+psycopg://u1:p1@pg.example:6543/db1"
    assert "persist_directory" not in kwargs


def test_unsupported_provider_raises(
    _isolate_rag: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """未知 provider 抛出 ValueError。"""
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "milvus")

    with pytest.raises(ValueError, match="VECTORSTORE_PROVIDER"):
        rag._get_vectorstore_from_config()


# ---------------- BM25 组装 ----------------


def test_bm25_uses_same_store_and_bm25_k(_isolate_rag: dict[str, Any]) -> None:
    """BM25 必须从同一个向量库实例构建，k 读 Config.BM25_K。"""
    calls = _isolate_rag["calls"]

    store = rag._get_vectorstore_from_config()
    rag._get_bm25_retriever()

    args, kwargs = calls["bm25"][0]
    assert args[0] is store
    assert kwargs["k"] == 5


def test_bm25_unsupported_on_postgres(
    _isolate_rag: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """postgres 下直接构建 BM25 检索器抛出 NotImplementedError。"""
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "postgres")

    with pytest.raises(NotImplementedError, match="Chroma"):
        rag._get_bm25_retriever()


def test_components_built_only_once(_isolate_rag: dict[str, Any]) -> None:
    """重复获取只构建一次向量库/向量检索器/BM25（缓存生效）。"""
    calls = _isolate_rag["calls"]

    for _ in range(3):
        rag._get_vectorstore_from_config()
        rag._get_vector_retriever()
        rag._get_bm25_retriever()

    assert len(calls["vectorstore"]) == 1
    assert len(calls["vector_retriever"]) == 1
    assert len(calls["bm25"]) == 1


# ---------------- 去重与节点 ----------------


def test_dedup_by_id_keeps_first_order() -> None:
    """按 id 去重并保持首次出现顺序；无 id 退回内容指纹。"""
    docs = [
        Document(id="a", page_content="A"),
        Document(id="a", page_content="A2"),
        Document(id="b", page_content="B"),
        Document(page_content="C"),
        Document(page_content="C"),
    ]

    result = rag._dedup_documents(docs)

    assert [d.id if d.id else d.page_content for d in result] == ["a", "b", "C"]
    assert result[0].page_content == "A"  # 保留首次出现的对象


def test_node_rag_merges_dedups_and_queries_both_rewritten_and_original(
    _isolate_rag: dict[str, Any],
) -> None:
    """改写词与原始问题各走一遍双路召回，按 id 去重，输出 dict 形态。"""
    log = _isolate_rag["log"]
    state = {"question": "原始问题", "rewritten_query": "改写后的检索词"}

    result = rag.node_rag(state)  # type: ignore[arg-type]

    docs = result["retrieved_docs"]
    assert [doc["id"] for doc in docs] == ["v1", "shared", "b1"]  # type: ignore[index]
    assert all(set(doc) == {"id", "page_content", "metadata"} for doc in docs)  # type: ignore[arg-type]
    # 两条查询各触发一次向量 + 一次 BM25
    assert log["invokes"] == {"vector": 2, "bm25": 2}
    fake_retrievers = _isolate_rag["retrievers"]
    # 改写词在前扩召回，原始问题在后兜底
    for retriever in fake_retrievers.values():
        assert retriever.queries == ["改写后的检索词", "原始问题"]


def test_build_queries_rewritten_first_original_last() -> None:
    """查询列表改写词在前、原始问题恒在末位兜底。"""
    state: dict[str, Any] = {"question": "原问题", "rewritten_query": "改写词"}

    assert rag._build_queries(state) == ["改写词", "原问题"]  # type: ignore[arg-type]


def test_node_rag_identical_rewritten_query_retrieves_once(
    _isolate_rag: dict[str, Any],
) -> None:
    """改写词与原问题相同（首轮短路）时只检索一轮，不做重复请求。"""
    log = _isolate_rag["log"]
    state = {"question": "同一个问题", "rewritten_query": "同一个问题"}

    rag.node_rag(state)  # type: ignore[arg-type]

    assert log["invokes"] == {"vector": 1, "bm25": 1}
    retriever = _isolate_rag["retrievers"]["vector"]
    assert retriever.queries == ["同一个问题"]


def test_node_rag_blank_rewritten_query_retrieves_once(
    _isolate_rag: dict[str, Any],
) -> None:
    """改写词为空白时按缺失处理，只用原问题检索一轮。"""
    log = _isolate_rag["log"]

    rag.node_rag({"question": "原问题", "rewritten_query": "   "})  # type: ignore[arg-type]

    assert log["invokes"] == {"vector": 1, "bm25": 1}


def test_node_rag_falls_back_to_question(_isolate_rag: dict[str, Any]) -> None:
    """rewritten_query 缺失时只使用原始 question 检索一轮。"""
    result = rag.node_rag({"question": "原始问题"})  # type: ignore[arg-type]

    docs = result["retrieved_docs"]
    assert len(docs) == 3  # type: ignore[arg-type]
    assert _isolate_rag["retrievers"]["vector"].queries == ["原始问题"]


def test_node_rag_postgres_skips_bm25(
    _isolate_rag: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """postgres 下节点不构建 BM25，仅返回向量路结果且不报错。"""
    calls = _isolate_rag["calls"]
    monkeypatch.setattr(Config, "VECTORSTORE_PROVIDER", "postgres")

    result = rag.node_rag({"question": "q"})  # type: ignore[arg-type]

    assert [doc["id"] for doc in result["retrieved_docs"]] == ["v1", "shared"]  # type: ignore[index]
    assert calls["bm25"] == []


def test_node_rag_degrades_when_bm25_index_unavailable(
    _isolate_rag: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """BM25 建索引抛 ValueError（如空库）时降级为仅向量路，不崩。"""

    def _boom(_query: str) -> list[Any]:
        raise ValueError("构建 BM25 索引至少需要 1 个文档")

    monkeypatch.setattr(rag, "_bm25_search", _boom)

    result = rag.node_rag({"question": "q"})  # type: ignore[arg-type]

    # 向量路结果照常返回（v1、shared），链路未中断
    assert [doc["id"] for doc in result["retrieved_docs"]] == ["v1", "shared"]  # type: ignore[index]
