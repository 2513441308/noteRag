"""BM25 检索器端到端单元测试（纯内存，不加载 embedding 模型、不联网）."""

from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from agent.retrieval.bm25 import (
    build_bm25_from_store,
    create_bm25_retriever,
    fetch_neighbor_documents,
    load_distinct_headers,
)

CORPUS = [
    "Use StateGraph to build your workflow in LangGraph.",
    "The weather is sunny today.",
    "Call graph.add_node to add nodes into the graph.",
    "Checkpointer persists conversation state with SqliteSaver.",
    "RunnableConfig carries configuration through the graph.",
    "Documents are split into chunks before indexing.",
    "Embedding models convert text into dense vectors.",
    "The cat sat on the warm mat.",
    "Python functions are first class objects.",
    "Retrieval augmented generation combines search and LLM.",
]


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    return [
        Document(page_content=text, metadata={"doc_id": i})
        for i, text in enumerate(CORPUS)
    ]


@pytest.fixture(scope="module")
def retriever(documents: list[Document]):
    return create_bm25_retriever(documents, k=3)


@pytest.mark.parametrize("query", ["stategraph", "STATEGRAPH", "StateGraph"])
def test_case_insensitive_match(retriever, query: str) -> None:
    hits = retriever.invoke(query)
    assert hits[0].metadata["doc_id"] == 0


def test_camel_subword_with_spaces(retriever) -> None:
    hits = retriever.invoke("state graph 怎么用")
    assert hits[0].metadata["doc_id"] == 0


def test_snake_case_api_name(retriever) -> None:
    assert retriever.invoke("sqlitesaver")[0].metadata["doc_id"] == 3
    assert retriever.invoke("add node")[0].metadata["doc_id"] == 2
    assert retriever.invoke("RunnableConfig 配置")[0].metadata["doc_id"] == 4


def test_k_limits_result_count(retriever, documents: list[Document]) -> None:
    assert len(retriever.invoke("graph")) <= 3
    wide = create_bm25_retriever(documents, k=100)
    assert len(wide.invoke("graph")) == len(documents)


def test_build_from_chroma_store(tmp_path, documents: list[Document]) -> None:
    # 图启动路径：Chroma -> load_all_documents -> BM25，验证"Chroma 对了 BM25 就对了"
    embeddings = DeterministicFakeEmbedding(size=32)
    store = Chroma(
        collection_name="bm25_roundtrip",
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "chroma"),
    )
    store.add_documents(documents)

    retriever = build_bm25_from_store(store, k=2)
    hits = retriever.invoke("runnableconfig")
    assert [doc.metadata["doc_id"] for doc in hits][0] == 4


# ---------------- 标题术语表聚合 ----------------


def test_load_distinct_headers_dedupes_and_limits(tmp_path: Path) -> None:
    """headers 去重聚合，无标题 chunk 忽略，limit 生效（纯内存 Chroma）。"""
    embeddings = DeterministicFakeEmbedding(size=32)
    store = Chroma(
        collection_name="headers_glossary",
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "chroma"),
    )
    store.add_documents(
        [
            Document(page_content="a", metadata={"headers": "核心概念 > StateGraph"}),
            Document(page_content="b", metadata={"headers": "核心概念 > StateGraph"}),
            Document(page_content="c", metadata={"headers": "核心概念 > 条件边"}),
            Document(page_content="d", metadata={}),  # 游离 chunk：跳过
        ]
    )

    headers = load_distinct_headers(store)

    assert set(headers) == {"核心概念 > StateGraph", "核心概念 > 条件边"}
    assert len(headers) == 2  # 重复路径只保留一次
    assert len(load_distinct_headers(store, limit=1)) == 1


def test_load_distinct_headers_empty_store(tmp_path: Path) -> None:
    """空集合返回空列表（调用方据此渲染「（无）」占位）。"""
    embeddings = DeterministicFakeEmbedding(size=32)
    store = Chroma(
        collection_name="headers_empty",
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "chroma"),
    )

    assert load_distinct_headers(store) == []


def test_load_distinct_headers_requires_collection() -> None:
    """非 Chroma 实现（无 _collection）抛 TypeError，与 load_all_documents 一致。"""
    with pytest.raises(TypeError):
        load_distinct_headers(object())  # type: ignore[arg-type]


# ---------------- 邻域兄弟块捞取 ----------------


def _build_indexed_store(tmp_path: Path) -> Chroma:
    """两个来源的带序号小块：a.md#0-4、b.md#0-2。"""
    embeddings = DeterministicFakeEmbedding(size=32)
    store = Chroma(
        collection_name="neighbor_chunks",
        embedding_function=embeddings,
        persist_directory=str(tmp_path / "chroma"),
    )
    store.add_documents(
        [
            Document(
                id=f"a{i}",
                page_content=f"a 文档第 {i} 片",
                metadata={"source": "/docs/a.md", "chunk_index": i},
            )
            for i in range(5)
        ]
        + [
            Document(
                id=f"b{i}",
                page_content=f"b 文档第 {i} 片",
                metadata={"source": "/docs/b.md", "chunk_index": i},
            )
            for i in range(3)
        ]
    )
    return store


def test_fetch_neighbors_returns_siblings_excluding_anchors(tmp_path: Path) -> None:
    """window=1 只回锚点前后各一片，锚点自身不返回，跨源互不串扰。"""
    store = _build_indexed_store(tmp_path)
    anchors = [
        Document(
            id="a2",
            page_content="a 文档第 2 片",
            metadata={"source": "/docs/a.md", "chunk_index": 2},
        ),
        Document(
            id="b1",
            page_content="b 文档第 1 片",
            metadata={"source": "/docs/b.md", "chunk_index": 1},
        ),
    ]

    neighbors = fetch_neighbor_documents(store, anchors, window=1)

    assert set(neighbors) == {
        ("/docs/a.md", 1),
        ("/docs/a.md", 3),
        ("/docs/b.md", 0),
        ("/docs/b.md", 2),
    }
    assert neighbors[("/docs/a.md", 3)].page_content == "a 文档第 3 片"
    assert neighbors[("/docs/a.md", 3)].metadata["source"] == "/docs/a.md"


def test_fetch_neighbors_merges_overlapping_ranges(tmp_path: Path) -> None:
    """同源多锚点区间合并：a#1 与 a#3 一次查回 #0-#4 中除锚点外的片。"""
    store = _build_indexed_store(tmp_path)
    anchors = [
        Document(page_content="x", metadata={"source": "/docs/a.md", "chunk_index": 1}),
        Document(page_content="y", metadata={"source": "/docs/a.md", "chunk_index": 3}),
    ]

    neighbors = fetch_neighbor_documents(store, anchors, window=1)

    assert set(neighbors) == {("/docs/a.md", 0), ("/docs/a.md", 2), ("/docs/a.md", 4)}


def test_fetch_neighbors_window_zero_is_noop(tmp_path: Path) -> None:
    """window<=0 不查库，直接返回空映射。"""
    store = _build_indexed_store(tmp_path)
    anchors = [
        Document(page_content="x", metadata={"source": "/docs/a.md", "chunk_index": 2})
    ]

    assert fetch_neighbor_documents(store, anchors, window=0) == {}


def test_fetch_neighbors_skips_anchors_without_index(tmp_path: Path) -> None:
    """缺 source/chunk_index 的锚点跳过；全缺时不查库、返回空。"""
    store = _build_indexed_store(tmp_path)
    anchors = [Document(page_content="x", metadata={"doc_id": 1})]

    assert fetch_neighbor_documents(store, anchors, window=1) == {}


def test_fetch_neighbors_requires_collection() -> None:
    """非 Chroma 实现抛 TypeError。"""
    with pytest.raises(TypeError):
        fetch_neighbor_documents(
            object(),  # type: ignore[arg-type]
            [Document(page_content="x", metadata={"source": "s", "chunk_index": 0})],
            window=1,
        )
