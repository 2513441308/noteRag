"""RAG 节点：双路召回（向量 + BM25）检索知识库.

组件工厂（``get_vectorstore`` / ``get_retriever`` / ``build_bm25_from_store``）
是不读配置的纯组件，本模块作为编排层集中从 ``Config`` 读取全部连接与召回参数，
再显式注入。三个重量级对象（向量库、向量检索器、BM25 检索器）均懒加载且
全局只构建一次：BM25 启动时需从 Chroma 集合导出全量 chunk 建索引，重复构建代价高。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from agent.config import ROOT_DIR, Config
from agent.embedding.embedding_factory import create_embedding_model
from agent.ingestion.vectorstore import build_postgres_dsn, get_vectorstore
from agent.retrieval.bm25 import build_bm25_from_store
from agent.retrieval.retrieval import get_retriever
from agent.state.state import MainState
from agent.utils.doc_utils import doc_to_dict

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_vectorstore_from_config() -> VectorStore:
    """按 ``Config`` 获取（或复用工厂缓存的）向量库实例.

    路径解析规则与 ingestion 侧 ``sync._resolve_store_config`` 完全一致：
    ``VECTORSTORE_PATH`` 为相对路径时按项目根 ``ROOT_DIR`` 解析，
    保证读写两侧落到同一个 Chroma 目录、共用工厂实例缓存。

    Returns:
        配置完成的向量库实例。

    Raises:
        ValueError: ``VECTORSTORE_PROVIDER`` 不是 chroma / postgres 时抛出。
    """
    embedding_model = create_embedding_model()

    provider = Config.VECTORSTORE_PROVIDER.lower()
    collection_name = Config.VECTORSTORE_COLLECTION_NAME

    if provider == "chroma":
        persist = Path(Config.VECTORSTORE_PATH)
        if not persist.is_absolute():
            persist = ROOT_DIR / persist
        return get_vectorstore(
            collection_name,
            embedding_model,
            provider=provider,
            persist_directory=persist,
        )

    if provider == "postgres":
        connection = build_postgres_dsn(
            Config.VECTORSTORE_POSTGRES_HOST,
            Config.VECTORSTORE_POSTGRES_PORT,
            Config.VECTORSTORE_POSTGRES_USER,
            Config.VECTORSTORE_POSTGRES_PASSWORD,
            Config.VECTORSTORE_POSTGRES_DB,
        )
        return get_vectorstore(
            collection_name,
            embedding_model,
            provider=provider,
            connection=connection,
        )

    raise ValueError(
        f"不支持的 VECTORSTORE_PROVIDER：{provider}，可选：chroma / postgres"
    )


@lru_cache(maxsize=1)
def _get_vector_retriever() -> BaseRetriever:
    """向量检索器，k 读 ``RETRIEVAL_K``，只构建一次."""
    return get_retriever(_get_vectorstore_from_config(), k=Config.RETRIEVAL_K)


@lru_cache(maxsize=1)
def _get_bm25_retriever() -> BaseRetriever:
    """BM25 检索器，从同一个 Config 向量库导出全量文档建一次索引，k 读 ``BM25_K``.

    Raises:
        NotImplementedError: 当前 BM25 只能从 Chroma 的底层集合导出文档，
            provider 为 postgres 时不支持。
    """
    if Config.VECTORSTORE_PROVIDER.lower() != "chroma":
        raise NotImplementedError(
            "BM25 双路召回目前仅支持 Chroma（需从底层集合导出全量文档建索引），"
            f"当前 VECTORSTORE_PROVIDER={Config.VECTORSTORE_PROVIDER}。"
        )
    return build_bm25_from_store(_get_vectorstore_from_config(), k=Config.BM25_K)


def _vectorstore_search(query: str) -> list[Document]:
    """向量检索与问题相关的文档片段.

    Args:
        query: 检索查询（通常为重构节点产出的 ``rewritten_query``）。

    Returns:
        召回的文档列表，条数由 ``RETRIEVAL_K`` 控制。
    """
    return _get_vector_retriever().invoke(query)


def _bm25_search(query: str) -> list[Document]:
    """BM25 关键词检索与问题相关的文档片段.

    Args:
        query: 检索查询（通常为重构节点产出的 ``rewritten_query``）。

    Returns:
        召回的文档列表，条数由 ``BM25_K`` 控制。
    """
    return _get_bm25_retriever().invoke(query)


def _dedup_documents(docs: list[Document]) -> list[Document]:
    """按文档 id 去重（无 id 时退回内容），保持首次出现顺序.

    Args:
        docs: 多路召回拼接后的文档列表。

    Returns:
        去重后的文档列表。
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.id if doc.id is not None else f"content:{doc.page_content}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _build_queries(state: MainState) -> list[str]:
    """构造本轮检索的去重查询列表：改写词优先，原始问题始终兜底.

    改写词可能漂移（被 LLM 引入历史噪声）或与文档词汇错位，裸问题是
    最后的保险，因此**原问题永远参与召回**，改写只负责扩召回而不是替换。
    改写词缺失/空白/与原问题相同时只保留一条，避免重复检索。

    Args:
        state: 当前图状态，需包含 ``question``，可选 ``rewritten_query``。

    Returns:
        去重保序的查询列表，首条为改写词（若有），末条为原始问题。
    """
    question = state["question"]
    rewritten = (state.get("rewritten_query") or "").strip()
    queries: list[str] = []
    if rewritten and rewritten != question.strip() and rewritten not in queries:
        queries.append(rewritten)
    queries.append(question)
    return queries


def node_rag(state: MainState) -> dict[str, object]:
    """双路召回：向量 + BM25 检索后按 id 合并去重，写入 ``retrieved_docs``.

    查询使用 :func:`_build_queries` 的结果——改写词与原始问题各自走一遍
    双路召回（改写词漂移时裸问题兜底；首轮改写词即原问题时不重复检索）。
    BM25 仅在 ``VECTORSTORE_PROVIDER=chroma`` 时启用，其他后端只走向量路
    并记录警告；BM25 索引暂不可用（如知识库为空尚未入库）时同样降级为
    仅向量召回，不让整条链路失败。

    Args:
        state: 当前图状态，需包含 ``question``，可选 ``rewritten_query``。

    Returns:
        仅包含 ``retrieved_docs`` 字段的状态更新字典（dict 形态文档）。
    """
    queries = _build_queries(state)
    provider = Config.VECTORSTORE_PROVIDER.lower()
    if provider != "chroma":
        logger.warning(
            "VECTORSTORE_PROVIDER=%s 不支持 BM25 导出建索引，本次仅走向量召回。",
            provider,
        )

    candidates: list[Document] = []
    use_bm25 = provider == "chroma"
    for query in queries:
        candidates.extend(_vectorstore_search(query))
        if not use_bm25:
            continue
        try:
            candidates.extend(_bm25_search(query))
        except ValueError:
            # 典型场景：集合为空导致 BM25 无法建索引，降级为仅向量路；
            # 失败是持久状态，本轮剩余查询不再重试
            logger.warning("BM25 索引不可用（知识库可能为空），本次仅走向量召回。")
            use_bm25 = False

    merged = _dedup_documents(candidates)
    logger.info(
        "%d 条查询双路召回 %d 篇，去重后 %d 篇",
        len(queries),
        len(candidates),
        len(merged),
    )
    return {"retrieved_docs": [doc_to_dict(doc) for doc in merged]}
