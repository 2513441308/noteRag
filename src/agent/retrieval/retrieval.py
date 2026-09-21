"""检索器工厂：基于已构建的向量库创建 LangChain Retriever.

本模块是纯组件，不读取任何全局配置：向量库实例、召回数量、
检索策略均由调用方显式传入，既可单独使用，也可由编排层注入。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def get_retriever(
    vectorstore: VectorStore,
    *,
    k: int = 3,
    search_type: Literal[
        "similarity", "mmr", "similarity_score_threshold"
    ] = "similarity",
    search_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseRetriever:
    """从向量库构造检索器.

    Args:
        vectorstore: 已构建的向量库实例（Chroma / PGVector 等）。
        k: 召回文档数量。
        search_type: 检索策略，similarity 为向量相似度，mmr 为最大边际相关性，
            similarity_score_threshold 为相似度阈值过滤。
        search_kwargs: 透传给底层检索的参数，例如 metadata 过滤 ``filter``、
            mmr 的 ``fetch_k``、阈值策略的 ``score_threshold``。
        **kwargs: 透传给 ``as_retriever`` 的其他参数。

    Returns:
        可直接 ``.invoke(query)`` / 接入链的检索器实例。
    """
    # 兜底为空字典，避免与默认参数共享可变对象
    final_search_kwargs = search_kwargs.copy() if search_kwargs is not None else {}
    final_search_kwargs["k"] = k

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=final_search_kwargs,
        **kwargs,
    )
