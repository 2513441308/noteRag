"""Rerank 精排节点：对双路召回结果重排取 top-k，并产出引用来源.

检索组件 ``build_reranker`` / ``rerank_documents`` 是不读配置的纯组件，
本模块作为编排层从 ``Config`` 注入全部 ``RERANK_*`` 参数。重排模型
（本地 CrossEncoder 权重或云端客户端）懒加载且全局只构建一次。

精排后还有两道上下文加工：

- **熔断**：``RERANK_SCORE_THRESHOLD`` 非空时，若最高分仍低于阈值，
  判定知识库未命中并清空候选（交给生成节点的「未找到」分支），
  避免候选全都不相关时「矮子里拔将军」硬答；
- **邻域扩展**：``RERANK_CONTEXT_WINDOW>0`` 时，按命中块的
  source + chunk_index 从 Chroma 捞回前后兄弟块，小块命中、带语境
  喂给生成，缓解长文档切分碎片化。仅 Chroma 支持，任何失败都静默
  降级为只使用命中块，不阻断主链路。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from agent.config import Config
from agent.retrieval.bm25 import fetch_neighbor_documents
from agent.retrieval.reranker import RerankerScorer, build_reranker, rerank_documents
from agent.state.state import MainState
from agent.utils.doc_utils import dict_to_doc, doc_to_dict

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker() -> RerankerScorer:
    """按 ``Config`` 构建（或复用）重排模型，本地/云端对节点透明."""
    return build_reranker(
        Config.RERANK_PROVIDER,
        Config.RERANK_MODEL,
        device=Config.RERANK_DEVICE,
        api_key=Config.RERANK_API_KEY,
        base_url=Config.RERANK_BASE_URL,
        timeout=Config.RERANK_API_TIMEOUT,
        max_retries=Config.RERANK_MAX_RETRIES,
    )


def extract_sources(docs: list[dict[str, Any]]) -> list[str]:
    """按文档顺序提取 ``metadata["source"]`` 并去重保序.

    Args:
        docs: state 中字典形态的文档列表（通常为 ranked_docs）。

    Returns:
        去重后的来源路径列表；缺失 source 字段的文档跳过。
    """
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        metadata = doc.get("metadata")
        source = metadata.get("source") if isinstance(metadata, dict) else None
        if source and source not in seen:
            seen.add(str(source))
            sources.append(str(source))
    return sources


def _is_indexed_chunk(doc: Document) -> bool:
    """文档是否带有邻域扩展所需的 source + 整数 chunk_index 元数据."""
    metadata = doc.metadata or {}
    return isinstance(metadata.get("source"), str) and isinstance(
        metadata.get("chunk_index"), int
    )


def _expand_with_neighbors(ranked: list[Document]) -> list[Document]:
    """把命中块前后 ``RERANK_CONTEXT_WINDOW`` 片的同文档兄弟块补进上下文.

    输出按**来源分组、组内按 chunk_index 升序**：来源顺序遵循命中块的
    首次出现顺序（与 sources 编号一致），组内恢复文档原序，让生成节点
    看到连贯论述而不是按分数打乱的碎片。缺索引元数据（旧库/非 Markdown）
    的块无法扩展，原样追加在末尾。

    Args:
        ranked: 精排/直通截断后的命中文档。

    Returns:
        扩展后的文档列表；窗口关闭、非 Chroma 后端或取数失败时原样返回。
    """
    window = Config.RERANK_CONTEXT_WINDOW
    if window <= 0 or not ranked:
        return ranked
    if not any(_is_indexed_chunk(doc) for doc in ranked):
        return ranked
    if Config.VECTORSTORE_PROVIDER.lower() != "chroma":
        logger.debug("非 Chroma 后端不支持邻域扩展，跳过。")
        return ranked

    try:
        # 复用 rag 节点缓存的同一个 Chroma 单例；懒导入避免节点间顶层耦合
        from agent.nodes.rag import _get_vectorstore_from_config

        neighbors = fetch_neighbor_documents(
            _get_vectorstore_from_config(), ranked, window=window
        )
    except Exception:
        # 邻域是增强而非必需：集合异常/后端不匹配都降级为仅命中块
        logger.warning("邻域扩展失败，本次仅使用命中块。", exc_info=True)
        return ranked

    ordered_sources: list[str] = []
    by_source: dict[str, list[Document]] = {}
    tail: list[Document] = []
    for doc in ranked:
        metadata = doc.metadata or {}
        source = metadata.get("source")
        if _is_indexed_chunk(doc):
            assert isinstance(source, str)
            if source not in by_source:
                by_source[source] = []
                ordered_sources.append(source)
            by_source[source].append(doc)
        else:
            tail.append(doc)

    for (source, _index), neighbor in neighbors.items():
        by_source.setdefault(source, []).append(neighbor)

    expanded: list[Document] = []
    for source in ordered_sources:
        group = by_source[source]
        # 稳定排序：同序号保持命中块原序，无 chunk_index 的块不会进此分组
        group.sort(key=lambda d: int((d.metadata or {})["chunk_index"]))
        expanded.extend(group)
    expanded.extend(tail)
    logger.info("邻域扩展：命中 %d 片 → 补全为 %d 片", len(ranked), len(expanded))
    return expanded


def node_rerank(state: MainState) -> dict[str, object]:
    """对 ``retrieved_docs`` 精排取 top-k，写入 ``ranked_docs`` 与 ``sources``.

    - ``RERANK_ENABLED=true``（默认）：用重排模型按相关度降序，取 ``RERANK_TOP_K``；
    - ``RERANK_ENABLED=false``：不加载模型，保留召回原序直通并截断到 top-k；
    - 无召回文档时两个字段均为空列表。

    打分查询与召回保持一致，优先用 ``rewritten_query``（指代消解后的
    自包含查询对 cross-encoder 更友好），缺失时回退原始 ``question``。
    开启重排且配置了 ``RERANK_SCORE_THRESHOLD`` 时，最高分低于阈值则
    熔断清空候选；通过熔断（或直通）后再做邻域扩展。

    Args:
        state: 当前图状态，需包含 ``retrieved_docs``。

    Returns:
        包含 ``ranked_docs``（dict 形态）与 ``sources`` 的状态更新字典。
    """
    raw_docs: list[dict[str, Any]] = list(state.get("retrieved_docs", []))
    documents = [dict_to_doc(doc) for doc in raw_docs]
    query = state.get("rewritten_query") or state["question"]

    if not documents:
        logger.info("无召回文档，跳过 rerank。")
        ranked: list[Document] = []
    elif Config.RERANK_ENABLED:
        ranked = rerank_documents(
            _get_reranker(), query, documents, top_k=Config.RERANK_TOP_K
        )
        threshold = Config.RERANK_SCORE_THRESHOLD
        top_score = float(ranked[0].metadata.get("rerank_score", 0.0))
        if threshold is not None and top_score < threshold:
            # 全部候选都不够相关：不硬答，交给生成节点的「知识库未命中」分支
            logger.info(
                "重排熔断：最高分 %.4f 低于阈值 %.4f，判定知识库未命中。",
                top_score,
                threshold,
            )
            ranked = []
    else:
        # 直通分支仍截断，保证送入生成的上下文规模可控
        ranked = documents[: Config.RERANK_TOP_K]
        logger.info(
            "RERANK_ENABLED=false，召回结果直通并截断到 top_k=%d。",
            Config.RERANK_TOP_K,
        )

    ranked = _expand_with_neighbors(ranked)

    ranked_dicts = [doc_to_dict(doc) for doc in ranked]
    return {
        "ranked_docs": ranked_dicts,
        "sources": extract_sources(ranked_dicts),
    }
