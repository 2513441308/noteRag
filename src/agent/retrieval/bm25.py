"""BM25 关键词检索组件（归一化中英混合分词）.

与向量检索互补的第二路召回：BM25 基于精确词项匹配，对函数名、配置名、
报错信息等关键词查询更敏感。索引在内存中构建，不落盘——图启动时从
向量库集合导出全量 chunk 建一次索引即可，适合万级文档以内的知识库。

纯组件，不读取任何全局配置：文档、召回条数均由调用方显式传入。
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Callable, Sequence

# jieba 未提供 py.typed 桩，已在 pyproject.toml 的 mypy overrides 中豁免
import jieba
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

# 分词器类型：文本 -> token 列表
Tokenizer = Callable[[str], list[str]]

# 驼峰边界：StateGraph -> State|Graph；HTTPServer -> HTTP|Server
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# 纯标点/符号 token（含下划线，jieba 会把 _ 单独切出来）
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")
# token 内部的连续字母数字片段，用于按符号切出英文子串
_ALNUM_RUN_RE = re.compile(r"[A-Za-z0-9]+")


def chinese_tokenize(text: str) -> list[str]:
    """Jieba 精确模式分词，剔除纯空白 token.

    .. deprecated::
        大小写敏感且标点（``_`` ``.`` ``(`` 等）会作为噪声词项入索引，
        仅保留用于对照；新代码请用 :func:`normalized_tokenize`。

    Args:
        text: 待分词文本（文档块或查询）。

    Returns:
        分词结果，如 "条件边怎么写" -> ["条件", "边", "怎么", "写"]。
    """
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def normalized_tokenize(text: str) -> list[str]:
    """中英混合归一化分词：jieba 切词后做大小写归一与驼峰展开.

    解决英文/代码语料下的两个精确匹配问题（文档与查询走同一函数，保证一致）：

    1. **大小写归一**：所有含拉丁字母的 token 统一小写，文档 ``StateGraph``
       与查询 ``stategraph`` / ``STATEGRAPH`` 不再漏匹配；
    2. **驼峰双形态索引**：``StateGraph`` 同时产出整体词 ``stategraph`` 与
       子词 ``state`` / ``graph``，覆盖 "state graph" 这类带空格/分词写法。
       子词通常高频低 IDF，罕见整体词仍保留区分度；
    3. **标点过滤**：``_`` ``.`` ``(`` ``=`` 等 jieba 单独切出的纯符号
       token 不入索引（蛇形 ``add_conditional_edges`` 与连字符
       ``add-conditional-edges`` 因此归一为同一组词项）；
    4. **中文无损保留**：中文 token 不经正则切词，``lower()`` 不影响汉字，
       如 "条件边" -> ["条件", "边"]。

    Args:
        text: 待分词文本（文档块或查询）。

    Returns:
        归一化后的 token 列表，不含空白与纯符号 token。
    """
    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip()
        if not token or _PUNCT_ONLY_RE.match(token):
            continue

        # 候选形态：整体小写 + 每个字母数字片段的整体形态与驼峰子词。
        # StateGraph -> stategraph + state + graph；标点粘连也能保住整体词
        forms: list[str] = [token.lower()]
        for run in _ALNUM_RUN_RE.findall(token):
            forms.append(run.lower())
            parts = _CAMEL_BOUNDARY_RE.split(run)
            if len(parts) > 1:
                forms.extend(part.lower() for part in parts if part)

        # 仅在同一原始 token 内去重，避免整体词/子词重复人为抬高词频；
        # 不同原始 token 产出的相同词不去重（那是合法 term frequency）
        seen: set[str] = set()
        for form in forms:
            if form and form not in seen:
                seen.add(form)
                tokens.append(form)
    return tokens


def create_bm25_retriever(
    documents: Sequence[Document],
    *,
    k: int = 5,
    tokenizer: Tokenizer = normalized_tokenize,
    **kwargs: Any,
) -> BaseRetriever:
    """用给定文档构建内存 BM25 检索器.

    Args:
        documents: 建索引的全部文档（通常来自 :func:`load_all_documents`）。
        k: 每次检索返回的文档条数。
        tokenizer: 分词函数，默认 :func:`normalized_tokenize`（大小写归一、
            驼峰展开、标点过滤，适合英文/中英混合语料）；纯中文历史行为可用
            :func:`chinese_tokenize`。
        **kwargs: 透传给 ``BM25Retriever.from_documents`` 的参数，
            如 ``bm25_params={"k1": 2.0, "b": 0.75}``。

    Returns:
        可直接 ``.invoke(query)`` 的 BM25 检索器，
        返回的 Document 保留建索引时的 metadata。

    Raises:
        文档列表为空、无法建立索引时抛出 ValueError。
    """
    documents = list(documents)
    if not documents:
        raise ValueError("构建 BM25 索引至少需要 1 个文档，请先完成知识库入库。")

    # 与 loader.py 一致：导入 community 的瞬间压掉其 sunset 横幅
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(
        documents,
        k=k,
        preprocess_func=tokenizer,
        **kwargs,
    )
    logger.info(f"BM25 索引构建完成：{len(documents)} 个文档块，k={k}")
    return retriever


def load_all_documents(vectorstore: VectorStore) -> list[Document]:
    """从向量库集合导出全部 chunk 为 ``Document`` 列表.

    用于图启动时把已入库的文档同步到内存 BM25 索引。目前适配
    Chroma（``langchain_chroma.Chroma``，底层暴露 ``_collection``）。

    Args:
        vectorstore: 已写入文档的向量库实例。

    Returns:
        集合内全部文档，``id`` 与 metadata 与入库时一致；空集合返回空列表。

    Raises:
        向量库实现未暴露可遍历的底层集合（如 PGVector）时抛出 TypeError。
    """
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        raise TypeError(
            f"暂不支持从 {type(vectorstore).__name__} 导出全量文档构建 BM25 索引，"
            "当前仅支持 Chroma（需要底层 _collection）。"
        )

    data = collection.get(include=["documents", "metadatas"])
    ids = data.get("ids") or []
    contents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    documents = [
        Document(id=doc_id, page_content=content, metadata=metadata or {})
        for doc_id, content, metadata in zip(ids, contents, metadatas)
    ]
    logger.info(f"从向量库集合导出 {len(documents)} 个文档块用于 BM25 索引")
    return documents


def load_distinct_headers(
    vectorstore: VectorStore,
    *,
    limit: int = 200,
) -> list[str]:
    """从集合 metadata 聚合去重的标题路径，作为改写器的知识库术语表.

    Markdown 切分器给每个 chunk 写入 ``headers``（形如
    ``核心概念 > StateGraph > 条件边``）。标题用词就是文档的真实术语，
    查询改写时向它对齐，可避免把「条件边」改写成文档里不存在的「分支判断」。
    只取 metadata、不拉正文，代价远小于 :func:`load_all_documents`。

    Args:
        vectorstore: 已写入文档的 Chroma 向量库实例。
        limit: 最多返回多少条标题路径（按集合遍历先后，够改写器参考即可）。

    Returns:
        去重保序的标题路径列表；集合为空或全为无标题历史 chunk 时返回空列表。

    Raises:
        向量库实现未暴露 ``_collection``（如 PGVector）时抛出 TypeError。
    """
    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        raise TypeError(
            f"暂不支持从 {type(vectorstore).__name__} 导出标题术语表，"
            "当前仅支持 Chroma（需要底层 _collection）。"
        )

    data = collection.get(include=["metadatas"])
    seen: set[str] = set()
    headers: list[str] = []
    for metadata in data.get("metadatas") or []:
        raw = (metadata or {}).get("headers")
        if not isinstance(raw, str):
            continue
        path = raw.strip()
        if path and path not in seen:
            seen.add(path)
            headers.append(path)
            if len(headers) >= limit:
                break
    return headers


def fetch_neighbor_documents(
    vectorstore: VectorStore,
    anchors: Sequence[Document],
    *,
    window: int = 1,
) -> dict[tuple[str, int], Document]:
    """按命中 chunk 的 ``source`` + ``chunk_index`` 捞回同文档邻域兄弟块.

    小块检索负责「找得准」，本函数负责把命中小块前后 ``window`` 片的原文
    补回来交给生成，解决长文档碎片化导致的语境丢失（small-to-big 的轻量
    平替，不需要两级入库）。同一来源的多个锚点会合并为一个区间，每源只
    发一次 ``collection.get``；锚点自身不返回（调用方已持有且带重排分数）。

    Args:
        vectorstore: 已写入文档的 Chroma 向量库实例。
        anchors: 重排后命中的文档，需带 source 与整数 chunk_index 元数据；
            缺任一者的锚点跳过。
        window: 向前后各扩展的片数，非正数时返回空字典。

    Returns:
        ``(source, chunk_index) -> Document`` 的邻域块映射（不含锚点自身）。

    Raises:
        向量库实现未暴露 ``_collection``（如 PGVector）时抛出 TypeError。
    """
    if window <= 0:
        return {}

    collection = getattr(vectorstore, "_collection", None)
    if collection is None:
        raise TypeError(
            f"暂不支持从 {type(vectorstore).__name__} 捞回邻域块，"
            "当前仅支持 Chroma（需要底层 _collection）。"
        )

    # 每源合并出一个连续的 [lo, hi] 区间，避免多锚点重复查询
    ranges: dict[str, tuple[int, int]] = {}
    anchor_keys: set[tuple[str, int]] = set()
    for doc in anchors:
        metadata = doc.metadata or {}
        source = metadata.get("source")
        index = metadata.get("chunk_index")
        if not isinstance(source, str) or not isinstance(index, int):
            continue
        anchor_keys.add((source, index))
        lo, hi = index - window, index + window
        if source in ranges:
            prev_lo, prev_hi = ranges[source]
            ranges[source] = (min(prev_lo, lo), max(prev_hi, hi))
        else:
            ranges[source] = (lo, hi)

    neighbors: dict[tuple[str, int], Document] = {}
    for source, (lo, hi) in ranges.items():
        where = {
            "$and": [
                {"source": source},
                {"chunk_index": {"$gte": lo}},
                {"chunk_index": {"$lte": hi}},
            ]
        }
        data = collection.get(where=where, include=["documents", "metadatas"])
        for doc_id, content, metadata in zip(
            data.get("ids") or [],
            data.get("documents") or [],
            data.get("metadatas") or [],
        ):
            meta = metadata or {}
            index = meta.get("chunk_index")
            if not isinstance(index, int):
                continue
            key = (source, index)
            if key in anchor_keys or key in neighbors:
                continue
            neighbors[key] = Document(id=doc_id, page_content=content, metadata=meta)
    return neighbors


def build_bm25_from_store(
    vectorstore: VectorStore,
    *,
    k: int = 5,
    tokenizer: Tokenizer = normalized_tokenize,
    **kwargs: Any,
) -> BaseRetriever:
    """便捷编排：从向量库导出全量文档并立即构建 BM25 检索器.

    Args:
        vectorstore: 已写入文档的 Chroma 向量库实例。
        k: 每次检索返回条数。
        tokenizer: 分词函数，默认 :func:`normalized_tokenize`。
        **kwargs: 透传给 :func:`create_bm25_retriever`。

    Returns:
        构建好的 BM25 检索器。

    Raises:
        向量库集合为空（尚未入库）时抛出 ValueError。
    """
    documents = load_all_documents(vectorstore)
    return create_bm25_retriever(documents, k=k, tokenizer=tokenizer, **kwargs)
