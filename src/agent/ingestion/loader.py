"""文档加载入口.

按文件后缀分发到 langchain-community 的各类加载器：
LangChain 1.x 把 Text/PyPDF/Python 等通用文件加载器统一维护在
``langchain-community`` 中（https://reference.langchain.com/python/langchain-community/document-loaders ）。
导入该包时会打印一条包级 sunset 横幅，这里仅在导入 community 的瞬间用
``catch_warnings`` 压掉这条特定告警，不影响其他任何 DeprecationWarning。

本模块同时是知识库入库的**核心编排层**：:func:`ingest_documents` 串起
"加载文档 → 结构切分 → 嵌入向量化 → 写入向量库" 完整链路。

配置依赖只存在于编排层（参数传 None 时回落 ``Config``）；下层组件均为
纯函数/纯工厂，连接信息与切分参数全部显式传入，可脱离 Config 单独使用：
切分逻辑见 :mod:`agent.ingestion.markdown_splitter`，
JSON 加载器见 :mod:`agent.ingestion.json_loader`，
向量库工厂见 :mod:`agent.ingestion.vectorstore`。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from agent.config import ROOT_DIR, Config
from agent.embedding.embedding_factory import create_embedding_model
from agent.ingestion.json_loader import GenericJSONLoader
from agent.ingestion.markdown_splitter import (
    MarkdownCodeTextSplitter,
    split_documents,
)
from agent.ingestion.vectorstore import build_postgres_dsn, get_vectorstore

# 向后兼容：允许继续从 loader 直接导入这些符号
__all__ = [
    "GenericJSONLoader",
    "MarkdownCodeTextSplitter",
    "get_vectorstore",
    "ingest_documents",
    "load_dir",
    "load_document",
    "split_documents",
]

logger = logging.getLogger(__name__)

# 仅在此最小作用域内忽略 community 的 sunset 横幅
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.document_loaders import (
        PyPDFLoader,
        PythonLoader,
        TextLoader,
    )


# 不标注 type[BaseLoader]：基类 __init__ 只有 **kwargs，标注后反而无法
# 传 file_path；保留具体加载器类的推断联合类型，调用签名才能精确匹配
_SUPPORT_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".py": PythonLoader,
    # Markdown 本质是纯文本，用 TextLoader 读取即可，无需引入沉重的 unstructured 依赖
    ".md": TextLoader,
    ".markdown": TextLoader,
    # JSONLoader 强制要求调用方提供与数据结构强绑定的 jq_schema，
    # 这里用标准库实现的通用加载器替代，自动兼容对象根/数组根
    ".json": GenericJSONLoader,
}

# 只有这些加载器支持 encoding 参数。
# 注意：必须用精确类匹配而非 issubclass —— PythonLoader 虽继承 TextLoader，
# 但其重写的 __init__ 只接受 file_path，不接受 encoding
_ENCODING_AWARE_LOADERS = frozenset({TextLoader, GenericJSONLoader})


def load_document(
    file_path: Path, encoding: str = "utf-8", **kwargs: Any
) -> list[Document]:
    """加载文档文件.

    Args:
        file_path: 文档文件路径。
        encoding: 文本类文件编码，默认 utf-8。
        **kwargs: 透传给具体加载器的额外参数。

    Returns:
        加载后的文档列表。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件类型不受支持。
    """
    if not file_path.exists():
        logger.error(f"文件不存在：{file_path}")
        raise FileNotFoundError(f"文件不存在：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORT_SUFFIX:
        logger.error(f"不支持的文件类型：{suffix}")
        raise ValueError(f"不支持的文件类型：{file_path.suffix}")

    loader_cls = _SUPPORT_SUFFIX[suffix]
    if loader_cls in _ENCODING_AWARE_LOADERS:
        kwargs["encoding"] = encoding
    # 加载器类联合中含 community 未完全标注类型的类，其 .load() 静态返回 Any
    return loader_cls(file_path, **kwargs).load()  # type: ignore[no-any-return]


def load_dir(
    dir_path: Path,
    recursive: bool = True,
    encoding: str = "utf-8",
) -> list[Document]:
    """加载目录下所有支持的文档文件，跳过加载失败的单个文件并记录日志.

    Args:
        dir_path: 目录路径。
        recursive: 是否递归加载子目录，默认 True。
        encoding: 文本类文件编码，默认 utf-8。

    Returns:
        加载后的文档列表。

    Raises:
        FileNotFoundError: 目录不存在。
        ValueError: 路径不是目录。
    """
    if not dir_path.exists():
        logger.error(f"目录不存在：{dir_path}")
        raise FileNotFoundError(f"目录不存在：{dir_path}")
    if not dir_path.is_dir():
        logger.error(f"路径不是目录：{dir_path}")
        raise ValueError(f"路径不是目录：{dir_path}")

    docs: list[Document] = []
    it = dir_path.rglob("*") if recursive else dir_path.glob("*")
    for file_path in it:
        # 后缀统一小写后再匹配，与 load_document 保持一致
        if not (file_path.is_file() and file_path.suffix.lower() in _SUPPORT_SUFFIX):
            continue
        try:
            docs.extend(load_document(file_path, encoding=encoding))
        except Exception as e:
            # 单个文件失败（损坏/编码错误/无权限等）不影响其余文件入库
            logger.warning(f"跳过加载失败的文件：{file_path} - {type(e).__name__}: {e}")
    return docs


# Chroma 等向量库只接受 str/int/float/bool 类型的 metadata 值，不接受 None/Path
_METADATA_SCALARS = (str, int, float, bool)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """把文档 metadata 规整为向量库可接受的标量字典.

    Args:
        metadata: 原始 metadata。

    Returns:
        值为 str/int/float/bool 的新字典，None 与非标量被剔除或字符串化。
    """
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, _METADATA_SCALARS) else str(value)
    return clean


def _resolve_paths(
    paths: str | Path | list[str | Path],
    recursive: bool,
    encoding: str,
) -> list[Document]:
    """把单个/多个文件或目录路径统一展开为已加载的 Document 列表.

    Args:
        paths: 文件或目录路径，或它们的列表（可混合）。
        recursive: 目录是否递归。
        encoding: 文本类文件编码。

    Returns:
        加载后的文档列表。
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    docs: list[Document] = []
    for raw in paths:
        path = Path(raw)
        try:
            if path.is_dir():
                docs.extend(load_dir(path, recursive=recursive, encoding=encoding))
            else:
                docs.extend(load_document(path, encoding=encoding))
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"跳过加载失败的文件：{path} - {type(e).__name__}: {e}")
    return docs


def ingest_documents(
    paths: str | Path | list[str | Path],
    *,
    collection_name: str | None = None,
    embedding_model: Embeddings | None = None,
    provider: str | None = None,
    persist_directory: str | Path | None = None,
    postgres_dsn: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    code_max_chars: int | None = None,
    recursive: bool = True,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> VectorStore:
    """端到端入库：加载文档 → 结构切分 → 嵌入向量化 → 写入向量库.

    所有配置参数默认 None，表示回落 :class:`agent.config.Config` 中的值；
    显式传参即可覆盖，也可以直接注入已构造好的 ``embedding_model``。
    下层组件不感知 Config，配置注入只发生在本编排层。

    Args:
        paths: 待入库的文件/目录路径，或二者混合的列表。
        collection_name: 向量库集合名，默认 Config.VECTORSTORE_COLLECTION_NAME。
        embedding_model: 已构造的嵌入模型；None 时按 Config 创建。
        provider: 向量库类型 chroma/postgres，默认 Config.VECTORSTORE_PROVIDER。
        persist_directory: chroma 持久化目录，默认 Config.VECTORSTORE_PATH。
        postgres_dsn: postgres DSN；None 时用 Config 的 POSTGRES_* 分项拼接。
        chunk_size: 切分目标字符数，默认 Config.CHUNK_SIZE。
        chunk_overlap: 文本重叠字符数，默认 Config.CHUNK_OVERLAP。
        code_max_chars: 代码块完整保留上限，默认 Config.CODE_MAX_CHARS。
        recursive: 目录是否递归加载，默认 True。
        encoding: 文本类文件编码，默认 utf-8。
        **kwargs: 透传给向量库构造函数的额外参数。

    Returns:
        写入完成的向量库实例（可直接 ``.as_retriever()`` 检索）。
    """
    # ---- 1. 配置解析：显式参数优先，None 回落 Config ----
    collection_name = collection_name or Config.VECTORSTORE_COLLECTION_NAME
    provider = (provider or Config.VECTORSTORE_PROVIDER).lower()
    chunk_size = Config.CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = Config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    code_max_chars = Config.CODE_MAX_CHARS if code_max_chars is None else code_max_chars

    connection: str | None = None
    persist_path: Path | None = None
    if provider == "chroma":
        persist_directory = (
            persist_directory
            if persist_directory is not None
            else Config.VECTORSTORE_PATH
        )
        persist_path = Path(persist_directory)
        if not persist_path.is_absolute():
            persist_path = ROOT_DIR / persist_path
    elif provider == "postgres":
        connection = postgres_dsn or build_postgres_dsn(
            Config.VECTORSTORE_POSTGRES_HOST,
            Config.VECTORSTORE_POSTGRES_PORT,
            Config.VECTORSTORE_POSTGRES_USER,
            Config.VECTORSTORE_POSTGRES_PASSWORD,
            Config.VECTORSTORE_POSTGRES_DB,
        )
    else:
        # 快速失败：避免未知 provider 以 connection=None、persist_path=None
        # 的空配置静默继续，把错误拖到向量库构造深处才暴露
        raise ValueError(
            f"不支持的向量库类型 provider={provider!r}，仅支持 'chroma' / 'postgres'"
        )

    # ---- 2. 加载 → 3. 切分 ----
    docs = _resolve_paths(paths, recursive=recursive, encoding=encoding)
    chunks = split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        code_max_chars=code_max_chars,
    )
    for chunk in chunks:  # 入库边界：规整 metadata，剔除 None/非标量
        chunk.metadata = _sanitize_metadata(chunk.metadata)
    logger.info(f"加载 {len(docs)} 个文档，切分得到 {len(chunks)} 个 chunk")

    # ---- 4. 嵌入模型（注入优先，未注入则按 Config 创建） ----
    if embedding_model is None:
        embedding_model = create_embedding_model()

    # ---- 5. 创建向量库 → 6. 写入 ----
    store = get_vectorstore(
        collection_name,
        embedding_model,
        provider=provider,
        persist_directory=persist_path,
        connection=connection,
        **kwargs,
    )
    ids = store.add_documents(chunks)
    logger.info(
        f"已向 {provider} 集合 {collection_name} 写入 {len(ids or chunks)} 条向量"
    )
    return store
