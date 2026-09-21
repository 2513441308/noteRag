"""向量数据库工厂（纯组件，不依赖项目配置）.

所有连接信息均通过函数参数显式传入，本模块不读取任何环境变量或 Config，
因此既可以被 :mod:`agent.ingestion.loader` 的编排函数注入配置使用，
也可以在脚本/notebook 中脱离项目配置单独调用。

支持：
- ``chroma``：本地持久化（或内存模式）；
- ``postgres``：PGVector，需要可选依赖 ``langchain-postgres``。

实例按 (provider, collection_name, 连接目标) 缓存复用。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)

# (provider, collection_name, target) -> 向量库实例
_vectorstores: dict[tuple[str, str, str], VectorStore] = {}


def build_postgres_dsn(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> str:
    """根据分项连接参数拼出 psycopg3 风格的 PostgreSQL DSN.

    Args:
        host: 主机地址。
        port: 端口。
        user: 用户名。
        password: 密码。
        database: 数据库名。

    Returns:
        ``postgresql+psycopg://user:password@host:port/database`` 形式的 DSN。
    """
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


def create_chroma(
    collection_name: str,
    embedding_model: Embeddings,
    persist_directory: str | Path | None = None,
    **kwargs: Any,
) -> VectorStore:
    """创建 Chroma 向量库.

    Args:
        collection_name: 集合名称。
        embedding_model: 嵌入模型实例。
        persist_directory: 持久化目录；None 时为纯内存模式。
        **kwargs: 透传给 ``langchain_chroma.Chroma`` 的额外参数。

    Returns:
        Chroma 实例。
    """
    from langchain_chroma import Chroma

    persist_dir = str(persist_directory) if persist_directory is not None else None
    if persist_dir:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        f"初始化 Chroma：collection={collection_name}, persist_dir={persist_dir or '(内存模式)'}"
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
        **kwargs,
    )


def create_postgres(
    collection_name: str,
    embedding_model: Embeddings,
    connection: str,
    **kwargs: Any,
) -> VectorStore:
    """创建 PGVector 向量库.

    Args:
        collection_name: 集合（表）名称。
        embedding_model: 嵌入模型实例。
        connection: PostgreSQL DSN 或已建立的连接对象，可用
            :func:`build_postgres_dsn` 生成分项参数对应的 DSN。
        **kwargs: 透传给 ``PGVector`` 的额外参数。

    Returns:
        PGVector 实例。

    Raises:
        未安装可选依赖 langchain-postgres 时抛出 ImportError。
    """
    try:
        from langchain_postgres import PGVector  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError(
            "使用 postgres 向量库需要安装可选依赖，请执行：uv add langchain-postgres"
        ) from e

    logger.info(f"初始化 PGVector：collection={collection_name}")
    return PGVector(  # type: ignore[no-any-return]  # 可选依赖未装时无类型桩
        collection_name=collection_name,
        connection=connection,
        embeddings=embedding_model,
        **kwargs,
    )


def get_vectorstore(
    collection_name: str,
    embedding_model: Embeddings,
    *,
    provider: str = "chroma",
    persist_directory: str | Path | None = None,
    connection: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """获取（或复用）向量数据库实例，连接信息全部显式传入.

    Args:
        collection_name: 集合名称，实例按 (provider, 集合, 连接目标) 缓存。
        embedding_model: 嵌入模型实例（LangChain Embeddings 接口）。
        provider: 向量库类型，``chroma`` 或 ``postgres``，默认 chroma。
        persist_directory: chroma 持久化目录，None 为内存模式。
        connection: postgres 的 DSN/连接；provider=postgres 时必填。
        **kwargs: 透传给底层向量库构造函数的额外参数。

    Returns:
        向量数据库实例。

    Raises:
        provider 不支持，或 postgres 未提供 connection 时抛出 ValueError。
    """
    provider = provider.lower()
    target = str(persist_directory) if provider == "chroma" else (connection or "")
    cache_key = (provider, collection_name, target)
    if cache_key in _vectorstores:
        return _vectorstores[cache_key]

    if provider == "chroma":
        store = create_chroma(
            collection_name, embedding_model, persist_directory, **kwargs
        )
    elif provider == "postgres":
        if not connection:
            raise ValueError(
                "provider=postgres 时必须提供 connection（DSN 或连接对象）"
            )
        store = create_postgres(collection_name, embedding_model, connection, **kwargs)
    else:
        raise ValueError(f"不支持的向量库类型：{provider}，可选：chroma / postgres")

    _vectorstores[cache_key] = store
    return store
