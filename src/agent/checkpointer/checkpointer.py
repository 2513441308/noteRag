"""Checkpointer 工厂：按 Config 返回 memory/redis/postgres 单例检查点存储器."""

import logging
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from agent.config import Config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_checkpointer(
    backend: str,
    redis_url: str | None,
    database_url: str | None,
    **kwargs: Any,
) -> BaseCheckpointSaver[Any]:
    """真正的构建逻辑，参数已解析成明确的值.

    Args:
        backend: 后端类型，"memory"、"redis"、"postgres"。
        redis_url: Redis 连接 URL。
        database_url: PostgreSQL 数据库 URL。
        **kwargs: 透传给具体 Saver 构造函数的额外参数。

    Returns:
        Checkpointer 实例。

    Raises:
        backend 无效、或 redis/postgres 缺少连接 URL 时抛出 ValueError。
    """
    if backend == "memory":
        return MemorySaver()

    if backend == "redis":
        if not redis_url:
            raise ValueError("backend=redis 但未配置 CP_REDIS_URL")
        # redis 为可选后端，依赖（langgraph-checkpoint-redis、redis）按需导入，
        # 未安装时 mypy/pyright 报无法解析，已在 pyproject 与行内显式忽略
        from langgraph.checkpoint.redis import (  # pyright: ignore[reportMissingImports]
            RedisSaver,
        )
        from redis import Redis  # pyright: ignore[reportMissingImports]

        client = Redis.from_url(redis_url)
        saver = RedisSaver(redis_client=client, **kwargs)
        saver.setup()
        return saver  # type: ignore[no-any-return]  # 可选依赖无类型桩

    if backend == "postgres":
        if not database_url:
            raise ValueError("backend=postgres 但未配置 CP_POSTGRES_URL")
        # postgres 为可选后端，依赖（langgraph-checkpoint-postgres、psycopg）
        # 按需导入，未安装时静态检查报无法解析，属预期
        from langgraph.checkpoint.postgres import (  # pyright: ignore[reportMissingImports]
            PostgresSaver,
        )
        from psycopg import Connection  # pyright: ignore[reportMissingImports]
        from psycopg.rows import (  # pyright: ignore[reportMissingImports]
            DictRow,
            dict_row,
        )
        from psycopg_pool import ConnectionPool  # pyright: ignore[reportMissingImports]

        pool = ConnectionPool(
            connection_class=Connection[DictRow],
            conninfo=database_url,
            max_size=10,
            min_size=0,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        saver = PostgresSaver(conn=pool, **kwargs)
        saver.setup()
        return saver  # type: ignore[no-any-return]  # 可选依赖无类型桩

    raise ValueError(
        f"不支持的 checkpointer 后端: {backend}，可选 memory/redis/postgres"
    )


def get_checkpointer(**kwargs: Any) -> BaseCheckpointSaver[Any]:
    """按当前配置返回单例 Checkpointer."""
    return _build_checkpointer(
        Config.CHECKPOINTER_BACKEND,
        Config.CP_REDIS_URL,
        Config.CP_POSTGRES_URL,
        **kwargs,
    )
