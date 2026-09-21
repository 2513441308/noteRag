"""嵌入模型工厂：按 Config 创建本地 BGE 或云端嵌入模型并缓存实例.

所有 provider（含云端 dashscope/openai）的实例都统一走缓存，避免重复
创建 httpx 客户端与连接池；构建过程按 provider 分派给 ``_build_embedding_model``。
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.embeddings import Embeddings

from agent.config import Config

# 云端服务商默认 base_url 映射
_EMBED_BASE_URLS = {
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
}

_INSTANCE_CACHE: dict[tuple[str, str], Embeddings] = {}
_INSTANCE_CACHE_LOCK = threading.Lock()


def create_embedding_model(
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """创建嵌入模型实例（同 provider+model 全局只构建一次）.

    Args:
        model: 嵌入模型名称，默认读取 Config.EMBEDDING_MODEL。
        provider: 嵌入服务商，默认读取 Config.EMBEDDING_PROVIDER。
        api_key: 服务商API密钥，本地BGE不需要密钥。
        base_url: API接口地址，云端服务使用。
        **kwargs: 透传给嵌入模型类的额外参数。

    Returns:
        LangChain 标准嵌入模型对象。

    Raises:
        云端服务商缺少 API Key，或者服务商不支持时抛出 ValueError。
    """
    provider = (provider or Config.EMBEDDING_PROVIDER).strip().lower()
    # 与 llm_client 保持一致：仅在未显式传入时回退默认值，空串也是显式选择
    model = model if model is not None else Config.EMBEDDING_MODEL
    api_key = api_key or Config.EMBEDDING_API_KEY
    base_url = base_url or Config.EMBEDDING_BASE_URL or _EMBED_BASE_URLS.get(provider)

    cache_key = (provider, model)
    with _INSTANCE_CACHE_LOCK:
        cached = _INSTANCE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    # 重型构建放在锁外执行，避免首次加载本地模型时长时间阻塞其他实例的缓存命中
    ins = _build_embedding_model(provider, model, api_key, base_url, kwargs)

    with _INSTANCE_CACHE_LOCK:
        _INSTANCE_CACHE[cache_key] = ins
    return ins


def _build_embedding_model(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    kwargs: dict[str, Any],
) -> Embeddings:
    """按 provider 实际构建嵌入模型实例（本函数不处理缓存）.

    Args:
        provider: 已归一化为小写的提供方名称。
        model: 模型名称或本地模型目录。
        api_key: 云端 API 密钥，bge 本地模型不使用。
        base_url: 云端兼容协议端点。
        kwargs: 透传给嵌入模型类的额外参数。

    Returns:
        未缓存的嵌入模型新实例。

    Raises:
        云端提供方缺少 API Key，或 provider 不受支持时抛出 ValueError。
    """
    if provider == "dashscope":
        if not api_key:
            raise ValueError(
                "服务商 dashscope 需要配置 EMBEDDING_API_KEY，请检查 .env 文件。"
            )
        # DashScope 官方嵌入目前无独立合作包，使用 langchain-community 提供的实现
        from langchain_community.embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            model=model,
            dashscope_api_key=api_key,
            **kwargs,
        )

    if provider == "openai":
        if not api_key:
            raise ValueError(
                "服务商 openai 需要配置 EMBEDDING_API_KEY，请检查 .env 文件。"
            )
        # 与 LLM 侧统一，使用 langchain-openai 合作包而非 community 的同名实现
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        # OpenAIEmbeddings 的 api_key 静态类型为 SecretStr，与 llm_client 同样显式包装
        return OpenAIEmbeddings(
            model=model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            **kwargs,
        )

    if provider == "bge":
        from langchain_huggingface import HuggingFaceEmbeddings
        from transformers.utils.logging import disable_progress_bar

        # transformers 未导出 py.typed，关闭进度条的调用无法静态检查
        disable_progress_bar()  # type: ignore[no-untyped-call]

        model_kwargs = kwargs.pop("model_kwargs", {"device": "cpu"})
        encode_kwargs = kwargs.pop(
            "encode_kwargs",
            {
                "normalize_embeddings": True,
                "batch_size": Config.EMBEDDING_BATCH_SIZE,
            },
        )
        # query_encode_kwargs 非空时会整体替代 encode_kwargs，需带上相同设置；
        # 同时为中文 BGE 查询拼接检索指令，保持与旧版 HuggingFaceBgeEmbeddings 一致
        query_encode_kwargs = kwargs.pop(
            "query_encode_kwargs",
            {**encode_kwargs, "prompt": "为这个句子生成表示以用于检索相关文章："},
        )
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_encode_kwargs=query_encode_kwargs,
            show_progress=False,
            **kwargs,
        )

    raise ValueError(f"不支持的嵌入服务商: {provider}")
