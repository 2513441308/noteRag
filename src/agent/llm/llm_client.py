"""LLM 客户端工厂：根据配置创建聊天大模型.

通过 ``agent.config.Config`` 读取 ``.env``，由 ``DEFAULT_LLM_PROVIDER`` 切换：

1. ``deepseek``（默认）: ``langchain-deepseek`` 的 ChatDeepSeek；
2. ``openai`` / ``qwen`` 等: OpenAI 兼容协议（ChatOpenAI + base_url），
   可对接通义千问、智谱、本地 vLLM / Ollama 等服务。

默认参数创建的实例按 ``(provider, model, temperature)`` 缓存复用，
避免重复构建底层 httpx 客户端；传入额外 ``kwargs`` 时绕过缓存逐个构建。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import SecretStr

from agent.config import Config

# 常见 OpenAI 兼容服务商的接入地址
_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def _require_secret_key() -> SecretStr:
    """读取并校验 API Key，包装为 SecretStr 返回.

    Raises:
        ValueError: 未配置 ``DEFAULT_LLM_API_KEY`` 时抛出。
    """
    api_key = Config.DEFAULT_LLM_API_KEY
    if not api_key:
        raise ValueError(
            "未找到 API Key，请在项目根目录 .env 中配置 DEFAULT_LLM_API_KEY。"
        )
    # 两类 Chat 模型的 api_key 字段均标注为 SecretStr，显式包装避免依赖
    # pydantic 运行时隐式转换（类型检查通过，也防止 key 进入明文 repr）
    return SecretStr(api_key)


def _build_model(
    provider: str,
    model: str,
    temperature: float,
    *,
    base_url: str | None,
    extra_kwargs: dict[str, Any],
) -> Any:
    """按服务商实际实例化聊天模型（不走缓存的内部构建函数）.

    Args:
        provider: 服务商标识，支持 deepseek / openai / qwen。
        model: 模型名。
        temperature: 采样温度。
        base_url: OpenAI 兼容服务的接入地址，deepseek 路径忽略。
        extra_kwargs: 透传给 Chat 模型类的额外参数。

    Returns:
        ChatDeepSeek 或 ChatOpenAI 实例。

    Raises:
        ValueError: 缺少 API Key 或服务商不支持时抛出。
    """
    secret_key = _require_secret_key()

    if provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model,
            api_key=secret_key,
            temperature=temperature,
            **extra_kwargs,
        )

    if provider in _BASE_URLS or provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=secret_key,
            base_url=base_url,
            temperature=temperature,
            **extra_kwargs,
        )

    raise ValueError(
        f"不支持的 DEFAULT_LLM_PROVIDER: {provider!r}，可选 deepseek / openai / qwen"
    )


@lru_cache(maxsize=8)
def _get_cached_model(provider: str, model: str, temperature: float) -> Any:
    """按 (provider, model, temperature) 返回缓存的模型实例.

    模型对象可安全跨节点共享：``with_structured_output`` 等调用会返回
    新的 Runnable 包装对象，不会修改被缓存的底层实例。
    """
    return _build_model(
        provider,
        model,
        temperature,
        base_url=Config.DEFAULT_LLM_BASE_URL or _BASE_URLS.get(provider),
        extra_kwargs={},
    )


def create_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """创建并返回一个聊天模型实例.

    不传额外参数时，按 ``(provider, model, temperature)`` 复用缓存实例；
    传入任意 ``kwargs``（如自定义 ``base_url``、``http_client``）时绕过缓存，
    因为这些参数不参与缓存键且可能不可哈希。

    Args:
        model: 模型名，默认取 ``Config.DEFAULT_LLM_MODEL``。
        temperature: 采样温度，RAG 场景默认 0（更确定、少幻觉）。
            显式传 0 会被保留，不会回退到默认值。
        provider: 服务商，默认取 ``Config.DEFAULT_LLM_PROVIDER``。
        **kwargs: 透传给具体 Chat 模型类的额外参数。

    Returns:
        ChatDeepSeek 或 ChatOpenAI 实例。

    Raises:
        ValueError: 缺少 API Key 或服务商不支持时抛出。
    """
    provider = (provider or Config.DEFAULT_LLM_PROVIDER).strip().lower()
    model = model if model is not None else Config.DEFAULT_LLM_MODEL
    # 注意不能用 ``temperature or 默认值``：显式传入 0 会被误判为假值
    temperature = (
        temperature if temperature is not None else Config.DEFAULT_LLM_TEMPERATURE
    )

    if kwargs:
        base_url = (
            kwargs.pop("base_url", None)
            or Config.DEFAULT_LLM_BASE_URL
            or _BASE_URLS.get(provider)
        )
        return _build_model(
            provider,
            model,
            temperature,
            base_url=base_url,
            extra_kwargs=kwargs,
        )

    return _get_cached_model(provider, model, temperature)
