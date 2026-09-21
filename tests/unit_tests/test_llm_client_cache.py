"""create_chat_model 实例缓存与参数处理的离线单元测试.

通过向 sys.modules 注入假的 langchain_deepseek / langchain_openai 模块，
避免构造真实 SDK 客户端（不联网、不创建 httpx 连接池）。
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from pydantic import SecretStr

from agent.config import Config
from agent.llm import llm_client
from agent.llm.llm_client import create_chat_model


class _RecordingModel:
    """记录构造参数的假模型基类。"""

    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        type(self).calls.append(kwargs)


class _FakeDeepSeek(_RecordingModel):
    calls: list[dict[str, Any]] = []


class _FakeOpenAI(_RecordingModel):
    calls: list[dict[str, Any]] = []


@pytest.fixture(autouse=True)
def _isolate_model_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """固定配置、注入假 SDK 模块，并在测试前后清空缓存与调用记录。"""
    monkeypatch.setattr(Config, "DEFAULT_LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "DEFAULT_LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(Config, "DEFAULT_LLM_MODEL", "test-model")
    monkeypatch.setattr(Config, "DEFAULT_LLM_TEMPERATURE", 0.0)
    monkeypatch.setattr(Config, "DEFAULT_LLM_BASE_URL", None)

    fake_deepseek = types.ModuleType("langchain_deepseek")
    fake_deepseek.ChatDeepSeek = _FakeDeepSeek  # type: ignore[attr-defined]
    fake_openai = types.ModuleType("langchain_openai")
    fake_openai.ChatOpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_deepseek", fake_deepseek)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_openai)

    for cls in (_FakeDeepSeek, _FakeOpenAI):
        cls.calls.clear()
    llm_client._get_cached_model.cache_clear()

    yield

    llm_client._get_cached_model.cache_clear()


def test_same_params_reuse_cached_instance() -> None:
    """相同参数两次创建应命中缓存，只构造一次且返回同一实例。"""
    first = create_chat_model()
    second = create_chat_model()

    assert first is second
    assert len(_FakeDeepSeek.calls) == 1
    assert _FakeDeepSeek.calls[0]["model"] == "test-model"
    assert _FakeDeepSeek.calls[0]["temperature"] == 0.0


def test_different_temperature_separate_instances() -> None:
    """temperature 不同属于不同缓存键。"""
    zero = create_chat_model(temperature=0.0)
    warm = create_chat_model(temperature=0.5)

    assert zero is not warm
    assert len(_FakeDeepSeek.calls) == 2


def test_different_provider_separate_instances() -> None:
    """provider 不同走不同实现类并分别缓存。"""
    deepseek_model = create_chat_model(provider="deepseek")
    qwen_model = create_chat_model(provider="qwen")

    assert deepseek_model is not qwen_model
    assert isinstance(deepseek_model, _FakeDeepSeek)
    assert isinstance(qwen_model, _FakeOpenAI)
    assert _FakeOpenAI.calls[0]["base_url"] == llm_client._BASE_URLS["qwen"]

    # 再次请求 qwen 应命中 OpenAI 侧缓存
    assert create_chat_model(provider="qwen") is qwen_model
    assert len(_FakeOpenAI.calls) == 1


def test_explicit_zero_temperature_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式传 temperature=0 不得因 ``or`` 回退到非零默认值。"""
    monkeypatch.setattr(Config, "DEFAULT_LLM_TEMPERATURE", 0.7)

    model = create_chat_model(temperature=0)

    assert _FakeDeepSeek.calls[0]["temperature"] == 0
    assert model.init_kwargs["temperature"] == 0


def test_kwargs_bypass_cache() -> None:
    """传入额外参数时必须绕过缓存，每次单独构建。"""
    first = create_chat_model(timeout=10)
    second = create_chat_model(timeout=10)

    assert first is not second
    assert len(_FakeDeepSeek.calls) == 2
    assert llm_client._get_cached_model.cache_info().currsize == 0


def test_custom_base_url_passes_through() -> None:
    """自定义 base_url 透传给 OpenAI 兼容实现，且不进缓存。"""
    model = create_chat_model(provider="openai", base_url="https://vllm.example.com/v1")

    assert isinstance(model, _FakeOpenAI)
    assert _FakeOpenAI.calls[0]["base_url"] == "https://vllm.example.com/v1"
    assert llm_client._get_cached_model.cache_info().currsize == 0


def test_config_base_url_used_for_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config.DEFAULT_LLM_BASE_URL 生效于缓存路径，覆盖内置端点。"""
    monkeypatch.setattr(Config, "DEFAULT_LLM_BASE_URL", "https://gateway.test/v1")

    model = create_chat_model(provider="openai")

    assert isinstance(model, _FakeOpenAI)
    assert _FakeOpenAI.calls[0]["base_url"] == "https://gateway.test/v1"


def test_explicit_base_url_beats_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式传入的 base_url 优先级高于环境变量配置。"""
    monkeypatch.setattr(Config, "DEFAULT_LLM_BASE_URL", "https://gateway.test/v1")

    create_chat_model(provider="openai", base_url="https://explicit.test/v1")

    assert _FakeOpenAI.calls[0]["base_url"] == "https://explicit.test/v1"


def test_api_key_is_wrapped_as_secretstr() -> None:
    """传入模型的 api_key 必须是 SecretStr，且值正确。"""
    create_chat_model()

    api_key = _FakeDeepSeek.calls[0]["api_key"]
    assert isinstance(api_key, SecretStr)
    assert api_key.get_secret_value() == "test-key"


def test_missing_api_key_raises_cached_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认路径缺 API Key 时抛出 ValueError。"""
    monkeypatch.setattr(Config, "DEFAULT_LLM_API_KEY", "")

    with pytest.raises(ValueError, match="DEFAULT_LLM_API_KEY"):
        create_chat_model()


def test_missing_api_key_raises_uncached_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """带 kwargs 的绕过缓存路径缺 API Key 时同样抛出 ValueError。"""
    monkeypatch.setattr(Config, "DEFAULT_LLM_API_KEY", "")

    with pytest.raises(ValueError, match="DEFAULT_LLM_API_KEY"):
        create_chat_model(timeout=5)


def test_unsupported_provider_raises() -> None:
    """不支持的服务商抛出 ValueError 并提示可选项。"""
    with pytest.raises(ValueError, match="不支持的 DEFAULT_LLM_PROVIDER"):
        create_chat_model(provider="anthropic")
