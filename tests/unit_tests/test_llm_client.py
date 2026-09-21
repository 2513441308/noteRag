"""LLM 客户端联网冒烟测试.

需要 .env 中配置 DEFAULT_LLM_API_KEY 且外网可达，默认不执行::

    .venv/bin/pytest tests/unit_tests/test_llm_client.py -m integration
"""

import pytest

pytestmark = pytest.mark.integration


def test_chat_model_invoke() -> None:
    """按默认配置创建聊天模型并完成一次最简单的问答往返."""
    from agent.llm.llm_client import create_chat_model

    llm = create_chat_model()
    response = llm.invoke("只回复两个字：你好")
    assert response.content
