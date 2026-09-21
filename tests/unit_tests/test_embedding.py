"""嵌入模型集成冒烟测试.

需要本地 bge-m3 模型权重，属于慢测试，默认不执行::

    .venv/bin/pytest tests/unit_tests/test_embedding.py -m integration
"""

import pytest

pytestmark = pytest.mark.integration


def test_create_embedding_model_returns_vectors_and_caches() -> None:
    """工厂可正常产出向量，且重复调用复用同一实例（避免重复加载权重）."""
    from agent.embedding.embedding_factory import create_embedding_model

    embed = create_embedding_model()
    vector = embed.embed_query("你好")
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(value, float) for value in vector)

    assert create_embedding_model() is embed
