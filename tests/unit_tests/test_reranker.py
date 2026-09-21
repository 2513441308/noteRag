"""Reranker 单元测试：云端协议用 httpx MockTransport 离线模拟，本地用桩对象.

不加载 2.2G 的 CrossEncoder 权重；create_reranker 的缓存行为通过注入假的
sentence_transformers 模块验证。
"""

import json
import sys
import types

import httpx
import pytest
from langchain_core.documents import Document

from agent.retrieval import reranker
from agent.retrieval.reranker import (
    _MISSING_SCORE,
    CloudReranker,
    _stable_fingerprint,
    build_reranker,
    create_reranker,
    rerank_documents,
)

# ---------------------------------------------------------------------------
# 辅助对象
# ---------------------------------------------------------------------------


class FakeScorer:
    """记录调用参数、返回预设分数的 RerankerScorer 测试替身."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict] = []

    def predict(
        self,
        pairs,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[float]:
        self.calls.append(
            {
                "n_pairs": len(pairs),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "first_query": pairs[0][0],
            }
        )
        return self.scores


def make_mock_client(script: list):
    """按脚本依次应答的 httpx.Client（MockTransport）.

    script 元素：``(status_code, payload_dict)`` 或要抛出的网络异常实例；
    请求次数超过脚本长度时持续返回最后一个结果（用于模拟持续故障）。
    """
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        outcome = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        status_code, payload = outcome
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def make_cloud(script: list, **kwargs) -> tuple[CloudReranker, list[httpx.Request]]:
    client, calls = make_mock_client(script)
    cloud = CloudReranker(
        "BAAI/bge-reranker-v2-m3",
        api_key="sk-test",
        base_url="https://api.example.com/v1",
        provider="siliconflow",
        retry_backoff=0,
        client=client,
        **kwargs,
    )
    return cloud, calls


# ---------------------------------------------------------------------------
# rerank_documents：纯排序契约
# ---------------------------------------------------------------------------


def _docs() -> list[Document]:
    return [
        Document(page_content="天气不错", metadata={"source": "weather"}),
        Document(page_content="StateGraph 条件边文档", metadata={"source": "graph"}),
        Document(page_content="番茄炒蛋做法", metadata={"source": "cook"}),
    ]


def test_rerank_orders_by_score_and_writes_metadata() -> None:
    scorer = FakeScorer([0.1, 0.99, -0.5])
    ranked = rerank_documents(scorer, "StateGraph 怎么用", _docs(), top_k=3)

    assert [doc.metadata["source"] for doc in ranked] == ["graph", "weather", "cook"]
    assert [doc.metadata["rerank_rank"] for doc in ranked] == [1, 2, 3]
    assert ranked[0].metadata["rerank_score"] == 0.99
    # 调用参数：查询下传、关闭进度条、批次透传
    assert scorer.calls[0]["first_query"] == "StateGraph 怎么用"
    assert scorer.calls[0]["show_progress_bar"] is False
    assert scorer.calls[0]["n_pairs"] == 3


def test_rerank_top_k_truncation_and_edges() -> None:
    docs = _docs()
    assert len(rerank_documents(FakeScorer([1.0, 2.0, 3.0]), "q", docs, top_k=2)) == 2
    assert len(rerank_documents(FakeScorer([1.0, 2.0, 3.0]), "q", docs, top_k=10)) == 3
    assert rerank_documents(FakeScorer([1.0]), "q", docs, top_k=0) == []
    assert rerank_documents(FakeScorer([1.0]), "q", [], top_k=3) == []


def test_rerank_does_not_mutate_input_documents() -> None:
    docs = _docs()
    ranked = rerank_documents(FakeScorer([0.1, 0.9, 0.2]), "q", docs, top_k=3)

    assert all("rerank_score" not in doc.metadata for doc in docs)
    assert ranked is not docs and ranked[0] is not docs[0]
    # 原有 metadata 保留
    assert ranked[0].metadata["source"] == "graph"


# ---------------------------------------------------------------------------
# CloudReranker.predict：协议适配
# ---------------------------------------------------------------------------


def test_cloud_predict_request_shape_and_auth() -> None:
    payload = {"results": [{"index": 0, "relevance_score": 0.9}]}
    cloud, calls = make_cloud([(200, payload)])

    scores = cloud.predict([("查询", "文档甲"), ("查询", "文档乙")])

    assert len(calls) == 1
    request = calls[0]
    assert str(request.url) == "https://api.example.com/v1/rerank"
    assert request.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "查询",
        "documents": ["文档甲", "文档乙"],
        "top_n": 2,
        "return_documents": False,
    }
    assert scores[0] == 0.9 and scores[1] == _MISSING_SCORE
    cloud.close()


def test_cloud_reorders_shuffled_indexes() -> None:
    # 服务端按相关度乱序返回 index，必须还原为输入文档顺序
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.3},
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ]
    }
    cloud, _ = make_cloud([(200, payload)])
    assert cloud.predict([("q", "d0"), ("q", "d1"), ("q", "d2")]) == [0.9, 0.1, 0.3]
    cloud.close()


def test_cloud_empty_pairs_short_circuits_without_http() -> None:
    cloud, calls = make_cloud([(200, {"results": []})])
    assert cloud.predict([]) == []
    assert calls == []
    cloud.close()


def test_cloud_multiple_queries_raises_before_http() -> None:
    cloud, calls = make_cloud([(200, {"results": []})])
    with pytest.raises(ValueError, match="同一 query"):
        cloud.predict([("query-a", "doc"), ("query-b", "doc")])
    assert calls == []  # 请求发出前就拦截
    cloud.close()


def test_cloud_endpoint_per_provider() -> None:
    client, _ = make_mock_client([(200, {"results": []})])
    cloud = CloudReranker(
        "qwen3-rerank",
        api_key="sk",
        base_url="https://dashscope.aliyuncs.com/compatible-api/v1/",  # 尾部斜杠也要兼容
        provider="dashscope",
        client=client,
    )
    assert cloud.endpoint == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    cloud.close()


def test_cloud_init_validation() -> None:
    with pytest.raises(ValueError, match="api_key"):
        CloudReranker("m", api_key="", base_url="https://x")
    with pytest.raises(ValueError, match="provider"):
        CloudReranker("m", api_key="sk", base_url="https://x", provider="unknown")


# ---------------------------------------------------------------------------
# 重试 / 错误处理
# ---------------------------------------------------------------------------


def test_cloud_401_fails_fast_without_retry() -> None:
    cloud, calls = make_cloud([(401, {"detail": "unauthorized"})])
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        cloud.predict([("q", "d")])
    assert excinfo.value.response.status_code == 401
    assert len(calls) == 1
    cloud.close()


def test_cloud_503_retries_then_succeeds() -> None:
    cloud, calls = make_cloud(
        [
            (503, {"error": "busy"}),
            (503, {"error": "busy"}),
            (200, {"results": [{"index": 0, "relevance_score": 0.7}]}),
        ],
        max_retries=2,
    )
    assert cloud.predict([("q", "d")]) == [0.7]
    assert len(calls) == 3
    cloud.close()


def test_cloud_503_retries_exhausted_raises_last_error() -> None:
    cloud, calls = make_cloud([(503, {})], max_retries=1)
    with pytest.raises(httpx.HTTPStatusError):
        cloud.predict([("q", "d")])
    assert len(calls) == 2  # 首次 + 1 次重试
    cloud.close()


def test_cloud_network_error_retries_then_succeeds() -> None:
    cloud, calls = make_cloud(
        [
            httpx.ConnectError("connection refused"),
            (200, {"results": [{"index": 0, "relevance_score": 0.5}]}),
        ],
        max_retries=1,
    )
    assert cloud.predict([("q", "d")]) == [0.5]
    assert len(calls) == 2
    cloud.close()


def test_cloud_injected_client_not_closed_by_cloud() -> None:
    client, _ = make_mock_client([(200, {"results": []})])
    cloud = CloudReranker("m", api_key="sk", base_url="https://x", client=client)
    cloud.predict([("q", "d")])
    cloud.close()
    assert not client.is_closed  # 注入的 client 由调用方负责
    client.close()


# ---------------------------------------------------------------------------
# build_reranker 工厂
# ---------------------------------------------------------------------------


def test_build_reranker_unknown_provider() -> None:
    with pytest.raises(ValueError, match="未知"):
        build_reranker("mystery", "model")


def test_build_reranker_requires_model_and_key() -> None:
    with pytest.raises(ValueError, match="本地"):
        build_reranker("local", None)
    with pytest.raises(ValueError, match="需要指定 model"):
        build_reranker("siliconflow", None, api_key="sk")
    with pytest.raises(ValueError, match="api_key"):
        build_reranker("siliconflow", "m", api_key=None)


def test_build_reranker_cloud_returns_adapter() -> None:
    client, _ = make_mock_client([(200, {"results": []})])
    rerank_obj = build_reranker("siliconflow", "m", api_key="sk", client=client)
    assert isinstance(rerank_obj, CloudReranker)
    assert rerank_obj.endpoint.endswith("/v1/rerank")
    client.close()


def test_fingerprint_order_independent() -> None:
    assert _stable_fingerprint({"a": 1, "b": 2}) == _stable_fingerprint(
        {"b": 2, "a": 1}
    )
    assert _stable_fingerprint({"a": 1}) != _stable_fingerprint({"a": 2})


# ---------------------------------------------------------------------------
# create_reranker 缓存（注入假 sentence_transformers，不下载/不加载权重）
# ---------------------------------------------------------------------------


def test_create_reranker_cache_key_includes_all_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict] = []

    class FakeCrossEncoder:
        def __init__(self, model_name, *, device="cpu", model_kwargs=None, **kwargs):
            created.append(
                {
                    "model": model_name,
                    "device": device,
                    "model_kwargs": model_kwargs,
                    **kwargs,
                }
            )

        def predict(self, pairs, **kwargs):
            return [0.0 for _ in pairs]

    fake_module = types.ModuleType("sentence_transformers")
    setattr(fake_module, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(reranker, "_reranker_cache", {})

    first = create_reranker("fake/model", device="cpu")
    second = create_reranker("fake/model", device="cpu")
    assert first is second
    assert len(created) == 1  # 第二次命中缓存

    # kwargs 书写顺序不同但内容相同 -> 同一实例
    assert create_reranker(
        "fake/model", device="cpu", max_length=5, num_beams=2
    ) is create_reranker("fake/model", device="cpu", num_beams=2, max_length=5)
    # model_kwargs 值不同 -> 不同实例
    create_reranker("fake/model", device="cpu", model_kwargs={"dtype": "fp16"})
    create_reranker("fake/model", device="cpu", model_kwargs={"dtype": "fp32"})
    create_reranker("fake/model", device="cuda")
    # 1(首次) + 1(kwargs) + 2(两种 dtype) + 1(cuda) = 5
    assert len(created) == 5
