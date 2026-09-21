"""Rerank 重排器组件（本地 cross-encoder 与云端 rerank API）.

纯组件，不读取任何全局配置：模型名、设备、API Key 均由调用方显式传入，
由编排层（图构建时）用 ``Config`` 注入。提供两种后端，对上层透明：

- ``local``：本地 ``sentence_transformers.CrossEncoder``（默认 bge-reranker），
  实例按 ``(model_name, device)`` 缓存，避免重复加载权重；
- ``siliconflow`` / ``dashscope``：云端 rerank API（Jina 风格
  ``query + documents`` 协议），不占本地内存，按量计费（硅基流动 bge 免费档）。

两种后端都实现 :class:`RerankerScorer` 协议（``predict(pairs) -> 分数序列``），
因此 :func:`rerank_documents` 与图节点无需感知后端差异。

典型用法::

    # 本地
    reranker = build_reranker("local", "BAAI/bge-reranker-v2-m3", device="cpu")
    # 云端（硅基流动）
    reranker = build_reranker(
        "siliconflow", "BAAI/bge-reranker-v2-m3", api_key="sk-..."
    )
    top_docs = rerank_documents(reranker, query, retrieved_docs, top_k=3)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict, cast, final

import httpx
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 云端 provider -> 默认 base_url（base_url 不含 rerank 端点路径）
_RERANK_BASE_URLS: dict[str, str] = {
    "siliconflow": "https://api.siliconflow.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-api/v1",
}
# 云端 provider -> rerank 端点路径（阿里云 qwen3-rerank 用复数 reranks）
_RERANK_ENDPOINTS: dict[str, str] = {
    "siliconflow": "rerank",
    "dashscope": "reranks",
}
# 云端未返回某文档分数时的兜底极低分（保证其排在末尾）
_MISSING_SCORE = -1e9
# 值得退避重试的 HTTP 状态：429 限流 + 5xx 网关/服务端抖动；
# 4xx（400/401/403 等）是请求本身的问题，重试无意义，立即抛出
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _stable_fingerprint(value: Any) -> str:
    """把任意构造参数序列化为稳定字符串，用作缓存指纹.

    键排序保证不同传参顺序得到同一指纹；``default=str`` 兜底处理不可
    JSON 序列化的对象（如各类 ``model_kwargs`` 配置实例）。
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class RerankerScorer(Protocol):
    """统一的重排模型协议：吃 (query, passage) 对，吐相关度分数.

    本地 ``CrossEncoder`` 与 :class:`CloudReranker` 均满足该协议，
    :func:`rerank_documents` 只依赖此接口。
    """

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Sequence[float]:
        """对一批 (查询, 文档) 对打分，返回与输入等长、同序的分数."""
        ...


class _RerankItem(TypedDict):
    """云端 rerank 响应 results 数组中单个元素的结构."""

    index: int
    relevance_score: float


# (model_name, device, model_kwargs 指纹, 其他构造参数指纹) -> 本地重排模型实例
_reranker_cache: dict[tuple[str, str, str, str], RerankerScorer] = {}
_reranker_lock = threading.Lock()


def create_reranker(
    model_name: str,
    *,
    device: str = "cpu",
    model_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> RerankerScorer:
    """创建（或复用缓存的）CrossEncoder 重排模型.

    Args:
        model_name: HuggingFace 模型名或本地路径，如
            ``BAAI/bge-reranker-v2-m3``（多语言，推荐中文场景）。
        device: 推理设备，``cpu`` / ``cuda`` / ``cuda:0`` 等。
        model_kwargs: 透传给底层模型构造的参数（如 torch dtype/trust_remote_code）。
        **kwargs: 透传给 ``sentence_transformers.CrossEncoder`` 的其他参数。

    Returns:
        满足 :class:`RerankerScorer` 协议的 CrossEncoder 实例。
    """
    # model_kwargs / kwargs 不同（如 dtype 差异）必须视为不同实例，
    # 否则第二次调用会拿到按旧参数加载并缓存的模型；指纹与参数书写顺序无关
    cache_key = (
        model_name,
        device,
        _stable_fingerprint(model_kwargs or {}),
        _stable_fingerprint(kwargs),
    )
    with _reranker_lock:
        cached = _reranker_cache.get(cache_key)
        if cached is not None:
            return cached

    from sentence_transformers import CrossEncoder
    from transformers.utils.logging import disable_progress_bar

    # 关闭权重加载时的 tqdm 进度条，保持服务日志干净
    # transformers 未导出 py.typed，该调用无法静态检查
    disable_progress_bar()  # type: ignore[no-untyped-call]
    logger.info(f"加载 Rerank 模型：{model_name}（device={device}）")
    reranker = CrossEncoder(
        model_name,
        device=device,
        model_kwargs=model_kwargs or {},
        **kwargs,
    )

    # CrossEncoder.predict 是含多种输入形态的重载签名，无法静态匹配
    # Protocol 的单一签名；其 (pairs, batch_size, show_progress_bar) 调用
    # 形态运行时完全兼容，故在信任边界显式转换。
    scorer = cast(RerankerScorer, reranker)
    with _reranker_lock:
        _reranker_cache[cache_key] = scorer
    return scorer


def rerank_documents(
    reranker: RerankerScorer,
    query: str,
    documents: Sequence[Document],
    top_k: int,
    *,
    batch_size: int = 32,
) -> list[Document]:
    """用重排模型对召回文档按相关度精排，取前 top_k.

    bge-reranker 输出未归一化的相关度 logits，数值越大越相关；
    本函数只使用其相对序，不依赖分数绝对含义。

    Args:
        reranker: 任何满足 :class:`RerankerScorer` 协议的对象
            （本地 CrossEncoder 或云端 CloudReranker）。
        query: 用于精排的查询（建议传改写后的检索查询）。
        documents: 多路召回合并后的候选文档。
        top_k: 返回文档数量；大于候选数时返回全部，小于等于 0 时返回空列表。
        batch_size: 模型推理批大小。

    Returns:
        按相关度降序的前 top_k 个文档，返回的是新 ``Document`` 对象，
        metadata 在原文基础上追加 ``rerank_score``（保留 4 位小数）
        与 ``rerank_rank``（从 1 开始的名次），不修改传入的原文档。
    """
    if not documents or top_k <= 0:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    raw_scores: Sequence[float] = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    ranked_pairs = sorted(
        zip(documents, raw_scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    results: list[Document] = []
    for rank, (doc, score) in enumerate(ranked_pairs[:top_k], start=1):
        metadata = dict(doc.metadata) if doc.metadata else {}
        rerank_score = round(float(score), 4)
        metadata["rerank_score"] = rerank_score
        metadata["rerank_rank"] = rank
        results.append(
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    top_score = results[0].metadata["rerank_score"] if results else "N/A"
    logger.info(
        f"Rerank 完成：{len(documents)} 个候选 -> 保留 {len(results)} 个，最高分 {top_score}"
    )
    return results


@final
class CloudReranker:
    """云端 rerank API 适配器，接口对齐本地 CrossEncoder 的 ``predict``.

    适配 Jina 风格的 rerank 协议（硅基流动 ``/v1/rerank``、阿里云百炼
    qwen3-rerank ``/compatible-api/v1/reranks``）：一次请求传一个 query 与
    多个 documents，服务端返回每个文档的相关度分数。

    :meth:`predict` 接收 ``[(query, passage), ...]``（与
    ``CrossEncoder.predict`` 同形），内部要求这些 pair 共享同一个 query
    （:func:`rerank_documents` 正是这样构造的），再把服务端按 ``index``
    返回的分数还原成与输入文档**同序**的列表。
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        provider: str = "siliconflow",
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_backoff: float = 0.5,
        client: httpx.Client | None = None,
    ) -> None:
        """初始化云端重排器.

        Args:
            model: 云端模型名，如 ``BAAI/bge-reranker-v2-m3``、
                ``Qwen/Qwen3-Reranker-0.6B``、``qwen3-rerank``。
            api_key: 服务端 API Key（Bearer Token）。
            base_url: API 根地址（不含 rerank 路径），如
                ``https://api.siliconflow.cn/v1``。
            provider: 服务提供方，用于选择端点路径（rerank/reranks）。
            timeout: 单次请求超时秒数。
            max_retries: 失败后的最大重试次数（总请求数 = max_retries + 1）。
                仅对网络异常、429、5xx 重试；401/400 等 4xx 立即抛出。
                设为 0 可关闭重试。
            retry_backoff: 指数退避基数秒数，第 n 次重试前等待
                ``retry_backoff * 2**n``（默认两次重试前分别等 0.5s、1s）。
            client: 可选的注入 ``httpx.Client``（主要用于测试传入 MockTransport）；
                不传则内部新建，并在 :meth:`close` 时关闭。

        Raises:
            ValueError: provider 不支持或缺少 api_key。
        """
        if not api_key:
            raise ValueError(f"云端 rerank（{provider}）缺少 api_key。")
        endpoint = _RERANK_ENDPOINTS.get(provider)
        if endpoint is None:
            raise ValueError(
                f"不支持的云端 rerank provider：{provider}，可选 {sorted(_RERANK_ENDPOINTS)}。"
            )

        self.model: str = model
        self.provider: str = provider
        self.endpoint: str = f"{base_url.rstrip('/')}/{endpoint}"
        self.max_retries: int = max(0, max_retries)
        self.retry_backoff: float = retry_backoff
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client: httpx.Client = client or httpx.Client(timeout=timeout)
        self._owns_client: bool = client is None

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[float]:
        """对一批 ``(query, passage)`` 调用云端 API，返回与输入同序的分数.

        ``batch_size`` / ``show_progress_bar`` 仅为对齐
        ``CrossEncoder.predict`` 的签名而保留（云端单批返回、无本地进度条）。

        Args:
            pairs: 共享同一 query 的 (查询, 文档) 对。
            batch_size: 接口对齐保留参数，云端一次请求处理全部文档。
            show_progress_bar: 接口对齐保留参数，云端不展示进度条。

        Returns:
            与 ``pairs`` 等长、同序的相关度分数；服务端未返回的文档位置填兜底极低分。

        Raises:
            pairs 中出现多个不同 query 时抛出 ValueError（云端协议一次只接受一个
            query，静默取首个会产生错误打分，故直接报错）；云端返回非 2xx
            （4xx 立即抛出；429/5xx 重试耗尽后抛出）抛出 httpx.HTTPStatusError；
            网络异常重试耗尽后抛出 httpx.TransportError。
        """
        del batch_size, show_progress_bar  # 云端模式下不适用，仅为接口对齐保留

        if not pairs:
            return []

        query = pairs[0][0]
        distinct_queries = {q for q, _ in pairs}
        if len(distinct_queries) > 1:
            raise ValueError(
                f"CloudReranker.predict 要求所有 pair 共享同一 query，"
                f"却收到 {len(distinct_queries)} 个不同 query；"
                "云端 rerank 协议一次请求只接受一个 query"
                "（多 query 场景需分组分别调用）。"
            )
        documents = [passage for _, passage in pairs]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }
        response = self._post_with_retry(payload)
        response_data: dict[str, Any] = response.json()
        results: Sequence[_RerankItem] = response_data.get("results", [])

        scores = [_MISSING_SCORE] * len(documents)
        for item in results:
            index = item["index"]
            if 0 <= index < len(scores):
                scores[index] = float(item["relevance_score"])
        logger.info(
            f"云端 rerank（{self.provider}/{self.model}）完成："
            f"{len(documents)} 个候选，最高分 {max(scores):.4f}"
        )
        return scores

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        """发送 rerank 请求，对网络抖动 / 429 / 5xx 做指数退避重试.

        不可重试的响应（除 429 外的 4xx）在第一次收到时立即
        ``raise_for_status``；可重试失败在 ``max_retries`` 次内退避重试，
        耗尽后抛出最后一个异常（保留服务端状态码或网络错误类型）。
        """
        total_attempts = self.max_retries + 1
        last_response: httpx.Response | None = None
        last_exc: httpx.TransportError | None = None

        for attempt in range(total_attempts):
            will_retry = attempt < self.max_retries
            try:
                last_response = self._client.post(
                    self.endpoint, json=payload, headers=self._headers
                )
            except httpx.TransportError as exc:
                # 连接错误 / 超时等：可重试的网络层失败
                last_exc = exc
                logger.warning(
                    f"云端 rerank 网络异常（第 {attempt + 1}/{total_attempts} 次）：{exc}"
                    + ("，准备退避重试。" if will_retry else "，重试次数已用完。")
                )
            else:
                last_exc = None
                if last_response.status_code not in _RETRYABLE_STATUS:
                    # 含成功 2xx 与不可重试 4xx：后者在此直接抛出
                    last_response.raise_for_status()
                    return last_response
                logger.warning(
                    f"云端 rerank 返回 {last_response.status_code}"
                    f"（第 {attempt + 1}/{total_attempts} 次）"
                    + ("，准备退避重试。" if will_retry else "，重试次数已用完。")
                )

            if will_retry:
                time.sleep(self.retry_backoff * (2**attempt))

        # 重试耗尽：优先抛出最后一次 HTTP 响应的状态异常，其次网络异常
        if last_response is not None and last_exc is None:
            last_response.raise_for_status()
        assert last_exc is not None  # 走到这里必有未恢复的网络异常
        raise last_exc

    def close(self) -> None:
        """关闭内部创建的 HTTP 客户端（注入的 client 由调用方负责）."""
        if self._owns_client:
            self._client.close()


def build_reranker(
    provider: str,
    model: str | None,
    *,
    device: str = "cpu",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 2,
    client: httpx.Client | None = None,
    **kwargs: Any,
) -> RerankerScorer:
    """按 provider 创建本地或云端 reranker（统一编排入口）.

    纯工厂函数，不读取 ``Config``：所有参数由编排层从配置注入。
    返回对象都满足 :class:`RerankerScorer` 协议，可直接传给
    :func:`rerank_documents`。

    Args:
        provider: ``local`` / ``siliconflow`` / ``dashscope``。
        model: 模型名。local 为 HF 模型名/路径，云端为 API 模型名。
        device: 本地模型推理设备（仅 local）。
        api_key: 云端 API Key（云端必填）。
        base_url: 云端根地址，留空时按 provider 取内置默认值。
        timeout: 云端请求超时秒数。
        max_retries: 云端失败重试次数（仅网络异常/429/5xx，0 关闭）。
        client: 可选注入的 ``httpx.Client``（测试用）。
        **kwargs: local 时透传给 :func:`create_reranker`（如 model_kwargs）。

    Returns:
        满足 :class:`RerankerScorer` 协议的本地或云端重排器。

    Raises:
        ValueError: provider 未知，或缺少 model / api_key。
    """
    provider = (provider or "local").strip().lower()

    if provider == "local":
        if not model:
            raise ValueError("本地 rerank 需要指定 model（HF 模型名或本地路径）。")
        return create_reranker(model, device=device, **kwargs)

    if provider in _RERANK_BASE_URLS:
        if not model:
            raise ValueError(f"云端 rerank（{provider}）需要指定 model。")
        return CloudReranker(
            model,
            api_key=api_key or "",
            base_url=base_url or _RERANK_BASE_URLS[provider],
            provider=provider,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    raise ValueError(
        f"未知 RERANK_PROVIDER：{provider}，可选 local / {sorted(_RERANK_BASE_URLS)}。"
    )
