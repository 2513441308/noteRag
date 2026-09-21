"""检索层离线评估的集成测试：真实本地嵌入 + 本地 CrossEncoder 跑黄金集.

与 ``test_graph.py`` 一样默认跳过（``-m 'not integration'``），需要本地
chroma 库与已缓存的 BAAI/bge-reranker-v2-m3 权重，不调用聊天 LLM、
不调云端 rerank。指标线按 20 题黄金集实跑基线（召回命中 1.0、
hit@3 0.95、hit@1 0.65、mrr@3 0.783）下调一档设置，作为防退化护栏，
刻意不卡满分以免脆弱。跑法：
``uv run pytest -m integration tests/integration_tests/test_retrieval_eval.py``
"""

from pathlib import Path

import pytest

from agent.retrieval.evaluation import load_golden_cases, run_evaluation

pytestmark = [pytest.mark.integration]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN = _PROJECT_ROOT / "eval" / "golden_retrieval.jsonl"


def test_golden_retrieval_metrics_above_baseline() -> None:
    """黄金集端到端检索指标不低于校准基线（本地模型，零 API）。"""
    summary = run_evaluation(_GOLDEN, ks=(1, 3))
    golden_count = len(load_golden_cases(_GOLDEN))

    assert summary["total"] == golden_count
    # 护栏：评估必须零 API 成本，若误接云端 provider 应立即失败
    assert summary["config"]["rerank_provider"].startswith("local")

    metrics = summary["metrics"]
    assert metrics["hit@3"] >= 0.85
    assert metrics["hit@1"] >= 0.50
    assert metrics["mrr@3"] >= 0.70
    # 双路合路的召回覆盖必须足够高，问题通常出在精排而非召回
    assert summary["retrieval_hit"] >= 0.95
    # 召回命中却被精排挤出 top3 的样例不应超过 3 条（基线为 1）
    assert summary["rerank_demoted"] <= 3
