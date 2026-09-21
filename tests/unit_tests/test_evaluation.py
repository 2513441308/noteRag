"""检索评估组件的离线单元测试.

用假检索器 / 按正文返回固定分数的假重排器，不加载真实模型、不碰 chroma，
覆盖：黄金集加载校验、source 匹配两条规则、Hit/Recall/MRR、双路合并降级、
精排产物、阈值四象限扫描、报告渲染/JSON 以及 CLI 参数解析。
"""

from __future__ import annotations

import importlib.util
import io
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from agent.retrieval import evaluation as eval_mod
from agent.retrieval.evaluation import (
    CaseEval,
    GoldenCase,
    RankedRow,
    build_summary,
    evaluate_case,
    format_report,
    hit_at_k,
    load_golden_cases,
    recall_at_k,
    reciprocal_rank_at_k,
    retrieve_documents,
    source_matches,
    summarize_scores,
    threshold_sweep,
    to_relative_source,
    write_json,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeRetriever:
    """返回固定文档的假检索器，可配置 invoke 时抛错。"""

    def __init__(self, docs: list[Document], error: Exception | None = None) -> None:
        self.docs = docs
        self.error = error
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return list(self.docs)


class _ScriptedReranker:
    """按正文查表返回固定 logits 的假重排器，签名对齐 RerankerScorer 协议。"""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.pairs: Sequence[tuple[str, str]] = []

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> Sequence[float]:
        del batch_size, show_progress_bar
        self.pairs = pairs
        return [self.scores.get(passage, 0.0) for _, passage in pairs]


def _doc(
    doc_id: str, content: str, source: str | None = None, **metadata: Any
) -> Document:
    """构造带 id / source 的测试文档。"""
    meta = dict(metadata)
    if source is not None:
        meta["source"] = source
    return Document(id=doc_id, page_content=content, metadata=meta)


# ---------------- 黄金集加载 ----------------


def test_load_golden_cases_parses_and_fills_ids(tmp_path: Path) -> None:
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        "# 头部注释\n"
        "\n"
        '{"question": "  问题一  ", "expected_sources": ["a/x.md"]}\n'
        '{"id": "custom", "question": "问题二", "expected_sources": ["b/y.md", "b/z.md"],'
        ' "note": " 备注 "}\n',
        encoding="utf-8",
    )

    cases = load_golden_cases(golden)

    assert [c.case_id for c in cases] == ["g01", "custom"]
    assert cases[0].question == "问题一"
    assert cases[0].expected_sources == ("a/x.md",)
    assert cases[0].note is None
    assert cases[1].expected_sources == ("b/y.md", "b/z.md")
    assert cases[1].note == "备注"


@pytest.mark.parametrize(
    "line",
    [
        '{"question": "x"}',  # 缺 expected_sources
        '{"question": "x", "expected_sources": []}',  # 空期望
        '{"question": "x", "expected_sources": [""]}',  # 空串来源
        '{"question": "", "expected_sources": ["a.md"]}',  # 空问题
        '{"question": 1, "expected_sources": ["a.md"]}',  # 类型错误
        "not-json",  # 非法 JSON
        "[1, 2]",  # 不是对象
    ],
)
def test_load_golden_cases_rejects_bad_rows(tmp_path: Path, line: str) -> None:
    golden = tmp_path / "bad.jsonl"
    golden.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_golden_cases(golden)


def test_load_golden_cases_empty_file_raises(tmp_path: Path) -> None:
    golden = tmp_path / "empty.jsonl"
    golden.write_text("# 只有注释\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="没有任何有效样例"):
        load_golden_cases(golden)


# ---------------- 路径匹配 ----------------


def test_source_matches_inside_and_outside_kb(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    inside = kb / "langgraph-docs" / "persistence.md"
    foreign = tmp_path / "other_kb" / "langgraph-docs" / "persistence.md"

    assert source_matches(str(inside), "langgraph-docs/persistence.md", kb)
    assert not source_matches(str(inside), "langgraph-docs/other.md", kb)
    # 越界绝对路径退回后缀兜底
    assert source_matches(str(foreign), "langgraph-docs/persistence.md", kb)
    assert not source_matches(str(foreign), "langgraph-docs/other.md", kb)

    # 已相对化的展示路径（评估明细形态）直接逐段比较
    assert source_matches(
        "langgraph-docs/persistence.md", "langgraph-docs/persistence.md", kb
    )
    assert not source_matches(
        "langgraph-docs/other.md", "langgraph-docs/persistence.md", kb
    )

    assert to_relative_source(str(inside), kb) == "langgraph-docs/persistence.md"
    assert to_relative_source(str(foreign), kb) is None


# ---------------- 纯指标 ----------------


def test_hit_recall_mrr_metrics(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    ranked = [
        str(kb / "docs" / "a.md"),
        str(kb / "docs" / "b.md"),
        str(kb / "docs" / "c.md"),
    ]
    expected = ("docs/b.md",)

    assert not hit_at_k(ranked, expected, kb, 1)
    assert hit_at_k(ranked, expected, kb, 3)
    assert reciprocal_rank_at_k(ranked, expected, kb, 3) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(ranked, expected, kb, 1) == 0.0
    assert recall_at_k(ranked, expected, kb, 3) == pytest.approx(1.0)

    multi_expected = ("docs/b.md", "docs/z.md")
    assert recall_at_k(ranked, multi_expected, kb, 3) == pytest.approx(0.5)
    assert recall_at_k(ranked, multi_expected, kb, 1) == pytest.approx(0.0)


# ---------------- 双路召回 ----------------


def test_retrieve_documents_merges_and_dedups() -> None:
    vector = _FakeRetriever([_doc("v1", "向量独有"), _doc("shared", "重复文档")])
    bm25 = _FakeRetriever([_doc("shared", "重复文档"), _doc("b1", "BM25 独有")])

    docs = retrieve_documents(
        "问题",
        vector_retriever=vector,
        bm25_retriever=bm25,
    )

    assert [doc.id for doc in docs] == ["v1", "shared", "b1"]
    assert vector.queries == ["问题"]
    assert bm25.queries == ["问题"]


def test_retrieve_documents_bm25_valueerror_falls_back_to_vector() -> None:
    vector = _FakeRetriever([_doc("v1", "向量文档")])
    bm25 = _FakeRetriever([], error=ValueError("空集合无法建 BM25 索引"))

    docs = retrieve_documents(
        "问题",
        vector_retriever=vector,
        bm25_retriever=bm25,
    )

    assert [doc.id for doc in docs] == ["v1"]


def test_retrieve_documents_without_bm25() -> None:
    vector = _FakeRetriever([_doc("v1", "向量文档")])
    docs = retrieve_documents(
        "问题",
        vector_retriever=vector,
        bm25_retriever=None,
    )
    assert [doc.id for doc in docs] == ["v1"]


# ---------------- 单条精排链路 ----------------


def test_evaluate_case_collects_ranked_rows(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    docs_dir = kb / "docs"
    vector = _FakeRetriever(
        [
            _doc("d1", "正文A", source=str(docs_dir / "a.md")),
            _doc("d2", "正文B", source=str(docs_dir / "b.md")),
        ]
    )
    bm25 = _FakeRetriever(
        [
            _doc("d2", "正文B", source=str(docs_dir / "b.md")),
            _doc("d3", "正文C", source=str(docs_dir / "c.md")),
        ]
    )
    reranker = _ScriptedReranker({"正文A": 9.0, "正文B": 1.0, "正文C": 5.0})
    case = GoldenCase(question="问题", expected_sources=("docs/a.md",), case_id="g01")

    result = evaluate_case(
        case,
        kb_dir=kb,
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=reranker,
        top_k=2,
    )

    assert result.retrieved_sources == ["docs/a.md", "docs/b.md", "docs/c.md"]
    assert [(row.source, row.score) for row in result.ranked] == [
        ("docs/a.md", 9.0),
        ("docs/c.md", 5.0),
    ]
    assert result.top_score == 9.0
    assert result.ranked_source_list() == ["docs/a.md", "docs/c.md"]


# ---------------- 阈值扫描 / 汇总 / 报告 ----------------


def _synthetic_results(kb: Path) -> list[CaseEval]:
    """构造三条样例：高分命中、低分未命中、无候选。"""
    hit_source = str(kb / "docs" / "hit.md")
    other_source = str(kb / "docs" / "other.md")
    case_hit = GoldenCase(
        question="命中题", expected_sources=("docs/hit.md",), case_id="g01"
    )
    case_miss = GoldenCase(
        question="未中题", expected_sources=("docs/miss.md",), case_id="g02"
    )
    case_empty = GoldenCase(
        question="无题", expected_sources=("docs/empty.md",), case_id="g03"
    )
    return [
        CaseEval(
            case=case_hit,
            retrieved_sources=["docs/hit.md", "docs/other.md"],
            ranked=[RankedRow(1, hit_source, 9.0), RankedRow(2, other_source, 3.0)],
        ),
        CaseEval(
            case=case_miss,
            retrieved_sources=["docs/other.md"],
            ranked=[RankedRow(1, other_source, 1.0)],
        ),
        CaseEval(case=case_empty, retrieved_sources=[], ranked=[]),
    ]


def test_threshold_sweep_four_quadrants(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    results = _synthetic_results(kb)

    rows = {
        row.threshold: row
        for row in threshold_sweep(results, kb_dir=kb, k=3, thresholds=(0.0, 4.0, 10.0))
    }

    assert (rows[0.0].kept_hits, rows[0.0].false_keeps, rows[0.0].true_rejects) == (
        1,
        1,
        1,
    )
    assert (rows[4.0].kept_hits, rows[4.0].false_rejects, rows[4.0].true_rejects) == (
        1,
        0,
        2,
    )
    assert (rows[10.0].false_rejects, rows[10.0].true_rejects) == (1, 2)


def test_build_summary_metrics_and_stats(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    summary = build_summary(_synthetic_results(kb), kb_dir=kb, ks=(1, 3))

    assert summary["total"] == 3
    assert summary["metrics"]["hit@1"] == pytest.approx(1 / 3)
    assert summary["metrics"]["mrr@1"] == pytest.approx(1 / 3)
    assert summary["retrieval_hit"] == pytest.approx(1 / 3)
    assert summary["rerank_demoted"] == 0
    stats = summary["score_stats"]
    assert stats is not None
    assert (stats["min"], stats["median"], stats["max"]) == (1.0, 5.0, 9.0)
    assert len(summary["threshold_sweep"]) == len(eval_mod.DEFAULT_THRESHOLD_GRID)
    assert [case["id"] for case in summary["cases"]] == ["g01", "g02", "g03"]


def test_build_summary_empty_results(tmp_path: Path) -> None:
    summary = build_summary([], kb_dir=tmp_path / "kb")
    assert summary["total"] == 0
    assert summary["metrics"]["hit@1"] == 0.0
    assert summary["retrieval_hit"] == 0.0
    assert summary["score_stats"] is None


def test_summarize_scores_none_without_candidates() -> None:
    case = CaseEval(
        case=GoldenCase(question="q", expected_sources=("a.md",)),
        retrieved_sources=[],
        ranked=[],
    )
    assert summarize_scores([case]) is None


def test_format_report_and_json_roundtrip(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    summary = build_summary(
        _synthetic_results(kb),
        kb_dir=kb,
        ks=(1, 3),
        config={"rerank_model": "fake-model"},
    )

    report = format_report(summary)
    assert "样例数：3" in report
    assert "fake-model" in report
    assert "熔断扫描" in report
    assert "未中题" in report  # 未达标样例列出问题
    assert "无题" in report

    buffer = io.StringIO()
    write_json(summary, buffer)
    payload = json.loads(buffer.getvalue())
    assert payload["total"] == 3
    assert payload["cases"][0]["ranked"][0]["score"] == 9.0


# ---------------- CLI ----------------


def _load_cli_module() -> Any:
    """按文件路径加载 scripts/eval_retrieval.py（scripts 不是包）。"""
    script_path = _PROJECT_ROOT / "scripts" / "eval_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_missing_golden_returns_2(tmp_path: Path) -> None:
    cli = _load_cli_module()
    assert cli.main(["--golden", str(tmp_path / "nope.jsonl")]) == 2


def test_cli_happy_path_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli_module()
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        '{"question": "q", "expected_sources": ["a.md"]}\n', encoding="utf-8"
    )
    out_path = tmp_path / "out" / "report.json"

    # 装配真实组件很重，单测里替换成空报告
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda path, **kwargs: build_summary([], kb_dir=tmp_path / "kb"),
    )

    code = cli.main(["--golden", str(golden), "--json", str(out_path)])

    assert code == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["total"] == 0


def test_cli_parse_ks_and_thresholds() -> None:
    cli = _load_cli_module()
    parser = cli.build_parser()
    args = parser.parse_args(["--ks", "5,1,1,3", "--thresholds", "2,0"])
    assert args.ks == (1, 3, 5)
    assert args.thresholds == (0.0, 2.0)

    # 缺省阈值为 None，交给 run_evaluation 按 provider 自适应
    default_args = parser.parse_args([])
    assert default_args.thresholds is None

    with pytest.raises(SystemExit):
        parser.parse_args(["--ks", "1,x"])
