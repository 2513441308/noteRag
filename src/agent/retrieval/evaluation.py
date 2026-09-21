"""检索层离线评估组件：黄金集 + Hit/Recall/MRR + rerank 分数分布与熔断扫描.

为「检索质量是否退化」「RERANK_SCORE_THRESHOLD 应标到多少」提供数据支撑，
跑的是**线上同构的检索链路**（向量 + BM25 双路召回 → CrossEncoder 精排），
但不调用聊天 LLM、不跑查询改写——直接用黄金集中的原始问题，既零 API 成本、
又隔离了改写器波动，量到的是检索/精排本身的质量。

与仓库其他组件一致：指标计算与路径匹配是不读配置的纯函数；组件装配
（向量库、BM25、重排模型）集中在 :func:`build_components_from_config`，
编排层（脚本 / 集成测试）显式传入。

黄金集 JSONL 每行一个样例::

    {"id": "g01", "question": "...", "expected_sources": ["langgraph-docs/x.md"]}

``expected_sources`` 写**知识库相对的 posix 路径**（与入库 metadata 的
source 绝对路径对应），一条样例可写多个等价来源（命中任一即算 Hit）。
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agent.retrieval.reranker import RerankerScorer, rerank_documents

logger = logging.getLogger(__name__)

# 报告默认统计的截断档位
DEFAULT_KS: tuple[int, ...] = (1, 3)
# rerank 绝对分熔断扫描网格（bge-reranker-v2-m3 在本栈输出 sigmoid 归一化分 0~1）
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


# ---------------- 黄金集 ----------------


@dataclass(frozen=True)
class GoldenCase:
    """一条检索评估样例.

    Attributes:
        question: 用户原始问题（不经改写器）。
        expected_sources: 可接受的知识库相对来源路径（posix），命中任一即算命中。
        case_id: 样例编号，JSONL 缺省时按序生成 ``g01``。
        note: 备注（标注意图等），纯人工维护用，不参与计算。
    """

    question: str
    expected_sources: tuple[str, ...]
    case_id: str = ""
    note: str | None = None


def load_golden_cases(path: Path) -> list[GoldenCase]:
    """从 JSONL 文件加载黄金集.

    Args:
        path: JSONL 文件路径，每行 ``{"question": ..., "expected_sources": [...]}``；
            空行与 ``#`` 开头的注释行允许出现，便于分段维护。

    Returns:
        黄金样例列表，保持文件中的顺序；缺省 id 按序号补 ``g01`` 起。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 某行不是合法 JSON、缺问题/期望来源或字段类型不对。
    """
    cases: list[GoldenCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"黄金集第 {line_no} 行不是合法 JSON：{exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"黄金集第 {line_no} 行必须是 JSON 对象。")
            question = row.get("question")
            expected = row.get("expected_sources")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"黄金集第 {line_no} 行缺少非空 question。")
            if (
                not isinstance(expected, list)
                or not expected
                or not all(isinstance(item, str) and item.strip() for item in expected)
            ):
                raise ValueError(
                    f"黄金集第 {line_no} 行的 expected_sources 必须是非空字符串数组。"
                )
            case_id = row.get("id")
            note = row.get("note")
            cases.append(
                GoldenCase(
                    question=question.strip(),
                    expected_sources=tuple(item.strip() for item in expected),
                    case_id=str(case_id).strip()
                    if isinstance(case_id, str) and case_id.strip()
                    else f"g{len(cases) + 1:02d}",
                    note=note.strip()
                    if isinstance(note, str) and note.strip()
                    else None,
                )
            )
    if not cases:
        raise ValueError(f"黄金集 {path} 中没有任何有效样例。")
    return cases


# ---------------- 路径匹配 ----------------


def to_relative_source(source: str, kb_dir: Path) -> str | None:
    """把入库的绝对 source 转成知识库相对 posix 路径，越界返回 None."""
    try:
        return Path(source).resolve().relative_to(kb_dir.resolve()).as_posix()
    except ValueError:
        return None


def source_matches(source: str, expected_rel: str, kb_dir: Path) -> bool:
    """判断一个入库 source 是否命中期望的知识库相对路径.

    匹配规则（任一满足即命中）：

    1. source 本身已是相对路径（评估展示形态）：与期望**逐段 posix 相等**；
    2. source 是绝对路径且解析后位于 ``kb_dir`` 内：相对路径与期望逐段相等
       （与知识库管理端 ``_rel`` 同规则，仓库换机器/换绝对路径不影响）；
    3. 绝对路径越界（如集合由其他机器/目录的同名知识库写入）：以
       ``/<expected_rel>`` 后缀兜底匹配。

    Args:
        source: 入库 metadata 中的来源（绝对路径）或已相对化的展示路径。
        expected_rel: 黄金集标注的知识库相对 posix 路径。
        kb_dir: 知识库根目录。

    Returns:
        是否命中该期望来源。
    """
    expected = PurePosixPath(expected_rel.strip().lstrip("/"))
    if not Path(source).is_absolute():
        # 评估明细里的来源已被 _display_source 相对化，直接逐段比较
        return PurePosixPath(str(source).replace("\\", "/")) == expected
    rel = to_relative_source(source, kb_dir)
    if rel is not None:
        return PurePosixPath(rel) == expected
    # source 不在当前知识库目录下（如集合来自其他机器的绝对路径）：退回后缀匹配
    normalized = str(source).replace("\\", "/").rstrip("/")
    return normalized.endswith("/" + expected.as_posix())


# ---------------- 指标（纯函数） ----------------


def _first_hit_rank(
    ranked_sources: list[str], expected_sources: tuple[str, ...], kb_dir: Path, k: int
) -> int | None:
    """前 k 个结果中第一个命中任一期望来源的 1-based 名次，未命中返回 None."""
    for rank, source in enumerate(ranked_sources[:k], start=1):
        if any(
            source_matches(source, expected, kb_dir) for expected in expected_sources
        ):
            return rank
    return None


def hit_at_k(
    ranked_sources: list[str], expected_sources: tuple[str, ...], kb_dir: Path, k: int
) -> bool:
    """Hit@k：前 k 个结果是否命中任一期望来源."""
    return _first_hit_rank(ranked_sources, expected_sources, kb_dir, k) is not None


def recall_at_k(
    ranked_sources: list[str], expected_sources: tuple[str, ...], kb_dir: Path, k: int
) -> float:
    """Recall@k：前 k 个结果覆盖了多少比例的期望来源（0~1）."""
    if not expected_sources:
        return 0.0
    top = ranked_sources[:k]
    found = sum(
        1
        for expected in expected_sources
        if any(source_matches(source, expected, kb_dir) for source in top)
    )
    return found / len(expected_sources)


def reciprocal_rank_at_k(
    ranked_sources: list[str], expected_sources: tuple[str, ...], kb_dir: Path, k: int
) -> float:
    """MRR 的单条贡献：首个命中名次的倒数，未命中为 0."""
    rank = _first_hit_rank(ranked_sources, expected_sources, kb_dir, k)
    return 1.0 / rank if rank is not None else 0.0


# ---------------- 单条样例执行 ----------------


@dataclass(frozen=True)
class RankedRow:
    """精排后的一行结果（只留评估需要的字段，便于序列化）."""

    rank: int
    source: str
    score: float


@dataclass(frozen=True)
class CaseEval:
    """一条黄金样例的评估产物."""

    case: GoldenCase
    retrieved_sources: list[str]
    ranked: list[RankedRow]

    @property
    def top_score(self) -> float | None:
        """精排第一名的绝对分；无候选时为 None."""
        return self.ranked[0].score if self.ranked else None

    def ranked_source_list(self) -> list[str]:
        """精排结果的来源序列（按名次），供指标函数消费."""
        return [row.source for row in self.ranked]


def _dedup_documents(docs: list[Document]) -> list[Document]:
    """按文档 id 去重（无 id 退回正文），保持首次出现顺序（与 rag 节点同规则）."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.id if doc.id is not None else f"content:{doc.page_content}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _display_source(source: Any, kb_dir: Path) -> str:
    """结果来源的展示形态：优先知识库相对路径，拿不到就保留原值."""
    if not isinstance(source, str):
        return str(source)
    return to_relative_source(source, kb_dir) or source


def retrieve_documents(
    question: str,
    *,
    vector_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever | None = None,
) -> list[Document]:
    """向量 + BM25 双路召回并按 id 合并去重（与 node_rag 单查询行为一致）.

    Args:
        question: 原始问题（评估不做 LLM 改写）。
        vector_retriever: 向量检索器（必需）。
        bm25_retriever: BM25 检索器；为 None（如 postgres 后端）时只走向量路；
            调用抛 ValueError（空集合无法建索引）时同样降级为仅向量召回。

    Returns:
        合并去重后的候选 Document 列表。
    """
    candidates: list[Document] = list(vector_retriever.invoke(question))
    if bm25_retriever is not None:
        try:
            candidates.extend(bm25_retriever.invoke(question))
        except ValueError:
            logger.warning("BM25 检索不可用（知识库可能为空），本次仅走向量召回。")
    return _dedup_documents(candidates)


def evaluate_case(
    case: GoldenCase,
    *,
    kb_dir: Path,
    vector_retriever: BaseRetriever,
    reranker: RerankerScorer,
    bm25_retriever: BaseRetriever | None = None,
    top_k: int,
) -> CaseEval:
    """跑一条样例的完整检索→精排链路，收集来源序列与绝对分.

    Args:
        case: 黄金样例。
        kb_dir: 知识库根目录，用于 source 相对化展示与匹配。
        vector_retriever: 向量检索器。
        reranker: 重排模型（本地 CrossEncoder 或云端适配器均可）。
        bm25_retriever: 可选 BM25 检索器。
        top_k: 精排保留条数（通常取 ``Config.RERANK_TOP_K``）。

    Returns:
        该样例的评估产物；精排行保留 ``rerank_score`` 绝对分，供阈值标定。
    """
    candidates = retrieve_documents(
        case.question,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )
    ranked_docs = rerank_documents(reranker, case.question, candidates, top_k=top_k)

    seen: set[str] = set()
    retrieved_sources: list[str] = []
    for doc in candidates:
        source = _display_source(doc.metadata.get("source"), kb_dir)
        if source not in seen:
            seen.add(source)
            retrieved_sources.append(source)

    ranked_rows = [
        RankedRow(
            rank=rank,
            source=_display_source(doc.metadata.get("source"), kb_dir),
            score=float(doc.metadata.get("rerank_score", 0.0)),
        )
        for rank, doc in enumerate(ranked_docs, start=1)
    ]
    return CaseEval(case=case, retrieved_sources=retrieved_sources, ranked=ranked_rows)


# ---------------- 汇总与熔断扫描 ----------------


@dataclass(frozen=True)
class ScoreStats:
    """全部样例 top1 重排分的分布统计."""

    count: int
    minimum: float
    median: float
    mean: float
    maximum: float


def summarize_scores(results: list[CaseEval]) -> ScoreStats | None:
    """统计每条样例精排第一名的绝对分；没有任何有效候选时返回 None."""
    scores = [result.top_score for result in results if result.top_score is not None]
    if not scores:
        return None
    return ScoreStats(
        count=len(scores),
        minimum=min(scores),
        median=statistics.median(scores),
        mean=statistics.fmean(scores),
        maximum=max(scores),
    )


@dataclass(frozen=True)
class ThresholdRow:
    """某熔断阈值下全部样例的四象限分布."""

    threshold: float
    total: int
    kept_hits: int
    """命中且分数达标被保留（正确保留）。"""
    false_rejects: int
    """期望来源命中却被熔断误杀（最需要避免的一类）。"""
    true_rejects: int
    """本就未命中、分数不达标被拒绝（正确熔断）。"""
    false_keeps: int
    """未命中但分数达标被保留（熔断漏网）。"""


def threshold_sweep(
    results: list[CaseEval],
    *,
    kb_dir: Path,
    k: int,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> list[ThresholdRow]:
    """对一组候选熔断阈值做四象限扫描，为阈值标定提供依据.

    判定基于真实 Hit@k 与 top1 绝对分：没有任何候选的样例在任意阈值下
    都视为被拒绝（与线上「无召回直接空」一致）。

    Args:
        results: 全部样例的评估产物。
        kb_dir: 知识库根目录。
        k: 判定命中的精排截断档位。
        thresholds: 待扫描的候选阈值（升序）。

    Returns:
        与 thresholds 同序的四象限统计行。
    """
    rows: list[ThresholdRow] = []
    for threshold in thresholds:
        kept_hits = false_rejects = true_rejects = false_keeps = 0
        for result in results:
            hit = hit_at_k(
                result.ranked_source_list(), result.case.expected_sources, kb_dir, k
            )
            top = result.top_score
            kept = top is not None and top >= threshold
            if kept and hit:
                kept_hits += 1
            elif kept:
                false_keeps += 1
            elif hit:
                false_rejects += 1
            else:
                true_rejects += 1
        rows.append(
            ThresholdRow(
                threshold=threshold,
                total=len(results),
                kept_hits=kept_hits,
                false_rejects=false_rejects,
                true_rejects=true_rejects,
                false_keeps=false_keeps,
            )
        )
    return rows


# ---------------- 汇总 dict / 文本报告 ----------------


def _retrieval_hit(result: CaseEval, kb_dir: Path) -> bool:
    """召回阶段（精排前的全部候选）是否覆盖任一期望来源."""
    return hit_at_k(
        result.retrieved_sources,
        result.case.expected_sources,
        kb_dir,
        len(result.retrieved_sources),
    )


def build_summary(
    results: list[CaseEval],
    *,
    kb_dir: Path,
    ks: tuple[int, ...] = DEFAULT_KS,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把逐样例结果聚合成可 JSON 序列化的评估报告 dict.

    Args:
        results: 全部样例的评估产物。
        kb_dir: 知识库根目录（匹配期望路径用）。
        ks: 统计的截断档位（如 ``(1, 3)``）。
        thresholds: 熔断扫描网格。
        config: 额外写入报告的配置快照（检索条数/模型等），可为空。

    Returns:
        报告 dict：含样例数、各档 Hit/Recall/MRR、召回阶段命中率、
        被精排挤出 top-k 的样例数、top1 分数统计、熔断扫描表与逐样例明细。
    """
    total = len(results)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = (
            statistics.fmean(
                float(
                    hit_at_k(r.ranked_source_list(), r.case.expected_sources, kb_dir, k)
                )
                for r in results
            )
            if total
            else 0.0
        )
        metrics[f"recall@{k}"] = (
            statistics.fmean(
                recall_at_k(r.ranked_source_list(), r.case.expected_sources, kb_dir, k)
                for r in results
            )
            if total
            else 0.0
        )
        metrics[f"mrr@{k}"] = (
            statistics.fmean(
                reciprocal_rank_at_k(
                    r.ranked_source_list(), r.case.expected_sources, kb_dir, k
                )
                for r in results
            )
            if total
            else 0.0
        )
    sweep_k = max(ks)
    retrieval_hits = sum(1 for r in results if _retrieval_hit(r, kb_dir))
    summary: dict[str, Any] = {
        "total": total,
        "config": config or {},
        "metrics": metrics,
        "retrieval_hit": retrieval_hits / total if total else 0.0,
        "rerank_demoted": sum(
            1
            for r in results
            if _retrieval_hit(r, kb_dir)
            and not hit_at_k(
                r.ranked_source_list(), r.case.expected_sources, kb_dir, sweep_k
            )
        ),
        "score_stats": None,
        "threshold_sweep": [
            {
                "threshold": row.threshold,
                "kept_hits": row.kept_hits,
                "false_rejects": row.false_rejects,
                "true_rejects": row.true_rejects,
                "false_keeps": row.false_keeps,
            }
            for row in threshold_sweep(
                results, kb_dir=kb_dir, k=sweep_k, thresholds=thresholds
            )
        ],
        "cases": [
            {
                "id": r.case.case_id,
                "question": r.case.question,
                "expected_sources": list(r.case.expected_sources),
                "retrieval_hit": _retrieval_hit(r, kb_dir),
                "top_score": r.top_score,
                "ranked": [
                    {"rank": row.rank, "source": row.source, "score": row.score}
                    for row in r.ranked
                ],
            }
            for r in results
        ],
    }
    stats = summarize_scores(results)
    if stats is not None:
        summary["score_stats"] = {
            "count": stats.count,
            "min": stats.minimum,
            "median": stats.median,
            "mean": stats.mean,
            "max": stats.maximum,
        }
    return summary


def _relative_expected_hit(
    ranked_sources: list[str], expected_sources: list[str]
) -> bool:
    """明细行已是相对路径，直接按 posix 逐段判等（报告渲染用）."""
    return any(
        PurePosixPath(source) == PurePosixPath(expected)
        for source in ranked_sources
        for expected in expected_sources
    )


def format_report(summary: dict[str, Any]) -> str:
    """把 :func:`build_summary` 的 dict 渲染成适合终端阅读的文本报告."""
    lines: list[str] = []
    cfg = summary.get("config") or {}
    if cfg:
        lines.append(
            "配置快照：" + "  ".join(f"{key}={value}" for key, value in cfg.items())
        )
    lines.append(f"样例数：{summary['total']}")
    lines.append("指标：")
    for key, value in summary["metrics"].items():
        lines.append(f"  {key:<10}= {value:.3f}")
    lines.append(f"  {'recall(召回阶段)':<18}= {summary['retrieval_hit']:.3f}")
    sweep_k = max(int(key.rsplit("@", 1)[1]) for key in summary["metrics"])
    lines.append(
        f"召回命中但被精排挤出 top{sweep_k} 的样例数：{summary['rerank_demoted']}"
    )
    stats = summary.get("score_stats")
    if stats is not None:
        lines.append(
            "top1 重排分："
            f"min={stats['min']:.3f}  median={stats['median']:.3f}  "
            f"mean={stats['mean']:.3f}  max={stats['max']:.3f}  (n={stats['count']})"
        )
    lines.append("熔断扫描（阈值 / 正确保留 / 误杀命中 / 正确拒绝 / 漏网保留）：")
    for row in summary["threshold_sweep"]:
        lines.append(
            f"  t={row['threshold']:>5.1f}  保留命中={row['kept_hits']:<3}"
            f"误杀={row['false_rejects']:<3}正确拒绝={row['true_rejects']:<3}"
            f"漏网={row['false_keeps']}"
        )
    failed = [
        case
        for case in summary["cases"]
        if not case["retrieval_hit"]
        or not _relative_expected_hit(
            [row["source"] for row in case["ranked"]], case["expected_sources"]
        )
    ]
    if failed:
        lines.append("未达标样例：")
        for case in failed:
            top = "N/A" if case["top_score"] is None else f"{case['top_score']:.2f}"
            lines.append(f"  [{case['id']}] top={top}  {case['question']}")
            lines.append(f"      期望：{case['expected_sources']}")
            lines.append(
                "      精排："
                + ", ".join(
                    f"{row['source']}({row['score']:.2f})" for row in case["ranked"]
                )
            )
    return "\n".join(lines)


def write_json(summary: dict[str, Any], handle: TextIO) -> None:
    """把报告 dict 以 JSON 写入给定流（便于落盘做跨版本回归对比）."""
    json.dump(summary, handle, ensure_ascii=False, indent=2)
    handle.write("\n")


# ---------------- 配置装配（编排层） ----------------


def resolve_kb_dir() -> Path:
    """按 ``Config`` 解析知识库根目录（相对路径锚定项目根，与 ingestion 一致）."""
    from agent.config import ROOT_DIR, Config

    docs_dir = Path(Config.KNOWLEDGE_DOCS_DIR)
    if not docs_dir.is_absolute():
        docs_dir = ROOT_DIR / docs_dir
    return docs_dir.resolve()


def build_components_from_config() -> tuple[
    Path, BaseRetriever, BaseRetriever | None, RerankerScorer
]:
    """按 ``Config`` 装配评估组件：(知识库目录, 向量检索器, BM25, 重排器).

    复用 rag 节点的懒加载单例（向量库/BM25 索引全局只建一次）；BM25 在
    非 Chroma 后端不支持时降级为 None，评估只走向量路。重排器固定用
    本地 CrossEncoder（模型名/设备仍读 ``Config``），刻意不跟随
    ``RERANK_PROVIDER`` 的云端配置：评估要零 API 成本、可离线复现。
    """
    from agent.config import Config
    from agent.nodes import rag
    from agent.retrieval.reranker import build_reranker

    kb_dir = resolve_kb_dir()
    vector_retriever = rag._get_vector_retriever()
    bm25_retriever: BaseRetriever | None
    try:
        bm25_retriever = rag._get_bm25_retriever()
    except NotImplementedError:
        logger.warning("当前后端不支持 BM25，评估只走向量召回。")
        bm25_retriever = None
    # 评估固定走本地 CrossEncoder（不跟随 .env 的云端 rerank）：零 API 成本、可离线复现
    reranker = build_reranker("local", Config.RERANK_MODEL, device=Config.RERANK_DEVICE)
    return kb_dir, vector_retriever, bm25_retriever, reranker


def run_evaluation(
    golden_path: Path,
    *,
    top_k: int | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
    thresholds: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """端到端跑一遍黄金集：装配组件 → 逐样例评估 → 汇总报告.

    Args:
        golden_path: 黄金集 JSONL 路径。
        top_k: 精排保留条数，缺省取 ``Config.RERANK_TOP_K``。
        ks: 报告统计的截断档位。
        thresholds: 熔断扫描网格；为 None 时用默认 0~1 归一化分网格。

    Returns:
        可 JSON 序列化的评估报告 dict（见 :func:`build_summary`）。
    """
    from agent.config import Config

    cases = load_golden_cases(golden_path)
    kb_dir, vector_retriever, bm25_retriever, reranker = build_components_from_config()
    effective_top_k = top_k if top_k is not None else Config.RERANK_TOP_K
    # 评估固定本地重排，默认网格用 0~1 归一化分
    effective_thresholds = (
        thresholds if thresholds is not None else DEFAULT_THRESHOLD_GRID
    )
    results = [
        evaluate_case(
            case,
            kb_dir=kb_dir,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            reranker=reranker,
            top_k=effective_top_k,
        )
        for case in cases
    ]
    config_snapshot = {
        "vectorstore_provider": Config.VECTORSTORE_PROVIDER,
        "retrieval_k": Config.RETRIEVAL_K,
        "bm25_k": Config.BM25_K,
        "rerank_provider": "local (eval 固定，不跟随 .env)",
        "rerank_model": Config.RERANK_MODEL,
        "rerank_top_k": effective_top_k,
        "rerank_score_threshold": Config.RERANK_SCORE_THRESHOLD,
    }
    return build_summary(
        results,
        kb_dir=kb_dir,
        ks=ks,
        thresholds=effective_thresholds,
        config=config_snapshot,
    )
