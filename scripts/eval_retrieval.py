#!/usr/bin/env python
"""检索层离线评估入口：跑黄金集，打印 Hit/MRR 报告，可选导出 JSON.

用法（在项目根目录）::

    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --json out/eval.json
    uv run python scripts/eval_retrieval.py --golden eval/golden_retrieval.jsonl

跑的是本地真实嵌入模型 + 本地/云端 rerank（按 .env 配置），不调用聊天 LLM；
首次加载模型约 20 秒属正常冷启动。熔断扫描表用于标定
``RERANK_SCORE_THRESHOLD``：优先选「误杀」为 0 且「正确拒绝」尽量多的档位。
评估固定用本地 CrossEncoder（不调云端、零 API 成本），可用 ``--thresholds``
覆盖默认 0~1 归一化分网格。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接 `python scripts/eval_retrieval.py` 运行（未 editable 安装时也能找到 src）
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.retrieval.evaluation import (  # noqa: E402
    DEFAULT_KS,
    format_report,
    run_evaluation,
    write_json,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLDEN = _PROJECT_ROOT / "eval" / "golden_retrieval.jsonl"


def _parse_ks(raw: str) -> tuple[int, ...]:
    """把 ``"1,3,5"`` 解析为正整数档位元组."""
    try:
        ks = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--ks 需要形如 1,3,5 的整数列表：{raw}"
        ) from exc
    if not ks or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError(f"--ks 档位必须全为正整数：{raw}")
    return tuple(sorted(set(ks)))


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    """把 ``"0,2,4"`` 解析为升序浮点阈值元组."""
    try:
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--thresholds 需要形如 0,2,4 的数字列表：{raw}"
        ) from exc
    return tuple(sorted(set(values)))


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器."""
    parser = argparse.ArgumentParser(
        description="检索层离线评估：黄金集双路召回 + 精排指标与熔断扫描"
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=_DEFAULT_GOLDEN,
        help=f"黄金集 JSONL 路径（默认 {_DEFAULT_GOLDEN.relative_to(_PROJECT_ROOT)}）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="精排保留条数，缺省读 RERANK_TOP_K",
    )
    parser.add_argument(
        "--ks",
        type=_parse_ks,
        default=DEFAULT_KS,
        help="指标统计档位，逗号分隔，如 1,3,5（默认 1,3）",
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=None,
        help="熔断扫描阈值网格，逗号分隔（默认 0.1,0.3,0.5,0.7,0.9）",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        dest="json_path",
        help="把完整报告（含逐样例明细）写入该 JSON 文件",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口：解析参数 → 跑评估 → 打印报告 / 写 JSON.

    Returns:
        进程退出码：0 成功，2 参数/文件错误。
    """
    args = build_parser().parse_args(argv)
    if not args.golden.is_file():
        print(f"黄金集不存在：{args.golden}", file=sys.stderr)
        return 2

    summary = run_evaluation(
        args.golden,
        top_k=args.top_k,
        ks=args.ks,
        thresholds=args.thresholds,
    )
    print(format_report(summary))

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        with args.json_path.open("w", encoding="utf-8") as handle:
            write_json(summary, handle)
        print(f"\n完整报告已写入：{args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
