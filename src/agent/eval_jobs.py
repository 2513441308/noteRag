"""检索评估的后台任务注册表（进程内内存态）.

评估跑的是真实本地嵌入 + 本地 CrossEncoder，20 题黄金集在 CPU 上约需
数分钟，不能占着 HTTP 请求同步等待：:func:`start_evaluation_job` 只做
登记并拉起一个 daemon 线程，前端凭 ``job_id`` 轮询状态。

设计取舍：

- **单任务互斥**：评估是 CPU 密集型，并发跑只会互相拖慢、抢模型，
  已有 running 任务时再触发抛 :class:`EvalJobBusy`（HTTP 409）；
- **内存态、不持久化**：评估只读且可随时重跑，重启丢失进行中/历史
  任务没有实际代价，只保留最近 :data:`_JOB_KEEP` 个已结束任务；
- **runner 可注入**：默认 runner 懒加载地调用
  ``retrieval.evaluation.run_evaluation``（避免本模块被 import 时
  拖入重依赖），测试传入脚本函数即可，不加载真实模型。
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 注册表最多保留的任务数（含当前任务），超出后丢弃最旧的已结束任务
_JOB_KEEP = 10


@dataclass
class EvalJob:
    """一次检索评估任务的可变状态.

    Attributes:
        job_id: 任务唯一 id（uuid hex），也是轮询路径参数。
        status: ``running`` / ``done`` / ``error``。
        started_at: 开始时间（UTC ISO8601）。
        finished_at: 结束时间（UTC ISO8601），进行中为 None。
        error: 失败时的 ``异常类型: 信息``，成功/进行中为 None。
        summary: 成功后的评估报告 dict（``build_summary`` 产物）。
    """

    job_id: str
    status: str
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为可直接 JSON 响应的 dict."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "summary": self.summary,
        }


class EvalJobBusy(Exception):
    """已有评估任务在运行时再次触发（调用方应映射为 HTTP 409）."""


def _utc_now_iso() -> str:
    """当前 UTC 时间的 ISO8601 字符串（带时区，前端可直接转本地时间）."""
    return datetime.now(UTC).isoformat()


def _default_runner() -> dict[str, Any]:
    """生产环境默认评估执行体：对项目内置黄金集跑完整检索评估.

    懒导入放在函数内：模型加载等重逻辑全部收在 ``run_evaluation`` 里，
    import 本模块本身保持轻量。
    """
    from agent.config import ROOT_DIR
    from agent.retrieval.evaluation import run_evaluation

    golden_path = ROOT_DIR / "eval" / "golden_retrieval.jsonl"
    return run_evaluation(golden_path)


_jobs: dict[str, EvalJob] = {}
# 任务按创建顺序排列的 id 序列（与 _jobs 同步增删）
_job_order: list[str] = []
_lock = threading.Lock()


def start_evaluation_job(
    runner: Callable[[], dict[str, Any]] | None = None,
) -> str:
    """登记并在后台线程启动一次评估，立即返回 job_id.

    Args:
        runner: 评估执行体（无参，返回报告 dict）；为 None 时用
            :func:`_default_runner`。运行时才解析全局默认 runner，
            便于测试 monkeypatch。

    Returns:
        新建任务的 job_id。

    Raises:
        EvalJobBusy: 已有 running 任务（本进程不允许评估并发）。
    """
    with _lock:
        if any(job.status == "running" for job in _jobs.values()):
            raise EvalJobBusy("已有检索评估任务在运行。")
        job_id = uuid.uuid4().hex
        _jobs[job_id] = EvalJob(
            job_id=job_id, status="running", started_at=_utc_now_iso()
        )
        _job_order.append(job_id)
        _trim_locked()

    effective_runner = runner if runner is not None else _default_runner
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, effective_runner),
        name=f"retrieval-eval-{job_id[:8]}",
        daemon=True,
    )
    thread.start()
    logger.info("检索评估任务已启动：%s", job_id)
    return job_id


def _run_job(job_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    """后台线程主体：执行 runner 并把结果/异常写回任务状态.

    模型加载或评估抛出的任何异常都在此兜底成 ``error`` 状态，
    绝不让异常逸出线程（daemon 线程未捕获异常只会打一条难排查的栈）。
    """
    try:
        summary = runner()
        with _lock:
            job = _jobs[job_id]
            job.status = "done"
            job.summary = summary
            job.finished_at = _utc_now_iso()
        logger.info("检索评估任务完成：%s", job_id)
    except Exception as exc:  # noqa: BLE001 - 后台任务需要兜底全部异常
        with _lock:
            job = _jobs[job_id]
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = _utc_now_iso()
        logger.exception("检索评估任务失败：%s", job_id)


def get_job(job_id: str) -> EvalJob | None:
    """按 id 取任务快照，不存在返回 None."""
    with _lock:
        return _jobs.get(job_id)


def latest_job_id() -> str | None:
    """最近创建的任务 id（无论状态），无任务时返回 None."""
    with _lock:
        return _job_order[-1] if _job_order else None


def _trim_locked() -> None:
    """任务数超上限时丢弃最旧的**已结束**任务（持锁时调用）.

    进行中的任务永不丢弃；若历史全在跑（互斥下实际不可能），宁可保留.
    """
    while len(_job_order) > _JOB_KEEP:
        for index, old_id in enumerate(_job_order):
            if _jobs[old_id].status != "running":
                _job_order.pop(index)
                del _jobs[old_id]
                break
        else:
            return


def reset_registry() -> None:
    """清空任务注册表（仅供测试隔离使用）."""
    with _lock:
        _jobs.clear()
        _job_order.clear()
