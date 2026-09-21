import { useEffect, useState } from "react";
import {
  getEvalJob,
  startEvalJob,
  type EvalCase,
  type EvalJob,
  type EvalSummary,
} from "../api/eval";
import { Modal } from "./Modal";

interface Props {
  open: boolean;
  onClose: () => void;
}

const POLL_INTERVAL_MS = 3000;

const STATUS_META: Record<EvalJob["status"], { label: string; cls: string }> = {
  running: { label: "评估进行中…", cls: "bg-amber-300/90 text-amber-950" },
  done: { label: "评估完成", cls: "bg-emerald-200/90 text-emerald-900" },
  error: { label: "评估失败", cls: "bg-red-200/90 text-red-900" },
};

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString("zh-CN", { hour12: false });
}

/** 精排 top-k 未命中任一期望来源（与后端 hit@k 判定一致）。 */
function caseMissed(c: EvalCase, k: number): boolean {
  if (!c.retrieval_hit) return true;
  return !c.ranked
    .slice(0, k)
    .some((row) => c.expected_sources.includes(row.source));
}

function scoreCls(score: number): string {
  if (score >= 0.7) return "text-emerald-800 font-bold";
  if (score >= 0.3) return "text-amber-800 font-semibold";
  return "text-red-700 font-semibold";
}

function MetricCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="note note-mint relative break-inside-avoid px-3 py-2 shadow-note-sm">
      <p className="text-[11px] font-semibold text-emerald-950/70">{label}</p>
      <p className="font-hand text-2xl leading-7 text-ink">{value}</p>
      {hint && <p className="text-[10px] text-emerald-950/60">{hint}</p>}
    </div>
  );
}

function ThresholdTable({ summary }: { summary: EvalSummary }) {
  return (
    <div className="overflow-x-auto rounded-sm border-2 border-amber-900/30 bg-white/70">
      <table className="w-full border-collapse text-center text-xs">
        <thead>
          <tr className="bg-amber-200/60 text-ink">
            <th className="px-2 py-1.5 text-left">熔断阈值</th>
            <th className="px-2 py-1.5">正确保留</th>
            <th className="px-2 py-1.5">误杀命中</th>
            <th className="px-2 py-1.5">正确拒绝</th>
            <th className="px-2 py-1.5">漏网保留</th>
          </tr>
        </thead>
        <tbody>
          {summary.threshold_sweep.map((row) => (
            <tr
              key={row.threshold}
              className="border-t border-dashed border-amber-900/20"
            >
              <td className="px-2 py-1.5 text-left font-mono">{row.threshold}</td>
              <td className="px-2 py-1.5 text-emerald-800">{row.kept_hits}</td>
              <td
                className={
                  row.false_rejects > 0
                    ? "bg-red-100/80 px-2 py-1.5 font-bold text-red-800"
                    : "px-2 py-1.5 text-ink/60"
                }
              >
                {row.false_rejects}
              </td>
              <td className="px-2 py-1.5 text-ink/70">{row.true_rejects}</td>
              <td
                className={
                  row.false_keeps > 0
                    ? "bg-amber-100/80 px-2 py-1.5 font-bold text-amber-800"
                    : "px-2 py-1.5 text-ink/60"
                }
              >
                {row.false_keeps}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FailedCases({ summary }: { summary: EvalSummary }) {
  const k = 3;
  const failed = summary.cases.filter((c) => caseMissed(c, k));
  if (failed.length === 0) {
    return (
      <p className="rounded-sm border-2 border-dashed border-emerald-700/50 bg-emerald-100/70 px-3 py-2 text-sm font-semibold text-emerald-900">
        🎉 全部 {summary.total} 条样例精排 top{k} 均命中期望来源。
      </p>
    );
  }
  return (
    <ul className="space-y-2">
      {failed.map((c) => (
        <li
          key={c.id}
          className="rounded-sm border-2 border-dashed border-amber-900/30 bg-white/70 px-3 py-2"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-sm bg-amber-300/80 px-1.5 py-0.5 text-[11px] font-bold text-amber-950">
              {c.id}
            </span>
            <span className="text-sm font-semibold text-ink">{c.question}</span>
            {!c.retrieval_hit && (
              <span className="rounded-sm bg-red-200/90 px-1.5 py-0.5 text-[10px] font-bold text-red-900">
                召回阶段未命中
              </span>
            )}
            {c.retrieval_hit && (
              <span className="rounded-sm bg-amber-200/90 px-1.5 py-0.5 text-[10px] font-bold text-amber-900">
                精排挤出 top{k}
              </span>
            )}
          </div>
          <p className="mt-1 text-[11px] text-ink/70">
            期望：{c.expected_sources.join(" / ")}
          </p>
          {c.ranked.length > 0 ? (
            <ol className="mt-1 space-y-0.5">
              {c.ranked.map((row) => {
                const expected = c.expected_sources.includes(row.source);
                return (
                  <li key={row.rank} className="flex items-center gap-2 text-xs">
                    <span className="w-4 shrink-0 text-ink/50">{row.rank}.</span>
                    <span
                      className={`min-w-0 flex-1 truncate ${
                        expected ? "text-emerald-800" : "text-ink"
                      }`}
                      title={row.source}
                    >
                      {expected ? "✓ " : ""}
                      {row.source}
                    </span>
                    <span className={`shrink-0 ${scoreCls(row.score)}`}>
                      {row.score.toFixed(3)}
                    </span>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="mt-1 text-xs text-red-700">熔断后无候选文档</p>
          )}
        </li>
      ))}
    </ul>
  );
}

export function EvalModal({ open, onClose }: Props) {
  const [job, setJob] = useState<EvalJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 打开时先尝试拉取最近一次任务（没有则进入空态）
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getEvalJob("latest")
      .then((latest) => {
        if (!cancelled) setJob(latest);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        // 404（后端 detail「还没有任何评估任务」）才是真正的空态；
        // 其余（网络抖动/5xx）提示错误，避免误清空。
        if (msg.includes("还没有")) {
          setJob(null);
        } else {
          setError(`读取最近一次评估失败：${msg}`);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const running = job?.status === "running";

  // 运行中轮询；弹窗关闭/任务结束后自动停止
  useEffect(() => {
    if (!open || !running || !job) return;
    const timer = setInterval(() => {
      getEvalJob(job.job_id)
        .then(setJob)
        .catch((err: unknown) =>
          setError(err instanceof Error ? err.message : String(err)),
        );
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [open, running, job]);

  const handleStart = async () => {
    setStarting(true);
    setError(null);
    try {
      const { job_id } = await startEvalJob();
      setJob({
        job_id,
        status: "running",
        started_at: new Date().toISOString(),
        finished_at: null,
        error: null,
        summary: job?.summary ?? null,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // 409：已有任务在跑（可能由别处触发），转为轮询那个任务
      if (msg.includes("已有")) {
        try {
          setJob(await getEvalJob("latest"));
        } catch (latestErr) {
          setError(
            latestErr instanceof Error ? latestErr.message : String(latestErr),
          );
        }
      } else {
        setError(msg);
      }
    } finally {
      setStarting(false);
    }
  };

  const summary = job?.summary ?? null;
  const sweepK = 3;

  return (
    <Modal
      open={open}
      onClose={onClose}
      label="检索评估"
      title="📊 检索评估"
      widthClass="w-[860px]"
      maxHeightClass="max-h-[88vh]"
      subtitle="黄金集 20 题 · 真实双路召回 + 本地 CrossEncoder 精排 · 不调聊天 LLM"
    >
      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-7 py-3">
        {/* 操作栏 */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={handleStart}
            disabled={starting || running}
            className="note note-mint relative px-5 py-2 text-sm font-bold shadow-note-sm transition hover:-translate-y-0.5 disabled:opacity-50"
          >
            {running ? "评估进行中…" : starting ? "提交中…" : "▶ 开始评估"}
          </button>
          {job && (
            <span
              className={`rounded-sm px-2 py-0.5 text-xs font-bold ${STATUS_META[job.status].cls}`}
            >
              {STATUS_META[job.status].label}
            </span>
          )}
          {job?.finished_at && (
            <span className="text-[11px] text-ink/60">
              完成于 {formatTime(job.finished_at)}
            </span>
          )}
          <span className="ml-auto text-[11px] text-ink/50">
            CPU 本地推理约 5–9 分钟，期间问答可正常使用
          </span>
        </div>

        {running && (
          <div className="rounded-sm border-2 border-dashed border-amber-700/50 bg-amber-100/70 px-3 py-2 text-sm text-amber-900">
            正在加载本地嵌入与 bge-reranker，逐题跑双路召回 + 精排，每 3 秒自动刷新…
          </div>
        )}

        {error && (
          <div className="rounded-sm border-2 border-dashed border-red-700/60 bg-red-100/70 px-3 py-2 text-sm text-red-900">
            {error}
          </div>
        )}

        {loading && !job && (
          <p className="py-10 text-center text-sm text-ink/60">正在读取最近一次评估…</p>
        )}

        {!loading && !summary && !running && !error && (
          <div className="flex flex-col items-center gap-2 py-12 text-center">
            <p className="font-hand text-2xl text-ink/80">还没有跑过评估</p>
            <p className="max-w-[420px] text-xs leading-5 text-ink/60">
              点击「开始评估」对内置黄金集跑一遍真实检索链路，产出 Hit/MRR、
              精排分数分布与熔断阈值扫描，用于发现「召回没问题但精排排错」
              这类回答层面看不出的回归。
            </p>
          </div>
        )}

        {summary && (
          <>
            {/* 指标卡 */}
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <MetricCard label="样例数" value={String(summary.total)} />
              <MetricCard label="Hit@1" value={pct(summary.metrics["hit@1"])} />
              <MetricCard
                label={`Hit@${sweepK}`}
                value={pct(summary.metrics["hit@3"])}
                hint="前三条精排结果命中期望来源"
              />
              <MetricCard label={`MRR@${sweepK}`} value={summary.metrics["mrr@3"].toFixed(3)} />
              <MetricCard
                label="召回阶段命中率"
                value={pct(summary.retrieval_hit)}
                hint="精排前的双路召回覆盖"
              />
              <MetricCard
                label="精排挤出"
                value={`${summary.rerank_demoted} 条`}
                hint="召回命中但被挤出 top3"
              />
            </div>

            {summary.score_stats && (
              <p className="text-xs text-ink/70">
                top1 精排分分布：min={summary.score_stats.min.toFixed(3)}  median=
                {summary.score_stats.median.toFixed(3)}  mean=
                {summary.score_stats.mean.toFixed(3)}  max=
                {summary.score_stats.max.toFixed(3)}（n={summary.score_stats.count}）
              </p>
            )}

            <details className="text-[11px] text-ink/55">
              <summary className="cursor-pointer select-none">配置快照</summary>
              <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5">
                {Object.entries(summary.config).map(([key, value]) => (
                  <span key={key}>
                    {key}={String(value)}
                  </span>
                ))}
              </div>
            </details>

            <section>
              <h3 className="mb-1 text-sm font-bold text-ink">
                熔断阈值扫描
                <span className="ml-2 text-[11px] font-normal text-ink/55">
                  红底=误杀正确命中，黄底=漏网未命中
                </span>
              </h3>
              <ThresholdTable summary={summary} />
            </section>

            <section className="pb-2">
              <h3 className="mb-1 text-sm font-bold text-ink">未达标样例</h3>
              <FailedCases summary={summary} />
            </section>
          </>
        )}
      </div>
    </Modal>
  );
}
