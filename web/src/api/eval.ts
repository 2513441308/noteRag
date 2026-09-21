/**
 * 检索层离线评估 API。
 * 评估在后台线程跑本地模型（数分钟）：POST 触发拿 job_id，GET 轮询。
 * 走当前页面 origin + /admin，由 Vite 同源代理到 langgraph dev。
 */

import { requestJson, resolveBaseUrl } from "./http";

const EVAL_URL = `${resolveBaseUrl(
  import.meta.env.VITE_ADMIN_URL as string | undefined,
  "/admin",
)}/eval/retrieval`;

export type EvalJobStatus = "running" | "done" | "error";

export interface EvalMetrics {
  "hit@1": number;
  "recall@1": number;
  "mrr@1": number;
  "hit@3": number;
  "recall@3": number;
  "mrr@3": number;
}

export interface ScoreStats {
  count: number;
  min: number;
  median: number;
  mean: number;
  max: number;
}

export interface ThresholdRow {
  threshold: number;
  /** 命中且分数达标（正确保留） */
  kept_hits: number;
  /** 命中却被熔断误杀，最需要关注 */
  false_rejects: number;
  /** 未命中且被拒绝（正确熔断） */
  true_rejects: number;
  /** 未命中但被保留（熔断漏网） */
  false_keeps: number;
}

export interface RankedRow {
  rank: number;
  source: string;
  score: number;
}

export interface EvalCase {
  id: string;
  question: string;
  expected_sources: string[];
  /** 召回阶段（精排前）是否覆盖期望来源 */
  retrieval_hit: boolean;
  top_score: number | null;
  ranked: RankedRow[];
}

export interface EvalSummary {
  total: number;
  config: Record<string, string | number>;
  metrics: EvalMetrics;
  /** 召回阶段命中率（精排前） */
  retrieval_hit: number;
  /** 召回命中但被精排挤出 top-k 的样例数 */
  rerank_demoted: number;
  score_stats: ScoreStats | null;
  threshold_sweep: ThresholdRow[];
  cases: EvalCase[];
}

export interface EvalJob {
  job_id: string;
  status: EvalJobStatus;
  started_at: string;
  finished_at: string | null;
  error: string | null;
  summary: EvalSummary | null;
}

export async function startEvalJob(): Promise<Pick<EvalJob, "job_id" | "status">> {
  return requestJson<Pick<EvalJob, "job_id" | "status">>("POST", EVAL_URL);
}

export async function getEvalJob(jobId: string): Promise<EvalJob> {
  return requestJson<EvalJob>("GET", `${EVAL_URL}/${encodeURIComponent(jobId)}`);
}
