/**
 * 前端公共 HTTP 工具：base URL 解析与 JSON 请求封装。
 * 三处后端地址（/lg、/admin、/admin/kb）共用同一套解析规则。
 */

/**
 * 解析后端 base URL。
 *
 * 优先级：环境变量覆盖 > 当前页面 origin + 同源代理前缀 > 本地 dev 兜底。
 *
 * @param envValue - `import.meta.env.VITE_*` 读出的地址（可能为 undefined）。
 * @param proxyPath - Vite dev server 上的同源代理前缀，如 `/lg`、`/admin`。
 * @returns 去掉尾部斜杠的 base URL。
 */
export function resolveBaseUrl(
  envValue: string | undefined,
  proxyPath: string,
): string {
  if (envValue) return envValue.replace(/\/$/, "");
  if (typeof window !== "undefined") {
    return `${window.location.origin}${proxyPath}`;
  }
  return `http://localhost:5173${proxyPath}`;
}

/**
 * 发送 JSON 请求并解析响应。
 *
 * 成功返回响应体（无正文或非 JSON 时返回 null）；非 2xx 时优先抛出后端
 * FastAPI 风格 `{detail}` 中的中文信息，否则抛 HTTP 状态错误。
 *
 * @param method - HTTP 方法。
 * @param url - 完整请求地址。
 * @param body - 请求体，传入时自动序列化为 JSON；undefined 表示不带正文。
 */
export async function requestJson<T>(
  method: string,
  url: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = (data as { detail?: string } | null)?.detail;
    throw new Error(detail ?? `请求失败：HTTP ${res.status}`);
  }
  return data as T;
}
