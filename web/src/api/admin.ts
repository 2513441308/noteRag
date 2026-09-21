/**
 * 调试设置（热更新）API。
 * 走当前页面 origin + /admin，由 Vite 同源代理到 langgraph dev 的 admin 路由。
 */

import { requestJson, resolveBaseUrl } from "./http";

const ADMIN_URL = resolveBaseUrl(
  import.meta.env.VITE_ADMIN_URL as string | undefined,
  "/admin",
);

export type FieldType = "str" | "secret" | "int" | "float" | "bool" | "select";

export interface ConfigField {
  key: string;
  group: string;
  label: string;
  type: FieldType;
  help: string;
  optional: boolean;
  options: string[] | null;
  value: string | number | boolean | null;
  /** 仅 secret 字段：当前是否已设置（明文不下发） */
  secret_set?: boolean;
}

export async function getConfig(): Promise<ConfigField[]> {
  const data = await requestJson<{ fields: ConfigField[] }>(
    "GET",
    `${ADMIN_URL}/config`,
  );
  return data.fields;
}

export async function saveConfig(
  values: Record<string, string | boolean | number>,
): Promise<{ applied: string[]; caches_cleared: number }> {
  return requestJson<{ applied: string[]; caches_cleared: number }>(
    "POST",
    `${ADMIN_URL}/config`,
    { values },
  );
}
