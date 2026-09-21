/**
 * 知识库管理 API。
 * 走当前页面 origin + /admin/kb，由 Vite 同源代理到 langgraph dev 的 admin 路由。
 * 文件内容上传用 base64（JSON），免去 multipart 依赖。
 */

import { requestJson, resolveBaseUrl } from "./http";

const KB_URL = `${resolveBaseUrl(
  import.meta.env.VITE_ADMIN_URL as string | undefined,
  "/admin",
)}/kb`;

export type KbStatus = "synced" | "modified" | "unindexed";

export interface KbDoc {
  /** 相对知识库根目录的 posix 路径 */
  path: string;
  name: string;
  suffix: string;
  editable: boolean;
  size_bytes: number;
  mtime: number;
  content_hash?: string | null;
  indexed: boolean;
  status: KbStatus;
  chunks: number;
}

export interface KbOrphan {
  source: string;
  rel: string | null;
  chunks: number;
}

export interface KbList {
  root: string;
  total_files: number;
  total_chunks: number;
  documents: KbDoc[];
  orphans: KbOrphan[];
  editable_suffixes: string[];
  supported_suffixes: string[];
  max_upload_bytes: number;
}

export interface KbWriteResult {
  ok: boolean;
  path: string;
  chunks: number;
  action: string;
}

const q = (path: string) => encodeURIComponent(path);

export async function listKb(): Promise<KbList> {
  return requestJson<KbList>("GET", `${KB_URL}/documents`);
}

export async function getKbText(path: string): Promise<{ content: string }> {
  return requestJson<{ content: string }>("GET", `${KB_URL}/document?path=${q(path)}`);
}

export async function createKbText(path: string, content: string): Promise<KbWriteResult> {
  return requestJson<KbWriteResult>("POST", `${KB_URL}/document`, { path, content });
}

export async function updateKbText(path: string, content: string): Promise<KbWriteResult> {
  return requestJson<KbWriteResult>("PUT", `${KB_URL}/document`, { path, content });
}

export async function uploadKb(
  path: string,
  contentBase64: string,
  overwrite: boolean,
): Promise<KbWriteResult> {
  return requestJson<KbWriteResult>("POST", `${KB_URL}/upload`, {
    path,
    content_base64: contentBase64,
    overwrite,
  });
}

export async function deleteKb(path: string): Promise<void> {
  await requestJson<unknown>("DELETE", `${KB_URL}/document?path=${q(path)}`);
}

export async function reindexKb(path: string): Promise<KbWriteResult> {
  return requestJson<KbWriteResult>("POST", `${KB_URL}/reindex?path=${q(path)}`);
}

export async function pruneKb(): Promise<{ removed: string[]; count: number }> {
  return requestJson<{ removed: string[]; count: number }>("POST", `${KB_URL}/prune`);
}

/** 把 File 编码为 base64（分块拼接，避免大文件 call stack 溢出） */
export async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}
