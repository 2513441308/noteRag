import { useEffect, useMemo, useRef, useState } from "react";
import {
  createKbText,
  deleteKb,
  fileToBase64,
  getKbText,
  listKb,
  pruneKb,
  reindexKb,
  updateKbText,
  uploadKb,
  type KbDoc,
  type KbList,
} from "../api/kb";
import { Modal } from "./Modal";

interface Props {
  open: boolean;
  onClose: () => void;
}

interface EditorState {
  create: boolean;
  path: string;
  content: string;
}

const ICONS: Record<string, string> = {
  ".md": "📝",
  ".markdown": "📝",
  ".txt": "📄",
  ".json": "🧩",
  ".py": "🐍",
  ".pdf": "📕",
};

const STATUS_META: Record<
  KbDoc["status"],
  { label: string; cls: string }
> = {
  synced: { label: "已同步", cls: "bg-emerald-200/80 text-emerald-900" },
  modified: { label: "已改动", cls: "bg-amber-300/90 text-amber-950" },
  unindexed: { label: "未入库", cls: "bg-red-200/90 text-red-900" },
};

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

export function KnowledgeModal({ open, onClose }: Props) {
  const [data, setData] = useState<KbList | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [busyPath, setBusyPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tip, setTip] = useState<string | null>(null);
  const [editor, setEditor] = useState<EditorState | null>(null);
  const [savingDoc, setSavingDoc] = useState(false);
  const [query, setQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 搜索在已加载的完整文档列表上做前端实时过滤：大小写不敏感，
  // 空格拆成多个关键词，需同时命中文件路径/名称才算匹配。
  const filteredDocs = useMemo(() => {
    const docs = data?.documents ?? [];
    const q = query.trim().toLowerCase();
    if (!q) return docs;
    const terms = q.split(/\s+/);
    return docs.filter((doc) => {
      const haystack = doc.path.toLowerCase();
      return terms.every((term) => haystack.includes(term));
    });
  }, [data, query]);

  const reload = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      setData(await listKb());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      setEditor(null);
      setTip(null);
      setError(null);
      setQuery("");
      void reload();
    }
  }, [open]);

  if (!open) return null;

  const closeEditor = () => setEditor(null);

  const startCreate = () => {
    setError(null);
    setTip(null);
    setEditor({ create: true, path: "", content: "" });
  };

  const startEdit = async (doc: KbDoc) => {
    setBusyPath(doc.path);
    setError(null);
    try {
      const { content } = await getKbText(doc.path);
      setEditor({ create: false, path: doc.path, content });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyPath(null);
    }
  };

  const saveEditor = async () => {
    if (!editor) return;
    const path = editor.path.trim();
    if (!path) {
      setError("请填写文件名（如 新笔记.md）。");
      return;
    }
    setSavingDoc(true);
    setError(null);
    try {
      const res = editor.create
        ? await createKbText(path, editor.content)
        : await updateKbText(path, editor.content);
      setTip(`已保存「${res.path}」，生成 ${res.chunks} 个切片，检索已热更新。`);
      setEditor(null);
      await reload(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingDoc(false);
    }
  };

  const handlePickFiles = () => fileInputRef.current?.click();

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    setTip(null);
    try {
      for (const file of Array.from(files)) {
        const b64 = await fileToBase64(file);
        try {
          await uploadKb(file.name, b64, false);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          if (msg.includes("已存在")) {
            const ok = window.confirm(
              `「${file.name}」已存在，要用新文件覆盖并重新索引吗？`,
            );
            if (!ok) continue;
            await uploadKb(file.name, b64, true);
          } else {
            throw err;
          }
        }
      }
      setTip(`已处理 ${files.length} 个文件。`);
      await reload(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleReindex = async (doc: KbDoc) => {
    setBusyPath(doc.path);
    setError(null);
    try {
      const res = await reindexKb(doc.path);
      setTip(`「${res.path}」已重新索引，共 ${res.chunks} 个切片。`);
      await reload(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyPath(null);
    }
  };

  const handleDelete = async (doc: KbDoc) => {
    if (!window.confirm(`确定从知识库删除「${doc.path}」吗？\n文件与其全部切片都会被移除，不可恢复。`)) {
      return;
    }
    setBusyPath(doc.path);
    setError(null);
    try {
      await deleteKb(doc.path);
      setTip(`已删除「${doc.path}」。`);
      await reload(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyPath(null);
    }
  };

  const handlePrune = async () => {
    const n = data?.orphans.length ?? 0;
    if (!window.confirm(`检测到 ${n} 个源文件已删除但切片残留，确定清理这些游离切片吗？`)) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await pruneKb();
      setTip(`已清理 ${res.count} 个文件的游离切片。`);
      await reload(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const accept = (
    data?.supported_suffixes ?? [".md", ".txt", ".json", ".py", ".pdf"]
  ).join(",");

  return (
    <Modal
      open={open}
      onClose={onClose}
      label="知识库管理"
      title="📚 知识库"
      widthClass="w-[780px]"
      maxHeightClass="max-h-[88vh]"
      subtitle={
        data
          ? `${data.total_files} 个文档 · ${data.total_chunks} 个切片 · 改完提问即时生效`
          : "管理源文档与向量索引，改完提问即时生效"
      }
    >
      {editor ? (
          /* ---------------- 文本编辑器视图 ---------------- */
          <div className="flex min-h-0 flex-1 flex-col px-7 py-3">
            <label className="mb-1 block text-xs font-semibold text-ink">
              文件路径（相对知识库根目录）
            </label>
            <input
              value={editor.path}
              disabled={!editor.create || savingDoc}
              onChange={(e) =>
                setEditor((prev) => (prev ? { ...prev, path: e.target.value } : prev))
              }
              placeholder="例如：langgraph/新笔记.md（支持 .md/.txt/.json/.py）"
              className="mb-3 w-full rounded-sm border-b-2 border-amber-900/40 bg-white/80 px-2 py-1.5 text-sm text-ink outline-none focus:border-amber-700 disabled:opacity-60"
            />
            <textarea
              value={editor.content}
              onChange={(e) =>
                setEditor((prev) => (prev ? { ...prev, content: e.target.value } : prev))
              }
              placeholder="在这里书写正文，保存后会自动切分并嵌入向量库…"
              className="min-h-[320px] flex-1 resize-none rounded-sm border-2 border-dashed border-amber-900/30 bg-white/80 p-3 font-mono text-[13px] leading-6 text-ink outline-none focus:border-amber-700"
            />
            {error && (
              <div className="mt-2 rounded-sm border-2 border-dashed border-red-700/60 bg-red-100/70 px-3 py-2 text-sm text-red-900">
                {error}
              </div>
            )}
            <div className="mt-3 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={closeEditor}
                disabled={savingDoc}
                className="px-4 py-2 text-sm font-bold text-amber-900/70 transition hover:text-ink disabled:opacity-50"
              >
                取消
              </button>
              <button
                type="button"
                onClick={saveEditor}
                disabled={savingDoc}
                className="note note-mint relative px-5 py-2 text-sm font-bold shadow-note-sm transition hover:-translate-y-0.5 disabled:opacity-50"
              >
                {savingDoc ? "嵌入中…" : "📌 保存并索引"}
              </button>
            </div>
          </div>
        ) : (
          /* ---------------- 列表视图 ---------------- */
          <>
            <div className="flex flex-wrap items-center gap-2 px-7 py-2">
              <button
                type="button"
                onClick={startCreate}
                disabled={busy}
                className="note note-mint relative px-3 py-1.5 text-xs font-bold shadow-note-sm transition hover:-translate-y-0.5 disabled:opacity-50"
              >
                ＋ 新建文本
              </button>
              <button
                type="button"
                onClick={handlePickFiles}
                disabled={busy}
                className="note relative bg-white/80 px-3 py-1.5 text-xs font-bold text-ink shadow-note-sm transition hover:-translate-y-0.5 disabled:opacity-50"
              >
                {busy ? "处理中…" : "📤 上传文件"}
              </button>
              <button
                type="button"
                onClick={() => void reload()}
                disabled={busy || loading}
                className="note relative bg-white/80 px-3 py-1.5 text-xs font-bold text-ink shadow-note-sm transition hover:-translate-y-0.5 disabled:opacity-50"
              >
                🔄 刷新
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={accept}
                className="hidden"
                onChange={(e) => void handleFiles(e.target.files)}
              />
              <span className="ml-auto text-[10px] text-amber-900/50">
                支持 {(data?.supported_suffixes ?? []).join(" / ")}
              </span>
            </div>

            {/* 搜索栏：在完整文档列表上实时过滤显示 */}
            <div className="flex items-center gap-2 px-7 pb-1 pt-1">
              <div className="relative flex-1">
                <span className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-sm">
                  🔍
                </span>
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="搜索文件名或路径（如 agent、langgraph/docs），空格分隔多个词"
                  className="w-full rounded-sm border-b-2 border-amber-900/40 bg-white/80 py-1.5 pl-8 pr-8 text-sm text-ink outline-none transition focus:border-amber-700"
                />
                {query && (
                  <button
                    type="button"
                    onClick={() => setQuery("")}
                    aria-label="清空搜索"
                    title="清空搜索"
                    className="absolute right-1.5 top-1/2 flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded-full bg-amber-900/15 text-xs font-bold text-amber-900/70 transition hover:bg-amber-900/25"
                  >
                    ×
                  </button>
                )}
              </div>
              {data && (
                <span className="shrink-0 text-[11px] font-semibold text-amber-900/60">
                  {query.trim() ? `${filteredDocs.length} / ${data.documents.length}` : `${data.documents.length} 个`}
                </span>
              )}
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-7 py-2">
              {error && (
                <div className="mb-3 rounded-sm border-2 border-dashed border-red-700/60 bg-red-100/70 px-3 py-2 text-sm text-red-900">
                  {error}
                </div>
              )}
              {tip && !error && (
                <div className="mb-3 rounded-sm border-2 border-dashed border-emerald-700/60 bg-emerald-100/70 px-3 py-2 text-sm text-emerald-900">
                  ✅ {tip}
                </div>
              )}

              {loading ? (
                <p className="py-12 text-center text-sm text-amber-900/70">
                  正在盘点文件柜…
                </p>
              ) : data && data.documents.length === 0 ? (
                <p className="px-1 py-12 text-center text-sm leading-7 text-amber-900/60">
                  知识库里还没有文档。
                  <br />
                  点「＋ 新建文本」写一篇，或「📤 上传文件」放进 PDF / Markdown。
                </p>
              ) : filteredDocs.length === 0 ? (
                <p className="px-1 py-12 text-center text-sm leading-7 text-amber-900/60">
                  没有匹配「{query.trim()}」的文档。
                  <br />
                  换个关键词，或点搜索框右侧 × 清空条件。
                </p>
              ) : (
                <ul className="space-y-2">
                  {filteredDocs.map((doc) => {
                    const meta = STATUS_META[doc.status];
                    const rowBusy = busyPath === doc.path;
                    return (
                      <li
                        key={doc.path}
                        className="note relative flex items-center gap-3 !bg-white/75 px-3 py-2 shadow-note-sm"
                      >
                        <span className="text-xl">{ICONS[doc.suffix] ?? "📎"}</span>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-bold text-ink" title={doc.path}>
                            {doc.path}
                          </p>
                          <p className="mt-0.5 text-[11px] text-amber-900/60">
                            {doc.chunks} 切片 · {formatSize(doc.size_bytes)}
                          </p>
                        </div>
                        <span
                          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold ${meta.cls}`}
                        >
                          {meta.label}
                        </span>
                        <span className="flex shrink-0 gap-1">
                          {doc.editable && (
                            <button
                              type="button"
                              title="在线编辑"
                              disabled={busy || rowBusy}
                              onClick={() => void startEdit(doc)}
                              className="flex h-6 w-6 items-center justify-center rounded-full bg-amber-700/80 text-xs text-amber-50 shadow transition hover:scale-110 hover:bg-amber-800 disabled:opacity-50"
                            >
                              ✏️
                            </button>
                          )}
                          <button
                            type="button"
                            title="重新索引"
                            disabled={busy || rowBusy}
                            onClick={() => void handleReindex(doc)}
                            className="flex h-6 w-6 items-center justify-center rounded-full bg-sky-700/80 text-xs text-amber-50 shadow transition hover:scale-110 hover:bg-sky-800 disabled:opacity-50"
                          >
                            🔁
                          </button>
                          <button
                            type="button"
                            title="删除"
                            disabled={busy || rowBusy}
                            onClick={() => void handleDelete(doc)}
                            className="flex h-6 w-6 items-center justify-center rounded-full bg-red-700/80 text-xs font-bold text-amber-50 shadow transition hover:scale-110 hover:bg-red-800 disabled:opacity-50"
                          >
                            ×
                          </button>
                        </span>
                      </li>
                    );
                  })}
                </ul>
              )}

              {data && data.orphans.length > 0 && (
                <button
                  type="button"
                  onClick={() => void handlePrune()}
                  disabled={busy}
                  className="mt-3 w-full rounded-sm border-2 border-dashed border-red-700/50 bg-red-100/60 px-3 py-2 text-xs font-bold text-red-900 transition hover:bg-red-100 disabled:opacity-50"
                >
                  🧹 检测到 {data.orphans.length} 个源文件已删但切片残留，点此清理
                </button>
              )}
            </div>
          </>
        )}
    </Modal>
  );
}
