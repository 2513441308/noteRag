import { useEffect, useRef, useState } from "react";
import type { ThreadSummary } from "../api/client";

interface Props {
  threads: ThreadSummary[];
  currentId: string | null;
  disabled: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onOpenSettings: () => void;
  onOpenKnowledge: () => void;
  onOpenEval: () => void;
}

const TILTS = [-1.4, 1.2, -0.8, 1.8, -1.8, 0.8];

export function Sidebar({
  threads,
  currentId,
  disabled,
  onNew,
  onSelect,
  onDelete,
  onRename,
  onOpenSettings,
  onOpenKnowledge,
  onOpenEval,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editingId]);

  const startEdit = (t: ThreadSummary) => {
    if (disabled) return;
    setEditingId(t.thread_id);
    setDraft(t.title);
  };

  const commit = () => {
    const id = editingId;
    const title = draft.trim();
    setEditingId(null);
    if (id && title) onRename(id, title);
  };

  const cancel = () => setEditingId(null);

  return (
    <aside className="relative z-10 flex h-full w-[260px] shrink-0 flex-col border-r-4 border-double border-amber-950/40 bg-cork-dark/35 backdrop-blur-[2px]">
      <div className="px-5 pb-4 pt-6">
        <h1 className="font-hand text-3xl leading-tight text-amber-50 drop-shadow-[0_2px_2px_rgba(0,0,0,.4)]">
          便签问答板
        </h1>
        <p className="mt-0.5 text-xs text-amber-100/70">
          RAG Agent · 贴一张问题，等一张答案
        </p>
      </div>

      <div className="px-4 pb-3">
        <button
          type="button"
          onClick={onNew}
          disabled={disabled}
          className={`note note-hover relative w-full px-3 py-2.5 text-left text-sm font-bold shadow-note-sm transition hover:-translate-y-0.5 disabled:opacity-50 ${
            currentId === null ? "note-mint shadow-note" : "note-yellow"
          }`}
          style={{ transform: "rotate(-1deg)" }}
        >
          {currentId === null && <span className="tape tape-sm" aria-hidden />}
          ＋ 写一张新便签
        </button>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto px-4 pb-6">
        {threads.map((t, i) => {
          const active = t.thread_id === currentId;
          const editing = editingId === t.thread_id;
          return (
            <div
              key={t.thread_id}
              role="button"
              tabIndex={0}
              aria-current={active}
              onClick={() => {
                if (!editing && !disabled) onSelect(t.thread_id);
              }}
              onDoubleClick={() => startEdit(t)}
              onKeyDown={(e) => {
                if (editing) return;
                if (!disabled && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  onSelect(t.thread_id);
                }
              }}
              className={`note note-hover group relative block w-full cursor-pointer px-3 py-2 text-sm transition ${
                active ? "note-mint shadow-note" : "shadow-note-sm opacity-90"
              }`}
              style={{
                transform: editing
                  ? "rotate(0deg)"
                  : active
                    ? "rotate(0deg)"
                    : `rotate(${TILTS[i % TILTS.length]}deg)`,
                background: active || editing ? undefined : "#fdf3c0",
              }}
            >
              {active && !editing && (
                <span className="tape tape-sm" aria-hidden />
              )}

              {editing ? (
                <input
                  ref={inputRef}
                  value={draft}
                  maxLength={60}
                  onChange={(e) => setDraft(e.target.value)}
                  onClick={(e) => e.stopPropagation()}
                  onBlur={commit}
                  onKeyDown={(e) => {
                    e.stopPropagation();
                    if (e.key === "Enter") commit();
                    if (e.key === "Escape") cancel();
                  }}
                  className="w-full rounded-sm border-b-2 border-dashed border-amber-700/60 bg-white/70 px-1 py-0.5 text-sm font-semibold text-ink outline-none"
                  placeholder="给便签起个名字…"
                />
              ) : (
                <span
                  title="双击可重命名"
                  className="line-clamp-2 break-words pr-11 font-semibold leading-5 text-ink"
                >
                  {active && <span className="pin mr-1 !h-2 !w-2 align-middle" />}
                  {t.title}
                </span>
              )}

              {!editing && (
                <span className="absolute right-1.5 top-1.5 flex gap-1 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
                  <button
                    type="button"
                    title="重命名（也可双击便签）"
                    aria-label={`重命名会话 ${t.title}`}
                    disabled={disabled}
                    onClick={(e) => {
                      e.stopPropagation();
                      startEdit(t);
                    }}
                    className="flex h-5 w-5 items-center justify-center rounded-full bg-amber-700/80 text-[11px] leading-none text-amber-50 shadow transition hover:scale-110 hover:bg-amber-800 disabled:cursor-default"
                  >
                    ✎
                  </button>
                  <button
                    type="button"
                    title="撕掉这张便签"
                    aria-label={`删除会话 ${t.title}`}
                    disabled={disabled}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (disabled) return;
                      if (
                        window.confirm(
                          `确定撕掉「${t.title}」这张便签吗？此操作不可恢复。`,
                        )
                      ) {
                        onDelete(t.thread_id);
                      }
                    }}
                    className="flex h-5 w-5 items-center justify-center rounded-full bg-red-700/80 text-xs font-bold leading-none text-amber-50 shadow transition hover:scale-110 hover:bg-red-800 disabled:cursor-default"
                  >
                    ×
                  </button>
                </span>
              )}
            </div>
          );
        })}
        {threads.length === 0 && (
          <p className="px-1 pt-6 text-center text-xs leading-5 text-amber-100/60">
            看板上还没有便签
            <br />
            在下方直接提问，就会贴出第一张～
          </p>
        )}
      </div>

      {/* 左下角：知识库与调试设置入口 */}
      <div className="space-y-1 border-t-2 border-double border-amber-950/30 p-3">
        <button
          type="button"
          onClick={onOpenKnowledge}
          className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-amber-100/80 transition hover:bg-amber-100/10 hover:text-amber-50"
          title="上传 / 新建 / 编辑 / 删除知识库文档，改完即时生效"
        >
          <span className="text-base">📚</span>
          <span>知识库</span>
          <span className="ml-auto text-[10px] text-amber-100/40">增删改查</span>
        </button>
        <button
          type="button"
          onClick={onOpenEval}
          className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-amber-100/80 transition hover:bg-amber-100/10 hover:text-amber-50"
          title="对黄金集跑真实检索链路，查看 Hit/MRR 与熔断阈值扫描"
        >
          <span className="text-base">📊</span>
          <span>检索评估</span>
          <span className="ml-auto text-[10px] text-amber-100/40">离线</span>
        </button>
        <button
          type="button"
          onClick={onOpenSettings}
          className="flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-amber-100/80 transition hover:bg-amber-100/10 hover:text-amber-50"
          title="调整模型 / 检索 / 压缩等环境变量，即时生效"
        >
          <span className="text-base">⚙️</span>
          <span>调试设置</span>
          <span className="ml-auto text-[10px] text-amber-100/40">热更新</span>
        </button>
      </div>
    </aside>
  );
}
