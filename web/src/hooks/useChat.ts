import { useCallback, useEffect, useRef, useState } from "react";
import {
  createThread,
  deleteThread as deleteThreadApi,
  listThreads,
  loadThreadMessages,
  renameThread as renameThreadApi,
  renameThreadBestEffort,
  streamAsk,
  type ChatMessage,
  type ThreadSummary,
} from "../api/client";

let localSeq = 0;
const nextId = () => `local-${Date.now()}-${localSeq++}`;

/**
 * 启动引导：只读取会话列表并打开最近一张；没有会话时保持「本地草稿态」
 * （currentId = null），不预先创建空会话——会话在发出第一条消息时才落库。
 * React StrictMode 开发模式会挂载两次，用模块级 Promise 共享同一次结果。
 */
type Bootstrap = {
  threads: ThreadSummary[];
  currentId: string | null;
  messages: ChatMessage[];
  error: string | null;
};
let bootstrapPromise: Promise<Bootstrap> | null = null;

function bootstrap(): Promise<Bootstrap> {
  if (bootstrapPromise) return bootstrapPromise;
  bootstrapPromise = (async () => {
    const list = await listThreads().catch((err) => {
      console.error("[useChat] listThreads 失败：", err);
      return [] as ThreadSummary[];
    });
    if (list.length === 0) {
      // 无会话：草稿态，等用户发第一条消息时再建
      return { threads: [], currentId: null, messages: [], error: null };
    }
    const first = list[0];
    const history = await loadThreadMessages(first.thread_id).catch((err) => {
      console.error("[useChat] 历史加载失败：", err);
      return [] as ChatMessage[];
    });
    return { threads: list, currentId: first.thread_id, messages: history, error: null };
  })();
  return bootstrapPromise;
}

export function useChat() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const streamingIdRef = useRef<string | null>(null);

  const refreshThreads = useCallback(async () => {
    try {
      setThreads(await listThreads());
      setError(null);
    } catch (err) {
      console.error("[useChat] 加载会话列表失败：", err);
      setError("无法连接后端，请确认 langgraph dev 已在 2024 端口启动");
    }
  }, []);

  // 启动时加载会话列表；有则打开最近一张，无则进入草稿态
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    bootstrap().then((result) => {
      if (cancelled) return;
      setThreads(result.threads);
      setCurrentId(result.currentId);
      setMessages(result.messages);
      if (result.error) setError(result.error);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
    // 仅启动时执行一次
  }, []);

  const selectThread = useCallback(async (threadId: string) => {
    if (streamingIdRef.current) return; // 流式中不切换，避免状态错乱
    setCurrentId(threadId);
    setError(null);
    try {
      setMessages(await loadThreadMessages(threadId));
    } catch (err) {
      console.error("[useChat] 历史加载失败：", err);
      setMessages([]);
      setError("历史加载失败（后端可能已重启，内存会话已清空）");
    }
  }, []);

  /** 新建便签：仅切到本地草稿态，不调后端；发第一条消息时才真正创建 */
  const newThread = useCallback(() => {
    if (streamingIdRef.current) return;
    setCurrentId(null);
    setMessages([]);
    setError(null);
  }, []);

  /** 删除会话；删的是当前会话时自动切到剩余最近一张，没有则回草稿态 */
  const removeThread = useCallback(
    async (threadId: string) => {
      if (streamingIdRef.current) return;
      const remaining = threads.filter((t) => t.thread_id !== threadId);
      try {
        await deleteThreadApi(threadId);
      } catch (err) {
        console.error("[useChat] 删除会话失败：", err);
        setError("删除失败，请检查后端服务");
        return;
      }
      setThreads(remaining);
      if (threadId === currentId) {
        const next = remaining[0] ?? null;
        setCurrentId(next?.thread_id ?? null);
        if (next) {
          setMessages(await loadThreadMessages(next.thread_id).catch(() => []));
        } else {
          setMessages([]);
        }
      }
    },
    [threads, currentId],
  );

  /** 重命名会话：乐观更新本地列表，后端失败时回滚标题 */
  const renameThread = useCallback(
    async (threadId: string, title: string) => {
      const trimmed = title.trim();
      if (!trimmed) return;
      const old = threads.find((t) => t.thread_id === threadId);
      setThreads((prev) =>
        prev.map((t) =>
          t.thread_id === threadId ? { ...t, title: trimmed } : t,
        ),
      );
      try {
        await renameThreadApi(threadId, trimmed);
      } catch (err) {
        console.error("[useChat] 重命名失败：", err);
        setError("重命名失败，请检查后端服务");
        if (old) {
          setThreads((prev) =>
            prev.map((t) =>
              t.thread_id === threadId ? { ...t, title: old.title } : t,
            ),
          );
        }
      }
    },
    [threads],
  );

  const send = useCallback(
    async (rawQuestion: string) => {
      const question = rawQuestion.trim();
      if (!question || streaming) return;

      // 草稿态：发第一条消息时才真正创建会话（避免产生空便签）
      let threadId = currentId;
      const createdNow = !threadId;
      if (!threadId) {
        try {
          const t = await createThread();
          threadId = t.thread_id;
          setThreads((prev) => [t, ...prev]);
          setCurrentId(t.thread_id);
        } catch (err) {
          console.error("[useChat] createThread 失败：", err);
          setError("无法连接后端，请确认 langgraph dev 已在 2024 端口启动");
          return;
        }
      }
      const targetThread = threadId;

      const userMsg: ChatMessage = { id: nextId(), role: "user", content: question };
      const aiMsg: ChatMessage = { id: nextId(), role: "assistant", content: "" };
      setMessages((prev) => [...prev, userMsg, aiMsg]);
      setStreaming(true);
      setError(null);
      streamingIdRef.current = aiMsg.id;

      const patchAi = (patch: Partial<ChatMessage>) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === aiMsg.id ? { ...m, ...patch } : m)),
        );

      await streamAsk(targetThread, question, {
        onToken: (delta) =>
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiMsg.id ? { ...m, content: m.content + delta } : m,
            ),
          ),
        onDone: ({ answer, sources }) => {
          patchAi({ content: answer, sources: sources.length ? sources : undefined });
        },
        onError: (message) => {
          patchAi({ content: `⚠️ 出错了：${message}` });
          setError(message);
        },
      });

      setStreaming(false);
      streamingIdRef.current = null;

      // 刚创建的会话，用首条问题作为标题，再刷新侧栏顺序
      if (createdNow) {
        await renameThreadBestEffort(
          targetThread,
          question.slice(0, 16) || "未命名便签",
        );
      }
      await refreshThreads();
    },
    [currentId, refreshThreads, streaming],
  );

  return {
    threads,
    currentId,
    messages,
    streaming,
    loading,
    error,
    newThread,
    removeThread,
    renameThread,
    selectThread,
    send,
  };
}
