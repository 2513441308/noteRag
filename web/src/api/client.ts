import { Client } from "@langchain/langgraph-sdk";
import { resolveBaseUrl } from "./http";

/**
 * 后端地址。
 * 必须是绝对地址：langgraph-sdk 内部用 `new URL(path)` 单参数解析，
 * 传 "/lg" 这种相对路径会直接抛 TypeError: Invalid URL，请求根本发不出去。
 * - 默认取当前页面 origin + /lg，经 Vite 同源代理转发到 2024，
 *   IDE 端口转发/预览换成任意域名都能自适应；
 * - 直连其他后端时用 .env 的 VITE_API_URL 覆盖。
 */
export const API_URL: string = resolveBaseUrl(
  import.meta.env.VITE_API_URL as string | undefined,
  "/lg",
);

/** langgraph.json 中注册的图 id */
export const ASSISTANT_ID = "agent";

export const client = new Client({ apiUrl: API_URL });

/** 能产出「最终回答」的节点：RAG 走 generate，闲聊走 common，
 *  route / searchrebuild 等节点也会吐 token，必须过滤掉 */
const ANSWER_NODES = new Set(["generate", "common"]);

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export interface ThreadSummary {
  thread_id: string;
  title: string;
  updated_at?: string;
}

interface StateMessage {
  type?: string;
  content?: unknown;
}

/** 把 SDK 返回的消息对象内容统一成字符串（兼容多模态 content 数组） */
function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) =>
        typeof part === "string"
          ? part
          : (part as { text?: string })?.text ?? "",
      )
      .join("");
  }
  return "";
}

/** 列出全部会话，按更新时间倒序 */
export async function listThreads(): Promise<ThreadSummary[]> {
  const threads = await client.threads.search({ limit: 50 });
  return threads
    .map((t) => ({
      thread_id: t.thread_id,
      title:
        (t.metadata as Record<string, unknown> | null | undefined)?.title as
          | string
          | undefined
          ? String(
              (t.metadata as Record<string, unknown>).title,
            )
          : "未命名便签",
      updated_at: t.updated_at,
    }))
    .sort(
      (a, b) =>
        new Date(b.updated_at ?? 0).getTime() -
        new Date(a.updated_at ?? 0).getTime(),
    );
}

/** 新建会话 */
export async function createThread(): Promise<ThreadSummary> {
  const t = await client.threads.create({});
  return { thread_id: t.thread_id, title: "未命名便签", updated_at: t.updated_at };
}

/** 更新会话标题（手动重命名），失败抛错供 UI 提示 */
export async function renameThread(
  threadId: string,
  title: string,
): Promise<void> {
  await client.threads.update(threadId, { metadata: { title } });
}

/** 更新会话标题（用首条问题命名，失败不影响主流程） */
export async function renameThreadBestEffort(
  threadId: string,
  title: string,
): Promise<void> {
  try {
    await renameThread(threadId, title);
  } catch {
    /* 标题仅用于侧栏展示，忽略 */
  }
}

/** 删除会话（同时清掉后端检查点中的历史） */
export async function deleteThread(threadId: string): Promise<void> {
  await client.threads.delete(threadId);
}

/** 从 checkpointer 状态恢复一个会话的可见问答历史（跳过工具消息） */
export async function loadThreadMessages(
  threadId: string,
): Promise<ChatMessage[]> {
  const state = await client.threads.getState(threadId);
  const raw = (state.values as { messages?: StateMessage[] })?.messages ?? [];
  const messages: ChatMessage[] = [];
  for (const m of raw) {
    if (m.type !== "human" && m.type !== "ai") continue;
    const content = contentToText(m.content).trim();
    if (!content) continue;
    messages.push({
      id: `${m.type}-${messages.length}`,
      role: m.type === "human" ? "user" : "assistant",
      content,
    });
  }
  // 最后一条 AI 消息补 sources（若有）
  const sources = (state.values as { sources?: unknown })?.sources;
  if (Array.isArray(sources) && messages.length > 0) {
    const lastAi = [...messages].reverse().find((m) => m.role === "assistant");
    if (lastAi) lastAi.sources = sources.map(String);
  }
  return messages;
}

export interface StreamHandlers {
  /** 收到回答 token（仅 generate/common 节点） */
  onToken: (delta: string) => void;
  /** 整轮结束，拿到最终答案与引用来源 */
  onDone: (final: { answer: string; sources: string[] }) => void;
  onError: (message: string) => void;
}

/** 发送问题并消费 SSE：边收 token 边回调，结束用最终 state 定稿 */
export async function streamAsk(
  threadId: string,
  question: string,
  handlers: StreamHandlers,
): Promise<void> {
  let streamed = "";
  // 注意：answer/sources 会跨轮残留在 state 中，节点中途的 values 事件
  // 可能仍是上一轮旧答案，因此只记录、不提前定稿，流结束时最后一个为准。
  let latestFinal: { answer: string; sources: string[] } | null = null;
  try {
    const stream = client.runs.stream(threadId, ASSISTANT_ID, {
      input: { question },
      streamMode: ["messages-tuple", "values"],
    });

    for await (const part of stream) {
      if (part.event === "messages") {
        const [chunk, metadata] = part.data as [
          { content?: unknown; type?: string },
          { langgraph_node?: string } & Record<string, unknown>,
        ];
        const node = metadata?.langgraph_node;
        // messages 流有两类内容：
        // 1. 模型实时吐出的 AIMessageChunk 分片（要逐字展示）；
        // 2. 节点把完整 human/ai 消息写入 state 时整段重放的消息
        //    （type 为 "human"/"ai"），拼进去会把问题和答案各重复一遍。
        if (chunk.type === "AIMessageChunk" && node && ANSWER_NODES.has(node)) {
          const delta = contentToText(chunk.content);
          if (delta) {
            streamed += delta;
            handlers.onToken(delta);
          }
        }
      } else if (part.event === "values") {
        const values = part.data as {
          answer?: unknown;
          sources?: unknown;
        };
        if (typeof values.answer === "string") {
          latestFinal = {
            answer: values.answer,
            sources: Array.isArray(values.sources)
              ? values.sources.map(String)
              : [],
          };
        }
      } else if (part.event === "error") {
        const data = part.data as { message?: string };
        throw new Error(data?.message ?? "服务器返回错误事件");
      }
    }

    if (latestFinal) {
      handlers.onDone(latestFinal);
    } else if (streamed) {
      handlers.onDone({ answer: streamed, sources: [] });
    } else {
      // 兜底：SSE 未带出最终状态时主动拉一次线程状态
      const state = await client.threads.getState(threadId);
      const values = state.values as { answer?: string; sources?: unknown };
      if (values.answer) {
        handlers.onDone({
          answer: values.answer,
          sources: Array.isArray(values.sources)
            ? values.sources.map(String)
            : [],
        });
      } else {
        handlers.onError("未获取到回答");
      }
    }
  } catch (err) {
    console.error("[streamAsk] 请求失败：", err);
    handlers.onError(err instanceof Error ? err.message : String(err));
  }
}
