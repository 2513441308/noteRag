import { useEffect, useRef } from "react";
import type { ChatMessage } from "../api/client";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
}

export function ChatWindow({ messages, streaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  // 流式中的一定是最后一张 AI 便签
  let lastAssistantId: string | null = null;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      lastAssistantId = messages[i].id;
      break;
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center px-6">
        <div
          className="note note-mint relative max-w-md px-8 py-7 text-center shadow-note"
          style={{ transform: "rotate(-1.5deg)" }}
        >
          <span className="tape" aria-hidden />
          <div className="font-hand text-3xl text-ink">欢迎来到便签问答板 📝</div>
          <p className="mt-3 text-sm leading-6 text-ink-soft">
            在下方便签写下任何关于{" "}
            <strong className="text-ink">LangGraph / 本项目知识库</strong>{" "}
            的问题，
            助手会翻资料、把答案写在一张薄荷绿便签上，并钉好引用来源。
            闲聊也欢迎，比如先问声好～
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-8 sm:px-8">
      <div className="mx-auto flex max-w-3xl flex-col gap-7">
        {messages.map((m, i) => (
          <MessageBubble
            key={m.id}
            message={m}
            index={i}
            streaming={streaming && m.id === lastAssistantId}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
