import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { createLowlight } from "lowlight";
// rehype-highlight 默认注册 highlight.js 全部约 190 种语言（gzip 后约 200KB），
// 这里只注册 RAG 对话中实际会出现的常用语言，显著减小产物体积。
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import plaintext from "highlight.js/lib/languages/plaintext";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";
import type { ChatMessage } from "../api/client";
import { SourcesCard } from "./SourcesCard";

const lowlight = createLowlight();
lowlight.register({
  bash,
  css,
  javascript,
  json,
  markdown,
  plaintext,
  python,
  typescript,
  xml,
  yaml,
});

/** 按索引取一个稳定的小倾角，让便签像手贴的一样错落 */
const TILTS = [-2, 1.4, -1.2, 2, -2.4, 1];

interface Props {
  message: ChatMessage;
  index: number;
  streaming: boolean;
}

export function MessageBubble({ message, index, streaming }: Props) {
  const isUser = message.role === "user";
  const tilt = TILTS[index % TILTS.length];
  const isTyping =
    !isUser && streaming && message.content.length === 0;

  return (
    <div
      className={`flex w-full note-drop ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`note note-hover relative max-w-[82%] ${
          isUser ? "note-yellow" : "note-mint"
        }`}
        style={{ transform: `rotate(${tilt}deg)` }}
      >
        <span className="tape" aria-hidden />
        <div
          className={`mb-1 flex items-center gap-1.5 text-xs font-bold text-ink-soft ${
            isUser ? "justify-end" : "justify-start"
          }`}
        >
          <span className="pin !h-2.5 !w-2.5" aria-hidden />
          {isUser ? "我" : "问答助手"}
        </div>
        {isUser ? (
          <div className="note-body whitespace-pre-wrap">{message.content}</div>
        ) : (
          <div
            className={`note-body ${streaming && message.content ? "typing-caret" : ""}`}
          >
            {isTyping ? (
              <span className="text-ink-soft">正在便签上翻资料…</span>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[
                  [rehypeHighlight, { lowlight, detect: false }],
                ]}
              >
                {message.content}
              </ReactMarkdown>
            )}
          </div>
        )}
        {!isUser && message.sources && message.sources.length > 0 && (
          <SourcesCard sources={message.sources} />
        )}
      </div>
    </div>
  );
}
