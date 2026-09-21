> ## Documentation Index
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# assistant-ui

> Headless React AI chat framework with a full runtime layer, bridged to useStream

export const ExampleEmbed = ({example, theme, minHeight = 500, maxHeight = 700}) => {
  var PROD_BASE = "https://ui-patterns.langchain.com";
  var iframeCache = (() => {
    const g = globalThis;
    if (!g.__lcExampleIframeCache) {
      g.__lcExampleIframeCache = new Map();
    }
    return g.__lcExampleIframeCache;
  })();
  function detectPageTheme() {
    if (typeof document === "undefined") return "light";
    const root = document.documentElement;
    if (root.classList.contains("dark") || root.getAttribute("data-theme") === "dark" || root.style.colorScheme === "dark") {
      return "dark";
    }
    return "light";
  }
  var LOCAL_BASE = "http://localhost";
  var LOCAL_PORTS = {
    "ai-elements": 4600,
    "assistant-ui": 4500
  };
  function isLocalhost() {
    return typeof location !== "undefined" && (location.hostname === "localhost" || location.hostname === "127.0.0.1");
  }
  var EMBED_CSS = `
[data-lc-ee] .lc-border{border-color:#B8DFFF}
[data-lc-ee].dark .lc-border{border-color:#1A2740}
[data-lc-ee] .lc-bg-surface{background-color:white}
[data-lc-ee].dark .lc-bg-surface{background-color:#0B1120}
[data-lc-ee] .lc-bg-wash{background-color:#F2FAFF}
[data-lc-ee].dark .lc-bg-wash{background-color:#030710}
[data-lc-ee] .lc-spinner{border-color:#B8DFFF;border-top-color:#7FC8FF}
[data-lc-ee].dark .lc-spinner{border-color:#1A2740;border-top-color:#7FC8FF}
`;
  const slotRef = useRef(null);
  const [ready, setReady] = useState(() => Boolean(iframeCache.get(example)?.iframe));
  const [iframeHeight, setIframeHeight] = useState(minHeight);
  const [pageTheme, setPageTheme] = useState(detectPageTheme);
  const effectiveTheme = theme ?? pageTheme;
  const effectiveThemeRef = useRef(effectiveTheme);
  effectiveThemeRef.current = effectiveTheme;
  useEffect(() => {
    if (document.getElementById("lc-ee-css")) return;
    const style = document.createElement("style");
    style.id = "lc-ee-css";
    style.textContent = EMBED_CSS;
    document.head.appendChild(style);
  }, []);
  useEffect(() => {
    setPageTheme(detectPageTheme());
    const observer = new MutationObserver(() => setPageTheme(detectPageTheme()));
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class", "data-theme", "style"]
    });
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const useLocal = isLocalhost();
    const localPort = LOCAL_PORTS[example];
    const src = useLocal && localPort ? `${LOCAL_BASE}:${localPort}/` : `${PROD_BASE}/${example}/`;
    let cached = iframeCache.get(example);
    if (cached?.hideTimer) {
      clearTimeout(cached.hideTimer);
      cached.hideTimer = void 0;
    }
    if (!cached) {
      const iframe = document.createElement("iframe");
      iframe.src = src;
      iframe.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms");
      iframe.setAttribute("allow", "clipboard-write");
      iframe.title = `${example} example`;
      Object.assign(iframe.style, {
        position: "fixed",
        border: "none",
        visibility: "hidden",
        pointerEvents: "auto",
        zIndex: "1",
        borderRadius: "15px"
      });
      document.body.appendChild(iframe);
      cached = {
        iframe
      };
      iframeCache.set(example, cached);
      window.addEventListener("message", e => {
        if (e.data?.type === "RESIZE" && iframeCache.get(example)?.iframe === iframe) {
          const h = Math.min(maxHeight, Math.max(minHeight, e.data.height));
          setIframeHeight(h);
        }
      });
      iframe.addEventListener("load", () => {
        iframe.style.visibility = "visible";
        setReady(true);
        try {
          iframe.contentWindow?.postMessage({
            type: "CHAT_LC_SET_THEME",
            theme: effectiveThemeRef.current
          }, "*");
        } catch {}
      });
    } else {
      cached.iframe.style.visibility = "visible";
      setReady(true);
    }
    function syncPosition() {
      const slot = slotRef.current;
      if (!slot) return;
      const rect = slot.getBoundingClientRect();
      const {style} = cached.iframe;
      style.top = `${rect.top}px`;
      style.left = `${rect.left}px`;
      style.width = `${rect.width}px`;
      style.setProperty("height", `${rect.height}px`, "important");
    }
    syncPosition();
    const ro = new ResizeObserver(syncPosition);
    if (slotRef.current) ro.observe(slotRef.current);
    document.addEventListener("scroll", syncPosition, {
      passive: true,
      capture: true
    });
    window.addEventListener("resize", syncPosition, {
      passive: true
    });
    let frameCount = 0;
    let rafId = 0;
    function initialSync() {
      syncPosition();
      if (++frameCount < 5) rafId = requestAnimationFrame(initialSync);
    }
    rafId = requestAnimationFrame(initialSync);
    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
      document.removeEventListener("scroll", syncPosition, {
        capture: true
      });
      window.removeEventListener("resize", syncPosition);
      cached.hideTimer = setTimeout(() => {
        if (cached?.iframe) cached.iframe.style.visibility = "hidden";
      }, 200);
    };
  }, [example, minHeight, maxHeight]);
  useEffect(() => {
    const cached = iframeCache.get(example);
    if (!cached?.iframe || !ready) return;
    try {
      cached.iframe.contentWindow?.postMessage({
        type: "CHAT_LC_SET_THEME",
        theme: effectiveTheme
      }, "*");
    } catch {}
  }, [effectiveTheme, ready, example]);
  return <div data-lc-ee="" className={effectiveTheme === "dark" ? "dark" : ""} style={{
    position: "relative",
    fontFamily: "inherit"
  }}>
      <div className="lc-border lc-bg-surface" style={{
    border: "1px solid",
    borderRadius: "16px",
    overflow: "hidden"
  }}>
        {}
        <div ref={slotRef} className="lc-bg-wash" style={{
    height: iframeHeight,
    position: "relative"
  }}>
          {!ready && <div style={{
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center"
  }}>
              <div className="lc-spinner" style={{
    width: 24,
    height: 24,
    border: "3px solid",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite"
  }} />
              <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
            </div>}
        </div>
      </div>
    </div>;
};

[assistant-ui](https://www.assistant-ui.com/) is a headless React UI framework for AI chat. It provides a full runtime layer—thread management, message branching, attachment handling—that connects to [`useStream`](https://reference.langchain.com/javascript/langchain-react/index/useStream) via the `useExternalStoreRuntime` adapter.

<ExampleEmbed example="assistant-ui" minHeight={700} />

<Tip>
  Clone and run the [full assistant-ui example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/assistant-ui-claude) to see a Claude-style chat interface wired to a LangChain agent with `useExternalStoreRuntime`.
</Tip>

## How it works

1. **Stream with [`useStream`](https://reference.langchain.com/javascript/langchain-react/index/useStream)**: connect to your agent and get reactive messages, loading state, and submit/cancel callbacks
2. **Adapt with `useExternalStoreRuntime`**: bridge `stream.messages` into assistant-ui's runtime format by converting `BaseMessage[]` to `ThreadMessageLike[]`
3. **Provide the runtime**: wrap your UI in `AssistantRuntimeProvider` and render any assistant-ui thread component

## Installation

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
bun add @assistant-ui/react @assistant-ui/react-markdown
```

## Wiring `useStream`

The `useExternalStoreRuntime` adapter bridges `stream.messages` into the assistant-ui runtime. Pass it to `AssistantRuntimeProvider` and render any thread component:

```tsx theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { useCallback, useMemo } from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { useStream } from "@langchain/react";
import { Thread } from "@assistant-ui/react";

export function Chat() {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "claude",
  });

  const onNew = useCallback(
    async (message: AppendMessage) => {
      const text = message.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("");
      await stream.submit({ messages: [{ type: "human", content: text }] });
    },
    [stream],
  );

  // Convert LangChain messages to assistant-ui's ThreadMessageLike format
  const messages = useMemo(
    () => toThreadMessages(stream.messages),
    [stream.messages],
  );

  const runtime = useExternalStoreRuntime<ThreadMessageLike>({
    messages,
    onNew,
    onCancel: () => stream.stop(),
    convertMessage: (m) => m,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  );
}
```

### Converting messages

`toThreadMessages` maps LangChain `BaseMessage[]` to the `ThreadMessageLike[]` format assistant-ui expects. Handle each message type — human, AI, and tool — and convert content blocks, tool calls, and reasoning tokens:

```tsx expandable theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from "langchain";
import type { ThreadMessageLike } from "@assistant-ui/react";

export function toThreadMessages(messages: BaseMessage[]): ThreadMessageLike[] {
  const result: ThreadMessageLike[] = [];

  for (const msg of messages) {
    if (HumanMessage.isInstance(msg)) {
      result.push({
        role: "user",
        content: [{ type: "text", text: msg.text }],
      });
    } else if (AIMessage.isInstance(msg)) {
      const parts: ThreadMessageLike["content"] = [];

      // Reasoning tokens
      const reasoning = msg.contentBlocks.find((block) => block.type === "reasoning")?.reasoning;
      if (reasoning) parts.push({ type: "reasoning", text: reasoning });

      // Tool calls
      for (const tc of msg.tool_calls ?? []) {
        parts.push({
          type: "tool-call",
          toolCallId: tc.id ?? "",
          toolName: tc.name,
          args: tc.args,
        });
      }

      // Text response
      const text = msg.text;
      if (text) parts.push({ type: "text", text });

      result.push({ role: "assistant", content: parts });
    } else if (ToolMessage.isInstance(msg)) {
      // Attach tool results to the preceding assistant message
      const last = result[result.length - 1];
      if (last?.role === "assistant") {
        for (const part of last.content) {
          if (
            part.type === "tool-call" &&
            part.toolCallId === msg.tool_call_id
          ) {
            (part as { result?: string }).result = msg.text;
          }
        }
      }
    }
  }

  return result;
}
```

## Customizing the thread UI

`<Thread />` ships a complete default thread UI including message list, composer, and scroll management. Customize individual parts by overriding component slots:

```tsx theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { Thread, ThreadMessages, Composer } from "@assistant-ui/react";

function CustomThread() {
  return (
    <Thread.Root>
      <ThreadMessages
        components={{
          UserMessage: MyUserMessage,
          AssistantMessage: MyAssistantMessage,
          ToolFallback: MyToolCard,
        }}
      />
      <Composer />
    </Thread.Root>
  );
}
```

## Best practices

* **Memoise message conversion:** wrap `toThreadMessages(stream.messages)` in `useMemo` to avoid re-running the conversion on every render
* **Handle attachments:** use `CompositeAttachmentAdapter` with `SimpleImageAttachmentAdapter` for image uploads; extend with custom adapters for files
* **Use branching:** assistant-ui has built-in message branching support via `MessageBranch`; pair edits with `useMessageMetadata` and `forkFrom` when you need LangGraph checkpoint forks
* **Thread persistence:** persist `threadId` with `onThreadId` and pass it back into [`useStream`](https://reference.langchain.com/javascript/langchain-react/index/useStream) on page load so assistant-ui reconnects to the same thread

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/frontend/integrations/assistant-ui.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
