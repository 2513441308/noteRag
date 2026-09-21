> ## Documentation Index
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# AI Elements

> Composable shadcn/ui-based components for AI chat interfaces with useStream

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

[AI Elements](https://elements.ai-sdk.dev/) is a composable, shadcn/ui-based component library purpose-built for AI chat interfaces. Components like `Conversation`, `Message`, `Tool`, `Reasoning`, and `PromptInput` are designed to drop directly into any React project and wire to `stream.messages` with minimal glue code.

<ExampleEmbed example="ai-elements" minHeight={700} />

<Tip>
  Clone and run the [full AI Elements example](https://github.com/langchain-ai/langgraphjs/tree/main/examples/ai-elements) to see tool call rendering, reasoning display, streaming messages, and more in a working project.
</Tip>

## How it works

1. **Install components as source files:** AI Elements ships via a CLI that adds components directly to your project (shadcn/ui registry style)
2. **Map messages to components:** iterate `stream.messages`, render `HumanMessage` instances as user bubbles and `AIMessage` instances as assistant responses
3. **Compose richer UIs:** wrap tool calls in `<Tool>`, reasoning in `<Reasoning>`, and everything in `<Conversation>` for scroll management

## Installation

Install AI Elements components via the CLI. They're added as editable source files into your project:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
npm install @langchain/react
npx ai-elements@latest add conversation message prompt-input tool reasoning suggestion
```

## Wiring useStream

Render AI Elements components directly from `stream.messages`. Each LangChain `BaseMessage` maps to a component:

```tsx theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { useStream } from "@langchain/react";
import { HumanMessage, AIMessage } from "langchain";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  Tool,
  ToolHeader,
  ToolContent,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import {
  Reasoning,
  ReasoningTrigger,
  ReasoningContent,
} from "@/components/ai-elements/reasoning";
import {
  PromptInput,
  PromptInputBody,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputSubmit,
} from "@/components/ai-elements/prompt-input";

function getReasoningText(msg: AIMessage) {
  return msg.contentBlocks.find((block) => block.type === "reasoning")?.reasoning ?? "";
}

function getTextContent(msg: AIMessage) {
  return msg.text;
}

function getToolCalls(msg: AIMessage) {
  return (msg.tool_calls ?? []).map((tc) => ({
    id: tc.id,
    name: tc.name,
    args: tc.args,
    state: "input-available" as const,
  }));
}

export function Chat() {
  const stream = useStream({
    apiUrl: "http://localhost:2024",
    assistantId: "ai_elements",
  });

  return (
    <div className="flex flex-col h-dvh">
      <Conversation className="flex-1">
        <ConversationContent>
          {stream.messages.map((msg, i) => {
            if (HumanMessage.isInstance(msg)) {
              return (
                <Message key={i} from="user">
                  <MessageContent>{msg.text}</MessageContent>
                </Message>
              );
            }
            if (AIMessage.isInstance(msg)) {
              return (
                <div key={i}>
                  {/* Reasoning block (shows when model emits thinking tokens) */}
                  <Reasoning>
                    <ReasoningTrigger />
                    <ReasoningContent>{getReasoningText(msg)}</ReasoningContent>
                  </Reasoning>

                  {/* Inline tool calls with input/output display */}
                  {getToolCalls(msg).map((tc) => (
                    <Tool key={tc.id} defaultOpen>
                      <ToolHeader type={`tool-${tc.name}`} state={tc.state} />
                      <ToolContent>
                        <ToolInput input={tc.args} />
                        {tc.output && (
                          <ToolOutput output={tc.output} errorText={undefined} />
                        )}
                      </ToolContent>
                    </Tool>
                  ))}

                  {/* Streamed text response */}
                  <Message from="assistant">
                    <MessageContent>
                      <MessageResponse>{getTextContent(msg)}</MessageResponse>
                    </MessageContent>
                  </Message>
                </div>
              );
            }
          })}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <PromptInput
        onSubmit={({ text }) =>
          stream.submit({ messages: [{ type: "human", content: text }] })
        }
      >
        <PromptInputBody>
          <PromptInputTextarea placeholder="Ask me something..." />
        </PromptInputBody>
        <PromptInputFooter>
          <PromptInputSubmit
            status={stream.isLoading ? "streaming" : "ready"}
          />
        </PromptInputFooter>
      </PromptInput>
    </div>
  );
}
```

## Best practices

* **Edit source files freely:** components ship in your project, not as an external package dependency, so you can change anything without forking
* **Use `MessageResponse` for streaming:** it handles streamed partial tokens correctly; avoid rendering raw message content directly during streaming
* **Wrap in `Conversation`:** the `Conversation` component manages scroll behaviour so new messages auto-scroll into view
* **Gate on `isInstance`:** use `HumanMessage.isInstance(msg)` and `AIMessage.isInstance(msg)` rather than checking `msg.getType()` for proper TypeScript narrowing

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/frontend/integrations/ai-elements.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
