> ## Documentation Index
> Fetch the complete documentation index at: https://docs.langchain.com/llms.txt
> Use this file to discover all available pages before exploring further.

# CopilotKit

> Use CopilotKit with LangGraph, Deep Agents, and React with custom endpoints, the Python AG-UI bridge, structured generative UI, and messaging-platform channels

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

[CopilotKit](https://www.copilotkit.ai/) provides a full React chat runtime and pairs especially well with LangGraph when you want the agent to return **structured UI payloads** instead of only plain text. In this pattern, your LangGraph deployment serves both the graph API and a custom CopilotKit endpoint, while the frontend parses assistant messages into dynamic React components.

On the server, the [copilotkit](https://pypi.org/project/copilotkit/) package provides [`CopilotKitMiddleware`](https://docs.copilotkit.ai) so a LangGraph graph, a LangChain agent, or a [Deep Agent](/oss/python/deepagents/overview) can speak the [Agent UI (AG-UI)](https://docs.ag-ui.com/) wire protocol, stream tool and message events to a chat UI, and read or write the shared **CopilotKit** slice of state, with helpers to mount a CopilotKit-compatible HTTP endpoint in front of your graph.

This approach is useful when you want:

* a ready-made chat runtime instead of wiring `stream.messages` yourself
* a custom server endpoint that can add provider-specific behavior next to your deployed graph
* structured generative UI rendered from a constrained component registry

[CopilotKit for LangGraph](https://docs.copilotkit.ai/langgraph) also documents [generative UI](https://docs.copilotkit.ai/langgraph/generative-ui), [human in the loop](https://docs.copilotkit.ai/langgraph/human-in-the-loop) (HITL), and [shared state](https://docs.copilotkit.ai/langgraph/shared-state) on top of the same middleware and clients.

<Info>
  For CopilotKit-specific APIs, UI patterns, and runtime configuration, see the
  [CopilotKit docs](https://docs.copilotkit.ai/langgraph). For a Deep Agent walkthrough, see
  [Deep Agents and CopilotKit](https://docs.copilotkit.ai/langgraph/deep-agents) in the CopilotKit docs.
</Info>

<ExampleEmbed example="copilotkit" minHeight={700} />

## How it works

At a high level, CopilotKit sits between your React app and the LangGraph deployment. The frontend sends conversation state to a custom `/api/copilotkit` route mounted alongside the graph API, that route forwards the request to LangGraph, and the response comes back with both assistant messages and any structured UI payloads your component registry can render.

1. **Deploy the graph as usual** using LangSmith or using a LangGraph development server.
2. **Extend the deployment with an HTTP app** that mounts a CopilotKit route next to the graph API.
3. **Wrap the frontend in `CopilotKit`** and point it at that custom runtime URL.
4. **Register dynamic UI components** and parse assistant responses into those components at render time.

```mermaid theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
%%{
  init: {
    "fontFamily": "monospace",
    "flowchart": {
      "curve": "curve"
    }
  }
}%%
graph LR
  USER["User input"]
  UI["CopilotKit React app"]
  ENDPOINT["/api/copilotkit"]
  GRAPH["LangGraph deployment"]
  RENDER["Hashbrown UI kit"]

  USER --> UI
  UI --> ENDPOINT
  ENDPOINT --> GRAPH
  GRAPH --> ENDPOINT
  ENDPOINT --> UI
  UI --> RENDER
```

## What you get on the Python server

The [copilotkit](https://pypi.org/project/copilotkit/) and related packages bridge a LangGraph deployment and CopilotKit clients.

| Component                                                                                            | Role                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CopilotKitMiddleware`                                                                               | Merges CopilotKit and AG-UI state and requests into your agent, including frontend [tool calls](/oss/python/langchain/agents#tools) and context. Add it to the `middleware` list for [create\_agent](https://reference.langchain.com/python/langchain/agents/factory/create_agent) or [create\_deep\_agent](https://reference.langchain.com/python/deepagents/graph/create_deep_agent). |
| `CopilotKitState` (subclass)                                                                         | [Custom state](/oss/python/langchain/short-term-memory): extend `CopilotKitState` so the CopilotKit key is part of graph state.                                                                                                                                                                                                                                                         |
| `LangGraphAGUIAgent`                                                                                 | Bundles a compiled graph with a name and description for the runtime.                                                                                                                                                                                                                                                                                                                   |
| `add_langgraph_fastapi_endpoint` (from [ag-ui-langgraph](https://pypi.org/project/ag-ui-langgraph/)) | Wires a **FastAPI** app so CopilotKit can run your graph on the same [LangGraph](/oss/python/langgraph/overview) process. Use it when you add a [custom `http` app in `langgraph.json`](#extend-the-langgraph-deployment-with-a-custom-endpoint) instead of a separate HTTP server.                                                                                                     |

`CopilotKitMiddleware` is the same middleware for [create\_deep\_agent](https://reference.langchain.com/python/deepagents/graph/create_deep_agent) and for a graph from [create\_agent](https://reference.langchain.com/python/langchain/agents/factory/create_agent) when you add it to the `middleware` list. For a `create_agent` graph with `CopilotKitState` and a FastAPI bridge, follow the [Python `main.py` example](#extend-the-langgraph-deployment-with-a-custom-endpoint) below. Structured generative UI (for example `useAgentContext` and an `output_schema` from the client) needs extra middleware that maps Copilot state to a [structured output](/oss/python/langchain/agents#structured-output) strategy, as in the expandable `src/middleware.py` example in the same section.

Mounting `app` on the `http` key in `langgraph.json` follows the usual [LangGraph or LangSmith deployment](/oss/python/langgraph/deploy) so one process serves the graph and the same FastAPI app to the CopilotKit client.

## Installation

For the backend endpoint:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
uv add copilotkit ag-ui-langgraph fastapi uvicorn
```

The middleware package sits alongside the Deep Agents stack. Install it with your [chat model](/oss/python/integrations/chat) package (this example uses OpenAI):

<CodeGroup>
  ```python pip theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  pip install -U deepagents copilotkit langchain-openai
  ```

  ```python uv theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  uv add deepagents copilotkit langchain-openai
  ```
</CodeGroup>

For the frontend app:

```bash theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
bun add @copilotkit/react-core @copilotkit/react-ui @hashbrownai/core @hashbrownai/react
```

## Use CopilotKit with a Deep Agent

Add `CopilotKitMiddleware` to the `middleware` list you pass to [create\_deep\_agent](https://reference.langchain.com/python/deepagents/graph/create_deep_agent). The middleware lets CopilotKit route frontend tool calls and align chat state with your graph. Keep any other [middleware you configure](/oss/python/deepagents/customization#middleware) in the same list.

The compiled graph is then ready to plug into a CopilotKit- or AG-UI–aware process (for example, the [FastAPI pattern below](#extend-the-langgraph-deployment-with-a-custom-endpoint)) or a guide such as [Deep Agents and CopilotKit](https://docs.copilotkit.ai/langgraph/deep-agents) in the CopilotKit documentation.

```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from deepagents import create_deep_agent
from copilotkit import CopilotKitMiddleware
from langgraph.checkpoint.memory import MemorySaver


def get_weather(location: str) -> str:
    """Return a simple weather string for a location."""
    return f"The weather in {location} is sunny."


agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    middleware=[CopilotKitMiddleware()],  # AG-UI, frontend tools, and context
    system_prompt="You are a helpful research assistant.",
    checkpointer=MemorySaver(),
)
```

## Extend the LangGraph deployment with a custom endpoint

The key idea is that the LangGraph deployment does not only serve graphs. It can also load an HTTP app, which lets you mount extra routes next to the deployment itself.

In `langgraph.json`, point `http.app` at your custom app entrypoint:

```json theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
{
  "dependencies": ["."],
  "graphs": {
    "copilotkit_shadify": "./main.py:agent"
  },
  "http": {
    "app": "./main.py:app"
  }
}
```

In Python, create a `FastAPI` app and expose the LangGraph agent through CopilotKit's AG-UI bridge:

```python main.py theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
from typing import Any, TypedDict

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import CopilotKitMiddleware, CopilotKitState, LangGraphAGUIAgent
from fastapi import FastAPI
from langchain.agents import create_agent

from src.middleware import apply_structured_output_schema, normalize_context


class AgentState(CopilotKitState):
    pass


class AgentContext(TypedDict, total=False):
    output_schema: dict[str, Any]


agent = create_agent(
    model="openai:gpt-5.5",
    middleware=[
        normalize_context,
        CopilotKitMiddleware(),
        apply_structured_output_schema,
    ],
    context_schema=AgentContext,
    state_schema=AgentState,
    system_prompt=(
        "You are a helpful UI assistant. Build visual responses using the "
        "available components."
    ),
)

app = FastAPI()

add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAGUIAgent(
        name="copilotkit_shadify",
        description="A UI assistant that returns structured component payloads.",
        graph=agent,
    ),
    path="/",
)
```

This custom app is the important extension point: it mounts a CopilotKit-aware runtime without replacing the underlying LangGraph deployment.

In Python, the equivalent work happens in middleware: normalize the CopilotKit context and forward the `output_schema` from `useAgentContext(...)` into the model's structured output configuration.

```python expandable src/middleware.py theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import json
from collections.abc import Mapping

from langchain.agents.middleware import before_agent, wrap_model_call
from langchain.agents.structured_output import ProviderStrategy


@wrap_model_call
async def apply_structured_output_schema(request, handler):
    schema = None
    runtime = getattr(request, "runtime", None)
    runtime_context = getattr(runtime, "context", None)

    if isinstance(runtime_context, Mapping):
        schema = runtime_context.get("output_schema")

    if schema is None and isinstance(getattr(request, "state", None), dict):
        copilot_context = request.state.get("copilotkit", {}).get("context")
        if isinstance(copilot_context, list):
            for item in copilot_context:
                if isinstance(item, dict) and item.get("description") == "output_schema":
                    schema = item.get("value")
                    break

    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            schema = None

    if isinstance(schema, dict):
        request = request.override(
            response_format=ProviderStrategy(schema=schema, strict=True),
        )

    return await handler(request)


@before_agent
def normalize_context(state, runtime):
    copilotkit_state = state.get("copilotkit", {})
    context = copilotkit_state.get("context")

    if isinstance(context, list):
        normalized = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in context
        ]
        return {"copilotkit": {**copilotkit_state, "context": normalized}}

    return None
```

The result is a clean separation of concerns:

* LangGraph still owns graph execution and persistence
* CopilotKit owns the chat-facing runtime contract
* your custom endpoint glues them together inside one deployment

Point your CopilotKit `runtimeUrl` at the route the FastAPI (or other) app exposes, not only the raw graph REST surface, when you use the [CopilotKit](https://docs.copilotkit.ai) runtime adapter.

Follow the CopilotKit documentation for [LangGraphHttpAgent](https://docs.copilotkit.ai/langgraph) or `LangGraphAgent` in the Node **CopilotRuntime**; the **Python** graph and middleware still define tool behavior and agent logic.
:::

## Structure the frontend app

On the frontend, wrap your app in `CopilotKit` and point it at the custom runtime URL:

```tsx theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat, useAgentContext } from "@copilotkit/react-core/v2";
import { s } from "@hashbrownai/core";

import { useChatKit } from "@/components/chat/chat-kit";
import { chatTheme } from "@/lib/chat-theme";

export function App() {
  return (
    <CopilotKit runtimeUrl={import.meta.env.VITE_RUNTIME_URL ?? "/api/copilotkit"}>
      <Page />
    </CopilotKit>
  );
}

function Page() {
  const chatKit = useChatKit();

  useAgentContext({
    description: "output_schema",
    value: s.toJsonSchema(chatKit.schema),
  });

  return <CopilotChat {...chatTheme} />;
}
```

There are two important pieces here:

* `runtimeUrl="/api/copilotkit"` sends the chat to your custom backend route rather than directly to the raw LangGraph API
* `useAgentContext(...)` sends the UI schema to the agent so the model knows what structured output format it should produce

## Register the dynamic components

The component registry lives in `useChatKit()`. This is where you define the set of components the agent is allowed to emit, such as cards, rows, columns, charts, code blocks, and buttons.

```tsx theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { s } from "@hashbrownai/core";
import { exposeComponent, exposeMarkdown, useUiKit } from "@hashbrownai/react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { CodeBlock } from "@/components/ui/code-block";
import { Row, Column } from "@/components/ui/layout";
import { SimpleChart } from "@/components/ui/simple-chart";

export function useChatKit() {
  return useUiKit({
    components: [
      exposeMarkdown(),
      exposeComponent(Card, {
        name: "card",
        description: "Card to wrap generative UI content.",
        children: "any",
      }),
      exposeComponent(Row, {
        name: "row",
        props: {
          gap: s.string("Tailwind gap size") as never,
        },
        children: "any",
      }),
      exposeComponent(Column, {
        name: "column",
        children: "any",
      }),
      exposeComponent(SimpleChart, {
        name: "chart",
        props: {
          labels: s.array("Category labels", s.string("A label")),
          values: s.array("Numeric values", s.number("A value")),
        },
        children: false,
      }),
      exposeComponent(CodeBlock, {
        name: "code_block",
        props: {
          code: s.streaming.string("The code to display"),
          language: s.string("Programming language") as never,
        },
        children: false,
      }),
      exposeComponent(Button, {
        name: "button",
        children: "text",
      }),
    ],
  });
}
```

This registry becomes the contract between the agent and the UI. The model is not generating arbitrary JSX. It is generating structured data that must validate against the components and props you exposed.

## Render assistant messages as dynamic UI

Once the assistant response arrives, the custom message renderer decides how to display it. In this example:

* assistant messages are parsed as structured JSON against the UI kit schema
* valid structured output is rendered as real React components
* user messages are rendered as ordinary chat bubbles

```tsx theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import type { AssistantMessage } from "@ag-ui/core";
import type { RenderMessageProps } from "@copilotkit/react-ui";
import { useJsonParser } from "@hashbrownai/react";
import { memo } from "react";

import { useChatKit } from "@/components/chat/chat-kit";
import { Squircle } from "@/components/squircle";

const AssistantMessageRenderer = memo(function AssistantMessageRenderer({
  message,
}: {
  message: AssistantMessage;
}) {
  const kit = useChatKit();
  const { value } = useJsonParser(message.content ?? "", kit.schema);

  if (!value) return null;

  return (
    <div className="group/msg mt-2 flex w-full justify-start">
      <div className="magic-text-output w-full px-1 py-1">{kit.render(value)}</div>
    </div>
  );
});

export function CustomMessageRenderer({ message }: RenderMessageProps) {
  if (message.role === "assistant") {
    return <AssistantMessageRenderer message={message} />;
  }

  return (
    <div className="flex w-full justify-end">
      <Squircle className="w-full max-w-[64ch] px-4 py-3">
        <pre>{typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2)}</pre>
      </Squircle>
    </div>
  );
}
```

This renderer pattern is what makes the integration feel native:

* CopilotKit handles chat state and transport
* the custom renderer decides how assistant payloads become UI
* [Hashbrown](https://hashbrown.dev/) turns validated structured data into concrete React elements

## Channels

The same agent that powers your in-app copilot can also run as a bot in Slack and other messaging platforms. [CopilotKit Channels](https://docs.copilotkit.ai/channels) connects your LangChain agent to a messaging platform through a managed CopilotKit Intelligence connection: Intelligence holds the platform credentials and message delivery, while your process runs the agent.

<Note>
  Channels require `@copilotkit/channels` 0.6.1 and `@copilotkit/runtime` 1.65.0, installed together as a tested pair, and Node.js 22 or later on a long-running host. The LangChain deployment the channel connects to can be Python or TypeScript.
</Note>

<Info>
  This page shows how Channels fit together with a LangChain agent. For the full Slack walkthrough with a LangGraph backend, see [Connect and run your agent in Slack](https://docs.copilotkit.ai/slack/langgraph-typescript/connect). For other agent backends and platform coverage, see the [Channels overview](https://docs.copilotkit.ai/channels).
</Info>

### How it fits together

CopilotKit Intelligence owns the platform connection (Slack, Microsoft Teams, and more) and its credentials. Your long-running Channels listener owns the agent, tools, and application logic. Each turn follows the same path:

1. A person messages your app in Slack.
2. Intelligence receives the platform event using the credentials configured for the Channel.
3. A persistent Intelligence gateway connection delivers the turn to your Channels listener.
4. The listener runs your LangChain agent over [AG-UI](https://docs.ag-ui.com/) and renders the reply.
5. Intelligence sends the result back as native Slack Block Kit.

Platform credentials never enter the agent process. Create a managed Channel in [CopilotKit Intelligence](https://docs.copilotkit.ai/slack/intelligence) and connect Slack: Intelligence stores the Slack credentials and gives you a **Channel code** (`CHANNEL_CODE`) and a project-scoped **Intelligence API key** (`INTELLIGENCE_API_KEY`) for the listener.

### Install

Install the tested SDK pair, then the TypeScript tooling:

<CodeGroup>
  ```bash npm theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  npm install --save-exact @copilotkit/channels@0.6.1 @copilotkit/runtime@1.65.0
  npm install -D tsx typescript @types/node
  ```

  ```bash pnpm theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  pnpm add --save-exact @copilotkit/channels@0.6.1 @copilotkit/runtime@1.65.0
  pnpm add -D tsx typescript @types/node
  ```

  ```bash yarn theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
  yarn add --exact @copilotkit/channels@0.6.1 @copilotkit/runtime@1.65.0
  yarn add -D tsx typescript @types/node
  ```
</CodeGroup>

Use a NodeNext TypeScript configuration:

```json icon="settings" title="tsconfig.json" theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "skipLibCheck": true,
    "noEmit": true,
    "types": ["node"]
  },
  "include": ["*.ts", "*.tsx"]
}
```

### Connect your LangChain agent

Return a fresh agent for each conversation, keyed by thread, so no state leaks across Slack threads. LangGraph Server speaks the LangGraph API, so use `LangGraphAgent` from `@copilotkit/runtime/langgraph`, not `HttpAgent`. Point it at the [deployment you already run](#extend-the-langgraph-deployment-with-a-custom-endpoint):

```ts icon="robot" title="agent.ts" theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { LangGraphAgent } from "@copilotkit/runtime/langgraph";

// A fresh agent per conversation, keyed by thread.
export function makeAgent(threadId: string) {
  const agent = new LangGraphAgent({
    deploymentUrl: process.env.LANGGRAPH_DEPLOYMENT_URL!,
    graphId: "copilotkit_shadify", // the graph you deployed above
  });
  agent.threadId = threadId;
  return agent;
}
```

### Run the Channel listener

Declare the Channel with the `Code` from Intelligence and forward each message to the agent. Creating the Node listener is what connects the Channel. Register process teardown before you create the listener so a Ctrl-C during the connect window still stops the Channel cleanly.

```ts icon="server" title="channel.ts" theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
import { createServer } from "node:http";

import { createChannel } from "@copilotkit/channels";
import { CopilotKitIntelligence, CopilotRuntime } from "@copilotkit/runtime/v2";
import { createCopilotNodeListener } from "@copilotkit/runtime/v2/node";

import { makeAgent } from "./agent.js";

function required(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

// `name` must match the Channel Code shown in CopilotKit Intelligence.
const channel = createChannel({
  name: required("CHANNEL_CODE"),
  identifyUser: "platform",
  agent: makeAgent,
});

// Each incoming message starts a turn; runAgent streams the reply back.
channel.onMessage(async ({ thread, message }) => {
  await thread.runAgent({
    prompt: message.contentParts?.length
      ? [
          ...(message.text ? [{ type: "text" as const, text: message.text }] : []),
          ...message.contentParts,
        ]
      : message.text,
    context: [{ description: "Originating platform", value: message.platform }],
  });
});

const intelligence = new CopilotKitIntelligence({
  apiKey: required("INTELLIGENCE_API_KEY"),
});

const runtime = new CopilotRuntime({
  agents: {},
  intelligence,
  channels: [channel],
});

// Wire teardown before the listener exists, because creating it starts the
// Channel. A Ctrl-C during the connect window then still tears it down.
let teardown: (() => Promise<void>) | undefined;
const shutdown = async () => {
  await teardown?.();
};
process.once("SIGINT", shutdown);
process.once("SIGTERM", shutdown);

const listener = createCopilotNodeListener({
  runtime,
  basePath: "/api/copilotkit",
});
const channels = listener.channels;
const server = createServer(listener);
teardown = async () => {
  await channels.stop();
  if (server.listening) server.close();
};

// Creating the listener starts the Channel; there is no `channel.start()`.
// `ready()` is optional; inspect `status()` before reporting the Channel online.
await channels.ready({ timeoutMs: 30_000 });
if (channels.status().overall !== "online") {
  throw new Error("Slack Channel is not online");
}

const port = Number(process.env.PORT ?? 3000);
server.listen(port, () => {
  console.log(`Slack Channel online; listening on :${port}`);
});
```

Set the runtime secrets, then start the process:

```bash .env theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
INTELLIGENCE_API_KEY=<project-api-key>
CHANNEL_CODE=support-slack
LANGGRAPH_DEPLOYMENT_URL=http://127.0.0.1:2024
PORT=3000
```

```bash Terminal theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}
node --env-file=.env --import tsx channel.ts
```

The Channel moves from **Waiting for runtime** to **Online** in Intelligence once the process connects.

### Other platforms

Managed Slack is generally available. Managed Teams is a controlled integration target. The Channels SDK also ships direct adapters for Teams, Discord, Telegram, and WhatsApp. See the [Channels overview](https://docs.copilotkit.ai/slack) for the current platform list and per-platform setup.

## Resources

* [Deep Agents and CopilotKit](https://docs.copilotkit.ai/langgraph/deep-agents) in the CopilotKit documentation — end-to-end Next.js, dev server, and **Deep Agent** path
* [CopilotKit: LangGraph features](https://docs.copilotkit.ai/langgraph) — generative UI, HITL, shared state
* [LangGraph deployment](/oss/python/langgraph/deploy) — production and dev server

## Best practices

* **Keep the custom endpoint thin:** use it to adapt CopilotKit to your graph deployment, not to duplicate business logic already inside the graph
* **Send the schema explicitly:** `useAgentContext` should describe the UI contract every time the page mounts
* **Register a constrained component set:** expose only the components and props you actually want the model to use
* **Treat rendering as a parsing step:** parse assistant content against your schema before rendering it
* **Keep user messages plain:** only assistant messages need the structured renderer; user messages can stay normal chat bubbles

***

<div className="source-links">
  <Callout icon="terminal-2">
    [Connect these docs](/use-these-docs) to Claude, VSCode, and more via MCP for real-time answers.
  </Callout>

  <Callout icon="edit">
    [Edit this page on GitHub](https://github.com/langchain-ai/docs/edit/main/src/oss/langchain/frontend/integrations/copilotkit.mdx) or [file an issue](https://github.com/langchain-ai/docs/issues/new/choose).
  </Callout>
</div>
