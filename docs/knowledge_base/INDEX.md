# LangGraph 垂直知识库索引

> 面向「LangGraph 开发辅助问答 RAG」项目的官方资料集。
> 仅用于课程/毕设本地演示，不商用。

## 目录结构

```
knowledge_base/
├── INDEX.md                    本索引
├── README-langgraph.md         langchain-ai/langgraph 仓库 README（来源：GitHub）
├── llms.txt                    官网文档精选链接清单（docs.langchain.com）
├── source-manifest.txt         官网 Python 文档完整语料清单（6.8MB，含全部 OSS 文档链接）
├── langgraph-docs/             官方文档正文（41 个 markdown 文件）
│   ├── overview.md / graph-api.md / functional-api.md / pregel.md ...  概念类
│   ├── quickstart.md / agentic-rag.md / sql-agent.md ...               教程类
│   ├── use-graph-api.md / use-subgraphs.md / use-time-travel.md ...    How-to 类
│   ├── persistence.md / checkpointers.md / stores.md ...               持久化/记忆
│   ├── streaming.md / interrupts.md / event-streaming.md ...           流式/中断
│   ├── errors/*.md              常见错误排查（GRAPH_RECURSION_LIMIT 等）
│   └── frontend/*.md            前端/Graph 执行可视化
└── langchain-docs/              LangChain 官方文档正文（73 个 markdown 文件）
    ├── agents.md / tools.md / models.md / retrieval.md ...  核心组件
    ├── messages.md / prompts.md / components.md ...         消息/提示/组件
    ├── errors/*.md              常见错误排查
    └── ...                      其余概念与 How-to 页面
```

## 文档统计

| 子库 | 文件数 | 大小 |
|---|---|---|
| langgraph-docs | 41 个 md | 约 7.6 MB |
| langchain-docs | 73 个 md | 约 1.9 MB |

- 语言：英文（官方原文）
- 说明：langchain-docs/retrieval.md 实际重定向到 DeepAgents 检索文档（内容有效）；academy 链接仅跳转外部站点，已剔除。

## 来源

1. LangGraph 官方文档（docs.langchain.com/oss/python/langgraph/），2026-09 拉取
2. LangChain 官方文档（docs.langchain.com/oss/python/langchain/），2026-09 拉取
3. GitHub 仓库 README：https://github.com/langchain-ai/langgraph
4. 官方文档链接清单：https://docs.langchain.com/oss/python/llms-full.txt

## 后续可选扩充（非必须）

- 中文技术博客/掘金/CSDN 精选文章（领域问答更贴近中文提问习惯）
- 官方示例 notebook：https://github.com/langchain-ai/langgraph-examples
- API 参考：https://reference.langchain.com/python/langgraph/

## 入库提示（对接 RAG 分块）

- 这些是 .md 文本（实为 MDX 源码），含大量 ```python 代码块，入库时建议代码块整体切分、不截断（项目创新点）。
- 分块脚本：`H:\langraphdocs\scripts\split_docs.py`
  - `python split_docs.py`（默认处理 langgraph-docs）
  - `python split_docs.py H:\langraphdocs\knowledge_base\langchain-docs H:\langraphdocs\knowledge_base\chunks_langchain.json`
- metadata 建议：`{"source": "official-docs", "topic": "graph-api"}`，按文件名主题打标，区分 langgraph / langchain 两个子库。
