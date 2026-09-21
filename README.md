# RAG Agent · 便签问答板

基于 LangGraph 编排的检索增强问答 Agent。问题依次经过意图路由、多轮查询改写、向量与 BM25 双路召回、Cross-Encoder 精排后生成带引用的答案，并内置滚动摘要式长期记忆。后端由 `langgraph dev` 托管，前端是软木板便签风格的流式对话网页，另提供知识库管理、参数热更新与检索层离线评估三个调试面板。

适用于希望在本地或内网运行、对检索质量可观测、对数据和成本可控的团队：本地 BGE 嵌入与重排模型让问答链路零 API 成本（只有对话模型走 LLM），CPU 即可运行。

## 功能概览

- **意图分流**：`route` 节点先判断问题是否与知识库相关，闲聊走 `common` 直答，领域问题才走检索链路，节省 token 与延迟。
- **多轮指代改写**：`searchrebuild` 结合近期历史把「它需要 Python 几版本？」改写成自包含检索词；没有历史时短路，不调用 LLM。
- **混合检索**：Chroma + BGE 向量语义召回与 BM25（jieba 分词）关键词召回双路并行，互为补充。
- **精排与熔断**：本地 bge-reranker 或云端（硅基流动、阿里云）Cross-Encoder 对合并结果精排取 top-k；可配置绝对分阈值，最高分仍不达标时判定知识库未命中，不硬答。
- **邻域扩展**：命中块前后自动补入同文档兄弟块一起喂给生成模型，实现「小块检索、带语境生成」（仅 Chroma，可关闭）。
- **带引用生成**：答案标注 `[1][2]` 来源编号，底部展示引用文件。
- **滚动摘要记忆**：消息条数或 token 超过阈值即触发压缩，旧历史按比例进入摘要、近期对话原样保留。
- **流式输出**：SSE 逐 token 返回，前端边收边渲染 Markdown 与代码高亮。
- **便签风 Web UI**：React + TypeScript + Tailwind，软木板底、鹅黄/薄荷便签、胶带/图钉、手写体；支持会话新建、切换、双击重命名与删除（懒创建，不产生空会话）。
- **三个网页调试面板**：📚 知识库（文档增删改查与热刷新）、📊 检索评估（黄金集离线回归）、⚙️ 调试设置（22 项参数热更新并写回 `.env`）。
- **离线评估与 HTTP 接口**：内置 20 题黄金集，支持网页或命令行触发，产出 Hit/MRR、精排分分布与熔断阈值扫描；评估通过 `/admin/eval/retrieval` 端点异步运行。

## 工作原理

```
START → route
        ├─ is_relevant=true  → searchrebuild → rag → rerank → generate ─┐
        └─ is_relevant=false → common ──────────────────────────────────┤
                                                                         ↓
                                              条件边 should_summarize?
                                                 ├─ true  → summarize → END
                                                 └─ false → END
```

| 节点 | 职责 |
| --- | --- |
| `route` | 结构化输出判定问题是否与知识库相关 |
| `searchrebuild` | 结合近期历史改写检索词（无历史时短路） |
| `rag` | 向量与 BM25 双路召回，合并去重 |
| `rerank` | Cross-Encoder 精排取 `RERANK_TOP_K`，执行分数熔断与邻域扩展，产出引用来源 |
| `generate` | 基于检索上下文生成带引用的答案 |
| `common` | 回答与知识库无关的通用闲聊 |
| `summarize` | 超过阈值时把旧消息压缩为滚动摘要 |

状态通过 `InputState / MainState / OutputState` 分层收窄，对外只暴露 `answer` 与 `sources`。状态定义见 [state.py](./src/agent/state/state.py)，编排在 [graph.py](./src/agent/graph.py)。

## 目录结构

```
RAG_Agent/
├── src/agent/
│   ├── graph.py              # LangGraph 工作流编排
│   ├── config.py             # 环境变量配置与热更新机制（权威清单）
│   ├── admin_server.py       # /admin 调试路由（Starlette，挂到 langgraph dev）
│   ├── eval_jobs.py          # 评估后台任务注册表（内存态、互斥、保留最近 10 次）
│   ├── state/                # Input/Main/Output State
│   ├── nodes/                # route/common/searchrebuild/rag/rerank/generate/summarize
│   ├── prompts/              # 各节点提示词模板
│   ├── llm/                  # LLM 工厂（deepseek/openai/qwen，lru 缓存）
│   ├── embedding/            # BGE 本地 / OpenAI / DashScope 嵌入工厂
│   ├── retrieval/            # 向量检索、BM25、CrossEncoder 重排、离线评估
│   ├── ingestion/            # 文档加载、切分、向量库增量同步
│   ├── checkpointer/         # memory / redis / postgres 检查点
│   └── utils/                # token 估算、历史处理、文档工具
├── eval/
│   └── golden_retrieval.jsonl  # 检索回归黄金集（20 题，可扩充）
├── scripts/
│   └── eval_retrieval.py     # 检索评估命令行入口
├── web/                      # React + TS + Vite + Tailwind 便签风前端
├── tests/                    # 离线单测（默认）与集成测试（-m integration）
├── docs/knowledge_base/      # 知识库源文档（Markdown 等）
├── langgraph.json            # langgraph dev 图与自定义 HTTP 路由声明
├── pyproject.toml
└── .env.example              # 环境变量模板
```

## 准备工作

开始之前，请确认环境满足以下要求：

- **Python 3.11 或更高版本**（langgraph-api 的硬性要求）
- **[uv](https://docs.astral.sh/uv/)**：依赖与虚拟环境管理工具
- **Node.js 18 或更高版本**与 npm：构建并运行前端
- **一个 LLM API Key**：默认使用 DeepSeek，也支持 OpenAI 兼容协议与通义千问
- **CPU 即可**：默认使用本地 BGE 嵌入与本地重排模型，首次加载约 20 秒；无 GPU 时 `pyproject.toml` 已配置 CPU 版 torch 镜像

## 快速开始

### 1. 安装依赖并配置环境变量

```bash
cd RAG_Agent
uv sync                       # 按 lockfile 创建 .venv 并安装全部依赖
cp .env.example .env          # 复制环境变量模板
```

编辑 `.env`，至少填入对话模型的 Key：

```text
DEFAULT_LLM_PROVIDER=deepseek
DEFAULT_LLM_MODEL=deepseek-chat
DEFAULT_LLM_API_KEY=sk-xxxxxx
```

### 2. 构建知识库（首次）

把文档放入 `docs/knowledge_base/`（仓库已内置一份 LangGraph/LangChain 文档），然后增量同步到 Chroma：

```bash
uv run python -m agent.ingestion.sync docs/knowledge_base                    # 预览变更（dry-run）
uv run python -m agent.ingestion.sync docs/knowledge_base --apply            # 执行增量同步
uv run python -m agent.ingestion.sync docs/knowledge_base --rebuild --apply  # 索引异常时全量重建
```

同步以文件内容的 SHA-256 做三路 diff（新增、更新、删除），幂等且可重复执行。通过命令行改库后需要重启图进程以重建 BM25 内存索引；通过下面的网页知识库面板改库则会自动热刷新。

### 3. 启动后端

```bash
uv run langgraph dev                       # 默认监听 127.0.0.1:2024
# 或显式指定参数
uv run langgraph dev --host 127.0.0.1 --port 2024 --no-browser
```

启动后可访问：

- API 文档（Swagger）：<http://127.0.0.1:2024/docs>
- LangGraph Studio（可视化调试图与状态）：启动日志中的 `smith.langchain.com/studio/?baseUrl=...` 链接
- 调试接口：`http://127.0.0.1:2024/admin/*`

### 4. 启动前端并开始提问

```bash
cd web
npm install
npm run dev                 # 输出 http://127.0.0.1:5173
```

在浏览器打开 Vite 输出的本地链接（在 IDE 中可直接 Ctrl/Cmd+点击终端链接走端口转发）。前端通过 Vite 同源代理访问后端：`/lg/*` 转发到 LangGraph API，`/admin/*` 转发到调试接口，因此无需配置跨域。

打开页面后：

- 在输入框直接发问，自动创建第一张便签；
- 点击侧栏「📚 知识库」在线管理文档；
- 点击侧栏「📊 检索评估」跑检索质量回归（见下一节）；
- 点击侧栏「⚙️ 调试设置」热更新运行参数。

## 运行检索层评估

检索评估回答一个从对话质量里很难看出的问题：**召回没问题，但精排是否把正确文档挤出了 top-k？** 评估对内置黄金集跑一遍真实双路召回与本地 CrossEncoder 精排，不调用聊天 LLM、不调云端 API，CPU 上约需 5–9 分钟（含模型冷启动）。

评估在后台线程运行，触发后立即返回，不阻塞问答服务与事件循环；同一时刻只允许一个评估任务，结果保留在内存中（最近 10 次，重启清空）。

### 在网页上运行评估

1. 点击侧栏「📊 检索评估」打开面板。面板会先加载最近一次评估结果（没有则显示空态）。
2. 点击「开始评估」。按钮变为「评估进行中…」并禁用，页面每 3 秒自动刷新进度。
3. 评估完成后查看：
   - **指标卡**：样例数、Hit@1、Hit@3、MRR@3、召回阶段命中率、被精排挤出的条数；
   - **精排分分布**：top1 分数的 min/median/mean/max，用于判断分数尺度；
   - **熔断阈值扫描表**：每个候选阈值下的正确保留、误杀命中（红底）、正确拒绝、漏网保留（黄底）；
   - **未达标样例**：逐条列出召回未命中或精排挤出 top3 的问题、期望来源与精排榜单。
4. 运行中再次点击触发会收到 409 冲突，面板自动转为轮询当前任务。

### 在命令行运行评估

```bash
# 默认读取 eval/golden_retrieval.jsonl，打印指标报告
uv run python scripts/eval_retrieval.py

# 导出含逐样例明细的完整 JSON 报告
uv run python scripts/eval_retrieval.py --json out/eval.json

# 自定义黄金集、k 档位与熔断扫描网格
uv run python scripts/eval_retrieval.py \
  --golden eval/golden_retrieval.jsonl \
  --ks 1,3,5 \
  --thresholds 0.1,0.3,0.5,0.7,0.9
```

参数说明：

| 参数 | 说明 |
| --- | --- |
| `--golden` | 黄金集 JSONL 路径，默认为 `eval/golden_retrieval.jsonl` |
| `--top-k` | 精排保留条数，缺省读取 `RERANK_TOP_K` |
| `--ks` | 指标统计档位，逗号分隔，默认为 `1,3` |
| `--thresholds` | 熔断扫描阈值网格，默认为 `0.1,0.3,0.5,0.7,0.9` |
| `--json` | 把完整报告（含逐样例明细）写入指定 JSON 文件 |

### 解读当前基线

20 题黄金集在本地模型上的校准基线：

| 指标 | 数值 | 含义 |
| --- | --- | --- |
| 召回阶段命中率 | 1.00 | 精排前的双路召回覆盖了全部期望来源 |
| Hit@1 | 0.65 | 65% 的问题精排首位即期望文档 |
| Hit@3 | 0.95 | 95% 的问题期望文档出现在精排前三 |
| MRR@3 | 0.783 | 期望文档在前三中的平均倒数排名 |

已知薄弱点是 g12（checkpointer 持久化问题）：`persistence.md` 已被召回，但精排把它排到了 `agents.md`、`streaming.md` 之后。熔断扫描还显示绝对阈值对中文问法偏严格——0.3 阈值会误杀两个高分特征不明显但命中正确的样例。**标定 `RERANK_SCORE_THRESHOLD` 时优先选择「误杀命中」为 0、「正确拒绝」尽量多的档位。**

### 维护黄金集

黄金集每行一个 JSON 样例，`#` 开头的行会被忽略：

```json
{"id": "g01", "question": "如何查看图的执行历史并回滚到之前某个状态？", "expected_sources": ["langgraph-docs/use-time-travel.md"], "note": "time travel 操作手册"}
```

编写要求：问题使用真实用户口吻（允许中文转述，不要堆砌文档标题关键词）；`expected_sources` 写知识库相对 posix 路径，可列多个等价来源；新增样例后重跑一次评估确认基线。

### 通过 HTTP 接口触发

评估接口随 `langgraph dev` 挂载在 `/admin/eval/retrieval`，仅供本地调试：

| 方法与路径 | 响应 | 说明 |
| --- | --- | --- |
| `POST /admin/eval/retrieval` | `202 {"job_id", "status":"running"}` | 触发一次评估，立即返回 |
| 同上（已有任务运行中） | `409 {"detail": ...}` | 任务互斥，拒绝重复触发 |
| `GET /admin/eval/retrieval/{job_id}` | `200` 任务对象 | 查询指定任务，含指标与逐样例汇总 |
| `GET /admin/eval/retrieval/latest` | `200` 任务对象 | 查询最近一次任务 |
| 同上（从未运行过） | `404 {"detail": ...}` | 尚无评估任务 |

## 配置运行参数

所有参数集中在 [config.py](./src/agent/config.py) 的 `Config` 类，从环境变量与 `.env` 读取，`.env.example` 提供带注释的完整模板。常用参数如下：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEFAULT_LLM_PROVIDER` | `deepseek` | 可选 `deepseek` / `openai` / `qwen` |
| `DEFAULT_LLM_MODEL` | `deepseek-chat` | 对话模型名 |
| `DEFAULT_LLM_TEMPERATURE` | `0` | RAG 场景建议保持 0 |
| `EMBEDDING_PROVIDER` | `bge` | 可选 `bge` 本地 / `openai` / `dashscope` |
| `EMBEDDING_MODEL` | 本地 BGE | `bge` 填本地模型目录，云端填 API 模型名 |
| `KNOWLEDGE_DOCS_DIR` | `docs/knowledge_base` | 知识库源文档根目录（相对路径按项目根解析） |
| `VECTORSTORE_PROVIDER` | `chroma` | 可选 `chroma` / `postgres` |
| `RETRIEVAL_K` | `3` | 向量召回条数 |
| `BM25_K` | `5` | BM25 召回条数 |
| `RERANK_ENABLED` | `true` | 关闭后召回结果保序直通 |
| `RERANK_PROVIDER` | `local` | 可选 `local` / `siliconflow` / `dashscope` |
| `RERANK_TOP_K` | `3` | 重排后送入生成的文档数 |
| `RERANK_SCORE_THRESHOLD` | 空（关闭） | 精排最高分低于此值则判定未命中、清空候选；本地 bge 为 0~1 归一化分，用评估扫描标定，勿跨模型照搬 |
| `RERANK_CONTEXT_WINDOW` | `1` | 命中块前后各补几片同文档兄弟块；`0` 关闭；仅 Chroma 生效 |
| `SUMMARY_TRIGGER_MESSAGES` | `10` | 消息条数超过即压缩 |
| `SUMMARY_TRIGGER_TOKENS` | `4000` | token 估算超过即压缩（与条数满足其一） |
| `SUMMARY_COMPRESS_RATIO` | `0.5` | 触发后旧历史进入摘要的比例 |
| `REBUILD_RETRIEVAL_K` | `6` | 改写检索词时携带的近期历史条数 |
| `CHECKPOINTER_BACKEND` | `memory` | 可选 `memory` / `redis` / `postgres` |

### 在网页上热更新参数

侧栏「⚙️ 调试设置」面板把 22 项参数分为 4 组：对话模型（5 项）、检索与重排（9 项）、嵌入模型（4 项）、历史压缩（4 项）。保存时后端依次执行：

1. 校验类型与取值，非法值整体拒绝且不生效；
2. 写入进程内运行时覆盖，下一次提问即使用新值；
3. 行级 upsert 回 `.env`，重启后保留；
4. 清空 LLM、节点链、嵌入与重排模型的实例缓存，强制按新配置重建。

密钥字段只回显脱敏标记（如 `sk-…55`），留空提交表示不修改。注意 `CHECKPOINTER_BACKEND`、`CHUNK_SIZE` 等涉及存储结构或启动期初始化的少数参数不支持热更新，需要修改 `.env` 后重启进程。该接口随 `langgraph dev` 同端口挂载，**仅供本地开发调试，请勿暴露到公网**。

### 在网页上管理知识库

侧栏「📚 知识库」面板对源文档目录与 Chroma 集合做文件级双写管理，无需命令行：

- **查看**：列出全部文档的切片数、体积、索引状态（已同步、已改动、未入库）与游离切片；顶部搜索框支持按文件名/路径实时过滤（大小写不敏感，空格分隔的多个词需同时命中）。
- **新增**：「＋ 新建文本」在线编写 `.md/.txt/.json/.py`，或「📤 上传文件」放入 PDF、Markdown 等文档（支持 `.pdf .md .markdown .txt .json .py`，单文件不超过 20 MiB）。
- **修改**：文本类直接在网页编辑器中改正文；二进制文件用「上传同名文件」覆盖；也可对单个文件执行「🔁 重新索引」。
- **删除**：删除文件并按 `source` 清除其全部切片，自动回收空目录。

每次写操作保存后都会热刷新检索索引（BM25 内存索引按新集合重建），下一次提问即使用新知识库，无需重启后端。接口挂在 `/admin/kb/*`，对文件路径做了根目录越界校验，同样仅供本地调试。批量入库仍建议使用命令行 `ingestion.sync`。

## 测试

```bash
# 离线单测：不依赖真实模型与外网，237 个用例
uv run pytest

# 追加运行集成测试（需要真实本地模型或 LLM API）
uv run pytest -m integration

# 只跑检索评估集成测试：真实本地嵌入与重排跑黄金集，零 LLM API，CPU 约 8 分钟
uv run pytest -m integration tests/integration_tests/test_retrieval_eval.py

# 前端类型检查（tsc）与生产构建
cd web && npm run build

# 后端静态检查
uv run ruff check src tests && uv run mypy --strict src
```

检索评估集成测试内置防退化护栏（hit@3 ≥ 0.85、hit@1 ≥ 0.50、MRR@3 ≥ 0.70、召回命中 ≥ 0.95、精排挤出 ≤ 3 条），指标线按真实基线下调一档设置，避免卡满分导致脆弱。

## 技术栈

- **编排与服务**：LangGraph 1.x、langgraph-api（dev 服务器与 Studio tracing）、Starlette（调试路由）
- **模型**：LangChain、langchain-deepseek、langchain-openai、sentence-transformers（BGE）
- **检索**：Chroma、rank-bm25 + jieba、Cross-Encoder 重排
- **持久化**：InMemory（默认）/ Redis / Postgres checkpointer
- **前端**：React 18、TypeScript、Vite 5、Tailwind 3、@langchain/langgraph-sdk、react-markdown
- **工程工具**：uv、pytest、ruff、mypy

## 常见问题

**首次提问很慢。**
本地 BGE 嵌入与重排模型首次加载约需 20 秒，之后缓存复用。也可把 `RERANK_PROVIDER` 改为 `siliconflow` 使用云端免费档位。

**重启后会话消失。**
默认使用 `memory` checkpointer，进程重启即清空。需要持久化时把 `CHECKPOINTER_BACKEND` 改为 `redis` 或 `postgres`。

**改了知识库不生效。**
网页「📚 知识库」面板中的增删改会自动热刷新并即时生效；命令行执行 `ingestion.sync --apply` 后需要重启后端，或在面板中对相关文件执行重新索引，以重建 BM25 索引。

**前端连不上后端。**
确认后端监听 2024、前端通过 Vite 链接（5173）访问，而不是直连沙箱内网 IP。代理规则见 [web/vite.config.ts](./web/vite.config.ts)。

**评估任务会影响正在进行的问答吗？**
评估运行在独立后台线程，触发请求立即返回，不占用服务事件循环；但评估全程跑本地 CPU 模型，机器繁忙时问答响应可能变慢。任务状态保存在内存中，保留最近 10 次已结束任务，服务重启后清空（评估只读、可随时重跑）。

**如何确定 `RERANK_SCORE_THRESHOLD` 的取值？**
跑一次检索评估，查看熔断阈值扫描表：选择「误杀命中」为 0 且「正确拒绝」尽量多的档位。绝对阈值依赖具体重排模型的分数尺度，本地 bge-reranker 输出 0~1 归一化分，切换模型后需要重新标定。

## License

MIT
