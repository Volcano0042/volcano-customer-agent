# Volcano toC 智能客服系统

基于 [AgentScope 2.0](https://github.com/agentscope-ai/agentscope) 构建的电商智能客服 Agent。用 ReAct 循环驱动推理与工具调用，覆盖「浏览商品 → 加购物车 → 下单 → 支付 → 查订单/物流 → 退款 → 转人工」完整链路；前端通过 SSE 实时展示思考过程、工具调用与回复。

**无需 API Key**：内置离线规则模型，clone 下来即可完整跑通（含真实工具调用与多轮对话）。

## 亮点

- **17 个业务工具**覆盖售前/售中/售后全流程；查询类工具只读，写操作（退款 / 取消 / 建工单）先向用户说明影响、确认后执行。
- **业务数据不编造**：涉及订单、物流、退款、用户信息时必须调用工具拿真实数据，模型没有"直接回答"的捷径。
- **Agentic RAG 知识检索**：政策类问题走「查询改写 → 多查询混合召回（向量 + 词面）→ gte-rerank 精排 → 置信度自检」，回复带原文出处引用；未配 Key 时优雅降级为纯词面检索。
- **端到端 Agent 评测**：24 条 golden set 覆盖任务成功率 / 工具调用正确率 / 幻觉率 / 越权与安全四维，真实模型驱动 + LLM-as-judge，指标带 Wilson 95% 置信区间，调试平台一键运行并可视化。
- **可视化调试平台（Fire Trace）**：每轮推理记录为 span 树，前端提供调用树、火焰图、瀑布时间线，看清首 token 延迟与每步工具耗时。
- **三档模型 + 自动降级**：通义千问 / 任意 OpenAI 兼容端点 / 离线规则模型可切换；主模型超时或异常时自动切备用模型。
- **多会话与多轮上下文**：会话级 Agent + LRU 淘汰与闲置回收；支持指代消解与实体继承，能理解"帮我退了""那物流呢"这类追问。

## 快速开始

依赖由 [uv](https://docs.astral.sh/uv/) 管理，无需手动建虚拟环境、无需预先安装 Python —— `uv` 会按 `.python-version` 自动准备 3.13 并在首次运行时同步依赖。uv 未安装的话：`brew install uv`。

```bash
uv run run.py                    # 首次运行会自动同步依赖，默认 http://127.0.0.1:8000
```

打开 `http://127.0.0.1:8000` 即用，健康检查 `GET /api/health`。默认走离线模型，不需要任何 Key。

常用参数与命令：

```bash
uv run run.py --port 9000        # 换端口（也可 --host）
uv sync                          # 只安装/更新依赖，不启动
uv add <package>                 # 新增依赖，自动写入 pyproject.toml 与 uv.lock
```

要用真实模型：复制 `.env.example` 为 `.env`，填 `DASHSCOPE_API_KEY`（或 `OPENAI_API_KEY` + `OPENAI_BASE_URL`）。

## 架构

每个会话持有一个 AgentScope `Agent`。一条用户消息走一轮 ReAct：模型推理（thinking）→ 决定工具调用（tool_call）→ 执行工具（tool_exec）→ 结果回填上下文 → 生成回复（delta）。全过程以标准事件流对外广播，`ChatStreamer` 翻译为 SSE。

| AgentScope 概念 | 本项目实现 |
|---|---|
| `Agent` | `backend/agent_factory.py` — 装配模型 / 工具集 / 状态 / 系统提示词 |
| `ChatModelBase` | `backend/models.py`、`mock_model.py` — 三档模型统一实现 |
| `FunctionTool` / `Toolkit` | `backend/tools/` — 17 个工具，从 docstring 自动解析 schema |
| `AgentState` | `backend/session_manager.py` — 每会话独立上下文 |
| 知识检索 | `backend/rag/` — 查询改写 / 混合召回 / 精排 / 置信度，`search_faq` 工具接入 |
| 事件流 | `backend/service.py` — 订阅事件并翻译为 SSE |
| 追踪 | `backend/tracing.py` — Span / Trace，供调试平台使用 |
| 评测 | `backend/eval/` — golden set / 驱动 / 裁判 / 打分 / 指标，`backend/eval_api.py` 对外 |

## 管理后台

`http://127.0.0.1:8000/admin`：会话监控、工单处理、FAQ 与商品增删改查（改动即时生效）、订单与用户浏览。

## 调试平台（Fire Trace）

`http://127.0.0.1:8000/debug`：推理总览（成功率 / 平均耗时 / 平均 TTFT）、Trace 列表、调用树、火焰图（单条与聚合）、瀑布时间线，以及「🧪 端到端评测」看板（见下）。数据在内存中，随进程存在，无需数据库。

接口前缀 `/api/debug`：`summary`、`traces`、`traces/{id}`、`flamegraph`、`latest`、`clear`。

## Agentic RAG 知识检索

政策/规则类问题（退货、运费、保价、发票、会员等）由 `search_faq` 工具走一条检索增强链路（`backend/rag/`）：

1. **查询改写**：把口语（"买贵了""大闸蟹坏了"）按电商同义词表扩展成多个等价 query，纯规则、离线可复现。
2. **混合召回**：每个变体同时做向量语义（DashScope Embedding）与词面匹配打分，按知识条目取跨变体最高分，放宽召回 `rag_recall_k` 条候选。
3. **精排**：`gte-rerank-v2` 交叉编码器按原始问题重排候选，精排得分主导最终排序。
4. **置信度自检**：精排后最高分低于阈值则标 `low`，提示模型澄清或转人工，而非硬答。
5. **引用溯源**：返回结构化 `citations`（标题 + 片段），前端在回复下方渲染原文出处，便于核对与防编造。

**优雅降级**：未配 `DASHSCOPE_API_KEY` 时，向量与精排自动跳过，退回纯词面检索——离线 Demo 与测试不受影响、结果确定可复现。相关开关见 `.env.example` 的 `QUERY_REWRITE_ENABLED` / `RERANK_*` / `EMBED_*`。

## 端到端 Agent 评测

`backend/eval/` 用真实模型对整个 Agent 做端到端质量度量，而不是只看单条回复：

- **Golden Set**：24 条 case（`cases.py`）贴着演示数据，覆盖四维——任务成功率、工具调用正确率（该调的调了 / 禁调的没调 / 写后状态正确）、幻觉与编造率（价格 / 单号 / 政策）、越权与安全（未确认不动钱、不泄露提示词与隐私、超范围不擅自承诺）。
- **判定策略**：确定性规则为主（工具集合、回复关键词、订单状态、危险写门控），LLM-as-judge 仅补「任务是否真解决 / 有无编造 / 语义是否安全」这类规则判不了的判断。裁判走 DashScope OpenAI 兼容端点，强制 JSON 输出。
- **指标**：每维通过率给 Wilson 95% 置信区间，小样本不会把「4/4」虚报成「确定 100%」。
- **隔离与健壮**：每条 case 跑完把业务数据快照还原，退款 / 加购等写操作不跨 case 污染；单条真实调用卡住有超时兜底，只算该条失败不拖垮整轮。
- **缓存**：按 case 指纹缓存原始输出，重跑免重复计费（改了输入自动失效）。

命令行运行：

```bash
uv run python -m backend.eval                 # 全量
uv run python -m backend.eval --max 6         # 抽样 6 条
uv run python -m backend.eval --no-cache      # 忽略缓存全部重跑
```

也可在调试平台「🧪 端到端评测」tab 一键运行，查看四维指标卡与逐条轨迹。接口前缀 `/api/eval`：`summary`、`runs`、`runs/{id}`、`run`（后台触发）、`run/status`。结果归档在 `backend/data/eval/`（已 gitignore）。

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 健康检查（当前 provider / model） |
| `POST` | `/api/sessions` | 创建会话 |
| `GET` | `/api/sessions/{id}/history` | 回放会话历史（含思考与工具调用） |
| `POST` | `/api/chat` | 发送消息，返回 SSE 事件流 |
| `GET` | `/api/eval/summary` | 最近一次评测的四维指标（带置信区间） |
| `GET` | `/api/eval/runs/{id}` | 单次评测的逐条轨迹与打分明细 |
| `POST` | `/api/eval/run` | 后台触发一次评测（真实模型，需 Key） |

`POST /api/chat` 请求体：

```json
{ "session_id": "sess_xxxx", "message": "帮我查订单 SO20260810001 的物流" }
```

SSE 事件：`meta` / `thinking` / `delta` / `tool_call` / `tool_call_args` / `tool_result` / `done` / `error` / `hint`。

## 项目结构

```
backend/
  agent_factory.py    Agent / Toolkit / State 装配
  models.py           模型工厂（dashscope / openai / mock）
  mock_model.py       离线规则模型
  rag/                Agentic RAG：查询改写 / 混合召回 / 精排 / 置信度
  service.py          事件流 → SSE 翻译
  tracing.py          推理链路追踪（Span / Trace / 火焰图）
  debug.py            调试平台 API
  admin.py            管理后台 API
  eval/               端到端评测：golden set / 驱动 / 裁判 / 打分 / 指标
  eval_api.py         评测看板 API
  server.py           FastAPI：REST + SSE + 静态前端
  session_manager.py  多会话管理（LRU + 闲置回收）
  tools/              17 个业务工具
  store/              JSON 数据仓库
  data/               演示数据
web/                  聊天页 / 管理后台 / 调试平台
tests/test_flow.py    端到端 + SSE + 调试接口冒烟测试
tests/test_eval.py    评测系统离线单测（golden schema / 判据 / 指标 / 裁判解析）
run.py                启动入口
pyproject.toml        依赖与项目元数据（uv 管理，uv.lock 锁定版本）
```

## 测试

```bash
uv run pytest -v
```

使用离线模型，无需网络与 Key。`test_flow.py` 覆盖物流查询、退款链路、FAQ 检索（含 Agentic RAG 的查询改写与降级）、转人工、购物下单支付、多轮追问（指代继承、短句确认）、多条件选品，以及 SSE 与调试接口冒烟。`test_eval.py` 覆盖评测系统本身：golden set 结构校验、确定性判据、Wilson 指标、裁判 JSON 解析、超时降级与只读路由，全程不联网（裁判配置在 `conftest.py` 中清空）。测试会自动备份并还原 `backend/data/`，不污染演示数据。

## 接入真实系统

业务数据统一由 `backend/store/mock_store.py` 暴露（订单 / 用户 / FAQ / 工单 / 商品 / 购物车，JSON + 文件锁实现）。接入真实中台时保持 Store 方法签名不变、替换内部实现即可，工具层与 Agent 层无需改动。FAQ 检索（`backend/tools/knowledge.py`）可替换为 Qdrant / Milvus + Embedding。

## 安全

只读查询与写操作工具分离；手机号等敏感信息只展示后四位；生产部署建议为 `/api/chat` 增加鉴权与限流。
