# Volcano toC 多智能体智能客服系统

基于 [AgentScope 2.0](https://github.com/agentscope-ai/agentscope) 构建的电商智能客服 Agent：以 ReAct 循环驱动推理与工具调用，覆盖「浏览商品 → 加购物车 → 下单 → 支付 → 查订单/物流 → 退款 → 转人工」完整链路。多智能体模式下拆成两层——监督者只做意图路由与转述、不持有业务工具，4 个专家子 agent 各自带着职责内的工具子集分域取数与处置。前端通过 SSE 实时展示思考过程、工具调用与回复。

**无需 API Key 即可体验**：内置离线规则模型，clone 后不做任何配置就能完整跑通（含真实工具调用与多轮对话）。

## 亮点

- **17 个业务工具**覆盖售前 / 售中 / 售后全流程；查询类工具只读，写操作（退款、取消、下单、支付、建工单）先向用户说明影响，确认后再执行。
- **业务数据不编造**：订单、物流、退款、用户信息一律以工具返回为准，模型没有「直接回答」的捷径。
- **Agentic RAG 知识检索**：政策类问题走「查询改写 → 多查询混合召回（向量 + 词面）→ gte-rerank 精排 → 置信度自检」，回复附带原文出处；未配 Key 时优雅降级为纯词面检索。
- **多智能体模式（可选）**：`MULTI_AGENT_ENABLED=true` 时改为「监督者路由 + 4 个专家子 agent」两层结构，专家各自只持有职责内的工具子集与独立上下文；默认关闭，走单 Agent 主链路。经三处延迟优化后，单域问题约 4~6s。
- **端到端 Agent 评测**：50 条 golden set 覆盖任务成功率、工具调用正确率、幻觉率、越权与安全四个维度，真实模型驱动 + LLM-as-judge 裁判，指标带 Wilson 95% 置信区间。多智能体配置实测：任务成功率 94%、工具调用正确率 88%、幻觉与安全 100%、整体 90%。
- **可视化调试平台（Fire Trace）**：每轮推理记录为 span 树，前端提供调用树、火焰图与瀑布时间线，可定位首 token 延迟与每步工具耗时。
- **三档模型 + 自动降级**：通义千问 / 任意 OpenAI 兼容端点 / 离线规则模型可切换；主模型超时或异常时自动切换到备用模型。
- **多会话与多轮上下文**：会话级 Agent，配 LRU 淘汰与闲置回收；支持指代消解与实体继承，能理解「帮我退了」「那物流呢」这类追问。

## 快速开始

依赖由 [uv](https://docs.astral.sh/uv/) 管理，无需手动建虚拟环境、无需预先安装 Python —— uv 会按 `.python-version` 自动准备 3.13，并在首次运行时同步依赖。尚未安装 uv 的话：`brew install uv`。

```bash
uv run run.py                    # 首次运行会自动同步依赖，默认 http://127.0.0.1:8000
```

打开 `http://127.0.0.1:8000` 即可使用，健康检查 `GET /api/health`。默认走离线模型，不需要任何 Key。

常用参数与命令：

```bash
uv run run.py --port 9000        # 换端口（也可 --host）
uv sync                          # 只安装/更新依赖，不启动
uv add <package>                 # 新增依赖，自动写入 pyproject.toml 与 uv.lock
```

接入真实模型：复制 `.env.example` 为 `.env`，填入 `DASHSCOPE_API_KEY`（或 `OPENAI_API_KEY` + `OPENAI_BASE_URL`）。多智能体、知识检索、评测等开关与默认值都写在该文件的注释里。

## 架构

每个会话持有一个 AgentScope `Agent`。一条用户消息走一轮 ReAct：模型推理（thinking）→ 决定工具调用（tool_call）→ 执行工具（tool_exec）→ 结果回填上下文 → 生成回复（delta）。全过程以标准事件流对外广播，由 `ChatStreamer` 翻译为 SSE。

| AgentScope 概念 | 本项目实现 |
|---|---|
| `Agent` | `backend/agent_factory.py` — 装配模型 / 工具集 / 状态 / 系统提示词 |
| `ChatModelBase` | `backend/models.py`、`mock_model.py` — 三档模型统一实现 |
| `FunctionTool` / `Toolkit` | `backend/tools/` — 17 个工具，从 docstring 自动解析 schema |
| `AgentState` | `backend/session_manager.py` — 每会话独立上下文 |
| 多智能体（可选） | `backend/multi_agent/` — 监督者 + 专家子 agent（agent-as-tool） |
| 知识检索 | `backend/rag/` — 查询改写 / 混合召回 / 精排 / 置信度，`search_faq` 工具接入 |
| 事件流 | `backend/service.py` — 订阅事件并翻译为 SSE |
| 追踪 | `backend/tracing.py` — Span / Trace，供调试平台使用 |
| 评测 | `backend/eval/` — golden set / 驱动 / 裁判 / 打分 / 指标，`backend/eval_api.py` 对外 |

## 多智能体模式

默认关闭（`MULTI_AGENT_ENABLED=false`，即单 Agent 主链路）。打开后改为 agent-as-tool 的两层结构（`backend/multi_agent/`）：

- **监督者**只做意图判断、编写自包含的任务描述与转述结果，自身不查数据、不执行业务，只持有 4 个委派工具；
- **专家子 agent** 按委派临时创建，各自只装配职责内的工具子集与独立上下文，跑完把结论回填给监督者。

一轮消息的编排：

1. 监督者读用户消息，判断该由哪个（或哪几个）专家处理；
2. 把订单号、手机号后四位、用户已确认的意愿等关键信息写成**自包含的 task**，调用 `delegate_to_*` 委派；
3. 专家在自己的上下文里跑一轮 ReAct（查订单、查政策、执行退款…），把结论与用到的工具回填；
4. 同一回合委派多个专家时由 AgentScope 并发执行；只委派一个且答案完整时按 `PASSTHROUGH_SINGLE_DELEGATION` 直接采用专家答案，否则由监督者综合转述。

| 委派工具 | 角色 | 工具子集 |
|---|---|---|
| `delegate_to_knowledge_agent` | 知识检索专家（只读） | `search_faq` |
| `delegate_to_logistics_agent` | 物流查询专家（只读） | `query_order`、`list_recent_orders`、`track_logistics` |
| `delegate_to_refund_agent` | 退款售后专家 | `query_order`、`list_recent_orders`、`check_refund_policy`、`apply_refund`、`cancel_order`、`create_ticket`、`transfer_to_human` |
| `delegate_to_shopping_agent` | 导购交易专家 | `list_products`、`query_product`、`view_cart`、`add_to_cart`、`update_cart_item`、`place_order`、`pay_order`、`query_user_profile` |

四个专家的工具子集合计覆盖全部 17 个业务工具，`tests/test_multi_agent.py` 有回归测试守住这一点。

转人工同理：`transfer_to_human` 只挂在退款售后专家名下，监督者手里没有，因此提示词显式规定「用户明确要求转人工 → 委派给该专家」。

### 配置项

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MULTI_AGENT_ENABLED` | `false` | 总开关，关闭时走单 Agent 主链路 |
| `SPECIALIST_MAX_ITERS` | `6` | 单个专家子 agent 的最大「推理-行动」轮次 |
| `SUPERVISOR_MODEL` | 空 | 监督者单独使用的模型，留空则与主模型同款 |
| `PASSTHROUGH_SINGLE_DELEGATION` | `true` | 只委派一个专家时直接采用其答案 |

### 延迟优化

两层结构比单 Agent 多出一次路由调用与一次终答调用，未做下述优化时单域问题约 22s（单 Agent 约 3.6s）；三处优化后同一问题约 4~6s：

1. **监督者使用快速模型**（`SUPERVISOR_MODEL`）。监督者不取数，只做路由与转述，用小模型即可胜任，实测路由轮 4.96s → 0.43s。配置了快速模型时降级链为「快速模型 → 主模型」，可用性不因换模型而降低。
2. **模型实例复用**。`create_chat_model` 按 (provider, model, 凭证) 做进程级缓存。每次委派都会新建专家子 agent，连带新建 `openai.AsyncClient`（连接池与 TLS 握手）；而模型实例不持有会话状态，可安全并发共享。
3. **单专家终答透传**（`PASSTHROUGH_SINGLE_DELEGATION`）。专家已经输出面向用户的正文，监督者再改写一遍既慢又易失真。当本回合恰好只有一个 `delegate_to_*` 结果、且该专家给出完整答案时，直接把这份答案作为本轮回复，省掉整轮模型调用；跨域的多个专家委派仍由监督者综合，闲聊、专家报错、转人工等路径不受影响。

三项优化只作用于多智能体模式，`MULTI_AGENT_ENABLED=false` 时的单 Agent 主链路不受影响；另外透传命中时回复会整段一次推送，不再逐字增量。

> 上述耗时取自同一问题（查询物流）的单次实测，受网络与模型侧波动影响。跨域问题（如「发货了吗 + 退货规则」）需两个专家并行取数并再综合一次，耗时高于单域问题。

`GET /api/health` 会返回当前模式与监督者模型（`mode`、`supervisor_model`），可用于确认配置是否生效。

### 评测结果

同一 golden set 50 条、同一主模型（`deepseek-v4.1-flash`）与裁判，多智能体（含三处延迟优化）配置实测：

| 维度 | 通过 / 总数 | 通过率 | 95% 置信区间 |
|---|---|---|---|
| 任务成功率 | 47/50 | **94%** | [0.84, 0.98] |
| 工具调用正确率 | 30/34 | **88%** | [0.73, 0.95] |
| 幻觉与编造（未被判编造） | 27/27 | **100%** | [0.88, 1.00] |
| 越权与安全 | 22/22 | **100%** | [0.85, 1.00] |
| 整体 | 45/50 | **90%** | [0.79, 0.96] |

- **Golden set 50 条**，贴着演示数据编写，按场景分布：越权与安全 16、知识检索 15、退款取消 7、订单 4、购物闭环 4、物流 2、转人工 2；每条 case 显式声明自己参与统计的维度（同一 case 可同时计入多个维度）。
- **被测链路**固定走多智能体（监督者 + 专家子 agent），与线上 `MULTI_AGENT_ENABLED=true` 时一致，不含任何 mock 或降级路径。
- **判定以确定性判据为主**：工具集合（`tools_any` / `tools_all` / `tools_forbidden`）、回复关键词、订单写后状态、危险写门控；LLM-as-judge 只补规则判不了的三件事——任务是否真正解决、有无编造、语义层是否安全。
- **隔离**：每条 case 开跑前回滚业务数据快照，退款、加购等写操作不跨 case 污染；单条调用超时只算该条失败，不拖垮整轮。
- **区间**：每维通过率配 Wilson 95% 置信区间，小样本不会把「4/4」当作确定的 100%。

## Agentic RAG 知识检索

政策 / 规则类问题（退货、运费、保价、发票、会员等）由 `search_faq` 工具走一条检索增强链路（`backend/rag/`）：

1. **查询改写**：按电商同义词表把口语（「买贵了」「大闸蟹坏了」）扩展成多个等价 query，纯规则、离线可复现。
2. **混合召回**：每个变体同时做向量语义（DashScope Embedding）与词面匹配打分，按知识条目取跨变体最高分，放宽召回 `rag_recall_k` 条候选。
3. **精排**：`gte-rerank-v2` 交叉编码器按原始问题重排候选，精排得分主导最终排序。
4. **置信度自检**：精排后最高分低于阈值则标为 `low`，提示模型澄清或转人工，而不是硬答。
5. **引用溯源**：返回结构化 `citations`（标题 + 片段），前端在回复下方渲染原文出处，便于核对与防编造。

**优雅降级**：未配 `DASHSCOPE_API_KEY` 时，向量与精排自动跳过，退回纯词面检索 —— 离线 Demo 与测试不受影响，结果确定可复现。相关开关见 `.env.example` 的 `QUERY_REWRITE_ENABLED` / `RERANK_*` / `EMBED_*`。

## 端到端 Agent 评测

`backend/eval/` 用真实模型对整个 Agent 做端到端质量度量，而不是只看单条回复：

- **被测链路**：固定走多智能体（`multi_agent=True`，监督者 + 专家子 agent），即线上开启 `MULTI_AGENT_ENABLED` 时正在跑的那条链路。
- **Golden Set**：50 条 case（`cases.py`）贴着演示数据，覆盖四个维度 —— 任务成功率、工具调用正确率（该调的调了 / 禁调的没调 / 写后状态正确）、幻觉与编造率（价格 / 单号 / 政策）、越权与安全（未确认不动钱、不泄露提示词与隐私、超范围不擅自承诺）。
- **判定策略**：以确定性规则为主（工具集合、回复关键词、订单状态、危险写门控），LLM-as-judge 只补「任务是否真正解决 / 有无编造 / 语义是否安全」这类规则判不了的判断。裁判走 DashScope OpenAI 兼容端点，强制 JSON 输出；轨迹里每个工具的真实返回也一并交给裁判，供其核对结论有没有依据。
- **指标**：每一维通过率给出 Wilson 95% 置信区间。
- **隔离与健壮**：每条 case 跑完即还原业务数据快照，退款、加购等写操作不跨 case 污染；单条真实调用超时只算该条失败，不拖垮整轮。
- **录制与线上一致**：驱动录的是**用户真正看得到的正文**——调工具前的过程性旁白（"I'll check…"）在线上会被 SSE 层丢弃，录制走同一套抑制规则（`backend/narration.py`）。
- **缓存**：按 case 指纹缓存实跑轨迹与裁判结果，命中即秒出、不再计费（输入有改动则自动失效）。`--no-cache` 与页面上的「忽略缓存重跑」只跳过读取，跑完仍写回缓存；驱动超时的空轨迹不落盘，裁判缺失的条目下次自动补判。判据或裁判视野变更时 `JUDGE_VERSION` +1，旧裁决作废并复用轨迹重判。

命令行运行：

```bash
uv run python -m backend.eval                 # 全量
uv run python -m backend.eval --max 6         # 抽样 6 条
uv run python -m backend.eval --cases id1,id2 # 只跑指定 case（改了某几条判据时用）
uv run python -m backend.eval --no-cache      # 忽略缓存全部重跑（跑完刷新缓存）
```

也可在调试平台的「🧪 端到端评测」tab 一键运行，查看四维指标卡与逐条轨迹；当前结果见「多智能体模式 → 评测结果」。接口前缀 `/api/eval`：`summary`、`runs`、`runs/{id}`、`run`（后台触发）、`run/status`、`cache`。结果归档在 `backend/data/eval/`（已 gitignore）。

## 管理后台

`http://127.0.0.1:8000/admin`：会话监控、工单处理、FAQ 与商品的增删改查（改动即时生效）、订单与用户浏览。

## 调试平台（Fire Trace）

`http://127.0.0.1:8000/debug`：推理总览（成功率 / 平均耗时 / 平均 TTFT）、Trace 列表、调用树、火焰图（单条与聚合）、瀑布时间线，以及「🧪 端到端评测」看板。追踪数据存放在内存中，随进程结束而清空，无需数据库。

接口前缀 `/api/debug`：`summary`、`traces`、`traces/{id}`、`flamegraph`、`latest`、`clear`。

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/health` | 健康检查（当前 provider / model / 模式 / 监督者模型） |
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
  multi_agent/        多智能体：监督者 / 专家子 agent / 终答透传
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
tests/test_multi_agent.py  多智能体离线单测（工具子集 / 委派 / 终答透传 / 模型选择）
run.py                启动入口
pyproject.toml        依赖与项目元数据（uv 管理，uv.lock 锁定版本）
```

## 测试

```bash
uv run pytest -v
```

全部使用离线模型，无需网络与 Key：

- `test_flow.py` —— 业务主链路与接口冒烟：物流查询、退款链路、FAQ 检索（含 Agentic RAG 的查询改写与降级）、转人工、购物下单支付、多轮追问（指代继承、短句确认）、多条件选品，以及 SSE 与调试接口。
- `test_eval.py` —— 评测系统本身：golden set 结构校验、确定性判据、Wilson 指标、裁判 JSON 解析、超时降级与只读路由（裁判配置在 `conftest.py` 中清空）。
- `test_multi_agent.py` —— 多智能体层：专家工具子集与安全边界、17 个业务工具的可达性、委派工具形状、专家答案与过程性旁白的区分、终答透传的命中与不命中判定、监督者模型选择与实例复用。

测试会自动备份并还原 `backend/data/`，不污染演示数据。

## 接入真实系统

业务数据统一由 `backend/store/mock_store.py` 暴露（订单 / 用户 / FAQ / 工单 / 商品 / 购物车，JSON + 文件锁实现）。接入真实中台时保持 Store 方法签名不变、只替换内部实现，工具层与 Agent 层无需改动。FAQ 检索（`backend/tools/knowledge.py`）可替换为 Qdrant / Milvus + Embedding。

## 安全

只读查询工具与写操作工具分离；手机号等敏感信息只展示后四位；配置对象 `Settings` 的 `repr` 不输出 API Key，避免密钥随日志或断言失败外泄；生产部署建议为 `/api/chat` 增加鉴权与限流。