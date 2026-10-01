# -*- coding: utf-8 -*-
"""端到端流程测试（使用 AgentScope 离线规则模型，无网络依赖）。"""
import json
import os
import re
import shutil
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "backend" / "data"

# 需要备份还原的业务数据文件
_DATA_FILES = ("orders.json", "tickets.json", "faq.json", "users.json", "cart.json", "products.json")


@pytest.fixture(autouse=True)
def _preserve_data(tmp_path):
    """备份并在结束时还原 JSON 数据，避免测试污染演示数据。

    memory.json（跨会话用户记忆）也一并隔离：测试前清空、结束后还原原状，
    否则上一条用例记住的用户档案会串到下一条，让结果不可复现。
    """
    backup = tmp_path / "data_bak"
    backup.mkdir()
    for name in _DATA_FILES:
        shutil.copy(_DATA / name, backup / name)

    memory = _DATA / "memory.json"
    memory_backup = backup / "memory.json"
    had_memory = memory.exists()
    if had_memory:
        shutil.copy(memory, memory_backup)
    memory.write_text("{}", encoding="utf-8")

    # 会话存档：HTTP 用例会走真实 server，persist 会把 AgentState 写到
    # backend/data/sessions/。记录测试前的文件，结束后删掉新增的。
    sessions_dir = _DATA / "sessions"
    before = (
        {p.name for p in sessions_dir.glob("*.json")}
        if sessions_dir.exists()
        else set()
    )

    yield

    for name in _DATA_FILES:
        shutil.copy(backup / name, _DATA / name)
    if had_memory:
        shutil.copy(memory_backup, memory)
    elif memory.exists():
        memory.unlink()
    if sessions_dir.exists():
        for path in sessions_dir.glob("*.json"):
            if path.name not in before:
                path.unlink()


def _make_agent(session_id: str = "test_sess"):
    os.environ["MODEL_PROVIDER"] = "mock"
    from backend.config import get_settings
    from backend.agent_factory import build_customer_service_agent
    from backend.mock_model import OfflineChatModel

    settings = get_settings()
    return build_customer_service_agent(
        settings,
        session_id,
        model=OfflineChatModel(),
    )


async def _collect_reply(agent, message: str):
    """驱动 Agent 回复一轮，返回 (调用的工具列表, 最终文本)。"""
    from agentscope.message import UserMsg, TextBlock
    from agentscope.event import (
        TextBlockDeltaEvent,
        ToolCallStartEvent,
        ReplyEndEvent,
    )

    text: list[str] = []
    tools: list[str] = []
    async for evt in agent.reply_stream(
        UserMsg(name="用户", content=[TextBlock(text=message)])
    ):
        if isinstance(evt, TextBlockDeltaEvent):
            text.append(evt.delta)
        elif isinstance(evt, ToolCallStartEvent):
            tools.append(evt.tool_call_name)
        elif isinstance(evt, ReplyEndEvent):
            assert str(evt.finished_reason) == "completed"
    return tools, "".join(text)


async def test_query_logistics():
    agent = _make_agent()
    tools, reply = await _collect_reply(agent, "帮我查一下订单 SO20260810001 的物流")
    assert "track_logistics" in tools
    assert "火山速运" in reply
    assert "SO20260810001" in reply or "VL8820472631" in reply


async def test_refund_chain_updates_store():
    agent = _make_agent()
    tools, reply = await _collect_reply(agent, "我想把 SO20260812003 这个订单退了")
    assert "check_refund_policy" in tools
    assert "apply_refund" in tools
    assert "提交" in reply or "TKF" in reply

    # 订单应被标记为退款中
    from backend.store.mock_store import order_store

    order = await order_store.find_by_id("SO20260812003")
    assert order["status"] == "退款中"


async def test_faq_search():
    agent = _make_agent()
    tools, reply = await _collect_reply(agent, "退货规则是什么？")
    assert "search_faq" in tools
    assert "七天" in reply


async def test_transfer_human():
    agent = _make_agent()
    tools, reply = await _collect_reply(agent, "我要转人工客服")
    assert "transfer_to_human" in tools
    assert "排队" in reply or "转人工" in reply




async def test_shopping_add_and_view_cart():
    agent = _make_agent()
    tools, reply = await _collect_reply(agent, "手机号后四位 3721，把 P1002 加进购物车")
    assert "add_to_cart" in tools
    assert "购物车" in reply

    agent2 = _make_agent("test_sess2")
    tools2, reply2 = await _collect_reply(agent2, "手机号后四位 3721，看看我的购物车")
    assert "view_cart" in tools2
    # 购物车渲染行小计：保温焖烧杯 单价129 × 2 = ¥258
    assert "保温焖烧杯" in reply2
    assert "258" in reply2


async def test_shopping_checkout_and_pay():
    agent = _make_agent()
    # 先加一件商品进购物车
    await _collect_reply(agent, "手机号后四位 3721，把 P1001 加进购物车")

    # 按购物车下单
    tools, reply = await _collect_reply(agent, "手机号后四位 3721，按购物车下单，地址用文三路 199 号")
    assert "place_order" in tools
    assert "SO" in reply

    from backend.store.mock_store import order_store
    orders = await order_store.find_by_phone_tail("3721")
    new_order = max((o for o in orders if o.get("status") == "待付款"), key=lambda o: o["created_at"])
    oid = new_order["order_id"]

    # 支付
    tools2, reply2 = await _collect_reply(agent, f"支付订单 {oid}")
    assert "pay_order" in tools2
    assert "支付" in reply2 or "发货" in reply2
    paid = await order_store.find_by_id(oid)
    assert paid["status"] == "待发货"


# ---------------- HTTP 接口冒烟 ----------------

@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


async def test_health_api_reports_supervisor_model():
    """健康检查上报模式与监督者模型。"""
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_import_app()),
        base_url="http://test",
    ) as client:
        r = await client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["mode"] in {"single_agent", "multi_agent"}
        assert "supervisor_model" in data


async def test_config_api():
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_import_app()),
        base_url="http://test",
    ) as client:
        r = await client.get("/api/config")
        assert r.status_code == 200
        data = r.json()
        assert data["agent_name"]
        assert data["quick_prompts"]


async def test_chat_api_sse():
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_import_app()),
        base_url="http://test",
    ) as client:
        r = await client.post(
            "/api/chat",
            json={"message": "运费怎么算？"},
        )
        assert r.status_code == 200
        body = r.text
        assert "event: meta" in body
        assert "event: delta" in body
        assert "event: done" in body

        # done 事件带上本轮 token 用量（离线模型也会估算上报）
        done = [
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and '"tokens_in"' in line
        ]
        assert done, body[-500:]
        assert done[-1]["tokens_in"] > 0
        assert done[-1]["tokens_out"] > 0


def _chat_streamer():
    from backend.config import get_settings
    from backend.service import ChatStreamer

    return ChatStreamer(get_settings(), "sess_narration")


def _deltas(chunks):
    """取出这批 SSE 片段里的 delta 文本。"""
    return [
        json.loads(c.split("data: ", 1)[1])["text"]
        for c in chunks
        if c.startswith("event: delta")
    ]


def test_narration_before_tool_call_is_dropped():
    """调工具前的过程性旁白（"I'll check…"）不得推给用户，工具轮后正文照常流式。"""
    from agentscope.event import (
        ModelCallEndEvent,
        TextBlockDeltaEvent,
        ToolCallStartEvent,
    )

    st = _chat_streamer()
    narrate = TextBlockDeltaEvent(
        reply_id="r", block_id="b1", delta="I'll check the order status for you.",
    )
    assert _deltas(st._translate(narrate)) == []

    st._translate(ToolCallStartEvent(
        reply_id="r", tool_call_id="c1", tool_call_name="query_order",
    ))
    st._translate(ModelCallEndEvent(reply_id="r", input_tokens=1, output_tokens=1))

    # 工具结果之上的正文立即逐块推送，不再整段缓冲到模型调用结束
    answer = TextBlockDeltaEvent(reply_id="r", block_id="b2", delta="订单已发货")
    assert _deltas(st._translate(answer)) == ["订单已发货"]


def test_short_answer_without_tool_is_flushed_at_model_end():
    """纯闲聊没有工具轮兜底：缓冲在模型调用结束时推送，不能丢。"""
    from agentscope.event import ModelCallEndEvent, TextBlockDeltaEvent

    st = _chat_streamer()
    hello = TextBlockDeltaEvent(reply_id="r", block_id="b1", delta="您好～")
    assert _deltas(st._translate(hello)) == []
    end = ModelCallEndEvent(reply_id="r", input_tokens=1, output_tokens=1)
    assert _deltas(st._translate(end)) == ["您好～"]


def test_long_text_without_tool_starts_streaming():
    """无工具调用时缓冲超限即转流式，避免整段延迟。"""
    from agentscope.event import TextBlockDeltaEvent

    st = _chat_streamer()
    long_delta = TextBlockDeltaEvent(reply_id="r", block_id="b1", delta="啰" * 200)
    assert _deltas(st._translate(long_delta)) == ["啰" * 200]
    more = TextBlockDeltaEvent(reply_id="r", block_id="b1", delta="嗦")
    assert _deltas(st._translate(more)) == ["嗦"]


def test_buffered_text_is_flushed_at_reply_end():
    """兜底：模型调用没正常收尾（如中途报错）时，缓冲里的正文也不能丢。"""
    from agentscope.event import ReplyEndEvent, TextBlockDeltaEvent

    st = _chat_streamer()
    st._translate(TextBlockDeltaEvent(reply_id="r", block_id="b1", delta="您好～"))
    end = st._translate(ReplyEndEvent(reply_id="r", session_id="s", finished_reason="completed"))
    assert _deltas(end) == ["您好～"]


async def test_debug_api_trace_pipeline():
    import httpx

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_import_app()),
        base_url="http://test",
    ) as client:
        # 先跑一轮对话产生 trace（物流查询必走工具调用）
        r = await client.post(
            "/api/chat",
            json={"message": "帮我查一下订单 SO20260810001 的物流"},
        )
        assert r.status_code == 200

        s = await client.get("/api/debug/summary")
        assert s.status_code == 200
        assert s.json()["total"] >= 1

        t = await client.get("/api/debug/traces")
        traces = t.json()["traces"]
        assert traces

        detail = await client.get(f"/api/debug/traces/{traces[0]['id']}")
        assert detail.status_code == 200
        trace = detail.json()["trace"]
        kinds = {sp["kind"] for sp in trace["spans"]}
        assert "tool" in kinds
        assert any(e["name"] == "reply_end" for e in trace["events"])

        # 火焰图：单条与聚合两种模式
        f_one = await client.get(f"/api/debug/flamegraph?trace_id={traces[0]['id']}")
        assert f_one.status_code == 200
        assert f_one.json()["flame"]["children"]
        f_all = await client.get("/api/debug/flamegraph")
        assert f_all.json()["flame"]["children"]

        c = await client.post("/api/debug/clear")
        assert c.status_code == 200
        assert (await client.get("/api/debug/summary")).json()["total"] == 0


def _import_app():
    os.environ["MODEL_PROVIDER"] = "mock"
    import backend.server  # noqa: F401
    return backend.server.app
async def test_multi_turn_coreference_refund():
    """多轮追问：第二轮用指代（不带订单号），应继承上文订单号进入退款链路。"""
    agent = _make_agent("sess_multi_refund")
    tools1, _reply1 = await _collect_reply(agent, "帮我查一下订单 SO20260810001 的物流")
    assert "track_logistics" in tools1

    tools2, _reply2 = await _collect_reply(agent, "帮我退了")
    assert "check_refund_policy" in tools2
    assert "apply_refund" in tools2

    from backend.store.mock_store import order_store

    order = await order_store.find_by_id("SO20260810001")
    assert order["status"] == "退款中"


async def test_multi_turn_ack_confirm_refund():
    """简短确认：上一轮给出「确认退款」的挂起意图，回"好的"应推进退款。"""
    agent = _make_agent("sess_multi_ack")
    tools1, reply1 = await _collect_reply(agent, "SO20260812003 能退吗？")
    assert "check_refund_policy" in tools1
    assert "确认退款" in reply1

    tools2, _reply2 = await _collect_reply(agent, "好的")
    assert "apply_refund" in tools2


async def test_multi_turn_bare_phone_answer():
    """裸数字回答：客服上一轮索要手机号后四位，用户只回四位数字应继续购物车上下文。"""
    agent = _make_agent("sess_multi_phone")
    tools1, reply1 = await _collect_reply(agent, "看看我的购物车")
    assert "view_cart" not in tools1
    assert "手机号" in reply1

    tools2, _reply2 = await _collect_reply(agent, "3721")
    assert "view_cart" in tools2


async def test_multi_turn_inherit_phone_for_logistics():
    """实体继承：手机号在上一轮给出，第二轮直接追问物流不再索要手机号。"""
    agent = _make_agent("sess_multi_inherit")
    tools1, _reply1 = await _collect_reply(agent, "手机号后四位 3721，看下我最近的订单")
    assert "list_recent_orders" in tools1

    tools2, _reply2 = await _collect_reply(agent, "在途那笔到哪儿了？")
    assert "track_logistics" in tools2

async def test_shop_price_filter_under_budget():
    """多条件：让 mock 按价格上限 (500以内) 过滤商品并复述。"""
    agent = _make_agent("sess_price_fit")
    tools, reply = await _collect_reply(agent, "推荐 500 以内的耳机")
    assert "list_products" in tools
    assert "500" in reply
    assert ("P1001" in reply and "以内" in reply) or "暂无" in reply


async def test_shop_price_filter_no_result():
    """预算过低时的兜底：明确告知并推荐最接近的在售款。"""
    agent = _make_agent("sess_price_none")
    tools, reply = await _collect_reply(agent, "推荐 300 以内的耳机")
    assert "list_products" in tools
    assert "300" in reply
    assert "实在" not in reply  # mock 不应假装存在
    assert ("P1001" in reply)  # 推荐最接近的耳机


# ---------------- 上下文策略 ----------------

def test_context_policy_from_settings():
    """上下文策略由配置驱动，而不是依赖 SDK 默认值。"""
    os.environ["MODEL_PROVIDER"] = "mock"
    from backend.config import get_settings
    from backend.agent_factory import _build_context_config

    settings = get_settings()
    cc = _build_context_config(settings)

    assert cc.trigger_ratio == settings.context_trigger_ratio
    assert cc.reserve_ratio == settings.context_reserve_ratio
    assert cc.context_buffer_ratio == settings.context_buffer_ratio
    assert cc.tool_result_limit == settings.tool_result_limit
    # 离线模型无法生成结构化摘要，必须显式打开截断兜底
    assert cc.compression_fallback_to_truncation is True
    # SDK 约束：reserve / buffer 都必须小于 trigger，否则构造 Agent 时报错
    assert cc.reserve_ratio < cc.trigger_ratio
    assert cc.context_buffer_ratio < cc.trigger_ratio
    # 与 SDK 默认值不同，证明确实覆盖了默认策略
    assert cc.trigger_ratio != 0.8 or cc.tool_result_limit != 50000


def test_react_iters_bounded():
    """一轮回复的推理轮次上限由配置控制（SDK 默认 50，客服场景过大）。"""
    agent = _make_agent("sess_react_iters")
    assert agent.react_config.max_iters == 12
    assert agent.react_config.max_iters < 50


async def test_long_conversation_stays_bounded():
    """长对话不会让上下文无限增长：超出窗口后走截断兜底，且仍能正常回复。

    把模型窗口调小以快速触发压缩阈值，否则需要灌入数十 KB 文本。
    """
    os.environ["MODEL_PROVIDER"] = "mock"
    from backend.config import get_settings
    from backend.agent_factory import build_customer_service_agent
    from backend.mock_model import OfflineChatModel

    model = OfflineChatModel()
    model.context_size = 6000  # 人为压小窗口
    agent = build_customer_service_agent(
        get_settings(), "sess_bounded", model=model,
    )

    turns = 8
    peak = 0
    for i in range(turns):
        # _collect_reply 内部已断言 finished_reason == "completed"
        await _collect_reply(agent, f"第{i}轮：帮我查订单 SO20260810001 的物流")
        peak = max(peak, len(agent.state.context))

    # 若无任何上下文策略，8 轮会累积到 16 条消息（实测峰值 6）
    assert peak < 2 * turns
    # 触发过压缩兜底：离线模型无法生成结构化摘要，必然落到截断并留下占位说明
    assert "truncated" in str(agent.state.summary)


# ---------------- 权限与成本控制 ----------------

def test_permission_is_allowlist_not_bypass():
    """权限策略是「显式工具白名单」，而不是 BYPASS 全放行。"""
    from agentscope.permission import PermissionMode

    from backend.agent_factory import build_permission_context
    from backend.tools import tool_names

    ctx = build_permission_context()

    # 不再是 BYPASS —— 那会连安全类 ASK 一起跳过
    assert ctx.mode is not PermissionMode.BYPASS
    # 白名单覆盖全部业务工具，且没有多余项
    assert set(ctx.allow_rules) == set(tool_names())
    assert len(ctx.allow_rules) >= 17


async def test_permission_allows_write_tools_but_not_unknown_ones():
    """写工具按白名单放行；白名单之外的工具不会被静默执行。

    这是本策略的核心性质。曾经用过 BYPASS（全放行，无防护）和裸 DONT_ASK
    （未列白即 DENY，写操作全废）—— 两者都被这一条用例挡住。
    """
    from agentscope.permission import PermissionBehavior, PermissionEngine
    from agentscope.tool import FunctionTool

    from backend.agent_factory import build_permission_context
    from backend.tools import build_toolkit

    ctx = build_permission_context()
    engine = PermissionEngine(ctx)
    tk = build_toolkit()

    # 写类业务工具必须放行
    for name in ("add_to_cart", "apply_refund", "place_order", "pay_order"):
        tool = await tk.get_tool(name)
        decision = await engine.check_permission(tool, {})
        assert decision.behavior is PermissionBehavior.ALLOW, name

    # 白名单之外的函数工具不得被静默放行（回落为 ASK，需用户确认）
    async def _rogue() -> str:
        """一个不在白名单里的工具。"""
        return "should not run silently"

    rogue = FunctionTool(_rogue)
    decision = await engine.check_permission(rogue, {})
    assert decision.behavior is PermissionBehavior.ASK
    assert decision.behavior is not PermissionBehavior.ALLOW


def test_reply_budget_wired_from_settings():
    """单轮 token 预算取自配置，且中间件已挂到 Agent 上。"""
    from agentscope.middleware import ReplyBudgetControlMiddleware

    os.environ["MODEL_PROVIDER"] = "mock"
    from backend.config import get_settings

    agent = _make_agent("sess_budget_wired")
    settings = get_settings()

    # 中间件按实现的钩子被分发到不同的内部列表，这里去重后统一收集
    found = {
        id(m): m
        for attr in dir(agent)
        if attr.endswith("_middlewares")
        for m in getattr(agent, attr)
        if isinstance(m, ReplyBudgetControlMiddleware)
    }
    assert len(found) == 1
    budget = next(iter(found.values()))
    assert budget.token_budget == settings.reply_token_budget
    assert budget.token_budget > 0


async def test_offline_model_reports_usage_so_budget_can_bite():
    """离线模型必须上报 token 用量，否则预算中间件在无 Key 环境下形同虚设。"""
    from agentscope.event import ModelCallEndEvent
    from agentscope.message import TextBlock, UserMsg

    from backend.mock_model import OfflineChatModel

    agent = _make_agent("sess_usage")
    assert isinstance(agent.model, OfflineChatModel)

    reported = []
    async for evt in agent.reply_stream(
        UserMsg(name="用户", content=[TextBlock(text="帮我查订单 SO20260810001 的物流")]),
    ):
        if isinstance(evt, ModelCallEndEvent):
            reported.append((evt.input_tokens, evt.output_tokens))

    assert reported, "未观察到 ModelCallEndEvent"
    assert all(i > 0 and o > 0 for i, o in reported), reported


async def test_reply_budget_stops_runaway_loop():
    """预算超限时注入收尾提示并累计用量到 middle_context。"""
    from agentscope.agent import Agent, ModelConfig, ReActConfig
    from agentscope.middleware import ReplyBudgetControlMiddleware
    from agentscope.state import AgentState

    from backend.agent_factory import build_permission_context
    from backend.config import get_settings
    from backend.mock_model import OfflineChatModel
    from backend.prompts import build_system_prompt
    from backend.tools import build_toolkit

    settings = get_settings()

    def _agent(budget: int) -> Agent:
        return Agent(
            name=settings.agent_name,
            system_prompt=build_system_prompt(settings.agent_name, settings.brand_name),
            model=OfflineChatModel(),
            toolkit=build_toolkit(),
            state=AgentState(
                session_id="sess_runaway", context=[],
                permission_context=build_permission_context(),
            ),
            middlewares=[ReplyBudgetControlMiddleware(token_budget=budget)],
            model_config=ModelConfig(max_retries=2),
            react_config=ReActConfig(max_iters=settings.max_react_iters),
        )

    def _hint_count(agent: Agent) -> int:
        return sum(
            len(msg.get_content_blocks("hint")) for msg in agent.state.context
        )

    question = "手机号后四位 3721，看下我最近的订单"

    # 预算充足：不注入收尾提示
    generous = _agent(settings.reply_token_budget)
    await _collect_reply(generous, question)
    baseline = _hint_count(generous)

    # 预算极小：必须多注入一次收尾提示，并记录累计用量
    tight = _agent(1)
    await _collect_reply(tight, question)
    assert _hint_count(tight) > baseline

    records = tight.state.middle_context.get("ReplyBudgetControlMiddleware") or {}
    assert records, tight.state.middle_context
    assert all(cost > 0 for cost in records.values())


# ---------------- 上下文实体继承层 ----------------

def test_extract_entities_pulls_order_and_phone():
    """本轮抽取：订单号里的数字不得被误读成手机号后四位。"""
    from backend.context import extract_entities

    ex = extract_entities("订单 SO20260810001 要退款，手机号后四位 3721")
    assert ex["order_id"] == "SO20260810001"
    assert ex["phone_tail"] == "3721"
    assert ex["reason"] == "用户申请退款"

    # 只给订单号时不应凭空造出手机号（SO20260810001 内含 8 位数字）
    only_order = extract_entities("查一下 SO20260810001")
    assert only_order["order_id"] == "SO20260810001"
    assert only_order["phone_tail"] == ""


def test_history_entities_ignores_tool_result_numbers():
    """回归：快递员脱敏手机号 / 积分这些**工具结果里的数字**不得被继承为用户手机号。

    修复前 ``_history_entities`` 把工具结果原文也纳入手机号扫描，且不剔除
    ``138****6621`` 这类脱敏串，于是 6621 被当成用户尾号；积分 2680 同理。
    这里直接对脱敏串与工具结果做单元级断言。
    """
    from agentscope.message import (
        AssistantMsg,
        TextBlock,
        ToolCallBlock,
        ToolResultBlock,
        UserMsg,
    )

    from backend.context import history_entities

    call = ToolCallBlock(
        id="c1", name="track_logistics", input='{"order_id": "SO20260810001"}',
    )
    result = ToolResultBlock(
        id="c1", name="track_logistics",
        output=[TextBlock(text='{"company": "火山速运", "courier": "赵师傅 138****6621"}')],
    )
    msgs = [
        AssistantMsg(name="客服", content=[call, result]),
        # 末尾必须有 user 消息，否则 boundary 落在列表末尾、什么都扫不到
        UserMsg(name="用户", content=[TextBlock(text="那物流呢")]),
    ]

    ents = history_entities(msgs)
    assert ents["phone_tail"] != "6621", ents
    assert ents["phone_tail"] == "", ents
    assert ents["order_id"] == "SO20260810001"   # 订单号仍应正常继承


async def test_regression_courier_phone_not_used_as_user_tail():
    """回归（端到端）：查完物流再追问，不得拿快递员尾号去查库。"""
    from backend.context import history_entities

    agent = _make_agent("sess_reg_courier")
    tools1, _ = await _collect_reply(agent, "帮我查一下订单 SO20260810001 的物流")
    assert "track_logistics" in tools1

    tools2, reply2 = await _collect_reply(agent, "那物流呢")
    # 症状：修复前这里会去查"手机号后四位 6621"，回复以「未找到」开头
    assert "未找到" not in reply2, reply2[:200]
    assert "6621" not in history_entities(agent.state.context)["phone_tail"]
    # 仍应正常拿到物流
    assert "track_logistics" in tools2


async def test_regression_points_not_used_as_user_tail():
    """回归（端到端）：查完会员积分再追问，不得拿积分值当手机号后四位。"""
    from backend.context import history_entities

    agent = _make_agent("sess_reg_points")
    tools1, _ = await _collect_reply(agent, "手机号后四位 3721，我的会员等级")
    assert "query_user_profile" in tools1

    tools2, reply2 = await _collect_reply(agent, "那物流呢")
    assert "未找到" not in reply2, reply2[:200]

    ents = history_entities(agent.state.context)
    assert ents["phone_tail"] == "3721", ents   # 用户自己报的尾号，而不是积分 2680


async def test_new_order_id_keeps_date_plus_sequence_format():
    """回归：新订单号必须是 SO+YYYYMMDD+3 位序号，不能把旧单号整串拼进来。"""
    from backend.store.mock_store import order_store

    order = await order_store.create(
        user_id="U10001", phone_tail="3721", items=[],
        amount=1.0, address="测试地址",
    )
    # 修复前会生成 SO2026092820260910012 这种 20 位畸形串
    assert re.fullmatch(r"SO\d{8}\d{3}", order["order_id"]), order["order_id"]


# ---------------- 跨会话用户记忆 ----------------

async def test_cross_session_memory():
    """跨会话记忆：换一个新会话，凭同一 user_id 仍能认出回头客。

    这是「记忆」区别于「会话历史」的关键 —— 会话是新的，Agent 实例是新的，
    但用户档案被长期记忆带了过来。
    """
    os.environ["MODEL_PROVIDER"] = "mock"
    from backend.config import get_settings
    from backend.agent_factory import build_customer_service_agent
    from backend.mock_model import OfflineChatModel
    from backend.store.memory_store import user_memory_store

    settings = get_settings()

    # 会话 A：登录态 U10001，问出会员档案
    agent_a = build_customer_service_agent(
        settings, "sess_mem_a", model=OfflineChatModel(), user_id="U10001",
    )
    tools, _reply = await _collect_reply(agent_a, "手机号后四位 3721，查下我的会员等级和积分")
    assert "query_user_profile" in tools

    # 事实被确定性抽取并落盘（不依赖模型决策，故离线模型下同样生效）
    record = await user_memory_store.get("3721")
    assert record is not None
    assert record["nickname"] == "追风的云"
    assert record["member_level"] == "黄金会员"
    assert record["user_id"] == "U10001"

    # 会话 B：全新会话 + 全新 Agent + 全新上下文，仅凭 user_id
    agent_b = build_customer_service_agent(
        settings, "sess_mem_b", model=OfflineChatModel(), user_id="U10001",
    )
    assert agent_b.state.context == []          # 上下文确实是空的

    prompt = await agent_b._get_system_prompt()  # noqa: SLF001 - 断言注入结果的最直接方式
    assert "已知用户档案" in prompt
    assert "追风的云" in prompt
    assert "黄金会员" in prompt
    # 记忆只用于个性化，业务数据仍须以工具为准
    assert "必须调用工具核实" in prompt


async def test_memory_learns_identity_without_login():
    """匿名会话：没有 user_id 时，身份从对话中问到的手机号后四位建立。"""
    os.environ["MODEL_PROVIDER"] = "mock"
    from backend.config import get_settings
    from backend.agent_factory import build_customer_service_agent
    from backend.mock_model import OfflineChatModel
    from backend.store.memory_store import user_memory_store

    agent = build_customer_service_agent(
        get_settings(), "sess_mem_anon", model=OfflineChatModel(),
    )
    tools, _reply = await _collect_reply(agent, "手机号后四位 3721，查下我的会员等级和积分")
    assert "query_user_profile" in tools

    record = await user_memory_store.get("3721")
    assert record is not None
    assert record["nickname"] == "追风的云"
    assert not record.get("user_id")            # 匿名会话不写 user_id


# ---------------- 会话状态持久化 ----------------

def test_session_id_rejects_path_traversal():
    """session_id 来自客户端且会拼进文件路径，必须拒绝穿越字符。"""
    from backend.store.session_store import is_safe_session_id

    assert is_safe_session_id("sess_01ff30695a91")
    assert not is_safe_session_id("../../etc/passwd")
    assert not is_safe_session_id("a/b")
    assert not is_safe_session_id("a\\b")
    assert not is_safe_session_id("..")
    assert not is_safe_session_id("")
    assert not is_safe_session_id("x" * 65)


async def test_session_state_persists_across_restart(tmp_path):
    """会话状态持久化：模拟进程重启后仍能恢复上下文并继续对话。"""
    os.environ["MODEL_PROVIDER"] = "mock"
    from backend.config import get_settings
    from backend.session_manager import SessionManager
    from backend.store.session_store import SessionStore
    from backend.mock_model import OfflineChatModel

    settings = get_settings()
    store = SessionStore(directory=tmp_path / "sessions")   # 不污染演示数据

    def _factory(session_id, state=None, user_id=""):
        from backend.agent_factory import build_customer_service_agent

        return build_customer_service_agent(
            settings, session_id, state=state,
            model=OfflineChatModel(), user_id=user_id,
        )

    # ---- 进程 1 ----
    m1 = SessionManager(settings, _factory, session_store=store)
    rt1 = await m1.create(user_id="U10001")
    sid = rt1.session_id
    await _collect_reply(rt1.agent, "帮我查订单 SO20260810001 的物流")
    assert await m1.persist(rt1) is True
    turns_before = len(rt1.agent.state.context)
    assert turns_before > 0

    # ---- 进程 2：全新 manager，内存中不含该会话 ----
    m2 = SessionManager(settings, _factory, session_store=store)
    assert m2.get(sid) is None

    rt2 = await m2.get_or_create(sid)
    assert len(rt2.agent.state.context) == turns_before   # 上下文完整恢复
    assert rt2.user_id == "U10001"                        # 身份一并恢复

    # 恢复后可直接继续对话（_collect_reply 内部断言 finished_reason == completed）
    tools, _reply = await _collect_reply(rt2.agent, "那物流呢")
    assert tools

    # 未知 session_id 不应伪造会话，而是新建
    rt3 = await m2.get_or_create("sess_ffffffffffff")
    assert rt3.agent.state.context == []


# ---------------- Agentic RAG：查询改写 / 精排降级 / 引用溯源 / 置信度 ----------------
# 全程离线（conftest 已关 EMBED_PROVIDER 与 RERANK_PROVIDER）。

def test_query_rewrite_expands_ecommerce_synonyms():
    """口语应被扩展出规范词查询（买贵了→保价/价格保护），且关掉开关时只用原句。"""
    from backend.rag.query_rewrite import expand_queries

    variants = expand_queries("买贵了能退差价吗", enabled=True)
    assert variants[0] == "买贵了能退差价吗"          # 原句恒在首位
    assert len(variants) > 1                          # 确实产生了扩展
    assert any("保价" in v or "价格保护" in v for v in variants)

    # 关闭扩展：只返回原句，行为退回旧实现
    assert expand_queries("买贵了能退差价吗", enabled=False) == ["买贵了能退差价吗"]


async def test_retriever_offline_returns_dict_with_citations():
    """降级（词面）路径下检索返回结构化 dict：results 带 snippet/source_id，附 confidence/queries。"""
    from backend.config import get_settings
    from backend.rag.retriever import KnowledgeBase
    from backend.store.mock_store import faq_store

    settings = get_settings()
    kb = KnowledgeBase(settings)
    entries = await faq_store.all()
    out = await kb.search("退货规则是什么？", entries)

    assert isinstance(out, dict)
    assert out["confidence"] == "high"                # 降级路径不误判低置信
    assert out["queries"] and out["queries"][0] == "退货规则是什么？"
    assert out["results"], "退货规则应命中 FAQ"
    top = out["results"][0]
    assert "七天" in top["title"]
    assert top["source_id"] and top["snippet"]        # 引用溯源字段齐备
    assert top["retrieval"] == "lexical"              # 离线为纯词面模式
    assert top["rerank"] is None                       # 未精排


async def test_search_faq_tool_exposes_citations():
    """工具层把命中整理成 citations，并保持 found/results 向后兼容。"""
    from backend.tools.knowledge import search_faq

    data = await search_faq("开发票要怎么申请？")
    assert data["found"] is True
    assert "results" in data and "citations" in data
    assert data["confidence"] == "high"
    cite = data["citations"][0]
    assert cite["id"] and cite["title"] and "snippet" in cite


async def test_search_faq_no_hit_falls_back_to_message():
    """完全无关的问题检索不到，退回 found=False 的转人工提示（旧语义不变）。"""
    from backend.tools.knowledge import search_faq

    data = await search_faq("火星移民指南")
    assert data["found"] is False
    assert "message" in data and "转人工" in data["message"]


async def test_reranker_degrades_when_unconfigured():
    """无 Key / provider=none 时精排不可用，rerank 返回 None 交检索层保持原序。"""
    from backend.config import get_settings
    from backend.rag.reranker import Reranker

    rr = Reranker(get_settings())
    assert rr.available is False                       # conftest 已关精排
    assert await rr.rerank("退货规则", ["七天无理由退货", "运费规则"]) is None
