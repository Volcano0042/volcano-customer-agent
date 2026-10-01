# -*- coding: utf-8 -*-
"""多智能体（监督者 + 专家子 agent）的离线单测。

全程不联网、不触发真实模型调用：只验证工具子集装配、委派工具形状、
权限白名单放行、监督者可构建，以及评测驱动里对子 agent 结果的解析逻辑。
"""
import json
from dataclasses import replace

import pytest
from agentscope.message import (
    Msg,
    TextBlock,
    ToolResultBlock,
    ToolResultState,
    UserMsg,
)
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.tool import FunctionTool

from backend.agent_factory import build_agent, build_permission_context
from backend.config import get_settings
from backend.eval.driver import Transcript, _specialist_inner_tools
from backend.models import create_chat_model, create_supervisor_model
from backend.multi_agent import (
    SPECIALIST_SPECS,
    SingleDelegationPassthroughModel,
    build_supervisor_agent,
    make_delegate_tool,
    run_specialist,
    single_delegation_answer,
)
from backend.multi_agent.prompts import build_specialist_prompt, build_supervisor_prompt
from backend.tools import build_toolkit, tool_names


# ---------------- 工具子集装配 ----------------
async def test_build_toolkit_filter_subset():
    tk = build_toolkit(["search_faq"])
    schemas = await tk.get_tool_schemas()
    assert len(schemas) == 1


def test_build_toolkit_filter_unknown_raises():
    with pytest.raises(ValueError):
        build_toolkit(["no_such_tool_xyz"])


# ---------------- 专家规格与安全边界 ----------------
def test_specialist_specs_tools_all_exist():
    known = set(tool_names())
    for spec in SPECIALIST_SPECS:
        for name in spec["tools"]:
            assert name in known, f"{spec['tool_name']} 引用了不存在的工具 {name}"


def test_write_tools_only_in_writable_specs():
    """危险写工具（退款/取消/下单/支付）只能出现在标为非只读的专家规格里。"""
    danger = {"apply_refund", "cancel_order", "pay_order", "place_order"}
    for spec in SPECIALIST_SPECS:
        if spec["read_only"]:
            assert not (danger & set(spec["tools"])), f"{spec['role']} 标只读却含写工具"


def test_every_business_tool_reachable_from_some_specialist():
    """回归：每个业务工具都必须有专家持有，否则多 agent 模式下不可达。"""
    reachable = {name for spec in SPECIALIST_SPECS for name in spec["tools"]}
    missing = sorted(set(tool_names()) - reachable)
    assert not missing, f"以下工具在多 agent 模式下没有任何专家可达: {missing}"


# ---------------- 委派工具形状 ----------------
def test_make_delegate_tool_shape():
    settings = get_settings()
    spec = SPECIALIST_SPECS[0]  # 知识检索专家，只读
    tool = make_delegate_tool(settings, spec, "sess-1")
    assert isinstance(tool, FunctionTool)
    assert tool.name == spec["tool_name"]
    assert tool.description == spec["description"]


# ---------------- 监督者可离线构建 ----------------
def test_build_supervisor_agent_constructs():
    settings = get_settings()
    agent = build_supervisor_agent(settings, session_id="test-sup")
    assert agent is not None


def test_build_agent_dispatch_single_by_default():
    """multi_agent 显式 False 一定走单 agent，不受全局开关影响。"""
    settings = get_settings()
    agent = build_agent(settings, session_id="test-single", multi_agent=False)
    assert agent is not None


def test_build_agent_dispatch_multi_when_flag_true():
    settings = get_settings()
    agent = build_agent(settings, session_id="test-multi", multi_agent=True)
    assert agent is not None


# ---------------- 权限白名单放行委派工具 ----------------
def test_permission_context_extra_allow():
    ctx = build_permission_context(extra_allow=["delegate_to_knowledge_agent"])
    assert "delegate_to_knowledge_agent" in ctx.allow_rules
    # 原有业务工具仍在白名单里
    for name in tool_names():
        assert name in ctx.allow_rules


# ---------------- 评测驱动对子 agent 结果的解析 ----------------
def test_specialist_inner_tools_parses_list():
    payload = json.dumps({"specialist": "物流查询专家", "answer": "在途", "tools_used": ["query_order", "track_logistics"]})
    assert _specialist_inner_tools(payload) == ["query_order", "track_logistics"]


def test_specialist_inner_tools_tolerates_junk():
    assert _specialist_inner_tools("") == []
    assert _specialist_inner_tools("not json") == []
    assert _specialist_inner_tools(json.dumps({"answer": "no tools key"})) == []


def test_transcript_dict_has_no_mode_field():
    """评测只剩多 agent 一条链路，轨迹里不再需要 mode 区分。"""
    assert "mode" not in Transcript(case_id="c1").to_dict()


# ---------------- 专家旁白不得混进结论 ----------------
async def test_run_specialist_drops_pre_tool_narration(monkeypatch):
    """专家调工具前的过程性旁白不能进 answer——监督者会把它当结论转述。"""
    from agentscope.event import (
        ModelCallEndEvent,
        ReplyEndEvent,
        TextBlockDeltaEvent,
        ToolCallStartEvent,
    )

    import backend.multi_agent.agents as agents_mod

    class FakeSpecialist:
        async def reply_stream(self, msg):
            yield TextBlockDeltaEvent(
                reply_id="r", block_id="b1",
                delta="I'll check the order details for you. ",
            )
            yield ToolCallStartEvent(
                reply_id="r", tool_call_id="c1", tool_call_name="query_order",
            )
            yield ModelCallEndEvent(reply_id="r", input_tokens=1, output_tokens=1)
            yield TextBlockDeltaEvent(
                reply_id="r", block_id="b2", delta="订单已发货，预计明天送达。",
            )
            yield ModelCallEndEvent(reply_id="r", input_tokens=1, output_tokens=1)
            yield ReplyEndEvent(
                reply_id="r", session_id="s", finished_reason="completed",
            )

    monkeypatch.setattr(agents_mod, "build_specialist_agent", lambda *a, **k: FakeSpecialist())
    result = await run_specialist(
        get_settings(), SPECIALIST_SPECS[1], "查订单 SO20260810001 的物流", "sess-x",
    )
    assert result["answer"] == "订单已发货，预计明天送达。"
    assert result["tools_used"] == ["query_order"]


async def test_run_specialist_keeps_answer_when_stream_ends_abruptly(monkeypatch):
    """模型调用没正常收尾（缺 ModelCallEnd）时，ReplyEnd 兜底不能把正文丢掉。"""
    from agentscope.event import ReplyEndEvent, TextBlockDeltaEvent

    import backend.multi_agent.agents as agents_mod

    class TruncatedSpecialist:
        async def reply_stream(self, msg):
            yield TextBlockDeltaEvent(reply_id="r", block_id="b1", delta="运费满 99 元包邮。")
            yield ReplyEndEvent(reply_id="r", session_id="s", finished_reason="completed")

    monkeypatch.setattr(
        agents_mod, "build_specialist_agent", lambda *a, **k: TruncatedSpecialist(),
    )
    result = await run_specialist(
        get_settings(), SPECIALIST_SPECS[0], "运费怎么算", "sess-y",
    )
    assert result["answer"] == "运费满 99 元包邮。"


# ---------------- 模型实例复用 ----------------
def test_create_chat_model_reuses_same_instance():
    settings = get_settings()
    assert create_chat_model(settings) is create_chat_model(settings)


def test_create_chat_model_separates_by_model_name():
    settings = get_settings()
    other = replace(settings, model_name="another-model-x")
    assert create_chat_model(other) is not create_chat_model(settings)


# ---------------- 监督者模型选择 ----------------
def test_supervisor_model_falls_back_to_main_when_unset():
    # 显式清空，不依赖本机 .env
    settings = replace(get_settings(), supervisor_model="")
    model = create_supervisor_model(settings)
    assert getattr(model, "primary", model) is create_chat_model(settings)


def test_supervisor_model_uses_configured_fast_model():
    settings = replace(get_settings(), supervisor_model="fast-model-x")
    if settings.model_provider == "mock":
        pytest.skip("离线 mock 模型没有真实模型名")
    model = create_supervisor_model(settings)
    assert getattr(model, "primary", model).model == "fast-model-x"


def test_supervisor_fast_model_degrades_to_main_model():
    settings = replace(
        get_settings(), supervisor_model="fast-model-x", degradation_enabled=True,
    )
    if settings.model_provider == "mock":
        pytest.skip("离线 mock 模型没有真实模型名")
    model = create_supervisor_model(settings)
    # 快速模型失败时降级回主模型
    assert model.primary.model == "fast-model-x"
    assert model.fallback is create_chat_model(settings)


# ---------------- 终答透传 ----------------
def _delegate_msg(*results: tuple[str, dict]) -> Msg:
    """造一条本轮工具结果 assistant 消息（形状同 AgentScope 写进上下文的）。"""
    return Msg(
        name="小V",
        role="assistant",
        content=[
            ToolResultBlock(
                id=f"call_{i}",
                name=name,
                output=[TextBlock(text=json.dumps(payload, ensure_ascii=False))],
                state=ToolResultState.SUCCESS,
            )
            for i, (name, payload) in enumerate(results)
        ],
    )


def test_single_delegation_answer_extracts_when_exactly_one():
    msg = _delegate_msg((
        "delegate_to_knowledge_agent",
        {"specialist": "知识检索专家", "answer": "满 99 元包邮。", "tools_used": []},
    ))
    messages = [UserMsg(name="用户", content=[TextBlock(text="运费怎么算")]), msg]
    assert single_delegation_answer(messages) == "满 99 元包邮。"


def test_single_delegation_answer_skips_cross_domain_delegation():
    """委派了多个专家时不透传，交回监督者综合。"""
    msg = _delegate_msg(
        ("delegate_to_knowledge_agent", {"answer": "满 99 包邮"}),
        ("delegate_to_logistics_agent", {"answer": "已发货"}),
    )
    assert single_delegation_answer([msg]) is None


def test_single_delegation_answer_rejects_unusable_results():
    cases = [
        [],                                                      # 空上下文
        [UserMsg(name="用户", content=[TextBlock(text="你好")])],  # 最后一条不是工具结果
        [_delegate_msg(("delegate_to_refund_agent", {"answer": "   "}))],          # 空答案
        [_delegate_msg(("delegate_to_refund_agent", {
            "answer": "x", "error": "TimeoutError",
        }))],
        [_delegate_msg(("delegate_to_refund_agent", {
            "answer": "已转人工", "handed_off": True,
        }))],
        [Msg(name="小V", role="assistant", content=[
            ToolResultBlock(id="c", name="delegate_to_refund_agent",
                            output="not json", state=ToolResultState.SUCCESS),
        ])],
    ]
    for messages in cases:
        assert single_delegation_answer(messages) is None


class _ExplodingModel(ChatModelBase):
    """一被调用就报错，用于证明透传命中时没发出模型调用。"""

    def __init__(self) -> None:
        super().__init__(
            credential=None, model="exploding", parameters=None, stream=True,
        )

    async def _call_api(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("透传命中时不应调用内层模型")


class _EchoModel(ChatModelBase):
    """固定回复的内层模型，用于验证未命中透传时正常放行。"""

    def __init__(self) -> None:
        super().__init__(credential=None, model="echo", parameters=None, stream=True)

    async def _call_api(self, model_name, messages=None, tools=None, tool_choice=None, **kwargs):
        return ChatResponse(content=[TextBlock(text="来自内层模型")], is_last=True)


async def test_passthrough_model_answers_without_model_call():
    model = SingleDelegationPassthroughModel(_ExplodingModel())
    resp = await model(messages=[_delegate_msg((
        "delegate_to_logistics_agent", {"answer": "已发货，预计明天送达。"},
    ))])
    assert isinstance(resp, ChatResponse)
    assert resp.content[0].text == "已发货，预计明天送达。"
    assert model.passthrough_count == 1


async def test_passthrough_model_forwards_when_not_single():
    model = SingleDelegationPassthroughModel(_EchoModel())
    resp = await model(messages=[UserMsg(name="用户", content=[TextBlock(text="你好")])])
    assert resp.content[0].text == "来自内层模型"
    assert model.passthrough_count == 0


def test_supervisor_wraps_model_for_passthrough():
    agent = build_supervisor_agent(get_settings(), session_id="test-wrap")
    assert isinstance(agent.model, SingleDelegationPassthroughModel)


def test_supervisor_passthrough_can_be_switched_off():
    settings = replace(get_settings(), passthrough_single_delegation=False)
    agent = build_supervisor_agent(settings, session_id="test-wrap-off")
    assert not isinstance(agent.model, SingleDelegationPassthroughModel)


def test_supervisor_prompt_tells_it_not_to_rewrite():
    text = build_supervisor_prompt("小V", "Volcano")
    assert "直接沿用专家给出的正文" in text


def test_supervisor_prompt_routes_direct_transfer_request():
    """回归：用户直接要求转人工时，监督者必须委派（它只持有 delegate_to_*）。"""
    text = build_supervisor_prompt("小V", "Volcano")
    assert "用户明确要求转人工" in text
    assert "必须委派退款售后专家" in text
    # 路由表里「转人工」归在退款售后专家名下，监督者才知道该委派给谁
    route = next(l for l in text.splitlines() if l.startswith("- 退款/售后处置"))
    assert "delegate_to_refund_agent" in route and "转人工" in route


def test_supervisor_prompt_requires_lookup_before_confirming():
    """回归：写操作要确认，但确认之前得先把只读查询委派出去，不能凭空发问。"""
    text = build_supervisor_prompt("小V", "Volcano")
    assert "澄清之前要先把只读查询委派出去" in text


def test_refund_specialist_owns_transfer_and_is_told_to_use_it():
    """退款售后专家是 transfer_to_human 的唯一持有者，其角色说明须明确要用它。"""
    spec = next(s for s in SPECIALIST_SPECS if s["tool_name"] == "delegate_to_refund_agent")
    assert "transfer_to_human" in spec["tools"]
    assert "直接 transfer_to_human" in build_specialist_prompt(spec["role"], "Volcano")
