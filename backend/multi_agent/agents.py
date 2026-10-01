# -*- coding: utf-8 -*-
"""监督者 + 专家子 agent 的多智能体装配（AgentScope 2.0 的 agent-as-tool 模式）。

专家子 agent 被封装成工具暴露给监督者；监督者在 ReAct 循环里"以调用的方式委派"，
每次委派临时新建一个只带自身工具子集、独立上下文的子 agent，跑完把结论回填给监督者。
"""
from __future__ import annotations

from agentscope.agent import (
    Agent,
    InjectionConfig,
    ModelConfig,
    ReActConfig,
)
from agentscope.event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    TextBlockDeltaEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.message import TextBlock, UserMsg
from agentscope.middleware import ReplyBudgetControlMiddleware
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit

from ..agent_factory import _build_context_config, build_permission_context
from ..config import Settings
from ..middleware import CustomerMemoryMiddleware
from ..models import create_chat_model, create_supervisor_model
from ..narration import NarrationFilter
from ..tools import build_toolkit
from .passthrough import SingleDelegationPassthroughModel
from .prompts import build_specialist_prompt, build_supervisor_prompt
from .tool_trace import collect as collect_tool_result


# 专家子 agent 规格：委派工具名、角色标签、给监督者看的说明、各自装配的工具子集
SPECIALIST_SPECS: list[dict] = [
    {
        "tool_name": "delegate_to_knowledge_agent",
        "role": "知识检索专家",
        "read_only": True,
        "description": (
            "电商政策/规则问答专家。用于运费、发货时效、退货退款规则、发票、"
            "会员积分、保价、地址修改规则等需要官方口径的问题。"
        ),
        "tools": ["search_faq"],
    },
    {
        "tool_name": "delegate_to_logistics_agent",
        "role": "物流查询专家",
        "read_only": True,
        "description": (
            "订单与物流查询专家。用于查询订单状态、物流轨迹，"
            "或按手机号后四位列举近期订单。"
        ),
        "tools": ["query_order", "list_recent_orders", "track_logistics"],
    },
    {
        "tool_name": "delegate_to_refund_agent",
        "role": "退款售后专家",
        "read_only": False,
        "description": (
            "退款与售后处置专家。用于判断退款资格、执行退款/取消订单、"
            "创建售后工单、用户明确要求转人工，以及需要转人工的复杂纠纷。"
            "委派执行类任务时，task 必须已写明用户已确认。"
        ),
        "tools": [
            "query_order", "list_recent_orders", "check_refund_policy",
            "apply_refund", "cancel_order", "create_ticket", "transfer_to_human",
        ],
    },
    {
        "tool_name": "delegate_to_shopping_agent",
        "role": "导购交易专家",
        "read_only": False,
        "description": (
            "商品导购与交易专家。用于浏览/搜索商品、查询商品详情与价格、"
            "查看或修改购物车、下单、支付，以及查询会员等级与积分。"
        ),
        "tools": [
            "list_products", "query_product", "view_cart", "add_to_cart",
            "update_cart_item", "place_order", "pay_order", "query_user_profile",
        ],
    },
]


def build_specialist_agent(settings: Settings, spec: dict, session_id: str) -> Agent:
    """按规格新建一个只带自身工具子集、独立权限与上下文的专家子 agent。"""
    return Agent(
        name=spec["role"],
        system_prompt=build_specialist_prompt(spec["role"], settings.brand_name),
        model=create_chat_model(settings),
        toolkit=build_toolkit(spec["tools"]),
        state=AgentState(
            session_id=session_id,
            context=[],
            permission_context=build_permission_context(),
        ),
        model_config=ModelConfig(max_retries=2),
        context_config=_build_context_config(settings),
        react_config=ReActConfig(max_iters=settings.specialist_max_iters),
        injection_config=InjectionConfig(timezone="Asia/Shanghai"),
    )


async def run_specialist(settings: Settings, spec: dict, task: str, session_id: str) -> dict:
    """驱动一个专家子 agent 完成单个委派任务，收集其最终文本与用到的工具。

    只收集工具轮之后的正文——调工具前的旁白不是业务结论（规则见 narration 模块）。
    """
    agent = build_specialist_agent(settings, spec, session_id)
    answer = ""
    tools_used: list[str] = []
    handed_off = False
    call_names: dict[str, str] = {}
    result_buf: dict[str, str] = {}
    # 旁白抑制：调工具前的文字丢弃，工具轮之后的才是结论（规则见 narration 模块）
    narration = NarrationFilter(buffer_limit=None)
    try:
        async for evt in agent.reply_stream(
            UserMsg(name="用户", content=[TextBlock(text=task)]),
        ):
            if isinstance(evt, TextBlockDeltaEvent):
                answer += narration.on_text(evt.delta)
            elif isinstance(evt, ModelCallEndEvent):
                answer += narration.on_model_end()
            elif isinstance(evt, ReplyEndEvent):
                answer += narration.on_reply_end()
            elif isinstance(evt, ToolCallStartEvent):
                narration.on_tool_call()
                call_names[evt.tool_call_id] = evt.tool_call_name
                if evt.tool_call_name not in tools_used:
                    tools_used.append(evt.tool_call_name)
                if evt.tool_call_name == "transfer_to_human":
                    handed_off = True
            elif isinstance(evt, ToolResultTextDeltaEvent):
                result_buf[evt.tool_call_id] = (
                    result_buf.get(evt.tool_call_id, "") + evt.delta
                )
            elif isinstance(evt, ToolResultEndEvent):
                # 旁路留一份真实返回：评测裁判要拿它核对结论有没有依据
                collect_tool_result(
                    call_names.get(evt.tool_call_id, evt.tool_call_id),
                    result_buf.get(evt.tool_call_id, ""),
                )
    except Exception as exc:  # noqa: BLE001
        return {
            "specialist": spec["role"],
            "answer": answer.strip(),
            "tools_used": tools_used,
            "handed_off": handed_off,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "specialist": spec["role"],
        "answer": answer.strip(),
        "tools_used": tools_used,
        "handed_off": handed_off,
    }


def make_delegate_tool(settings: Settings, spec: dict, parent_session_id: str) -> FunctionTool:
    """把一个专家子 agent 包装成监督者可调用的委派工具（agent-as-tool）。"""
    async def delegate(task: str) -> dict:
        """把自包含的任务转交给该专家子 agent 处理并返回其结论。

        Args:
            task: 完整、自包含的任务描述，须包含订单号 / 手机号后四位 / 用户已确认的意愿等关键信息（子 agent 看不到原始对话）。
        """
        sub_session = f"{parent_session_id}::{spec['tool_name']}"
        return await run_specialist(settings, spec, task, sub_session)

    return FunctionTool(
        func=delegate,
        name=spec["tool_name"],
        description=spec["description"],
        is_read_only=spec["read_only"],
    )


def build_supervisor_agent(
    settings: Settings,
    session_id: str,
    state: AgentState | None = None,
    model=None,
    user_id: str = "",
) -> Agent:
    """构造前台监督者：只持有委派工具，业务与政策一律委派给专家子 agent。

    model 显式传入时（测试/评测注入）用它，否则用 create_supervisor_model；
    再按开关套一层终答透传包装。
    """
    delegate_names = [spec["tool_name"] for spec in SPECIALIST_SPECS]
    if state is None:
        state = AgentState(
            session_id=session_id,
            context=[],
            permission_context=build_permission_context(extra_allow=delegate_names),
        )
    toolkit = Toolkit(
        tools=[
            make_delegate_tool(settings, spec, session_id)
            for spec in SPECIALIST_SPECS
        ],
    )
    supervisor_model = model or create_supervisor_model(settings)
    if settings.passthrough_single_delegation:
        supervisor_model = SingleDelegationPassthroughModel(supervisor_model)
    return Agent(
        name=settings.agent_name,
        system_prompt=build_supervisor_prompt(settings.agent_name, settings.brand_name),
        model=supervisor_model,
        toolkit=toolkit,
        state=state,
        middlewares=[
            CustomerMemoryMiddleware(user_id=user_id),
            ReplyBudgetControlMiddleware(
                token_budget=settings.reply_token_budget,
                output_token_weight=settings.reply_output_token_weight,
            ),
        ],
        model_config=ModelConfig(max_retries=2),
        context_config=_build_context_config(settings),
        react_config=ReActConfig(max_iters=settings.max_react_iters),
        injection_config=InjectionConfig(timezone="Asia/Shanghai"),
    )
