# -*- coding: utf-8 -*-
"""基于 AgentScope 2.0 SDK 层构建客服智能体。

编排层次对应 AgentScope 2.0 架构：
- Building blocks: Agent（ReAct 式推理-行动循环）
- Model: DashScopeChatModel / OpenAIChatModel / OfflineChatModel
- Toolkit: FunctionTool 注册业务工具
- State: AgentState 维护会话上下文 + 权限上下文（服务端托管，默认拒绝 + 工具白名单）
- Config: ContextConfig / ReActConfig 显式声明上下文与轮次策略（见 _build_context_config）
"""
from agentscope.agent import (
    Agent,
    ContextConfig,
    InjectionConfig,
    ModelConfig,
    ReActConfig,
)
from agentscope.model import ChatModelBase
from agentscope.middleware import ReplyBudgetControlMiddleware
from agentscope.permission import (
    PermissionBehavior,
    PermissionContext,
    PermissionMode,
    PermissionRule,
)
from agentscope.state import AgentState

from .config import Settings
from .middleware import CustomerMemoryMiddleware
from .models import create_degraded_model
from .prompts import build_system_prompt
from .tools import build_toolkit, tool_names


def build_customer_service_agent(
    settings: Settings,
    session_id: str,
    state: AgentState | None = None,
    model: ChatModelBase | None = None,
    user_id: str = "",
) -> Agent:
    """为单个会话创建一个客服 Agent 实例。

    Args:
        settings: 全局配置。
        session_id: 会话 ID（会写入 AgentState，便于事件追踪与持久化）。
        state: 恢复已有会话时传入完整 AgentState（含上下文与中间件状态）；
            缺省时新建空会话。用完整 state 而非消息列表，是为了连同
            ``middle_context`` / ``tool_context`` 一起恢复。
        model: 可选的自定义模型实例（测试注入用）。
        user_id: 用户标识（登录态），供跨会话记忆识别回头客；空表示匿名。
    """
    if state is None:
        state = AgentState(
            session_id=session_id,
            context=[],
            permission_context=build_permission_context(),
        )
    return Agent(
        name=settings.agent_name,
        system_prompt=build_system_prompt(settings.agent_name, settings.brand_name),
        model=model or create_degraded_model(settings),
        toolkit=build_toolkit(),
        state=state,
        # 跨会话用户记忆：注入「已知用户档案」并在回复后落盘
        middlewares=[
            CustomerMemoryMiddleware(user_id=user_id),
            # 单轮 token 预算：超限即注入收尾提示并禁用工具，防止无限工具循环
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


def build_permission_context() -> PermissionContext:
    """构造权限策略：显式工具白名单，清单之外一律不静默执行。

    别改成 ``DONT_ASK``：自定义工具返回的 ASK 会被它转成 DENY，写操作全被拒。
    """
    return PermissionContext(
        mode=PermissionMode.DEFAULT,
        allow_rules={
            name: [
                PermissionRule(
                    tool_name=name,
                    rule_content=None,      # 空表示匹配该工具的全部调用
                    behavior=PermissionBehavior.ALLOW,
                    source="projectSettings",
                ),
            ]
            for name in tool_names()
        },
    )


def _build_context_config(settings: Settings) -> ContextConfig:
    """构造上下文策略：显式声明而不依赖 SDK 默认值。

    离线模型无法输出结构化摘要，压缩走"截断最旧上下文"兜底，显式打开让行为可预期。
    """
    return ContextConfig(
        trigger_ratio=settings.context_trigger_ratio,
        reserve_ratio=settings.context_reserve_ratio,
        context_buffer_ratio=settings.context_buffer_ratio,
        tool_result_limit=settings.tool_result_limit,
        compression_fallback_to_truncation=True,
    )