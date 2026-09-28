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
            # 跨会话用户记忆：注入「已知用户档案」并在回复后落盘
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
    """构造权限策略：显式工具白名单。

    这里刻意避开原实现使用的 ``BYPASS``：``BYPASS`` 会连**安全类 ASK 也一并跳过**
    （见 ``PermissionEngine._check_bypass``），等价于没有防护。

    但也不能简单地换成 ``DONT_ASK``。实测结论：``FunctionTool`` 对**所有**自定义
    工具都返回 ASK（"Custom function tools must be explicitly allowed"），
    只有只读调用能靠"只读快路径"提前放行；而 ``DONT_ASK`` 会在工具自身检查那一步
    就把 ASK 转成 DENY，**根本走不到白名单** —— 结果是「加购物车」「申请退款」这类
    写操作全被拒，业务直接残废。

    ``DEFAULT`` 的求值顺序才是可用的：工具返回的普通 ASK 会继续往下走到白名单
    （第 5 步），命中的放行；未被白名单命中的才回落到 ASK 请用户确认。同时
    ``DEFAULT`` 不会跳过 ``bypass_immune`` 的安全类 ASK，护栏仍在。

    最终形态：**业务工具按清单放行，清单之外一律不静默执行**。
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
    """构造上下文策略。

    显式声明而非依赖 SDK 默认值，原因有二：

    1. 默认 ``trigger_ratio=0.8`` 与 ``tool_result_limit=50000`` 面向通用场景，
       对客服（单轮工具结果小、对话轮次多）偏宽松，长会话容易贴到窗口上限。
    2. 离线规则模型无法输出结构化摘要，压缩必然走"截断最旧上下文"这条兜底路径。
       这里显式打开该兜底，让行为可预期、可测试，而不是靠 SDK 默认值默默生效。
    """
    return ContextConfig(
        trigger_ratio=settings.context_trigger_ratio,
        reserve_ratio=settings.context_reserve_ratio,
        context_buffer_ratio=settings.context_buffer_ratio,
        tool_result_limit=settings.tool_result_limit,
        compression_fallback_to_truncation=True,
    )