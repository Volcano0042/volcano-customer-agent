# -*- coding: utf-8 -*-
"""多智能体层：监督者路由 + 专家子 agent（agent-as-tool）。"""
from .agents import (
    SPECIALIST_SPECS,
    build_specialist_agent,
    build_supervisor_agent,
    make_delegate_tool,
    run_specialist,
)
from .passthrough import (
    SingleDelegationPassthroughModel,
    single_delegation_answer,
)

__all__ = [
    "SPECIALIST_SPECS",
    "SingleDelegationPassthroughModel",
    "build_specialist_agent",
    "build_supervisor_agent",
    "make_delegate_tool",
    "run_specialist",
    "single_delegation_answer",
]
