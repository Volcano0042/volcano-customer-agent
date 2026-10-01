# -*- coding: utf-8 -*-
"""专家内部工具返回的旁路采集。

多 agent 链路里专家的业务工具跑在委派工具内部，顶层事件流只有 delegate_to_* 的结果。
这里用 ContextVar 旁路收集工具返回，不改变给模型的工具返回值（避免撑大监督者上下文、
影响线上行为）；未开采集时是空操作。
"""
from __future__ import annotations

from contextvars import ContextVar, Token

# 单条工具返回的留存上限
RESULT_LIMIT = 800

_COLLECT: ContextVar[list | None] = ContextVar("specialist_tool_results", default=None)


def collect(name: str, result: str) -> None:
    """专家内部某个工具跑完时调用。"""
    log = _COLLECT.get()
    if log is None:
        return
    log.append({"name": name, "result": (result or "")[:RESULT_LIMIT]})


def start() -> tuple[list, Token]:
    """开始采集，返回 (容器, 还原令牌)。"""
    log: list = []
    return log, _COLLECT.set(log)


def stop(token: Token) -> None:
    _COLLECT.reset(token)


def by_name(log: list) -> dict[str, str]:
    """按工具名取首次返回，供轨迹把内部工具条目补上真实依据。"""
    out: dict[str, str] = {}
    for entry in log:
        out.setdefault(entry["name"], entry["result"])
    return out