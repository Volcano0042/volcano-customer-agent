# -*- coding: utf-8 -*-
"""客服业务工具集。"""
from typing import Any

from agentscope.tool import FunctionTool, Toolkit

from .knowledge import search_faq
from .orders import list_recent_orders, query_order, track_logistics
from .refunds import apply_refund, cancel_order, check_refund_policy
from .support import create_ticket, query_user_profile, transfer_to_human
from .shop import (
    add_to_cart,
    list_products,
    pay_order,
    place_order,
    query_product,
    update_cart_item,
    view_cart,
)


# 工具清单：(实现函数, 是否只读)，是 Toolkit 与权限白名单的唯一数据源
TOOL_SPECS: tuple[tuple[Any, bool], ...] = (
    # ---- 只读：查订单 / 物流 / 资料 / 政策 / 商品 ----
    (query_order, True),
    (list_recent_orders, True),
    (track_logistics, True),
    (query_user_profile, True),
    (search_faq, True),
    (check_refund_policy, True),
    (list_products, True),
    (query_product, True),
    (view_cart, True),
    # ---- 写：退款 / 取消 / 工单 / 转人工 / 购物车 / 下单 / 支付 ----
    (apply_refund, False),
    (cancel_order, False),
    (create_ticket, False),
    (transfer_to_human, False),
    (add_to_cart, False),
    (update_cart_item, False),
    (place_order, False),
    (pay_order, False),
)


def build_toolkit(tool_filter: list[str] | None = None) -> Toolkit:
    """构建 Toolkit（AgentScope 2.0 工具层）。

    tool_filter 为 None 时纳入全部业务工具；给定名单时只保留这些工具
    （供多智能体的专家子 agent 装配各自的工具子集，顺序按传入名单）。
    """
    if tool_filter is None:
        specs = list(TOOL_SPECS)
    else:
        wanted = set(tool_filter)
        by_name = {fn.__name__: (fn, ro) for fn, ro in TOOL_SPECS}
        missing = wanted - by_name.keys()
        if missing:
            raise ValueError(f"build_toolkit 收到未知工具名: {sorted(missing)}")
        specs = [by_name[name] for name in tool_filter]
    return Toolkit(
        tools=[FunctionTool(fn, is_read_only=ro) for fn, ro in specs],
    )


def tool_names() -> list[str]:
    """本 Agent 的业务工具名列表（权限白名单的单一数据源）。"""
    return [fn.__name__ for fn, _ in TOOL_SPECS]


__all__ = ["build_toolkit", "tool_names", "TOOL_SPECS"]