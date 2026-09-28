# -*- coding: utf-8 -*-
"""跨会话用户记忆中间件：on_system_prompt 注入已知档案，on_reply 抽取事实落盘。

读写都是纯代码逻辑，不依赖模型决策，故离线模型下同样有效。
"""
from typing import Any

from agentscope.message import Msg
from agentscope.middleware import MiddlewareBase

from ..store.memory_store import UserMemoryStore, user_memory_store
from ..store.mock_store import user_store

# 抽取事实时关注的工具：调用参数里有手机号后四位，或结果里有档案/订单信息
_PHONE_TOOLS = {
    "query_user_profile",
    "list_recent_orders",
    "view_cart",
    "add_to_cart",
    "update_cart_item",
    "place_order",
    "pay_order",
    "apply_refund",
    "cancel_order",
}
_PROFILE_TOOLS = {"query_user_profile"}
_ORDER_TOOLS = {"query_order", "list_recent_orders", "track_logistics", "place_order"}


def _parse_block(block: Any) -> dict:
    """把工具结果块的 output 统一解析成 dict。"""
    import json

    output = getattr(block, "output", "")
    if isinstance(output, list):
        output = "".join(getattr(x, "text", "") for x in output)
    try:
        parsed = json.loads(output or "{}")
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class CustomerMemoryMiddleware(MiddlewareBase):
    """用户长期记忆：跨会话记住回头客。

    Args:
        user_id: 登录态用户标识（如 U10001）；空表示匿名会话。
        store: 记忆存储，默认全局单例（测试可注入临时实例）。
    """

    def __init__(
        self,
        user_id: str = "",
        store: UserMemoryStore | None = None,
    ) -> None:
        self._user_id = user_id or ""
        self._store = store or user_memory_store
        # 记忆键（手机号后四位）：可能来自登录态解析，也可能在对话中学到
        self._key = ""
        self._resolved = False

    # ---------------- 身份解析 ----------------

    async def _resolve_key(self) -> str:
        """由登录态 user_id 解析出记忆键（手机号后四位）。只尝试一次。"""
        if self._key or self._resolved:
            return self._key
        self._resolved = True
        if self._user_id:
            profile = await user_store.find_by_id(self._user_id)
            if profile:
                self._key = str(profile.get("phone_tail") or "")
        return self._key

    @staticmethod
    def _learn_key(msgs: list[Msg]) -> str:
        """从工具调用参数里学出手机号后四位（取最后一次出现的）。"""
        key = ""
        for msg in msgs:
            for block in msg.get_content_blocks("tool_call"):
                if getattr(block, "name", "") not in _PHONE_TOOLS:
                    continue
                import json

                try:
                    args = json.loads(getattr(block, "input", "") or "{}")
                except (ValueError, TypeError):
                    continue
                tail = str(args.get("phone_tail") or "").strip()
                if tail.isdigit() and len(tail) == 4:
                    key = tail
        return key

    # ---------------- 事实抽取 ----------------

    @staticmethod
    def _extract_facts(msgs: list[Msg]) -> dict:
        """从上下文中抽取可长期保留的用户事实：身份取自工具入参，业务事实取自工具结果。"""
        calls: dict[str, str] = {}
        facts: dict[str, Any] = {}
        order_ids: list[str] = []

        for msg in msgs:
            for block in msg.get_content_blocks("tool_call"):
                calls[block.id] = getattr(block, "name", "")

            for block in msg.get_content_blocks("tool_result"):
                name = calls.get(block.id) or getattr(block, "name", "")
                data = _parse_block(block)
                if not data or not data.get("found", True):
                    continue

                if name in _PROFILE_TOOLS:
                    facts.setdefault("nickname", data.get("nickname"))
                    facts.setdefault("member_level", data.get("member_level"))
                    facts.setdefault("points", data.get("points"))

                if name in _ORDER_TOOLS:
                    order = data.get("order")
                    if isinstance(order, dict) and order.get("order_id"):
                        order_ids.append(str(order["order_id"]))
                    for item in data.get("orders") or []:
                        if isinstance(item, dict) and item.get("order_id"):
                            order_ids.append(str(item["order_id"]))

        if order_ids:
            facts["order_ids"] = order_ids
        return {k: v for k, v in facts.items() if v not in (None, "", [])}

    # ---------------- 钩子 ----------------

    async def on_system_prompt(self, agent, current_prompt: str) -> str:
        """把已知用户档案注入系统提示词。"""
        # 身份可能在本轮对话中学到（用户报出手机号后四位）
        if not self._key:
            self._key = self._learn_key(agent.state.context)
        if not self._key:
            await self._resolve_key()

        record = await self._store.get(self._key) if self._key else None
        if not record:
            return current_prompt
        return current_prompt + self._render(record)

    async def on_reply(self, agent, input_kwargs: dict, next_handler):
        """一轮回复结束后抽取事实并落盘（此时工具结果才刚写入上下文）。"""
        async for evt in next_handler():
            yield evt

        msgs = agent.state.context
        learned = self._learn_key(msgs)
        if learned:
            self._key = learned
        if not self._key:
            await self._resolve_key()
        if not self._key:
            return

        facts = self._extract_facts(msgs)
        if self._user_id:
            facts["user_id"] = self._user_id
        if facts:
            await self._store.upsert(self._key, **facts)

    # ---------------- 渲染 ----------------

    @staticmethod
    def _render(record: dict) -> str:
        lines = ["", "# 已知用户档案（来自历史会话的记忆）"]
        if record.get("phone_tail") or record.get("key"):
            lines.append(f"- 手机号后四位：{record.get('phone_tail') or record.get('key')}")
        if record.get("nickname"):
            lines.append(f"- 昵称：{record['nickname']}")
        if record.get("member_level"):
            lines.append(f"- 会员等级：{record['member_level']}")
        if record.get("points") is not None:
            lines.append(f"- 积分：{record['points']}")
        if record.get("order_ids"):
            lines.append(f"- 近期订单：{'、'.join(record['order_ids'][-5:])}")
        if record.get("updated_at"):
            turns = record.get("turns")
            suffix = f"（累计 {turns} 轮）" if turns else ""
            lines.append(f"- 上次互动：{record['updated_at']}{suffix}")

        lines.append("")
        lines.append(
            "这位用户是回头客，可直接用昵称称呼，不必再索要手机号后四位。"
            "但订单状态、物流、退款资格、积分余额这类业务数据仍**必须调用工具核实最新值**，"
            "不得仅凭以上记忆作答；记忆与工具结果冲突时，以工具结果为准。"
        )
        return "\n".join(lines) + "\n"