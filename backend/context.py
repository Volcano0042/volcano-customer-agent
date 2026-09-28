# -*- coding: utf-8 -*-
"""上下文实体继承层：确定性的槽位抽取与指代消解。

客服对话里有大量"帮我退了""那物流呢""3721"这类省略句 —— 它们本身不含任何
实体，必须从更早的对话里把订单号 / 手机号后四位 / 货号捞回来，否则工具调用
无从下手。

这套逻辑此前埋在 ``OfflineChatModel`` 内部。放在那里有两个问题：

1. **和模型耦合**：它是纯对话状态推理，与"用哪个模型"无关，却只有离线模型
   享受得到（换真实模型就整段丢失）。
2. **不可单独测试**：想验证指代消解对不对，得先把整个模型跑起来。

现在抽成本模块后它是一组纯函数：输入消息列表，输出实体字典。谁都能调 ——
``OfflineChatModel`` 用它驱动工具调用；将来接真实模型时，也可以由中间件在
``on_system_prompt`` 里消费同一份结果。

（关于"为什么不直接做成中间件把实体注入提示词"：实测过，见 ``merge_entities``
的说明 —— 继承来的订单号可能是错的，注入反而误导模型。）

术语：**boundary** 指"最后一条用户消息"的下标。历史实体只从 boundary **之前**
的消息里回收 —— 当前这句用户消息由 ``extract_entities`` 单独处理。
"""
import re
from typing import Any

from agentscope.message import Msg

ORDER_RE = re.compile(r"SO\d{8,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
SKU_RE = re.compile(r"P\d{3,4}", re.IGNORECASE)
# 脱敏手机号，如 138****6621。必须整体剔除：否则尾号 6621 会被当成用户手机号
MASKED_PHONE_RE = re.compile(r"\d{3}\*{4}\d{4}")

# 裸数字回答（用户只回了"3721"这种），后接上一轮挂起的问题
BARE_PHONE_RE = re.compile(r"^\s*(\d{4})\s*$")
# 简短确认词，后接上一轮挂起的意图
ACK_WORDS = ("好的", "好", "嗯", "嗯嗯", "确认", "是的", "对", "行", "可以",
             "就这样", "没问题", "ok", "OK")


def extract_entities(text: str) -> dict[str, str]:
    """从用户当前这句话里提取订单号 / 手机号后四位 / 退款原因 / 货号。"""
    order_match = ORDER_RE.search(text)
    order_id = order_match.group(0).upper() if order_match else ""
    # 先摘掉订单号，避免 SO20260810001 里的数字被当成手机号后四位
    masked = ORDER_RE.sub("", text)
    tail_match = PHONE_RE.search(masked)
    phone_tail = tail_match.group(1) if tail_match else ""
    reason = "用户申请退款"
    for kw, r in (
        ("破损", "商品破损/质量问题"),
        ("损坏", "商品损坏/质量问题"),
        ("坏了", "商品损坏"),
        ("质量", "质量问题"),
        ("拍错", "拍错了"),
        ("买错", "买错了"),
        ("下错", "下错单了"),
        ("不想要", "不想要了"),
        ("不喜欢", "不合适/不喜欢"),
        ("七天", "七天无理由退货"),
    ):
        if kw in text:
            reason = r
            break
    sku_match = SKU_RE.search(text)
    sku = sku_match.group(0).upper() if sku_match else ""
    return {"order_id": order_id, "phone_tail": phone_tail, "reason": reason, "sku": sku}


def _boundary(messages: list[Msg]) -> int:
    """返回最后一条用户消息的下标；没有用户消息时返回列表长度。"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            return i
    return len(messages)


def _collect_text(messages: list[Msg], *, include_results: bool = True) -> str:
    """把消息的正文、工具入参、工具结果拼成一段可供正则扫描的文本。

    ``include_results=False`` 时跳过工具**结果**（只保留消息正文与工具入参）。
    区分二者很关键：工具结果是系统数据，里面的数字未必属于用户。
    """
    parts: list[str] = []
    for m in messages:
        parts.append(m.get_text_content() or "")
        for block in m.get_content_blocks("tool_call"):
            parts.append(str(getattr(block, "input", "") or ""))
        if not include_results:
            continue
        for block in m.get_content_blocks("tool_result"):
            output: Any = getattr(block, "output", "")
            if isinstance(output, list):
                output = " ".join(getattr(x, "text", "") for x in output)
            parts.append(str(output))
    return "\n".join(parts)


def history_entities(messages: list[Msg]) -> dict[str, str]:
    """从更早的对话上下文中回收最近提到的订单号 / 手机号后四位 / 货号。

    扫描最后一条用户消息**之前**的消息，让"帮我退了""那物流呢"这类指代上文的
    短句能继承实体，实现多轮对话。

    注意 boundary 语义：调用时必须已经包含本轮的 user 消息，否则会一直扫到
    列表末尾、把本轮内容也当成"历史"。

    订单号 / 货号允许来自工具结果（``list_recent_orders`` 的返回值里就带着
    用户自己的订单号，是合法的继承来源）。**手机尾号则严格不看工具结果** ——
    工具结果里的四位数字经常是别人的或别的字段：

    - ``track_logistics`` 结果里的快递员手机 ``138****6621`` → 会被误当成用户尾号
    - ``query_user_profile`` 结果里的积分 ``2680`` → 同样会被误认

    用户自己的手机尾号只可能出现在他本人说的话里（或模型据此填的工具入参中），
    所以只从这两个来源回收。
    """
    history = messages[: _boundary(messages)]

    # 订单号 / 货号：含工具结果
    full_blob = _collect_text(history)
    orders = ORDER_RE.findall(full_blob)
    skus = SKU_RE.findall(full_blob)

    # 手机尾号：只看消息正文与工具入参，且先剔除脱敏手机号与订单号 / 货号 / 小数
    user_blob = _collect_text(history, include_results=False)
    masked = MASKED_PHONE_RE.sub(" ", user_blob)
    masked = ORDER_RE.sub(" ", masked)
    masked = SKU_RE.sub(" ", masked)
    masked = re.sub(r"\d+\.\d+", " ", masked)
    phones = [
        p for p in PHONE_RE.findall(masked)
        if not re.fullmatch(r"(19|20)\d{2}", p)   # 排除时间戳里的年份
    ]
    return {
        "order_id": orders[-1].upper() if orders else "",
        "phone_tail": phones[-1] if phones else "",
        "sku": skus[-1].upper() if skus else "",
    }


def pending_intent(messages: list[Msg]) -> str:
    """从上一条客服回复里识别"挂起的问题"，供短句 / 裸数字回答继承意图。"""
    for m in reversed(messages[: _boundary(messages)]):
        if m.role != "assistant":
            continue
        text = m.get_text_content() or ""
        if not text:
            continue
        if "确认退款" in text or ("退款" in text and "确认" in text):
            return "refund"
        if "下单" in text:
            return "checkout"
        if "购物车" in text:
            return "cart"
        if "手机号" in text or "订单号" in text or "定位订单" in text:
            return "order"
        if "物流" in text or "快递" in text:
            return "logistics"
        return ""
    return ""


def resolve_followup(intent: str, user_text: str, messages: list[Msg]) -> str:
    """多轮对话的意图修补：裸数字回答 / 简短确认映射到上一轮挂起的意图。"""
    text = (user_text or "").strip()
    if BARE_PHONE_RE.fullmatch(text):
        return pending_intent(messages) or intent
    if intent in ("fallback", "greeting"):
        ack = text.strip("。！!？?~～ ")
        if ack in ACK_WORDS and pending_intent(messages) == "refund":
            return "refund"
    return intent


def merge_entities(user_text: str, messages: list[Msg]) -> dict[str, str]:
    """本轮抽取 + 历史继承，得到这一轮可用的实体集合。

    只做槽位合并，不含任何意图相关的决策。

    **注意继承值可能是错的，不要不加甄别地当成事实使用。** 具体地：
    ``history_entities`` 里的订单号取的是"历史文本中最后出现的订单号"，
    当上下文里有一份 ``list_recent_orders`` 的结果（一列 5 笔订单）时，
    取到的是数组末尾那笔，**未必是用户正在聊的那笔**。

    所以 ``resolve_turn`` 会在拿到意图后做一次甄别（见其物流特例）；而把
    合并结果直接展示给模型（例如注入系统提示词）是不安全的 —— 实测中它会把
    ``SO20260721004`` 这个无关订单号当成"用户想查的订单"塞给模型。

    合并规则：本轮明确提到的值优先；本轮没提的槽位才用历史值补。
    """
    entities = extract_entities(user_text)
    inherited = history_entities(messages)
    for key in ("order_id", "phone_tail", "sku"):
        if not entities.get(key) and inherited.get(key):
            entities[key] = inherited[key]
    return entities


def resolve_turn(
    user_text: str,
    messages: list[Msg],
    intent: str,
) -> tuple[str, dict[str, str]]:
    """一轮对话的实体与意图解析：本轮抽取 + 历史继承 + 追问修补。

    返回 ``(intent, entities)``。这是离线模型侧的**主要入口** —— 调用方不必
    关心历史继承的细节，拿到结果直接用于决策。
    """
    entities = extract_entities(user_text)
    current_order = entities.get("order_id")

    # 先修补意图：裸数字 / 简短确认要映射到上一轮挂起的问题，
    # 后面的物流特例依赖修补**之后**的意图
    intent = resolve_followup(intent, user_text, messages)

    entities = merge_entities(user_text, messages)

    # 特例：追问物流、订单号是继承来的、且手机位可用时，优先走"手机号定位在途
    # 订单"的语义路径，避免误选历史列表里的尾单。
    # 这是**决策策略**而非槽位解析，所以只留在这里，不下沉到 merge_entities。
    if intent == "logistics" and not current_order and entities.get("phone_tail"):
        entities["order_id"] = ""

    return intent, entities