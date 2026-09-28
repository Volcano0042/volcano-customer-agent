# -*- coding: utf-8 -*-
"""上下文实体继承层：从历史对话回收订单号 / 手机号后四位 / 货号，并修补追问意图。

纯函数，与模型无关。boundary 指"最后一条用户消息"的下标。
"""
import re
from typing import Any

from agentscope.message import Msg

ORDER_RE = re.compile(r"SO\d{8,}", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
SKU_RE = re.compile(r"P\d{3,4}", re.IGNORECASE)
MASKED_PHONE_RE = re.compile(r"\d{3}\*{4}\d{4}")   # 脱敏手机号，如 138****6621

BARE_PHONE_RE = re.compile(r"^\s*(\d{4})\s*$")     # 用户只回了"3721"这种
ACK_WORDS = ("好的", "好", "嗯", "嗯嗯", "确认", "是的", "对", "行", "可以",
             "就这样", "没问题", "ok", "OK")


def extract_entities(text: str) -> dict[str, str]:
    """从用户当前这句话里提取订单号 / 手机号后四位 / 退款原因 / 货号。"""
    order_match = ORDER_RE.search(text)
    order_id = order_match.group(0).upper() if order_match else ""
    masked = ORDER_RE.sub("", text)   # 先摘掉订单号，免得其中数字被当成手机尾号
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
    """把消息正文、工具入参、工具结果拼成候选文本；include_results=False 时跳过工具结果。"""
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
    """从 boundary 之前的历史里回收订单号 / 货号（含工具结果）与手机尾号（不含）。

    手机尾号只看正文与工具入参：工具结果里的四位数字常是快递员手机或积分。
    调用时须已包含本轮 user 消息，否则 boundary 会落到列表末尾。
    """
    history = messages[: _boundary(messages)]

    full_blob = _collect_text(history)
    orders = ORDER_RE.findall(full_blob)
    skus = SKU_RE.findall(full_blob)

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
    """本轮抽取 + 历史继承：本轮提到的优先，没提的槽位才用历史值补。

    继承来的订单号可能是历史里的尾单，未必是用户正在聊的那笔，不要直接当事实用。
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
    """一轮对话的入口：本轮抽取 + 历史继承 + 追问修补，返回 (intent, entities)。"""
    entities = extract_entities(user_text)
    current_order = entities.get("order_id")

    intent = resolve_followup(intent, user_text, messages)   # 物流特例依赖修补后的意图
    entities = merge_entities(user_text, messages)

    # 物流追问 + 订单号是继承来的：改走手机尾号定位在途订单，避免误选历史尾单
    if intent == "logistics" and not current_order and entities.get("phone_tail"):
        entities["order_id"] = ""

    return intent, entities