# -*- coding: utf-8 -*-
"""FAQ 检索工具：底层调用 backend/rag 的改写 + 混合召回 + 精排流水线。

每条命中带 source_id 与 snippet，工具汇总为 citations 供 Agent 按来源作答、前端溯源。
无 Key 或调用失败时逐级降级到纯词面检索，found / results 结构不变；
接入更强向量库（Milvus / Qdrant）或调整策略只换 rag/ 内实现。
"""
from ..config import get_settings
from ..rag import get_knowledge_base
from ..store.mock_store import faq_store


async def search_faq(question: str) -> dict:
    """在商城知识库中检索常见问题的官方解答（运费、发货时效、退货规则、发票、会员积分、改地址、保价、保修、优惠券、生鲜售后等）。遇到政策性问题时优先使用本工具，并依据返回的 citations 作答，禁止凭印象编造条款。

    Args:
        question: 用户的问题原文或关键描述。
    """
    settings = get_settings()
    entries = await faq_store.all()
    kb = await get_knowledge_base(settings)
    hit = await kb.search(question, entries)
    results = hit.get("results", [])
    confidence = hit.get("confidence", "high")

    if not results:
        return {
            "found": False,
            "message": "知识库中未检索到相关解答，请考虑转人工或创建工单",
        }

    citations = [
        {
            "id": r.get("source_id", ""),
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in results
    ]

    out: dict = {"found": True, "results": results, "citations": citations}
    if confidence == "low":
        out["confidence"] = "low"
        out["hint"] = (
            "检索到的内容与问题相关度偏低，请先向用户澄清具体诉求再作答；"
            "若仍不确定，不要臆断政策细节，转人工或创建工单更稳妥。"
        )
    else:
        out["confidence"] = "high"
    return out
