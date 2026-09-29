# -*- coding: utf-8 -*-
"""FAQ 知识库检索工具（RAG：稠密向量 + 词面混合召回）。

底层为 backend/rag/：调用 DashScope Embedding API 把知识条目向量化建索引，
检索时融合语义相似度与关键词命中。无 Key 或调用失败时自动退回纯词面检索，
工具签名与返回结构不变，接入更强向量库（Milvus / Qdrant）也只换 retriever 实现。
"""
from ..config import get_settings
from ..rag import get_knowledge_base
from ..store.mock_store import faq_store


async def search_faq(question: str) -> dict:
    """在商城知识库中检索常见问题的官方解答（运费、发货时效、退货规则、发票、会员积分、改地址等）。遇到政策性问题时优先使用本工具，禁止凭印象作答。

    Args:
        question: 用户的问题原文或关键描述。
    """
    settings = get_settings()
    entries = await faq_store.all()
    kb = await get_knowledge_base(settings)
    results = await kb.search(question, entries)
    if not results:
        return {
            "found": False,
            "message": "知识库中未检索到相关解答，请考虑转人工或创建工单",
        }
    return {"found": True, "results": results}
