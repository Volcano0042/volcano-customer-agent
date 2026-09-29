# -*- coding: utf-8 -*-
"""知识库检索：查询改写 → 混合召回并集 → 精排 → 置信度自检。

- 文档 = 标题 + 正文 + 关键词，向量化建稠密索引；得分 = alpha·余弦 + (1-alpha)·归一词面分；
- 精排开启时用原始问题对候选重排并主导排序，未开启 / 调用失败则跳过、沿用召回原序；
- 无 Embedding（provider=none）时退回纯词面——离线 Demo 与测试行为确定、不受影响；
- 索引按「条目指纹」惰性重建，FAQ 增删改后下次检索自动刷新。
"""
from __future__ import annotations

import asyncio

from ..config import Settings
from .embedder import Embedder, text_hash
from .query_rewrite import expand_queries
from .reranker import Reranker


def _bigrams(text: str) -> set[str]:
    text = "".join(ch for ch in text.lower() if not ch.isspace())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i : i + 2] for i in range(len(text) - 1)}


def lexical_score(query: str, entry: dict) -> float:
    """字符二元组 Jaccard + 关键词命中，作为语义检索的补强与离线兜底。"""
    query_bg = _bigrams(query)
    if not query_bg:
        return 0.0
    keyword_hits = sum(1 for kw in entry.get("keywords", []) if kw in query)
    keyword_score = keyword_hits * 3.0
    doc_bg = _bigrams(entry.get("title", "") + " " + entry.get("content", ""))
    union = query_bg | doc_bg
    jaccard = len(query_bg & doc_bg) / len(union) if union else 0.0
    return keyword_score + jaccard * 4.0


def _doc_text(entry: dict) -> str:
    """送入向量模型的文档表示：标题、正文、关键词拼接。"""
    kws = " ".join(entry.get("keywords", []))
    return f"{entry.get('title', '')}。{entry.get('content', '')}。{kws}"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a**0.5 * norm_b**0.5)


def _snippet(content: str, limit: int = 120) -> str:
    """截取正文前若干字符作为引用摘要，供前端「依据」条与溯源展示。"""
    text = (content or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class KnowledgeBase:
    """一个进程内共享的知识库索引 + 检索器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = Embedder(settings)
        self.reranker = Reranker(settings)
        self._entries: list[dict] = []
        self._vectors: list[list[float]] | None = None   # 与 _entries 同序；None 表示未向量化
        self._sig: tuple | None = None
        self._lock = asyncio.Lock()

    async def _ensure_index(self, entries: list[dict]) -> None:
        """条目指纹变化时才（重）建向量索引；嵌入失败则退回词面并留待下次重试。"""
        sig = tuple((e.get("id"), text_hash(_doc_text(e))) for e in entries)
        if self._vectors is not None and sig == self._sig:
            return
        async with self._lock:
            # 拿锁后二次确认，避免并发重复建索引
            if self._vectors is not None and sig == self._sig:
                return
            vectors = None
            if self.embedder.available:
                try:
                    vectors = await self.embedder.embed([_doc_text(e) for e in entries])
                except Exception:
                    vectors = None
            self._entries = list(entries)
            self._vectors = vectors
            # 向量化失败时不记录 sig，下次检索会再试；纯词面模式则记 sig 免重复拷贝
            self._sig = sig if vectors is not None or not self.embedder.available else None

    async def search(self, query: str, entries: list[dict]) -> dict:
        """执行一次检索，返回 {results, confidence, queries}。

        confidence 仅在精排开启时按精排分判定 high/low，降级路径恒为 high；
        未命中由工具层按空 results 处理，语义与旧实现一致。
        """
        valid = [e for e in entries if e.get("title") or e.get("content")]
        await self._ensure_index(valid)

        hybrid_mode = self._vectors is not None and self.embedder.available
        alpha = self.settings.rag_alpha
        threshold = self.settings.rag_min_score if hybrid_mode else 0.05

        # 1) 查询改写：原句 + 若干规范变体（离线纯规则，始终可用）
        queries = expand_queries(
            query, enabled=self.settings.query_rewrite_enabled
        ) or [query]

        # 2) 多路召回并按条目取并集：保留每条命中的最优混合分
        best: dict[int, tuple[float, float | None, float]] = {}
        for q in queries:
            qvec: list[float] | None = None
            if hybrid_mode:
                try:
                    qvec = (await self.embedder.embed([q]))[0]
                except Exception:
                    qvec = None
            for i, entry in enumerate(self._entries):
                ls = lexical_score(q, entry)
                lb = ls / (1.0 + ls)          # 词面分压到 [0,1)
                if qvec is not None:
                    ds = max(0.0, _cosine(qvec, self._vectors[i]))
                    score = alpha * ds + (1.0 - alpha) * lb
                else:
                    ds = None
                    score = ls                # 纯词面：沿用旧阈值与量纲
                if score <= threshold:
                    continue
                prev = best.get(i)
                if prev is None or score > prev[0]:
                    best[i] = (score, ds, lb)

        if not best:
            return {"results": [], "confidence": "high", "queries": queries}

        # 3) 短列表：按混合分放宽召回，交给精排收敛
        ranked = sorted(
            best.items(), key=lambda kv: kv[1][0], reverse=True
        )[: self.settings.rag_recall_k]

        # 4) 精排：用「原始问题」对候选做交叉编码重排，失败/未启用则保持召回原序
        rerank_map: dict[int, float] | None = None
        if self.reranker.available:
            docs = [_doc_text(self._entries[i]) for i, _ in ranked]
            scores = await self.reranker.rerank(query, docs)
            if scores is not None:
                rerank_map = {ranked[k][0]: scores[k] for k in range(len(ranked))}

        # 汇总每条命中的最终分与模式：精排开启时精排分主导排序
        rows: list[tuple[dict, float, float | None, float, float | None, str]] = []
        for i, (score, ds, lb) in ranked:
            rr = rerank_map.get(i) if rerank_map else None
            if rr is not None:
                final = rr
                mode = "hybrid+rerank" if ds is not None else "lexical+rerank"
            else:
                final = score
                mode = "hybrid" if ds is not None else "lexical"
            rows.append((self._entries[i], final, ds, lb, rr, mode))

        rows.sort(key=lambda x: x[1], reverse=True)
        top = rows[: self.settings.rag_top_k]

        # 5) 置信度自检：仅在有精排分时武断判低置信，降级路径不误伤
        if rerank_map is not None and top:
            confidence = (
                "high" if top[0][1] >= self.settings.rag_high_conf_bar else "low"
            )
        else:
            confidence = "high"

        return {
            "results": [
                {
                    "title": e.get("title", ""),
                    "content": e.get("content", ""),
                    "snippet": _snippet(e.get("content", "")),
                    "source_id": e.get("id", ""),
                    "score": round(final, 4),
                    "retrieval": mode,
                    "semantic": round(ds, 4) if ds is not None else None,
                    "lexical": round(lbv, 4),
                    "rerank": round(rrv, 4) if rrv is not None else None,
                }
                for e, final, ds, lbv, rrv, mode in top
            ],
            "confidence": confidence,
            "queries": queries,
        }


_kb: KnowledgeBase | None = None
_kb_lock = asyncio.Lock()


async def get_knowledge_base(settings: Settings) -> KnowledgeBase:
    """进程内单例：首次调用时构建，之后复用同一份向量索引。"""
    global _kb
    if _kb is None:
        async with _kb_lock:
            if _kb is None:
                _kb = KnowledgeBase(settings)
    return _kb
