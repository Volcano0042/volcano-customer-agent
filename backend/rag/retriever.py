# -*- coding: utf-8 -*-
"""知识库检索：稠密向量 + 词面打分的混合召回。

设计要点：
- 每条 FAQ 文档文本 = 标题 + 正文 + 关键词，向量化后建稠密索引；
- 检索得分 = alpha · 余弦相似度 + (1 - alpha) · 归一化词面分，兼顾语义与精确关键词；
- 无 Embedding（provider=none）或调用失败时，退回纯词面打分，阈值与行为对齐旧实现，
  保证离线 Demo 和自动化测试不受影响；
- 索引按「条目指纹」惰性重建，FAQ 增删改后下一次检索自动刷新。
"""
from __future__ import annotations

import asyncio

from ..config import Settings
from .embedder import Embedder, text_hash


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


class KnowledgeBase:
    """一个进程内共享的知识库索引 + 检索器。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = Embedder(settings)
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

    async def search(self, query: str, entries: list[dict]) -> list[dict]:
        """返回按相关度降序的检索命中（已过滤低于阈值的条目）。"""
        valid = [e for e in entries if e.get("title") or e.get("content")]
        await self._ensure_index(valid)

        lex = [lexical_score(query, e) for e in self._entries]

        qvec: list[float] | None = None
        if self._vectors is not None and self.embedder.available:
            try:
                qvec = (await self.embedder.embed([query]))[0]
            except Exception:
                qvec = None

        alpha = self.settings.rag_alpha
        scored: list[tuple[dict, float, float | None, float, str]] = []
        for i, entry in enumerate(self._entries):
            ls = lex[i]
            lb = ls / (1.0 + ls)              # 词面分压到 [0,1)
            if qvec is not None:
                ds = max(0.0, _cosine(qvec, self._vectors[i]))
                score = alpha * ds + (1.0 - alpha) * lb
                threshold = self.settings.rag_min_score
                mode = "hybrid"
            else:
                ds = None
                score = ls                     # 纯词面：沿用旧阈值与量纲
                threshold = 0.05
                mode = "lexical"
            if score > threshold:
                scored.append((entry, score, ds, lb, mode))

        scored.sort(key=lambda x: x[1], reverse=True)
        hits = scored[: self.settings.rag_top_k]
        return [
            {
                "title": e.get("title", ""),
                "content": e.get("content", ""),
                "source_id": e.get("id", ""),
                "score": round(score, 4),
                "retrieval": mode,
                "semantic": round(ds, 4) if ds is not None else None,
                "lexical": round(lbv, 4),
            }
            for e, score, ds, lbv, mode in hits
        ]


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
