# -*- coding: utf-8 -*-
"""精排：调用 DashScope 原生 text-rerank，对召回候选按原始问题做相关性重排。

未配置（缺 provider / model / base_url / Key）或调用失败时统一返回 None，
由检索层沿用召回原序——离线 Demo 与测试不受影响。
"""
from __future__ import annotations

import httpx

from ..config import Settings


class Reranker:
    """基于远程交叉编码器的文档精排器。"""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.dashscope_api_key or ""
        self._provider = settings.rerank_provider
        self._model = settings.rerank_model
        self._url = (settings.rerank_base_url or "").strip()
        self._timeout = settings.rerank_timeout

    @property
    def available(self) -> bool:
        """是否具备调用远程精排的条件。任一要素缺失即视为未启用。"""
        return (
            self._provider == "dashscope"
            and bool(self._api_key)
            and bool(self._model)
            and bool(self._url)
        )

    async def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        """返回与 documents 同序的相关性得分列表；未启用或失败时返回 None。"""
        if not self.available or not documents or not query:
            return None

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": len(documents)},  # 全部候选都打分
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, headers=headers, json=payload)
                resp.raise_for_status()
                results = resp.json()["output"]["results"]
        except Exception:
            return None  # 异常时退回不精排

        scores = [0.0] * len(documents)
        for item in results:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(documents):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores
