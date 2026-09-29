# -*- coding: utf-8 -*-
"""文本向量化：调用 DashScope OpenAI 兼容 embeddings 接口。

带按文本哈希的磁盘缓存（backend/data/embeddings_cache.json），服务重启或 FAQ 微调时只补嵌新增文本，
避免每次检索都重复请求计费。任何网络 / 鉴权失败都向上抛异常，由检索层决定回退。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from ..config import Settings

_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings_cache.json"


def text_hash(text: str) -> str:
    """文本内容指纹，用作缓存键：内容变了指纹就变，天然实现失效。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Embedder:
    """批量文本向量化，带磁盘缓存。"""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.dashscope_api_key or ""
        self._provider = settings.embed_provider
        self._model = settings.embed_model
        self._base_url = (settings.embed_base_url or "").rstrip("/")
        self._timeout = settings.embed_timeout
        self._batch = max(1, settings.embed_batch_size)
        self._cache: dict[str, list[float]] = self._load_disk()
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        """是否具备调用远程 Embedding 的条件。"""
        return self._provider == "dashscope" and bool(self._api_key) and bool(self._base_url)

    # ------------------------------------------------------------------
    # 缓存读写
    # ------------------------------------------------------------------
    def _load_disk(self) -> dict[str, list[float]]:
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def _persist_disk(self) -> None:
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            # 缓存落盘失败不影响检索，最坏情况下次重新嵌入
            pass

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """返回与 texts 同序的向量列表；命中缓存的不再请求 API。"""
        if not self.available:
            raise RuntimeError("Embedding 后端未启用（缺少可用 Key 或 provider=none）")

        # 去重后只挑出缓存里没有的文本
        pending: list[str] = []
        seen: set[str] = set()
        for t in texts:
            h = text_hash(t)
            if h in self._cache or t in seen:
                continue
            seen.add(t)
            pending.append(t)

        if pending:
            async with self._lock:
                # 拿锁后再查一次，避免并发重复请求同一批文本
                to_call = [t for t in pending if text_hash(t) not in self._cache]
                if to_call:
                    vectors = await self._call_api(to_call)
                    for t, v in zip(to_call, vectors):
                        self._cache[text_hash(t)] = v
                    self._persist_disk()

        return [self._cache[text_hash(t)] for t in texts]

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """分批调用远程接口，保持与入参同序。"""
        out: list[list[float]] = []
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for i in range(0, len(texts), self._batch):
                chunk = texts[i : i + self._batch]
                resp = await client.post(
                    url,
                    headers=headers,
                    json={"model": self._model, "input": chunk,
                          "encoding_format": "float"},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                # 兼容返回乱序：按 index 归位
                ordered = sorted(data, key=lambda d: d.get("index", 0))
                out.extend(item["embedding"] for item in ordered)
        return out
