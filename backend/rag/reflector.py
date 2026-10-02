# -*- coding: utf-8 -*-
"""反思式检索评估器（Self-RAG / CRAG）：让模型判召回资料能否作答，不足则给出改写查询。

走 DashScope OpenAI 兼容 /chat/completions，强制 JSON 输出；未配置或调用失败返回 None，
由上层跳过反思、沿用原置信度——离线与测试不受影响。
"""
from __future__ import annotations

import json

import httpx

from ..config import Settings

_PROMPT = """你是电商客服知识库的检索质检员。判断【检索到的资料】能否准确回答【用户问题】，只输出一个 JSON 对象，不要输出多余文字。

判定：
- sufficient：资料是否足以准确回答该问题（相关且信息完整为 true，仅沾边或答非所问为 false）。
- rewrite：若不足以回答，给出一个更可能命中的改写查询（换同义词、补规范术语、或拆成更具体的子问题）；足够则留空字符串。
- reason：一句话依据。

【用户问题】
{question}

【检索到的资料】
{docs}

只输出如下 JSON：
{{"sufficient": true/false, "rewrite": "改写查询或空串", "reason": "一句话"}}"""


class RetrievalReflector:
    """检索评估器客户端。缺配置或调用失败即不可用，由上层降级。"""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.dashscope_api_key or ""
        self._enabled = settings.rag_reflect_enabled
        self._model = settings.reflect_model
        self._base_url = settings.reflect_base_url
        self._timeout = settings.reflect_timeout

    @property
    def available(self) -> bool:
        """开关、Key、模型、端点任一缺失即视为未启用。"""
        return bool(
            self._enabled
            and self._api_key
            and self._model
            and self._base_url
        )

    @staticmethod
    def _format_docs(results: list[dict]) -> str:
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            body = r.get("snippet") or r.get("content") or ""
            lines.append(f"{i}. {r.get('title', '')}：{body}")
        return "\n".join(lines)

    async def evaluate(self, question: str, results: list[dict]) -> dict | None:
        """返回 {sufficient, rewrite, reason}；不可用、无资料或解析失败返回 None。"""
        if not self.available or not results:
            return None

        prompt = _PROMPT.format(
            question=question, docs=self._format_docs(results)
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

        return _parse_verdict(content)


def _parse_verdict(content: str) -> dict | None:
    """从模型返回文本里抽出 JSON，容忍 markdown 代码块包裹。"""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    return {
        "sufficient": bool(obj.get("sufficient")),
        "rewrite": str(obj.get("rewrite", "") or "").strip(),
        "reason": str(obj.get("reason", "")),
    }
