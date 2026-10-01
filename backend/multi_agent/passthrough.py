# -*- coding: utf-8 -*-
"""终答透传：只委派一个专家时，直接把专家写好的答案作为本轮回复。

用一个模型包装器免掉监督者"拿到专家结果后再重写一遍"的第二次模型调用。
只在恰好一个委派、答案完整、没报错没转人工时生效；跨域的多专家委派仍交给监督者综合。
"""
from __future__ import annotations

import json
from typing import Any

from agentscope.message import TextBlock
from agentscope.model import ChatModelBase, ChatResponse

# 委派工具名前缀（对应 SPECIALIST_SPECS 的 tool_name）
DELEGATE_PREFIX = "delegate_to_"


def _blocks_to_text(output: str | list) -> str:
    """工具结果可能是原始字符串，也可能是 TextBlock/DataBlock 列表。"""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "".join(getattr(b, "text", "") for b in output)
    return ""


def _answer_from_output(output: str | list) -> str | None:
    """从委派工具结果的 JSON 里取专家答案，取不到返回 None。"""
    try:
        payload = json.loads(_blocks_to_text(output))
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("error") or payload.get("handed_off"):
        return None
    answer = payload.get("answer")
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    return answer or None


def single_delegation_answer(messages: list) -> str | None:
    """本轮只委派了一个专家时返回该专家的答案，否则返回 None。

    只看最后一条消息：AgentScope 把同一轮的工具调用与结果写进同一条 assistant 消息，
    新用户消息会追加在其后。
    """
    if not messages:
        return None
    last = messages[-1]
    getter = getattr(last, "get_content_blocks", None)
    if getter is None:
        return None
    results = getter("tool_result") or []
    delegates = [
        b for b in results
        if str(getattr(b, "name", "")).startswith(DELEGATE_PREFIX)
    ]
    if len(delegates) != 1:
        return None
    return _answer_from_output(delegates[0].output)


class SingleDelegationPassthroughModel(ChatModelBase):
    """只委派一个专家时透传其答案，否则转交被包装的模型。"""

    def __init__(self, inner: ChatModelBase) -> None:
        # 从被包装模型复制 ChatModelBase 需要的属性（同 DegradedChatModel）
        super().__init__(
            credential=getattr(inner, "credential", None),
            model=getattr(inner, "model", "passthrough-wrapper"),
            parameters=getattr(inner, "parameters", None),
            stream=getattr(inner, "stream", True),
            max_retries=0,  # 透传命中时不调用模型；未命中则交给 inner 自己重试
            retry_delay=getattr(inner, "retry_delay", 1.0),
            context_size=getattr(inner, "context_size", 32768),
        )
        self.inner = inner
        self.passthrough_count = 0

    def __getattr__(self, name: str):
        """未定义的属性代理给被包装模型（如 formatter）。"""
        return getattr(self.inner, name)

    async def __call__(
        self,
        messages: list | None = None,
        tools: list[dict] | None = None,
        tool_choice: Any | None = None,
        **kwargs: Any,
    ):
        answer = single_delegation_answer(messages or [])
        if answer is not None:
            self.passthrough_count += 1
            return ChatResponse(content=[TextBlock(text=answer)], is_last=True)
        return await self.inner(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"SingleDelegationPassthroughModel(inner={self.inner!r}, "
            f"passed_through={self.passthrough_count})"
        )