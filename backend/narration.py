# -*- coding: utf-8 -*-
"""旁白抑制：区分"面向用户的正文"与"调工具前的过程性旁白"。

模型常在发起工具调用前先吐一句 "I'll check…"/"我先帮您查一下"，这不是业务结论。
规则：本次模型调用若发起了工具调用，它之前的文字是旁白，工具轮之后的才是正文。
三条链路复用同一实现——SSE 输出（``service.ChatStreamer``）、专家结论收集
（``multi_agent.agents.run_specialist``）、评测轨迹录制（``eval.driver``）。
"""
from __future__ import annotations

# 尚无工具调用时的正文缓冲上限；超过即判定为正文，避免纯闲聊场景整段延迟。
NARRATION_BUFFER_LIMIT = 120


class NarrationFilter:
    """按事件流区分旁白与正文，只返回应给用户看的文本。

    buffer_limit: 无工具调用时的缓冲上限，超过即转为正文；None 表示不设上限
    （专家子 agent、评测录制只关心最终文本，不需要逐块输出）。
    """

    def __init__(self, buffer_limit: int | None = NARRATION_BUFFER_LIMIT) -> None:
        self._buffer_limit = buffer_limit
        self._pending = ""              # 尚未定性为旁白或正文的缓冲
        self._saw_tool_in_call = False  # 本次模型调用是否发起了工具调用
        self._answer_phase = False      # 本回复是否已进入正文阶段

    def on_text(self, delta: str) -> str:
        """文本增量：返回可以立即输出的正文，仍在缓冲中则返回空串。"""
        if self._answer_phase:
            return delta
        self._pending += delta
        if self._buffer_limit is not None and len(self._pending) >= self._buffer_limit:
            self._answer_phase = True
            flushed, self._pending = self._pending, ""
            return flushed
        return ""

    def on_tool_call(self) -> None:
        """本段模型调用发起了工具调用：缓冲里的文字是旁白，丢弃，并进入正文阶段。"""
        self._pending = ""
        self._saw_tool_in_call = True
        self._answer_phase = True

    def on_model_end(self) -> str:
        """一次模型调用结束：没发起工具调用时，缓冲的内容就是正文。"""
        flushed = ""
        if not self._saw_tool_in_call and self._pending:
            flushed, self._pending = self._pending, ""
        self._saw_tool_in_call = False
        return flushed

    def on_reply_end(self) -> str:
        """回复结束兜底：正常已在 on_model_end 吐出，这里防漏（如模型调用中途报错）。"""
        flushed, self._pending = self._pending, ""
        return flushed