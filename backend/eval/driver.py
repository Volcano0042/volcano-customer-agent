# -*- coding: utf-8 -*-
"""用真实模型驱动客服 Agent 跑完一个 case 的多轮对话，录制逐轮轨迹。

只走主模型（不降级），失败如实抛出，供评测客观反映线上表现。
评测固定走"监督者 + 专家子 agent"链路（`multi_agent=True`），与线上开启
`MULTI_AGENT_ENABLED` 时的行为一致。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from agentscope.event import (
    ModelCallEndEvent,
    ReplyEndEvent,
    RequireUserConfirmEvent,
    TextBlockDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.message import TextBlock, UserMsg

from ..agent_factory import build_agent
from ..config import Settings
from ..models import create_chat_model
from ..multi_agent import tool_trace
from ..narration import NarrationFilter
from .cases import DANGER_WRITE


def _specialist_inner_tools(result_text: str) -> list[str]:
    """从委派工具返回的 JSON 里取出专家子 agent 实际调用的业务工具名。

    多 agent 模式下顶层只有 delegate_to_*，业务工具（search_faq / apply_refund 等）
    在子 agent 结果里；解出来才能让工具与安全维度的确定性判据照常生效。
    """
    if not result_text:
        return []
    try:
        payload = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(payload, dict):
        inner = payload.get("tools_used")
        if isinstance(inner, list):
            return [str(n) for n in inner]
    return []


@dataclass
class TurnRecord:
    index: int
    user: str
    text: str = ""
    tools: list[dict] = field(default_factory=list)
    danger_writes: list[str] = field(default_factory=list)
    confirm_requested: bool = False
    finished_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]


@dataclass
class Transcript:
    case_id: str
    turns: list[TurnRecord] = field(default_factory=list)

    @property
    def all_tool_names(self) -> list[str]:
        seen: list[str] = []
        for turn in self.turns:
            for name in turn.tool_names:
                if name not in seen:
                    seen.append(name)
        return seen

    @property
    def tools_by_turn(self) -> list[list[str]]:
        return [turn.tool_names for turn in self.turns]

    @property
    def final_text(self) -> str:
        return self.turns[-1].text if self.turns else ""

    @property
    def all_text(self) -> str:
        return "\n".join(turn.text for turn in self.turns if turn.text)

    @property
    def transferred(self) -> bool:
        return "transfer_to_human" in self.all_tool_names

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "transferred": self.transferred,
            "turns": [
                {
                    "index": t.index,
                    "user": t.user,
                    "text": t.text,
                    "tools": t.tools,
                    "danger_writes": t.danger_writes,
                    "confirm_requested": t.confirm_requested,
                    "finished_reason": t.finished_reason,
                    "tokens_in": t.tokens_in,
                    "tokens_out": t.tokens_out,
                    "error": t.error,
                }
                for t in self.turns
            ],
        }


async def drive_case(settings: Settings, case: dict) -> Transcript:
    """新建会话，逐轮把 case.turns 喂给真实模型，返回录制好的轨迹。

    固定走多智能体链路：评测度量的是线上正在跑的那条链路。
    """
    transcript = Transcript(case_id=case["id"])
    agent = build_agent(
        settings,
        session_id=f"eval-{case['id']}",
        model=create_chat_model(settings),
        multi_agent=True,
    )

    for idx, message in enumerate(case["turns"]):
        rec = TurnRecord(index=idx, user=message)
        transcript.turns.append(rec)

        args_buf: dict[str, str] = {}
        result_buf: dict[str, str] = {}
        names: dict[str, str] = {}

        # 旁白抑制：只录用户看得到的文本，与线上 SSE 一致（规则见 narration 模块）。
        narration = NarrationFilter(buffer_limit=None)
        inner_log, log_token = tool_trace.start()
        try:
            async for evt in agent.reply_stream(
                UserMsg(name="用户", content=[TextBlock(text=message)]),
            ):
                if isinstance(evt, ToolCallStartEvent):
                    narration.on_tool_call()
                    names[evt.tool_call_id] = evt.tool_call_name
                elif isinstance(evt, ToolCallDeltaEvent):
                    args_buf[evt.tool_call_id] = (
                        args_buf.get(evt.tool_call_id, "") + evt.delta
                    )
                elif isinstance(evt, TextBlockDeltaEvent):
                    rec.text += narration.on_text(evt.delta)
                elif isinstance(evt, RequireUserConfirmEvent):
                    rec.confirm_requested = True
                elif isinstance(evt, ToolResultTextDeltaEvent):
                    result_buf[evt.tool_call_id] = (
                        result_buf.get(evt.tool_call_id, "") + evt.delta
                    )
                elif isinstance(evt, ToolResultEndEvent):
                    name = names.get(evt.tool_call_id, evt.tool_call_id)
                    raw_result = result_buf.get(evt.tool_call_id, "")
                    tool = {
                        "name": name,
                        "args": args_buf.get(evt.tool_call_id, ""),
                        "state": str(evt.state),
                        "result": raw_result[:2000],
                    }
                    rec.tools.append(tool)
                    if name in DANGER_WRITE:
                        rec.danger_writes.append(name)
                    # 展开委派结果里的专家业务工具，计入工具/安全维度判据；
                    # 返回内容取自 tool_trace 旁路，供裁判核对结论依据
                    inner_results = tool_trace.by_name(inner_log)
                    for inner in _specialist_inner_tools(raw_result):
                        rec.tools.append({
                            "name": inner,
                            "args": "",
                            "state": "specialist",
                            "result": inner_results.get(inner, ""),
                        })
                        if inner in DANGER_WRITE:
                            rec.danger_writes.append(inner)
                elif isinstance(evt, ModelCallEndEvent):
                    rec.text += narration.on_model_end()
                    rec.tokens_in += max(0, evt.input_tokens)
                    rec.tokens_out += max(0, evt.output_tokens)
                elif isinstance(evt, ReplyEndEvent):
                    rec.text += narration.on_reply_end()
                    rec.finished_reason = str(evt.finished_reason)
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"
        finally:
            tool_trace.stop(log_token)

    return transcript
