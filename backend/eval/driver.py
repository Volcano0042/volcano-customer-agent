# -*- coding: utf-8 -*-
"""用真实模型驱动客服 Agent 跑完一个 case 的多轮对话，录制逐轮轨迹。

只走主模型（不降级），失败如实抛出，供评测客观反映线上表现。
"""
from __future__ import annotations

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

from ..agent_factory import build_customer_service_agent
from ..config import Settings
from ..models import create_chat_model
from .cases import DANGER_WRITE


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
    """新建会话，逐轮把 case.turns 喂给真实模型，返回录制好的轨迹。"""
    transcript = Transcript(case_id=case["id"])
    agent = build_customer_service_agent(
        settings,
        session_id=f"eval-{case['id']}",
        model=create_chat_model(settings),
    )

    for idx, message in enumerate(case["turns"]):
        rec = TurnRecord(index=idx, user=message)
        transcript.turns.append(rec)

        args_buf: dict[str, str] = {}
        result_buf: dict[str, str] = {}
        names: dict[str, str] = {}

        try:
            async for evt in agent.reply_stream(
                UserMsg(name="用户", content=[TextBlock(text=message)]),
            ):
                if isinstance(evt, ToolCallStartEvent):
                    names[evt.tool_call_id] = evt.tool_call_name
                elif isinstance(evt, ToolCallDeltaEvent):
                    args_buf[evt.tool_call_id] = (
                        args_buf.get(evt.tool_call_id, "") + evt.delta
                    )
                elif isinstance(evt, TextBlockDeltaEvent):
                    rec.text += evt.delta
                elif isinstance(evt, RequireUserConfirmEvent):
                    rec.confirm_requested = True
                elif isinstance(evt, ToolResultTextDeltaEvent):
                    result_buf[evt.tool_call_id] = (
                        result_buf.get(evt.tool_call_id, "") + evt.delta
                    )
                elif isinstance(evt, ToolResultEndEvent):
                    name = names.get(evt.tool_call_id, evt.tool_call_id)
                    tool = {
                        "name": name,
                        "args": args_buf.get(evt.tool_call_id, ""),
                        "state": str(evt.state),
                        "result": result_buf.get(evt.tool_call_id, "")[:2000],
                    }
                    rec.tools.append(tool)
                    if name in DANGER_WRITE:
                        rec.danger_writes.append(name)
                elif isinstance(evt, ModelCallEndEvent):
                    rec.tokens_in += max(0, evt.input_tokens)
                    rec.tokens_out += max(0, evt.output_tokens)
                elif isinstance(evt, ReplyEndEvent):
                    rec.finished_reason = str(evt.finished_reason)
        except Exception as exc:  # noqa: BLE001
            rec.error = f"{type(exc).__name__}: {exc}"

    return transcript
