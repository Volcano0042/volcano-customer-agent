# -*- coding: utf-8 -*-
"""会话状态持久化：让会话在进程重启后仍可恢复。

AgentScope 2.0 的 ``AgentState`` 是纯 pydantic 模型，``model_dump_json()`` /
``model_validate_json()`` 可直接往返，所以这里不需要任何额外依赖 ——
与项目「clone 即可跑」的定位一致。

落盘位置：``backend/data/sessions/{session_id}.json``
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from agentscope.state import AgentState

from .mock_store import now_cn

_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"

# session_id 来自客户端，会被拼进文件路径，因此必须严格校验以防目录穿越。
# 本服务生成的 id 形如 sess_<12位十六进制>，这里放宽到通用安全字符集。
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 默认最多保留的会话存档数
DEFAULT_MAX_RECORDS = 500


def is_safe_session_id(session_id: str) -> bool:
    """校验 session_id 是否可安全用作文件名。"""
    return bool(session_id) and bool(_SAFE_ID.fullmatch(session_id))


class SessionStore:
    """会话存档（AgentState + 会话元信息）。"""

    def __init__(self, directory: Path | None = None) -> None:
        self._dir = directory or _SESSIONS_DIR

    def _path(self, session_id: str) -> Path | None:
        if not is_safe_session_id(session_id):
            return None
        return self._dir / f"{session_id}.json"

    async def save(
        self,
        session_id: str,
        state: AgentState,
        meta: dict,
    ) -> bool:
        """保存一个会话。返回是否写入成功。"""
        import asyncio

        path = self._path(session_id)
        if path is None:
            return False

        def _do() -> None:
            self._dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": {**meta, "session_id": session_id, "saved_at": now_cn()},
                # AgentState 自带 JSON 往返能力，thinking / tool_call / tool_result
                # / hint 等各类内容块都能原样保留
                "state": json.loads(state.model_dump_json()),
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), "utf-8",
            )

        try:
            await asyncio.to_thread(_do)
            return True
        except OSError:
            # 持久化是旁路能力，磁盘异常不应中断对话
            return False

    async def load(self, session_id: str) -> dict | None:
        """读取一个会话存档，返回 {"state": AgentState, "meta": dict}。"""
        import asyncio

        path = self._path(session_id)
        if path is None or not path.exists():
            return None

        def _do() -> dict | None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
                state = AgentState.model_validate(payload["state"])
            except (OSError, ValueError, KeyError, TypeError):
                return None
            return {"state": state, "meta": payload.get("meta") or {}}

        return await asyncio.to_thread(_do)

    async def delete(self, session_id: str) -> bool:
        import asyncio

        path = self._path(session_id)
        if path is None or not path.exists():
            return False
        try:
            await asyncio.to_thread(path.unlink)
            return True
        except OSError:
            return False

    async def list_ids(self) -> list[str]:
        import asyncio

        def _do() -> list[str]:
            if not self._dir.exists():
                return []
            return [
                p.stem
                for p in sorted(
                    self._dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            ]

        return await asyncio.to_thread(_do)

    async def prune(self, max_records: int = DEFAULT_MAX_RECORDS) -> int:
        """按最近修改时间裁剪存档数量，避免 data/ 无限增长。"""
        import asyncio

        def _do() -> int:
            if not self._dir.exists():
                return 0
            files = sorted(
                self._dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            removed = 0
            for path in files[max_records:]:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
            return removed

        return await asyncio.to_thread(_do)


session_store = SessionStore()