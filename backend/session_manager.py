# -*- coding: utf-8 -*-
"""多会话管理：会话 <-> 客服 Agent 实例的映射，含闲置回收。

toC 场景特点：并发用户多、单会话生命周期短、需要按用户隔离上下文。
每个会话持有独立的 Agent（其 AgentState.context 即该用户的对话历史）。

注：``from __future__ import annotations`` 不可删 —— 本模块的 ``list`` 方法会在类
命名空间遮蔽内置 ``list``，只注解惰性求值才能让 ``list[X]`` 不在类体求值时炸。
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentscope.agent import Agent

from .config import Settings


@dataclass
class SessionRuntime:
    """一个运行中的客服会话。"""

    session_id: str         # 会话唯一标识
    agent: Agent            # 该会话专属的 Agent 实例
    user_id: str = ""       # 用户标识（登录态传入），空表示匿名会话
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)    # 异步锁，保证同一会话串行执行
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    last_active_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    handed_off: bool = False    # 是否已转人工（客服场景特有字段）
    preview: str = ""           # 会话预览文本（供列表页快速展示）


class SessionManager:
    """会话注册表：实例常驻内存，AgentState 另存档到磁盘以便重启后恢复。

    存档是可选旁路，存储不可用时不影响对话。
    """

    def __init__(
        self,
        settings: Settings,
        agent_factory: Any,
        session_store: Any = None,
    ) -> None:
        self._settings = settings
        self._agent_factory = agent_factory  # 可延后 import，避免循环引用
        self._sessions: dict[str, SessionRuntime] = {}  # 进程内存中的会话注册表
        self._gc_task: asyncio.Task | None = None   # 垃圾回收任务
        if session_store is None:
            from .store.session_store import session_store as _default_store

            session_store = _default_store
        self._store = session_store

    # 生命周期管理
    async def start(self) -> None:
        self._gc_task = asyncio.create_task(self._gc_loop())

    async def stop(self) -> None:
        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass

    # 创建会话
    async def create(self, user_id: str = "") -> SessionRuntime:
        # 内存上限淘汰（LRU：按 last_active_at 排序剔除最久未活动的会话）
        if len(self._sessions) >= self._settings.max_sessions:
            stale = sorted(
                self._sessions.values(),
                key=lambda s: s.last_active_at,
            )
            for old in stale[: max(1, len(stale) - self._settings.max_sessions + 1)]:
                # 淘汰前先存档，之后仍可按 session_id 恢复
                await self.persist(old)
                self._sessions.pop(old.session_id, None)

        session_id = "sess_" + uuid.uuid4().hex[:12]
        runtime = SessionRuntime(
            session_id=session_id,
            agent=self._agent_factory(session_id, user_id=user_id),
            user_id=user_id,
        )
        self._sessions[session_id] = runtime
        return runtime

    # 获取（如果传了 session_id 且已存在）或创建
    async def get_or_create(
        self,
        session_id: str | None,
        user_id: str = "",
    ) -> SessionRuntime:
        if session_id and session_id in self._sessions:
            runtime = self._sessions[session_id]
            runtime.last_active_at = datetime.now().isoformat(timespec="seconds")
            # 匿名会话在后续请求中补上登录态时，补齐身份
            if user_id and not runtime.user_id:
                runtime.user_id = user_id
            return runtime

        # 内存里没有（进程重启过，或已被 LRU/TTL 回收）：尝试从存档恢复
        if session_id:
            restored = await self.restore(session_id, user_id=user_id)
            if restored is not None:
                return restored

        return await self.create(user_id=user_id)

    async def restore(
        self,
        session_id: str,
        user_id: str = "",
    ) -> SessionRuntime | None:
        """从磁盘存档恢复一个会话，失败返回 None（调用方回退到新建）。"""
        payload = await self._store.load(session_id)
        if not payload:
            return None

        meta = payload["meta"]
        resolved_user = user_id or str(meta.get("user_id") or "")
        runtime = SessionRuntime(
            session_id=session_id,
            agent=self._agent_factory(
                session_id,
                state=payload["state"],
                user_id=resolved_user,
            ),
            user_id=resolved_user,
            created_at=meta.get("created_at") or datetime.now().isoformat(timespec="seconds"),
            last_active_at=datetime.now().isoformat(timespec="seconds"),
            handed_off=bool(meta.get("handed_off")),
            preview=str(meta.get("preview") or ""),
        )
        self._sessions[session_id] = runtime
        return runtime

    async def persist(self, runtime: SessionRuntime) -> bool:
        """把会话状态存档。失败不抛异常（持久化是旁路能力）。"""
        return await self._store.save(
            runtime.session_id,
            runtime.agent.state,
            meta={
                "user_id": runtime.user_id,
                "created_at": runtime.created_at,
                "last_active_at": runtime.last_active_at,
                "handed_off": runtime.handed_off,
                "preview": runtime.preview,
            },
        )

    def get(self, session_id: str) -> SessionRuntime | None:
        return self._sessions.get(session_id)

    def list(self) -> list[SessionRuntime]:
        return list(self._sessions.values())

    def find_by_user(self, user_id: str) -> list[SessionRuntime]:
        """按用户标识找出该用户的所有会话（供管理后台与调试平台使用）。"""
        if not user_id:
            return []
        return [s for s in self._sessions.values() if s.user_id == user_id]

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # 会话列表页快速展示，不用加载完整历史
    def _update_preview(self, runtime: SessionRuntime) -> None:
        history = runtime.agent.state.context
        for msg in reversed(history):
            if msg.role == "user":
                preview = msg.get_text_content() or ""
                runtime.preview = preview[:32] if preview else ""
                return

    async def _gc_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            ttl = self._settings.session_ttl_minutes * 60
            now = datetime.now().timestamp()
            expired = [
                sid
                for sid, s in self._sessions.items()
                if now - datetime.fromisoformat(s.last_active_at).timestamp() > ttl
            ]
            for sid in expired:
                runtime = self._sessions.get(sid)
                if runtime is not None:
                    # 先存档再回收：闲置会话仍可凭 session_id 恢复
                    await self.persist(runtime)
                self._sessions.pop(sid, None)