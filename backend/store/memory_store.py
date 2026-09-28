# -*- coding: utf-8 -*-
"""跨会话用户记忆存储。

与 ``mock_store.py`` 分开是有意为之：``mock_store`` 是 README 里写明的
「接真实中台时的替换点」，而用户记忆属于 Agent 自身状态 ——
接入真实中台时不该被一起换掉，所以它单独成模块。

存储形态与项目其余数据保持一致：单文件 JSON + 文件锁，零外部依赖。

    backend/data/memory.json
    {
      "3721": {
        "phone_tail": "3721",
        "user_id": "U10001",
        "nickname": "追风的云",
        "member_level": "黄金会员",
        "points": 2680,
        "order_ids": ["SO20260810001"],
        "turns": 3,
        "first_seen_at": "2026-09-28 21:50",
        "updated_at": "2026-09-28 22:05"
      }
    }

身份键优先用手机号后四位（项目里购物车、订单查询都以它为实际主键）；
若客户端带登录态 ``user_id``，则由 ``UserStore`` 解析出手机号后四位后再落库。
"""
import asyncio
import json
from pathlib import Path
from typing import Any

from .mock_store import now_cn

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FILENAME = "memory.json"

# 单条记忆里最多保留的订单号数量，避免无限增长
_MAX_ORDER_IDS = 10

_lock = asyncio.Lock()


async def _read() -> dict[str, dict]:
    path = _DATA_DIR / _FILENAME
    if not path.exists():
        return {}
    text = await asyncio.to_thread(path.read_text, encoding="utf-8-sig")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        # 记忆是旁路能力，损坏时不应让主链路失败
        return {}
    return data if isinstance(data, dict) else {}


async def _write(data: dict[str, dict]) -> None:
    path = _DATA_DIR / _FILENAME
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    await asyncio.to_thread(path.write_text, payload, "utf-8")


class UserMemoryStore:
    """用户长期记忆（进程外持久化，可跨会话、跨重启）。"""

    async def get(self, key: str) -> dict | None:
        """按键取一条记忆，key 为手机号后四位或 user_id。"""
        if not key:
            return None
        return (await _read()).get(str(key))

    async def all(self) -> dict[str, dict]:
        return await _read()

    async def upsert(self, key: str, **fields: Any) -> dict:
        """写入 / 合并一条记忆。

        合并语义：只覆盖本次提供的字段；``order_ids`` 做去重合并并保留最近
        ``_MAX_ORDER_IDS`` 条；``first_seen_at`` 只在首次写入时设置。
        """
        if not key:
            raise ValueError("记忆键不能为空")
        key = str(key)
        async with _lock:
            data = await _read()
            record = data.get(key) or {}
            record.setdefault("first_seen_at", now_cn())

            for name, value in fields.items():
                if value in (None, "", []):
                    continue
                if name == "order_ids":
                    merged = list(record.get("order_ids") or [])
                    for oid in value:
                        if oid and oid not in merged:
                            merged.append(oid)
                    record["order_ids"] = merged[-_MAX_ORDER_IDS:]
                else:
                    record[name] = value

            record["key"] = key
            record["updated_at"] = now_cn()
            record["turns"] = int(record.get("turns", 0)) + 1
            data[key] = record
            await _write(data)
            return record

    async def clear(self) -> int:
        """清空全部记忆（测试与调试用），返回清除条数。"""
        async with _lock:
            data = await _read()
            count = len(data)
            await _write({})
            return count


user_memory_store = UserMemoryStore()