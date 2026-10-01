# -*- coding: utf-8 -*-
"""评测编排：快照还原保证 case 隔离，逐条驱动真实模型 + 裁判 + 打分，聚合归档。

每个 case 前把可变数据文件回滚到快照，避免退款 / 加购等写操作跨 case 污染。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ..config import Settings, get_settings
from ..store.mock_store import order_store
from .cases import GOLDEN_CASES, case_fingerprint
from .driver import drive_case
from .judge import JUDGE_VERSION, Judge
from .metrics import aggregate
from . import storage
from .score import score_case

_TZ_CN = timezone(timedelta(hours=8))
_MUTABLE_FILES = ("orders.json", "tickets.json", "cart.json", "memory.json")
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _snapshot() -> dict[str, str | None]:
    snap: dict[str, str | None] = {}
    for name in _MUTABLE_FILES:
        path = _DATA_DIR / name
        snap[name] = path.read_text(encoding="utf-8") if path.exists() else None
    return snap


def _restore(snap: dict[str, str | None]) -> None:
    for name, text in snap.items():
        path = _DATA_DIR / name
        if text is None:
            if path.exists():
                path.unlink()
        else:
            path.write_text(text, encoding="utf-8")


async def _post_orders(case: dict) -> dict[str, str]:
    """跑完后读取该 case 关注的订单状态，供 expect_store 判定。"""
    want = (case.get("expect", {}) or {}).get("expect_store", {})
    oid = want.get("order_id", "")
    if not oid:
        return {}
    order = await order_store.find_by_id(oid)
    return {oid: order.get("status", "")} if order else {oid: ""}


def _timeout_transcript(case: dict, err: str) -> dict:
    """驱动超时时构造占位轨迹：按单轮空输出计，确定性判据自然判失败。"""
    return {
        "case_id": case["id"],
        "transferred": False,
        "turns": [
            {
                "index": i,
                "user": msg,
                "text": "",
                "tools": [],
                "danger_writes": [],
                "confirm_requested": False,
                "finished_reason": "timeout",
                "tokens_in": 0,
                "tokens_out": 0,
                "error": err,
            }
            for i, msg in enumerate(case["turns"])
        ],
    }


async def run_eval(
    settings: Settings | None = None,
    max_cases: int | None = None,
    use_cache: bool | None = None,
    case_ids: list[str] | None = None,
) -> dict:
    """执行一次端到端评测，返回并归档 run 结果。

    固定走"监督者+专家子 agent"链路，与线上开启 ``MULTI_AGENT_ENABLED`` 时一致。
    ``use_cache=False`` 只跳过读取，实跑出的新轨迹仍写回缓存；
    ``case_ids`` 只跑指定几条（改了某几条 case 的判据时用）。
    """
    settings = settings or get_settings()
    limit = max_cases if max_cases is not None else settings.eval_max_cases
    # use_cache 只管"读"：忽略缓存重跑仍写回缓存；EVAL_CACHE_ENABLED=false 才是既不读也不写的总开关。
    cache_enabled = settings.eval_cache_enabled
    read_cache = cache_enabled and use_cache is not False
    judge = Judge(settings)
    cases = GOLDEN_CASES[:limit] if limit and limit > 0 else GOLDEN_CASES
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["id"] in wanted]
    # run_id 在开跑时确定，避免同秒的两次短跑撞名、互相覆盖归档。
    run_id = storage.new_run_id()

    snapshot = _snapshot()
    results: list[dict] = []
    try:
        for case in cases:
            _restore(snapshot)  # case 隔离：先回滚数据
            fp = case_fingerprint(case)
            cached = storage.load_cache(fp) if read_cache else None
            from_cache = cached is not None
            timed_out = False

            if cached:
                transcript = cached["transcript"]
                # JUDGE_VERSION 变更：轨迹仍可用，旧裁决作废重判
                verdict = cached.get("verdict") if cached.get("judge_version") == JUDGE_VERSION else None
            else:
                try:
                    tr = await asyncio.wait_for(
                        drive_case(settings, case), timeout=settings.eval_case_timeout,
                    )
                    transcript = tr.to_dict()
                except asyncio.TimeoutError:
                    transcript = _timeout_transcript(
                        case, f"驱动超时 >{settings.eval_case_timeout:.0f}s",
                    )
                    timed_out = True
                verdict = None

            # 裁决缺失（上次裁判超时）时补判，否则该维度不进分母。
            if verdict is None and judge.available:
                try:
                    verdict = await asyncio.wait_for(
                        judge.judge(case, transcript),
                        timeout=settings.eval_judge_timeout + 15,
                    )
                except asyncio.TimeoutError:
                    verdict = None

            # 超时轨迹不是有效模型输出，不写缓存；缓存中已有同版本裁决时不重写。
            unchanged = (
                cached is not None
                and cached.get("judge_version") == JUDGE_VERSION
                and cached.get("verdict") == verdict
            )
            if cache_enabled and not timed_out and not unchanged:
                storage.save_cache(fp, transcript, verdict, JUDGE_VERSION)

            post = await _post_orders(case)
            scored = score_case(case, transcript, verdict, post)
            results.append(
                {
                    "case_id": case["id"],
                    "category": case.get("category", ""),
                    "dims": case.get("dims", []),
                    "goal": case.get("goal", ""),
                    "from_cache": from_cache,
                    "transcript": transcript,
                    "verdict": verdict,
                    "post_orders": post,
                    "scores": scored,
                    "overall_pass": scored["overall_pass"],
                },
            )
    finally:
        _restore(snapshot)  # 全部跑完，数据还原到评测前

    run = {
        "run_id": run_id,
        "created_at": datetime.now(_TZ_CN).isoformat(timespec="seconds"),
        "model": settings.model_name,
        "model_provider": settings.model_provider,
        "judge_model": settings.eval_judge_model or "none",
        "judge_available": judge.available,
        "case_count": len(results),
        "metrics": aggregate(results),
        "results": results,
    }
    storage.save_run(run)
    return run
