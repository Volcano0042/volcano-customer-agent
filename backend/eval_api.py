# -*- coding: utf-8 -*-
"""评测看板 API：读取历次 run、触发的后台评测任务与运行状态。

评测会短暂改写业务数据（快照还原），故同一时刻只允许一个 run 在跑。
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .eval import storage
from .eval.runner import run_eval


class RunRequest(BaseModel):
    max_cases: int | None = None
    no_cache: bool = False


def create_eval_router() -> APIRouter:
    router = APIRouter(prefix="/api/eval", tags=["eval"])

    # 后台任务状态（进程内单例）
    job: dict = {"running": False, "run_id": None, "error": None, "requested": None}

    async def _bg(max_cases: int | None, no_cache: bool) -> None:
        try:
            run = await run_eval(max_cases=max_cases, use_cache=False if no_cache else None)
            job["run_id"] = run["run_id"]
        except Exception as exc:  # noqa: BLE001
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["running"] = False

    @router.get("/summary")
    async def summary() -> dict:
        run = storage.latest_run()
        if not run:
            return {"available": False}
        return {
            "available": True,
            "run_id": run["run_id"],
            "created_at": run["created_at"],
            "model": run["model"],
            "judge_model": run["judge_model"],
            "judge_available": run["judge_available"],
            "case_count": run["case_count"],
            "metrics": run["metrics"],
        }

    @router.get("/runs")
    async def runs() -> dict:
        return {"runs": storage.list_runs()}

    @router.get("/runs/{run_id}")
    async def run_detail(run_id: str) -> dict:
        run = storage.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run 不存在")
        return {"run": run}

    @router.get("/run/status")
    async def run_status() -> dict:
        return job

    @router.post("/run")
    async def start_run(req: RunRequest) -> dict:
        if job["running"]:
            raise HTTPException(status_code=409, detail="已有评测在进行中")
        job.update({"running": True, "run_id": None, "error": None})
        asyncio.create_task(_bg(req.max_cases, req.no_cache))
        return {"started": True, "max_cases": req.max_cases, "no_cache": req.no_cache}

    return router
