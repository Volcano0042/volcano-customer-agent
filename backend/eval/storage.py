# -*- coding: utf-8 -*-
"""评测结果与缓存落盘：backend/data/eval/ 下按 run 归档，另按 case 指纹缓存原始输出。

缓存命中可跳过重新驱动模型与重新计费，只做确定性重算分。原子写避免半截文件。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
_CACHE_DIR = _EVAL_DIR / "cache"
_RUNS_DIR = _EVAL_DIR / "runs"
_TZ_CN = timezone(timedelta(hours=8))


def _ensure_dirs() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


# ---------------- case 级缓存 ----------------
def load_cache(fingerprint: str) -> dict | None:
    return _read(_CACHE_DIR / f"{fingerprint}.json")


def save_cache(fingerprint: str, transcript: dict, verdict: dict | None) -> None:
    _ensure_dirs()
    _atomic_write(
        _CACHE_DIR / f"{fingerprint}.json",
        {"transcript": transcript, "verdict": verdict},
    )


# ---------------- run 级归档 ----------------
def new_run_id() -> str:
    return datetime.now(_TZ_CN).strftime("run-%Y%m%d-%H%M%S")


def save_run(run: dict) -> Path:
    _ensure_dirs()
    path = _RUNS_DIR / f"{run['run_id']}.json"
    _atomic_write(path, run)
    return path


def get_run(run_id: str) -> dict | None:
    return _read(_RUNS_DIR / f"{run_id}.json")


def list_runs(limit: int = 50) -> list[dict]:
    """按时间倒序返回历次 run 的轻量摘要（不含逐 case 明细）。"""
    if not _RUNS_DIR.exists():
        return []
    files = sorted(_RUNS_DIR.glob("run-*.json"), key=lambda p: p.name, reverse=True)
    summaries: list[dict] = []
    for path in files[:limit]:
        run = _read(path)
        if not run:
            continue
        summaries.append(
            {
                "run_id": run.get("run_id"),
                "created_at": run.get("created_at"),
                "model": run.get("model"),
                "judge_model": run.get("judge_model"),
                "case_count": len(run.get("results", [])),
                "metrics": run.get("metrics"),
            },
        )
    return summaries


def latest_run() -> dict | None:
    runs = list_runs(limit=1)
    return get_run(runs[0]["run_id"]) if runs else None
