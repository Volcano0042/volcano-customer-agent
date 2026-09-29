"""四维指标的聚合：每个维度算通过率 + Wilson 95% 置信区间（小样本更稳健）。

Wilson 区间在小样本 / 极端比例下比正态近似更合理，避免 0 或 100% 时区间塌成一点。
纯计算，无外部依赖，便于离线单测。
"""
from __future__ import annotations

import math

DIMENSIONS = ("task", "tool", "hallucination", "safety")
_Z = 1.96  # 95% 置信


def wilson_interval(passes: int, n: int, z: float = _Z) -> tuple[float, float] | None:
    """二项比例的 Wilson 95% 区间；n<=0 返回 None。"""
    if n <= 0:
        return None
    phat = passes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(
        phat * (1 - phat) / n + z * z / (4 * n * n)
    )
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def aggregate(case_results: list[dict]) -> dict:
    """按维度汇总通过率与置信区间；只统计该维度被判到（出现在 dims 里）的 case。"""
    metrics: dict[str, dict] = {}
    for dim in DIMENSIONS:
        judged = [r for r in case_results if dim in r.get("scores", {}).get("dims", {})]
        n = len(judged)
        passes = sum(1 for r in judged if r["scores"]["dims"][dim]["pass"])
        metrics[dim] = {
            "n": n,
            "passed": passes,
            "rate": round(passes / n, 4) if n else None,
            "ci95": wilson_interval(passes, n),
        }

    overall_pass = sum(1 for r in case_results if r.get("overall_pass"))
    metrics["_overall"] = {
        "n": len(case_results),
        "passed": overall_pass,
        "rate": round(overall_pass / len(case_results), 4) if case_results else None,
        "ci95": wilson_interval(overall_pass, len(case_results)),
    }
    return metrics
