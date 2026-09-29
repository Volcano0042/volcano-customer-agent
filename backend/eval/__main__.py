# -*- coding: utf-8 -*-
"""命令行入口：python -m backend.eval [--max N] [--no-cache]"""
from __future__ import annotations

import argparse
import asyncio

from ..config import get_settings
from .runner import run_eval


def _fmt_ci(ci) -> str:
    if not ci:
        return "-"
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _print_summary(run: dict) -> None:
    print(f"\n评测 {run['run_id']} · 模型={run['model']} · 裁判={run['judge_model']}"
          f"{'（不可用，仅确定性判据）' if not run['judge_available'] else ''}")
    print(f"{'维度':<14}{'通过/总数':<12}{'通过率':<10}95% 置信区间")
    m = run["metrics"]
    for dim in ("task", "tool", "hallucination", "safety"):
        row = m[dim]
        rate = f"{row['rate']:.0%}" if row["rate"] is not None else "-"
        print(f"{dim:<14}{str(row['passed']) + '/' + str(row['n']):<12}{rate:<10}{_fmt_ci(row['ci95'])}")
    ov = m["_overall"]
    rate = f"{ov['rate']:.0%}" if ov["rate"] is not None else "-"
    print(f"{'整体':<12}{str(ov['passed']) + '/' + str(ov['n']):<12}{rate:<10}{_fmt_ci(ov['ci95'])}")
    print("\n逐条结果：")
    for r in run["results"]:
        flag = "PASS" if r["overall_pass"] else "FAIL"
        src = "缓存" if r["from_cache"] else "实跑"
        fails = [d for d, v in r["scores"]["dims"].items() if not v["pass"]]
        note = f"（未过：{','.join(fails)}）" if fails else ""
        print(f"  [{flag}] {r['case_id']:<28}{src}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Volcano 客服 Agent 端到端评测")
    parser.add_argument("--max", type=int, default=None, help="最多跑几条 case")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存，全部重新实跑")
    args = parser.parse_args()

    run = asyncio.run(
        run_eval(
            settings=get_settings(),
            max_cases=args.max,
            use_cache=False if args.no_cache else None,
        ),
    )
    _print_summary(run)


if __name__ == "__main__":
    main()
