# -*- coding: utf-8 -*-
"""端到端评测系统的离线单测：golden set 结构、确定性判据、指标数学、裁判解析、路由形状。

全部不联网：裁判配置已在 conftest 清空，评测 run 接口不被触发（只测 GET 只读端点）。
"""
import pytest

from backend.eval.cases import DANGER_WRITE, GOLDEN_CASES, case_fingerprint
from backend.eval.judge import _parse_verdict
from backend.eval.metrics import aggregate, wilson_interval
from backend.eval.score import score_case
from backend.tools import tool_names

VALID_DIMS = {"task", "tool", "hallucination", "safety"}


def _turn(index, text="", tools=(), danger=()):
    return {
        "index": index,
        "user": "u",
        "text": text,
        "tools": [{"name": n, "args": "", "state": "ok", "result": ""} for n in tools],
        "danger_writes": list(danger),
        "confirm_requested": False,
        "finished_reason": "stop",
        "tokens_in": 0,
        "tokens_out": 0,
        "error": "",
    }


def _transcript(turns):
    return {"case_id": "x", "transferred": False, "turns": list(turns)}


def _case(cid):
    return next(c for c in GOLDEN_CASES if c["id"] == cid)


# ---------------- golden set 结构 ----------------
def test_case_ids_unique_and_fingerprint_stable():
    ids = [c["id"] for c in GOLDEN_CASES]
    assert len(ids) == len(set(ids)), "case id 必须唯一"
    c = GOLDEN_CASES[0]
    assert case_fingerprint(c) == case_fingerprint(c)  # 纯函数，可复现


def test_case_required_fields_and_dims():
    for c in GOLDEN_CASES:
        for key in ("id", "category", "dims", "turns", "goal", "ground_truth"):
            assert key in c, f"{c.get('id')} 缺字段 {key}"
        assert set(c["dims"]) <= VALID_DIMS, f"{c['id']} 含非法维度"
        assert c["turns"] and all(isinstance(t, str) for t in c["turns"])
        assert set(c["dims"]), f"{c['id']} 未声明任何维度"


def test_case_referenced_tools_exist():
    known = set(tool_names()) | DANGER_WRITE
    for c in GOLDEN_CASES:
        exp = c.get("expect", {}) or {}
        for key in ("tools_any", "tools_all", "tools_forbidden"):
            for t in exp.get(key, []):
                assert t in known, f"{c['id']} 引用了不存在的工具 {t}"


def test_all_four_dimensions_covered():
    covered = set()
    for c in GOLDEN_CASES:
        covered |= set(c["dims"])
    assert covered == VALID_DIMS


def test_safety_cases_reference_danger_writes():
    """安全/门控类 case 的 tools_forbidden 若涉及写操作，须在 DANGER_WRITE 名单内。"""
    for c in GOLDEN_CASES:
        exp = c.get("expect", {}) or {}
        for t in exp.get("tools_forbidden", []):
            if t in {"apply_refund", "cancel_order", "pay_order", "place_order"}:
                assert t in DANGER_WRITE


# ---------------- 确定性判据 ----------------
def test_score_knowledge_case_pass_with_expected_tool_and_text():
    c = _case("kn_return_policy")
    tr = _transcript([_turn(0, "自签收次日起七天内支持七天无理由退货", ["search_faq"])])
    res = score_case(c, tr, None, None)
    assert res["dims"]["tool"]["pass"] is True
    assert res["dims"]["task"]["pass"] is True
    assert res["overall_pass"] is True


def test_score_knowledge_case_fails_when_tool_missing():
    c = _case("kn_return_policy")
    tr = _transcript([_turn(0, "七天无理由", [])])  # 没调 search_faq
    res = score_case(c, tr, None, None)
    assert res["dims"]["tool"]["pass"] is False
    assert res["deterministic_pass"] is False


def test_score_forbidden_tool_fails_tool_and_safety():
    c = _case("order_ambiguous")  # tools_forbidden: query_order / track_logistics
    tr = _transcript([_turn(0, "帮你查了", ["track_logistics"])])
    res = score_case(c, tr, None, None)
    assert res["dims"]["tool"]["pass"] is False


def test_score_danger_write_on_turn0_fails_safety():
    c = _case("refund_skip_confirm")  # confirm_before_write
    tr = _transcript([_turn(0, "已退款", ["apply_refund"], danger=["apply_refund"])])
    res = score_case(c, tr, None, None)
    assert res["dims"]["safety"]["pass"] is False
    assert res["overall_pass"] is False


def test_score_multiturn_write_after_check_passes():
    c = _case("refund_flow_multiturn")
    tr = _transcript([
        _turn(0, "可全额退款，确认吗？", ["check_refund_policy"]),
        _turn(1, "已提交退款", ["apply_refund"], danger=["apply_refund"]),
    ])
    post = {"SO20260812003": "退款中"}
    res = score_case(c, tr, None, post)
    assert res["dims"]["safety"]["pass"] is True   # 首轮未写
    assert res["dims"]["tool"]["pass"] is True      # tools_all + expect_store 命中
    assert res["overall_pass"] is True


def test_score_expect_store_mismatch_fails():
    c = _case("cancel_pending_pay")
    tr = _transcript([_turn(0, "已取消", ["cancel_order"])])
    res = score_case(c, tr, None, {"SO20260901006": "待付款"})  # 状态没改成已取消
    assert res["dims"]["tool"]["pass"] is False


def test_score_answer_not_contains_detects_hallucination():
    c = _case("order_not_found")  # answer_not_contains: VL8820472631
    tr = _transcript([_turn(0, "您的物流单号是 VL8820472631", ["query_order"])])
    res = score_case(c, tr, None, None)
    assert res["dims"]["hallucination"]["pass"] is False


def test_score_transfer_expectation():
    c = _case("transfer_direct")
    tr_pass = _transcript([_turn(0, "已转人工", ["transfer_to_human"])])
    tr_fail = _transcript([_turn(0, "我来帮您", [])])
    assert score_case(c, tr_pass, None, None)["overall_pass"] is True
    assert score_case(c, tr_fail, None, None)["overall_pass"] is False


def test_judge_verdict_can_flip_task_dim():
    """purchase_recommend 无 answer_contains，task 维度只由裁判决定：判失败即不通过。"""
    c = _case("purchase_recommend")
    tr = _transcript([_turn(0, "推荐这款耳机", ["list_products"])])
    # 无裁判时 task 维度无信号、不进入 dims
    assert "task" not in score_case(c, tr, None, None)["dims"]
    bad = score_case(c, tr, {"task_success": False, "hallucination": False, "safety_ok": True}, None)
    assert bad["dims"]["task"]["pass"] is False
    good = score_case(c, tr, {"task_success": True, "hallucination": False, "safety_ok": True}, None)
    assert good["dims"]["task"]["pass"] is True


# ---------------- 指标数学 ----------------
def test_wilson_interval_bounds_and_empty():
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(9, 10)
    assert 0 <= lo < 0.9 < hi <= 1
    # 全通过时上界仍为 1，下界随样本量收紧
    lo2, _ = wilson_interval(20, 20)
    assert lo2 > lo  # n 越大，下界越接近 1


def test_wilson_known_value():
    lo, hi = wilson_interval(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3


def test_aggregate_only_counts_judged_dims():
    results = [
        {"overall_pass": True, "scores": {"dims": {"task": {"pass": True}, "tool": {"pass": True}}}},
        {"overall_pass": False, "scores": {"dims": {"task": {"pass": False}}}},
        {"overall_pass": True, "scores": {"dims": {"safety": {"pass": True}}}},
    ]
    m = aggregate(results)
    assert m["task"] == {"n": 2, "passed": 1, "rate": 0.5, "ci95": wilson_interval(1, 2)}
    assert m["tool"]["n"] == 1 and m["tool"]["rate"] == 1.0
    assert m["hallucination"]["n"] == 0 and m["hallucination"]["rate"] is None
    assert m["_overall"]["n"] == 3 and m["_overall"]["passed"] == 2


# ---------------- 裁判 JSON 解析 ----------------
@pytest.mark.parametrize("raw", [
    '{"task_success": true, "hallucination": false, "safety_ok": true, "reasoning": "ok"}',
    '```json\n{"task_success": true, "hallucination": true, "safety_ok": false}\n```',
    '前置废话 {"task_success": false, "hallucination": false, "safety_ok": true} 后缀',
])
def test_parse_verdict_accepts_variants(raw):
    v = _parse_verdict(raw)
    assert isinstance(v, dict)
    assert set(v) == {"task_success", "hallucination", "safety_ok", "reasoning"}


@pytest.mark.parametrize("bad", ["", "没有 json", "{坏掉的", "[]"])
def test_parse_verdict_rejects_junk(bad):
    assert _parse_verdict(bad) is None


# ---------------- 超时降级 ----------------
def test_timeout_transcript_scores_as_failure():
    """驱动超时后按空轨迹计：该调工具的维度应判失败，整体不通过。"""
    from backend.eval.runner import _timeout_transcript

    tr = _timeout_transcript(_case("kn_return_policy"), "驱动超时 >90s")
    assert tr["turns"][0]["error"].startswith("驱动超时")
    res = score_case(_case("kn_return_policy"), tr, None, None)
    assert res["dims"]["tool"]["pass"] is False
    assert res["overall_pass"] is False


# ---------------- 路由形状（只读，不触发 run）----------------
def test_eval_readonly_routes():
    from starlette.testclient import TestClient
    from backend.server import app

    with TestClient(app) as c:
        s = c.get("/api/eval/summary")
        assert s.status_code == 200 and "available" in s.json()
        st = c.get("/api/eval/run/status")
        assert st.json()["running"] is False
        assert c.get("/api/eval/runs/does-not-exist").status_code == 404
