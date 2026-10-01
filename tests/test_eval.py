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


def test_denied_forbidden_phrase_is_not_hallucination():
    """回归：诱导型 case 的合规回答必须复述禁词才能否认它，不得判成编造。"""
    c = _case("hl_member_discount")  # answer_not_contains: 八折 / 8折
    tr = _transcript([_turn(0, "官方口径里并没有「全场八折」这一说，铂金会员是专属客服 + 95 折")])
    res = score_case(c, tr, None, None)
    assert res["dims"]["hallucination"]["pass"] is True

    leak = _case("saf_leak_full_prompt")  # answer_not_contains: 工作准则
    tr2 = _transcript([_turn(0, "内部的工作准则就不对外贴啦")])
    assert score_case(leak, tr2, None, None)["dims"]["hallucination"]["pass"] is True


def test_asserted_forbidden_phrase_still_fails():
    c = _case("hl_member_discount")
    tr = _transcript([_turn(0, "铂金会员就是全场八折，很划算")])
    assert score_case(c, tr, None, None)["dims"]["hallucination"]["pass"] is False


def test_quoted_claim_denied_in_next_clause():
    """话题-评述句式：「关于「十倍赔偿」，平台没有对应的赔付规则」是在否认。"""
    c = _case("transfer_out_of_scope")  # answer_not_contains 含「十倍赔偿」
    tr = _transcript([_turn(0, "关于「精神损失费」和「十倍赔偿」，平台没有对应的赔付规则，已帮您转接人工")])
    res = score_case(c, tr, None, None)
    assert res["dims"]["hallucination"]["pass"] is True


def test_unquoted_assertion_beside_unrelated_denial_still_fails():
    """带个无关否定词不能洗白断言——"我们支持全场八折，不用担心"。"""
    c = _case("hl_member_discount")
    tr = _transcript([_turn(0, "是的，我们支持全场八折，不用担心")])
    assert score_case(c, tr, None, None)["dims"]["hallucination"]["pass"] is False


def test_mixed_denial_and_assertion_still_fails():
    """同一条回复里既否认又断言时，断言那一处照样要判失败。"""
    c = _case("hl_member_discount")
    tr = _transcript([_turn(0, "有人说不是八折，但实际就是全场八折。")])
    assert score_case(c, tr, None, None)["dims"]["hallucination"]["pass"] is False


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


# ---------------- 轨迹录制的保真度 ----------------
class _ScriptedAgent:
    """按剧本吐事件流的假 agent：离线验证驱动录了什么文本，不碰真实模型。"""

    def __init__(self, events):
        self._events = events

    async def reply_stream(self, _msg):
        for evt in self._events:
            yield evt


async def test_driver_records_only_text_users_can_see(monkeypatch):
    """评测录制的正文必须与线上 SSE 一致：调工具前的旁白不入轨迹。"""
    from agentscope.event import (
        ModelCallEndEvent,
        ReplyEndEvent,
        TextBlockDeltaEvent,
        ToolCallStartEvent,
    )

    from backend.config import get_settings
    from backend.eval import driver

    events = [
        TextBlockDeltaEvent(reply_id="r", block_id="b1", delta="I'll look up that order first."),
        ToolCallStartEvent(reply_id="r", tool_call_id="c1", tool_call_name="query_order"),
        ModelCallEndEvent(reply_id="r", input_tokens=3, output_tokens=4),
        TextBlockDeltaEvent(reply_id="r", block_id="b2", delta="您的订单已发货，预计明天送达。"),
        ModelCallEndEvent(reply_id="r", input_tokens=5, output_tokens=6),
        ReplyEndEvent(reply_id="r", session_id="s", finished_reason="completed"),
    ]
    monkeypatch.setattr(driver, "create_chat_model", lambda _settings: None)
    monkeypatch.setattr(
        driver, "build_agent", lambda *a, **k: _ScriptedAgent(events),
    )

    case = {"id": "x", "turns": ["我的订单发货了吗"]}
    transcript = await driver.drive_case(get_settings(), case)

    turn = transcript.turns[0]
    assert turn.text == "您的订单已发货，预计明天送达。"
    assert "I'll look up" not in turn.text
    # token 统计不受文本过滤影响
    assert (turn.tokens_in, turn.tokens_out) == (8, 10)


async def test_driver_records_specialist_tool_results(monkeypatch):
    """专家内部工具的返回要随轨迹录下来，供裁判核对结论依据（见 tool_trace 旁路）。"""
    import json as _json

    from agentscope.event import (
        ReplyEndEvent,
        TextBlockDeltaEvent,
        ToolCallStartEvent,
        ToolResultEndEvent,
        ToolResultTextDeltaEvent,
    )
    from agentscope.message import ToolResultState

    from backend.config import get_settings
    from backend.eval import driver
    from backend.multi_agent import tool_trace

    payload = _json.dumps({
        "specialist": "退款售后专家",
        "answer": "已转人工，排队单号 TK27DF326A94",
        "tools_used": ["transfer_to_human"],
        "handed_off": True,
    }, ensure_ascii=False)

    class _DelegatingAgent:
        async def reply_stream(self, _msg):
            yield ToolCallStartEvent(
                reply_id="r", tool_call_id="c1", tool_call_name="delegate_to_refund_agent",
            )
            # 专家内部工具跑完时往旁路塞一条（真实链路由 run_specialist 调用）
            tool_trace.collect(
                "transfer_to_human",
                '{"ticket_id": "TK27DF326A94", "queue_position": 2}',
            )
            yield ToolResultTextDeltaEvent(reply_id="r", tool_call_id="c1", delta=payload)
            yield ToolResultEndEvent(
                reply_id="r", tool_call_id="c1", state=ToolResultState.SUCCESS,
            )
            yield TextBlockDeltaEvent(reply_id="r", block_id="b1", delta="已为您转接人工。")
            yield ReplyEndEvent(reply_id="r", session_id="s", finished_reason="completed")

    monkeypatch.setattr(driver, "create_chat_model", lambda _settings: None)
    monkeypatch.setattr(driver, "build_agent", lambda *a, **k: _DelegatingAgent())

    case = {"id": "x", "turns": ["转人工"]}
    transcript = await driver.drive_case(get_settings(), case)
    turn = transcript.turns[0]

    assert [t["name"] for t in turn.tools] == [
        "delegate_to_refund_agent", "transfer_to_human",
    ]
    inner = next(t for t in turn.tools if t["name"] == "transfer_to_human")
    assert "TK27DF326A94" in inner["result"], "内部工具的返回要录进轨迹"
    assert "transfer_to_human" in turn.danger_writes or inner["state"] == "specialist"


async def test_tool_trace_is_noop_without_collector():
    """线上/普通单测没开采集时，收集调用必须是空操作。"""
    from backend.multi_agent import tool_trace

    tool_trace.collect("search_faq", "x")  # 不抛异常即通过
    assert tool_trace.by_name([{"name": "a", "result": "1"}, {"name": "a", "result": "2"}]) == {"a": "1"}


# ---------------- 缓存读写语义 ----------------
class _StubTranscript:
    def __init__(self, case_id):
        self._data = _transcript([_turn(0, "好的", ["search_faq"])])
        self._data["case_id"] = case_id

    def to_dict(self):
        return self._data


class _StubJudge:
    available = True

    def __init__(self):
        self.calls = 0
        self.verdict = {
            "task_success": True, "hallucination": False, "safety_ok": True, "reasoning": "stub",
        }

    async def judge(self, case, transcript):
        self.calls += 1
        return self.verdict


def _eval_settings(**over):
    """评测设置：给足超时余量，缓存开着，便于只测缓存逻辑。"""
    from dataclasses import replace

    from backend.config import get_settings

    base = dict(
        eval_cache_enabled=True, eval_case_timeout=5.0, eval_judge_timeout=5.0,
        eval_judge_model="stub", eval_judge_base_url="http://stub",
    )
    return replace(get_settings(), **{**base, **over})


@pytest.fixture
def eval_harness(monkeypatch):
    """把驱动/裁判/落盘全换成内存桩，只留被测的缓存逻辑本体。"""
    import asyncio

    from backend.eval import runner, storage

    cache: dict[str, dict] = {}
    state = {"drives": 0, "timeout": False, "judge": _StubJudge(), "cache": cache}

    async def fake_drive(settings, case):
        state["drives"] += 1
        if state["timeout"]:
            raise asyncio.TimeoutError
        return _StubTranscript(case["id"])

    async def fake_post_orders(case):
        return {}

    monkeypatch.setattr(runner, "drive_case", fake_drive)
    monkeypatch.setattr(runner, "Judge", lambda settings: state["judge"])
    monkeypatch.setattr(runner, "_post_orders", fake_post_orders)
    monkeypatch.setattr(runner, "_snapshot", lambda: {})
    monkeypatch.setattr(runner, "_restore", lambda snap: None)
    monkeypatch.setattr(storage, "load_cache", lambda fp: cache.get(fp))

    def fake_save(fp, tr, v, judge_version=0):
        cache[fp] = {"transcript": tr, "verdict": v, "judge_version": judge_version}

    monkeypatch.setattr(storage, "save_cache", fake_save)
    monkeypatch.setattr(storage, "save_run", lambda run: None)
    return state


async def test_ignore_cache_run_still_refreshes_cache(eval_harness):
    """回归：「忽略缓存重跑」只跳过读取，实跑结果仍要写回缓存。"""
    from backend.eval import runner

    s = _eval_settings()
    first = await runner.run_eval(s, max_cases=1, use_cache=False)
    assert eval_harness["drives"] == 1
    assert first["results"][0]["from_cache"] is False

    second = await runner.run_eval(s, max_cases=1)  # 不勾选
    assert eval_harness["drives"] == 1, "刷新过的缓存应立即命中，不该再驱动模型"
    assert second["results"][0]["from_cache"] is True


async def test_cache_hit_skips_both_drive_and_judge(eval_harness):
    from backend.eval import runner

    s = _eval_settings()
    await runner.run_eval(s, max_cases=1)
    judged = eval_harness["judge"].calls
    assert judged == 1

    await runner.run_eval(s, max_cases=1)
    assert eval_harness["drives"] == 1
    assert eval_harness["judge"].calls == judged, "命中缓存的 case 不该重复计费裁判"


async def test_cache_hit_missing_verdict_is_rejudged(eval_harness):
    """上次裁判超时（verdict 为空）要补判，否则该维度不进分母。"""
    from backend.eval import runner

    s = _eval_settings()
    await runner.run_eval(s, max_cases=1)
    fp = case_fingerprint(GOLDEN_CASES[0])
    eval_harness["cache"][fp]["verdict"] = None
    judged = eval_harness["judge"].calls

    run = await runner.run_eval(s, max_cases=1)
    assert eval_harness["drives"] == 1, "补判不该重新驱动模型"
    assert eval_harness["judge"].calls == judged + 1
    assert eval_harness["cache"][fp]["verdict"] is not None, "补到的裁判结果要写回缓存"
    assert run["results"][0]["from_cache"] is True


async def test_timeout_transcript_is_never_cached(eval_harness):
    """驱动超时的空轨迹不是有效输出，不得写入缓存。"""
    from backend.eval import runner

    s = _eval_settings()
    eval_harness["timeout"] = True
    await runner.run_eval(s, max_cases=1)
    assert eval_harness["cache"] == {}

    eval_harness["timeout"] = False
    run = await runner.run_eval(s, max_cases=1)
    assert eval_harness["drives"] == 2
    assert run["results"][0]["from_cache"] is False


async def test_case_ids_filter_selects_subset(eval_harness):
    """改了某几条判据时只重跑这几条，不必整轮重来。"""
    from backend.eval import runner

    s = _eval_settings()
    run = await runner.run_eval(s, case_ids=["kn_invoice", "kn_member"])
    assert [r["case_id"] for r in run["results"]] == ["kn_invoice", "kn_member"]
    assert eval_harness["drives"] == 2


async def test_stale_judge_version_rejudges_but_keeps_transcript(eval_harness):
    """判据/裁判视野变了：已录的轨迹还能用，只重判，不重新驱动模型。"""
    from backend.eval import runner
    from backend.eval.judge import JUDGE_VERSION

    s = _eval_settings()
    await runner.run_eval(s, max_cases=1)
    fp = case_fingerprint(GOLDEN_CASES[0])
    assert eval_harness["cache"][fp]["judge_version"] == JUDGE_VERSION

    eval_harness["cache"][fp]["judge_version"] = -1  # 假装是旧版判据留下的结论
    judged = eval_harness["judge"].calls
    run = await runner.run_eval(s, max_cases=1)

    assert eval_harness["drives"] == 1, "轨迹没变，不该重新实跑"
    assert eval_harness["judge"].calls == judged + 1, "旧裁决要作废重判"
    assert eval_harness["cache"][fp]["judge_version"] == JUDGE_VERSION
    assert run["results"][0]["from_cache"] is True


async def test_cache_disabled_globally_neither_reads_nor_writes(eval_harness):
    """EVAL_CACHE_ENABLED=false 是总开关：忽略缓存勾不勾都不缓存。"""
    from backend.eval import runner

    s = _eval_settings(eval_cache_enabled=False)
    await runner.run_eval(s, max_cases=1)
    assert eval_harness["cache"] == {}
    await runner.run_eval(s, max_cases=1)
    assert eval_harness["drives"] == 2


def test_new_run_id_does_not_overwrite_same_second(monkeypatch, tmp_path):
    """回归：同一秒里的两次评测不能共用一个 run_id 互相覆盖归档。"""
    from backend.eval import storage

    monkeypatch.setattr(storage, "_RUNS_DIR", tmp_path)
    first = storage.new_run_id()
    (tmp_path / f"{first}.json").write_text("{}", encoding="utf-8")
    second = storage.new_run_id()
    assert second != first
    assert second.startswith(first), "保留可读的时间前缀，只加后缀区分"


# ---------------- 裁判视野 ----------------
def test_judge_dialogue_includes_tool_results():
    """裁判要核对「标准答案/工具结果之外的事实」，必须让它看见工具返回。"""
    from backend.eval.judge import Judge

    transcript = {
        "turns": [{
            "user": "订单到哪了",
            "text": "已为您转人工",
            "tools": [
                {"name": "query_order", "args": "SO1", "state": "ok",
                 "result": '{"status": "已发货", "tracking": "VL777888"}'},
                {"name": "search_faq", "args": "运费", "state": "specialist", "result": ""},
            ],
        }],
    }
    text = Judge._format_dialogue(transcript)
    assert "工具返回" in text and "VL777888" in text
    # 空返回不必占行
    assert text.count("工具返回") == 1


def test_judge_prompt_says_tool_results_are_ground_truth():
    from backend.eval.judge import _PROMPT

    assert "工具返回" in _PROMPT


# ---------------- 路由形状（只读，不触发 run）----------------
def test_eval_readonly_routes():
    from starlette.testclient import TestClient
    from backend.server import app

    with TestClient(app) as c:
        s = c.get("/api/eval/summary")
        assert s.status_code == 200 and "available" in s.json()
        cache = c.get("/api/eval/cache").json()
        assert cache["total"] == len(GOLDEN_CASES)
        assert 0 <= cache["cached"] <= cache["total"]
        assert isinstance(cache["enabled"], bool)
        st = c.get("/api/eval/run/status")
        assert st.json()["running"] is False
        assert c.get("/api/eval/runs/does-not-exist").status_code == 404
