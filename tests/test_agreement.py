"""Judge agreement must be measured on what the judge sees in a real run.

Pins a real bug: the agreement script used to re-judge each labelled reply with
no tool calls, so the judge scored faithfulness blind and the agreement number
described a judge that never runs in the suite.
"""
from __future__ import annotations

from harness import agreement
from agents.support.cases import SupportCase as Case
from harness.schema import AgentOutput, JudgeVerdict

CASE = Case(id="order_status", message="Where is ORD-1041?", lang="en", account_id="ACC-77", expect_tool="lookup_order")

CALLS = [{
    "name": "lookup_order",
    "args": {"order_id": "ORD-1041", "account_id": "ACC-77"},
    "result": '{"found": true, "status": "shipped", "carrier": "Aramex"}',
}]

RUN = {"case_id": "order_status", "repetition": 0, "output": {"reply": "ORD-1041 shipped with Aramex.", "tool_calls": CALLS}}


def verdict(score: int) -> JudgeVerdict:
    return JudgeVerdict(helpfulness=score, faithfulness=score, tone=score, reasoning="r")


def test_label_rows_keep_the_tool_calls_and_existing_verdicts():
    [row] = agreement.label_rows([RUN], {("order_status", 0): "pass"}, limit=30)
    assert row["tool_calls"] == CALLS
    assert row["human_verdict"] == "pass"


def test_the_judge_sees_the_tool_calls_and_results():
    seen: list[AgentOutput] = []

    def judge(case: Case, out: AgentOutput) -> JudgeVerdict:
        seen.append(out)
        return verdict(5)

    rows = agreement.label_rows([RUN], {("order_status", 0): "pass"}, limit=30)
    agreement.measure(rows, {CASE.id: CASE}, judge, judge_min=4)
    assert seen[0].tool_calls == CALLS


def test_agreement_and_disagreements_are_counted():
    rows = [
        {**agreement.label_rows([RUN], {}, 30)[0], "human_verdict": "pass"},
        {**agreement.label_rows([RUN], {}, 30)[0], "repetition": 1, "human_verdict": "fail"},
    ]
    result = agreement.measure(rows, {CASE.id: CASE}, lambda c, o: verdict(5), judge_min=4)
    assert (result.agreed, result.total, result.rate) == (1, 2, 0.5)
    assert result.disagreements[0][:3] == ("order_status", "fail", "pass")


def test_rows_labelled_without_tool_calls_are_stale_not_judged_blind():
    old_row = {"case_id": "order_status", "repetition": 0, "reply": "x", "human_verdict": "pass"}
    called = []
    result = agreement.measure([old_row], {CASE.id: CASE}, lambda c, o: called.append(o) or verdict(5), judge_min=4)
    assert result.stale == ["order_status#0"] and result.total == 0 and not called


def test_unlabelled_rows_are_ignored():
    row = {**agreement.label_rows([RUN], {}, 30)[0], "human_verdict": ""}
    assert agreement.measure([row], {CASE.id: CASE}, lambda c, o: verdict(5), judge_min=4).total == 0


# --- an honest agreement number -------------------------------------------------


def labelled(verdicts: list[str]) -> list[dict]:
    """Rows labelled by a human, one per verdict."""
    base = agreement.label_rows([RUN], {}, 30)[0]
    return [{**base, "repetition": i, "human_verdict": v} for i, v in enumerate(verdicts)]


# 9 good answers, 1 bad one: the usual shape of a labelled set.
NINE_PASS_ONE_FAIL = labelled(["pass"] * 9 + ["fail"])


def test_a_judge_that_passes_everything_looks_good_on_raw_agreement_only():
    """The trap kappa exists for: 90% agreement, measuring nothing."""
    result = agreement.measure(NINE_PASS_ONE_FAIL, {CASE.id: CASE}, lambda c, o: verdict(5), judge_min=4)
    assert result.rate == 0.9
    assert result.kappa == 0.0
    assert result.lenient == 1  # and it waved through the one bad answer
    ok, reasons = agreement.trusted(result)
    assert not ok and any("kappa" in r for r in reasons) and any("lenient" in r for r in reasons)


def test_a_judge_that_tracks_the_human_is_trusted():
    labels = labelled(["pass"] * 7 + ["fail"] * 3)
    scores = iter([5] * 7 + [2] * 3)
    result = agreement.measure(labels, {CASE.id: CASE}, lambda c, o: verdict(next(scores)), judge_min=4)
    assert (result.rate, result.kappa) == (1.0, 1.0)
    assert agreement.trusted(result) == (True, [])


def test_disagreements_are_split_by_direction():
    labels = labelled(["pass", "pass", "fail", "fail"])
    scores = iter([2, 5, 5, 2])  # strict on #0, lenient on #2
    result = agreement.measure(labels, {CASE.id: CASE}, lambda c, o: verdict(next(scores)), judge_min=4)
    assert (result.strict, result.lenient) == (1, 1)


def test_a_strict_judge_can_still_be_trusted_a_lenient_one_cannot():
    labels = labelled(["pass"] * 14 + ["fail"] * 6)
    strict = iter([2] + [5] * 13 + [2] * 6)  # fails one good answer
    lenient = iter([5] * 14 + [5] + [2] * 5)  # passes one bad answer
    s = agreement.measure(labels, {CASE.id: CASE}, lambda c, o: verdict(next(strict)), judge_min=4)
    l = agreement.measure(labels, {CASE.id: CASE}, lambda c, o: verdict(next(lenient)), judge_min=4)
    assert s.rate == l.rate == 0.95
    assert agreement.trusted(s)[0] and not agreement.trusted(l)[0]


def test_kappa_is_undefined_when_every_label_and_verdict_agree_on_pass():
    """All-pass labels cannot tell a good judge from a rubber stamp; say so."""
    result = agreement.measure(labelled(["pass"] * 5), {CASE.id: CASE}, lambda c, o: verdict(5), judge_min=4)
    assert result.rate == 1.0 and result.kappa is None
    assert not agreement.trusted(result)[0]


def test_rows_without_a_verdict_are_listed_not_scored():
    result = agreement.measure(labelled(["pass", "fail"]), {CASE.id: CASE}, lambda c, o: None, judge_min=4)
    assert result.total == 0 and result.unjudged == ["order_status#0", "order_status#1"]
