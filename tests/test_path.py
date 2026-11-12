"""Path checks, proven with planted faults.

Every fault here produces an answer whose figures are real -- the figure check
passes them all. Only the path shows the agent asked for the wrong slice of
the data: the wrong store, the wrong day, a month cut short.
"""
from __future__ import annotations

import json

import pytest

from agents.shop import dataset_cases, graders, tools
from harness import path
from harness.schema import AgentOutput


def call(name: str, **args) -> dict:
    result, is_error = tools.execute(name, args)
    return {"name": name, "args": args, "result": result if is_error else json.dumps(result), "is_error": is_error}


def sales(store: str, start: str, end: str) -> dict:
    return call("get_sales", store_id=store, start_date=start, end_date=end)


CASES = dataset_cases()
YESTERDAY, LAST_WEEK, BEST_STORE = CASES["revenue-yesterday"], CASES["store-last-week"], CASES["best-store-september"]


def run(case, calls, reply="ok") -> dict[str, bool]:
    out = AgentOutput(reply=reply, tool_calls=calls)
    return {g.name: g.passed for g in (graders.right_calls(case, out), graders.no_repeated_calls(case, out))}


# --- every valid path passes ------------------------------------------------------

VALID = [
    ("yesterday, all stores at once", YESTERDAY, [sales("all", "2026-10-14", "2026-10-14")]),
    ("yesterday, store by store", YESTERDAY, [sales(s, "2026-10-14", "2026-10-14") for s in ("online", "harbor", "mill")]),
    ("last week as Monday to Sunday", LAST_WEEK, [sales("mill", "2026-10-05", "2026-10-11")]),
    ("last week as the last 7 days", LAST_WEEK, [sales("mill", "2026-10-08", "2026-10-14")]),
    ("each store, any order, after listing stores", BEST_STORE,
     [call("list_stores")] + [sales(s, "2026-09-01", "2026-09-30") for s in ("mill", "online", "harbor")]),
]


@pytest.mark.parametrize("label,case,calls", VALID, ids=[v[0] for v in VALID])
def test_valid_paths_pass(label, case, calls):
    assert run(case, calls) == {"right_calls": True, "no_repeated_calls": True}


# --- planted faults are caught ------------------------------------------------------

FAULTS = [
    ("the wrong store", LAST_WEEK, [sales("harbor", "2026-10-05", "2026-10-11")], "right_calls"),
    ("today, not yesterday (no data yet)", YESTERDAY, [sales("all", "2026-10-15", "2026-10-15")], "right_calls"),
    ("the day before yesterday", YESTERDAY, [sales("all", "2026-10-13", "2026-10-13")], "right_calls"),
    ("one store missing from the comparison", BEST_STORE,
     [sales(s, "2026-09-01", "2026-09-30") for s in ("harbor", "online")], "right_calls"),
    ("September cut short", BEST_STORE, [sales(s, "2026-09-01", "2026-09-29") for s in ("harbor", "mill", "online")], "right_calls"),
    ("a week that is neither reading", LAST_WEEK, [sales("mill", "2026-10-04", "2026-10-10")], "right_calls"),
    ("half of the per-store path", YESTERDAY, [sales("harbor", "2026-10-14", "2026-10-14")], "right_calls"),
    ("the same call twice", YESTERDAY, [sales("all", "2026-10-14", "2026-10-14")] * 2, "no_repeated_calls"),
    ("retrying a failed call unchanged", YESTERDAY,
     [sales("all", "2026-10-15", "2026-10-15")] * 2 + [sales("all", "2026-10-14", "2026-10-14")], "no_repeated_calls"),
]


@pytest.mark.parametrize("label,case,calls,grader", FAULTS, ids=[f[0] for f in FAULTS])
def test_planted_faults_are_caught(label, case, calls, grader):
    assert run(case, calls)[grader] is False


def test_a_failed_call_never_satisfies_a_requirement():
    """Asking for the right thing and getting an error is not having the data."""
    spec = {"tool": "get_sales", "store_id": "all", "start_date": "2026-10-14", "end_date": "2026-10-14"}
    failed = {**sales("all", "2026-10-14", "2026-10-14"), "is_error": True}
    assert not path.satisfied([spec], [failed])


def test_failure_names_what_was_needed_and_what_was_made():
    out = AgentOutput(reply="ok", tool_calls=[sales("harbor", "2026-10-05", "2026-10-11")])
    detail = graders.right_calls(LAST_WEEK, out).detail
    assert detail.startswith("missing (get_sales(store_id=mill") and "made ['get_sales(store_id=harbor" in detail


# --- the answer names the period it used ----------------------------------------------

MON_SUN = [sales("mill", "2026-10-05", "2026-10-11")]


@pytest.mark.parametrize("reply", [
    "Last week (Monday 5 to Sunday 11 October) Mill Lane took $2,930.00.",
    "Mill Lane, October 5-11: $2,930.00.",
    "From 2026-10-05 to 2026-10-11, Mill Lane made $2,930.00.",
    "الأسبوع الماضي (٥ إلى ١١ أكتوبر) حقق فرع ميل لين ٢٬٩٣٠ دولار.",
])
def test_period_stated_in_any_form_passes(reply):
    assert graders.states_period(LAST_WEEK, AgentOutput(reply=reply, tool_calls=MON_SUN)).passed


@pytest.mark.parametrize("label,reply", [
    ("no dates at all", "Last week Mill Lane took $2,930.00."),
    ("dates of the other reading", "Over 8-14 October Mill Lane took $2,930.00."),
    ("only the start date", "Since 5 October Mill Lane took $2,930.00."),
])
def test_period_faults_are_caught(label, reply):
    assert not graders.states_period(LAST_WEEK, AgentOutput(reply=reply, tool_calls=MON_SUN)).passed
