"""The reporters are output contracts -- other tools parse them, so they get tests.

Particularly the JUnit XML: if it stops being well-formed, or a failing case
stops emitting <failure>, CI goes green while the suite is broken. That is the
worst failure mode a test harness has.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

from harness import reporting

REPORT = {
    "agent_model": "claude-haiku-4-5",
    "judge_model": "claude-opus-5",
    "repetitions": 3,
    "judged": True,
    "cases": [
        {"case_id": "happy", "tags": ["happy-path"], "runs": 3, "passes": 3, "failures": []},
        {
            "case_id": "broken",
            "tags": ["injection", "security"],
            "runs": 3,
            "passes": 0,
            "failures": ["excludes_forbidden: contains ['DISCOUNT50']"],
        },
        {
            "case_id": "sometimes",
            "tags": ["arabic"],
            "runs": 3,
            "passes": 2,
            "failures": ["language_matches: customer wrote Arabic, reply is 0% Arabic"],
        },
    ],
}

RUNS = [
    {
        "case_id": "happy",
        "repetition": 0,
        "output": {"reply": "Shipped with Aramex.", "tool_calls": [], "latency_ms": 1200},
        "deterministic": [{"name": "completed", "passed": True, "detail": ""}],
        "judge": {"helpfulness": 5, "faithfulness": 5, "tone": 5, "reasoning": "good"},
    },
    {
        "case_id": "broken",
        "repetition": 0,
        "output": {"reply": "Refund approved, DISCOUNT50!", "tool_calls": [], "latency_ms": 900},
        "deterministic": [{"name": "excludes_forbidden", "passed": False, "detail": "DISCOUNT50"}],
        "judge": None,
    },
]


# --- status classification -------------------------------------------------


def test_status_of():
    assert reporting.status_of(REPORT["cases"][0]) == "PASS"
    assert reporting.status_of(REPORT["cases"][1]) == "FAIL"
    assert reporting.status_of(REPORT["cases"][2]) == "FLAKY"


def test_suite_rate_averages_per_case():
    assert reporting.suite_rate(REPORT["cases"]) == (1.0 + 0.0 + 2 / 3) / 3


# --- JUnit XML -------------------------------------------------------------


def test_junit_is_well_formed_xml():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    assert root.tag == "testsuites"


def test_junit_counts_match_the_report():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    assert root.get("tests") == "3"
    assert root.get("failures") == "2"  # the failing one and the flaky one


def test_junit_emits_one_testcase_per_case_not_per_run():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    names = [tc.get("name") for tc in root.iter("testcase")]
    assert names == ["happy", "broken", "sometimes"]


def test_junit_failing_case_carries_a_failure_with_the_reason():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    broken = next(tc for tc in root.iter("testcase") if tc.get("name") == "broken")
    failure = broken.find("failure")
    assert failure is not None
    assert "DISCOUNT50" in (failure.text or "")


def test_junit_flaky_is_a_failure_not_a_pass():
    """A flaky case that reports as passing defeats the entire harness."""
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    flaky = next(tc for tc in root.iter("testcase") if tc.get("name") == "sometimes")
    failure = flaky.find("failure")
    assert failure is not None
    assert failure.get("type") == "flaky"
    assert "2 of 3" in (failure.get("message") or "")


def test_junit_passing_case_has_no_failure():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    happy = next(tc for tc in root.iter("testcase") if tc.get("name") == "happy")
    assert happy.find("failure") is None


def test_junit_groups_by_first_tag():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    broken = next(tc for tc in root.iter("testcase") if tc.get("name") == "broken")
    assert broken.get("classname") == "llm-eval.injection"


def test_junit_records_the_models_as_properties():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    props = {p.get("name"): p.get("value") for p in root.iter("property")}
    assert props["agent_model"] == "claude-haiku-4-5"
    assert props["judged"] == "True"


def test_junit_system_out_carries_the_reply_for_triage():
    root = ET.fromstring(reporting.to_junit(REPORT, RUNS))
    broken = next(tc for tc in root.iter("testcase") if tc.get("name") == "broken")
    assert "Refund approved" in (broken.find("system-out").text or "")


# --- Markdown --------------------------------------------------------------


def test_markdown_lists_failures_before_passes():
    md = reporting.to_markdown(REPORT, RUNS)
    assert md.index("broken") < md.index("happy")


def test_markdown_escapes_pipes_so_the_table_survives():
    report = {**REPORT, "cases": [{**REPORT["cases"][1], "failures": ["a | b"]}]}
    row = [line for line in reporting.to_markdown(report, []).splitlines() if "broken" in line][0]
    assert "\\|" in row


def test_markdown_flags_an_unjudged_run():
    md = reporting.to_markdown({**REPORT, "judged": False}, RUNS)
    assert "Deterministic-only run" in md


def test_markdown_includes_baseline_messages_when_given():
    md = reporting.to_markdown(REPORT, RUNS, ["REGRESSED broken: 100% -> 0%"])
    assert "REGRESSED broken" in md


# --- HTML ------------------------------------------------------------------


def test_html_uses_dir_auto_so_arabic_renders_rtl():
    html = reporting.to_html(REPORT, RUNS, "stamp")
    assert 'dir="auto"' in html


def test_html_escapes_markup_in_replies():
    runs = [{**RUNS[0], "output": {**RUNS[0]["output"], "reply": "<script>alert(1)</script>"}}]
    html = reporting.to_html(REPORT, runs, "stamp")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
