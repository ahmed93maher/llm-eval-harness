"""The assessment is read by someone who did not run the eval, so what it
claims -- and above all what it says it does NOT show -- is tested."""
from __future__ import annotations

from harness import assessment
from harness.schema import Case

CASES = {
    "a": Case(id="a", message="What was revenue yesterday?", lang="en", tags=["figures"]),
    "b": Case(id="b", message="How are sales?", lang="en", tags=["clarification"]),
}


def run(case_id: str, rep: int, passed: bool, **output) -> dict:
    grade = {"name": "figures_grounded", "passed": passed, "applicable": True,
             "detail": "" if passed else "not in the tool results: ['$9.99']"}
    return {"case_id": case_id, "repetition": rep, "output": {"reply": f"answer {rep}", **output},
            "deterministic": [grade, {"name": "asks", "passed": True, "applicable": case_id == "b", "detail": ""}]}


RUNS = [run("a", 0, True), run("a", 1, False), run("b", 0, True), run("b", 1, True)]
REPORT = {
    "agent": "shop", "agent_model": "m", "repetitions": 2, "judged": False, "source": "hand-authored+planted-fault",
    "cases": [{"case_id": "a", "tags": ["figures"], "runs": 2, "passes": 1, "failures": []},
              {"case_id": "b", "tags": ["clarification"], "runs": 2, "passes": 2, "failures": []}],
    "graders": [{"name": "figures_grounded", "passed": 3, "applicable": 4}, {"name": "asks", "passed": 2, "applicable": 2}],
}
DESC = {"figures_grounded": "Every figure traces to the data."}


def build(**kw) -> str:
    return assessment.build({**REPORT, **kw.pop("report", {})}, kw.pop("runs", RUNS), CASES, DESC, **kw)


def test_leads_with_the_answer():
    md = build()
    summary = md.split("## What this does not show")[0]
    assert "75% (3/4) of runs passed every check" in summary
    assert "1 passed only some (flaky)" in summary
    assert "**figures_grounded** — Every figure traces to the data. Failed 1 of 4 runs." in summary


def test_says_what_it_does_not_show():
    caveats = build().split("## What this does not show")[1].split("## By check")[0]
    assert "replayed from fixtures" in caveats  # not the model's behaviour
    assert "No LLM judge" in caveats
    assert "2 repetitions per case" in caveats
    assert "Thinly tested" in caveats and "`asks`" in caveats  # 2 runs < 3


def test_a_live_judged_well_repeated_run_has_fewer_caveats():
    caveats = build(report={"source": "live", "judged": True, "repetitions": 5}).split("## What this does not show")[1]
    caveats = caveats.split("## By check")[0]
    assert "replayed" not in caveats and "No LLM judge" not in caveats and "repetitions per case" not in caveats


def test_every_failure_comes_with_its_evidence():
    evidence = build().split("## Failures, with evidence")[1]
    assert "**`a`, run 2.**" in evidence
    assert "- Asked: What was revenue yesterday?" in evidence
    assert "- Answered: answer 1" in evidence
    assert "- Why it failed: not in the tool results: ['$9.99']" in evidence


def test_cost_is_not_guessed_when_not_measured():
    assert "Not measured" in build().split("## Cost and latency per task")[1]


def test_cost_and_latency_when_measured():
    measured = [run("a", i, True, input_tokens=1000, output_tokens=200, turns=2, latency_ms=1000 * (i + 1)) for i in range(4)]
    md = build(runs=measured, price_per_mtok=(1.0, 5.0)).split("## Cost and latency per task")[1]
    assert "| Input tokens | median 1,000 |" in md
    assert "| Cost | median $0.0020, total $0.01 for 4 runs |" in md  # 1000*1 + 200*5 per million
    assert "| Latency | median 2.5 s, p95 4.0 s |" in md


def test_cost_needs_a_price_and_says_so():
    measured = [run("a", 0, True, input_tokens=1000, output_tokens=200, turns=1)]
    assert "set the model's price" in build(runs=measured)


def test_descriptions_drop_code_pointers_and_whitespace():
    doc = """Every figure is a tool result, or plain arithmetic on tool
    results (see harness/figures.py). The check for the wrong number.

    Details nobody reading the report needs."""
    assert assessment.describe(doc) == "Every figure is a tool result, or plain arithmetic on tool results. The check for the wrong number."
