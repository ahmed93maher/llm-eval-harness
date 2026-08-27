"""The dataset and the aggregation logic are tested too.

A harness that miscounts pass rates is worse than no harness -- it reports
confidence it hasn't earned.
"""
from __future__ import annotations

import pytest

from harness import dataset
from harness.runner import summarise
from harness.schema import AgentOutput, Case, GraderResult, RunResult


# --- the shipped dataset must be valid -------------------------------------


def test_shipped_dataset_is_valid():
    cases = dataset.load_cases()
    df = dataset.validate(cases)
    assert len(df) == len(cases)


def test_dataset_has_adversarial_coverage():
    tags = {t for c in dataset.load_cases() for t in c.tags}
    for required in ("injection", "missing-data", "pii", "arabic", "hallucination"):
        assert required in tags, f"no case tagged {required}"


def test_duplicate_ids_rejected():
    c = Case(id="dup", message="hi", lang="en", account_id="A", tags=["x"], must_refuse=True)
    with pytest.raises(dataset.DatasetError, match="duplicate"):
        dataset.validate([c, c.model_copy()])


def test_case_without_expectation_rejected():
    """A case that asserts nothing always passes and proves nothing."""
    toothless = Case(id="t", message="hi", lang="en", account_id="A", tags=["x"])
    arabic = Case(id="a", message="مرحبا", lang="ar", account_id="A", tags=["injection", "pii", "missing-data"], must_refuse=True)
    with pytest.raises(dataset.DatasetError, match="no expectation"):
        dataset.validate([toothless, arabic])


# --- pass-rate aggregation -------------------------------------------------


def _run(case_id: str, rep: int, passed: bool) -> RunResult:
    return RunResult(
        case_id=case_id,
        repetition=rep,
        output=AgentOutput(reply="ok"),
        deterministic=[GraderResult(name="g", passed=passed, detail="" if passed else "boom")],
    )


def test_flaky_detection():
    case = Case(id="c", message="hi", lang="en", account_id="A", tags=["x"], must_refuse=True)
    runs = [_run("c", 0, True), _run("c", 1, False), _run("c", 2, True)]
    summary = summarise(case, runs)
    assert summary.passes == 2
    assert summary.pass_rate == pytest.approx(2 / 3)
    assert summary.flaky is True


def test_all_pass_is_not_flaky():
    case = Case(id="c", message="hi", lang="en", account_id="A", tags=["x"], must_refuse=True)
    summary = summarise(case, [_run("c", i, True) for i in range(3)])
    assert summary.pass_rate == 1.0
    assert summary.flaky is False


def test_all_fail_is_not_flaky_it_is_broken():
    case = Case(id="c", message="hi", lang="en", account_id="A", tags=["x"], must_refuse=True)
    summary = summarise(case, [_run("c", i, False) for i in range(3)])
    assert summary.pass_rate == 0.0
    assert summary.flaky is False
    assert summary.failures


def test_judge_below_threshold_fails_the_run():
    from harness.schema import JudgeVerdict

    r = RunResult(
        case_id="c",
        repetition=0,
        output=AgentOutput(reply="ok"),
        deterministic=[GraderResult(name="g", passed=True)],
        judge=JudgeVerdict(helpfulness=5, faithfulness=2, tone=5, reasoning="invented a date"),
    )
    assert r.deterministic_passed is True
    assert r.passed(judge_min=4) is False  # one low dimension sinks the run
