"""Run every case N times, grade each run, aggregate into pass rates."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from harness import config, fixtures
from harness.agent import run_agent
from harness.graders import deterministic, judge as judge_mod
from harness.schema import Case, CaseSummary, GraderRate, RunResult, SuiteReport


def run_case(case: Case, repetitions: int, use_judge: bool) -> list[RunResult]:
    results: list[RunResult] = []
    for i in range(repetitions):
        # Offline, repetition i replays recording i, so the variance between
        # live runs survives into the offline pass rate.
        out = fixtures.load(case.id, i) if config.FAKE else run_agent(case.message, case.account_id)
        verdict = judge_mod.judge(case, out) if use_judge else None
        results.append(
            RunResult(
                case_id=case.id,
                repetition=i,
                output=out,
                deterministic=deterministic.grade(case, out),
                judge=verdict,
            )
        )
    return results


def summarise(case: Case, runs: list[RunResult]) -> CaseSummary:
    passes = sum(1 for r in runs if r.passed(config.JUDGE_MIN_SCORE))
    failures: list[str] = []
    for r in runs:
        failures.extend(r.failures())
        if r.judge is not None:
            low = min(r.judge.helpfulness, r.judge.faithfulness, r.judge.tone)
            if low < config.JUDGE_MIN_SCORE:
                failures.append(f"judge: {r.judge.reasoning} (low score {low})")
    # de-duplicate while preserving order
    seen: set[str] = set()
    unique = [f for f in failures if not (f in seen or seen.add(f))]
    return CaseSummary(case_id=case.id, tags=case.tags, runs=len(runs), passes=passes, failures=unique)


def grader_rates(runs: list[RunResult]) -> list[GraderRate]:
    """Per-grader pass rates over the runs each grader applied to.

    A case pass rate says which scenario broke; this says which property broke
    across scenarios -- "language_matches fails 1 run in 5" points at the
    prompt, wherever the case was.
    """
    rates: dict[str, GraderRate] = {}
    for run in runs:
        for g in run.deterministic:
            rate = rates.setdefault(g.name, GraderRate(name=g.name, passed=0, applicable=0))
            if g.applicable:
                rate.applicable += 1
                rate.passed += g.passed
    return list(rates.values())


def fixture_source(cases: list[Case]) -> str:
    """What an offline run is grading: real recordings, hand-written doubles, or both."""
    sources = set()
    for case in cases:
        for p in sorted((fixtures.FIXTURES / case.id).glob("*.json")):
            sources.add(json.loads(p.read_text(encoding="utf-8")).get("source", "recorded"))
    return "+".join(sorted(sources)) or "none"


def run_suite(cases: list[Case], repetitions: int | None = None, use_judge: bool = True) -> tuple[SuiteReport, list[RunResult]]:
    repetitions = repetitions or config.REPETITIONS
    summaries: list[CaseSummary] = []
    all_runs: list[RunResult] = []

    for case in cases:
        runs = run_case(case, repetitions, use_judge)
        all_runs.extend(runs)
        summary = summarise(case, runs)
        summaries.append(summary)
        mark = "PASS" if summary.pass_rate >= config.PASS_RATE_THRESHOLD else "FAIL"
        if summary.flaky:
            mark = "FLAKY"
        print(f"  {mark:5} {case.id:32} {summary.passes}/{summary.runs}")
        for f in summary.failures[:3]:
            print(f"        - {f}")

    report = SuiteReport(
        cases=summaries,
        agent_model=config.AGENT_MODEL,
        judge_model=config.JUDGE_MODEL if use_judge else "",
        repetitions=repetitions,
        judged=use_judge and not config.FAKE,
        graders=grader_rates(all_runs),
        source=fixture_source(cases) if config.FAKE else "live",
    )
    return report, all_runs


def save(report: SuiteReport, runs: list[RunResult]) -> tuple[str, str]:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = config.RESULTS / f"report-{stamp}.json"
    runs_path = config.RESULTS / f"runs-{stamp}.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    runs_path.write_text(
        json.dumps([r.model_dump() for r in runs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(report_path), str(runs_path)
