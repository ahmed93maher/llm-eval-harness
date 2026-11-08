"""Regression gating against a committed baseline.

The baseline is a file of per-case pass rates from a known-good run. A new run is
compared against it, and CI fails when quality drops -- which is drift detection
with the honest name on it. Prompt edits, model upgrades and dependency bumps all
land here.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness import config
from harness.schema import SuiteReport

def path_for(agent: str) -> Path:
    """One baseline per agent: their pass rates measure different things."""
    return config.BASELINES / f"{agent}.json"


def write(report: SuiteReport, path: Path | None = None) -> Path:
    path = path or path_for(report.agent)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_model": report.agent_model,
        "judge_model": report.judge_model,
        "repetitions": report.repetitions,
        "judged": report.judged,
        "suite_pass_rate": report.suite_pass_rate,
        "cases": {c.case_id: c.pass_rate for c in report.cases},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def compare(report: SuiteReport, path: Path | None = None) -> tuple[bool, list[str]]:
    """Returns (ok, messages). ok=False means this run is a regression."""
    path = path or path_for(report.agent)
    if not path.exists():
        return True, [f"no baseline at {path} -- run scripts/update_baseline.py --agent {report.agent} to create one"]

    base = json.loads(path.read_text(encoding="utf-8"))
    messages: list[str] = []
    ok = True

    if base.get("judged") and not report.judged:
        messages.append("baseline is judged but this run is deterministic-only -- not comparable")
        return True, messages

    if base.get("agent_model") != report.agent_model:
        messages.append(
            f"model changed: baseline {base.get('agent_model')} -> now {report.agent_model}"
        )

    base_cases: dict[str, float] = base.get("cases", {})
    for case in report.cases:
        before = base_cases.get(case.case_id)
        if before is None:
            messages.append(f"NEW    {case.case_id}: {case.pass_rate:.0%} (not in baseline)")
            continue
        delta = case.pass_rate - before
        if delta < -config.BASELINE_TOLERANCE:
            ok = False
            messages.append(
                f"REGRESSED {case.case_id}: {before:.0%} -> {case.pass_rate:.0%} ({delta:+.0%})"
            )
        elif delta > config.BASELINE_TOLERANCE:
            messages.append(
                f"IMPROVED  {case.case_id}: {before:.0%} -> {case.pass_rate:.0%} ({delta:+.0%})"
            )

    for missing in set(base_cases) - {c.case_id for c in report.cases}:
        messages.append(f"MISSING {missing}: in baseline but not in this run")

    suite_delta = report.suite_pass_rate - base.get("suite_pass_rate", 0.0)
    messages.append(
        f"suite: {base.get('suite_pass_rate', 0):.1%} -> {report.suite_pass_rate:.1%} ({suite_delta:+.1%})"
    )
    return ok, messages
