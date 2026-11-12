"""Judge-vs-human agreement: the calibration that makes judged scores meaningful.

Kept here rather than in scripts/judge_agreement.py so it can be unit-tested;
the script is a thin command-line wrapper.

The judge must be measured on exactly what it sees in a real run: the reply AND
the tool calls with their results. Measured on the reply alone, it cannot tell
a grounded claim from an invented one, and the agreement number describes a
judge that never runs in the suite.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from harness.schema import AgentOutput, Case, JudgeVerdict

JudgeFn = Callable[[Case, AgentOutput], JudgeVerdict | None]


def label_rows(runs: list[dict[str, Any]], existing: dict[tuple[str, int], str], limit: int) -> list[dict[str, Any]]:
    """Rows to label by hand, keeping any verdicts already given."""
    rows = []
    for r in runs[:limit]:
        key = (r["case_id"], r["repetition"])
        rows.append({
            "case_id": r["case_id"],
            "repetition": r["repetition"],
            "reply": r["output"]["reply"],
            # The judge's context: without it, faithfulness is unscoreable.
            "tool_calls": r["output"].get("tool_calls", []),
            "human_verdict": existing.get(key, ""),  # fill in "pass" or "fail" by hand
        })
    return rows


@dataclass
class Agreement:
    agreed: int = 0
    total: int = 0
    disagreements: list[tuple[str, str, str, str]] = field(default_factory=list)
    # Rows labelled before tool calls were recorded: re-run --init to refresh them.
    stale: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0


def measure(rows: list[dict[str, Any]], cases: dict[str, Case], judge: JudgeFn, judge_min: int) -> Agreement:
    result = Agreement()
    for row in rows:
        if row.get("human_verdict") not in ("pass", "fail"):
            continue
        case = cases.get(row["case_id"])
        if case is None:
            continue
        if "tool_calls" not in row:
            result.stale.append(f"{row['case_id']}#{row['repetition']}")
            continue
        out = AgentOutput(reply=row["reply"], tool_calls=row["tool_calls"])
        verdict = judge(case, out)
        if verdict is None:
            continue
        low = min(verdict.helpfulness, verdict.faithfulness, verdict.tone)
        judge_verdict = "pass" if low >= judge_min else "fail"
        result.total += 1
        if judge_verdict == row["human_verdict"]:
            result.agreed += 1
        else:
            result.disagreements.append((row["case_id"], row["human_verdict"], judge_verdict, verdict.reasoning))
    return result
