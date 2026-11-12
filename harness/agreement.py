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
    """How a judge's verdicts line up with human labels.

    Raw agreement alone flatters a judge. When most labels are "pass", a judge
    that passes everything agrees with most of them while measuring nothing.
    So this also reports Cohen's kappa (agreement beyond what chance would
    give, 0 = no better than chance, 1 = perfect), and splits disagreements by
    direction: a lenient judge (passes what a human failed) hides real
    failures, which is worse than a strict one.
    """

    agreed: int = 0
    total: int = 0
    disagreements: list[tuple[str, str, str, str]] = field(default_factory=list)
    # Rows labelled before tool calls were recorded: re-run --init to refresh them.
    stale: list[str] = field(default_factory=list)
    # Rows the judge returned no verdict for (refusal, truncation): not scored.
    unjudged: list[str] = field(default_factory=list)
    human_pass: int = 0
    judge_pass: int = 0

    @property
    def rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    @property
    def lenient(self) -> int:
        """Judge passed, human failed: the dangerous direction."""
        return sum(1 for _, human, jv, _ in self.disagreements if human == "fail" and jv == "pass")

    @property
    def strict(self) -> int:
        """Judge failed, human passed: noisy, but it hides nothing."""
        return sum(1 for _, human, jv, _ in self.disagreements if human == "pass" and jv == "fail")

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa. None when chance agreement is total (every label and
        every verdict the same), where agreement says nothing about the judge."""
        if not self.total:
            return None
        h, j = self.human_pass / self.total, self.judge_pass / self.total
        chance = h * j + (1 - h) * (1 - j)
        if chance >= 1:
            return None
        return (self.rate - chance) / (1 - chance)


# The bar a judge must clear before its scores are trusted. Kappa 0.6 is the
# conventional floor for "substantial" agreement; raw agreement alone can pass
# 0.85 with a judge that measures nothing.
MIN_RATE = 0.85
MIN_KAPPA = 0.6
MAX_LENIENT = 0


def trusted(a: Agreement) -> tuple[bool, list[str]]:
    """Whether the judge clears the bar, and why not when it does not."""
    reasons = []
    if a.rate < MIN_RATE:
        reasons.append(f"agreement {a.rate:.0%} < {MIN_RATE:.0%}")
    if a.kappa is None:
        reasons.append("kappa undefined: label a mix of passes and fails")
    elif a.kappa < MIN_KAPPA:
        reasons.append(f"kappa {a.kappa:.2f} < {MIN_KAPPA}")
    if a.lenient > MAX_LENIENT:
        reasons.append(f"{a.lenient} lenient disagreement(s): the judge passed answers a human failed")
    return not reasons, reasons


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
            result.unjudged.append(f"{row['case_id']}#{row['repetition']}")
            continue
        low = min(verdict.helpfulness, verdict.faithfulness, verdict.tone)
        judge_verdict = "pass" if low >= judge_min else "fail"
        result.total += 1
        result.human_pass += row["human_verdict"] == "pass"
        result.judge_pass += judge_verdict == "pass"
        if judge_verdict == row["human_verdict"]:
            result.agreed += 1
        else:
            result.disagreements.append((row["case_id"], row["human_verdict"], judge_verdict, verdict.reasoning))
    return result
