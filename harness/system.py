"""The one thing the harness needs to know about an agent.

Everything in harness/ is agent-agnostic: running cases N times, replaying
recordings, aggregating pass rates, judging, calibrating the judge, gating on a
baseline, reporting. Everything that is about one particular agent -- its
prompt, its tools, its graders, its rubric, what makes one of its cases
falsifiable -- lives in agents/<name>/ and reaches the harness only through
this description.

The split is what lets one harness test two quite different agents, and it is
also the honest boundary for reuse: what moves to another product is this
package, not the agents.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from harness.schema import AgentOutput, Case, GraderResult

Grader = Callable[[Case, AgentOutput], GraderResult]


@dataclass(frozen=True)
class SystemUnderTest:
    name: str
    case_model: type[Case]
    data_dir: Path
    # One live run of one case. Must not raise for an API failure: a failed run
    # is a result to grade (see AgentOutput.error).
    run: Callable[[Case], AgentOutput]
    graders: tuple[Grader, ...]
    # What a good answer looks like for this agent; the judge scores against it.
    rubric: str
    # Agent-specific dataset expectations, each returning problems found. The
    # generic ones (unique ids, tags, language coverage) are in harness.dataset.
    dataset_checks: tuple[Callable[[pd.DataFrame], list[str]], ...] = field(default=())
    min_arabic_share: float = 0.25

    @property
    def cases_path(self) -> Path:
        return self.data_dir / "cases.yaml"

    @property
    def fixtures_dir(self) -> Path:
        return self.data_dir / "fixtures"

    @property
    def labels_path(self) -> Path:
        return self.data_dir / "human_labels.yaml"

    def grade(self, case: Case, out: AgentOutput) -> list[GraderResult]:
        return [g(case, out) for g in self.graders]
