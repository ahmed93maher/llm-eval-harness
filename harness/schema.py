"""Typed contracts shared by the dataset, the agent, the graders and the runner."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Case(BaseModel):
    """One eval case. Mirrors an entry in data/cases.yaml."""

    id: str
    message: str = Field(description="what the customer sends")
    lang: Literal["en", "ar"]
    account_id: str
    tags: list[str] = Field(default_factory=list)

    # --- expectations ------------------------------------------------------
    expect_tool: str | None = None
    expect_tool_args: dict[str, Any] = Field(default_factory=dict)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    must_ask_for: list[str] = Field(
        default_factory=list,
        description="agent must request this missing information rather than guess",
    )
    must_refuse: bool = False
    rubric: str = Field(
        default="",
        description="what a good reply looks like; handed to the judge",
    )


class AgentOutput(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0
    refused: bool = Field(
        default=False,
        description="API returned stop_reason=refusal -- not the same as the agent politely declining",
    )
    error: str | None = None


class GraderResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class JudgeVerdict(BaseModel):
    """Structured output from the LLM judge. Scores are 1-5."""

    helpfulness: int = Field(ge=1, le=5)
    faithfulness: int = Field(
        ge=1, le=5, description="5 = every claim is grounded in the context provided"
    )
    tone: int = Field(ge=1, le=5)
    reasoning: str = Field(description="one or two sentences, cite the deciding detail")


class RunResult(BaseModel):
    """One execution of one case."""

    case_id: str
    repetition: int
    output: AgentOutput
    deterministic: list[GraderResult] = Field(default_factory=list)
    judge: JudgeVerdict | None = None

    @property
    def deterministic_passed(self) -> bool:
        return all(g.passed for g in self.deterministic)

    def passed(self, judge_min: int) -> bool:
        if not self.deterministic_passed or self.output.error:
            return False
        if self.judge is None:
            return True  # deterministic-only run (offline / CI push job)
        return min(self.judge.helpfulness, self.judge.faithfulness, self.judge.tone) >= judge_min

    def failures(self) -> list[str]:
        out = [f"{g.name}: {g.detail}" for g in self.deterministic if not g.passed]
        if self.output.error:
            out.append(f"error: {self.output.error}")
        return out


class CaseSummary(BaseModel):
    case_id: str
    tags: list[str]
    runs: int
    passes: int
    failures: list[str] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0

    @property
    def flaky(self) -> bool:
        """Passed sometimes. The most informative state in the whole harness."""
        return 0 < self.passes < self.runs


class SuiteReport(BaseModel):
    cases: list[CaseSummary]
    agent_model: str
    judge_model: str
    repetitions: int
    judged: bool

    @property
    def suite_pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.pass_rate for c in self.cases) / len(self.cases)

    @property
    def flaky_cases(self) -> list[CaseSummary]:
        return [c for c in self.cases if c.flaky]
