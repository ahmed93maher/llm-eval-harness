"""The shop agent's cases: what each one expects, and what makes the dataset sound."""
from __future__ import annotations

import pandas as pd
from pydantic import ConfigDict, Field

from agents.shop import tools
from harness.schema import Case


class ShopCase(Case):
    """One manager question. Mirrors an entry in data/cases.yaml."""

    model_config = ConfigDict(extra="forbid")  # a typo in cases.yaml is an error, not a missing check

    # Tools that must each be called at least once.
    expect_tools: list[str] = Field(default_factory=list)
    # The right behaviour is to answer without any tool call (ask for the
    # missing period, decline an out-of-scope question).
    expect_no_tools: bool = False
    # The data cannot answer the question, so the answer must contain no
    # money figure at all: any figure would be invented.
    must_not_state_figures: bool = False
    notes: str = ""


def falsifiable(df: pd.DataFrame) -> list[str]:
    """Every case needs at least one expectation, or it always passes and proves nothing."""

    def has_expectation(row) -> bool:
        return bool(row["expect_tools"] or row["expect_no_tools"] or row["must_not_state_figures"] or row["rubric"])

    toothless = df.loc[~df.apply(has_expectation, axis=1), "id"].tolist()
    return [f"cases with no expectation (always pass, prove nothing): {toothless}"] if toothless else []


def consistent_tool_expectations(df: pd.DataFrame) -> list[str]:
    known = {t["name"] for t in tools.TOOLS}
    problems = []
    for _, row in df.iterrows():
        unknown = set(row["expect_tools"]) - known
        if unknown:
            problems.append(f"{row['id']}: expects unknown tools {sorted(unknown)}")
        if row["expect_tools"] and row["expect_no_tools"]:
            problems.append(f"{row['id']}: expects tools and no tools at once")
    return problems


def coverage(df: pd.DataFrame) -> list[str]:
    """Figures alone would hide the cases where the right answer is not a figure."""
    tags = {t for row in df["tags"] for t in row}
    return [f"no case tagged {t!r}" for t in ("figures", "clarification", "no-data", "out-of-scope") if t not in tags]


CHECKS = (falsifiable, consistent_tool_expectations, coverage)
