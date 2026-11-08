"""Load the eval dataset and validate it before a single API call is spent.

A bad dataset produces confident, meaningless numbers. These expectations are the
same kind you would express in Great Expectations or pandera -- kept as explicit
pandas checks here so the repo has no heavyweight dependency, and so each
expectation is readable next to the reason it exists.
"""
from __future__ import annotations

import pandas as pd
import yaml

from harness.schema import Case
from harness.system import SystemUnderTest


class DatasetError(AssertionError):
    """Raised when the dataset itself is broken."""


def load_cases(system: SystemUnderTest) -> list[Case]:
    with open(system.cases_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or []
    return [system.case_model(**entry) for entry in raw]


def to_frame(cases: list[Case]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in cases])


def validate(system: SystemUnderTest, cases: list[Case]) -> pd.DataFrame:
    """Expectations about the dataset. Raises DatasetError listing every violation.

    The checks here hold for any agent; each agent adds its own (what makes one
    of its cases falsifiable, which adversarial tags it must cover).
    """
    df = to_frame(cases)
    problems: list[str] = []

    if df.empty:
        raise DatasetError("dataset is empty")

    # ids are unique -- duplicates silently double-weight a case
    dupes = df["id"][df["id"].duplicated()].tolist()
    if dupes:
        problems.append(f"duplicate case ids: {dupes}")

    # no empty messages
    if (df["message"].str.strip() == "").any():
        problems.append("one or more cases have an empty message")

    # every case carries at least one tag -- tags are how we slice results
    untagged = df.loc[df["tags"].apply(len) == 0, "id"].tolist()
    if untagged:
        problems.append(f"untagged cases: {untagged}")

    # language coverage -- an Arabic-facing product needs Arabic cases
    arabic_share = (df["lang"] == "ar").mean()
    if arabic_share < system.min_arabic_share:
        problems.append(
            f"Arabic cases are only {arabic_share:.0%} of the suite (want >= {system.min_arabic_share:.0%})"
        )

    for check in system.dataset_checks:
        problems.extend(check(df))

    if problems:
        raise DatasetError("dataset validation failed:\n  - " + "\n  - ".join(problems))
    return df


def summarise(df: pd.DataFrame) -> str:
    by_lang = df["lang"].value_counts().to_dict()
    tags = pd.Series([t for row in df["tags"] for t in row]).value_counts().to_dict()
    return f"{len(df)} cases | languages: {by_lang} | tags: {tags}"
