"""Recorded agent runs for offline replay: one file per (case, repetition).

Recordings live in each agent's data directory: data/fixtures/<case>/<n>.json.

Why per repetition: the offline suite exists to grade real model behaviour
without an API key. With one sample per case, every repetition replays the same
output, so an offline pass rate can only be 0% or 100% and a flaky case can
never show up. Recording each repetition keeps the run-to-run variance the live
model actually produced.

Why keyed by case id, not by message: editing a case's message must not
silently orphan its recording, and two cases may send the same message with
different expectations.

A missing recording is a failed run, not a skipped one: it fails `completed`
loudly instead of letting a new case pass by having nothing to grade.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from harness import config
from harness.schema import AgentOutput

# "recorded" marks genuine model output; "hand-authored" a correct test double
# written by a person; "planted-fault" a test double with a deliberate fault,
# which the graders must catch. The report says which ones it graded.
Source = Literal["recorded", "hand-authored", "planted-fault"]


def path(root: Path, case_id: str, repetition: int) -> Path:
    return root / case_id / f"{repetition}.json"


def save(
    root: Path, case_id: str, repetition: int, out: AgentOutput, source: Source = "recorded", fault: str = ""
) -> Path:
    p = path(root, case_id, repetition)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "agent_model": config.AGENT_MODEL if source == "recorded" else None,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": out.model_dump(),
    }
    if fault:
        payload["fault"] = fault  # what was planted, for the test that it is caught
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def load(root: Path, case_id: str, repetition: int) -> AgentOutput:
    p = path(root, case_id, repetition)
    if not p.exists():
        return AgentOutput(reply="", error=f"no fixture recorded for {case_id}#{repetition}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    return AgentOutput.model_validate(payload["output"])


def clear(root: Path, case_id: str) -> None:
    """Drop a case's old recordings before re-recording it, so a smaller
    re-recording never leaves repetitions from an older model behind."""
    for p in (root / case_id).glob("*.json"):
        p.unlink()


def missing(root: Path, case_ids: list[str], repetitions: int) -> list[str]:
    """Recordings an offline run would need but does not have, as "case#rep".

    Checked before an offline run starts: graded, a gap would fail some
    repetitions and pass others, and look exactly like a flaky agent.
    """
    return [f"{c}#{r}" for c in case_ids for r in range(repetitions) if not path(root, c, r).exists()]


def sources(root: Path, case_ids: list[str]) -> str:
    """What an offline run is grading: real recordings, hand-written doubles, or both."""
    found = set()
    for c in case_ids:
        for p in (root / c).glob("*.json"):
            found.add(json.loads(p.read_text(encoding="utf-8")).get("source", "recorded"))
    return "+".join(sorted(found)) or "none"


def read_meta(root: Path, case_id: str, repetition: int) -> dict:
    """A recording's metadata: source, model, and the planted fault if any."""
    payload = json.loads(path(root, case_id, repetition).read_text(encoding="utf-8"))
    return {k: v for k, v in payload.items() if k != "output"}
