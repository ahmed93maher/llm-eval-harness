"""Graders any agent can use: the run finished, and it answered in the user's language.

Agent-specific graders live with their agent (agents/<name>/graders.py); these
two hold for any agent that talks to people, so they are written once.
"""
from __future__ import annotations

import re

from harness.schema import AgentOutput, Case, GraderResult

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def _ok(name: str, detail: str = "") -> GraderResult:
    return GraderResult(name=name, passed=True, detail=detail)


def _fail(name: str, detail: str) -> GraderResult:
    return GraderResult(name=name, passed=False, detail=detail)


def completed(case: Case, out: AgentOutput) -> GraderResult:
    """The run finished with an answer: no error, not refused, not cut off."""
    if out.error:
        return _fail("completed", out.error)
    if out.stop_reason == "max_tokens":
        # The reply was cut off mid-sentence; it may still read as plausible.
        return _fail("completed", "reply truncated at max_tokens")
    if not out.reply.strip():
        return _fail("completed", "empty reply")
    return _ok("completed")


def language_matches(case: Case, out: AgentOutput) -> GraderResult:
    """Arabic in, Arabic out. A real bug class, and invisible to an English-only tester."""
    if not out.reply.strip():
        return _fail("language_matches", "empty reply")
    letters = [c for c in out.reply if c.isalpha()]
    if not letters:
        return _fail("language_matches", "reply contains no letters")
    arabic_ratio = sum(1 for c in letters if ARABIC_RE.match(c)) / len(letters)
    if case.lang == "ar" and arabic_ratio < 0.5:
        return _fail("language_matches", f"user wrote Arabic, reply is {arabic_ratio:.0%} Arabic")
    if case.lang == "en" and arabic_ratio > 0.5:
        return _fail("language_matches", f"user wrote English, reply is {arabic_ratio:.0%} Arabic")
    return _ok("language_matches", f"arabic_ratio={arabic_ratio:.2f}")
