"""The systems under test. Each one describes itself as a harness.system.SystemUnderTest."""
from __future__ import annotations

from harness.system import SystemUnderTest


def registry() -> dict[str, SystemUnderTest]:
    from agents import support

    return {s.name: s for s in (support.SYSTEM,)}


def get(name: str) -> SystemUnderTest:
    systems = registry()
    if name not in systems:
        raise SystemExit(f"unknown agent {name!r}; known: {', '.join(systems)}")
    return systems[name]
