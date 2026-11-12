"""Validate every agent's eval dataset. Runs in CI before anything spends an API call."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents  # noqa: E402
from harness import dataset  # noqa: E402


def main() -> int:
    failed = 0
    for name, system in agents.registry().items():
        try:
            df = dataset.validate(system, dataset.load_cases(system))
        except dataset.DatasetError as exc:
            print(f"{name}: {exc}")
            failed += 1
            continue
        print(f"{name}: dataset OK -- {dataset.summarise(df)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
