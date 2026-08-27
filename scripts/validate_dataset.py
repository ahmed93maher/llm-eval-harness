"""Validate the eval dataset. Runs in CI before anything spends an API call."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import dataset  # noqa: E402


def main() -> int:
    cases = dataset.load_cases()
    try:
        df = dataset.validate(cases)
    except dataset.DatasetError as exc:
        print(exc)
        return 1
    print("dataset OK")
    print(dataset.summarise(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
