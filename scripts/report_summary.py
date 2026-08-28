"""Write a Markdown summary of a run.

  python scripts/report_summary.py            # print to stdout
  python scripts/report_summary.py --github   # append to $GITHUB_STEP_SUMMARY

In CI the --github form puts a real table on the workflow run page: pass rates,
which cases failed and why, and the baseline comparison -- so nobody has to
download an artifact to find out what broke.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import baseline, reporting  # noqa: E402
from harness.schema import SuiteReport  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="path to a report-*.json (default: newest)")
    ap.add_argument(
        "--github",
        action="store_true",
        help="append to $GITHUB_STEP_SUMMARY instead of printing",
    )
    ap.add_argument("--no-baseline", action="store_true", help="skip the baseline comparison")
    args = ap.parse_args()

    path = Path(args.report) if args.report else reporting.newest_report()
    if path is None:
        print("no results yet -- run scripts/run_eval.py first")
        return 1

    report, runs = reporting.load_results(path)

    messages: list[str] = []
    if not args.no_baseline:
        # Re-derive the comparison from the saved report so the summary matches
        # what the gate decided, without re-running anything.
        _, messages = baseline.compare(SuiteReport(**report))

    md = reporting.to_markdown(report, runs, messages)

    target = os.environ.get("GITHUB_STEP_SUMMARY") if args.github else None
    if target:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(md)
        print(f"summary appended to {target}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
