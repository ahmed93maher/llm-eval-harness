"""Record the current quality as the new baseline.

Run this deliberately, never automatically -- a baseline updated on every run
ratchets quality downward one invisible notch at a time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents  # noqa: E402
from harness import baseline, config, dataset, runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="support", help="which system under test (agents/<name>)")
    system = agents.get(ap.parse_args().agent)
    cases = dataset.load_cases(system)
    dataset.validate(system, cases)
    use_judge = not config.FAKE
    print(f"running {len(cases)} cases x {config.REPETITIONS} reps to establish a baseline\n")
    report, _ = runner.run_suite(system, cases, use_judge=use_judge)

    if report.flaky_cases:
        print("\nRefusing nothing, but note: these cases are flaky and will make the baseline noisy:")
        for c in report.flaky_cases:
            print(f"  {c.case_id}: {c.passes}/{c.runs}")

    path = baseline.write(report)
    print(f"\nsuite pass rate {report.suite_pass_rate:.1%} written to {path}")
    print("commit this file -- the diff is the record of how quality moved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
