"""Write JUnit XML so other tools can ingest the results.

  python scripts/report_junit.py                      # newest run -> results/junit.xml
  python scripts/report_junit.py -o build/junit.xml
  python scripts/report_junit.py results/report-20260828T215006Z.json

JUnit XML is the lingua franca of test results: Jira/Xray imports it, Jenkins
renders it natively, and GitHub Actions test reporters read it. One <testcase>
per eval case; a flaky case is reported as a failure with type="flaky", because
JUnit has no flaky state and silently passing an intermittent case is exactly
what this harness exists to prevent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import config, reporting  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="path to a report-*.json (default: newest)")
    ap.add_argument("--agent", default=None, help="newest report of this agent (default: newest of any)")
    ap.add_argument("-o", "--out", default=None, help="output path (default: results/junit-<agent>.xml)")
    ap.add_argument("--suite-name", default=None, help="testsuite name in the XML (default: llm-eval-<agent>)")
    args = ap.parse_args()

    path = Path(args.report) if args.report else reporting.newest_report(args.agent)
    if path is None:
        print("no results yet -- run scripts/run_eval.py first")
        return 1

    report, runs = reporting.load_results(path)
    agent = report.get("agent") or "support"
    xml = reporting.to_junit(report, runs, suite_name=args.suite_name or f"llm-eval-{agent}")

    out_path = Path(args.out) if args.out else config.RESULTS / f"junit-{agent}.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")

    failed = sum(1 for c in report["cases"] if reporting.status_of(c) != "PASS")
    print(f"wrote {out_path} ({len(report['cases'])} testcases, {failed} failing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
