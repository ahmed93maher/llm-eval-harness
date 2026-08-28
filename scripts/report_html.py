"""Render a run as a self-contained HTML report.

  python scripts/report_html.py                 # newest run in results/
  python scripts/report_html.py results/report-20260828T215006Z.json
  python scripts/report_html.py --open          # and open it in the browser
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import reporting  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="path to a report-*.json (default: newest)")
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    args = ap.parse_args()

    path = Path(args.report) if args.report else reporting.newest_report()
    if path is None:
        print("no results yet -- run scripts/run_eval.py first")
        return 1

    report, runs = reporting.load_results(path)
    out_path = path.with_suffix(".html")
    out_path.write_text(
        reporting.to_html(report, runs, path.stem.replace("report-", "")), encoding="utf-8"
    )
    print(f"wrote {out_path}")
    if args.open:
        webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
