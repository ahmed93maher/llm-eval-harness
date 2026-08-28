"""Render a run's results as a single self-contained HTML file.

  python scripts/report_html.py                 # newest run in results/
  python scripts/report_html.py results/report-20260828T215006Z.json
  python scripts/report_html.py --open          # and open it in the browser

The JSON files are for machines and archives. This is the one a human reads:
pass rates, what failed and why, and the actual replies -- including Arabic,
rendered right-to-left rather than mangled.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import config  # noqa: E402

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
  --pass: #15803d; --pass-bg: #dcfce7; --fail: #b91c1c; --fail-bg: #fee2e2;
  --flaky: #b45309; --flaky-bg: #fef3c7; --chip: #f3f4f6; --card: #fafafa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --fg: #e5e7eb; --muted: #9ca3af; --line: #262b33;
    --pass: #4ade80; --pass-bg: #14321f; --fail: #f87171; --fail-bg: #3b1618;
    --flaky: #fbbf24; --flaky-bg: #3a2c0a; --chip: #1b2028; --card: #151922;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
.sub code { background: var(--chip); padding: 1px 6px; border-radius: 4px; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 18px; min-width: 120px; flex: 1;
}
.card .n { font-size: 26px; font-weight: 650; letter-spacing: -0.5px; }
.card .l { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }
.banner {
  border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; font-weight: 600;
  border: 1px solid var(--line);
}
.banner.ok { background: var(--pass-bg); color: var(--pass); }
.banner.bad { background: var(--fail-bg); color: var(--fail); }
details {
  border: 1px solid var(--line); border-radius: 10px; margin-bottom: 10px;
  background: var(--card); overflow: hidden;
}
summary {
  cursor: pointer; padding: 12px 16px; display: grid; align-items: center; gap: 12px;
  grid-template-columns: 90px 1fr auto auto; list-style: none;
}
summary::-webkit-details-marker { display: none; }
summary:hover { background: var(--chip); }
.badge {
  font-size: 11px; font-weight: 700; letter-spacing: .6px; padding: 4px 8px;
  border-radius: 5px; text-align: center;
}
.badge.PASS { background: var(--pass-bg); color: var(--pass); }
.badge.FAIL { background: var(--fail-bg); color: var(--fail); }
.badge.FLAKY { background: var(--flaky-bg); color: var(--flaky); }
.cid { font-weight: 600; font-family: ui-monospace, "Cascadia Mono", monospace; font-size: 13px; }
.tags { display: flex; gap: 5px; flex-wrap: wrap; }
.tag { background: var(--chip); color: var(--muted); font-size: 11px; padding: 2px 7px; border-radius: 999px; }
.rate { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 13px; }
.body { padding: 4px 16px 16px; border-top: 1px solid var(--line); }
.fail-line {
  background: var(--fail-bg); color: var(--fail); border-radius: 6px;
  padding: 8px 12px; margin: 10px 0 0; font-size: 13px;
  font-family: ui-monospace, "Cascadia Mono", monospace;
}
.run { border-top: 1px dashed var(--line); padding: 14px 0 4px; }
.run h4 { margin: 0 0 8px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; }
.reply {
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 14px; white-space: pre-wrap; margin-bottom: 10px;
}
.graders { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
.g { font-size: 11px; padding: 3px 8px; border-radius: 5px; background: var(--pass-bg); color: var(--pass); }
.g.no { background: var(--fail-bg); color: var(--fail); }
.judge { font-size: 13px; color: var(--muted); }
.judge b { color: var(--fg); }
.meta { color: var(--muted); font-size: 12px; }
"""


def _load(report_path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    runs_path = report_path.with_name(report_path.name.replace("report-", "runs-"))
    runs = json.loads(runs_path.read_text(encoding="utf-8")) if runs_path.exists() else []
    return report, runs


def _status(case: dict) -> str:
    if 0 < case["passes"] < case["runs"]:
        return "FLAKY"
    return "PASS" if case["passes"] == case["runs"] else "FAIL"


def _esc(text: str) -> str:
    return html.escape(text or "")


def _run_block(run: dict) -> str:
    out = run.get("output", {})
    graders = "".join(
        f'<span class="g{"" if g["passed"] else " no"}">{_esc(g["name"])}</span>'
        for g in run.get("deterministic", [])
    )
    judge = run.get("judge")
    judge_html = ""
    if judge:
        judge_html = (
            f'<div class="judge">judge &mdash; helpfulness <b>{judge["helpfulness"]}</b>, '
            f'faithfulness <b>{judge["faithfulness"]}</b>, tone <b>{judge["tone"]}</b>: '
            f'{_esc(judge["reasoning"])}</div>'
        )
    tools = out.get("tool_calls") or []
    tool_html = ""
    if tools:
        names = ", ".join(f'{_esc(t.get("name", "?"))}({_esc(json.dumps(t.get("args", {})))})' for t in tools)
        tool_html = f'<div class="meta">tools: {names}</div>'
    err = f'<div class="fail-line">{_esc(out.get("error"))}</div>' if out.get("error") else ""
    # dir="auto" lets the browser pick RTL for Arabic replies and LTR for English.
    return (
        f'<div class="run"><h4>run {run.get("repetition", 0) + 1}'
        f' &middot; {out.get("latency_ms", 0)} ms</h4>'
        f'<div class="reply" dir="auto">{_esc(out.get("reply"))}</div>'
        f'<div class="graders">{graders}</div>{tool_html}{judge_html}{err}</div>'
    )


def render(report: dict, runs: list[dict], stamp: str) -> str:
    cases = report["cases"]
    by_case: dict[str, list[dict]] = {}
    for r in runs:
        by_case.setdefault(r["case_id"], []).append(r)

    passed = sum(1 for c in cases if _status(c) == "PASS")
    failed = sum(1 for c in cases if _status(c) == "FAIL")
    flaky = sum(1 for c in cases if _status(c) == "FLAKY")
    suite_rate = sum(c["passes"] / c["runs"] for c in cases) / len(cases) if cases else 0

    rows = []
    for case in sorted(cases, key=lambda c: (_status(c) == "PASS", c["case_id"])):
        status = _status(case)
        tags = "".join(f'<span class="tag">{_esc(t)}</span>' for t in case["tags"])
        fails = "".join(f'<div class="fail-line">{_esc(f)}</div>' for f in case["failures"])
        detail = "".join(_run_block(r) for r in by_case.get(case["case_id"], []))
        rows.append(
            f"<details{' open' if status != 'PASS' else ''}>"
            f'<summary><span class="badge {status}">{status}</span>'
            f'<span class="cid">{_esc(case["case_id"])}</span>'
            f'<span class="tags">{tags}</span>'
            f'<span class="rate">{case["passes"]}/{case["runs"]}</span></summary>'
            f'<div class="body">{fails}{detail}</div></details>'
        )

    judged = report.get("judged")
    banner = (
        '<div class="banner ok">Judged run &mdash; deterministic invariants and rubric scoring.</div>'
        if judged
        else '<div class="banner bad">Deterministic-only run &mdash; the judge did not score these '
        "replies, so subjective quality is unmeasured.</div>"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Eval report {stamp}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>LLM eval report</h1>
<div class="sub">{stamp} &middot; agent <code>{_esc(report["agent_model"])}</code>
{f'&middot; judge <code>{_esc(report["judge_model"])}</code>' if report.get("judge_model") else ""}
&middot; {report["repetitions"]} repetition(s) per case</div>
{banner}
<div class="cards">
  <div class="card"><div class="n">{suite_rate:.0%}</div><div class="l">suite pass rate</div></div>
  <div class="card"><div class="n">{len(cases)}</div><div class="l">cases</div></div>
  <div class="card"><div class="n">{passed}</div><div class="l">passed</div></div>
  <div class="card"><div class="n">{failed}</div><div class="l">failed</div></div>
  <div class="card"><div class="n">{flaky}</div><div class="l">flaky</div></div>
</div>
{"".join(rows)}
</div></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="path to a report-*.json (default: newest)")
    ap.add_argument("--open", action="store_true", help="open in the default browser")
    args = ap.parse_args()

    if args.report:
        path = Path(args.report)
    else:
        found = sorted(glob.glob(str(config.RESULTS / "report-*.json")))
        if not found:
            print("no results yet -- run scripts/run_eval.py first")
            return 1
        path = Path(found[-1])

    report, runs = _load(path)
    stamp = path.stem.replace("report-", "")
    out_path = path.with_suffix(".html")
    out_path.write_text(render(report, runs, stamp), encoding="utf-8")
    print(f"wrote {out_path}")
    if args.open:
        webbrowser.open(out_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
