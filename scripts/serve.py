"""Minimal HTTP wrapper around the agent, so k6 has something to hit.

  python scripts/serve.py            # live agent on :8000
  EVAL_FAKE=1 python scripts/serve.py  # fixture mode: load-test the plumbing for free

POST /reply  {"message": "...", "account_id": "ACC-77"}
          -> {"reply": "...", "tool_calls": [...], "latency_ms": 1234}

stdlib only, single-threaded per connection but threaded across them -- enough to
generate real concurrency against the model API, which is the point.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import config, dataset, fixtures  # noqa: E402
from harness.agent import run_agent  # noqa: E402
from harness.schema import AgentOutput  # noqa: E402

# Offline, a message from the dataset replays its case's first recording; any
# other message gets an obvious placeholder. Enough to load-test the plumbing.
CASE_BY_MESSAGE = {c.message: c.id for c in dataset.load_cases()}


def reply(message: str, account_id: str) -> AgentOutput:
    if not config.FAKE:
        return run_agent(message, account_id)
    case_id = CASE_BY_MESSAGE.get(message)
    if case_id is None:
        return AgentOutput(reply="(no fixture recorded for this message)")
    return fixtures.load(case_id, 0)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/reply":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send(400, {"error": f"bad request: {exc}"})

        message = payload.get("message", "")
        account_id = payload.get("account_id", "ACC-77")
        if not message:
            return self._send(400, {"error": "message is required"})

        out = reply(message, account_id)
        if out.error:
            # Surface upstream rate limiting as 429 so k6 can measure it properly
            # instead of burying it in a generic 500.
            code = 429 if "RateLimit" in out.error else 502
            return self._send(code, {"error": out.error, "latency_ms": out.latency_ms})
        return self._send(200, out.model_dump())

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - name fixed by the base class
        pass  # quiet under load


def main() -> int:
    port = 8000
    mode = "FIXTURES" if config.FAKE else config.AGENT_MODEL
    print(f"serving on http://localhost:{port}/reply  ({mode})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
