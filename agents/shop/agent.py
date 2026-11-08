"""The system under test: a sales assistant for the manager of Tidewater Goods,
a fictional three-store shop, answering questions about seeded sales data.

Where the support agent mostly has to behave (stay in language, refuse, not
leak), this one mostly has to be right: its answers are figures, and every
figure has a ground truth in the seeded back office.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from agents.shop import backoffice, tools
from harness import config
from harness.schema import AgentOutput

if TYPE_CHECKING:
    import anthropic
    from anthropic.types import MessageParam, ToolResultBlockParam

SYSTEM_PROMPT = f"""You are the sales assistant of Tidewater Goods, a shop with three stores. \
You answer the store manager's questions about sales.

Today is {backoffice.TODAY.isoformat()}. Sales data covers {backoffice.FIRST_DAY.isoformat()} to \
{backoffice.LAST_DAY.isoformat()}.

Rules:
- Every figure you state must come from a tool result, or be calculated from tool results. \
Never estimate or invent a figure.
- If the question names no time period, ask which period the manager means before calling any tool.
- If the question names no store, cover all stores combined.
- If the data cannot answer the question, say so plainly.
- Answer in the language the manager wrote in.
- Be brief: lead with the answer, then the supporting figures."""


def run_agent(message: str, client: anthropic.Anthropic | None = None) -> AgentOutput:
    """Run the agent on one manager question and record its whole trajectory.

    Never raises for an API failure: a failed run is a result to grade.
    """
    import anthropic

    client = client or anthropic.Anthropic()
    started = time.perf_counter()
    out = AgentOutput(reply="")
    messages: list[MessageParam] = [{"role": "user", "content": message}]

    try:
        while True:
            if out.turns >= config.AGENT_MAX_TURNS:
                out.error = f"turn limit ({config.AGENT_MAX_TURNS}) reached"
                break

            response = client.messages.create(
                model=config.AGENT_MODEL,
                max_tokens=config.AGENT_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=tools.TOOLS,  # type: ignore[arg-type]
                output_config={"effort": config.AGENT_EFFORT},
                messages=messages,
            )
            out.turns += 1
            out.stop_reason = response.stop_reason
            out.input_tokens += response.usage.input_tokens
            out.output_tokens += response.usage.output_tokens

            if response.stop_reason == "refusal":
                out.refused = True
                out.error = "api_refusal"
                break

            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                out.reply = "".join(b.text for b in response.content if b.type == "text").strip()
                break

            # The full content goes back, thinking blocks included.
            messages.append({"role": "assistant", "content": response.content})
            results: list[ToolResultBlockParam] = []
            for call in calls:
                args = dict(call.input) if isinstance(call.input, dict) else {}
                result, is_error = tools.execute(call.name, args)
                # Recorded exactly as the model received it: a figure is only
                # checkable against what the agent was actually told.
                content = result if isinstance(result, str) else json.dumps(result)
                out.tool_calls.append({"name": call.name, "args": args, "result": content, "is_error": is_error})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error})
            # All results of one step go back in ONE user message.
            messages.append({"role": "user", "content": results})

    except anthropic.APIError as exc:
        out.error = f"{type(exc).__name__}: {exc}"

    out.latency_ms = int((time.perf_counter() - started) * 1000)
    return out
