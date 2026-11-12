"""The subjective half: an LLM judge scoring against a written rubric.

Everything a regex can decide is decided by the agent's own graders. What is left --
was this helpful, is it grounded in the context it was given, is the tone right --
needs judgement. We make that judgement comparable across runs by writing the
rubric down and returning numbers, not prose. The mechanism is shared; each
agent supplies its own rubric (agents/<name>/rubric.py).

The judge is an instrument. scripts/judge_agreement.py measures it against
hand-labelled examples; until you have run that, treat these scores as unvalidated.
"""
from __future__ import annotations

import json

from harness import config
from harness.schema import AgentOutput, Case, JudgeVerdict

def _format_context(out: AgentOutput) -> str:
    """Every tool call the agent made, and what it got back.

    Faithfulness cannot be scored against the calls alone -- "shipped with
    Aramex" is only groundable if the judge can see that the tool said so.
    """
    if not out.tool_calls:
        return "(no tools called - the agent answered from the conversation alone)"
    lines: list[str] = []
    for i, call in enumerate(out.tool_calls, 1):
        args = json.dumps(call.get("args", {}), ensure_ascii=False, sort_keys=True)
        lines.append(f"{i}. called {call.get('name')}({args})")
        result = call.get("result")
        lines.append(f"   returned: {result if result is not None else '(not recorded)'}")
    return "\n".join(lines)


def _prompt(case: Case, out: AgentOutput) -> str:
    extra = f"\n\nCase-specific expectation:\n{case.rubric}" if case.rubric else ""
    return (
        f"USER MESSAGE ({case.lang}):\n{case.message}\n\n"
        f"CONTEXT THE AGENT RECEIVED (tool calls and what they returned):\n"
        f"{_format_context(out)}\n\n"
        f"AGENT REPLY:\n{out.reply}\n\n"
        f"Every factual claim in the reply must be supported by the context above. "
        f"A claim that is plausible but absent from the context is not faithful."
        f"{extra}"
    )


def judge(rubric: str, case: Case, out: AgentOutput) -> JudgeVerdict | None:
    """Score one reply against an agent's rubric. Returns None when the reply
    never happened, or when the judge did not produce a verdict."""
    if out.error or not out.reply.strip():
        return None
    if config.FAKE:
        # Offline mode never fabricates scores -- an unjudged run is reported as
        # unjudged, so nobody mistakes a free CI run for a full eval.
        return None

    import anthropic

    client = anthropic.Anthropic()
    # No refusal fallback to another model, deliberately: the judge is a
    # calibrated instrument, and silently swapping the model that scores a run
    # would make its scores incomparable with the ones agreement was measured on.
    response = client.messages.parse(
        model=config.JUDGE_MODEL,
        max_tokens=config.JUDGE_MAX_TOKENS,
        system=rubric,
        messages=[{"role": "user", "content": _prompt(case, out)}],
        output_format=JudgeVerdict,
        output_config={"effort": config.JUDGE_EFFORT},
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        return None  # reported as unjudged, never as a score
    return response.parsed_output
