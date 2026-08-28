"""The subjective half: an LLM judge scoring against a written rubric.

Everything a regex can decide is decided in deterministic.py. What is left --
was this helpful, is it grounded in the context it was given, is the tone right --
needs judgement. We make that judgement comparable across runs by writing the
rubric down and returning numbers, not prose.

The judge is an instrument. scripts/judge_agreement.py measures it against
hand-labelled examples; until you have run that, treat these scores as unvalidated.
"""
from __future__ import annotations

import json

from harness import config
from harness.schema import AgentOutput, Case, JudgeVerdict

RUBRIC = """You are grading a customer-support agent's reply. Score three \
dimensions from 1 to 5. Be strict: 4 means "I would send this to a customer", \
5 means "I could not improve it".

HELPFULNESS
5 - fully resolves the request, or asks precisely the one question needed to proceed
4 - addresses the request, minor omission
3 - partially addresses it, or is vague
2 - deflects without helping
1 - ignores what was asked

FAITHFULNESS (the most important dimension)
5 - every factual claim is supported by the tool results / context shown below
4 - grounded, with one harmlessly imprecise statement
3 - a claim that is probably true but is not in the context
2 - states an order fact that does not appear in the context
1 - invents a status, date, amount, or policy

TONE
5 - natural, courteous, correct register for a text message in the customer's language
4 - fine, slightly stiff or slightly long
3 - noticeably robotic, or over-familiar
2 - curt, or dismissive of a frustrated customer
1 - rude, or blames the customer

Judge only what is in front of you. Do not reward a reply for sounding confident."""


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
        f"CUSTOMER MESSAGE ({case.lang}):\n{case.message}\n\n"
        f"CONTEXT THE AGENT RECEIVED (tool calls and what they returned):\n"
        f"{_format_context(out)}\n\n"
        f"AGENT REPLY:\n{out.reply}\n\n"
        f"Every factual claim in the reply must be supported by the context above. "
        f"A claim that is plausible but absent from the context is not faithful."
        f"{extra}"
    )


def judge(case: Case, out: AgentOutput) -> JudgeVerdict | None:
    """Score one reply. Returns None when the reply never happened."""
    if out.error or not out.reply.strip():
        return None
    if config.FAKE:
        # Offline mode never fabricates scores -- an unjudged run is reported as
        # unjudged, so nobody mistakes a free CI run for a full eval.
        return None

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=config.JUDGE_MODEL,
        max_tokens=config.JUDGE_MAX_TOKENS,
        system=RUBRIC,
        messages=[{"role": "user", "content": _prompt(case, out)}],
        output_format=JudgeVerdict,
    )
    return response.parsed_output
