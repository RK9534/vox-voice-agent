"""
Multi-agent orchestration.

Flow:
  1. Router (no tools, cheap/fast call) classifies the query as
     'scheduling' or 'clinical_info'.
  2. The chosen specialist agent runs its own tool-calling loop
     (schedule_appointment / search_docs) using a system-prompt variant
     selected by the feedback bandit (app/feedback.py).
  3. Feedback (thumbs up/down) recorded after the fact updates that
     variant's stats, so future routing to that agent skews toward
     whichever prompt style has performed better — the "improves with
     every interaction" loop, in its simplest honest form.
"""
import time
from dataclasses import dataclass, field
from typing import List

import anthropic

from . import config, db, feedback
from .tools import SCHEDULING_TOOLS, FAQ_TOOLS, execute_tool

ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a clinic voice copilot.
Read the user's message and respond with EXACTLY ONE WORD:
- "scheduling" if it's about booking, rescheduling, canceling, or asking about appointment slots/hours.
- "clinical_info" for anything else (general questions, wellness info, what the copilot can help with).
Respond with only that one word, nothing else.
"""

# Two system-prompt variants per specialist agent — the bandit picks between
# them based on real thumbs-up feedback over time.
SCHEDULING_VARIANTS = {
    "concise": (
        "You are a scheduling assistant for a clinic. Be brief and efficient. "
        "Confirm patient name, preferred day, and reason, then call schedule_appointment. "
        "Use search_docs if asked about hours or policy. Never diagnose."
    ),
    "warm": (
        "You are a friendly, reassuring scheduling assistant for a clinic. Confirm the "
        "patient's name, preferred day, and reason for the visit, then call schedule_appointment. "
        "Use search_docs for hours/policy questions. Keep a warm, patient tone. Never diagnose."
    ),
}

FAQ_VARIANTS = {
    "concise": (
        "You are a clinic information assistant. Answer briefly using search_docs for anything "
        "about visit types, wellness, or clinic policy. Never diagnose or recommend medication — "
        "redirect health-specific questions to a clinician."
    ),
    "detailed": (
        "You are a thorough, patient-education-focused clinic information assistant. Use search_docs "
        "and explain context clearly using plain language. Never diagnose or recommend medication — "
        "always redirect health-specific questions to a clinician."
    ),
}


@dataclass
class AgentResult:
    answer: str
    agent: str
    variant: str
    request_id: str
    tool_calls: List[dict] = field(default_factory=list)
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class VoxOrchestrator:
    def __init__(self, api_key: str = None, model: str = None):
        self.client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)
        self.model = model or config.MODEL_NAME

    def _route(self, query: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            system=ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip().lower()
        return "scheduling" if "scheduling" in text else "clinical_info"

    def _run_specialist(self, system_prompt: str, tools: list, query: str):
        messages = [{"role": "user", "content": query}]
        tool_calls_log = []
        in_tok, out_tok = 0, 0

        for _ in range(config.MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
            in_tok += response.usage.input_tokens
            out_tok += response.usage.output_tokens

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                return text, tool_calls_log, in_tok, out_tok

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = execute_tool(block.name, block.input)
                tool_calls_log.append({"name": block.name, "input": block.input, "result": result})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            messages.append({"role": "user", "content": tool_results})

        return "[Max tool iterations reached]", tool_calls_log, in_tok, out_tok

    def run(self, query: str) -> AgentResult:
        start = time.time()
        route = self._route(query)

        if route == "scheduling":
            agent_name = "scheduling_agent"
            variants = SCHEDULING_VARIANTS
            tools = SCHEDULING_TOOLS
        else:
            agent_name = "faq_agent"
            variants = FAQ_VARIANTS
            tools = FAQ_TOOLS

        variant_name = feedback.choose_variant(agent_name, list(variants.keys()))
        system_prompt = variants[variant_name]

        answer, tool_calls, in_tok, out_tok = self._run_specialist(system_prompt, tools, query)

        request_id = db.new_request_id()
        db.record_use(agent_name, variant_name)
        db.record_request(request_id, agent_name, variant_name, query, answer)

        return AgentResult(
            answer=answer,
            agent=agent_name,
            variant=variant_name,
            request_id=request_id,
            tool_calls=tool_calls,
            latency_seconds=time.time() - start,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
