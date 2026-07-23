"""
Tool definitions available to the specialist agents.
"""
import random

from . import rag

# --- Tool 1: mock appointment scheduler ---------------------------------
# In production this would call a real scheduling/EHR API. Kept as a
# deterministic mock here so the POC is self-contained and testable
# without external services.

_SLOT_POOL = ["Mon 9:30 AM", "Mon 2:00 PM", "Tue 11:00 AM", "Wed 10:15 AM", "Thu 3:45 PM"]


def schedule_appointment_tool(patient_name: str, preferred_day: str, reason: str) -> str:
    slot = random.choice(_SLOT_POOL)
    return (
        f"Appointment confirmed for {patient_name}: {slot} "
        f"(requested day: {preferred_day}). Reason on file: '{reason}'. "
        f"A reminder will be sent 24 hours before the visit."
    )


# --- Tool 2: knowledge base search (RAG) --------------------------------

def search_docs_tool(query: str) -> str:
    index = rag.get_index()
    results = index.retrieve(query)
    if not results:
        return "No relevant documents found."
    formatted = []
    for r in results:
        formatted.append(f"[source: {r['source']}, score: {r['score']:.2f}]\n{r['text']}")
    return "\n\n---\n\n".join(formatted)


# --- Tool schemas ---------------------------------------------------------

SCHEDULING_TOOLS = [
    {
        "name": "schedule_appointment",
        "description": "Book a clinic appointment for a patient. Always confirm the patient's "
                       "name, preferred day, and reason for visit before calling this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "preferred_day": {"type": "string", "description": "e.g. 'Monday', 'this Wednesday'"},
                "reason": {"type": "string", "description": "Brief reason, used only for routing, not diagnosis"},
            },
            "required": ["patient_name", "preferred_day", "reason"],
        },
    },
    {
        "name": "search_docs",
        "description": "Search clinic policy docs for scheduling rules, hours, or visit types.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

FAQ_TOOLS = [
    {
        "name": "search_docs",
        "description": "Search the clinic knowledge base for general, non-diagnostic information "
                       "about visit types, wellness, and what the copilot can/cannot help with.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "schedule_appointment":
        return schedule_appointment_tool(
            tool_input.get("patient_name", "the patient"),
            tool_input.get("preferred_day", "unspecified"),
            tool_input.get("reason", "unspecified"),
        )
    if name == "search_docs":
        return search_docs_tool(tool_input.get("query", ""))
    return f"Unknown tool: {name}"
