"""
Sanity-check RAG, tools, and the feedback bandit WITHOUT calling the
Anthropic API. Run this first to confirm the local pieces work.

Usage: python -m scripts.quick_test
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools import schedule_appointment_tool, search_docs_tool  # noqa: E402
from app import db, feedback  # noqa: E402


def main():
    print("=== Scheduling tool ===")
    print(schedule_appointment_tool("Priya", "Thursday", "follow-up"))

    print("\n=== RAG search tool ===")
    for q in ["What are the clinic hours on Saturday?", "What can the copilot not do?"]:
        print(f"\nQuery: {q}")
        print(search_docs_tool(q)[:300], "...")

    print("\n=== Feedback bandit ===")
    db.init_db()
    for _ in range(5):
        v = feedback.choose_variant("faq_agent", ["concise", "detailed"])
        db.record_use("faq_agent", v)
        print("Chosen variant:", v)
    print("Stats:", db.get_variant_stats("faq_agent"))


if __name__ == "__main__":
    main()
