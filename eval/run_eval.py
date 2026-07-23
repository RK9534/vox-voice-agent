"""
Evaluation harness for Vox.

Checks TWO things per question, since this is a multi-agent system:
  1. Routing accuracy — did the router send it to the right specialist?
  2. Answer quality — keyword match against expected_keywords.

Logs latency and token usage too. Re-run after any prompt/model change.

Usage: python -m eval.run_eval
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agents import VoxOrchestrator  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "eval_questions.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.csv")


def keyword_pass(answer: str, expected_keywords: list) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


def main():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    orchestrator = VoxOrchestrator()
    rows = []
    routing_correct = 0
    answer_correct = 0

    for q in questions:
        result = orchestrator.run(q["query"])
        route_ok = result.agent == q["expected_agent"]
        answer_ok = keyword_pass(result.answer, q["expected_keywords"])
        routing_correct += int(route_ok)
        answer_correct += int(answer_ok)

        rows.append({
            "id": q["id"],
            "query": q["query"],
            "routed_to": result.agent,
            "expected_agent": q["expected_agent"],
            "routing_correct": route_ok,
            "variant": result.variant,
            "answer_pass": answer_ok,
            "latency_seconds": round(result.latency_seconds, 3),
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "answer": result.answer.replace("\n", " ")[:200],
        })
        print(f"[{'OK' if route_ok and answer_ok else 'CHECK'}] {q['id']}: "
              f"routed={result.agent} ({'correct' if route_ok else 'WRONG'}), "
              f"answer={'pass' if answer_ok else 'fail'}")

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(questions)
    avg_latency = sum(r["latency_seconds"] for r in rows) / total
    total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in rows)

    print("\n--- Summary ---")
    print(f"Routing accuracy: {routing_correct}/{total} ({100*routing_correct/total:.0f}%)")
    print(f"Answer pass rate: {answer_correct}/{total} ({100*answer_correct/total:.0f}%)")
    print(f"Avg latency: {avg_latency:.2f}s")
    print(f"Total tokens used: {total_tokens}")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
