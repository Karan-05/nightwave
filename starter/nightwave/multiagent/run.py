"""Entry point for the multi-agent pipeline.

Drop-in replacement for nightwave.agent.answer():
    from nightwave.multiagent.run import answer_multi
    ans = answer_multi("What was Madison wearing?")

Also runnable as a standalone eval:
    python -m nightwave.multiagent.run

Boots Neo4j seed on first run if NEO4J_URI is set and the graph is empty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nightwave.agent import Answer
from nightwave.multiagent.graph_db import _get_driver, get_mode
from nightwave.multiagent.orchestrator import run_pipeline

MULTI_REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "eval_report_multiagent.json"
QUESTIONS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "questions.json"


def _maybe_seed() -> None:
    """Seed Neo4j if connected and the graph is empty."""
    if get_mode() != "neo4j":
        return
    from nightwave.multiagent.graph_db import run_cypher
    from nightwave.multiagent.seed import seed

    try:
        count = run_cypher("MATCH (n) RETURN count(n) AS c")[0]["c"]
        if count == 0:
            print("[run] Neo4j is empty — seeding from case_data.json…")
            seed()
    except Exception as exc:
        print(f"[run] Seed check failed: {exc}")


def answer_multi(question: str, question_id: str = "") -> Answer:
    """Multi-agent answer. Mirrors the signature of nightwave.agent.answer()."""
    _get_driver()  # attempt Neo4j connection (sets mode)
    _maybe_seed()
    return run_pipeline(question, question_id=question_id)


def run_eval_multi() -> dict:
    """Score all three questions with the multi-agent pipeline."""
    from nightwave.eval import PASS_THRESHOLD, score_question

    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    results = []

    print("=" * 65)
    print("  Nightwave Multi-Agent Eval — Madison Fields Case")
    print(f"  Graph mode: {get_mode()}")
    print("=" * 65)

    for q in questions:
        print(f"\n[{q['id'].upper()}] {q['tier']}")
        print(f"  {q['question'][:70]}...")

        ans = answer_multi(q["question"], question_id=q["id"])
        result = score_question(ans, q)
        results.append(result)

        s = result["scores"]
        print(f"  confidence : {ans.confidence:.2f}")
        print(f"  citations  : {len(ans.citations)}")
        print(f"  overall    : {s['overall']:.3f}")
        print(
            f"  correct    : {s['correctness']['overall']:.3f}  "
            f"grounding: {s['citation_grounding']['grounding_score']:.3f}  "
            f"must-cite: {s['must_cite']['score']:.3f}  "
            f"calibration: {s['confidence_calibration']['calibration_score']:.3f}"
        )
        if s["citation_grounding"]["hallucinated"]:
            print(f"  ⚠ HALLUCINATED IDs: {s['citation_grounding']['invalid_evidence_ids']}")
        if s["correctness"]["phrase_misses"]:
            print(f"  ⚠ missing phrases: {s['correctness']['phrase_misses']}")

    overall_score = sum(r["scores"]["overall"] for r in results) / len(results)
    passed = overall_score >= PASS_THRESHOLD

    report = {
        "overall_score": round(overall_score, 3),
        "passed": passed,
        "pass_threshold": PASS_THRESHOLD,
        "architecture": "multi-agent",
        "graph_mode": get_mode(),
        "per_question": results,
    }

    MULTI_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 65)
    status = "PASSED" if passed else "FAILED"
    print(f"  {status} — overall: {overall_score:.3f}  (threshold: {PASS_THRESHOLD})")
    print(f"  Report → {MULTI_REPORT_PATH}")
    print("=" * 65)

    return report


if __name__ == "__main__":
    _get_driver()
    report = run_eval_multi()
    sys.exit(0 if report["passed"] else 1)
