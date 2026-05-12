"""Evaluation harness.

Scores answer() on the three questions in questions.json. Metrics:

  1. Correctness (35%)  — must_contain_phrases + must_reference_entities/topics
  2. Citation grounding (30%) — every evidence_id must resolve to a real file;
     adversarial probe: hallucinated IDs trigger a 70% score penalty
  3. Must-cite coverage (20%) — required evidence_ids from expected must appear
  4. Confidence calibration (15%) — in expected_range or above min_confidence

Pass threshold: 0.60.

Deterministic scoring: run_eval() separates agent execution from scoring.
Raw answers are written to raw_answers.json after the LLM pass. Pass
--cached (or run with EVAL_USE_CACHE=1) to score from disk without re-rolling.

    # Full run (LLM + score):
    python -m nightwave.eval

    # Deterministic score-only run (no LLM):
    python -m nightwave.eval --cached
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from nightwave.agent import Answer, Citation
from nightwave.agent import answer as answer_single
from nightwave.case import load_case_json

QUESTIONS_PATH = Path(__file__).resolve().parent.parent.parent / "questions.json"
REPORT_PATH = Path(__file__).resolve().parent.parent / "eval_report.json"
RAW_ANSWERS_PATH = Path(__file__).resolve().parent.parent / "raw_answers.json"
PASS_THRESHOLD = 0.60

_case = load_case_json()
_valid_evidence_ids: set[str] = {ev["evidence_id"] for ev in _case.get("evidence", [])}
# Map entity id → name for entity presence checks
_entity_names: dict[str, str] = {e["id"]: e.get("name", "") for e in _case.get("entities", [])}
_valid_artifact_ids: set[str] = (
    {e["id"] for e in _case.get("entities", [])}
    | {h["id"] for h in _case.get("hypotheses", [])}
    | {lead["id"] for lead in _case.get("leads", [])}
    | {ev["id"] for ev in _case.get("events", [])}
    | {loc["id"] for loc in _case.get("locations", [])}
)


# ─── Scorers ─────────────────────────────────────────────────────────────────


_SYNONYMS: dict[str, list[str]] = {
    "camo": ["camouflage", "camo"],
    "camouflage": ["camouflage", "camo"],
}


def _phrase_match(phrase: str, text: str) -> bool:
    """Flexible phrase match: handles quoted words and common abbreviations."""
    p = phrase.lower()
    # Direct match
    if p in text:
        return True
    # Strip punctuation around each word and retry
    stripped = re.sub(r'["\']', "", text)
    if p in stripped:
        return True
    # Synonym expansion: replace each word with its synonyms and test
    words = p.split()
    for i, w in enumerate(words):
        for syn in _SYNONYMS.get(w, []):
            candidate = " ".join(words[:i] + [syn] + words[i + 1 :])
            if candidate in text or candidate in stripped:
                return True
    return False


def score_correctness(ans: Answer, expected: dict) -> dict:
    text = ans.response.lower()

    phrases = expected.get("must_contain_phrases", [])
    phrase_hits = [p for p in phrases if _phrase_match(p, text)]
    phrase_score = len(phrase_hits) / len(phrases) if phrases else 1.0

    entities = expected.get("must_reference_entities", [])
    entity_hits = []
    for eid in entities:
        # Accept entity ID directly in text, or entity's display name
        name = _entity_names.get(eid, "")
        if eid in ans.response or (name and name.lower() in text):
            entity_hits.append(eid)
    entity_score = len(entity_hits) / len(entities) if entities else 1.0

    topics = expected.get("must_reference_leads_or_topics", [])

    def _topic_match(topic: str) -> bool:
        # Exact phrase first
        if topic.lower() in text:
            return True
        # All significant words present anywhere in the response
        words = [w for w in topic.lower().split() if len(w) > 3]
        return all(w in text for w in words)

    topic_hits = [t for t in topics if _topic_match(t)]
    topic_score = len(topic_hits) / len(topics) if topics else 1.0

    # Q3: check min_actions count via numbered/bulleted items
    min_actions = expected.get("min_actions", 0)
    action_ok = True
    if min_actions:
        count = len(re.findall(r"(?:^|\n)\s*(?:\d+\.|[-•*])\s+\S", ans.response))
        action_ok = count >= min_actions

    overall = (phrase_score + entity_score + topic_score) / 3

    return {
        "phrase_score": round(phrase_score, 3),
        "phrase_hits": phrase_hits,
        "phrase_misses": [p for p in phrases if p not in phrase_hits],
        "entity_score": round(entity_score, 3),
        "entity_hits": entity_hits,
        "entity_misses": [e for e in entities if e not in entity_hits],
        "topic_score": round(topic_score, 3),
        "topic_hits": topic_hits,
        "topic_misses": [t for t in topics if t not in topic_hits],
        "action_count_ok": action_ok,
        "overall": round(overall, 3),
    }


def score_citation_grounding(ans: Answer) -> dict:
    """
    Validate every citation the agent returned.
    Adversarial probe: any hallucinated evidence_id triggers the hard penalty.
    """
    invalid_ids: list[str] = []
    empty_excerpts: list[str] = []

    for c in ans.citations:
        if c.evidence_id not in _valid_evidence_ids:
            invalid_ids.append(c.evidence_id)
        if not c.excerpt.strip():
            empty_excerpts.append(c.evidence_id)

    total = len(ans.citations)
    valid_count = total - len(invalid_ids)
    grounding_score = valid_count / total if total > 0 else 0.0

    return {
        "total_citations": total,
        "valid_citations": valid_count,
        "invalid_evidence_ids": invalid_ids,
        "empty_excerpts": empty_excerpts,
        "hallucinated": len(invalid_ids) > 0,
        "grounding_score": round(grounding_score, 3),
    }


def score_must_cite(ans: Answer, expected: dict) -> dict:
    """Check that required evidence IDs (from expected) appear in the agent's citations."""
    cited_ids = {c.evidence_id for c in ans.citations}
    # Also check if the ID or hypothesis ID appears in the response text
    required: list[str] = expected.get("must_cite_evidence_ids", []) + expected.get(
        "must_cite_evidence_or_hypotheses", []
    )
    hits = [r for r in required if r in cited_ids or r in ans.response]
    score = len(hits) / len(required) if required else 1.0

    return {
        "required": required,
        "hits": hits,
        "misses": [r for r in required if r not in hits],
        "score": round(score, 3),
    }


def score_confidence_calibration(ans: Answer, expected: dict) -> dict:
    conf = ans.confidence

    if "confidence_range" in expected:
        lo, hi = expected["confidence_range"]
        in_range = lo <= conf <= hi
        midpoint = (lo + hi) / 2
        # Partial credit: how far outside the range is the confidence?
        cal_score = 1.0 if in_range else max(0.0, 1.0 - abs(conf - midpoint) * 2)
        return {
            "confidence": conf,
            "expected_range": [lo, hi],
            "in_range": in_range,
            "calibration_score": round(cal_score, 3),
        }

    if "min_confidence" in expected:
        min_c = expected["min_confidence"]
        above_min = conf >= min_c
        cal_score = 1.0 if above_min else conf / min_c
        return {
            "confidence": conf,
            "min_confidence": min_c,
            "above_min": above_min,
            "calibration_score": round(cal_score, 3),
        }

    return {"confidence": conf, "calibration_score": 1.0}


def score_question(ans: Answer, q: dict) -> dict:
    expected = q["expected"]
    correctness = score_correctness(ans, expected)
    grounding = score_citation_grounding(ans)
    must_cite = score_must_cite(ans, expected)
    calibration = score_confidence_calibration(ans, expected)

    overall = (
        correctness["overall"] * 0.35
        + grounding["grounding_score"] * 0.30
        + must_cite["score"] * 0.20
        + calibration["calibration_score"] * 0.15
    )

    # Hard penalty for hallucinated citations — the primary failure mode
    if grounding["hallucinated"]:
        overall *= 0.30

    return {
        "question_id": q["id"],
        "tier": q["tier"],
        "question": q["question"],
        "response": ans.response,
        "confidence": ans.confidence,
        "citations": [
            {
                "evidence_id": c.evidence_id,
                "source_path": c.source_path,
                "excerpt": c.excerpt[:120],
                "confidence": c.confidence,
            }
            for c in ans.citations
        ],
        "trace_steps": len(ans.trace),
        "scores": {
            "correctness": correctness,
            "citation_grounding": grounding,
            "must_cite": must_cite,
            "confidence_calibration": calibration,
            "overall": round(overall, 3),
        },
    }


# ─── Serialisation ────────────────────────────────────────────────────────────


def _serialize_answers(answers: list[tuple[str, Answer]]) -> str:
    """Serialise (question_id, Answer) pairs to JSON for deterministic replay."""
    out = []
    for qid, ans in answers:
        out.append(
            {
                "question_id": qid,
                "question": ans.question,
                "response": ans.response,
                "confidence": ans.confidence,
                "citations": [
                    {
                        "evidence_id": c.evidence_id,
                        "source_path": c.source_path,
                        "locator": c.locator,
                        "excerpt": c.excerpt,
                        "confidence": c.confidence,
                    }
                    for c in ans.citations
                ],
                "trace_len": len(ans.trace),
            }
        )
    return json.dumps(out, indent=2, default=str)


def _deserialize_answers(raw: str) -> dict[str, Answer]:
    """Load serialised answers; returns {question_id: Answer}."""
    items = json.loads(raw)
    result: dict[str, Answer] = {}
    for item in items:
        citations = [
            Citation(
                evidence_id=c["evidence_id"],
                source_path=c.get("source_path", ""),
                locator=c.get("locator", {}),
                excerpt=c["excerpt"],
                confidence=float(c["confidence"]),
            )
            for c in item.get("citations", [])
        ]
        result[item["question_id"]] = Answer(
            question=item["question"],
            response=item["response"],
            confidence=float(item["confidence"]),
            citations=citations,
            trace=[],  # trace not needed for scoring
        )
    return result


# ─── Runner ───────────────────────────────────────────────────────────────────


def run_eval(use_cache: bool = False) -> dict:
    """Run the full eval. Writes eval_report.json. Returns the report dict.

    use_cache=True: load answers from raw_answers.json (no LLM calls).
    use_cache=False: call answer() for each question, then write raw_answers.json.
    """
    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    results = []
    architecture = os.getenv("NIGHTWAVE_ARCHITECTURE", "multiagent").lower()

    print("=" * 65)
    print("  Nightwave Agent Eval — Madison Fields Case")
    print(f"  Architecture: {architecture}")
    if use_cache:
        print("  [CACHED — scoring from raw_answers.json, no LLM calls]")
    print("=" * 65)

    # ── Agent execution pass (skipped in cached mode) ─────────────────────────
    cached_answers: dict[str, Answer] = {}
    if use_cache:
        if not RAW_ANSWERS_PATH.exists():
            raise FileNotFoundError(
                f"{RAW_ANSWERS_PATH} not found. Run without --cached first."
            )
        cached_answers = _deserialize_answers(RAW_ANSWERS_PATH.read_text())
    else:
        collected: list[tuple[str, Answer]] = []
        if architecture == "single":
            def answer_fn(question: str, question_id: str) -> Answer:
                return answer_single(question)

            graph_mode = "n/a"
        else:
            from nightwave.multiagent.graph_db import get_mode
            from nightwave.multiagent.run import answer_multi

            answer_fn = answer_multi
            graph_mode = get_mode()
        for q in questions:
            print(f"\n[{q['id'].upper()}] Running agent…")
            ans = answer_fn(q["question"], q["id"])
            collected.append((q["id"], ans))
            cached_answers[q["id"]] = ans
        RAW_ANSWERS_PATH.write_text(_serialize_answers(collected))
        print(f"\n  Raw answers → {RAW_ANSWERS_PATH}")

    # ── Scoring pass (always deterministic) ───────────────────────────────────
    for q in questions:
        print(f"\n[{q['id'].upper()}] {q['tier']}")
        print(f"  {q['question'][:70]}...")

        ans = cached_answers[q["id"]]
        result = score_question(ans, q)
        results.append(result)

        s = result["scores"]
        print(f"  confidence : {ans.confidence:.2f}")
        print(f"  citations  : {len(ans.citations)}")
        print(f"  overall    : {s['overall']:.3f}")
        print(f"  correct    : {s['correctness']['overall']:.3f}  "
              f"grounding: {s['citation_grounding']['grounding_score']:.3f}  "
              f"must-cite: {s['must_cite']['score']:.3f}  "
              f"calibration: {s['confidence_calibration']['calibration_score']:.3f}")

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
        "architecture": architecture,
        "graph_mode": graph_mode if not use_cache else "cached",
        "per_question": results,
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 65)
    status = "PASSED" if passed else "FAILED"
    print(f"  {status} — overall: {overall_score:.3f}  (threshold: {PASS_THRESHOLD})")
    print(f"  Report → {REPORT_PATH}")
    print("=" * 65)

    return report


if __name__ == "__main__":
    use_cache = "--cached" in sys.argv or os.getenv("EVAL_USE_CACHE", "") == "1"
    report = run_eval(use_cache=use_cache)
    sys.exit(0 if report["passed"] else 1)
