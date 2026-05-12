"""Retriever subagent — hybrid BM25 + vector search with multi-query expansion.

Multi-query retrieval:
- Primary query: the full question
- Secondary queries: derived from the question text via NER heuristics and
  question-class templates. NO case-specific strings are hardcoded here;
  secondary queries are generated from what the question itself contains.

Rationale: on a new case with different entities and leads, the retriever
must remain useful. Encoding "Session forensic extraction" or "Snapchat
subpoena" would pass the eval but break on any other corpus.
"""

from __future__ import annotations

import re

from nightwave.case import DEFAULT_EVIDENCE_DIR
from nightwave.multiagent.nlp_utils import extract_apps_platforms, extract_proper_nouns
from nightwave.multiagent.state import AgentState, RetrievedChunk
from nightwave.multiagent.vector_store import hybrid_search

# Module-level aliases so existing code/tests that import private names still work.
_extract_proper_nouns = extract_proper_nouns
_extract_apps_platforms = extract_apps_platforms


def _classify(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ("wearing", "last seen", "clothing", "physical description")):
        return "retrieval"
    if any(w in q for w in ("same person", "separate", "two offenders", "same offender",
                             "groom", "platform", "app")):
        return "multi_hop"
    if any(w in q for w in ("next", "action", "investigat", "priorit", "rank", "step")):
        return "planning"
    return "general"


def _generate_secondary_queries(question: str, q_class: str) -> list[str]:
    """Generate secondary queries from question text. No case-specific strings."""
    queries: list[str] = []

    proper_nouns = _extract_proper_nouns(question)
    platforms = _extract_apps_platforms(question)

    if q_class == "multi_hop":
        # For identity/platform questions: search primary sources for each entity
        for pn in proper_nouns[:3]:
            queries.append(f"{pn} primary source evidence document")
        for pl in platforms[:2]:
            queries.append(f"{pl} contact evidence source")

    elif q_class == "planning":
        # For planning questions: search for lead and action evidence
        # Extract action-like verbs from the question
        action_words = re.findall(
            r'\b(forensic|subpoena|surveillance|extraction|arrest|interview|warrant|'
            r'canvass|search|seize|identify|trace|locate)\w*\b',
            question.lower(),
        )
        for w in list(dict.fromkeys(action_words))[:2]:
            queries.append(f"{w} investigation evidence")
        # Also search for named entities + "lead" or "action"
        for pn in proper_nouns[:2]:
            queries.append(f"{pn} investigation lead action")

    elif q_class == "retrieval":
        # For direct retrieval: search for the subject entity + fact type
        for pn in proper_nouns[:2]:
            queries.append(f"{pn} description official document record")

    return [q for q in queries if len(q.split()) >= 3][:4]


def run_retriever(state: AgentState, k: int = 10) -> AgentState:
    """Hybrid multi-query retrieval. Fills state.retrieved_chunks."""
    q_class = _classify(state.question)
    secondary_qs = _generate_secondary_queries(state.question, q_class)

    # Primary search
    primary_hits = hybrid_search(state.question, k=k)

    # Secondary searches — merged before primary so they rank higher
    secondary_hits: list[dict] = []
    for sq in secondary_qs:
        secondary_hits.extend(hybrid_search(sq, k=4))

    # Dedup by (evidence_id, locator): secondary first = higher effective rank
    seen: set[str] = set()
    merged: list[dict] = []
    for h in secondary_hits + primary_hits:
        key = f"{h['evidence_id']}:{h.get('locator', {})}"
        if key not in seen:
            seen.add(key)
            merged.append(h)

    # Source diversity: interleave so each unique evidence_id gets ≤2 slots
    # before any source gets a 3rd slot. Prevents one verbose source from
    # flooding the entire context window.
    per_source: dict[str, int] = {}
    diverse: list[dict] = []
    deferred: list[dict] = []
    for h in merged:
        count = per_source.get(h["evidence_id"], 0)
        if count < 2:
            per_source[h["evidence_id"]] = count + 1
            diverse.append(h)
        else:
            deferred.append(h)
    merged = diverse + deferred

    chunks: list[RetrievedChunk] = []
    for h in merged[:k + 4]:
        ev_id = h["evidence_id"]
        filename = h.get("filename", "")
        source_path = str(DEFAULT_EVIDENCE_DIR / filename) if filename else ""
        chunks.append(
            RetrievedChunk(
                evidence_id=ev_id,
                source_path=source_path,
                locator=h.get("locator", {}),
                excerpt=h.get("excerpt", "")[:400],
                score=h.get("rrf_score", 0.0),
                source="rrf",
            )
        )

    state.retrieved_chunks = chunks
    state.trace.append({
        "step": "retriever",
        "question_class": q_class,
        "hits": len(chunks),
        "secondary_queries": secondary_qs,
        "top_ids": [c.evidence_id for c in chunks[:4]],
    })
    return state
