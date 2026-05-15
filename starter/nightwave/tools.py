"""Tools the agent calls.

Implementation choices:
- search_evidence: BM25-ish token scoring over an in-memory corpus.
  14 files, ~50KB — no embedding or BM25 library needed. Text files are
  chunked at 5-line boundaries; the PDF is parsed with pypdf; ai_summary
  provides a fallback for images and videos. Summaries penalised 20% so
  raw text wins on ties.

- get_entity: O(1) dict lookup by id or lowercased name/alias. Citations
  eagerly attached from a pre-built index.

- query_graph: Routes through typed sub-queries (connected_to, events_for,
  hypotheses_and_leads). Never lets the LLM write SQL.

- list_citations: Index lookup by artifact_id.

Every tool return includes citation dicts so every factual claim is traceable.
"""

from __future__ import annotations

from typing import Any

from nightwave.case_store import CaseIndex, tokenize

# ─── Tokenizer (defined before _CaseIndex so the class can reference it) ─────


def _tokenize(text: str) -> list[str]:
    return tokenize(text)


# ─── In-memory index (built once at module load) ──────────────────────────────


_INDEX = CaseIndex()


# ─── BM25-ish scoring ─────────────────────────────────────────────────────────


def _bm25_score(query_tokens: list[str], doc: dict[str, Any]) -> float:
    k1, b = 1.5, 0.75
    doc_tokens = _tokenize(doc["text"])
    dl = len(doc_tokens)
    avg_dl = _INDEX._avg_dl

    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1

    score = 0.0
    for token in query_tokens:
        tf = tf_map.get(token, 0)
        if tf == 0:
            continue
        # Simplified IDF=1.0: 14-doc corpus is too small for meaningful DF weighting
        score += tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))

    if doc.get("is_summary"):
        score *= 0.8  # penalise summaries so raw text wins when both match
    return score


# ─── Tools ───────────────────────────────────────────────────────────────────


def search_evidence(query: str, k: int = 5, case_id: str | None = None) -> list[dict[str, Any]]:
    """BM25-ish search over the in-memory evidence corpus.

    Returns up to k hits, each with evidence_id, filename, excerpt, locator,
    score, and evidence_confidence — everything the agent needs to cite.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    active_case_id = case_id or _INDEX.case_id
    scored = [
        (doc, _bm25_score(q_tokens, doc))
        for doc in _INDEX.corpus
        if doc.get("case_id") == active_case_id
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results: list[dict[str, Any]] = []
    for doc, score in scored[:k]:
        if score == 0:
            break
        ev = _INDEX.evidence_by_id.get(doc["evidence_id"], {})
        results.append(
            {
                "evidence_id": doc["evidence_id"],
                "case_id": active_case_id,
                "filename": doc["filename"],
                "excerpt": doc["text"][:400],
                "locator": doc["locator"],
                "score": round(score, 4),
                "source_type": doc["source_type"],
                "evidence_name": ev.get("evidence_name", ""),
                "evidence_confidence": ev.get("confidence", 0.5),
            }
        )
    return results


def get_entity(name_or_id: str, case_id: str | None = None) -> dict[str, Any] | None:
    """Look up an entity by id or name (case-insensitive, alias-aware).

    Returns the entity record with citations attached, or None if not found.
    """
    needle = name_or_id.strip()
    active_case_id = case_id or _INDEX.case_id
    if active_case_id != _INDEX.case_id:
        return None
    entity = _INDEX.entities_by_id.get(needle) or _INDEX.entities_by_name.get(needle.lower())

    if entity is None:
        # Substring match as last resort
        for name, e in _INDEX.entities_by_name.items():
            if needle.lower() in name:
                entity = e
                break

    if entity is None:
        return None

    result = dict(entity)
    result["citations"] = _INDEX.citations_by_artifact.get(entity["id"], [])
    return result


def query_graph(question: str, case_id: str | None = None) -> list[dict[str, Any]]:
    """Resolve a structural question against the case graph.

    Routes to typed sub-queries; never lets the LLM write SQL.
    Covers: entity connections, timelines, leads, hypotheses.
    """
    q = question.lower()
    active_case_id = case_id or _INDEX.case_id
    if active_case_id != _INDEX.case_id:
        return []
    results: list[dict[str, Any]] = []

    def _connected_to(entity_id: str) -> list[dict]:
        hits = []
        for rel in _INDEX.relationships:
            other_id = None
            if rel.get("source_entity_id") == entity_id:
                other_id = rel.get("target_entity_id")
            elif rel.get("target_entity_id") == entity_id:
                other_id = rel.get("source_entity_id")
            if other_id:
                other = _INDEX.entities_by_id.get(other_id, {})
                hits.append(
                    {
                        "artifact_type": "relationship",
                        **rel,
                        "related_entity_name": other.get("name", other_id),
                        "citations": _INDEX.citations_by_artifact.get(rel["id"], []),
                    }
                )
        return hits

    def _events_for(entity_id: str) -> list[dict]:
        hits = []
        for ev in _INDEX.events:
            if entity_id in ev.get("entity_ids", []):
                hits.append(
                    {
                        "artifact_type": "event",
                        **ev,
                        "citations": _INDEX.citations_by_artifact.get(ev["id"], []),
                    }
                )
        return hits

    def _all_leads_and_hypotheses() -> list[dict]:
        items = []
        for h in _INDEX.hypotheses:
            items.append({"artifact_type": "hypothesis", **h})
        for lead in _INDEX.leads:
            items.append({"artifact_type": "lead", **lead})
        return items

    relationship_terms = (
        "connected",
        "linked",
        "related",
        "relationship",
        "knows",
        "involved",
    )
    if any(kw in q for kw in relationship_terms):
        for eid, entity in _INDEX.entities_by_id.items():
            if entity.get("name", "").lower() in q:
                results.extend(_connected_to(eid))
                results.extend(_events_for(eid))
                break
        if not results:  # fallback: return all relationships
            for rel in _INDEX.relationships:
                results.append({"artifact_type": "relationship", **rel})

    planning_terms = (
        "hypothes",
        "lead",
        "suspect",
        "theory",
        "action",
        "next step",
        "priorit",
    )
    if any(kw in q for kw in planning_terms):
        results.extend(_all_leads_and_hypotheses())

    if any(kw in q for kw in ("event", "happened", "timeline", "when", "chronolog")):
        for ev in _INDEX.events:
            results.append(
                {
                    "artifact_type": "event",
                    **ev,
                    "citations": _INDEX.citations_by_artifact.get(ev["id"], []),
                }
            )

    # Fallback: return entities + leads + hypotheses
    if not results:
        for eid, entity in _INDEX.entities_by_id.items():
            results.append(
                {
                    "artifact_type": "entity",
                    **entity,
                    "citations": _INDEX.citations_by_artifact.get(eid, []),
                }
            )
        results.extend(_all_leads_and_hypotheses())

    return results[:25]  # cap to stay within context window


def list_citations(artifact_id: str, case_id: str | None = None) -> list[dict[str, Any]]:
    """Return every citation pointing to artifact_id, with evidence metadata attached."""
    active_case_id = case_id or _INDEX.case_id
    if active_case_id != _INDEX.case_id:
        return []
    citations = _INDEX.citations_by_artifact.get(artifact_id, [])
    result = []
    for c in citations:
        ev = _INDEX.evidence_by_id.get(c.get("evidence_id", ""), {})
        result.append(
            {
                **c,
                "evidence_name": ev.get("evidence_name", ""),
                "evidence_type": ev.get("evidence_type", "unknown"),
                "evidence_confidence": ev.get("confidence", 0.5),
            }
        )
    return result
