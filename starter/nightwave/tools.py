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

import re
import sqlite3
from typing import Any

try:
    from pypdf import PdfReader

    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False

from nightwave.case import DEFAULT_DB, DEFAULT_EVIDENCE_DIR, load_case_json

# ─── Tokenizer (defined before _CaseIndex so the class can reference it) ─────


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ─── In-memory index (built once at module load) ──────────────────────────────


class _CaseIndex:
    """Pre-computed views over the case data. Avoids re-reading files per call."""

    def __init__(self) -> None:
        self.case = load_case_json()
        self.case_id: str = self.case["case_id"]

        self.entities_by_id: dict[str, dict] = {}
        self.entities_by_name: dict[str, dict] = {}
        for e in self.case.get("entities", []):
            self.entities_by_id[e["id"]] = e
            self.entities_by_name[e.get("name", "").lower().strip()] = e
            for alias in e.get("aliases", []):
                if alias:
                    self.entities_by_name[alias.lower().strip()] = e

        self.citations_by_artifact: dict[str, list[dict]] = {}
        for c in self.case.get("citations", []):
            aid = c.get("artifact_id", "")
            if aid:
                self.citations_by_artifact.setdefault(aid, []).append(c)

        self.evidence_by_id: dict[str, dict] = {}
        for ev in self.case.get("evidence", []):
            self.evidence_by_id[ev["evidence_id"]] = ev

        self.leads: list[dict] = self.case.get("leads", [])
        self.hypotheses: list[dict] = self.case.get("hypotheses", [])
        self.relationships: list[dict] = self.case.get("relationships", [])
        self.events: list[dict] = self.case.get("events", [])

        self.corpus: list[dict[str, Any]] = self._build_corpus()
        self._avg_dl: float = (
            sum(len(_tokenize(d["text"])) for d in self.corpus) / len(self.corpus)
            if self.corpus
            else 100.0
        )

    def _build_corpus(self) -> list[dict[str, Any]]:
        corpus: list[dict[str, Any]] = []
        for ev in self.case.get("evidence", []):
            ev_id = ev["evidence_id"]
            filename = ev["filename"]
            path = DEFAULT_EVIDENCE_DIR / filename

            if ev.get("mime_type") == "text/plain" or path.suffix == ".txt":
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    lines = text.splitlines()
                    for i in range(0, len(lines), 5):
                        chunk = " ".join(lines[i : i + 5]).strip()
                        if chunk:
                            corpus.append(
                                {
                                    "evidence_id": ev_id,
                                    "filename": filename,
                                    "text": chunk,
                                    "locator": {
                                        "type": "text",
                                        "line_start": i + 1,
                                        "line_end": min(i + 5, len(lines)),
                                    },
                                    "source_type": "text",
                                    "is_summary": False,
                                }
                            )
                except Exception:
                    pass

            elif ev.get("mime_type") == "application/pdf" or path.suffix == ".pdf":
                if _HAS_PYPDF:
                    try:
                        reader = PdfReader(str(path))
                        for pg_num, page in enumerate(reader.pages, 1):
                            text = (page.extract_text() or "").strip()
                            if text:
                                corpus.append(
                                    {
                                        "evidence_id": ev_id,
                                        "filename": filename,
                                        "text": text,
                                        "locator": {"type": "pdf", "page": pg_num},
                                        "source_type": "document",
                                        "is_summary": False,
                                    }
                                )
                    except Exception:
                        pass

            # Always include ai_summary + description as fallback / supplemental
            summary = f"{ev.get('description', '')} {ev.get('ai_summary', '')}".strip()
            if summary:
                corpus.append(
                    {
                        "evidence_id": ev_id,
                        "filename": filename,
                        "text": summary,
                        "locator": {"type": "text", "line_start": 1, "line_end": 1},
                        "source_type": ev.get("evidence_type", "unknown"),
                        "is_summary": True,
                    }
                )

        return corpus

    def get_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DEFAULT_DB))
        conn.row_factory = sqlite3.Row
        return conn


_INDEX = _CaseIndex()


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


def search_evidence(query: str, k: int = 5) -> list[dict[str, Any]]:
    """BM25-ish search over the in-memory evidence corpus.

    Returns up to k hits, each with evidence_id, filename, excerpt, locator,
    score, and evidence_confidence — everything the agent needs to cite.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored = [(doc, _bm25_score(q_tokens, doc)) for doc in _INDEX.corpus]
    scored.sort(key=lambda x: x[1], reverse=True)

    results: list[dict[str, Any]] = []
    for doc, score in scored[:k]:
        if score == 0:
            break
        ev = _INDEX.evidence_by_id.get(doc["evidence_id"], {})
        results.append(
            {
                "evidence_id": doc["evidence_id"],
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


def get_entity(name_or_id: str) -> dict[str, Any] | None:
    """Look up an entity by id or name (case-insensitive, alias-aware).

    Returns the entity record with citations attached, or None if not found.
    """
    needle = name_or_id.strip()
    entity = _INDEX.entities_by_id.get(needle) or _INDEX.entities_by_name.get(
        needle.lower()
    )

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


def query_graph(question: str) -> list[dict[str, Any]]:
    """Resolve a structural question against the case graph.

    Routes to typed sub-queries; never lets the LLM write SQL.
    Covers: entity connections, timelines, leads, hypotheses.
    """
    q = question.lower()
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


def list_citations(artifact_id: str) -> list[dict[str, Any]]:
    """Return every citation pointing to artifact_id, with evidence metadata attached."""
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
