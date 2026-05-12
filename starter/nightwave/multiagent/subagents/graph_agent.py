"""Graph Agent subagent — structured traversal over Neo4j (or memory fallback).

Runs async so orchestrator can gather it with retriever in parallel.
Populates state.graph_context with entities, hypotheses, leads, relationships.

Three Cypher query shapes (matching the three question tiers):
  Q1 (retrieval)       — entity lookup + citations
  Q2 (multi-hop)       — path traversal between entities via platforms
  Q3 (planning)        — leads + hypotheses sorted by priority/confidence
"""

from __future__ import annotations

from typing import Any

from nightwave.multiagent.graph_db import get_mode, run_cypher
from nightwave.multiagent.nlp_utils import extract_proper_nouns
from nightwave.multiagent.state import AgentState

# ── Cypher shapes ─────────────────────────────────────────────────────────────

def _cypher_entity_context(name_fragment: str) -> list[dict]:
    """All entities + their relationships where name contains fragment."""
    return run_cypher(
        """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
        OPTIONAL MATCH (e)-[r]-(other:Entity)
        RETURN e, collect({rel: type(r), other: other.name, conf: r.confidence}) AS rels
        """,
        {"name": name_fragment},
    )


def _cypher_entity_with_citations(entity_id: str) -> list[dict]:
    """Return entity + all evidence IDs that cite it (for grounding the LLM)."""
    rows = run_cypher(
        """
        MATCH (e:Entity {id: $eid})
        OPTIONAL MATCH (e)-[:CITES]->(ev:Evidence)
        RETURN e.id AS id, e.name AS name, e.entity_type AS entity_type,
               e.summary AS summary,
               collect(ev.evidence_id) AS cited_by_evidence_ids
        """,
        {"eid": entity_id},
    )
    return [{**r, "artifact_type": "entity"} for r in rows]


def _cypher_platform_path() -> list[dict]:
    """Return entities + their evidence citations for platform-identity questions."""
    rows = run_cypher(
        """
        MATCH (a:Entity)-[r]-(b:Entity)
        WHERE a.id <> b.id
        WITH a, b, type(r) AS rel_type, r.confidence AS conf
        OPTIONAL MATCH (a)-[:CITES]->(ev_a:Evidence)
        OPTIONAL MATCH (b)-[:CITES]->(ev_b:Evidence)
        RETURN a.id AS from_id, a.name AS from_name,
               b.id AS to_id, b.name AS to_name,
               rel_type, conf,
               collect(DISTINCT ev_a.evidence_id) AS from_evidence,
               collect(DISTINCT ev_b.evidence_id) AS to_evidence
        LIMIT 20
        """
    )
    return [{**r, "artifact_type": "relationship"} for r in rows]


def _cypher_leads_and_hypotheses() -> list[dict]:
    """All leads + hypotheses with supporting hypotheses listed."""
    rows = run_cypher(
        """
        MATCH (l:Lead)
        OPTIONAL MATCH (l)<-[:SUPPORTS]-(h:Hypothesis)
        RETURN l.id AS id, l.title AS title, l.priority AS priority,
               l.status AS status, collect(h.id) AS supporting_hypothesis_ids
        ORDER BY
          CASE l.priority WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                          WHEN 'medium' THEN 3 ELSE 4 END
        """
    )
    return [{**r, "artifact_type": "lead"} for r in rows]


def _cypher_hypotheses() -> list[dict]:
    rows = run_cypher(
        """
        MATCH (h:Hypothesis)
        RETURN h.id AS id, h.title AS statement, h.description AS description,
               h.confidence AS confidence, h.status AS status
        ORDER BY h.confidence DESC
        """
    )
    return [{**r, "artifact_type": "hypothesis"} for r in rows]


# ── Memory fallback ───────────────────────────────────────────────────────────

def _memory_graph_context(question: str) -> list[dict[str, Any]]:
    """In-memory fallback using the case index directly."""
    from nightwave.tools import _INDEX
    q = question.lower()
    results: list[dict[str, Any]] = []

    if any(w in q for w in ("hypothes", "lead", "suspect", "theory", "action", "next", "priorit", "plan")):
        for h in _INDEX.hypotheses:
            results.append({
                "artifact_type": "hypothesis",
                "id": h["id"],
                "statement": h.get("statement", h.get("title", "")),
                "confidence": h.get("confidence", 0.5),
                "status": h.get("status", ""),
            })
        for lead in _INDEX.leads:
            results.append({
                "artifact_type": "lead",
                "id": lead["id"],
                "title": lead.get("title", ""),
                "priority": lead.get("priority", ""),
                "status": lead.get("status", ""),
                "description": lead.get("description", ""),
            })

    if any(w in q for w in ("connected", "related", "relationship", "person", "who", "same", "platform", "app", "contact")):
        for rel in _INDEX.relationships:
            results.append({"artifact_type": "relationship", **rel})
        for eid, entity in _INDEX.entities_by_id.items():
            results.append({
                "artifact_type": "entity",
                **entity,
                "citations": _INDEX.citations_by_artifact.get(eid, []),
            })

    if any(w in q for w in ("event", "when", "timeline", "happened")):
        for ev in _INDEX.events:
            results.append({"artifact_type": "event", **ev})

    # Default: all leads + hypotheses
    if not results:
        for h in _INDEX.hypotheses:
            results.append({"artifact_type": "hypothesis", "id": h["id"],
                "statement": h.get("statement", ""), "confidence": h.get("confidence", 0.5)})
        for lead in _INDEX.leads:
            results.append({"artifact_type": "lead", "id": lead["id"],
                "title": lead.get("title", ""), "priority": lead.get("priority", "")})

    return results[:30]


# ── Main entry ────────────────────────────────────────────────────────────────

def _classify_question(q: str) -> str:
    """Q1=retrieval, Q2=multi_hop, Q3=planning."""
    q_lower = q.lower()
    if any(w in q_lower for w in ("wearing", "last seen", "clothing", "description", "physical")):
        return "retrieval"
    multi_hop_terms = (
        "same person",
        "separate",
        "groomed",
        "two",
        "both platform",
        "both app",
    )
    if any(w in q_lower for w in multi_hop_terms):
        return "multi_hop"
    if any(w in q_lower for w in ("next", "action", "priorit", "investigat", "rank", "step")):
        return "planning"
    return "general"


def run_graph_agent(state: AgentState) -> AgentState:
    """Populate state.graph_context via Cypher or memory fallback."""
    q_class = _classify_question(state.question)
    ctx: list[dict[str, Any]] = []

    if get_mode() == "neo4j":
        try:
            if q_class == "retrieval":
                nouns = extract_proper_nouns(state.question)
                for noun in nouns[:4]:
                    ctx += _cypher_entity_context(noun)
                if not ctx:
                    ctx = _cypher_leads_and_hypotheses()
            elif q_class == "multi_hop":
                from nightwave.tools import _INDEX
                ctx = _cypher_platform_path()
                ctx += _cypher_hypotheses()
                nouns = extract_proper_nouns(state.question)
                matched_ids = [
                    eid for eid, ent in _INDEX.entities_by_id.items()
                    if any(n.lower() in ent.get("name", "").lower() for n in nouns)
                ]
                for eid in matched_ids[:4]:
                    ctx += _cypher_entity_with_citations(eid)
            else:  # planning / general
                ctx = _cypher_leads_and_hypotheses()
                ctx += _cypher_hypotheses()
        except Exception as exc:
            state.trace.append({
                "step": "graph_agent",
                "warning": f"Cypher failed: {exc}, using memory",
            })
            ctx = _memory_graph_context(state.question)
    else:
        ctx = _memory_graph_context(state.question)

    state.graph_context = ctx[:30]
    state.trace.append({
        "step": "graph_agent",
        "mode": get_mode(),
        "question_class": q_class,
        "context_items": len(state.graph_context),
    })
    return state
