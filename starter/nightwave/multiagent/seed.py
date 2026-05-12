"""Idempotent Neo4j seed from case_data.json.

Run once (or every time — MERGE is idempotent):
    python -m nightwave.multiagent.seed

Schema
------
Nodes  : Entity, Event, Lead, Hypothesis, Location, Evidence
Edges  : KNOWS, WITNESSED, OCCURS_AT, CITES, SUPPORTS{confidence},
         CONTRADICTS{confidence}, TAGGED

All IDs use the case_data `id` or `evidence_id` fields as stable keys.
"""

from __future__ import annotations

from nightwave.case import load_case_json
from nightwave.multiagent.graph_db import _get_driver, get_mode, run_cypher


def _merge_node(label: str, props: dict) -> None:
    key = "id" if "id" in props else "evidence_id"
    safe = {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool))}
    cypher = (
        f"MERGE (n:{label} {{{key}: $key_val}}) "
        "SET n += $props"
    )
    run_cypher(cypher, {"key_val": props[key], "props": safe})


def _merge_edge(
    from_label: str, from_id: str,
    rel: str,
    to_label: str, to_id: str,
    props: dict | None = None,
) -> None:
    cypher = (
        f"MATCH (a:{from_label} {{id: $from_id}}), (b:{to_label} {{id: $to_id}}) "
        f"MERGE (a)-[r:{rel}]->(b) "
        "SET r += $props"
    )
    run_cypher(cypher, {
        "from_id": from_id,
        "to_id": to_id,
        "props": props or {},
    })


def seed() -> None:
    if get_mode() != "neo4j":
        print("[seed] Neo4j not available — skipping seed")
        return

    case = load_case_json()

    # ── Constraints (idempotent) ──────────────────────────────────────────────
    for label, key in [
        ("Entity", "id"), ("Event", "id"), ("Lead", "id"),
        ("Hypothesis", "id"), ("Location", "id"), ("Evidence", "evidence_id"),
    ]:
        try:
            run_cypher(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
            )
        except Exception:
            pass  # older Neo4j versions

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for e in case.get("entities", []):
        _merge_node("Entity", {
            "id": e["id"],
            "name": e.get("name", ""),
            "entity_type": e.get("entity_type", ""),
            "status": e.get("status", ""),
            "age": e.get("age") or 0,
            "summary": (e.get("summary") or "")[:500],
        })

    for ev in case.get("events", []):
        _merge_node("Event", {
            "id": ev["id"],
            "event_type": ev.get("event_type", ""),
            "description": (ev.get("description") or "")[:500],
            "occurred_at": str(ev.get("occurred_at") or ""),
            "confidence": float(ev.get("confidence") or 0.5),
        })

    for loc in case.get("locations", []):
        _merge_node("Location", {
            "id": loc["id"],
            "name": loc.get("name", ""),
            "location_type": loc.get("location_type", ""),
            "address": loc.get("address") or "",
        })

    for hyp in case.get("hypotheses", []):
        _merge_node("Hypothesis", {
            "id": hyp["id"],
            "title": hyp.get("title") or hyp.get("statement", ""),
            "description": (hyp.get("description") or hyp.get("reasoning") or "")[:500],
            "confidence": float(hyp.get("confidence") or 0.5),
            "status": hyp.get("status", ""),
        })

    for lead in case.get("leads", []):
        _merge_node("Lead", {
            "id": lead["id"],
            "title": lead.get("title", ""),
            "description": (lead.get("description") or "")[:500],
            "priority": lead.get("priority", ""),
            "status": lead.get("status", ""),
        })

    for ev in case.get("evidence", []):
        _merge_node("Evidence", {
            "evidence_id": ev["evidence_id"],
            "id": ev["evidence_id"],  # duplicate for edge matching
            "evidence_name": ev.get("evidence_name", ""),
            "evidence_type": ev.get("evidence_type", ""),
            "confidence": float(ev.get("confidence") or 0.5),
            "priority": ev.get("priority", ""),
        })

    # ── Entity→Entity relationships (KNOWS / typed) ───────────────────────────
    for rel in case.get("relationships", []):
        rel_type = rel.get("relationship_type", "KNOWS").upper().replace(" ", "_").replace("-", "_")
        try:
            _merge_edge(
                "Entity", rel["source_entity_id"],
                rel_type,
                "Entity", rel["target_entity_id"],
                {
                    "confidence": float(rel.get("confidence") or 0.5),
                    "description": rel.get("description", "")[:200],
                },
            )
        except Exception:
            pass  # entity may not exist for non-entity nodes

    # ── Events link entities and locations ───────────────────────────────────
    for ev in case.get("events", []):
        for eid in ev.get("entity_ids", []):
            try:
                _merge_edge("Entity", eid, "WITNESSED", "Event", ev["id"])
            except Exception:
                pass
        for lid in ev.get("location_ids", []):
            try:
                _merge_edge("Event", ev["id"], "OCCURS_AT", "Location", lid)
            except Exception:
                pass

    # ── Citations → Evidence ──────────────────────────────────────────────────
    for cit in case.get("citations", []):
        aid = cit.get("artifact_id", "")
        eid = cit.get("evidence_id", "")
        if not aid or not eid:
            continue
        # Determine source node label
        for label, id_field in [
            ("Entity", "id"), ("Event", "id"), ("Lead", "id"),
            ("Hypothesis", "id"), ("Location", "id"),
        ]:
            check = run_cypher(
                f"MATCH (n:{label} {{{id_field}: $id}}) RETURN count(n) AS c",
                {"id": aid},
            )
            if check and check[0].get("c", 0) > 0:
                try:
                    run_cypher(
                        f"MATCH (a:{label} {{{id_field}: $aid}}), "
                        "(e:Evidence {evidence_id: $eid}) "
                        "MERGE (a)-[r:CITES]->(e) "
                        "SET r.excerpt = $excerpt",
                        {"aid": aid, "eid": eid, "excerpt": (cit.get("excerpt") or "")[:200]},
                    )
                except Exception:
                    pass
                break

    # ── Hypotheses support/contradict leads ──────────────────────────────────
    for lead in case.get("leads", []):
        source_hypothesis_id = lead.get("source_hypothesis_id")
        if not source_hypothesis_id:
            continue
        try:
            _merge_edge(
                "Hypothesis",
                source_hypothesis_id,
                "SUPPORTS",
                "Lead",
                lead["id"],
                {"confidence": 1.0, "source": "lead.source_hypothesis_id"},
            )
        except Exception:
            pass

    for hyp in case.get("hypotheses", []):
        for lid in hyp.get("supporting_lead_ids", []):
            try:
                _merge_edge(
                    "Hypothesis", hyp["id"], "SUPPORTS", "Lead", lid,
                    {"confidence": float(hyp.get("confidence") or 0.5)},
                )
            except Exception:
                pass
        for lid in hyp.get("contradicting_lead_ids", []):
            try:
                _merge_edge(
                    "Hypothesis", hyp["id"], "CONTRADICTS", "Lead", lid,
                    {"confidence": float(hyp.get("confidence") or 0.5)},
                )
            except Exception:
                pass

    print("[seed] Neo4j seed complete.")


if __name__ == "__main__":
    _get_driver()  # force connection attempt
    seed()
