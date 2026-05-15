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


def _case_id() -> str:
    return str(load_case_json().get("case_id", ""))


def _merge_node(label: str, props: dict) -> None:
    key = "id" if "id" in props else "evidence_id"
    safe = {k: v for k, v in props.items() if isinstance(v, (str, int, float, bool)) or v is None}
    params = {"case_id": props["case_id"], "key_val": props[key], "props": safe}
    cypher = f"MERGE (n:{label} {{case_id: $case_id, {key}: $key_val}}) SET n += $props"
    try:
        run_cypher(cypher, params)
    except Exception:
        # Existing local databases may already have single-property uniqueness
        # constraints from earlier seeds. Preserve idempotence and stamp case_id.
        fallback = f"MERGE (n:{label} {{{key}: $key_val}}) SET n += $props"
        run_cypher(fallback, params)


def _merge_edge(
    from_label: str,
    from_id: str,
    rel: str,
    to_label: str,
    to_id: str,
    props: dict | None = None,
) -> None:
    case_id = _case_id()
    cypher = (
        f"MATCH (a:{from_label} {{case_id: $case_id, id: $from_id}}), "
        f"(b:{to_label} {{case_id: $case_id, id: $to_id}}) "
        f"MERGE (a)-[r:{rel}]->(b) "
        "SET r += $props"
    )
    run_cypher(
        cypher,
        {
            "case_id": case_id,
            "from_id": from_id,
            "to_id": to_id,
            "props": {"case_id": case_id, **(props or {})},
        },
    )


def validate_seed(case: dict | None = None) -> dict:
    """Assert Neo4j contains this case's core nodes and relationship evidence."""
    case = case or load_case_json()
    case_id = case["case_id"]
    expected = {
        "Entity": len(case.get("entities", [])),
        "Event": len(case.get("events", [])),
        "Lead": len(case.get("leads", [])),
        "Hypothesis": len(case.get("hypotheses", [])),
        "Location": len(case.get("locations", [])),
        "Evidence": len(case.get("evidence", [])),
    }
    actual: dict[str, int] = {}
    for label in expected:
        rows = run_cypher(
            f"MATCH (n:{label} {{case_id: $case_id}}) RETURN count(n) AS count",
            {"case_id": case_id},
        )
        actual[label] = int(rows[0]["count"]) if rows else 0

    missing = {
        label: {"expected": count, "actual": actual[label]}
        for label, count in expected.items()
        if actual[label] < count
    }
    relationship_rows = run_cypher(
        """
        MATCH (:Entity {case_id: $case_id})-[r]-(:Entity {case_id: $case_id})
        WHERE r.id IS NOT NULL
        RETURN count(DISTINCT r.id) AS count
        """,
        {"case_id": case_id},
    )
    relationship_count = int(relationship_rows[0]["count"]) if relationship_rows else 0
    if relationship_count < len(case.get("relationships", [])):
        missing["Relationship"] = {
            "expected": len(case.get("relationships", [])),
            "actual": relationship_count,
        }

    if missing:
        raise RuntimeError(f"Neo4j seed validation failed for case {case_id}: {missing}")

    return {"case_id": case_id, "nodes": actual, "relationships": relationship_count}


def seed() -> None:
    if get_mode() != "neo4j":
        raise RuntimeError("Neo4j seed requires graph mode 'neo4j'; got memory fallback")

    case = load_case_json()
    case_id = case["case_id"]

    # ── Constraints (idempotent) ──────────────────────────────────────────────
    for label, key in [
        ("Entity", "id"),
        ("Event", "id"),
        ("Lead", "id"),
        ("Hypothesis", "id"),
        ("Location", "id"),
        ("Evidence", "evidence_id"),
    ]:
        try:
            run_cypher(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                f"REQUIRE (n.case_id, n.{key}) IS UNIQUE"
            )
        except Exception:
            pass  # older Neo4j versions

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for e in case.get("entities", []):
        _merge_node(
            "Entity",
            {
                "id": e["id"],
                "case_id": case_id,
                "name": e.get("name", ""),
                "entity_type": e.get("entity_type", ""),
                "status": e.get("status", ""),
                "age": e.get("age") or 0,
                "summary": (e.get("summary") or "")[:500],
            },
        )

    for ev in case.get("events", []):
        _merge_node(
            "Event",
            {
                "id": ev["id"],
                "case_id": case_id,
                "event_type": ev.get("event_type", ""),
                "description": (ev.get("description") or "")[:500],
                "occurred_at": str(ev.get("occurred_at") or ""),
                "confidence": float(ev.get("confidence") or 0.5),
            },
        )

    for loc in case.get("locations", []):
        _merge_node(
            "Location",
            {
                "id": loc["id"],
                "case_id": case_id,
                "name": loc.get("name", ""),
                "location_type": loc.get("location_type", ""),
                "address": loc.get("address") or "",
            },
        )

    for hyp in case.get("hypotheses", []):
        _merge_node(
            "Hypothesis",
            {
                "id": hyp["id"],
                "case_id": case_id,
                "title": hyp.get("title") or hyp.get("statement", ""),
                "description": (hyp.get("description") or hyp.get("reasoning") or "")[:500],
                "confidence": float(hyp.get("confidence") or 0.5),
                "status": hyp.get("status", ""),
            },
        )

    for lead in case.get("leads", []):
        _merge_node(
            "Lead",
            {
                "id": lead["id"],
                "case_id": case_id,
                "title": lead.get("title", ""),
                "description": (lead.get("description") or "")[:500],
                "priority": lead.get("priority", ""),
                "status": lead.get("status", ""),
            },
        )

    for ev in case.get("evidence", []):
        _merge_node(
            "Evidence",
            {
                "evidence_id": ev["evidence_id"],
                "id": ev["evidence_id"],  # duplicate for edge matching
                "case_id": case_id,
                "evidence_name": ev.get("evidence_name", ""),
                "evidence_type": ev.get("evidence_type", ""),
                "confidence": float(ev.get("confidence") or 0.5),
                "priority": ev.get("priority", ""),
            },
        )

    # ── Entity→Entity relationships (KNOWS / typed) ───────────────────────────
    citations_by_artifact: dict[str, list[str]] = {}
    for cit in case.get("citations", []):
        citations_by_artifact.setdefault(cit.get("artifact_id", ""), []).append(cit["evidence_id"])

    for rel in case.get("relationships", []):
        rel_type = rel.get("relationship_type", "KNOWS").upper().replace(" ", "_").replace("-", "_")
        try:
            _merge_edge(
                "Entity",
                rel["source_entity_id"],
                rel_type,
                "Entity",
                rel["target_entity_id"],
                {
                    "id": rel.get("id", ""),
                    "case_id": case_id,
                    "confidence": float(rel.get("confidence") or 0.5),
                    "description": rel.get("description", "")[:200],
                    "platform": rel.get("properties", {}).get("platform", ""),
                    "citation_evidence_ids": citations_by_artifact.get(rel.get("id", ""), []),
                },
            )
        except Exception as exc:
            raise RuntimeError(f"Failed seeding relationship {rel.get('id')}: {exc}") from exc

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
            ("Entity", "id"),
            ("Event", "id"),
            ("Lead", "id"),
            ("Hypothesis", "id"),
            ("Location", "id"),
        ]:
            check = run_cypher(
                f"MATCH (n:{label} {{case_id: $case_id, {id_field}: $id}}) RETURN count(n) AS c",
                {"case_id": case_id, "id": aid},
            )
            if check and check[0].get("c", 0) > 0:
                try:
                    run_cypher(
                        f"MATCH (a:{label} {{case_id: $case_id, {id_field}: $aid}}), "
                        "(e:Evidence {case_id: $case_id, evidence_id: $eid}) "
                        "MERGE (a)-[r:CITES]->(e) "
                        "SET r += $props",
                        {
                            "case_id": case_id,
                            "aid": aid,
                            "eid": eid,
                            "props": {
                                "case_id": case_id,
                                "id": cit.get("id", ""),
                                "excerpt": (cit.get("excerpt") or "")[:200],
                            },
                        },
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed seeding citation {cit.get('id')}: {exc}") from exc
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
                    "Hypothesis",
                    hyp["id"],
                    "SUPPORTS",
                    "Lead",
                    lid,
                    {"confidence": float(hyp.get("confidence") or 0.5)},
                )
            except Exception:
                pass
        for lid in hyp.get("contradicting_lead_ids", []):
            try:
                _merge_edge(
                    "Hypothesis",
                    hyp["id"],
                    "CONTRADICTS",
                    "Lead",
                    lid,
                    {"confidence": float(hyp.get("confidence") or 0.5)},
                )
            except Exception:
                pass

    validation = validate_seed(case)
    print(f"[seed] Neo4j seed complete. validation={validation}")


if __name__ == "__main__":
    _get_driver()  # force connection attempt
    seed()
