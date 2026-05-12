"""Neo4j driver wrapper with transparent in-memory fallback.

If NEO4J_URI is set and reachable, all queries go to Neo4j.
If not (e.g. Docker not started), falls back to the in-memory
dict graph already built in nightwave.tools._INDEX — zero
capability loss, just no Cypher.

Connection is a module-level singleton; callers never touch the driver.
"""

from __future__ import annotations

import os
from typing import Any

_driver = None
_mode: str = "memory"  # "neo4j" | "memory"
_initialized: bool = False


def _get_driver():
    global _driver, _mode, _initialized
    if _initialized:
        return _driver

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "nightwave")

    try:
        from neo4j import GraphDatabase

        d = GraphDatabase.driver(uri, auth=(user, password))
        d.verify_connectivity()
        _driver = d
        _mode = "neo4j"
        print(f"[graph_db] Connected to Neo4j at {uri}")
    except Exception as exc:
        print(f"[graph_db] Neo4j unavailable ({exc}); using in-memory fallback")
        _driver = None
        _mode = "memory"

    _initialized = True
    return _driver


def get_mode() -> str:
    _get_driver()
    return _mode


def run_cypher(cypher: str, params: dict | None = None) -> list[dict[str, Any]]:
    """Execute Cypher and return rows as plain dicts. Raises if not in Neo4j mode."""
    driver = _get_driver()
    if driver is None:
        raise RuntimeError("Neo4j not available — use in-memory graph path")
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [dict(record) for record in result]


def close() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None
