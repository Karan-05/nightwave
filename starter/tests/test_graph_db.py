"""Graph DB connection policy tests."""

from __future__ import annotations

import pytest

from nightwave.multiagent import graph_db


def test_requires_neo4j_env_flag(monkeypatch) -> None:
    monkeypatch.setenv("NIGHTWAVE_REQUIRE_NEO4J", "1")
    assert graph_db._requires_neo4j() is True

    monkeypatch.setenv("NIGHTWAVE_REQUIRE_NEO4J", "false")
    assert graph_db._requires_neo4j() is False


def test_required_neo4j_raises_instead_of_falling_back(monkeypatch) -> None:
    monkeypatch.setenv("NIGHTWAVE_REQUIRE_NEO4J", "1")
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:1")
    monkeypatch.setattr(graph_db, "_initialized", False)
    monkeypatch.setattr(graph_db, "_driver", None)
    monkeypatch.setattr(graph_db, "_mode", "memory")

    with pytest.raises(RuntimeError, match="Neo4j is required but unavailable"):
        graph_db._get_driver()

    monkeypatch.setattr(graph_db, "_initialized", False)
    monkeypatch.setattr(graph_db, "_driver", None)
    monkeypatch.setattr(graph_db, "_mode", "memory")
