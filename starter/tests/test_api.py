"""FastAPI surface tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from nightwave.api import app


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["case_id"]
    assert payload["graph_mode"] in {"neo4j", "memory"}


def test_answer_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_MODE", "deterministic")
    client = TestClient(app)

    response = client.post(
        "/v1/answer",
        json={
            "question": "What was Madison Fields wearing when last seen?",
            "question_id": "q1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["confidence"] >= 0.85
    assert payload["citations"]
    assert payload["graph_mode"] in {"neo4j", "memory"}


def test_answer_endpoint_rejects_blank_question() -> None:
    client = TestClient(app)

    response = client.post("/v1/answer", json={"question": "", "question_id": "empty"})

    assert response.status_code == 422


def test_eval_endpoint_rejects_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("NIGHTWAVE_ENABLE_EVAL_ENDPOINT", raising=False)
    client = TestClient(app)

    response = client.post("/v1/eval")

    assert response.status_code == 403


def test_eval_endpoint_returns_multiagent_report(monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_MODE", "deterministic")
    monkeypatch.setenv("NIGHTWAVE_ENABLE_EVAL_ENDPOINT", "1")
    client = TestClient(app)

    response = client.post("/v1/eval")

    assert response.status_code == 200
    payload = response.json()
    assert payload["architecture"] == "multi-agent"
    assert payload["passed"] is True
    assert payload["overall_score"] >= 0.95
