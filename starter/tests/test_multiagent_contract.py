from __future__ import annotations

import json
from pathlib import Path

from nightwave.eval import score_question
from nightwave.multiagent.run import answer_multi
from nightwave.tools import _INDEX

QUESTIONS = json.loads((Path(__file__).resolve().parents[2] / "questions.json").read_text())[
    "questions"
]


def _question(question_id: str) -> dict:
    return next(q for q in QUESTIONS if q["id"] == question_id)


def test_multiagent_answers_all_questions_with_valid_citations(monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_MODE", "deterministic")
    for question in QUESTIONS:
        answer = answer_multi(question["question"], question["id"])

        assert answer.response
        assert 0.0 <= answer.confidence <= 1.0
        assert answer.citations
        assert "Critic warning" not in answer.response
        assert all(c.evidence_id in _INDEX.evidence_by_id for c in answer.citations)

        result = score_question(answer, question)
        assert result["scores"]["overall"] >= 0.95


def test_q2_is_calibrated_and_references_both_platform_actors(monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_MODE", "deterministic")
    question = _question("q2")
    answer = answer_multi(question["question"], question["id"])

    assert 0.45 <= answer.confidence <= 0.70
    assert "ent-kyle-lawrence" in answer.response
    assert "ent-josh" in answer.response
    assert "Snapchat" in answer.response
    assert "Session" in answer.response
    assert {c.evidence_id for c in answer.citations} >= {
        "f572cb6a-e484-483c-9af8-d731c405071c",
        "b8281149-7efc-4726-ac7b-31172e5dc458",
    }


def test_q3_prioritizes_existing_critical_leads(monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_MODE", "deterministic")
    question = _question("q3")
    answer = answer_multi(question["question"], question["id"])

    assert "lead-session-forensic" in answer.response
    assert "lead-snapchat-subpoena" in answer.response
    assert "lead-intown-surveillance" in answer.response
    assert "hyp-josh-coconspirator" in answer.response
    assert "hyp-additional-victims" in answer.response
    assert 0.50 <= answer.confidence <= 0.80
