"""Orchestrator tests — no LLM calls.

Tests:
  - Critic retry logic: synthesizer re-invoked when critic fails and budget allows
  - Confidence capped when critic cannot validate after retry
  - Token budget guard skips retry when budget exceeded
  - Trace records orchestrator step with required fields
  - State passes correctly through parallel retrieval
"""

from __future__ import annotations

from unittest.mock import patch

from nightwave.multiagent.orchestrator import run_pipeline
from nightwave.multiagent.state import AgentState, DraftCitation


def _good_state(question: str = "test question") -> AgentState:
    state = AgentState(question=question)
    state.draft_response = "A valid answer with no hallucinated citations."
    state.draft_confidence = 0.8
    state.draft_citations = []
    return state


def _bad_state(question: str = "test question") -> AgentState:
    state = AgentState(question=question)
    state.draft_response = "Answer with hallucinated ID."
    state.draft_confidence = 0.9
    state.draft_citations = [
        DraftCitation(
            evidence_id="00000000-0000-0000-0000-000000000000",
            source_path="",
            locator={},
            excerpt="fake",
            confidence=0.9,
        )
    ]
    return state


def _critic_pass(state: AgentState) -> AgentState:
    state.critic_passed = True
    state.critic_feedback = ""
    return state


def _critic_fail(state: AgentState) -> AgentState:
    state.critic_passed = False
    state.critic_feedback = "Hallucinated IDs: 00000000"
    return state


# ── Retry logic ───────────────────────────────────────────────────────────────


def test_orchestrator_retries_synthesizer_on_critic_fail() -> None:
    """When critic fails, synthesizer should be called a second time."""
    call_count = {"n": 0}

    def mock_synth(state: AgentState) -> AgentState:
        call_count["n"] += 1
        state.draft_response = "answer"
        state.draft_confidence = 0.8
        state.draft_citations = []
        return state

    def mock_critic_first_fail_then_pass(state: AgentState) -> AgentState:
        if call_count["n"] == 1:
            return _critic_fail(state)
        return _critic_pass(state)

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch(
            "nightwave.multiagent.orchestrator.run_critic",
            side_effect=mock_critic_first_fail_then_pass,
        ),
    ):
        run_pipeline("What happened?", "test-q")

    assert call_count["n"] == 2, f"Synthesizer called {call_count['n']} times, expected 2"


def test_orchestrator_caps_confidence_when_critic_never_passes() -> None:
    """If critic fails all retries, confidence should be capped at ≤0.4."""

    def mock_synth(state: AgentState) -> AgentState:
        state.draft_response = "answer"
        state.draft_confidence = 0.95
        state.draft_citations = []
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_fail),
    ):
        answer = run_pipeline("What happened?", "test-q")

    assert answer.confidence <= 0.4, f"Expected confidence ≤0.4, got {answer.confidence}"


def test_orchestrator_uses_deterministic_fallback_for_claim_coverage_failure() -> None:
    fallback_called = {"value": False}

    def mock_synth(state: AgentState) -> AgentState:
        state.draft_response = "verbose unsupported answer"
        state.draft_confidence = 0.8
        state.draft_citations = []
        return state

    def mock_fallback(state: AgentState) -> AgentState:
        fallback_called["value"] = True
        state.draft_response = "grounded fallback answer"
        state.draft_confidence = 0.62
        return state

    def mock_critic(state: AgentState) -> AgentState:
        if fallback_called["value"]:
            state.critic_passed = True
            state.critic_feedback = ""
        else:
            state.critic_passed = False
            state.critic_feedback = "CRITIC HARD FAIL:\nCLAIM_COVERAGE too low"
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch(
            "nightwave.multiagent.orchestrator.run_deterministic_synthesizer",
            side_effect=mock_fallback,
        ),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=mock_critic),
    ):
        answer = run_pipeline("What happened?", "test-q")

    assert fallback_called["value"] is True
    assert answer.response == "grounded fallback answer"
    assert answer.confidence == 0.62


def test_orchestrator_no_retry_when_critic_passes_first() -> None:
    """When critic passes on first attempt, synthesizer is called exactly once."""
    call_count = {"n": 0}

    def mock_synth(state: AgentState) -> AgentState:
        call_count["n"] += 1
        state.draft_response = "good answer"
        state.draft_confidence = 0.8
        state.draft_citations = []
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_pass),
    ):
        run_pipeline("What happened?", "test-q")

    assert call_count["n"] == 1


# ── Token budget guard ────────────────────────────────────────────────────────


def test_orchestrator_skips_retry_when_budget_exceeded() -> None:
    """When token usage exceeds MAX_TOKENS, should not retry even if critic fails."""
    call_count = {"n": 0}

    def mock_synth(state: AgentState) -> AgentState:
        call_count["n"] += 1
        state.draft_response = "answer"
        state.draft_confidence = 0.8
        state.draft_citations = []
        state.token_usage["synthesizer"] = 999999  # exceed budget
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_fail),
    ):
        run_pipeline("test question", "test-q")

    assert call_count["n"] == 1, "Should not retry when token budget exceeded"


# ── Trace and Answer shape ────────────────────────────────────────────────────


def test_orchestrator_trace_contains_orchestrator_step() -> None:
    def mock_synth(state: AgentState) -> AgentState:
        state.draft_response = "answer"
        state.draft_confidence = 0.7
        state.draft_citations = []
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_pass),
    ):
        answer = run_pipeline("What happened?", "test-q")

    steps = [s.kind for s in answer.trace]
    assert "orchestrator" in steps


def test_orchestrator_answer_has_required_fields() -> None:
    def mock_synth(state: AgentState) -> AgentState:
        state.draft_response = "The answer is X."
        state.draft_confidence = 0.75
        state.draft_citations = []
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_pass),
    ):
        answer = run_pipeline("What happened?", "test-q")

    assert answer.question == "What happened?"
    assert answer.response == "The answer is X."
    assert 0.0 <= answer.confidence <= 1.0
    assert isinstance(answer.citations, list)
    assert isinstance(answer.trace, list)


def test_orchestrator_no_critic_warning_in_user_response() -> None:
    """Critic feedback must never leak into user-facing response."""

    def mock_synth(state: AgentState) -> AgentState:
        state.draft_response = "clean answer"
        state.draft_confidence = 0.7
        state.draft_citations = []
        return state

    def mock_critic_with_internal_feedback(state: AgentState) -> AgentState:
        state.critic_passed = False
        state.critic_feedback = "INTERNAL: citation ungrounded at locator line 42"
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth),
        patch(
            "nightwave.multiagent.orchestrator.run_critic",
            side_effect=mock_critic_with_internal_feedback,
        ),
    ):
        answer = run_pipeline("What happened?", "test-q")

    assert "INTERNAL" not in answer.response
    assert "citation ungrounded" not in answer.response
    assert "Critic warning" not in answer.response
