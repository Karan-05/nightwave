"""Additional orchestrator/state edge-case tests.

All tests are deterministic — no LLM calls. All external IO is mocked via
unittest.mock.patch.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nightwave.multiagent.state import AgentState, DraftCitation, RetrievedChunk
from nightwave.multiagent.orchestrator import run_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_retriever_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            evidence_id="aaaaaaaa-0000-0000-0000-000000000001",
            source_path="/evidence/doc1.pdf",
            locator={"page": 1},
            excerpt="some text",
            score=0.9,
            source="rrf",
        )
    ]


def _make_graph_items(count: int = 3) -> list[dict]:
    return [{"artifact_type": "hypothesis", "id": f"hyp-{i}", "statement": f"stmt {i}", "confidence": 0.5}
            for i in range(count)]


def _critic_pass(state: AgentState) -> AgentState:
    state.critic_passed = True
    state.critic_feedback = ""
    return state


def _critic_fail(state: AgentState) -> AgentState:
    state.critic_passed = False
    state.critic_feedback = "Validation failure."
    return state


def _synth_noop(state: AgentState) -> AgentState:
    """Minimal synthesizer that sets a valid draft response."""
    state.draft_response = "answer"
    state.draft_confidence = 0.7
    state.draft_citations = []
    return state


# ── Test 1: State isolation ────────────────────────────────────────────────────


def test_state_isolation_retrieved_chunks_and_graph_context_are_independent() -> None:
    """retrieved_chunks and graph_context must be written independently."""
    chunks = _make_retriever_chunks()
    graph_items = _make_graph_items(5)

    def mock_retriever(state: AgentState) -> AgentState:
        state.retrieved_chunks = chunks
        state.trace.append({"step": "retriever", "hits": len(chunks)})
        return state

    def mock_graph_agent(state: AgentState) -> AgentState:
        state.graph_context = graph_items
        state.trace.append({"step": "graph_agent", "context_items": len(graph_items)})
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=mock_retriever),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=mock_graph_agent),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=_synth_noop),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_pass),
    ):
        answer = run_pipeline("What happened?", "test-isolation")

    # Both fields must be present in the final answer trace steps — we verify via
    # confirming both step kinds appear (retrieved_chunks & graph_context wrote their traces)
    step_kinds = {s.kind for s in answer.trace}
    assert "retriever" in step_kinds, "retriever step missing from trace"
    assert "graph_agent" in step_kinds, "graph_agent step missing from trace"

    # Verify the orchestrator correctly merged both — we can check trace counts
    retriever_steps = [s for s in answer.trace if s.kind == "retriever"]
    graph_steps = [s for s in answer.trace if s.kind == "graph_agent"]
    assert len(retriever_steps) >= 1
    assert len(graph_steps) >= 1

    # Confirm hits/context_items were recorded separately
    assert retriever_steps[0].detail.get("hits") == len(chunks)
    assert graph_steps[0].detail.get("context_items") == len(graph_items)


# ── Test 2: Token budget accumulation ────────────────────────────────────────


def test_token_budget_accumulates_across_synthesizer_retries() -> None:
    """When synthesizer is called twice (retry path), tokens accumulate additively."""
    PER_CALL_TOKENS = 1000

    def mock_synth_accumulates(state: AgentState) -> AgentState:
        state.draft_response = "answer"
        state.draft_confidence = 0.7
        state.draft_citations = []
        # Simulate what the real synthesizer does: accumulate
        state.token_usage["synthesizer"] = state.token_usage.get("synthesizer", 0) + PER_CALL_TOKENS
        state.trace.append({"step": "synthesizer", "tokens": PER_CALL_TOKENS})
        return state

    call_count = {"n": 0}

    def mock_critic_fail_then_pass(state: AgentState) -> AgentState:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _critic_fail(state)
        return _critic_pass(state)

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth_accumulates),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=mock_critic_fail_then_pass),
    ):
        answer = run_pipeline("What happened?", "test-tokens")

    # Synthesizer must have been called at least twice
    synth_steps = [s for s in answer.trace if s.kind == "synthesizer"]
    assert len(synth_steps) >= 2, f"Expected >=2 synthesizer calls, got {len(synth_steps)}"

    # Total tokens must be at least 2x per-call amount
    orchestrator_step = next(s for s in answer.trace if s.kind == "orchestrator")
    total_synth_tokens = orchestrator_step.detail.get("token_usage", {}).get("synthesizer", 0)
    assert total_synth_tokens >= 2 * PER_CALL_TOKENS, (
        f"Expected synthesizer tokens >= {2 * PER_CALL_TOKENS}, got {total_synth_tokens}"
    )


# ── Test 3: Graph context cap is in graph_agent, not orchestrator ─────────────


def test_orchestrator_does_not_impose_additional_graph_context_cap() -> None:
    """Orchestrator passes through graph_context as-is (cap lives in graph_agent)."""
    LARGE_CONTEXT_SIZE = 50
    large_graph = _make_graph_items(LARGE_CONTEXT_SIZE)

    # Track what state.graph_context looks like when synthesizer sees it
    seen_graph_size = {"n": 0}

    def mock_graph_agent(state: AgentState) -> AgentState:
        # Simulate graph_agent setting a large context (it normally caps at 30,
        # but we bypass that here to test the orchestrator specifically)
        state.graph_context = large_graph
        state.trace.append({"step": "graph_agent", "context_items": len(large_graph)})
        return state

    def mock_synth_records(state: AgentState) -> AgentState:
        seen_graph_size["n"] = len(state.graph_context)
        state.draft_response = "answer"
        state.draft_confidence = 0.7
        state.draft_citations = []
        state.trace.append({"step": "synthesizer"})
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=mock_graph_agent),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth_records),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_pass),
    ):
        run_pipeline("What happened?", "test-cap")

    # The orchestrator must not have trimmed graph_context before passing to synthesizer
    assert seen_graph_size["n"] == LARGE_CONTEXT_SIZE, (
        f"Orchestrator should not cap graph_context: expected {LARGE_CONTEXT_SIZE}, "
        f"synthesizer saw {seen_graph_size['n']}"
    )


# ── Test 4: Confidence capping below draft_confidence floor ──────────────────


def test_confidence_capping_when_draft_confidence_already_low() -> None:
    """min(0.3, 0.4) == 0.3; result must not go below 0.0 or exceed 0.4."""
    LOW_CONFIDENCE = 0.3

    def mock_synth_low(state: AgentState) -> AgentState:
        state.draft_response = "low-conf answer"
        state.draft_confidence = LOW_CONFIDENCE
        state.draft_citations = []
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth_low),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_fail),
    ):
        answer = run_pipeline("What happened?", "test-low-conf")

    assert 0.0 <= answer.confidence <= 0.4, (
        f"Expected 0.0 <= confidence <= 0.4, got {answer.confidence}"
    )
    # Specifically: min(0.3, 0.4) = 0.3, so should be exactly 0.3
    assert answer.confidence == LOW_CONFIDENCE, (
        f"Expected capped confidence == {LOW_CONFIDENCE} (min(0.3,0.4)), got {answer.confidence}"
    )


# ── Test 5: Answer.citations comes from state.draft_citations after critic pass


def test_answer_citations_match_draft_citations_exactly_after_critic_pass() -> None:
    """When critic passes, Answer.citations must exactly mirror state.draft_citations."""
    expected_citations = [
        DraftCitation(
            evidence_id="aaaaaaaa-0000-0000-0000-000000000001",
            source_path="/evidence/doc1.pdf",
            locator={"page": 3},
            excerpt="Verbatim excerpt one",
            confidence=0.85,
        ),
        DraftCitation(
            evidence_id="bbbbbbbb-0000-0000-0000-000000000002",
            source_path="/evidence/doc2.pdf",
            locator={"page": 7},
            excerpt="Verbatim excerpt two",
            confidence=0.90,
        ),
    ]

    def mock_synth_with_citations(state: AgentState) -> AgentState:
        state.draft_response = "answer with citations"
        state.draft_confidence = 0.8
        state.draft_citations = expected_citations[:]  # copy
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=lambda s: s),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth_with_citations),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=_critic_pass),
    ):
        answer = run_pipeline("What happened?", "test-citations")

    assert len(answer.citations) == len(expected_citations), (
        f"Expected {len(expected_citations)} citations, got {len(answer.citations)}"
    )
    for i, (actual, expected) in enumerate(zip(answer.citations, expected_citations)):
        assert actual.evidence_id == expected.evidence_id, (
            f"Citation[{i}] evidence_id mismatch: {actual.evidence_id!r} != {expected.evidence_id!r}"
        )
        assert actual.excerpt == expected.excerpt, (
            f"Citation[{i}] excerpt mismatch: {actual.excerpt!r} != {expected.excerpt!r}"
        )
        assert actual.confidence == expected.confidence, (
            f"Citation[{i}] confidence mismatch: {actual.confidence} != {expected.confidence}"
        )
        assert actual.locator == expected.locator, (
            f"Citation[{i}] locator mismatch: {actual.locator} != {expected.locator}"
        )


# ── Test 6: Multiple trace entries after retry ────────────────────────────────


def test_trace_contains_at_least_4_distinct_step_kinds_after_retry() -> None:
    """After one retry, the trace must show steps from retriever, graph_agent,
    synthesizer (x2), critic (x2), and orchestrator — at least 4 distinct kinds."""
    call_count = {"n": 0}

    def mock_synth_with_trace(state: AgentState) -> AgentState:
        state.draft_response = "answer"
        state.draft_confidence = 0.7
        state.draft_citations = []
        state.trace.append({"step": "synthesizer"})
        return state

    def mock_retriever_with_trace(state: AgentState) -> AgentState:
        state.trace.append({"step": "retriever", "hits": 0})
        return state

    def mock_graph_with_trace(state: AgentState) -> AgentState:
        state.trace.append({"step": "graph_agent", "context_items": 0})
        return state

    def mock_critic_with_trace(state: AgentState) -> AgentState:
        call_count["n"] += 1
        state.trace.append({"step": "critic", "passed": call_count["n"] >= 2})
        if call_count["n"] == 1:
            state.critic_passed = False
            state.critic_feedback = "retry"
        else:
            state.critic_passed = True
            state.critic_feedback = ""
        return state

    with (
        patch("nightwave.multiagent.orchestrator.run_retriever", side_effect=mock_retriever_with_trace),
        patch("nightwave.multiagent.orchestrator.run_graph_agent", side_effect=mock_graph_with_trace),
        patch("nightwave.multiagent.orchestrator.run_synthesizer", side_effect=mock_synth_with_trace),
        patch("nightwave.multiagent.orchestrator.run_critic", side_effect=mock_critic_with_trace),
    ):
        answer = run_pipeline("What happened?", "test-trace-multi")

    step_kinds = {s.kind for s in answer.trace}
    assert len(step_kinds) >= 4, (
        f"Expected at least 4 distinct step kinds, got {len(step_kinds)}: {step_kinds}"
    )

    # Specifically verify the most important kinds are present
    for expected_kind in ("retriever", "graph_agent", "synthesizer", "orchestrator"):
        assert expected_kind in step_kinds, f"Step kind {expected_kind!r} missing from trace"

    # Verify synthesizer was called at least twice (retry happened)
    synth_steps = [s for s in answer.trace if s.kind == "synthesizer"]
    assert len(synth_steps) >= 2, f"Expected >=2 synthesizer steps, got {len(synth_steps)}"


# ── Test 7: AgentState initialization defaults ────────────────────────────────


def test_agent_state_initialization_defaults() -> None:
    """A freshly constructed AgentState must have the expected default values."""
    state = AgentState(question="test question")

    assert state.retrieved_chunks == [], (
        f"retrieved_chunks default should be [], got {state.retrieved_chunks!r}"
    )
    assert state.graph_context == [], (
        f"graph_context default should be [], got {state.graph_context!r}"
    )
    assert state.draft_citations == [], (
        f"draft_citations default should be [], got {state.draft_citations!r}"
    )
    # The dataclass definition sets critic_passed: bool = False
    assert state.critic_passed is False, (
        f"critic_passed default should be False, got {state.critic_passed!r}"
    )
    assert state.token_usage == {}, (
        f"token_usage default should be {{}}, got {state.token_usage!r}"
    )

    # Verify list defaults are independent instances (no shared-mutable default)
    state2 = AgentState(question="another question")
    state.retrieved_chunks.append(
        RetrievedChunk(
            evidence_id="x", source_path="", locator={}, excerpt="", score=0.0, source="rrf"
        )
    )
    assert state2.retrieved_chunks == [], "list defaults must not be shared between instances"
    state.graph_context.append({"x": 1})
    assert state2.graph_context == [], "graph_context defaults must not be shared between instances"
