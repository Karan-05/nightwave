"""Deterministic graph_agent tests — no LLM calls, no Neo4j required.

Tests cover:
  1.  Classification: Q2 without platform names ("same person" / "two")
  2.  Classification: retrieval priority over multi_hop (both signals present)
  3.  Classification: "what timeline" has no planning keywords → "general"
  4.  Memory context: entity/connection question includes relationships + entities
  5.  Memory context: event question includes events
  6.  Memory context: fallback default returns leads + hypotheses
  7.  Memory context: result cap ≤ 30
  8.  run_graph_agent: state shape (graph_context is list, ≤30, trace has entry)
  9.  run_graph_agent: planning question populates at least one hypothesis
  10. run_graph_agent: empty question does not crash
"""

from __future__ import annotations

from nightwave.multiagent.state import AgentState
from nightwave.multiagent.subagents.graph_agent import (
    _classify_question,
    _memory_graph_context,
    run_graph_agent,
)

# ── Helper ────────────────────────────────────────────────────────────────────


def _make_state(question: str) -> AgentState:
    return AgentState(question=question)


# ── 1. Classification: Q2 without platform names ──────────────────────────────


def test_classify_q2_same_person_no_platform_names() -> None:
    """'same person' alone should trigger multi_hop classification."""
    q = "Did the same person contact Madison through two different accounts?"
    assert _classify_question(q) == "multi_hop"


def test_classify_q2_keyword_two_triggers_multi_hop() -> None:
    """'two' (without 'same person') should classify as multi_hop."""
    q = "Could there be two distinct individuals involved in this case?"
    assert _classify_question(q) == "multi_hop"


# ── 2. Classification: retrieval takes priority over multi_hop ────────────────


def test_classify_retrieval_beats_multi_hop_on_overlap() -> None:
    """A question with both 'wearing' (retrieval) and 'two' (multi_hop) should
    return 'retrieval' because retrieval is checked first."""
    q = "What was the suspect wearing when communicating across two platforms?"
    assert _classify_question(q) == "retrieval"


def test_classify_retrieval_on_wearing_alone() -> None:
    """'wearing' alone is enough to trigger retrieval."""
    q = "What was the suspect wearing when they sent the Snapchat message?"
    assert _classify_question(q) == "retrieval"


def test_classify_last_seen_is_retrieval() -> None:
    """'last seen' triggers retrieval regardless of other words."""
    q = "Where was Madison last seen?"
    assert _classify_question(q) == "retrieval"


# ── 3. Classification: 'what timeline' with no action keywords → "general" ───


def test_classify_timeline_only_is_general() -> None:
    """'what timeline' with no retrieval/multi_hop/planning keywords → 'general'.

    Note: 'investigat' is a planning keyword (substring-matches 'investigators'),
    so the question must avoid all planning keywords to land in 'general'.
    Keywords to avoid: next, action, priorit, investigat, rank, step.
    Also avoid retrieval: wearing, last seen, clothing, description, physical.
    Also avoid multi_hop: same person, separate, groomed, two, both platform, both app.
    """
    q = "What is the overall timeline of this case?"
    # 'timeline' is not in retrieval, multi_hop, or planning keyword lists
    result = _classify_question(q)
    assert result == "general"


def test_classify_empty_is_general() -> None:
    """Empty question falls through to the default 'general' bucket."""
    assert _classify_question("") == "general"


def test_classify_general_generic_question() -> None:
    """A question with no matching keywords returns 'general'."""
    q = "What is this case about?"
    assert _classify_question(q) == "general"


# ── 4. Memory context: entity question → relationships + entities ─────────────


def test_memory_graph_context_entity_question_includes_relationships() -> None:
    """'Who is connected to Madison?' triggers the 'connected'/'who' routing."""
    results = _memory_graph_context("Who is connected to Madison?")
    artifact_types = {r.get("artifact_type") for r in results}
    assert "relationship" in artifact_types, (
        "Expected 'relationship' items when question contains 'connected'/'who'"
    )


def test_memory_graph_context_entity_question_includes_entities() -> None:
    """'Who is connected to Madison?' should also return entity items."""
    results = _memory_graph_context("Who is connected to Madison?")
    artifact_types = {r.get("artifact_type") for r in results}
    assert "entity" in artifact_types, (
        "Expected 'entity' items when question contains 'connected'/'who'"
    )


# ── 5. Memory context: event question → events ────────────────────────────────


def test_memory_graph_context_event_question_includes_events() -> None:
    """'When did the incident happen?' triggers the event/timeline routing."""
    results = _memory_graph_context("When did the incident happen?")
    artifact_types = {r.get("artifact_type") for r in results}
    assert "event" in artifact_types, (
        "Expected 'event' items when question contains 'when'/'happen'"
    )


def test_memory_graph_context_timeline_keyword_includes_events() -> None:
    """Explicit 'timeline' keyword should also populate events."""
    results = _memory_graph_context("Show me the timeline of events.")
    artifact_types = {r.get("artifact_type") for r in results}
    assert "event" in artifact_types


# ── 6. Memory context: fallback → leads + hypotheses ─────────────────────────


def test_memory_graph_context_fallback_default_returns_leads_and_hypotheses() -> None:
    """A garbled question with no routing keywords hits the default branch."""
    results = _memory_graph_context("xyz 12345 lorem ipsum")
    artifact_types = {r.get("artifact_type") for r in results}
    assert artifact_types, "Default branch must return something"
    # Default branch only returns leads and hypotheses
    assert artifact_types <= {"lead", "hypothesis"}, (
        f"Default branch should only return leads/hypotheses, got: {artifact_types}"
    )


def test_memory_graph_context_fallback_has_leads() -> None:
    """Default branch includes at least one lead (the case has 6)."""
    results = _memory_graph_context("xyz 12345 lorem ipsum")
    leads = [r for r in results if r.get("artifact_type") == "lead"]
    assert len(leads) >= 1


def test_memory_graph_context_fallback_has_hypotheses() -> None:
    """Default branch includes at least one hypothesis (the case has 4)."""
    results = _memory_graph_context("xyz 12345 lorem ipsum")
    hypotheses = [r for r in results if r.get("artifact_type") == "hypothesis"]
    assert len(hypotheses) >= 1


# ── 7. Memory context: result cap ≤ 30 ───────────────────────────────────────


def test_memory_graph_context_cap_is_applied() -> None:
    """Result must be ≤ 30 items regardless of how many the index contains.

    Using a question that hits all three routing branches to maximise output
    before the cap is applied.
    """
    # Triggers planning, entity, AND event blocks simultaneously
    q = "Who is the connected person involved in the timeline of next steps?"
    results = _memory_graph_context(q)
    assert len(results) <= 30, f"Expected ≤30 results, got {len(results)}"


def test_memory_graph_context_fallback_cap_is_applied() -> None:
    """Cap applies to the default branch too."""
    results = _memory_graph_context("xyz 12345 lorem ipsum")
    assert len(results) <= 30


# ── 8. run_graph_agent: state shape ──────────────────────────────────────────


def test_run_graph_agent_returns_list() -> None:
    """graph_context must be a list after run_graph_agent."""
    state = _make_state("Who is connected to Madison?")
    result = run_graph_agent(state)
    assert isinstance(result.graph_context, list)


def test_run_graph_agent_context_cap() -> None:
    """graph_context must not exceed 30 items."""
    state = _make_state("Who is the connected person involved in the timeline of next steps?")
    result = run_graph_agent(state)
    assert len(result.graph_context) <= 30


def test_run_graph_agent_trace_has_graph_agent_entry() -> None:
    """After run_graph_agent, state.trace must contain an entry with step='graph_agent'."""
    state = _make_state("Who is connected to Madison?")
    result = run_graph_agent(state)
    assert any(
        e.get("step") == "graph_agent" or e.get("kind") == "graph_agent" for e in result.trace
    ), f"No graph_agent entry in trace: {result.trace}"


def test_run_graph_agent_trace_entry_has_question_class() -> None:
    """The trace entry should record question_class."""
    state = _make_state("Who is connected to Madison?")
    result = run_graph_agent(state)
    graph_entries = [
        e for e in result.trace if e.get("step") == "graph_agent" or e.get("kind") == "graph_agent"
    ]
    assert graph_entries, "No graph_agent trace entry found"
    assert "question_class" in graph_entries[-1], (
        f"question_class missing from trace entry: {graph_entries[-1]}"
    )


# ── 9. run_graph_agent: planning question → at least one hypothesis ───────────


def test_run_graph_agent_planning_includes_hypothesis() -> None:
    """A planning question should produce at least one hypothesis in graph_context."""
    state = _make_state("What are the next steps in the investigation?")
    result = run_graph_agent(state)
    hypothesis_items = [
        item for item in result.graph_context if item.get("artifact_type") == "hypothesis"
    ]
    assert len(hypothesis_items) >= 1, (
        "Planning question should include at least one hypothesis in graph_context"
    )


def test_run_graph_agent_planning_includes_lead() -> None:
    """A planning question should also include at least one lead."""
    state = _make_state("What are the next steps in the investigation?")
    result = run_graph_agent(state)
    lead_items = [item for item in result.graph_context if item.get("artifact_type") == "lead"]
    assert len(lead_items) >= 1, (
        "Planning question should include at least one lead in graph_context"
    )


# ── 10. run_graph_agent: empty question does not crash ────────────────────────


def test_run_graph_agent_empty_question_does_not_crash() -> None:
    """run_graph_agent with an empty question must not raise any exception."""
    state = _make_state("")
    result = run_graph_agent(state)
    # Must complete without exception
    assert isinstance(result.graph_context, list)
    assert len(result.graph_context) <= 30


def test_run_graph_agent_empty_question_has_trace() -> None:
    """Empty question should still add a trace entry."""
    state = _make_state("")
    result = run_graph_agent(state)
    assert any(
        e.get("step") == "graph_agent" or e.get("kind") == "graph_agent" for e in result.trace
    ), f"Expected graph_agent trace entry even for empty question, got: {result.trace}"


def test_run_graph_agent_respects_case_id_isolation() -> None:
    state = AgentState(question="Who is connected to Madison?", case_id="wrong-case")

    result = run_graph_agent(state)

    assert result.graph_context == []
    assert result.trace[-1]["mode"] == "case_mismatch"
    assert result.trace[-1]["case_id"] == "wrong-case"


# ── Bonus regression: multi_hop classifier correctly uses tuple, not set ──────


def test_classify_multi_hop_all_keywords() -> None:
    """Verify every multi_hop keyword individually triggers the classification."""
    multi_hop_keywords = [
        "same person",
        "separate",
        "groomed",
        "two",
        "both platform",
        "both app",
    ]
    for kw in multi_hop_keywords:
        result = _classify_question(f"question about {kw} something")
        assert result == "multi_hop", f"Expected 'multi_hop' for keyword '{kw}', got '{result}'"


def test_classify_planning_keywords() -> None:
    """Verify planning keywords each individually trigger 'planning'."""
    planning_keywords = ["next", "action", "priorit", "investigat", "rank", "step"]
    for kw in planning_keywords:
        # These must not overlap with retrieval or multi_hop keywords
        result = _classify_question(f"should we {kw} this case")
        assert result == "planning", f"Expected 'planning' for keyword '{kw}', got '{result}'"
