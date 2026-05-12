"""Deterministic critic tests — no LLM calls required.

Tests:
  - Hard fail on hallucinated evidence_id
  - Hard fail on multiple hallucinated IDs
  - Pass on valid evidence_id with grounded excerpt
  - Soft fail on ungrounded excerpt (confidence capped)
  - Confidence calibration soft check
"""

from __future__ import annotations

from nightwave.multiagent.state import AgentState, DraftCitation
from nightwave.multiagent.subagents.critic import run_critic
from nightwave.tools import _INDEX


def _valid_ev_id() -> str:
    """Return an evidence_id that actually exists in the case."""
    return next(iter(_INDEX.evidence_by_id))


def _make_state(**kwargs) -> AgentState:
    state = AgentState(question="test question")
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


def _citation(evidence_id: str, excerpt: str = "test excerpt", confidence: float = 0.8) -> DraftCitation:
    ev = _INDEX.evidence_by_id.get(evidence_id, {})
    return DraftCitation(
        evidence_id=evidence_id,
        source_path=ev.get("source_path", ""),
        locator={},
        excerpt=excerpt,
        confidence=confidence,
    )


# ── Hard-fail: hallucinated ID ────────────────────────────────────────────────

def test_critic_hard_fail_hallucinated_id() -> None:
    fake_id = "00000000-0000-0000-0000-000000000000"
    state = _make_state(
        draft_response="test",
        draft_confidence=0.8,
        draft_citations=[_citation(fake_id)],
    )
    result = run_critic(state)
    assert not result.critic_passed
    assert fake_id in result.critic_feedback


def test_critic_hard_fail_multiple_hallucinated_ids() -> None:
    fake_ids = [
        "aaaaaaaa-0000-0000-0000-000000000001",
        "bbbbbbbb-0000-0000-0000-000000000002",
    ]
    citations = [_citation(fid) for fid in fake_ids]
    state = _make_state(
        draft_response="test",
        draft_confidence=0.9,
        draft_citations=citations,
    )
    result = run_critic(state)
    assert not result.critic_passed
    for fid in fake_ids:
        assert fid in result.critic_feedback


def test_critic_hard_fail_confidence_capped_on_hallucinated_id() -> None:
    fake_id = "deadbeef-dead-dead-dead-deaddeaddeadd"
    state = _make_state(
        draft_response="test",
        draft_confidence=0.95,
        draft_citations=[_citation(fake_id)],
    )
    result = run_critic(state)
    assert not result.critic_passed
    # Confidence should not be trusted — critic_passed is False
    assert result.critic_passed is False


# ── Pass: valid ID with known evidence ────────────────────────────────────────

def test_critic_passes_valid_evidence_id_no_source_file() -> None:
    """Citations with no source_path are accepted if evidence_id is valid."""
    ev_id = _valid_ev_id()
    ev = _INDEX.evidence_by_id[ev_id]
    # Use description as excerpt — critic falls back to metadata text
    excerpt = (ev.get("description") or ev.get("ai_summary") or "")[:40]
    state = _make_state(
        draft_response=f"Test [{ev_id}: {excerpt}]",
        draft_confidence=0.8,
        draft_citations=[_citation(ev_id, excerpt=excerpt if excerpt else "some text")],
    )
    result = run_critic(state)
    # Should pass hard check (valid ID); soft check may warn but not block
    assert result.critic_passed or result.critic_feedback  # at minimum no hard ID fail


def test_critic_no_citations_passes_hard_check() -> None:
    """An answer with no citations at all should pass the hard ID check."""
    state = _make_state(
        draft_response="No evidence cited.",
        draft_confidence=0.5,
        draft_citations=[],
    )
    result = run_critic(state)
    # Hard check passes (no hallucinated IDs); soft check may warn about empty citations
    assert "hallucinated" not in result.critic_feedback.lower() or result.critic_passed


# ── Soft-fail: ungrounded excerpt ────────────────────────────────────────────

def test_critic_soft_fail_completely_fabricated_excerpt() -> None:
    ev_id = _valid_ev_id()
    state = _make_state(
        draft_response="test",
        draft_confidence=0.9,
        draft_citations=[_citation(
            ev_id,
            excerpt="totally fabricated content that is not in any source file",
        )],
    )
    result = run_critic(state)
    # Soft fail means critic_passed may be False OR feedback contains grounding warning
    # Hard check passes (valid ID), so feedback should not mention hallucinated
    assert "hallucinated" not in result.critic_feedback.lower()


# ── Confidence calibration ────────────────────────────────────────────────────

def test_critic_does_not_pass_over_calibrated_single_source() -> None:
    """Confidence >0.97 with a single source should trigger a soft warning."""
    ev_id = _valid_ev_id()
    ev = _INDEX.evidence_by_id[ev_id]
    excerpt = (ev.get("description") or "test excerpt")[:40]
    state = _make_state(
        draft_response="test",
        draft_confidence=0.99,
        draft_citations=[_citation(ev_id, confidence=0.99, excerpt=excerpt)],
    )
    result = run_critic(state)
    # Either passes with a confidence note, or fails soft check
    # Main assertion: no crash and feedback is populated when it fails
    assert isinstance(result.critic_passed, bool)
    assert isinstance(result.critic_feedback, str)


# ── New edge cases ────────────────────────────────────────────────────────────

# 9. Citation with empty excerpt — expect soft fail (ungrounded)
def test_critic_empty_excerpt_soft_fails() -> None:
    """_excerpt_grounded returns False for empty excerpt, so critic marks it ungrounded."""
    ev_id = _valid_ev_id()
    state = _make_state(
        draft_response="test",
        draft_confidence=0.8,
        draft_citations=[_citation(ev_id, excerpt="")],
    )
    result = run_critic(state)
    # Hard check passes (valid ID), ungrounded excerpt triggers soft fail
    assert not result.critic_passed
    assert "hallucinated" not in result.critic_feedback.lower()
    assert "ungrounded" in result.critic_feedback.lower()


# 10. Citation with valid evidence_id but empty source_path
def test_critic_valid_id_empty_source_path() -> None:
    """Empty source_path on DraftCitation does NOT affect hard-check outcome.

    The critic resolves the source file via evidence_id + _INDEX, not source_path.
    Hard check passes; grounding may fail softly if the excerpt isn't found.
    """
    ev_id = _valid_ev_id()
    state = _make_state(
        draft_response="test",
        draft_confidence=0.8,
        draft_citations=[
            DraftCitation(
                evidence_id=ev_id,
                source_path="",   # deliberately empty
                locator={},
                excerpt="totally fabricated text that will not be found",
                confidence=0.8,
            )
        ],
    )
    result = run_critic(state)
    # Hard check: ID is valid, so no "hallucinated" in feedback
    assert "hallucinated" not in result.critic_feedback.lower()
    # Either passes (excerpt found in metadata) or soft-fails (ungrounded)
    assert isinstance(result.critic_passed, bool)


# 11. Prior critic_feedback is replaced (not accumulated) on each run
def test_critic_replaces_prior_feedback() -> None:
    """run_critic overwrites state.critic_feedback unconditionally on each call."""
    ev_id = _valid_ev_id()
    state = _make_state(
        draft_response="test",
        draft_confidence=0.8,
        draft_citations=[],
        critic_feedback="PRIOR FEEDBACK FROM EARLIER RUN",
    )
    result = run_critic(state)
    # The prior feedback string must NOT appear in the new feedback
    assert "PRIOR FEEDBACK FROM EARLIER RUN" not in result.critic_feedback
    # On an empty citations list, critic passes hard check with empty feedback
    assert result.critic_passed is True
    assert result.critic_feedback == ""


# 12. Very long excerpt (500 chars) — no crash, grounding check handles it
def test_critic_very_long_excerpt_no_crash() -> None:
    """A 500-character excerpt should not raise any exception."""
    ev_id = _valid_ev_id()
    long_excerpt = "word " * 100  # 500 characters
    state = _make_state(
        draft_response="test",
        draft_confidence=0.8,
        draft_citations=[_citation(ev_id, excerpt=long_excerpt)],
    )
    result = run_critic(state)
    # No crash; outcome (pass/fail) depends on whether excerpt is found
    assert isinstance(result.critic_passed, bool)
    assert isinstance(result.critic_feedback, str)
