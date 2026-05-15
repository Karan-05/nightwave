"""Deterministic retriever tests — no LLM calls.

Tests:
  - Source diversity: no single evidence_id takes >2 slots before others appear
  - Q1 clothing query surfaces the Ohio AG alert (163faaac) in top 5
  - Q2 query surfaces both must-cite IDs (f572cb6a, b8281149) in top k
  - Secondary query generation produces relevant, non-case-specific queries
  - Proper noun extraction from question text
  - Platform/app extraction from question text
"""

from __future__ import annotations

import re

from nightwave.multiagent.state import AgentState
from nightwave.multiagent.subagents.retriever import (
    _classify,
    _extract_apps_platforms,
    _extract_proper_nouns,
    _generate_secondary_queries,
    run_retriever,
)
from nightwave.multiagent.vector_store import hybrid_search

Q1 = (
    "What was Madison Fields wearing when she was last seen, and where in the case "
    "files does this information come from?"
)
Q2 = (
    "Did the same person groom Madison on Snapchat and on the Session app, or are "
    "these two separate offenders? Cite the specific evidence that supports your "
    "answer and explain how confident you are."
)
Q3 = (
    "Madison's phone went dead the evening of February 13. As of the case state on "
    "March 1, list the top three next investigative actions ranked by impact."
)

OHIO_ALERT_ID = "163faaac-d742-4160-aa62-070f9ecb96cb"
DOJ_PRESS_ID = "f572cb6a-e484-483c-9af8-d731c405071c"
JOSH_NEWS_ID = "b8281149-7efc-4726-ac7b-31172e5dc458"


# ── Source diversity cap ──────────────────────────────────────────────────────


def test_retriever_source_diversity_no_source_exceeds_cap() -> None:
    """After run_retriever, no single evidence_id should hold >2 of the first 6 slots."""
    state = AgentState(question=Q2)
    result = run_retriever(state, k=10)
    chunks = result.retrieved_chunks[:6]
    from collections import Counter

    counts = Counter(c.evidence_id for c in chunks)
    for ev_id, count in counts.items():
        assert count <= 2, f"{ev_id[:8]} appears {count} times in first 6 slots (cap=2)"


def test_retriever_source_diversity_q1() -> None:
    """For Q1, check that diversity holds in the first 4 slots (initial pass guarantee).
    Q1 has fewer relevant unique sources so the deferred list may repeat at slot 5+."""
    state = AgentState(question=Q1)
    result = run_retriever(state, k=10)
    chunks = result.retrieved_chunks[:4]
    from collections import Counter

    counts = Counter(c.evidence_id for c in chunks)
    for _ev_id, count in counts.items():
        assert count <= 2, "Source diversity violated in first 4 slots for Q1"


# ── Must-cite coverage ────────────────────────────────────────────────────────


def test_retriever_q1_surfaces_ohio_alert() -> None:
    """Ohio AG alert (163faaac) must appear in top 5 retrieved chunks for Q1."""
    state = AgentState(question=Q1)
    result = run_retriever(state, k=10)
    top5_ids = {c.evidence_id for c in result.retrieved_chunks[:5]}
    top5_short_ids = [c.evidence_id[:8] for c in result.retrieved_chunks[:5]]
    assert OHIO_ALERT_ID in top5_ids, f"Ohio AG alert not in top 5. Got: {top5_short_ids}"


def test_retriever_q2_surfaces_both_must_cite_ids() -> None:
    """Both must-cite IDs for Q2 must appear somewhere in retrieved chunks."""
    state = AgentState(question=Q2)
    result = run_retriever(state, k=10)
    retrieved_ids = {c.evidence_id for c in result.retrieved_chunks}
    assert DOJ_PRESS_ID in retrieved_ids, "DOJ press release (f572cb6a) not retrieved for Q2"
    assert JOSH_NEWS_ID in retrieved_ids, (
        "Josh/Session news article (b8281149) not retrieved for Q2"
    )


def test_retriever_q2_doj_press_in_top8() -> None:
    """DOJ press release must be in top 8 (diverse) slots, not buried."""
    state = AgentState(question=Q2)
    result = run_retriever(state, k=10)
    top8_ids = [c.evidence_id for c in result.retrieved_chunks[:8]]
    assert DOJ_PRESS_ID in top8_ids, (
        f"DOJ press release not in top 8. Got: {[e[:8] for e in top8_ids]}"
    )


# ── Question classification ───────────────────────────────────────────────────


def test_classify_q1_is_retrieval() -> None:
    assert _classify(Q1) == "retrieval"


def test_classify_q2_is_multi_hop() -> None:
    assert _classify(Q2) == "multi_hop"


def test_classify_q3_is_planning() -> None:
    assert _classify(Q3) == "planning"


def test_classify_generic_fallback() -> None:
    assert _classify("What is the suspect's motive?") == "general"


# ── Proper noun extraction ────────────────────────────────────────────────────


def test_extract_proper_nouns_from_q2() -> None:
    nouns = _extract_proper_nouns(Q2)
    assert "Madison" in nouns
    # Should include platform names that start with capital letters
    assert len(nouns) >= 1


def test_extract_proper_nouns_filters_stopwords() -> None:
    nouns = _extract_proper_nouns("What was the suspect wearing?")
    # Stop words like "What" should be filtered
    assert "What" not in nouns


def test_extract_proper_nouns_deduplicates() -> None:
    text = "Madison Fields met Madison Fields near Madison."
    nouns = _extract_proper_nouns(text)
    assert nouns.count("Madison") <= 1 and nouns.count("Madison Fields") <= 1


# ── Platform extraction ───────────────────────────────────────────────────────


def test_extract_apps_platforms_session_app() -> None:
    platforms = _extract_apps_platforms("messaging via the Session app")
    assert "Session" in platforms


def test_extract_apps_platforms_quoted() -> None:
    platforms = _extract_apps_platforms('contact on "Snapchat" and other apps')
    assert "Snapchat" in platforms


def test_extract_apps_platforms_no_false_positives() -> None:
    platforms = _extract_apps_platforms("the suspect walked down the street")
    assert len(platforms) == 0


# ── Secondary query generation ────────────────────────────────────────────────


def test_secondary_queries_multi_hop_uses_question_text() -> None:
    qs = _generate_secondary_queries(Q2, "multi_hop")
    # All queries should be >= 3 words
    for q in qs:
        assert len(q.split()) >= 3
    # Should be non-empty
    assert len(qs) >= 1


def test_secondary_queries_planning_extracts_action_words() -> None:
    q = "Should we conduct a forensic extraction of the phone and issue a surveillance warrant?"
    qs = _generate_secondary_queries(q, "planning")
    joined = " ".join(qs).lower()
    # At least one action word should appear
    assert any(w in joined for w in ("forensic", "surveillance", "warrant"))


def test_secondary_queries_retrieval_uses_proper_nouns() -> None:
    q = "What was Sarah Mitchell wearing when last seen?"
    qs = _generate_secondary_queries(q, "retrieval")
    joined = " ".join(qs)
    assert "Sarah" in joined or "Mitchell" in joined


def test_secondary_queries_no_hardcoded_case_strings() -> None:
    """Ensure no case-specific strings are hardcoded."""
    qs = _generate_secondary_queries("unrelated question", "general")
    for q in qs:
        assert "Session" not in q or "session" not in q.lower()
        assert "Josh" not in q
        assert "Snapchat" not in q
        assert "Lawrence" not in q
        assert "InTown" not in q


# ── Hybrid search unit tests ──────────────────────────────────────────────────


def test_hybrid_search_returns_dicts_with_required_keys() -> None:
    hits = hybrid_search("missing child clothing description", k=5)
    assert len(hits) >= 1
    for h in hits:
        assert "evidence_id" in h
        assert "rrf_score" in h
        assert "excerpt" in h


def test_hybrid_search_scores_are_positive() -> None:
    hits = hybrid_search("forensic phone extraction", k=5)
    for h in hits:
        assert h["rrf_score"] >= 0.0


def test_hybrid_search_k_upper_bound() -> None:
    hits = hybrid_search("test query", k=3)
    assert len(hits) <= 3 * 2  # may return slightly more due to dedup logic in store


# ── New edge case tests ───────────────────────────────────────────────────────


# 1. Classification short-circuit: question has BOTH "wearing" AND "same person"
def test_classify_retrieval_short_circuits_before_multi_hop() -> None:
    """'wearing' is checked before 'same person'; result must be 'retrieval'."""
    q = "What was the same person wearing when seen on Snapchat?"
    assert _classify(q) == "retrieval", (
        "_classify should return 'retrieval' when 'wearing' appears, "
        "even if 'same person' is also present"
    )


# 2. Proper noun extraction: all-caps word (known limitation)
def test_extract_proper_nouns_allcaps_not_matched() -> None:
    """The regex [A-Z][a-z]+ requires at least one lowercase letter after the
    initial uppercase. ALL-CAPS words like 'MADISON' will NOT be captured.
    This test documents that known limitation."""
    nouns = _extract_proper_nouns("What did MADISON do?")
    assert "MADISON" not in nouns, (
        "ALL-CAPS words cannot match the Title-Case regex [A-Z][a-z]+ — "
        "this is a known limitation of the heuristic extractor"
    )


# 3. Proper noun extraction: numeric token is never returned
def test_extract_proper_nouns_no_numeric_token() -> None:
    """Digit tokens such as '3' cannot match [A-Z][a-z]+ and must not appear."""
    nouns = _extract_proper_nouns("3 suspects left the building")
    assert "3" not in nouns


# 4. Proper noun extraction: mixed-case multiword name
def test_extract_proper_nouns_multiword_name() -> None:
    """'Kyle D Lawrence': 'D' has no lowercase continuation so the 3-word
    phrase is split into 'Kyle' and 'Lawrence' by findall. Both are captured."""
    nouns = _extract_proper_nouns("Kyle D Lawrence was identified")
    assert "Kyle" in nouns, f"Expected 'Kyle' in {nouns}"
    assert "Lawrence" in nouns, f"Expected 'Lawrence' in {nouns}"
    # The single-char bridge 'D' alone does NOT appear
    assert "D" not in nouns


# 5. Platform extraction: platform at start of sentence, no quotes, no 'app'
def test_extract_apps_platforms_unquoted_no_app_suffix() -> None:
    """'Snapchat' at start of sentence with no quotation marks and no nearby
    'app' keyword — neither regex branch fires, so nothing is extracted."""
    platforms = _extract_apps_platforms("Snapchat was used to groom the victim")
    assert "Snapchat" not in platforms, (
        "Unquoted platform names with no 'app' nearby should not be extracted"
    )
    assert platforms == [], f"Expected empty list, got {platforms}"


# 6. Secondary queries: 'general' class yields empty list
def test_secondary_queries_general_class_returns_empty() -> None:
    """No branch in _generate_secondary_queries handles 'general'; result must
    be an empty list (no case-specific strings are ever injected)."""
    qs = _generate_secondary_queries("What happened?", "general")
    assert qs == [], f"Expected [], got {qs}"


# 7. run_retriever with k=1: at most k+4=5 chunks, all with required fields
def test_run_retriever_k1_chunk_count_and_fields() -> None:
    """With k=1 the slice merged[:k+4] = merged[:5], so at most 5 chunks."""
    state = AgentState(question=Q1)
    result = run_retriever(state, k=1)
    chunks = result.retrieved_chunks
    assert len(chunks) <= 5, f"Expected ≤5 chunks with k=1, got {len(chunks)}"
    for c in chunks:
        assert isinstance(c.evidence_id, str) and c.evidence_id, (
            f"Chunk has invalid evidence_id: {c.evidence_id!r}"
        )
        # excerpt is a string (may be empty if source had no text, but must exist)
        assert isinstance(c.excerpt, str), f"Chunk excerpt must be a string, got {type(c.excerpt)}"


# 8. Trace entry shape after run_retriever
def test_run_retriever_trace_entry_shape() -> None:
    """run_retriever appends exactly one entry with the documented keys."""
    state = AgentState(question=Q1)
    run_retriever(state, k=5)
    assert len(state.trace) == 1, f"Expected exactly 1 trace entry, got {len(state.trace)}"
    entry = state.trace[0]
    expected_keys = {
        "step",
        "case_id",
        "question_class",
        "hits",
        "secondary_queries",
        "top_ids",
    }
    assert set(entry.keys()) == expected_keys, (
        f"Trace entry has unexpected keys: {set(entry.keys())} vs {expected_keys}"
    )


def test_run_retriever_respects_case_id_isolation() -> None:
    state = AgentState(
        question="What was Madison wearing when last seen?",
        case_id="wrong-case",
    )

    result = run_retriever(state)

    assert result.retrieved_chunks == []
    entry = result.trace[-1]
    assert entry["case_id"] == "wrong-case"
    assert entry["step"] == "retriever"
    assert isinstance(entry["hits"], int)
    assert isinstance(entry["secondary_queries"], list)
    assert isinstance(entry["top_ids"], list)


# 9. Source diversity: diverse bucket comes before deferred bucket (ordering invariant)
def test_source_diversity_diverse_before_deferred_ordering() -> None:
    """After run_retriever, the diverse bucket (≤2 per source) is placed before
    the deferred bucket (overflow items that hit a 3rd+ slot for their source).
    Structural invariant: once we encounter a chunk whose evidence_id has
    already appeared ≥2 times earlier in the list, no chunk with a *new*
    (previously unseen) evidence_id must appear after it.
    This verifies the 'diverse + deferred' merge order, not just the cap."""
    state = AgentState(question=Q2)
    result = run_retriever(state, k=3)
    chunks = result.retrieved_chunks
    # k+4 = 7, so at most 7 chunks returned
    assert len(chunks) <= 7, f"Expected ≤7 chunks with k=3, got {len(chunks)}"

    # Walk the list tracking per-source counts.
    # Once any source hits its 3rd slot (deferred zone), no new source should appear.
    per_source_counts: dict[str, int] = {}
    deferred_zone_started = False
    seen_ids: set[str] = set()

    for c in chunks:
        ev_id = c.evidence_id
        count_so_far = per_source_counts.get(ev_id, 0)

        if count_so_far >= 2:
            # This is a deferred item — mark that we've entered the deferred zone
            deferred_zone_started = True

        if deferred_zone_started and ev_id not in seen_ids:
            raise AssertionError(
                f"New evidence_id {ev_id[:8]} appeared after deferred zone started — "
                f"diverse + deferred ordering is violated"
            )

        per_source_counts[ev_id] = count_so_far + 1
        seen_ids.add(ev_id)


# 10. Proper noun extraction: special regex characters in input don't crash
def test_extract_proper_nouns_special_regex_chars_in_input() -> None:
    """Parentheses, apostrophes, and other regex metacharacters in the *input*
    text (not the pattern) must not cause re.findall to raise."""
    q = "What is the suspect's (primary) connection?"
    try:
        nouns = _extract_proper_nouns(q)
    except re.error as exc:
        raise AssertionError(
            f"_extract_proper_nouns raised re.error on input with special chars: {exc}"
        ) from exc
    # Result must be a plain list; apostrophe/parens in input are silently ignored
    assert isinstance(nouns, list)
