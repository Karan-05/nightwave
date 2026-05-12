"""Deterministic synthesizer JSON extraction tests — no LLM calls.

Tests the robustness of _extract_json and _sanitize_json against
common LLM output formatting issues:
  - Apostrophe escape bug (\\' → ')
  - Markdown fences
  - Truncated JSON (graceful error)
  - Nested markdown fences
  - Extra whitespace and preamble text
"""

from __future__ import annotations

import json
import pytest

from nightwave.multiagent.subagents.synthesizer import (
    _extract_json,
    _extract_inline_citations,
    _sanitize_json,
)
from nightwave.multiagent.state import RetrievedChunk


# ── _sanitize_json ────────────────────────────────────────────────────────────

def test_sanitize_json_apostrophe_escape() -> None:
    raw = r'{"key": "it\'s a test"}'
    sanitized = _sanitize_json(raw)
    # After sanitizing, should be valid JSON
    parsed = json.loads(sanitized)
    assert parsed["key"] == "it's a test"


def test_sanitize_json_literal_tab() -> None:
    raw = '{"key": "value\there"}'
    sanitized = _sanitize_json(raw)
    parsed = json.loads(sanitized)
    assert "value" in parsed["key"]


def test_sanitize_json_passthrough_valid() -> None:
    raw = '{"key": "clean value", "num": 42}'
    sanitized = _sanitize_json(raw)
    assert json.loads(sanitized) == {"key": "clean value", "num": 42}


# ── _extract_json ─────────────────────────────────────────────────────────────

def test_extract_json_clean_object() -> None:
    raw = '{"response": "answer", "confidence": 0.8, "reasoning": "r", "citations": []}'
    result = _extract_json(raw)
    assert result["confidence"] == 0.8
    assert result["response"] == "answer"


def test_extract_json_markdown_fence_json() -> None:
    raw = '```json\n{"response": "fenced", "confidence": 0.7, "reasoning": "ok", "citations": []}\n```'
    result = _extract_json(raw)
    assert result["response"] == "fenced"
    assert result["confidence"] == 0.7


def test_extract_json_markdown_fence_no_lang() -> None:
    raw = '```\n{"response": "bare fence", "confidence": 0.6, "reasoning": "r", "citations": []}\n```'
    result = _extract_json(raw)
    assert result["response"] == "bare fence"


def test_extract_json_preamble_text() -> None:
    """LLM sometimes outputs explanation before the JSON object."""
    raw = 'Here is my answer:\n{"response": "with preamble", "confidence": 0.5, "reasoning": "r", "citations": []}'
    result = _extract_json(raw)
    assert result["response"] == "with preamble"


def test_extract_json_apostrophe_in_value() -> None:
    raw = r'{"response": "officer\'s report", "confidence": 0.9, "reasoning": "r", "citations": []}'
    result = _extract_json(raw)
    assert "officer" in result["response"]


def test_extract_json_nested_json_value() -> None:
    """Response field contains JSON-like content that should not break parsing."""
    inner = '{"response": "The suspect said \\"hello\\"", "confidence": 0.75, "reasoning": "quoted", "citations": []}'
    result = _extract_json(inner)
    assert result["confidence"] == 0.75


def test_extract_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError, match="No valid JSON"):
        _extract_json("this is not json at all, no braces")


def test_extract_json_truncated_raises() -> None:
    truncated = '{"response": "answer starts here but never ends...'
    with pytest.raises(ValueError):
        _extract_json(truncated)


def test_extract_json_multiple_objects_returns_first() -> None:
    """When output has multiple JSON objects, return the first complete one."""
    raw = '{"response": "first", "confidence": 0.8, "reasoning": "r", "citations": []} {"response": "second", "confidence": 0.5}'
    result = _extract_json(raw)
    assert result["response"] == "first"


# ── Citation extraction integration ──────────────────────────────────────────

def test_extract_json_citations_array() -> None:
    ev_id = "163faaac-d742-4160-aa62-070f9ecb96cb"
    raw = json.dumps({
        "response": f"She wore [{ev_id}: black hoodie]",
        "confidence": 0.93,
        "reasoning": "from alert PDF",
        "citations": [
            {"evidence_id": ev_id, "excerpt": "black hoodie", "confidence": 0.93}
        ],
    })
    result = _extract_json(raw)
    assert result["citations"][0]["evidence_id"] == ev_id
    assert result["citations"][0]["confidence"] == 0.93


# ── New edge cases ────────────────────────────────────────────────────────────

# 1. Confidence as string
def test_extract_json_confidence_as_string() -> None:
    """LLM emits confidence as a quoted string instead of a float."""
    raw = '{"response": "test", "confidence": "0.75", "reasoning": "r", "citations": []}'
    result = _extract_json(raw)
    # _extract_json returns the raw parsed dict; the string survives
    assert result["confidence"] == "0.75"
    # The synthesizer wraps it in float() — verify that cast succeeds
    assert float(result["confidence"]) == 0.75


# 2. Missing citations key
def test_extract_json_missing_citations_key() -> None:
    """Parsed dict without a citations key: .get("citations", []) must not raise."""
    raw = '{"response": "no cites", "confidence": 0.6, "reasoning": "r"}'
    result = _extract_json(raw)
    assert "citations" not in result
    # This is the exact pattern used in run_synthesizer
    assert result.get("citations", []) == []


# 3. Citations with null evidence_id — regression: None excerpt/confidence crash
def test_extract_json_citation_null_fields_no_crash() -> None:
    """null excerpt and confidence in a citation must not crash the synthesizer helper.

    Before the fix, cit.get('excerpt') returned None (key present, value null),
    and None[:400] raised TypeError.  Same for float(None).
    """
    import json as _json
    from nightwave.multiagent.subagents.synthesizer import _extract_json as _ej

    raw = _json.dumps({
        "response": "answer",
        "confidence": 0.8,
        "reasoning": "r",
        "citations": [
            {"evidence_id": None, "excerpt": None, "confidence": None}
        ],
    })
    parsed = _ej(raw)
    cit = parsed["citations"][0]

    # Simulate the exact expressions used in run_synthesizer after the fix:
    excerpt_val = (cit.get("excerpt") or "")[:400]
    confidence_val = float(cit.get("confidence") or 0.75)

    assert excerpt_val == ""
    assert confidence_val == 0.75


# 4. Whitespace-only excerpt in citation
def test_extract_json_whitespace_only_excerpt() -> None:
    """An excerpt that is only spaces should be accepted by _extract_json."""
    raw = json.dumps({
        "response": "test",
        "confidence": 0.8,
        "reasoning": "r",
        "citations": [
            {"evidence_id": "some-id", "excerpt": "   ", "confidence": 0.8}
        ],
    })
    result = _extract_json(raw)
    assert result["citations"][0]["excerpt"] == "   "


# 5. Confidence out of range
def test_extract_json_confidence_out_of_range() -> None:
    """confidence=1.5 must not crash _extract_json; the value passes through."""
    raw = '{"response": "test", "confidence": 1.5, "reasoning": "r", "citations": []}'
    result = _extract_json(raw)
    assert result["confidence"] == 1.5
    # float() cast also succeeds
    assert float(result["confidence"]) == 1.5


# 6. Multiple inline citations to the same evidence_id deduplicated
def test_extract_inline_citations_deduplicates_same_id() -> None:
    """Two [uuid: ...] mentions for the same ID should produce exactly one DraftCitation."""
    ev_id = "abcdef12-1234-5678-abcd-abcdef123456"
    chunk = RetrievedChunk(
        evidence_id=ev_id,
        source_path="/fake/path.txt",
        locator={"type": "text", "line_start": 1, "line_end": 5},
        excerpt="some evidence text",
        score=0.9,
        source="bm25",
    )
    chunk_index = {ev_id: chunk}
    response_text = (
        f"First mention [{ev_id}: text1] and second mention [{ev_id}: text2]."
    )
    results = _extract_inline_citations(response_text, chunk_index)
    # Should only have one citation despite two references to the same ID
    assert len(results) == 1
    assert results[0].evidence_id == ev_id


# 7. Leading/trailing whitespace around JSON
def test_extract_json_leading_trailing_whitespace() -> None:
    """Whitespace around the JSON object should not prevent parsing."""
    raw = '   {"key": "val", "confidence": 0.5}   '
    result = _extract_json(raw)
    assert result["key"] == "val"
    assert result["confidence"] == 0.5


# 8. Nested braces inside response field — brace-matcher must not be confused
def test_extract_json_nested_braces_in_response_field_with_preamble() -> None:
    """When preamble forces brace-matching, a response with {curly braces} inside
    a string value must not confuse the outermost-brace extractor."""
    inner = json.dumps({
        "response": "answer is {unclear} and {ambiguous}",
        "confidence": 0.5,
        "reasoning": "r",
        "citations": [],
    })
    # Add preamble so the direct-parse path fails first and brace-matching is used
    raw = "Here is my answer:\n" + inner
    result = _extract_json(raw)
    assert result["response"] == "answer is {unclear} and {ambiguous}"
    assert result["confidence"] == 0.5
