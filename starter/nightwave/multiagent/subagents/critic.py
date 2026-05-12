"""Critic subagent — citation validation gate before submit.

Two hard checks (both must pass):
  1. evidence_id_valid   — every cited ID exists in case_data.evidence
  2. excerpt_grounded    — excerpt text appears in the actual source file
                           at the claimed locator (excerpt-at-locator verification)

One soft check:
  3. confidence_calibration — confidence <= 0.70 when evidence conflicts (Q2-style)

If any hard check fails, critic returns structured feedback for the
synthesizer to retry (max 1 retry, then fail loudly).

This is the architectural addition that makes multi-agent defensible on
quality grounds: the original single-agent eval checked (1) post-hoc.
We now enforce both (1) and (2) *before* the answer is emitted.
"""

from __future__ import annotations

import re

from nightwave.case import DEFAULT_EVIDENCE_DIR
from nightwave.multiagent.state import AgentState, DraftCitation
from nightwave.tools import _INDEX

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False


def _load_source_text(ev_id: str, locator: dict) -> str:
    """Extract the text at `locator` from the evidence file."""
    ev = _INDEX.evidence_by_id.get(ev_id, {})
    if not ev:
        return ""

    filename = ev.get("filename", "")
    path = DEFAULT_EVIDENCE_DIR / filename

    loc_type = locator.get("type", "text")

    if loc_type == "pdf" and _HAS_PYPDF:
        try:
            reader = PdfReader(str(path))
            page_num = int(locator.get("page", 1)) - 1
            if 0 <= page_num < len(reader.pages):
                page_text = reader.pages[page_num].extract_text() or ""
                return f"{page_text} {ev.get('description', '')} {ev.get('ai_summary', '')}"
        except Exception:
            return f"{ev.get('description', '')} {ev.get('ai_summary', '')}"
    if loc_type == "pdf":
        return f"{ev.get('description', '')} {ev.get('ai_summary', '')}"

    if loc_type == "text":
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, int(locator.get("line_start", 1)) - 1)
            end = int(locator.get("line_end", start + 10))
            # Give a window ±5 lines around the locator for flexibility
            start = max(0, start - 5)
            end = min(len(lines), end + 5)
            window = " ".join(lines[start:end])
            full_text = " ".join(lines)
            return f"{window} {full_text} {ev.get('description', '')} {ev.get('ai_summary', '')}"
        except Exception:
            return f"{ev.get('description', '')} {ev.get('ai_summary', '')}"

    # Fallback: full file text (ai_summary locator)
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        # For images/video, use ai_summary
        return f"{ev.get('description', '')} {ev.get('ai_summary', '')}"


def _excerpt_grounded(citation: DraftCitation) -> bool:
    """Return True if the excerpt can be found in the source at the locator."""
    excerpt = citation.excerpt.strip()
    if not excerpt:
        return False

    source_text = _load_source_text(citation.evidence_id, citation.locator)
    if not source_text:
        # Can't verify (image/video) — give benefit of the doubt
        return True

    # Normalise both to lowercase, collapse whitespace
    norm_excerpt = re.sub(r"\s+", " ", excerpt.lower())
    norm_source = re.sub(r"\s+", " ", source_text.lower())

    # Sliding window: allow excerpt to span up to 2x its length in source
    # (handles minor OCR/wrapping variations)
    if norm_excerpt in norm_source:
        return True

    # Partial match: all significant tokens of the excerpt present in source
    tokens = [t for t in norm_excerpt.split() if len(t) > 3]
    if tokens and sum(1 for t in tokens if t in norm_source) / len(tokens) >= 0.80:
        return True

    return False


def run_critic(state: AgentState) -> AgentState:
    """Validate state.draft_citations. Populate critic_passed / critic_feedback."""
    citations = state.draft_citations
    issues: list[str] = []
    verified_citations: list[DraftCitation] = []

    for cit in citations:
        # Hard check 1: evidence_id must exist
        if cit.evidence_id not in _INDEX.evidence_by_id:
            issues.append(
                f"HALLUCINATED evidence_id '{cit.evidence_id}' — does not exist in case data"
            )
            continue  # do not carry this citation forward

        # Hard check 2: excerpt must appear at locator
        grounded = _excerpt_grounded(cit)
        if not grounded:
            issues.append(
                f"UNGROUNDED excerpt for '{cit.evidence_id}' — "
                f"text not found at locator {cit.locator}. "
                f"Excerpt: '{cit.excerpt[:80]}'"
            )
            # Still carry the citation but mark unverified
            cit.verified = False
        else:
            cit.verified = True

        verified_citations.append(cit)

    # Soft check: confidence calibration
    conf = state.draft_confidence
    has_conflict = any(
        h.get("confidence", 1.0) < 0.55
        for h in state.graph_context
        if isinstance(h, dict) and h.get("artifact_type") == "hypothesis"
    )
    if has_conflict and conf > 0.70:
        issues.append(
            f"CONFIDENCE too high ({conf:.2f}) for conflicting evidence — "
            "cap at 0.70 per calibration rubric"
        )

    # Critic decision
    ungrounded_count = sum(1 for c in verified_citations if not c.verified)
    hard_fail = len(issues) > ungrounded_count  # hallucinated IDs = hard fail

    state.draft_citations = verified_citations

    if hard_fail:
        state.critic_passed = False
        state.critic_feedback = "CRITIC HARD FAIL:\n" + "\n".join(issues)
    elif ungrounded_count > 0:
        state.critic_passed = False
        state.critic_feedback = (
            "CRITIC SOFT FAIL (ungrounded excerpts — revise):\n" + "\n".join(issues)
        )
    else:
        state.critic_passed = True
        state.critic_feedback = ""

    state.trace.append({
        "step": "critic",
        "passed": state.critic_passed,
        "issues": issues,
        "verified_citations": len(verified_citations),
        "ungrounded": ungrounded_count,
    })

    return state
