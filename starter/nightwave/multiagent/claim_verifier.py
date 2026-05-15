"""Claim-to-citation support checks backed by an entailment judge."""

from __future__ import annotations

import re
from typing import Any

from nightwave.multiagent.entailment import EntailmentJudge, get_entailment_judge
from nightwave.multiagent.state import DraftCitation

_INLINE_CITATION_RE = re.compile(r"\[([a-f0-9\-]{36}):\s*(.+?)\]", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"\b[a-z0-9][a-z0-9'-]{2,}\b", flags=re.IGNORECASE)
_STOPWORDS = {
    "about",
    "according",
    "also",
    "and",
    "are",
    "because",
    "been",
    "but",
    "case",
    "comes",
    "could",
    "document",
    "does",
    "evidence",
    "for",
    "from",
    "has",
    "have",
    "her",
    "his",
    "information",
    "into",
    "not",
    "notes",
    "official",
    "primarily",
    "reported",
    "same",
    "source",
    "that",
    "the",
    "their",
    "this",
    "was",
    "were",
    "which",
    "with",
}
_FACT_VERBS = {
    "active",
    "admitted",
    "carrying",
    "confessed",
    "communicated",
    "confirmed",
    "contact",
    "departed",
    "groomed",
    "is",
    "known",
    "left",
    "linked",
    "located",
    "met",
    "reported",
    "seen",
    "told",
    "traveled",
    "unconfirmed",
    "unknown",
    "used",
    "using",
    "wearing",
    "went",
    "wore",
}


def _tokens(text: str) -> set[str]:
    return {w.lower().strip("'") for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _candidate_claims(response: str) -> list[str]:
    claims: list[str] = []
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|"):
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.:]\s*", "", line)
        if line.lower().startswith(("expected output:", "supporting artifacts:")):
            continue
        if not line or len(_tokens(line)) < 2:
            continue
        protected = re.sub(r"\b([A-Z])\.", r"\1<dot>", line)
        parts = [part.replace("<dot>", ".") for part in re.split(r"(?<=[.!?])\s+", protected)]
        for part in parts:
            part = part.strip()
            if part.lower().startswith(("expected output:", "supporting artifacts:")):
                continue
            part_tokens = _tokens(part)
            if len(part_tokens) >= 2 and (part_tokens & _FACT_VERBS):
                claims.append(part)
    return claims


def _claim_text(claim: str) -> str:
    return _INLINE_CITATION_RE.sub(" ", claim).strip()


def _inline_excerpts(claim: str) -> list[str]:
    return [match.group(2).strip("'\" ") for match in _INLINE_CITATION_RE.finditer(claim)]


def verify_claim_coverage(
    response: str,
    citations: list[DraftCitation],
    judge: EntailmentJudge | None = None,
) -> dict[str, Any]:
    """Return deterministic coverage stats for response claims."""
    judge = judge or get_entailment_judge(_tokens)
    claims = _candidate_claims(response)
    uncovered: list[str] = []
    supported = 0
    support_scores: list[float] = []

    for claim in claims:
        evidence_snippets = _inline_excerpts(claim)
        evidence_snippets.extend(citation.excerpt for citation in citations)
        decision = judge.supports(_claim_text(claim), evidence_snippets)
        support_scores.append(decision.score)
        if decision.supported:
            supported += 1
        else:
            uncovered.append(claim)

    coverage = supported / len(claims) if claims else 1.0
    return {
        "claim_count": len(claims),
        "supported_claim_count": supported,
        "coverage": round(coverage, 3),
        "uncovered_claims": uncovered[:5],
        "entailment_backend": judge.backend,
        "min_entailment_score": round(min(support_scores), 4) if support_scores else 1.0,
    }
