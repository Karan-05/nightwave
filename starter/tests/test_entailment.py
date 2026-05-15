"""Entailment judge tests."""

from __future__ import annotations

from nightwave.multiagent.claim_verifier import _tokens, verify_claim_coverage
from nightwave.multiagent.entailment import EntailmentDecision, LexicalEntailmentJudge


class FakeEntailmentJudge:
    backend = "fake_nli"

    def supports(self, claim: str, evidence_snippets: list[str]) -> EntailmentDecision:
        if "departed the hotel" in claim.lower() and any(
            "left the intown suites" in snippet.lower() for snippet in evidence_snippets
        ):
            return EntailmentDecision(True, 0.91, self.backend)
        return EntailmentDecision(False, 0.08, self.backend)


def test_lexical_entailment_judge_scores_overlap() -> None:
    judge = LexicalEntailmentJudge(_tokens)

    decision = judge.supports(
        "Madison was wearing a black Champion sweatshirt",
        ["wearing black Champion hooded sweatshirt"],
    )

    assert decision.supported is True
    assert decision.backend == "lexical"
    assert decision.score > 0


def test_claim_verifier_accepts_dedicated_entailment_judge_support() -> None:
    result = verify_claim_coverage(
        "Madison departed the hotel.",
        [],
        judge=FakeEntailmentJudge(),
    )

    assert result["claim_count"] == 1
    assert result["supported_claim_count"] == 0

    result = verify_claim_coverage(
        "Madison departed the hotel [b8281149-7efc-4726-ac7b-31172e5dc458: "
        "Madison left the InTown Suites].",
        [],
        judge=FakeEntailmentJudge(),
    )

    assert result["claim_count"] == 1
    assert result["supported_claim_count"] == 1
    assert result["entailment_backend"] == "fake_nli"
    assert result["min_entailment_score"] == 0.91


def test_claim_verifier_rejects_when_entailment_judge_rejects() -> None:
    result = verify_claim_coverage(
        "A suspect confessed in March [163faaac-d742-4160-aa62-070f9ecb96cb: "
        "Madison wore a black hoodie].",
        [],
        judge=FakeEntailmentJudge(),
    )

    assert result["supported_claim_count"] == 0
    assert result["uncovered_claims"]
