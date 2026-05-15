"""Claim-support entailment backends.

The default backend is deterministic and offline-safe for CI. Production can
switch to a dedicated NLI cross-encoder by setting:

    NIGHTWAVE_ENTAILMENT_BACKEND=cross_encoder
    NIGHTWAVE_ENTAILMENT_MODEL=cross-encoder/nli-deberta-v3-base

The verifier treats this module as a judge over (evidence snippets, claim)
pairs; callers do not need to know whether the judge is lexical or model-based.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EntailmentDecision:
    supported: bool
    score: float
    backend: str


class EntailmentJudge(Protocol):
    backend: str

    def supports(self, claim: str, evidence_snippets: list[str]) -> EntailmentDecision: ...


class LexicalEntailmentJudge:
    """Deterministic entailment approximation used for offline CI."""

    backend = "lexical"

    def __init__(self, token_fn, threshold: float = 0.45) -> None:
        self._token_fn = token_fn
        self._threshold = threshold

    def supports(self, claim: str, evidence_snippets: list[str]) -> EntailmentDecision:
        claim_tokens = self._token_fn(claim)
        if not claim_tokens:
            return EntailmentDecision(True, 1.0, self.backend)

        best = 0.0
        combined_tokens: set[str] = set()
        for snippet in evidence_snippets:
            snippet_tokens = self._token_fn(snippet)
            combined_tokens.update(snippet_tokens)
            if not snippet_tokens:
                continue
            overlap = len(claim_tokens & snippet_tokens) / max(len(claim_tokens), 1)
            best = max(best, overlap)

        combined_overlap = len(claim_tokens & combined_tokens) / max(len(claim_tokens), 1)
        best = max(best, combined_overlap)
        return EntailmentDecision(best >= self._threshold, round(best, 4), self.backend)


class CrossEncoderEntailmentJudge:
    """Dedicated NLI judge using sentence-transformers CrossEncoder."""

    backend = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.62,
        entailment_index: int = 1,
    ) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

        self._model = CrossEncoder(model_name)
        self._threshold = threshold
        self._entailment_index = entailment_index

    def supports(self, claim: str, evidence_snippets: list[str]) -> EntailmentDecision:
        snippets = [snippet for snippet in evidence_snippets if snippet.strip()]
        if not snippets:
            return EntailmentDecision(False, 0.0, self.backend)

        pairs = [(snippet, claim) for snippet in snippets]
        raw_scores = self._model.predict(pairs)
        best = 0.0
        for raw in raw_scores:
            if hasattr(raw, "tolist"):
                values = raw.tolist()
            elif isinstance(raw, (list, tuple)):
                values = list(raw)
            else:
                values = [float(raw)]

            if len(values) == 1:
                entailment = float(values[0])
            else:
                exp = [math.exp(float(v)) for v in values]
                total = sum(exp) or 1.0
                probs = [v / total for v in exp]
                entailment = probs[min(self._entailment_index, len(probs) - 1)]
            best = max(best, entailment)

        return EntailmentDecision(best >= self._threshold, round(best, 4), self.backend)


def get_entailment_judge(token_fn) -> EntailmentJudge:
    backend = os.getenv("NIGHTWAVE_ENTAILMENT_BACKEND", "lexical").lower()
    if backend == "cross_encoder":
        model_name = os.getenv(
            "NIGHTWAVE_ENTAILMENT_MODEL",
            "cross-encoder/nli-deberta-v3-base",
        )
        threshold = float(os.getenv("NIGHTWAVE_ENTAILMENT_THRESHOLD", "0.62"))
        entailment_index = int(os.getenv("NIGHTWAVE_ENTAILMENT_INDEX", "1"))
        return CrossEncoderEntailmentJudge(
            model_name=model_name,
            threshold=threshold,
            entailment_index=entailment_index,
        )
    return LexicalEntailmentJudge(token_fn)
