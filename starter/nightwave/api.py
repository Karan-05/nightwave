"""HTTP API for the Nightwave multi-agent harness."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nightwave.multiagent.graph_db import get_mode
from nightwave.multiagent.run import answer_multi, run_eval_multi
from nightwave.tools import _INDEX

app = FastAPI(
    title="Nightwave Take-Home Agent API",
    version="0.1.0",
    description="Citation-governed investigative QA over the Madison Fields case.",
)


def _case_id() -> str:
    return _INDEX.case_id or ""


def _eval_endpoint_enabled() -> bool:
    return os.getenv("NIGHTWAVE_ENABLE_EVAL_ENDPOINT", "").lower() in {"1", "true", "yes"}


class CitationOut(BaseModel):
    evidence_id: str
    source_path: str
    locator: dict[str, Any]
    excerpt: str
    confidence: float


class TraceStepOut(BaseModel):
    kind: str
    detail: dict[str, Any]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    question_id: str = ""


class AnswerResponse(BaseModel):
    question: str
    response: str
    confidence: float
    citations: list[CitationOut]
    trace: list[TraceStepOut]
    graph_mode: str
    case_id: str


class HealthResponse(BaseModel):
    ok: bool
    graph_mode: str
    case_id: str
    synth_mode: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ok=True,
        graph_mode=get_mode(),
        case_id=_case_id(),
        synth_mode=os.getenv("SYNTH_MODE", "live"),
    )


@app.post("/v1/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must contain non-whitespace text")

    result = answer_multi(question, request.question_id)
    return AnswerResponse(
        question=result.question,
        response=result.response,
        confidence=result.confidence,
        citations=[
            CitationOut(
                evidence_id=c.evidence_id,
                source_path=c.source_path,
                locator=c.locator,
                excerpt=c.excerpt,
                confidence=c.confidence,
            )
            for c in result.citations
        ],
        trace=[TraceStepOut(kind=step.kind, detail=step.detail) for step in result.trace],
        graph_mode=get_mode(),
        case_id=_case_id(),
    )


@app.post("/v1/eval")
def eval_multiagent() -> dict[str, Any]:
    if not _eval_endpoint_enabled():
        raise HTTPException(
            status_code=403,
            detail="Eval endpoint is disabled. Set NIGHTWAVE_ENABLE_EVAL_ENDPOINT=1 to enable.",
        )
    return run_eval_multi()
