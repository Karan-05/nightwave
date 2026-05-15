"""Shared typed state passed through the orchestrator→subagent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nightwave.case import current_case_id


@dataclass
class RetrievedChunk:
    evidence_id: str
    source_path: str
    locator: dict
    excerpt: str
    score: float
    source: str  # "bm25" | "vector" | "graph" | "rrf"
    case_id: str = field(default_factory=current_case_id)


@dataclass
class DraftCitation:
    evidence_id: str
    source_path: str
    locator: dict
    excerpt: str
    confidence: float
    verified: bool = False  # set True by critic after locator check


@dataclass
class AgentState:
    question: str
    question_id: str = ""
    case_id: str = field(default_factory=current_case_id)

    # Retrieval outputs (populated by retriever + graph_agent in parallel)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    graph_context: list[dict[str, Any]] = field(default_factory=list)

    # Synthesizer output
    draft_response: str = ""
    draft_confidence: float = 0.5
    draft_reasoning: str = ""
    draft_citations: list[DraftCitation] = field(default_factory=list)

    # Critic feedback
    critic_passed: bool = False
    critic_feedback: str = ""
    critic_attempts: int = 0

    # Final answer (set after critic passes)
    final_response: str = ""
    final_confidence: float = 0.5
    final_citations: list[DraftCitation] = field(default_factory=list)

    # Observability
    token_usage: dict[str, int] = field(default_factory=dict)  # subagent → tokens
    trace: list[dict[str, Any]] = field(default_factory=list)
