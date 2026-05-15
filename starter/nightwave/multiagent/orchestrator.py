"""Orchestrator — typed state machine for the multi-agent pipeline.

Observability: every pipeline run appends a structured record to
agent_trace.jsonl (JSON Lines) so individual question runs can be
audited without reading the full eval report.

Cost tracking: each synthesizer call logs cost_usd; orchestrator
aggregates and surfaces total_cost_usd in the trace.


Pipeline (per question):
  ┌─────────────────────────────────────┐
  │         asyncio.gather              │ ← parallel, independent
  │  [Retriever]    [Graph Agent]       │
  └──────────────┬──────────────────────┘
                 ▼
           [Synthesizer]                  ← structured JSON output
                 ▼
             [Critic]                     ← hard citation validation
          ┌────┴─────┐
        PASS        FAIL (max 1 retry)
          │              │
          ▼              ▼ (re-run synthesizer with feedback)
       [emit]         [emit with warning]

Async: retriever + graph_agent run concurrently (asyncio.gather).
Serial: synthesizer → critic (critic depends on synthesizer output).

Token budget: configurable via MAX_TOKENS_PER_QUESTION env var (default 8000).
If budget would be exceeded, skip retry and emit with critic feedback attached.
"""

from __future__ import annotations

import asyncio
import os
import time

from nightwave.agent import Answer, Citation, Step
from nightwave.multiagent.observability import RunObserver
from nightwave.multiagent.state import AgentState
from nightwave.multiagent.subagents.critic import run_critic
from nightwave.multiagent.subagents.graph_agent import run_graph_agent
from nightwave.multiagent.subagents.retriever import run_retriever
from nightwave.multiagent.subagents.synthesizer import (
    run_deterministic_synthesizer,
    run_synthesizer,
)

_MAX_TOKENS = int(os.getenv("MAX_TOKENS_PER_QUESTION", "8000"))
_MAX_CRITIC_RETRIES = 1


def _run_in_executor(fn, *args):
    """Wrap a sync function for asyncio.gather."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, fn, *args)


async def _parallel_retrieval(state: AgentState) -> AgentState:
    """Run retriever and graph_agent concurrently, merge results into state."""
    import concurrent.futures

    loop = asyncio.get_event_loop()

    # Clone state for parallel mutation (each subagent writes different fields)
    import copy

    state_r = copy.deepcopy(state)
    state_g = copy.deepcopy(state)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_r = loop.run_in_executor(pool, run_retriever, state_r)
        future_g = loop.run_in_executor(pool, run_graph_agent, state_g)
        state_r, state_g = await asyncio.gather(future_r, future_g)

    # Merge: each wrote to its own field, no conflict
    state.retrieved_chunks = state_r.retrieved_chunks
    state.graph_context = state_g.graph_context
    state.trace += state_r.trace + state_g.trace

    return state


def run_pipeline(question: str, question_id: str = "") -> Answer:
    """Run the full multi-agent pipeline for a single question."""
    t_start = time.time()

    state = AgentState(question=question, question_id=question_id)
    observer = RunObserver(question_id=question_id, case_id=state.case_id)
    state.trace.append(observer.trace_event("orchestrator_start"))

    # ── Stage 1: parallel retrieval ───────────────────────────────────────────
    with observer.time_stage("parallel_retrieval"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            state = loop.run_until_complete(_parallel_retrieval(state))
        finally:
            loop.close()

    # ── Stage 2: synthesizer ──────────────────────────────────────────────────
    with observer.time_stage("synthesizer_initial"):
        state = run_synthesizer(state)

    # ── Stage 3: critic gate (with one retry) ─────────────────────────────────
    with observer.time_stage("critic_gate"):
        for attempt in range(_MAX_CRITIC_RETRIES + 1):
            state.critic_attempts = attempt
            state = run_critic(state)

            if state.critic_passed:
                break

            total_tokens = sum(state.token_usage.values())
            if attempt < _MAX_CRITIC_RETRIES and total_tokens < _MAX_TOKENS:
                # Re-run synthesizer with critic feedback
                state = run_synthesizer(state)
            else:
                feedback_can_be_repaired = (
                    "CONFIDENCE too high" in state.critic_feedback
                    or "UNGROUNDED" in state.critic_feedback
                    or "CLAIM_COVERAGE" in state.critic_feedback
                )
                can_use_fallback = (
                    feedback_can_be_repaired
                    and "HALLUCINATED" not in state.critic_feedback
                    and "NO_CITATIONS" not in state.critic_feedback
                    and "INLINE_CITATION_MISMATCH" not in state.critic_feedback
                )
                if not state.critic_passed and can_use_fallback:
                    state.trace.append(
                        observer.trace_event(
                            "orchestrator_fallback",
                            reason="critic_failed_after_retry",
                            critic_feedback=state.critic_feedback,
                        )
                    )
                    state = run_deterministic_synthesizer(state)
                    state.critic_attempts = attempt + 1
                    state = run_critic(state)

                # Budget exceeded or no retries left. Keep internal critic feedback in
                # trace only; user-facing answers must never contain validator chatter.
                if not state.critic_passed:
                    state.draft_confidence = min(state.draft_confidence, 0.4)
                break

    # ── Stage 4: emit final Answer ────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 2)
    total_tokens = sum(state.token_usage.values())
    total_cost = sum(t.get("cost_usd", 0.0) for t in state.trace if isinstance(t, dict))

    orchestrator_record = {
        "step": "orchestrator",
        "run_id": observer.run_id,
        "case_id": state.case_id,
        "question_id": state.question_id,
        "elapsed_s": elapsed,
        "stage_timings_s": observer.stage_timings,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 5),
        "token_usage": state.token_usage,
        "critic_attempts": state.critic_attempts,
        "critic_passed": state.critic_passed,
        "citations_emitted": len(state.draft_citations),
        "confidence": state.draft_confidence,
    }
    state.trace.append(orchestrator_record)

    # Append to JSON Lines trace file for observability
    try:
        observer.write_run_record({**orchestrator_record, "trace": state.trace})
    except Exception:
        pass

    citations = [
        Citation(
            evidence_id=c.evidence_id,
            source_path=c.source_path,
            locator=c.locator,
            excerpt=c.excerpt,
            confidence=c.confidence,
        )
        for c in state.draft_citations
    ]

    trace = [Step(kind=t.get("step", "unknown"), detail=t) for t in state.trace]

    return Answer(
        question=question,
        response=state.draft_response,
        confidence=state.draft_confidence,
        citations=citations,
        trace=trace,
    )
