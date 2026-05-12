# Solution - Nightwave Take-Home

## Time and Tools

Approximately 8–10 hours across two sessions — past the 6h soft cap. I went over to harden the citation pipeline and add the provider adapter; what's still on my list is in "What I Would Do Next." I used Claude Code (Anthropic's agentic CLI) substantively throughout — for code generation, test scaffolding, iteration on the graph seeder, and debugging the synthesizer JSON parsing edge cases. The architecture decisions, eval design, and citation discipline model were mine; Claude Code was my pair programmer on implementation.

## Product Shape

A citation-governed investigative reasoning pipeline. The product constraint: a detective must be able to ask a question and get an answer whose claims trace back to case evidence. That requires two capabilities working together — raw evidence retrieval for facts and excerpts, and graph reasoning for entities, leads, hypotheses, and relationships.

## Why Not a Single ReAct Loop?

The README warns against multi-agent theater, and I want to defend the choice directly.

A single ReAct loop with three tools can likely answer all three questions — the questions are answerable from BM25 alone, and citation validity isn't enforced by the retrieval harness. I did not run a single-agent eval against the live API on this pass, so I can't report a score delta. The case for multi-agent rests on the qualitative argument below, not a score gap.

The reason multi-agent is correct here is **citation circuit closure**. A ReAct loop that calls `search_evidence` and `submit_answer` produces answers, but no agent validates that the `evidence_id` values actually exist in the case data before they reach the detective. The critic closes that circuit. In production — where LLM outputs vary across runs and cases — the critic is the difference between a system that provably cites real evidence and one that hopes the LLM got it right. This matters in an investigative context: a hallucinated citation ID in a disclosure memo is a Brady problem, not a UX bug.

The graph agent is the other non-optional piece. Hypothesis confidence scores (hyp-josh-coconspirator at 0.47, hyp-josh-equals-lawrence at 0.34) require structured traversal over entities and relationships. BM25 search over a flat text index can retrieve the text that names them, but not reason about their relative status or the edges between them.

The pipeline:

1. Retriever agent — hybrid BM25 + dense embedding search with reciprocal rank fusion
2. Graph agent — typed traversal over entities, relationships, events, leads, hypotheses
3. Synthesizer agent — structured JSON answer with calibrated confidence
4. Critic agent — validates every citation before it reaches the detective

The original single-agent ReAct harness remains available with `NIGHTWAVE_ARCHITECTURE=single`.

## Architecture

### Retriever

Hybrid retrieval when dependencies are available: BM25 lexical search over parsed evidence chunks, dense embeddings via `sentence-transformers`, reciprocal rank fusion for reranking. Degrades to BM25 if `numpy` or the model is unavailable — keeps the submission runnable without network access or model downloads.

### Graph Agent

Designed around Neo4j with an in-memory fallback. Neo4j fits because the case state is relational: Madison → Josh → Session, Madison → Kyle Lawrence → Snapchat, hypotheses → supporting evidence. The seeder uses idempotent `MERGE` statements and creates `SUPPORTS` edges from `lead.source_hypothesis_id`. Final industrial run used Colima + Docker Compose to start Neo4j 5 Community; seeded the graph; verified node/edge counts; ran the eval with `graph_mode: neo4j`.

### Synthesizer

Receives only retrieved evidence chunks and graph context. Uses Claude with `temperature=0` and enforces structured JSON output via a strict system prompt. Output schema (`response`, `confidence`, `reasoning`, `citations`) is validated and sanitized — handling LLM edge cases like invalid `\'` apostrophe escapes, markdown fences, and truncated responses. On parse failure, raw output is returned with degraded confidence rather than crashing.

### Critic

Validates that every `evidence_id` exists in `case_data.json`, that excerpts are grounded in source text or evidence metadata, and that confidence is not over-calibrated when graph hypotheses conflict. Critic feedback stays internal trace data — earlier versions leaked warnings into the user-facing answer, which is unacceptable product behavior. If validation fails after retry, confidence is capped rather than the response polluted.

## Tool Choices

- `search_evidence`: lexical search over parsed raw evidence, PDF text when available, metadata summaries
- `get_entity`: entity lookup by id/name/alias with citations attached
- `query_graph`: typed graph traversal over entities, relationships, events, leads, hypotheses
- `list_citations`: artifact-to-citation lookup
- `submit_answer`: structured final answer contract for the single-agent baseline

## Confidence Model

| Question type | Expected range | Rationale |
|---|---:|---|
| Single-source official document | 0.85–0.97 | Q1 directly answered by the Ohio AG alert |
| Conflicting identity reasoning | 0.45–0.70 | Q2 has two active hypotheses; Session identity unresolved |
| Investigative planning | 0.50–0.78 | Q3 ranks actions from case state; outcomes uncertain |

Live LLM confidences: Q1 0.93, Q2 0.54, Q3 0.68.

Q2 is intentionally moderate. The system should not claim certainty where the case graph keeps both identity hypotheses active.

## Eval Methodology

| Metric | Weight | Purpose |
|---|---:|---|
| Correctness | 35% | Required facts, entities, topics, and action count |
| Citation grounding | 30% | Every citation must resolve to real case evidence |
| Must-cite coverage | 20% | Required evidence or hypothesis IDs must appear |
| Confidence calibration | 15% | Confidence must match the expected uncertainty band |

A hallucinated citation ID triggers a hard penalty. Raw answers are written to `raw_answers.json`; reports to `eval_report.json` and `eval_report_multiagent.json`.

Final result:

```text
overall_score: 1.000
passed: true
architecture: multiagent
graph_mode: neo4j
```

Validation:

```text
ruff check .                         # passed
mypy nightwave                       # passed
python -m compileall nightwave tests # passed
pytest -q --ignore=tests/test_multiagent_contract.py   # 136 passed, no LLM required
pytest tests/test_multiagent_contract.py               # 3 passed, live Anthropic + Neo4j
```

Test breakdown (10 files, 136 deterministic tests):

| File | Tests | Surface |
|---|---:|---|
| `test_retriever.py` | 32 | Diversity cap, classification, proper-noun/platform extraction, trace shape, k=1 |
| `test_graph_agent.py` | 27 | Memory routing, classification priority, state shape, cap enforcement |
| `test_synthesizer_json.py` | 21 | JSON parsing, null field crashes, citation dedup, confidence coercion |
| `test_critic.py` | 11 | Hard-fail hallucination guard, soft grounding, empty excerpts, long excerpts |
| `test_adversarial.py` | 9 | Fake IDs, prompt injection, non-existent entity, confidence capping |
| `test_orchestrator.py` | 7 | Retry logic, confidence cap, token budget guard, no-leak of critic feedback |
| `test_orchestrator_state.py` | 7 | State isolation, token accumulation, graph cap passthrough, mutable default safety |
| `test_no_hardcoding.py` | 3 | AST scan: no entity IDs, Cypher args, or platform keywords in dispatch lists |
| `test_llm_adapter.py` | 19 | Protocol conformance, factory dispatch for all 5 providers, model defaults, env-var overrides, missing-key errors |

Adversarial eval (`questions_adversarial.json`) covers 6 cases: out-of-scope date, hallucinated entity, fake evidence probe, prompt injection, pre-case fabrication, leading question. The structural guards — hallucination blocking, confidence capping, prompt injection resistance — are all deterministically tested. Automatically scoring LLM refusal prose against `acceptable_responses` / `forbidden_phrases` would require an LLM judge and add complexity I deliberately kept out of scope.

## Provider-Neutral LLM Adapter

The synthesizer dispatches through a thin adapter (`nightwave/multiagent/llm.py`). Two env vars control it:

```bash
NIGHTWAVE_LLM_PROVIDER=anthropic   # default; also: openai, gemini, bedrock, ollama
NIGHTWAVE_LLM_MODEL=gpt-4o         # overrides per-provider default
```

| Provider | Key env var | Default model |
|---|---|---|
| anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| openai | `OPENAI_API_KEY` | `gpt-4o` |
| gemini | `GOOGLE_API_KEY` | `gemini-2.0-flash` |
| bedrock | boto3 credential chain | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| ollama | none (`OLLAMA_BASE_URL` for endpoint) | `llama3.2` |

None of the adapters use native structured output modes. All five go through the same prompt-only JSON enforcement and `_extract_json` fallback, so the synthesizer validation path and critic work identically across providers.

## Scalability Bottlenecks

**1. Module-level case singleton.** `_INDEX` is loaded once from a single `case_data.json`. Scaling to 100 cases requires a `CaseStore` that fetches by `case_id` from Postgres, lazy-loaded per request.

**2. `asyncio.new_event_loop()` per question.** The orchestrator creates and discards an event loop per question. In a server context this needs to run inside a FastAPI background task queue, not spin up a new loop per call.

**3. No streaming.** The pipeline returns a full `Answer` after all four stages complete. The SSE contract in `schema/agent/sherlock.types.ts` already defines `sherlock_thinking`, `sherlock_tool_call`, and `sherlock_complete` events — wiring the orchestrator to emit SSE per step is the next integration for the production UI.

## What I Would Do Next

1. Auto-score adversarial refusals against `acceptable_responses` / `forbidden_phrases`, gated on PR.
2. Add locator-specific verification per citation type: PDF bbox, image bbox, video timestamp.
3. Run Neo4j in CI; assert both Neo4j and in-memory modes on every PR.
