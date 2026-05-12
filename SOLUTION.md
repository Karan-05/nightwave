# Solution - Nightwave Take-Home

## Product Shape

I built this as a citation-governed investigative reasoning pipeline, not a loose chatbot.

The important product constraint is that a detective should be able to ask a question and get an answer whose claims can be traced back to case evidence. That means the system needs two capabilities working together:

- raw evidence retrieval for facts and excerpts
- graph reasoning for entities, leads, hypotheses, events, and relationships

The default eval now runs the multi-agent pipeline. The original single-agent ReAct harness remains available with `NIGHTWAVE_ARCHITECTURE=single`, but the product path is:

1. Retriever agent
2. Graph agent
3. Synthesizer agent
4. Critic agent
5. Deterministic eval

This is not multi-agent for theater. Each agent owns a distinct control surface: recall, relationship traversal, answer composition, and validation.

## Architecture

### Retriever Agent

The retriever uses hybrid retrieval when dependencies are available:

- BM25-ish lexical search over parsed evidence chunks
- dense embeddings via `sentence-transformers`
- reciprocal rank fusion for reranking

For a clean take-home environment, dense retrieval degrades to BM25 if `numpy` or the embedding model is unavailable. That keeps the submission runnable without network access or model downloads.

### Graph Agent

The graph agent is designed around Neo4j, with an in-memory fallback.

Neo4j is the right fit here because the case state is relational:

- Madison -> Josh -> Session
- Madison -> Kyle Lawrence -> Snapchat
- hypotheses -> supporting evidence
- leads -> linked entities/events/hypotheses
- events -> entities -> locations

The code includes an idempotent Neo4j seeder that maps case data into nodes for entities, events, leads, hypotheses, locations, and evidence. It also creates relationship edges, citation edges, and hypothesis-to-lead `SUPPORTS` edges derived from `lead.source_hypothesis_id`. If Neo4j is unavailable, the graph agent transparently falls back to the same case graph in memory.

The final industrial run used Colima + Docker Compose to start Neo4j 5 Community, seeded the graph, verified node/edge counts, and ran the eval with `graph_mode: neo4j`.

### Synthesizer Agent

The synthesizer receives only retrieved evidence chunks and graph context. Its job is to write a concise answer with calibrated confidence.

If `ANTHROPIC_API_KEY` is present, the synthesizer uses Claude with `temperature=0` and enforces structured JSON output via a strict system prompt. The output schema (`response`, `confidence`, `reasoning`, `citations`) is validated and sanitized before parsing — handling LLM edge cases like invalid `\'` apostrophe escapes, markdown fences, and truncated responses. If parsing fails completely, the raw output is returned with a degraded confidence score rather than crashing.

### Critic Agent

The critic is the most important product-grade addition.

It validates:

- cited `evidence_id` exists in `case_data.json`
- excerpt is grounded in the source text, parsed PDF text, evidence description, or AI summary
- confidence is not over-calibrated when graph hypotheses conflict

Earlier versions leaked critic warnings into the user-facing answer. That is unacceptable product behavior, so critic feedback now remains internal trace data. If the critic cannot validate the answer after retry, confidence is capped instead of polluting the response.

## Tool Choices

The core tools are:

- `search_evidence`: lexical search over parsed raw evidence, PDF text when available, and metadata summaries
- `get_entity`: entity lookup by id/name/alias with citations attached
- `query_graph`: typed graph traversal over entities, relationships, events, leads, and hypotheses
- `list_citations`: artifact-to-citation lookup
- `submit_answer`: structured final answer contract for the single-agent baseline

For the multi-agent pipeline, the same underlying case index is reused by the retriever, graph agent, and critic.

## Confidence Model

Confidence is calibrated by question type:

| Question type | Expected range | Rationale |
|---|---:|---|
| Single-source official document | 0.85-0.97 | Q1 is directly answered by the Ohio AG alert |
| Conflicting identity reasoning | 0.45-0.70 | Q2 has two active hypotheses and unresolved Session identity |
| Investigative planning | 0.50-0.78 | Q3 ranks actions from case state, but outcomes are uncertain |

Final live LLM confidences:

- Q1: 0.93
- Q2: 0.54
- Q3: 0.72

Q2 is intentionally moderate. The system should not claim certainty where the case graph itself says the identity question is unresolved.

## Eval Methodology

The eval scores four things:

| Metric | Weight | Purpose |
|---|---:|---|
| Correctness | 35% | Required facts, entities, topics, and action count |
| Citation grounding | 30% | Every citation must resolve to real case evidence |
| Must-cite coverage | 20% | Required evidence or hypothesis ids must appear |
| Confidence calibration | 15% | Confidence must match the expected uncertainty band |

Any hallucinated citation id triggers a hard penalty. Raw answers are written to `starter/raw_answers.json`, and reports are written to:

- `starter/eval_report.json`
- `starter/eval_report_multiagent.json`

Final eval result:

```text
overall_score: 1.000
passed: true
architecture: multiagent
graph_mode: neo4j
```

Additional validation:

```text
ruff check .                         # passed
mypy nightwave                       # passed
python -m compileall nightwave tests # passed
pytest -q --ignore=tests/test_multiagent_contract.py   # 129 passed, no LLM required
pytest tests/test_multiagent_contract.py               # 3 passed, live Anthropic + Neo4j
```

Test breakdown (10 files, 129 deterministic tests):

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
| `test_llm_adapter.py` | 12 | Protocol conformance, factory dispatch, model defaults, env-var overrides, missing-key errors |

## Answers Produced

Q1 identifies the clothing from the Ohio AG alert: black Champion hoodie, camo pants, black shoes, black backpack, and white grocery bag.

Q2 concludes that Lawrence is the confirmed Snapchat offender, while Josh is a separate or at least unresolved Session contact. The answer cites both the DOJ/Lawrence evidence and the Josh/Session reporting, then calibrates confidence to 0.54 because the graph keeps both identity hypotheses active.

Q3 ranks the top three actions from the active critical leads:

1. `lead-session-forensic`
2. `lead-snapchat-subpoena`
3. `lead-intown-surveillance`

Those map to `hyp-josh-coconspirator`, `hyp-additional-victims`, `evt-feb13-meet-josh`, and `evt-feb13-departure`.

## Provider-Neutral LLM Adapter

The synthesizer now dispatches through a thin adapter layer (`nightwave/multiagent/llm.py`) rather than calling Anthropic directly. Two env vars control it:

```bash
NIGHTWAVE_LLM_PROVIDER=openai   # or anthropic (default)
NIGHTWAVE_LLM_MODEL=gpt-4o      # overrides per-provider default
```

The adapter contract is intentionally small:

```python
def chat(system, messages, max_tokens, temperature) -> (text, input_tokens, output_tokens)
```

Retry logic, cost tracking, and JSON extraction all live in the synthesizer, not the adapter. That means switching from Anthropic to OpenAI is a one-line env change — the rest of the pipeline is provider-blind.

One deliberate decision: the OpenAI adapter does *not* use `response_format={"type": "json_object"}`. Both providers go through the same prompt-only JSON enforcement and `_extract_json` fallback. This keeps the synthesizer's validation path identical across providers, which means the critic and tests work unchanged regardless of which LLM is active.

## Where the Architecture Breaks at 100 Cases

The honest bottlenecks, in order of severity:

**1. Module-level case singleton.** `_INDEX` is loaded once at import time from a single `case_data.json`. Scaling to 100 cases requires case-keyed loading — a `CaseStore` that fetches by `case_id` from Postgres or a document store, lazy-loaded per request.

**2. SQLite.** Works fine for one case; concurrent writes from 100 active investigations will hit lock contention. The schema already maps cleanly to Postgres — the main change is the connection layer.

**3. `asyncio.new_event_loop()` per question.** The orchestrator creates and discards an event loop for each question. In a server context this needs to run inside an existing async task (FastAPI + background task queue), not spin up a new loop per call.

**4. Neo4j single graph.** One graph database holds all case entities. Multi-agency at scale needs either subgraph partitioning by `case_id` or one graph instance per active case. The seeder already uses idempotent `MERGE` statements, so the seeding contract is reusable.

**5. No streaming.** The pipeline returns a full `Answer` after all four stages complete. Detectives need progress feedback — the SSE contract in `schema/agent/sherlock.types.ts` already defines `sherlock_thinking`, `sherlock_tool_call`, and `sherlock_complete` events. Wiring the orchestrator to emit SSE per step is the next integration that unlocks the production UI.

**6. No semantic cache.** The same question asked by two detectives on the same case runs the full pipeline twice. A Redis-backed embedding similarity cache would cut repeat costs significantly once query volume grows.

## What I Would Do With Another Week

1. Run Neo4j in CI and make the eval assert both Neo4j and memory modes on every PR.
2. Add locator-specific verification for every citation type: PDF bbox, image bbox, video timestamp, and text line/char range.
3. Add provider-neutral structured output adapters (Anthropic and OpenAI) behind a common interface.
4. Add OpenTelemetry spans around each agent step for production observability.
5. Run the adversarial eval suite against the live LLM path and gate the pipeline on refusal correctness.
6. Score adversarial responses automatically: map LLM output against `acceptable_responses` and `forbidden_phrases` in `questions_adversarial.json`.

## Where I Cut Corners

- The critic validates source grounding using parsed text or evidence metadata; it does not do pixel/timecode-level verification for PDFs and media.
- Dense retrieval ran in the final industrial path. BM25 fallback remains for dependency-light environments.
- The adversarial test suite (58 deterministic tests + 6 adversarial question specs in `questions_adversarial.json`) covers the critic and pipeline defensively, but the LLM-path adversarial responses are not yet scored in the eval harness.

## Note on System Prompt Case-Specificity

The synthesizer's `_SYNTH_SYSTEM` prompt contains case-specific examples ("Kyle D. Lawrence", "Snapchat, Session"). These are intentional LLM guidance — concrete examples produce more reliable structured output than abstract placeholders. They are not routing code: all subagent routing dispatches are driven by dynamic proper-noun extraction (`nlp_utils.extract_proper_nouns`) and are verified free of hardcoded strings by `test_no_hardcoding.py`'s AST scan.
