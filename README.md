# Nightwave Take-Home Submission

This repo contains a completed citation-governed, multi-agent investigative QA
pipeline for the Madison Fields case. The production-shaped path uses Neo4j for
case relationships, a FastAPI service for HTTP access, deterministic evaluation
for repeatable review, and CI that runs lint, type checks, tests, Neo4j seeding,
and the multi-agent eval.

## Reviewer Quickstart

Prerequisites:

- Python 3.11+
- Docker Compose
- Docker Desktop or Colima

From the repo root:

```bash
make setup
make api-up
make api-health
make api-answer
make api-eval
```

Expected API/eval shape:

```text
GET /health       -> ok=true, graph_mode=neo4j
POST /v1/answer   -> cited answer with real evidence_id values
POST /v1/eval     -> passed=true, overall_score=1.0, graph_mode=neo4j
```

The deterministic review path does not require an LLM key. Live LLM runs can be
enabled with `ANTHROPIC_API_KEY` or another provider configured through the
adapter described in `SOLUTION.md`.

## Local Verification

```bash
make neo4j-up
make seed
make lint
make typecheck
make compile
make test
make eval
```

Current verified status:

- GitHub Actions CI passes on `main`
- `162` deterministic tests pass locally
- Docker/Colima API service and Neo4j run healthy
- Deterministic multi-agent eval passes with `overall_score: 1.000`
- `/v1/answer` rejects blank and whitespace-only questions with `422`

## Key Files

- `RUNBOOK.md` - setup, Docker/API commands, Neo4j and eval operations
- `SOLUTION.md` - architecture, tradeoffs, eval methodology, test coverage
- `starter/nightwave/api.py` - FastAPI HTTP service
- `starter/nightwave/multiagent/` - retriever, graph agent, synthesizer, critic
- `.github/workflows/ci.yml` - CI with Neo4j service and deterministic eval

## HTTP API

```bash
curl -fsS http://localhost:8000/health

curl -fsS -X POST http://localhost:8000/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Madison Fields wearing when last seen?","question_id":"q1"}'

curl -fsS -X POST http://localhost:8000/v1/eval
```

The Docker review path intentionally enables `/v1/eval` via
`NIGHTWAVE_ENABLE_EVAL_ENDPOINT=1`. Outside that path, the endpoint is disabled
by default because it runs the full eval and writes a report.

---

# Original Nightwave Take-Home Brief

Hey Karan,

Great talking with you on Wednesday. The interview was the design conversation — this take-home is the implementation conversation. Goal: I see you write actual code that does the thing we talked about, you get a real feel for what working on the engine of Nightwave looks like.

## The scenario

This package contains a real Nightwave case — the Madison Fields missing-person investigation — exported in the same shape we use in production. You have:

- `case_data.json` — the full case as a single JSON document. Entities, relationships, events, locations, leads, hypotheses, citations, evidence metadata.
- `nightwave.db` — the same data as a SQLite file, schema identical to `services/evidence_pipeline_new/db.py` in our codebase. 10 tables.
- `evidence/` — the 14 raw evidence files (PDFs, txt, images, videos) the pipeline ingested.
- `schema/` — the actual Pydantic models, SQLite schema, and TypeScript interfaces from the codebase.
- `starter/` — minimal Python scaffolding so you don't waste the first hour on plumbing.
- `questions.json` — the three questions your agent has to answer.

A detective opened this case. The pipeline ran. Now they want an agent that reasons over the case state and answers investigative questions with citations. **Build that agent.**

## Your job

Write a small agent harness in Python that answers the three questions in `questions.json`. Then write an eval that scores it.

Concretely:

1. **Implement `starter/nightwave/tools.py`** — at least the four tool stubs we shipped. Add others if you want. Tool implementations are up to you (substring search, BM25, embeddings, hybrid, SQL, NetworkX, anything). Make a real tradeoff and explain it.
2. **Implement `starter/nightwave/agent.py`** — the harness that turns a question into a cited answer with a confidence score and a reasoning trace. Use whatever LLM you want; bring your own key.
3. **Implement `starter/nightwave/eval.py`** — a runner that scores the agent on the three questions per the criteria in `questions.json`. Outputs a JSON report and an exit code.
4. **Write a short `SOLUTION.md`** in the root explaining your design decisions: harness shape (single ReAct loop? multi-agent with reviewer?), tool choices, confidence model, eval methodology, what you'd do with another week, where you cut corners.

That's it. No frontend, no infra, no deployment. Just the engine.

## The three questions (preview)

See `questions.json` for the full schema. Short version:

- **Q1 — retrieval.** What was Madison wearing when last seen, and where does the information come from? *(Single-source factual lookup. Tests citation discipline. Should be high-confidence.)*
- **Q2 — multi-hop reasoning.** Did the same person groom Madison on both Snapchat and Session, or are these two separate offenders? *(Tests reasoning over multiple files plus calibrated uncertainty. Real answer is "probably separate," but the evidence is genuinely ambiguous — confidence should reflect that.)*
- **Q3 — uncertainty + planning.** Top three next investigative actions, ranked by impact, with rationale and citations. *(Tests prioritization + grounding to the case's existing leads/hypotheses without inventing.)*

## Rules

- **4 hours of work.** Spread across as many days as you want, just be honest about how you spent the time. Anything past 6 hours, stop and submit what you have plus a note about what you'd have done next.
- **Due 5 days from receipt.** No bonus for early; no penalty for using the full window.
- **Use any libraries, AI tools, or references you want.** Cursor, Copilot, Claude Code — fair game. Document the prompts/sessions if you used them substantively. Your job is to ship something good, not to prove you can hand-write everything.
- **Bring your own LLM key.** Anthropic, OpenAI, Gemini, Bedrock, Ollama — anything. We'd rather see the harness work end-to-end on whatever you're fastest with than have you fight a provider you've never used.
- **Make the LLM client easy to swap.** In the role you'd be doing this work — provider selection by env var, contract isolated behind a thin adapter. Bonus credit, not required.
- **Citations are non-negotiable.** Every factual claim from the agent must be backed by a citation that resolves to a real evidence_id and locator. Untraceable answers are not answers — that's the rule the production system enforces, and yours should too.
- **Confidence has to be honest.** Don't return 0.95 on Q2. The honest answer is in the 0.5 range and that's the answer we want.
- **Simple beats clever.** If a single ReAct loop with three tools answers all three questions cleanly, that beats a five-agent orchestration with a reviewer subagent. We push back hard on over-engineering — see `CLAUDE.md` in our repo for the full philosophy. *(I'll forgive over-engineering on the call if you can defend it.)*

## What I want back

- A zip / GitHub repo / Vercel-style deploy with the implemented `tools.py`, `agent.py`, `eval.py`, plus `SOLUTION.md`.
- A rendered eval report (`eval_report.json`) showing the agent's answers, scores, and confidence on the three questions.
- Anything else you want to share — alternate designs you considered, a Loom walkthrough, the prompts you used.

## What I'm looking for

In rough order of weight:

1. **Architecture decisions.** Did you make the right tradeoffs on harness shape, tool boundaries, and confidence handling? Can you defend them on the call?
2. **Citation discipline.** Every answer should be traceable to evidence. If the agent returns a fabricated citation, you've shipped a hallucination machine.
3. **Eval design.** Does the eval actually measure what matters — correctness, grounding, calibration — or does it just check that the agent ran without errors?
4. **Code quality.** Clean Python. Async where it matters. No unused abstractions. We have a no-files-over-500-lines rule and we mean it.
5. **Speed of building.** This is the most important attribute we listed in the JD. 4 hours is tight on purpose. The role moves at this pace constantly.
6. **Communication.** `SOLUTION.md` should tell me what you decided and why, in plain words, in 2-3 pages max.

## What I'm specifically not testing

- **Whether you can wire up real CJIS controls or production observability.** Not in 4 hours. We'll talk about that on the role-offer call.
- **Whether you got the "right" answer to Q2 or Q3.** There isn't one. I care about the reasoning chain and the calibration.
- **Whether your eval is exhaustive.** Three questions is fine. Add more if you have time and want to flex.

## Compensation

You're getting paid $500 for this regardless of outcome. If you take the role this folds into your first week. If you don't, you still get paid — your time is worth something either way.

## Submission

Send me a link by **end of day Tuesday**. We'll do a 60-minute review call later that week — you walk me through it live, I ask questions. Be ready for: "what did you cut," "why this harness shape," "show me how confidence flows from tool output through the model into the final answer," and "if I asked you to make this work for 100 cases instead of one, where would the architecture break first?"

Have fun. I'm rooting for you.

— Neel

---

## Appendix — getting started

```bash
# Clone or unzip into a folder
cd karan-takehome

# (Optional) create a venv
python3 -m venv .venv && source .venv/bin/activate

# Install (you'll need to uncomment one LLM SDK in starter/pyproject.toml)
cd starter
pip install -e .[dev]

# Sanity-check that the case loader works
python3 -c "from nightwave.case import load_case_json; c = load_case_json(); print(c['title']); print(f'{len(c[\"entities\"])} entities, {len(c[\"events\"])} events, {len(c[\"citations\"])} citations')"

# Inspect the SQLite directly
sqlite3 ../nightwave.db "SELECT name, entity_type, confidence, priority FROM entities ORDER BY confidence DESC LIMIT 5;"

# Check raw evidence
ls ../evidence

# Read the schema (real source from our codebase)
cat ../schema/db_schema.py
cat ../schema/citation_model.py
cat ../schema/agent/sherlock.types.ts   # SSE / chat contract
```

Once you've got `agent.py` and `eval.py` filled in:

```bash
# Run the eval
cd starter
python3 -m nightwave.eval
# or
python3 run_eval.py    # if you write a top-level entry point
```

## Schema cheat sheet

The full source is in `schema/`. Quick orientation:

- **Entity** — `id, name, entity_type, description, short_description, properties, aliases, confidence, priority, conflicts, created_*, updated_*`
- **Relationship** — `id, source_entity_id, target_entity_id, relationship_type, description, short_description, strength, confidence, priority`
- **Event** — `id, description, short_description, event_type, timestamp, timestamp_end, entity_ids, location_id, confidence, priority`
- **Location** — `id, name, address, city, state, zip, country, latitude, longitude, description, short_description, entity_ids, event_ids, confidence, priority`
- **Lead** — `id, title, description, priority, status, assigned_to, linked_entity_ids, linked_event_ids, linked_location_ids, source_hypothesis_id, action_items, resolution`
- **Hypothesis** — `id, statement, reasoning, confidence, status, supporting_evidence, refuting_evidence, linked_entity_ids, linked_event_ids, linked_location_ids, resolution, resolution_reasoning`
- **Citation** — `id, evidence_id, source_path, type, artifact_id, artifact_type, locator, excerpt, confidence`

Locator types: `text` (line/char range), `pdf` (page + bbox), `audio`/`video` (start/end seconds), `image` (bbox). See `schema/citation_model.py` and the per-type locators inline in our `services/evidence_pipeline_new/processors/` (we'd ship those if it mattered for this take-home; it doesn't).

Citations are the moat. Build around them.
