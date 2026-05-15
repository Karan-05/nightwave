# Nightwave Submission Runbook

This repo is ready to run locally with Python plus Docker/Colima for Neo4j.

## Prerequisites

- Python 3.11+
- Docker Desktop or Colima
- Docker Compose
- Optional for live LLM eval: `ANTHROPIC_API_KEY`

## One-Command Path

From the repo root:

```bash
make setup
make neo4j-up
make seed
make test
make eval
```

That runs the deterministic, CI-safe path with Neo4j required. It does not need
an LLM key.

## HTTP API

Start the API and Neo4j together:

```bash
make api-up
make api-health
make api-answer
make api-eval
```

Manual curl examples:

```bash
curl -fsS http://localhost:8000/health

curl -fsS -X POST http://localhost:8000/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"What was Madison Fields wearing when last seen?","question_id":"q1"}'

curl -fsS -X POST http://localhost:8000/v1/eval
```

API endpoints:

- `GET /health`
- `POST /v1/answer`
- `POST /v1/eval`

The Docker API path installs the lean `.[api]` extra and disables dense
retrieval by default, so reviewers do not need Torch/model downloads for the
deterministic run. Compose explicitly enables `/v1/eval` with
`NIGHTWAVE_ENABLE_EVAL_ENDPOINT=1`; outside the review path it is disabled by
default because it runs the full eval and writes a report. For local dense
retrieval or cross-encoder entailment:

```bash
cd starter
python -m pip install -e ".[dense]"
```

## Manual Setup

```bash
cd starter
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
docker-compose up -d neo4j
NIGHTWAVE_REQUIRE_NEO4J=1 python -m nightwave.multiagent.seed
pytest -q
NIGHTWAVE_REQUIRE_NEO4J=1 SYNTH_MODE=deterministic python -m nightwave.multiagent.run
```

Expected deterministic eval:

```text
PASSED — overall: 1.000
graph_mode: neo4j
```

## Live LLM Eval

```bash
cd starter
export ANTHROPIC_API_KEY=...
export NIGHTWAVE_REQUIRE_NEO4J=1
python -m nightwave.multiagent.run
```

## Useful Commands

```bash
make lint          # ruff
make typecheck     # mypy
make compile       # compileall
make test          # pytest
make eval          # deterministic multi-agent eval against Neo4j
make api-up        # build/start FastAPI + Neo4j via Docker Compose
make api-health    # curl /health
make api-answer    # curl /v1/answer
make api-eval      # curl /v1/eval
make reports       # print eval report status
make neo4j-down    # stop Neo4j
```

## Architecture Notes

- Multi-agent pipeline: retriever, graph agent, synthesizer, critic.
- Neo4j is required in CI and in `make eval`; memory fallback remains available
  only for local non-required runs.
- Case reads go through `CaseStore` / `CaseIndex`.
- Claim support goes through an entailment judge interface. CI uses deterministic
  lexical entailment; production can enable a cross-encoder NLI judge with:

```bash
export NIGHTWAVE_ENTAILMENT_BACKEND=cross_encoder
export NIGHTWAVE_ENTAILMENT_MODEL=cross-encoder/nli-deberta-v3-base
```

- Observability writes `starter/agent_trace.jsonl` and
  `starter/agent_metrics.json`; both are runtime artifacts and gitignored.
