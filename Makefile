PYTHON ?= python3
STARTER := starter

.PHONY: setup neo4j-up api-up api-down neo4j-down seed lint typecheck compile test eval reports api-health api-answer api-eval all

setup:
	cd $(STARTER) && $(PYTHON) -m pip install -e ".[dev]"

neo4j-up:
	cd $(STARTER) && docker-compose up -d neo4j

api-up:
	cd $(STARTER) && docker-compose up -d --build neo4j api

api-down:
	cd $(STARTER) && docker-compose stop api

neo4j-down:
	cd $(STARTER) && docker-compose down

seed:
	cd $(STARTER) && NIGHTWAVE_REQUIRE_NEO4J=1 $(PYTHON) -m nightwave.multiagent.seed

lint:
	cd $(STARTER) && $(PYTHON) -m ruff check .

typecheck:
	cd $(STARTER) && $(PYTHON) -m mypy nightwave

compile:
	cd $(STARTER) && $(PYTHON) -m compileall nightwave tests

test:
	cd $(STARTER) && $(PYTHON) -m pytest -q

eval:
	cd $(STARTER) && NIGHTWAVE_REQUIRE_NEO4J=1 NIGHTWAVE_DISABLE_DENSE=1 SYNTH_MODE=deterministic $(PYTHON) -m nightwave.multiagent.run

reports:
	cd $(STARTER) && $(PYTHON) -c "import json; paths=['eval_report.json','eval_report_multiagent.json']; [print(p, (r:=json.load(open(p))).get('passed'), r.get('overall_score'), r.get('architecture'), r.get('graph_mode')) for p in paths]"

api-health:
	curl -fsS http://localhost:8000/health

api-answer:
	curl -fsS -X POST http://localhost:8000/v1/answer \
		-H 'Content-Type: application/json' \
		-d '{"question":"What was Madison Fields wearing when last seen?","question_id":"q1"}'

api-eval:
	curl -fsS -X POST http://localhost:8000/v1/eval

all: lint typecheck compile test seed eval reports
