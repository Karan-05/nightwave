"""Case loaders. Provided so you don't burn time on plumbing.

Two ways to read the case state:

  - load_case_json(path="../case_data.json")  -> dict
      Returns the full case dump as a Python dict — entities, relationships,
      events, locations, leads, hypotheses, citations, evidence.

  - connect_db(path="../nightwave.db") -> sqlite3.Connection
      Read-only-ish handle to the SQLite file. The schema matches the real
      Nightwave codebase (see ../schema/db_schema.py). 10 tables.

Use whichever feels more natural for the tool you're writing. Most tools
will probably want the JSON for read-side queries; the DB is there if you
want to write SQL or test query performance.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

# Path resolution: walk up from this file to find the package root,
# so callers don't have to think about cwd.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # starter/ -> karan-takehome/

DEFAULT_JSON = _ROOT / "case_data.json"
DEFAULT_DB = _ROOT / "nightwave.db"
DEFAULT_EVIDENCE_DIR = _ROOT / "evidence"


def load_case_json(path: str | Path = DEFAULT_JSON) -> dict[str, Any]:
    """Load the full case dump as a dict.

    Top-level keys: case_id, title, status, summary, evidence, entities,
    relationships, events, locations, leads, hypotheses, citations,
    agent_analysis_metadata.
    """
    return json.loads(Path(path).read_text())


def connect_db(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open the SQLite case file. Tables: entities, relationships, events,
    locations, leads, hypotheses, citations, pipeline_jobs, audit_log,
    schema_version. JSON columns (properties, aliases, conflicts,
    entity_ids, etc.) are stored as TEXT — parse with json.loads().

    The DB is single-case (only the Madison Fields case). Don't filter
    by case_id unless you want to be defensive.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def evidence_path(filename: str) -> Path:
    """Return the absolute path to a raw evidence file by filename."""
    return DEFAULT_EVIDENCE_DIR / filename


def evidence_list() -> list[Path]:
    """Return absolute paths to every raw evidence file in evidence/."""
    return sorted(DEFAULT_EVIDENCE_DIR.iterdir())


# ---- Tiny helpers (use them or write your own) ----

def find_entity(case: dict, name_or_id: str) -> dict | None:
    """Return the entity matching by id or by case-insensitive name. None if absent."""
    needle = name_or_id.lower().strip()
    for e in case.get("entities", []):
        if e["id"] == name_or_id or e.get("name", "").lower().strip() == needle:
            return e
    return None


def citations_for(case: dict, artifact_id: str) -> list[dict]:
    """Return every citation row that points to a given artifact (entity, event, etc.)."""
    return [c for c in case.get("citations", []) if c["artifact_id"] == artifact_id]


def evidence_text(filename: str) -> str:
    """Read a raw text/PDF/etc evidence file as UTF-8. Will raise on binary files —
    that's intentional, you're expected to pick the right tool for the file type."""
    return evidence_path(filename).read_text(encoding="utf-8")
