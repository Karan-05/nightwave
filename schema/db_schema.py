"""
Unified Nightwave database — one nightwave.db for everything.

Tables:
  entities, relationships, events, locations  (core ontology)
  leads, hypotheses                           (agent/user created)
  citations                                   (Citation + Locator model)
  pipeline_jobs                               (persistent job queue)
  audit_log                                   (append-only, CJIS compliance)

Auto-creates on first connection. WAL mode. case_id indexes on ALL tables.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2  # v2: short_description on entities/relationships/events/locations

_DEFAULT_DB_PATH = os.getenv(
    "NIGHTWAVE_DB_PATH",
    str(Path("data") / "nightwave.db"),
)

_SCHEMA = """
-- =====================================================================
-- CORE ONTOLOGY (pipeline extracts, agent/user can edit)
-- =====================================================================

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    name TEXT DEFAULT '',
    entity_type TEXT DEFAULT 'unknown',
    description TEXT DEFAULT '',
    short_description TEXT DEFAULT '',
    properties TEXT DEFAULT '{}',
    aliases TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0.5,
    priority TEXT DEFAULT 'medium',
    conflicts TEXT DEFAULT '[]',
    created_by TEXT DEFAULT 'pipeline',
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_case ON entities(case_id);
-- Semantic identity: one row per (case, canonical name). Enforces the
-- "identity is not accidental" principle — duplicates become DB-impossible.
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_case_name
  ON entities(case_id, LOWER(TRIM(name)));

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    source_entity_id TEXT DEFAULT '',
    target_entity_id TEXT DEFAULT '',
    relationship_type TEXT DEFAULT 'unknown',
    description TEXT DEFAULT '',
    short_description TEXT DEFAULT '',
    strength REAL DEFAULT 1.0,
    properties TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.5,
    priority TEXT DEFAULT 'medium',
    conflicts TEXT DEFAULT '[]',
    created_by TEXT DEFAULT 'pipeline',
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_relationships_case ON relationships(case_id);
-- Identity: one relationship per (case, src, tgt, type). Two files extracting
-- "Alice knows Bob" produce one row with unioned citations, not two.
CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_identity
  ON relationships(case_id, source_entity_id, target_entity_id, relationship_type);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    description TEXT DEFAULT '',
    short_description TEXT DEFAULT '',
    event_type TEXT DEFAULT 'unknown',
    timestamp TEXT DEFAULT '',
    timestamp_end TEXT DEFAULT '',
    entity_ids TEXT DEFAULT '[]',
    location_id TEXT,
    properties TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.5,
    priority TEXT DEFAULT 'medium',
    conflicts TEXT DEFAULT '[]',
    created_by TEXT DEFAULT 'pipeline',
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_case ON events(case_id);
-- Event identity: (case, description-prefix, timestamp). "Meeting at the
-- warehouse 11pm" in three files collapses to one event row.
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_identity
  ON events(case_id, LOWER(SUBSTR(description, 1, 80)), timestamp);

CREATE TABLE IF NOT EXISTS locations (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    name TEXT DEFAULT '',
    address TEXT DEFAULT '',
    city TEXT DEFAULT '',
    state TEXT DEFAULT '',
    zip TEXT DEFAULT '',
    country TEXT DEFAULT '',
    latitude REAL,
    longitude REAL,
    description TEXT DEFAULT '',
    short_description TEXT DEFAULT '',
    entity_ids TEXT DEFAULT '[]',
    event_ids TEXT DEFAULT '[]',
    properties TEXT DEFAULT '{}',
    confidence REAL DEFAULT 0.5,
    priority TEXT DEFAULT 'medium',
    conflicts TEXT DEFAULT '[]',
    created_by TEXT DEFAULT 'pipeline',
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_locations_case ON locations(case_id);
-- Location identity: (case, canonical name). Warehouse on 5th Ave is one place.
CREATE UNIQUE INDEX IF NOT EXISTS idx_locations_case_name
  ON locations(case_id, LOWER(TRIM(name)));

-- =====================================================================
-- AGENT/USER ARTIFACTS (leads, hypotheses)
-- =====================================================================

CREATE TABLE IF NOT EXISTS leads (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    title TEXT DEFAULT '',
    description TEXT DEFAULT '',
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'new',
    assigned_to TEXT DEFAULT '',
    linked_entity_ids TEXT DEFAULT '[]',
    linked_event_ids TEXT DEFAULT '[]',
    linked_location_ids TEXT DEFAULT '[]',
    source_hypothesis_id TEXT DEFAULT '',
    action_items TEXT DEFAULT '[]',
    resolution TEXT DEFAULT '',
    properties TEXT DEFAULT '{}',
    created_by TEXT DEFAULT 'agent',
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_leads_case ON leads(case_id);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    statement TEXT DEFAULT '',
    reasoning TEXT DEFAULT '',
    confidence REAL DEFAULT 0.5,
    status TEXT DEFAULT 'active',
    supporting_evidence TEXT DEFAULT '[]',
    refuting_evidence TEXT DEFAULT '[]',
    linked_entity_ids TEXT DEFAULT '[]',
    linked_event_ids TEXT DEFAULT '[]',
    linked_location_ids TEXT DEFAULT '[]',
    resolution TEXT DEFAULT '',
    resolution_reasoning TEXT DEFAULT '',
    properties TEXT DEFAULT '{}',
    created_by TEXT DEFAULT 'agent',
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_hypotheses_case ON hypotheses(case_id);

-- =====================================================================
-- CITATIONS (Citation + Locator model)
-- =====================================================================

CREATE TABLE IF NOT EXISTS citations (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    evidence_id TEXT,
    source_path TEXT DEFAULT '',
    type TEXT NOT NULL DEFAULT 'unknown',
    artifact_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    locator TEXT NOT NULL DEFAULT '{}',
    excerpt TEXT DEFAULT '',
    confidence REAL DEFAULT 0.5,
    priority TEXT DEFAULT 'medium',
    created_by TEXT DEFAULT 'pipeline',
    created_at TEXT NOT NULL,
    updated_by TEXT,
    updated_at TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_citations_case ON citations(case_id);
CREATE INDEX IF NOT EXISTS idx_citations_artifact ON citations(artifact_id);
CREATE INDEX IF NOT EXISTS idx_citations_evidence ON citations(evidence_id);
-- Citation identity: one citation per (artifact, locator JSON). Prevents
-- duplicate citation rows when the same artifact gets linked from multiple
-- extractions of the same source position.
CREATE UNIQUE INDEX IF NOT EXISTS idx_citations_identity
  ON citations(artifact_id, locator);

-- =====================================================================
-- PIPELINE JOBS (persistent queue — survives browser close, restart, deploy)
--
-- Lifecycle: queued → stage1 → stage2 → completed | failed
--
-- Persistence guarantee: the moment a detective uploads evidence,
-- the job row is written to disk. Processing continues server-side
-- regardless of the detective's browser/device state. When they log
-- back in (even on a different device), they query this table to see
-- what's processing, what's done, and what failed.
--
-- Crash recovery: on startup, the worker scans for stuck jobs:
--   status='stage1' → reset to 'queued' (stage 1 is idempotent)
--   status='stage2' + checkpoint exists → resume from checkpoint
--   status='stage2' + no checkpoint → reset to 'queued'
--
-- Idempotency: (case_id, content_hash) is unique. Re-uploading the
-- same file returns the existing job, never reprocesses.
-- =====================================================================

CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    evidence_id TEXT NOT NULL,
    filename TEXT DEFAULT '',
    file_size INTEGER DEFAULT 0,
    content_type TEXT DEFAULT '',
    content_hash TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued',
    progress REAL DEFAULT 0.0,
    current_step TEXT DEFAULT '',
    s3_key TEXT DEFAULT '',
    checkpoint_key TEXT,
    error TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    pipeline_version TEXT NOT NULL DEFAULT '2.0',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON pipeline_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_case ON pipeline_jobs(case_id);
CREATE INDEX IF NOT EXISTS idx_jobs_evidence ON pipeline_jobs(evidence_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotent ON pipeline_jobs(case_id, content_hash)
    WHERE content_hash != '' AND status != 'failed';

-- =====================================================================
-- AUDIT LOG (append-only, CJIS compliance)
--
-- Every pipeline operation, every agent action, every user edit,
-- every citation access writes here. 12-month retention minimum.
--
-- Actions include:
--   evidence_uploaded    — detective uploaded a file
--   job_created          — pipeline job created
--   stage1_started       — processing started
--   stage1_completed     — text extraction done
--   stage2_started       — LLM extraction started
--   stage2_completed     — entities/citations committed
--   extraction_decision  — LLM decided to create/assign an entity (debug trail)
--   entity_merged        — dedup merged two entities
--   citation_created     — citation linked to artifact
--   artifact_created     — agent or user created an artifact
--   artifact_updated     — agent or user modified an artifact
--   artifact_deleted     — soft delete
--   session_started      — detective opened a case
--   session_restored     — detective returned to a case
--
-- The triggers below prevent UPDATE and DELETE on this table.
-- Application code uses INSERT only. Corrections use new rows, not mutations.
-- =====================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    case_id TEXT NOT NULL,
    evidence_id TEXT,
    artifact_id TEXT,
    citation_id TEXT,
    details TEXT DEFAULT '{}',
    request_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_log(case_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);

-- Immutability: no UPDATE or DELETE allowed on audit_log. Ever.
CREATE TRIGGER IF NOT EXISTS audit_log_no_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: UPDATE not allowed');
END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only: DELETE not allowed');
END;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
"""


def init_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Initialize the unified nightwave.db. Auto-creates tables if missing.

    Call once on startup. Returns a connection with WAL mode enabled.
    Safe to call multiple times (CREATE IF NOT EXISTS).
    """
    path = db_path or _DEFAULT_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")

    conn.executescript(_SCHEMA)

    # Schema version check + migration
    try:
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        current = row["version"] if row else 0
    except sqlite3.OperationalError:
        current = 0

    if current < SCHEMA_VERSION:
        _migrate(conn, current, SCHEMA_VERSION)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()

    logger.info("nightwave.db ready at %s (schema v%d, WAL mode)", path, SCHEMA_VERSION)
    return conn


def get_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a connection to nightwave.db for short-lived operations."""
    path = db_path or _DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _migrate(conn: sqlite3.Connection, from_version: int, to_version: int) -> None:
    """Run schema migrations. Add ALTER TABLE statements as schema evolves."""
    logger.info("Migrating nightwave.db v%d → v%d", from_version, to_version)
    if from_version < 2:
        # v2: short_description column on entities/relationships/events/locations.
        # Inspector renders this as the 1-7 word UI label so a list of names
        # reads like a brief instead of a roster. Idempotent ALTERs (catch
        # OperationalError if a partial migration already added the column).
        for table in ("entities", "relationships", "events", "locations"):
            try:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN short_description TEXT DEFAULT ''"
                )
                logger.info("  added short_description to %s", table)
            except sqlite3.OperationalError as exc:
                if "duplicate column" in str(exc).lower():
                    logger.info("  short_description already on %s — skipping", table)
                else:
                    raise


def log_audit(
    conn: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    case_id: str,
    evidence_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    citation_id: Optional[str] = None,
    details: Optional[dict] = None,
    request_id: Optional[str] = None,
) -> None:
    """Append to the audit log. Never fails silently — raises on error."""
    conn.execute(
        """INSERT INTO audit_log
           (actor, action, case_id, evidence_id, artifact_id, citation_id, details, request_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            actor,
            action,
            case_id,
            evidence_id,
            artifact_id,
            citation_id,
            json.dumps(details or {}, default=str),
            request_id,
        ),
    )
    conn.commit()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
