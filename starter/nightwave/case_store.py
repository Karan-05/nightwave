"""Case storage boundary for multi-case runtime support.

The take-home ships one JSON case file, but production code should not reach
directly into a module-level singleton for every read. This module introduces a
small store/index abstraction that can be backed by JSON today and by Postgres
or object storage later without changing the agents' read paths.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from nightwave.case import DEFAULT_DB, DEFAULT_EVIDENCE_DIR, DEFAULT_JSON, load_case_json

try:
    from pypdf import PdfReader

    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class CaseStore(Protocol):
    """Storage contract for loading case state by case_id."""

    def load_case(self, case_id: str | None = None) -> dict[str, Any]: ...

    def evidence_dir(self, case_id: str | None = None) -> Path: ...


@dataclass(frozen=True)
class JsonCaseStore:
    """Single-file store used by the take-home runtime."""

    path: Path = DEFAULT_JSON
    evidence_root: Path = DEFAULT_EVIDENCE_DIR

    def load_case(self, case_id: str | None = None) -> dict[str, Any]:
        case = load_case_json(self.path)
        if case_id is not None and case.get("case_id") != case_id:
            raise KeyError(f"Case {case_id!r} is not available in {self.path}")
        return case

    def evidence_dir(self, case_id: str | None = None) -> Path:
        # Single-case filesystem layout; case_id is accepted to preserve the
        # production interface for stores with per-case evidence roots.
        return self.evidence_root


@dataclass
class CaseIndex:
    """Pre-computed read model for one case."""

    store: CaseStore = field(default_factory=JsonCaseStore)
    case_id: str | None = None

    def __post_init__(self) -> None:
        self.case = self.store.load_case(self.case_id)
        self.case_id = str(self.case["case_id"])
        self.evidence_dir = self.store.evidence_dir(self.case_id)

        self.entities_by_id: dict[str, dict] = {}
        self.entities_by_name: dict[str, dict] = {}
        for entity in self.case.get("entities", []):
            self.entities_by_id[entity["id"]] = entity
            self.entities_by_name[entity.get("name", "").lower().strip()] = entity
            for alias in entity.get("aliases", []):
                if alias:
                    self.entities_by_name[alias.lower().strip()] = entity

        self.citations_by_artifact: dict[str, list[dict]] = {}
        for citation in self.case.get("citations", []):
            artifact_id = citation.get("artifact_id", "")
            if artifact_id:
                self.citations_by_artifact.setdefault(artifact_id, []).append(citation)

        self.evidence_by_id: dict[str, dict] = {
            evidence["evidence_id"]: evidence for evidence in self.case.get("evidence", [])
        }
        self.leads: list[dict] = self.case.get("leads", [])
        self.hypotheses: list[dict] = self.case.get("hypotheses", [])
        self.relationships: list[dict] = self.case.get("relationships", [])
        self.events: list[dict] = self.case.get("events", [])

        self.corpus: list[dict[str, Any]] = self._build_corpus()
        self._avg_dl: float = (
            sum(len(tokenize(doc["text"])) for doc in self.corpus) / len(self.corpus)
            if self.corpus
            else 100.0
        )

    def _build_corpus(self) -> list[dict[str, Any]]:
        corpus: list[dict[str, Any]] = []
        for evidence in self.case.get("evidence", []):
            evidence_id = evidence["evidence_id"]
            filename = evidence["filename"]
            path = self.evidence_dir / filename

            if evidence.get("mime_type") == "text/plain" or path.suffix == ".txt":
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    lines = text.splitlines()
                    for i in range(0, len(lines), 5):
                        chunk = " ".join(lines[i : i + 5]).strip()
                        if chunk:
                            corpus.append(
                                {
                                    "evidence_id": evidence_id,
                                    "case_id": self.case_id,
                                    "filename": filename,
                                    "text": chunk,
                                    "locator": {
                                        "type": "text",
                                        "line_start": i + 1,
                                        "line_end": min(i + 5, len(lines)),
                                    },
                                    "source_type": "text",
                                    "is_summary": False,
                                }
                            )
                except Exception:
                    pass

            elif evidence.get("mime_type") == "application/pdf" or path.suffix == ".pdf":
                if _HAS_PYPDF:
                    try:
                        reader = PdfReader(str(path))
                        for page_num, page in enumerate(reader.pages, 1):
                            text = (page.extract_text() or "").strip()
                            if text:
                                corpus.append(
                                    {
                                        "evidence_id": evidence_id,
                                        "case_id": self.case_id,
                                        "filename": filename,
                                        "text": text,
                                        "locator": {"type": "pdf", "page": page_num},
                                        "source_type": "document",
                                        "is_summary": False,
                                    }
                                )
                    except Exception:
                        pass

            summary = f"{evidence.get('description', '')} {evidence.get('ai_summary', '')}".strip()
            if summary:
                corpus.append(
                    {
                        "evidence_id": evidence_id,
                        "case_id": self.case_id,
                        "filename": filename,
                        "text": summary,
                        "locator": {"type": "text", "line_start": 1, "line_end": 1},
                        "source_type": evidence.get("evidence_type", "unknown"),
                        "is_summary": True,
                    }
                )

        return corpus

    def get_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DEFAULT_DB))
        conn.row_factory = sqlite3.Row
        return conn
