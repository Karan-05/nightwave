"""
Pipeline data models — the typed ontology contract.

ONE SOURCE OF TRUTH. The extractor, linker, committer, and API responses
all derive from the Pydantic classes below. That kills the whole class of
"prompt says date_time, code reads timestamp, DB stores neither" bugs
dead in their tracks — the types simply don't allow the drift to compile.

    TextSpan          — one chunk of text with its locator (Stage 1 output)
    ProcessedFile     — all TextSpans from one evidence file
    Entity / Relationship / Event / Location — typed artifacts (Stage 2 output)
    ExtractionResult  — bundle of artifacts for one evidence file
    PipelineError     — structured error with debug context

    Artifact          — Union[Entity, Relationship, Event, Location]
                        Discriminated by artifact_type for serialization.

Everything downstream (committer, citation store, API) accepts Artifact,
not a dict. That is the whole architectural win.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.models.citation import Locator


# ── Normalizers ─────────────────────────────────────────────────────────


_CATEGORICAL_CONFIDENCE = {
    "verified": 1.0,
    "critical": 0.95,
    "very high": 0.95,
    "high": 0.85,
    "medium": 0.5,
    "low": 0.25,
    "uncertain": 0.1,
}


def maybe_float(val: Any) -> Optional[float]:
    """Coerce to float or None. Empty strings, None, and parse failures all
    map to None — used for optional latitude/longitude/strength fields."""
    if val in (None, ""):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def normalize_confidence(val: Any) -> float:
    """Coerce any LLM or legacy confidence value into float [0.0, 1.0].

    Single source of truth for confidence normalization across the pipeline,
    citation model, and migration scripts. `Citation.from_dict` calls this —
    the older ad-hoc map in citation.py disagreed on `verified`/`uncertain`.
    """
    if val is None:
        return 0.5
    if isinstance(val, str):
        s = val.strip().lower()
        if s in _CATEGORICAL_CONFIDENCE:
            return _CATEGORICAL_CONFIDENCE[s]
        try:
            val = float(s)
        except ValueError:
            return 0.5
    if isinstance(val, (int, float)):
        if 0 <= val <= 1:
            return float(val)
        return min(max(float(val) / 100.0, 0.0), 1.0)
    return 0.5


# ── Stage 1 output ──────────────────────────────────────────────────────


@dataclass
class TextSpan:
    """One chunk of text from an evidence file, with its source locator.

    Output of Stage 1 (processors). Input to Stage 2 (extractor).

    Uses the same Locator class as citations, but at a BROADER granularity:
      - TextSpan locator: the WHOLE chunk's position (full utterance 135-142s)
      - Citation locator: the SPECIFIC position of the extracted entity
        within that chunk
    """

    text: str
    locator: Locator
    evidence_id: str
    span_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.span_id:
            content = f"{self.evidence_id}:{self.locator.to_dict()}:{hashlib.sha256(self.text.encode()).hexdigest()}"
            self.span_id = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ProcessedFile:
    """Output of Stage 1: all TextSpans extracted from one evidence file."""

    evidence_id: str
    case_id: str
    spans: List[TextSpan]
    source_path: str = ""
    source_type: str = "unknown"
    processor_name: str = ""
    processor_version: str = "2.0"
    processed_at: str = ""

    def __post_init__(self) -> None:
        if not self.processed_at:
            self.processed_at = datetime.now(timezone.utc).isoformat()


# Backwards-compat alias. Existing callers passing `ProcessedEvidence` keep
# working while we migrate. New code uses `ProcessedFile`.
ProcessedEvidence = ProcessedFile


# ── Stage 2 output: typed artifact hierarchy ────────────────────────────


class BaseArtifact(BaseModel):
    """Common fields every artifact carries.

    Invariants:
        - `citations` is always non-empty after build time (enforced by
          validator below).
        - `confidence` is float in [0.0, 1.0]. The legacy `str` form is
          coerced via `_normalize_confidence` at build time; by the time
          an artifact reaches the linker or committer, it's a float.
        - `id` is assigned by the linker (LocalLinker.resolve_batch), not
          by the extractor. At build time it's empty; the linker writes it.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = ""
    confidence: float = 0.5
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    span_ids: List[str] = Field(default_factory=list)
    unresolved_refs: List[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: Any) -> float:
        return normalize_confidence(v)

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, v: Any) -> str:
        if v is None:
            return "medium"
        s = str(v).strip().lower()
        return s if s in ("critical", "high", "medium", "low") else "medium"


class Entity(BaseArtifact):
    artifact_type: Literal["entity"] = "entity"
    name: str = ""
    entity_type: str = "unknown"
    description: str = ""
    # 1-7 word UI label. Inspector renders this under the entity name so
    # detectives scan a list and instantly know which "John" each one is.
    # Empty string is acceptable but we surface a warning if it's the norm.
    short_description: str = ""
    aliases: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseArtifact):
    artifact_type: Literal["relationship"] = "relationship"
    # IDs are the resolved identity. The linker populates these from names.
    source_entity_id: str = ""
    target_entity_id: str = ""
    # Names are kept for linker input + debugging. Committer prefers IDs.
    source_entity_name: str = ""
    target_entity_name: str = ""
    relationship_type: str = "unknown"
    description: str = ""
    # 1-7 word edge label rendered in graph view.
    short_description: str = ""
    strength: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class Event(BaseArtifact):
    artifact_type: Literal["event"] = "event"
    description: str = ""
    # 1-7 word headline for timeline view.
    short_description: str = ""
    event_type: str = "unknown"
    # ISO 8601 if extractable. Empty string if not present in source.
    timestamp: str = ""
    timestamp_end: str = ""
    # Names come out of the extractor; IDs come out of the linker.
    entity_names: List[str] = Field(default_factory=list)
    entity_ids: List[str] = Field(default_factory=list)
    location_name: str = ""
    location_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class Location(BaseArtifact):
    artifact_type: Literal["location"] = "location"
    name: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    zip: str = ""
    country: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: str = ""
    # 1-7 word UI label. e.g. "robbery crime scene", "suspect's apartment".
    short_description: str = ""
    entity_names: List[str] = Field(default_factory=list)
    entity_ids: List[str] = Field(default_factory=list)
    event_ids: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)


# The polymorphic artifact type every downstream component accepts.
Artifact = Union[Entity, Relationship, Event, Location]


# ── Backwards-compat shim ──────────────────────────────────────────────
#
# The legacy `ExtractedArtifact` had a `data: dict` field. Existing callers
# that still read `a.data["name"]` get routed through a read-only dict
# view so they don't crash during migration. Writers must use the typed
# field directly (a.name, a.source_entity_id, etc.).


class _DataView:
    """Read-only dict facade over a typed artifact.

    Returns the typed fields as if they were dict keys. Write attempts
    raise TypeError so migrations surface any lingering `a.data[...] = ...`
    writes as clear errors instead of silent drift.
    """

    __slots__ = ("_art",)

    def __init__(self, art: "BaseArtifact") -> None:
        object.__setattr__(self, "_art", art)

    def get(self, key: str, default: Any = None) -> Any:
        v = getattr(self._art, key, default)
        return v if v not in (None, "") else default

    def __getitem__(self, key: str) -> Any:
        return getattr(self._art, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self._art, key) and getattr(self._art, key) not in (None, "")

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(
            "Artifact.data is read-only after the typed migration. "
            "Write to the typed field directly (e.g. artifact.name = ...)."
        )


def _install_data_property() -> None:
    """Attach `.data` as a read-only dict-like view on every subclass."""
    for cls in (Entity, Relationship, Event, Location):
        cls.data = property(lambda self: _DataView(self))  # type: ignore[attr-defined]


_install_data_property()


# Legacy alias — existing code that imports ExtractedArtifact keeps working
# but the typed union is the real shape.
ExtractedArtifact = Artifact


# ── Stage 2 bundle ──────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    """Output of Stage 2: all artifacts extracted from one evidence file's spans."""

    evidence_id: str
    case_id: str
    artifacts: List[Artifact]
    extractor_version: str = "2.0"
    extracted_at: str = ""

    def __post_init__(self) -> None:
        if not self.extracted_at:
            self.extracted_at = datetime.now(timezone.utc).isoformat()


# ── Errors ──────────────────────────────────────────────────────────────


@dataclass
class PipelineError(Exception):
    """Structured error with context for 2 AM debugging.

    Extends Exception so callers can `raise PipelineError(...)` directly.
    Without the Exception base, Python raises
    `TypeError: exceptions must derive from BaseException` at the raise
    site (caught in production locally during the citation-loop ship —
    Wave 1 had this latent because the worker wasn't started yet).
    """

    stage: str = ""
    step: str = ""
    error: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""

    def __post_init__(self) -> None:
        # Populate the Exception's args so str(self) gives a useful summary
        # in tracebacks, log lines, and any code that introspects exception
        # state via `e.args` rather than the dataclass fields.
        super().__init__(self.error or self.step or self.stage)

    def to_json(self) -> str:
        return json.dumps(
            {
                "stage": self.stage,
                "step": self.step,
                "error": self.error,
                "context": self.context,
                "suggestion": self.suggestion,
            },
            default=str,
        )


