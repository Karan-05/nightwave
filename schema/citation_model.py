"""
Nightwave Citation + Locator System

    Citation = WHAT source + WHICH artifact it supports
    Locator  = exactly WHERE within that source the support is found

Core types only. Per-file-type Locators live in their processor files
(services/evidence_pipeline_new/processors/*.py) and auto-register here.
Adding a new source type = 1 processor file with its Locator. That's it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Type


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    """All source types in Nightwave."""
    DOCUMENT = "document"
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    SPREADSHEET = "spreadsheet"
    TEXT = "text"
    LINK = "link"
    WEB_SEARCH = "web_search"
    SOCIAL_MEDIA = "social_media"
    PUBLIC_RECORD = "public_record"
    NEWS_ARTICLE = "news_article"
    DATABASE_QUERY = "database_query"
    AI_ANALYSIS = "ai_analysis"
    TRANSCRIPTION = "transcription"
    OCR = "ocr"
    ARTIFACT = "artifact"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Locator base + registry
# ---------------------------------------------------------------------------

_LOCATOR_REGISTRY: Dict[str, Type["Locator"]] = {}


def register_locator(cls: Type["Locator"]) -> Type["Locator"]:
    """Decorator. Registers a Locator subclass by its `type` field default.

    Use in processor files:
        @register_locator
        @dataclass
        class AudioLocator(Locator):
            type: str = "audio"
            start: Optional[float] = None
            end: Optional[float] = None
    """
    loc_type = cls.__dataclass_fields__["type"].default
    if not loc_type:
        raise ValueError(f"Locator {cls.__name__} must define a 'type' default")
    _LOCATOR_REGISTRY[loc_type] = cls
    return cls


def locator_from_dict(data: dict) -> "Locator":
    """Deserialize a locator dict to the correct typed subclass."""
    if not data:
        return Locator()
    loc_type = data.get("type", "unknown")
    cls = _LOCATOR_REGISTRY.get(loc_type)
    if cls is None:
        return Locator(type=loc_type)
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


@dataclass
class Locator:
    """Base locator. Subclasses add source-specific position fields.

    Subclasses are defined in processor files and registered via @register_locator.
    """
    type: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def is_subset_of(self, other: "Locator") -> bool:
        """Return True if self points to a position contained inside `other`.

        Used by the citation bounds check to verify the LLM's emitted locator
        falls within one of the source-span locators it was given. If not, the
        citation falls back to the broad span locator (no hallucinated
        click-through targets).

        Default: True only if types match. Subclasses override for axis-by-axis
        containment (page/line/timestamp/bbox/cell range).
        """
        return self.type == other.type


# ---------------------------------------------------------------------------
# Citation model
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """Universal citation. Every claim in Nightwave traces to one of these.

    Connects a source (evidence file, URL, or another artifact)
    to an artifact (entity, relationship, event, location, lead, hypothesis).
    The Locator tells exactly WHERE within the source the support is found.

    IDs are always programmatic (uuid4), never LLM-generated.
    """

    # Identity
    id: str = ""

    # Source
    evidence_id: Optional[str] = None
    source_path: str = ""
    type: str = "unknown"               # SourceType value

    # Target artifact
    artifact_id: str = ""
    artifact_type: str = ""             # entity, relationship, event, location, lead, hypothesis

    # Location within source
    locator: Optional[Locator] = None

    # Content
    excerpt: str = ""                   # Short text describing what this citation proves/supports.
                                        # Shown inline in the frontend when hovering over a citation.
                                        # NOT the full source text — just enough to understand
                                        # WHY this citation exists and WHAT it supports.
                                        # Example: "Suspect mentioned at 2:15 discussing the warehouse"
    confidence: float = 0.5             # Float 0.0-1.0. How certain is this citation accurate.
                                        # LLM outputs 0-100, we normalize to 0-1.

    # Scoping
    case_id: str = ""

    # Provenance
    created_by: str = "pipeline"        # pipeline, agent, user
    created_at: str = ""
    updated_by: Optional[str] = None
    updated_at: Optional[str] = None

    # Open metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "evidence_id": self.evidence_id,
            "source_path": self.source_path,
            "type": self.type,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "locator": self.locator.to_dict() if self.locator else None,
            "excerpt": self.excerpt,
            "confidence": self.confidence,
            "case_id": self.case_id,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
            "metadata": self.metadata if self.metadata else None,
        }
        return {k: v for k, v in d.items() if v is not None and v != ""}

    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        locator_data = data.get("locator")
        locator = locator_from_dict(locator_data) if locator_data else None

        # Single source of truth across the pipeline. Avoids the silent
        # mismatch the previous in-place map had with normalize_confidence
        # (e.g. "verified" used to land at 1.0 here but 0.5 elsewhere).
        from services.evidence_pipeline_new.models import normalize_confidence
        conf = normalize_confidence(data.get("confidence", 0.5))

        return cls(
            id=data.get("id", ""),
            evidence_id=data.get("evidence_id"),
            source_path=data.get("source_path", ""),
            type=data.get("type", "unknown"),
            artifact_id=data.get("artifact_id", ""),
            artifact_type=data.get("artifact_type", ""),
            locator=locator,
            excerpt=data.get("excerpt", ""),
            confidence=conf,
            case_id=data.get("case_id", ""),
            created_by=data.get("created_by", "pipeline"),
            created_at=data.get("created_at", ""),
            updated_by=data.get("updated_by"),
            updated_at=data.get("updated_at"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def locator_json(self) -> str:
        """Locator as JSON string for the SQLite column."""
        if self.locator is None:
            return "{}"
        return json.dumps(self.locator.to_dict())

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def format_inline(self) -> str:
        """Compact inline format for agent responses."""
        name = self.source_path.rsplit("/", 1)[-1] if self.source_path else "?"
        short = (self.excerpt[:60] + "...") if len(self.excerpt) > 60 else self.excerpt

        if self.type in ("audio", "video") and self.locator:
            start = getattr(self.locator, "start", None) or ""
            return f'[{name}@{start}s:"{short}"]'
        elif self.type == "document" and self.locator:
            page = getattr(self.locator, "page", None)
            if page:
                return f'[{name}:p{page}:"{short}"]'
        elif self.type == "spreadsheet" and self.locator:
            ref = getattr(self.locator, "cell_ref", None)
            sheet = getattr(self.locator, "sheet", None)
            loc = f"#{sheet}!{ref}" if sheet and ref else ""
            return f'[{name}{loc}:"{short}"]'

        return f'[{name}:"{short}"]'

    def format_full(self) -> str:
        """Full multi-line format for reports."""
        lines = [f"Source: {self.source_path}"]
        if self.type:
            lines.append(f"Type: {self.type}")
        if self.locator:
            loc = self.locator.to_dict()
            loc.pop("type", None)
            if loc:
                lines.append(f"Location: {loc}")
        if self.excerpt:
            lines.append(f'Excerpt: "{self.excerpt}"')
        if self.confidence:
            lines.append(f"Confidence: {self.confidence}")
        return "\n".join(lines)
