"""Structured observability helpers for multi-agent runs."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

DEFAULT_TRACE_PATH = (
    Path(os.getenv("NIGHTWAVE_TRACE_PATH", ""))
    if os.getenv("NIGHTWAVE_TRACE_PATH")
    else Path(__file__).resolve().parent.parent.parent / "agent_trace.jsonl"
)
DEFAULT_METRICS_PATH = DEFAULT_TRACE_PATH.with_name("agent_metrics.json")
_WRITE_LOCK = Lock()


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs_total": 0, "critic_failures_total": 0, "stage_timings_s": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs_total": 0, "critic_failures_total": 0, "stage_timings_s": {}}


def _update_metrics(path: Path, payload: dict[str, Any]) -> None:
    metrics = _load_metrics(path)
    metrics["runs_total"] = int(metrics.get("runs_total", 0)) + 1
    if payload.get("critic_passed") is False:
        metrics["critic_failures_total"] = int(metrics.get("critic_failures_total", 0)) + 1

    stage_totals = metrics.setdefault("stage_timings_s", {})
    for stage, elapsed in payload.get("stage_timings_s", {}).items():
        current = stage_totals.setdefault(stage, {"count": 0, "total": 0.0, "max": 0.0})
        current["count"] += 1
        current["total"] = round(float(current["total"]) + float(elapsed), 4)
        current["max"] = max(float(current["max"]), float(elapsed))
        current["avg"] = round(float(current["total"]) / current["count"], 4)

    path.write_text(json.dumps(metrics, sort_keys=True, indent=2), encoding="utf-8")


@dataclass
class RunObserver:
    """Collects stage timings and writes one JSONL record per question run."""

    question_id: str
    case_id: str
    trace_path: Path = DEFAULT_TRACE_PATH
    metrics_path: Path = DEFAULT_METRICS_PATH
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    stage_timings: dict[str, float] = field(default_factory=dict)

    def time_stage(self, name: str):
        observer = self

        class _Timer:
            def __enter__(self) -> None:
                self._started_at = time.time()

            def __exit__(self, exc_type, exc, tb) -> None:
                observer.stage_timings[name] = round(time.time() - self._started_at, 4)

        return _Timer()

    def trace_event(self, step: str, **detail: Any) -> dict[str, Any]:
        return {
            "step": step,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "question_id": self.question_id,
            **detail,
        }

    def write_run_record(self, record: dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "question_id": self.question_id,
            "elapsed_s": round(time.time() - self.started_at, 4),
            "stage_timings_s": self.stage_timings,
            **record,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
            _update_metrics(self.metrics_path, payload)
