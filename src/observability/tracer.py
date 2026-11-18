"""Structured JSON-Lines observability and span tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any
import uuid
from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    """Single immutable structured JSONL trace record."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    trace_id: str = Field(min_length=8)
    span_id: str = Field(min_length=8)
    parent_span_id: str | None = None
    component: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    model_id: str = "none"
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    status_code: int = Field(default=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


def new_trace_id() -> str:
    """Generate a compact 16-hex-char trace identifier."""
    return uuid.uuid4().hex[:16]


def new_span_id() -> str:
    """Generate a compact 12-hex-char span identifier."""
    return uuid.uuid4().hex[:12]


@dataclass
class SpanTimer:
    """Context helper measuring monotonic wall-clock latency in milliseconds."""

    tracer: "StructuredTracer"
    trace_id: str
    component: str
    event_type: str
    parent_span_id: str | None = None
    span_id: str = field(default_factory=new_span_id)
    model_id: str = "none"
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    status_code: int = 200
    metadata: dict[str, Any] = field(default_factory=dict)
    _start_ns: int = field(default=0, init=False)

    def __enter__(self) -> "SpanTimer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        if exc_type is not None:
            self.status_code = 500
            self.metadata["exception_type"] = exc_type.__name__
            self.metadata["exception_message"] = str(exc_val)
        self.tracer.record(
            TraceEvent(
                trace_id=self.trace_id,
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
                component=self.component,
                event_type=self.event_type,
                model_id=self.model_id,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cached_input_tokens=self.cached_input_tokens,
                latency_ms=round(elapsed_ms, 3),
                status_code=self.status_code,
                metadata=self.metadata,
            )
        )
        return False


class StructuredTracer:
    """Append-only JSONL tracer with in-memory event indexing by trace_id."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        """Persist a validated TraceEvent to disk as single-line JSON and cache in memory."""
        self._events.append(event)
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def span(
        self,
        trace_id: str,
        component: str,
        event_type: str,
        parent_span_id: str | None = None,
        **metadata: Any,
    ) -> SpanTimer:
        """Open a timed tracing span."""
        return SpanTimer(
            tracer=self,
            trace_id=trace_id,
            component=component,
            event_type=event_type,
            parent_span_id=parent_span_id,
            metadata=dict(metadata),
        )

    def events_for_trace(self, trace_id: str) -> list[TraceEvent]:
        """Return all recorded events matching `trace_id` in chronological order."""
        return [ev for ev in self._events if ev.trace_id == trace_id]

    def summary_for_trace(self, trace_id: str) -> dict[str, Any]:
        """Aggregate token counts, latencies, and status codes for a single trace."""
        evs = self.events_for_trace(trace_id)
        return {
            "trace_id": trace_id,
            "span_count": len(evs),
            "total_input_tokens": sum(e.input_tokens for e in evs),
            "total_output_tokens": sum(e.output_tokens for e in evs),
            "total_cached_input_tokens": sum(e.cached_input_tokens for e in evs),
            "total_latency_ms": round(sum(e.latency_ms for e in evs), 3),
            "error_spans": sum(1 for e in evs if e.status_code >= 400),
        }
