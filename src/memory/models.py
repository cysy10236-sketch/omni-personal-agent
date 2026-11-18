"""Data models for the hybrid episodic-semantic-graph memory subsystem."""

from __future__ import annotations

from enum import Enum
import time
from typing import Any
from pydantic import BaseModel, Field, field_validator


class MemoryTier(str, Enum):
    """Lifecycle tier of a persisted personal memory item."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class EntityRelation(BaseModel):
    """Typed edge between two canonical entities in the personal knowledge graph."""

    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    memory_id: str = Field(default="")

    @field_validator("source", "target", mode="before")
    @classmethod
    def _normalize_node(cls, value: str) -> str:
        return str(value).strip()


class MemoryRecord(BaseModel):
    """Atomic personal memory record with provenance, decay parameters, and slot metadata."""

    memory_id: str = Field(min_length=4)
    content: str = Field(min_length=1)
    tier: MemoryTier = Field(default=MemoryTier.SEMANTIC)
    session_id: str = Field(default="default")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    entities: list[str] = Field(default_factory=list)
    relations: list[EntityRelation] = Field(default_factory=list)
    subject: str | None = None
    slot: str | None = None
    slot_value: str | None = None
    is_active: bool = True
    superseded_by: str | None = None
    source_Provenance: str = Field(default="user_turn")
    seq: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchHit(BaseModel):
    """Ranked hybrid search hit with channel score decomposition."""

    record: MemoryRecord
    final_score: float
    vector_score: float
    lexical_score: float
    graph_score: float
    decay_weight: float
    matched_via: list[str] = Field(default_factory=list)


class MemorySearchResponse(BaseModel):
    """Result envelope for a memory retrieval operation including abstention status."""

    query: str
    hits: list[MemorySearchHit] = Field(default_factory=list)
    abstained: bool = False
    max_raw_score: float = 0.0
    threshold_used: float = 0.0
    latency_ms: float = 0.0


class ConsolidationReport(BaseModel):
    """Metrics emitted by an incremental watermark memory consolidation pass."""

    previous_watermark: int
    new_watermark: int
    scanned_records: int
    duplicates_merged: int
    contradictions_superseded: int
    superseded_ids: list[str] = Field(default_factory=list)
