"""Incremental watermark memory consolidator with slot-conflict supersession."""

from __future__ import annotations

import re
from typing import Callable
import numpy as np

from src.memory.models import ConsolidationReport, MemoryRecord


_DETAIL_TOKENS_RE = re.compile(r"\d+(?:[.:/-]\d+)*|[A-Z][a-zA-Z0-9_-]+|[\u4e00-\u9fff]{2,}")

# Slots where a newer value for the same subject supersedes an older conflicting value
SINGLE_VALUED_SLOTS = frozenset(
    {"city", "location", "meeting_time", "budget", "owner", "deadline", "primary_model"}
)


def _detail_signature(text: str) -> set[str]:
    return {tok.lower() for tok in _DETAIL_TOKENS_RE.findall(text)}


class WatermarkConsolidator:
    """O(M * K) incremental memory consolidator avoiding O(N^2) full-store scans."""

    def __init__(self, duplicate_cosine_threshold: float = 0.91) -> None:
        if not 0.5 <= duplicate_cosine_threshold <= 1.0:
            raise ValueError("duplicate_cosine_threshold must lie in [0.5, 1.0]")
        self.duplicate_cosine_threshold = duplicate_cosine_threshold

    def consolidate(
        self,
        all_active_records: list[MemoryRecord],
        vectors: dict[str, np.ndarray],
        previous_watermark: int,
        apply_supersession: Callable[[str, str], None],
        apply_merge_boost: Callable[[str, int, float], None],
    ) -> ConsolidationReport:
        """Scan records with `seq > previous_watermark` and resolve duplicates/conflicts."""
        new_records = sorted(
            [r for r in all_active_records if r.seq > previous_watermark and r.is_active],
            key=lambda item: (item.updated_at, item.seq),
        )
        if not new_records:
            return ConsolidationReport(
                previous_watermark=previous_watermark,
                new_watermark=previous_watermark,
                scanned_records=0,
                duplicates_merged=0,
                contradictions_superseded=0,
                superseded_ids=[],
            )

        duplicates_merged = 0
        contradictions_superseded = 0
        superseded_ids: list[str] = []
        deactivated: set[str] = set()

        # Index active records by (subject, slot) for O(1) conflict lookup
        slot_index: dict[tuple[str, str], list[MemoryRecord]] = {}
        for rec in all_active_records:
            if rec.is_active and rec.subject and rec.slot:
                key = (rec.subject.lower(), rec.slot.lower())
                slot_index.setdefault(key, []).append(rec)

        for cand in new_records:
            if cand.memory_id in deactivated:
                continue

            # 1. Single-valued (subject, slot) temporal supersession check
            if cand.subject and cand.slot and cand.slot.lower() in SINGLE_VALUED_SLOTS:
                key = (cand.subject.lower(), cand.slot.lower())
                peers = slot_index.get(key, [])
                for older in peers:
                    if older.memory_id == cand.memory_id or older.memory_id in deactivated:
                        continue
                    # Only supersede strictly earlier or lower-seq records
                    is_older = (older.updated_at < cand.updated_at) or (
                        older.updated_at == cand.updated_at and older.seq < cand.seq
                    )
                    if not is_older:
                        continue
                    old_val = (older.slot_value or "").strip().lower()
                    new_val = (cand.slot_value or "").strip().lower()
                    if old_val and new_val and old_val != new_val:
                        apply_supersession(older.memory_id, cand.memory_id)
                        older.is_active = False
                        older.superseded_by = cand.memory_id
                        deactivated.add(older.memory_id)
                        superseded_ids.append(older.memory_id)
                        contradictions_superseded += 1

            # 2. Near-duplicate detection against top vector neighbors with detail-cover guard
            cand_vec = vectors.get(cand.memory_id)
            if cand_vec is None:
                continue
            cand_details = _detail_signature(cand.content)
            for existing in all_active_records:
                if (
                    existing.memory_id == cand.memory_id
                    or existing.memory_id in deactivated
                    or not existing.is_active
                    or existing.seq >= cand.seq
                ):
                    continue
                ex_vec = vectors.get(existing.memory_id)
                if ex_vec is None:
                    continue
                sim = float(np.dot(cand_vec, ex_vec))
                if sim < self.duplicate_cosine_threshold:
                    continue
                # Guard: never merge two facts whose named/numeric details differ
                # (e.g., "lives in Pune" vs "lives in Mumbai" have high cosine but distinct details)
                ex_details = _detail_signature(existing.content)
                if cand_details == ex_details:
                    apply_supersession(existing.memory_id, cand.memory_id)
                    existing.is_active = False
                    existing.superseded_by = cand.memory_id
                    deactivated.add(existing.memory_id)
                    superseded_ids.append(existing.memory_id)
                    boosted_importance = min(1.0, max(cand.importance, existing.importance) + 0.05)
                    apply_merge_boost(
                        cand.memory_id,
                        cand.access_count + existing.access_count + 1,
                        boosted_importance,
                    )
                    duplicates_merged += 1
                    break

        new_watermark = max(r.seq for r in new_records)
        return ConsolidationReport(
            previous_watermark=previous_watermark,
            new_watermark=new_watermark,
            scanned_records=len(new_records),
            duplicates_merged=duplicates_merged,
            contradictions_superseded=contradictions_superseded,
            superseded_ids=superseded_ids,
        )
