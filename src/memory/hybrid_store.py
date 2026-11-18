"""Hybrid Vector + SQLite FTS5 + NetworkX Adamic-Adar Graph Memory Engine."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid
import networkx as nx
import numpy as np

from src.memory.consolidator import WatermarkConsolidator
from src.memory.models import (
    ConsolidationReport,
    EntityRelation,
    MemoryRecord,
    MemorySearchHit,
    MemorySearchResponse,
    MemoryTier,
)
from src.memory.tokenizer import (
    DeterministicHashEmbedder,
    build_fts5_match_query,
    extract_entities,
    fts5_segmented_text,
    infer_subject_slot_value,
)


class HybridMemoryStore:
    """Persistent hybrid personal memory store with async write-through buffering."""

    def __init__(
        self,
        db_path: Path,
        embedding_dim: int = 256,
        base_half_life_hours: float = 168.0,
        abstention_threshold: float = 0.18,
        alpha_importance: float = 2.0,
        beta_access: float = 0.8,
        weights: tuple[float, float, float] = (0.45, 0.30, 0.25),
    ) -> None:
        if abs(sum(weights) - 1.0) > 1e-5:
            raise ValueError("Channel weights (vector, lexical, graph) must sum to 1.0")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.base_half_life_hours = base_half_life_hours
        self.abstention_threshold = abstention_threshold
        self.alpha_importance = alpha_importance
        self.beta_access = beta_access
        self.lambda_vec, self.lambda_lex, self.lambda_graph = weights

        self.embedder = DeterministicHashEmbedder(dim=embedding_dim)
        self.consolidator = WatermarkConsolidator()

        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

        self._records: dict[str, MemoryRecord] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._entity_graph: nx.Graph = nx.Graph()
        self._entity_to_memories: dict[str, set[str]] = {}
        self._seq_counter: int = 0

        self._write_queue: asyncio.Queue[MemoryRecord | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._dirty_access_ids: set[str] = set()
        self._load_existing_state()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                seq INTEGER NOT NULL,
                content TEXT NOT NULL,
                segmented_text TEXT NOT NULL,
                tier TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                importance REAL NOT NULL,
                access_count INTEGER NOT NULL,
                entities_json TEXT NOT NULL,
                relations_json TEXT NOT NULL,
                subject TEXT,
                slot TEXT,
                slot_value TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                superseded_by TEXT,
                source_provenance TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                vector_blob BLOB NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id UNINDEXED,
                segmented_text
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def _load_existing_state(self) -> None:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT * FROM memories ORDER BY seq ASC").fetchall()
        for row in rows:
            relations = [
                EntityRelation.model_validate(item)
                for item in json.loads(row["relations_json"])
            ]
            rec = MemoryRecord(
                memory_id=row["memory_id"],
                seq=int(row["seq"]),
                content=row["content"],
                tier=MemoryTier(row["tier"]),
                session_id=row["session_id"],
                created_at=float(row["created_at"]),
                updated_at=float(row["updated_at"]),
                importance=float(row["importance"]),
                access_count=int(row["access_count"]),
                entities=json.loads(row["entities_json"]),
                relations=relations,
                subject=row["subject"],
                slot=row["slot"],
                slot_value=row["slot_value"],
                is_active=bool(row["is_active"]),
                superseded_by=row["superseded_by"],
                source_Provenance=row["source_provenance"],
                metadata=json.loads(row["metadata_json"]),
            )
            vec = np.frombuffer(row["vector_blob"], dtype=np.float32).copy()
            self._records[rec.memory_id] = rec
            self._vectors[rec.memory_id] = vec
            self._seq_counter = max(self._seq_counter, rec.seq)
            if rec.is_active:
                self._index_graph_for_record(rec)

    def _get_watermark(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'consolidation_watermark'"
        ).fetchone()
        return int(row["value"]) if row else 0

    def _set_watermark(self, watermark: int) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES('consolidation_watermark', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(watermark),),
        )
        self._conn.commit()

    def _index_graph_for_record(self, rec: MemoryRecord) -> None:
        canonical_entities = [e.strip() for e in rec.entities if e.strip()]
        for ent in canonical_entities:
            norm = ent.lower()
            self._entity_graph.add_node(norm, label=ent)
            self._entity_to_memories.setdefault(norm, set()).add(rec.memory_id)

        # Co-occurrence edges between entities mentioned in the same atomic memory
        for i in range(len(canonical_entities)):
            for j in range(i + 1, len(canonical_entities)):
                u, v = canonical_entities[i].lower(), canonical_entities[j].lower()
                if u != v:
                    self._entity_graph.add_edge(u, v, relation="co_occurs", memory_id=rec.memory_id)

        # Explicit typed entity relations
        for rel in rec.relations:
            u, v = rel.source.lower(), rel.target.lower()
            self._entity_graph.add_node(u, label=rel.source)
            self._entity_graph.add_node(v, label=rel.target)
            self._entity_to_memories.setdefault(u, set()).add(rec.memory_id)
            self._entity_to_memories.setdefault(v, set()).add(rec.memory_id)
            if u != v:
                self._entity_graph.add_edge(
                    u, v, relation=rel.relation, confidence=rel.confidence, memory_id=rec.memory_id
                )

    def _prepare_record(
        self,
        content: str,
        *,
        tier: MemoryTier = MemoryTier.SEMANTIC,
        session_id: str = "default",
        importance: float = 0.5,
        entities: list[str] | None = None,
        relations: list[EntityRelation] | None = None,
        subject: str | None = None,
        slot: str | None = None,
        slot_value: str | None = None,
        timestamp: float | None = None,
        source_provenance: str = "user_turn",
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> tuple[MemoryRecord, np.ndarray]:
        now = timestamp if timestamp is not None else time.time()
        extracted_ents = list(entities) if entities is not None else extract_entities(content)
        if subject is None and slot is None:
            subject, slot, slot_value = infer_subject_slot_value(content)
        if subject and subject.lower() not in [e.lower() for e in extracted_ents]:
            extracted_ents.append(subject)
        if slot_value and len(slot_value) >= 2 and slot_value not in extracted_ents:
            extracted_ents.append(slot_value)

        self._seq_counter += 1
        mid = memory_id or f"mem_{uuid.uuid4().hex[:12]}"
        rec = MemoryRecord(
            memory_id=mid,
            seq=self._seq_counter,
            content=content.strip(),
            tier=tier,
            session_id=session_id,
            created_at=now,
            updated_at=now,
            importance=importance,
            access_count=0,
            entities=extracted_ents,
            relations=list(relations or []),
            subject=subject,
            slot=slot,
            slot_value=slot_value,
            is_active=True,
            superseded_by=None,
            source_Provenance=source_provenance,
            metadata=dict(metadata or {}),
        )
        subject_hint = f"subj_self {rec.subject}" if rec.subject == "user" else (rec.subject or "")
        vec = self.embedder.embed_one(f"{rec.content} {subject_hint}".strip())
        return rec, vec

    def _persist_record_sync(self, rec: MemoryRecord, vec: np.ndarray) -> None:
        subj_tag = "subj_self" if rec.subject == "user" else (rec.subject or "")
        seg = fts5_segmented_text(f"{rec.content} {' '.join(rec.entities)} {subj_tag}")
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO memories (
                memory_id, seq, content, segmented_text, tier, session_id,
                created_at, updated_at, importance, access_count,
                entities_json, relations_json, subject, slot, slot_value,
                is_active, superseded_by, source_provenance, metadata_json, vector_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                content = excluded.content,
                segmented_text = excluded.segmented_text,
                updated_at = excluded.updated_at,
                importance = excluded.importance,
                access_count = excluded.access_count,
                is_active = excluded.is_active,
                superseded_by = excluded.superseded_by
            """,
            (
                rec.memory_id,
                rec.seq,
                rec.content,
                seg,
                rec.tier.value,
                rec.session_id,
                rec.created_at,
                rec.updated_at,
                rec.importance,
                rec.access_count,
                json.dumps(rec.entities, ensure_ascii=False),
                json.dumps([r.model_dump() for r in rec.relations], ensure_ascii=False),
                rec.subject,
                rec.slot,
                rec.slot_value,
                1 if rec.is_active else 0,
                rec.superseded_by,
                rec.source_Provenance,
                json.dumps(rec.metadata, ensure_ascii=False),
                vec.astype(np.float32).tobytes(),
            ),
        )
        cur.execute("DELETE FROM memories_fts WHERE memory_id = ?", (rec.memory_id,))
        cur.execute(
            "INSERT INTO memories_fts(memory_id, segmented_text) VALUES (?, ?)",
            (rec.memory_id, seg),
        )
        self._conn.commit()

    def add_memory(self, content: str, **kwargs: Any) -> MemoryRecord:
        """Synchronous write + immediate incremental consolidation."""
        rec, vec = self._prepare_record(content, **kwargs)
        self._records[rec.memory_id] = rec
        self._vectors[rec.memory_id] = vec
        self._index_graph_for_record(rec)
        self._persist_record_sync(rec, vec)
        self.run_consolidation()
        return rec

    async def add_memory_async(self, content: str, **kwargs: Any) -> MemoryRecord:
        """Async write-through memory insertion: immediately searchable in RAM, persisted via queue."""
        rec, vec = self._prepare_record(content, **kwargs)
        # Immediate write-through into RAM structures for zero-latency read-your-writes
        self._records[rec.memory_id] = rec
        self._vectors[rec.memory_id] = vec
        self._index_graph_for_record(rec)

        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._background_writer_loop())
        await self._write_queue.put(rec)
        return rec

    async def _background_writer_loop(self) -> None:
        while True:
            item = await self._write_queue.get()
            try:
                if item is None:
                    return
                vec = self._vectors[item.memory_id]
                self._persist_record_sync(item, vec)
                self.run_consolidation()
            finally:
                self._write_queue.task_done()

    def _flush_dirty_access_counts(self) -> None:
        if not self._dirty_access_ids:
            return
        rows = [
            (self._records[mid].access_count, mid)
            for mid in self._dirty_access_ids
            if mid in self._records
        ]
        if rows:
            self._conn.executemany(
                "UPDATE memories SET access_count = ? WHERE memory_id = ?",
                rows,
            )
            self._conn.commit()
        self._dirty_access_ids.clear()

    async def flush(self) -> None:
        """Wait until all queued async memory writes and consolidations are committed to SQLite."""
        await self._write_queue.join()
        self._flush_dirty_access_counts()

    async def aclose(self) -> None:
        """Cleanly shut down background writer and close SQLite connection."""
        if self._worker_task is not None and not self._worker_task.done():
            await self._write_queue.put(None)
            await self._write_queue.join()
            await self._worker_task
        self._flush_dirty_access_counts()
        self._conn.close()

    def close(self) -> None:
        self._flush_dirty_access_counts()
        self._conn.close()

    def run_consolidation(self) -> ConsolidationReport:
        """Execute incremental watermark consolidation over newly inserted records."""
        prev_wm = self._get_watermark()
        active = [r for r in self._records.values() if r.is_active]

        def _mark_superseded(old_id: str, new_id: str) -> None:
            self._conn.execute(
                "UPDATE memories SET is_active = 0, superseded_by = ? WHERE memory_id = ?",
                (new_id, old_id),
            )

        def _boost_merged(kept_id: str, new_count: int, new_imp: float) -> None:
            if kept_id in self._records:
                self._records[kept_id].access_count = new_count
                self._records[kept_id].importance = new_imp
            self._conn.execute(
                "UPDATE memories SET access_count = ?, importance = ? WHERE memory_id = ?",
                (new_count, new_imp, kept_id),
            )

        report = self.consolidator.consolidate(
            all_active_records=active,
            vectors=self._vectors,
            previous_watermark=prev_wm,
            apply_supersession=_mark_superseded,
            apply_merge_boost=_boost_merged,
        )
        self._conn.commit()
        if report.new_watermark > prev_wm:
            self._set_watermark(report.new_watermark)
        return report

    def _compute_decay_weight(self, rec: MemoryRecord, now_ts: float) -> float:
        delta_hours = max(0.0, (now_ts - rec.updated_at) / 3600.0)
        effective_half_life = self.base_half_life_hours * (
            1.0
            + self.alpha_importance * rec.importance
            + self.beta_access * math.log1p(rec.access_count)
        )
        return math.exp(-math.log(2.0) * delta_hours / max(1e-3, effective_half_life))

    def _lexical_scores(self, query: str, candidate_ids: set[str]) -> dict[str, float]:
        q_tokens = set(fts5_segmented_text(query).split())
        if not q_tokens:
            return {}

        fts_expr = build_fts5_match_query(query)
        raw_bm25: dict[str, float] = {}
        if fts_expr:
            rows = self._conn.execute(
                "SELECT memory_id, bm25(memories_fts) AS bscore "
                "FROM memories_fts WHERE memories_fts MATCH ? LIMIT 40",
                (fts_expr,),
            ).fetchall()
            for row in rows:
                mid = row["memory_id"]
                if mid in candidate_ids:
                    raw_bm25[mid] = max(1e-4, -float(row["bscore"]))

        # Calibrated saturation score: avoids turning a single accidental weak match into 1.0
        calibrated: dict[str, float] = {}
        denom_cap = max(1.0, min(4.0, float(len(q_tokens))))
        for mid in candidate_ids:
            rec = self._records[mid]
            subj_tag = "subj_self" if rec.subject == "user" else (rec.subject or "")
            doc_tokens = set(
                fts5_segmented_text(f"{rec.content} {' '.join(rec.entities)} {subj_tag}").split()
            )
            overlap_cnt = len(q_tokens & doc_tokens)
            if overlap_cnt == 0 and mid not in raw_bm25:
                continue
            coverage = min(1.0, overlap_cnt / denom_cap)
            bm = raw_bm25.get(mid, float(overlap_cnt) * 0.8)
            sat = bm / (bm + 2.2)
            calibrated[mid] = round(min(1.0, 0.55 * sat + 0.45 * coverage), 4)

        return calibrated

    def _adamic_adar_graph_scores(
        self, query_entities: list[str], seed_memory_ids: list[str], candidate_ids: set[str]
    ) -> dict[str, float]:
        seed_nodes: set[str] = {
            e.lower() for e in query_entities if e.lower() in self._entity_graph
        }
        # Only fall back to top lexical/vector seed entities when the query had no explicit entities
        if not seed_nodes:
            for smid in seed_memory_ids:
                srec = self._records.get(smid)
                if srec:
                    for e in srec.entities:
                        if e.lower() in self._entity_graph:
                            seed_nodes.add(e.lower())

        if not seed_nodes:
            return {}

        raw_graph: dict[str, float] = {}
        for mid in candidate_ids:
            rec = self._records[mid]
            cand_nodes = {e.lower() for e in rec.entities if e.lower() in self._entity_graph}
            if not cand_nodes:
                continue

            overlap_score = len(seed_nodes & cand_nodes) / max(1, len(seed_nodes))
            topo_score = 0.0
            for u in seed_nodes:
                u_nbrs = set(self._entity_graph.neighbors(u))
                for v in cand_nodes - {u}:
                    if v in u_nbrs:
                        topo_score += 1.0
                    # Adamic-Adar over shared neighbors w with deg(w) >= 2
                    v_nbrs = set(self._entity_graph.neighbors(v))
                    for w in u_nbrs & v_nbrs:
                        deg_w = self._entity_graph.degree(w)
                        if deg_w >= 2:
                            topo_score += 1.0 / math.log(1.0 + deg_w)

            combined = 0.55 * overlap_score + 0.45 * topo_score
            if combined > 0.0:
                raw_graph[mid] = combined

        if not raw_graph:
            return {}
        max_g = max(raw_graph.values())
        return {mid: min(1.0, val / max(1e-6, max_g)) for mid, val in raw_graph.items()}

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        now_ts: float | None = None,
        record_access: bool = True,
    ) -> MemorySearchResponse:
        """Execute tri-channel hybrid search with decay weighting and abstention gate."""
        t0 = time.perf_counter_ns()
        now = now_ts if now_ts is not None else time.time()
        active_ids = {mid for mid, rec in self._records.items() if rec.is_active}
        if not query.strip() or not active_ids:
            return MemorySearchResponse(
                query=query,
                hits=[],
                abstained=True,
                max_raw_score=0.0,
                threshold_used=self.abstention_threshold,
                latency_ms=round((time.perf_counter_ns() - t0) / 1_000_000.0, 3),
            )

        q_vec = self.embedder.embed_one(query)
        vec_scores: dict[str, float] = {}
        for mid in active_ids:
            dvec = self._vectors[mid]
            sim = float(np.dot(q_vec, dvec))
            if sim > 0.0:
                vec_scores[mid] = min(1.0, sim)

        lex_scores = self._lexical_scores(query, active_ids)

        # Top lexical + strong vector hits seed 1-hop / Adamic-Adar entity graph expansion
        prelim = sorted(
            active_ids,
            key=lambda m: 0.55 * vec_scores.get(m, 0.0) + 0.45 * lex_scores.get(m, 0.0),
            reverse=True,
        )
        top_seeds = [
            m
            for m in prelim[:4]
            if (vec_scores.get(m, 0.0) >= 0.22 or lex_scores.get(m, 0.0) >= 0.30)
        ]
        q_entities = extract_entities(query)
        graph_scores = self._adamic_adar_graph_scores(q_entities, top_seeds, active_ids)

        scored_hits: list[MemorySearchHit] = []
        max_raw_score = 0.0
        for mid in active_ids:
            rec = self._records[mid]
            v_s = vec_scores.get(mid, 0.0)
            l_s = lex_scores.get(mid, 0.0)
            g_s = graph_scores.get(mid, 0.0)
            # Require at least minimal lexical/vector anchor unless query explicitly named an entity
            if v_s < 0.08 and l_s == 0.0 and not q_entities and not top_seeds:
                continue
            decay = self._compute_decay_weight(rec, now)
            combined = decay * (
                self.lambda_vec * v_s + self.lambda_lex * l_s + self.lambda_graph * g_s
            )
            max_raw_score = max(max_raw_score, combined)
            if combined >= self.abstention_threshold:
                via: list[str] = []
                if v_s >= 0.12:
                    via.append("vector")
                if l_s > 0.0:
                    via.append("fts5_bm25")
                if g_s > 0.0:
                    via.append("adamic_adar_graph")
                scored_hits.append(
                    MemorySearchHit(
                        record=rec,
                        final_score=round(combined, 4),
                        vector_score=round(v_s, 4),
                        lexical_score=round(l_s, 4),
                        graph_score=round(g_s, 4),
                        decay_weight=round(decay, 4),
                        matched_via=via,
                    )
                )

        scored_hits.sort(key=lambda item: item.final_score, reverse=True)
        final_hits = scored_hits[:top_k]

        if record_access and final_hits:
            for hit in final_hits:
                hit.record.access_count += 1
                self._dirty_access_ids.add(hit.record.memory_id)

        elapsed_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 3)
        return MemorySearchResponse(
            query=query,
            hits=final_hits,
            abstained=(len(final_hits) == 0),
            max_raw_score=round(max_raw_score, 4),
            threshold_used=self.abstention_threshold,
            latency_ms=elapsed_ms,
        )

    def search_flat_baseline(self, query: str, *, top_k: int = 5) -> MemorySearchResponse:
        """Standard flat vector+BM25 retrieval without graph expansion, supersession, or abstention."""
        t0 = time.perf_counter_ns()
        all_ids = set(self._records.keys())
        if not query.strip() or not all_ids:
            return MemorySearchResponse(query=query, hits=[], abstained=False)

        q_vec = self.embedder.embed_one(query)
        lex_scores = self._lexical_scores(query, all_ids)
        hits: list[MemorySearchHit] = []
        for mid in all_ids:
            rec = self._records[mid]
            v_s = max(0.0, float(np.dot(q_vec, self._vectors[mid])))
            l_s = lex_scores.get(mid, 0.0)
            score = 0.6 * v_s + 0.4 * l_s
            hits.append(
                MemorySearchHit(
                    record=rec,
                    final_score=round(score, 4),
                    vector_score=round(v_s, 4),
                    lexical_score=round(l_s, 4),
                    graph_score=0.0,
                    decay_weight=1.0,
                    matched_via=["flat_baseline"],
                )
            )
        hits.sort(key=lambda item: item.final_score, reverse=True)
        selected = hits[:top_k]
        return MemorySearchResponse(
            query=query,
            hits=selected,
            abstained=False,
            max_raw_score=selected[0].final_score if selected else 0.0,
            threshold_used=0.0,
            latency_ms=round((time.perf_counter_ns() - t0) / 1_000_000.0, 3),
        )

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._records.values() if r.is_active)

    @property
    def total_count(self) -> int:
        return len(self._records)
