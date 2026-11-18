"""Unit tests for HybridMemoryStore, WatermarkConsolidator, Ebbinghaus decay, and Adamic-Adar graph."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.memory.hybrid_store import HybridMemoryStore
from src.memory.models import EntityRelation, MemoryTier
from src.memory.tokenizer import multilingual_tokens


def test_multilingual_tokenizer_handles_cjk_and_english_stems() -> None:
    toks = multilingual_tokens("我已经搬到了上海，updating Project-Orion budgets!")
    assert "上海" in toks
    assert "concept_city" in toks
    assert "concept_budget" in toks
    assert "updat" in toks


def test_slot_conflict_supersession_and_detail_cover_guard(tmp_path: Path) -> None:
    store = HybridMemoryStore(db_path=tmp_path / "mem.sqlite3")
    try:
        r1 = store.add_memory(
            "我住在杭州西湖区。",
            subject="user",
            slot="city",
            slot_value="杭州",
            timestamp=1_000.0,
        )
        r2 = store.add_memory(
            "我已经搬到了上海徐汇区。",
            subject="user",
            slot="city",
            slot_value="上海",
            timestamp=2_000.0,
        )
        # Distinct people living in different cities must NOT be merged or supersede each other
        r3 = store.add_memory(
            "Alice lives in Pune.",
            subject="alice",
            slot="city",
            slot_value="Pune",
            timestamp=2_100.0,
        )
        r4 = store.add_memory(
            "Bob lives in Mumbai.",
            subject="bob",
            slot="city",
            slot_value="Mumbai",
            timestamp=2_200.0,
        )

        assert not store._records[r1.memory_id].is_active
        assert store._records[r1.memory_id].superseded_by == r2.memory_id
        assert store._records[r2.memory_id].is_active
        assert store._records[r3.memory_id].is_active
        assert store._records[r4.memory_id].is_active

        resp = store.search("我现在住在哪个城市？", top_k=3, now_ts=2_500.0)
        assert not resp.abstained
        assert resp.hits[0].record.memory_id == r2.memory_id
        assert all(h.record.memory_id != r1.memory_id for h in resp.hits)
    finally:
        store.close()


def test_ebbinghaus_decay_monotonicity_with_importance_and_access(tmp_path: Path) -> None:
    store = HybridMemoryStore(db_path=tmp_path / "decay.sqlite3", base_half_life_hours=24.0)
    try:
        low_imp = store.add_memory("Temporary scratch memo alpha", importance=0.1, timestamp=0.0)
        high_imp = store.add_memory("Critical architecture invariant alpha", importance=0.95, timestamp=0.0)
        high_imp.access_count = 10

        now_after_7_days = 7 * 24 * 3600.0
        w_low = store._compute_decay_weight(low_imp, now_after_7_days)
        w_high = store._compute_decay_weight(high_imp, now_after_7_days)
        assert 0.0 < w_low < w_high <= 1.0
    finally:
        store.close()


def test_adamic_adar_two_hop_entity_expansion_and_abstention(tmp_path: Path) -> None:
    store = HybridMemoryStore(db_path=tmp_path / "graph.sqlite3")
    try:
        store.add_memory(
            "Nebula-Engine 推理集群由负责人 ZhaoMing 维护。",
            entities=["Nebula-Engine", "ZhaoMing"],
            relations=[EntityRelation(source="Nebula-Engine", relation="owned_by", target="ZhaoMing")],
        )
        hop2 = store.add_memory(
            "ZhaoMing 要求线上服务必须启用 INT4 权重量化与张量并行校验。",
            entities=["ZhaoMing", "INT4-Spec"],
            relations=[EntityRelation(source="ZhaoMing", relation="mandates", target="INT4-Spec")],
        )

        res = store.search("在 Nebula-Engine 上部署服务有什么负责人要求？", top_k=4)
        recalled_ids = [h.record.memory_id for h in res.hits]
        assert hop2.memory_id in recalled_ids

        # Completely unrelated query must trigger abstention gate
        abstain_res = store.search("光合作用中叶绿体卡尔文循环的化学方程式是什么？", top_k=4)
        assert abstain_res.abstained is True
        assert abstain_res.hits == []
    finally:
        store.close()


@pytest.mark.asyncio
async def test_async_write_through_immediate_read_your_writes(tmp_path: Path) -> None:
    store = HybridMemoryStore(db_path=tmp_path / "async_mem.sqlite3")
    try:
        rec = await store.add_memory_async(
            "星云实验室服务器 SSH 跳板端口调整为 22022。",
            tier=MemoryTier.EPISODIC,
            entities=["星云实验室", "22022"],
        )
        # Must be searchable immediately before awaiting flush()
        immediate = store.search("星云实验室跳板端口是多少？", top_k=2)
        assert any(h.record.memory_id == rec.memory_id for h in immediate.hits)
        await store.flush()
    finally:
        await store.aclose()
