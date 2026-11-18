"""Multilingual Zero-LLM Tool Budget Pruner with MCP Pair-Preserving Constraint."""

from __future__ import annotations

import json
import math
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from src.memory.tokenizer import DeterministicHashEmbedder, multilingual_tokens


def estimate_tokens(text: str) -> int:
    """Deterministic token count estimator handling Latin words and CJK characters."""
    if not text:
        return 0
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    non_cjk_len = len(text) - cjk_chars
    return max(1, int(math.ceil(cjk_chars * 0.75 + non_cjk_len / 3.8)))


class ToolSchemaSpec(BaseModel):
    """Metadata and OpenAI-compatible function schema for a single agent tool."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    namespace: str = "core"
    mcp_role: str = Field(default="standalone", pattern=r"^(standalone|discover|act)$")
    always_include: bool = False
    keywords_bilingual: list[str] = Field(default_factory=list)

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def schema_token_cost(self) -> int:
        serialized = json.dumps(self.to_openai_tool(), ensure_ascii=False)
        return estimate_tokens(serialized)


class ToolSelectionResult(BaseModel):
    """Output of MultilingualToolSelector including schema token compression metrics."""

    selected_tool_names: list[str]
    scores: dict[str, float]
    full_schema_tokens: int
    selected_schema_tokens: int
    token_savings_ratio: float


class MultilingualToolSelector:
    """Prunes tool schemas for SLM/LLM context efficiency without extra LLM calls."""

    def __init__(self, embedder: DeterministicHashEmbedder | None = None) -> None:
        self.embedder = embedder or DeterministicHashEmbedder(dim=256)
        self._tool_vec_cache: dict[str, np.ndarray] = {}
        self._tool_token_cache: dict[str, set[str]] = {}

    def _index_tool(self, spec: ToolSchemaSpec) -> tuple[np.ndarray, set[str]]:
        if spec.name not in self._tool_vec_cache:
            rich_desc = (
                f"{spec.name.replace('_', ' ')} {spec.namespace} "
                f"{spec.description} {' '.join(spec.keywords_bilingual)}"
            )
            self._tool_vec_cache[spec.name] = self.embedder.embed_one(rich_desc)
            self._tool_token_cache[spec.name] = set(multilingual_tokens(rich_desc))
        return self._tool_vec_cache[spec.name], self._tool_token_cache[spec.name]

    def select_tools(
        self,
        query: str,
        all_tools: list[ToolSchemaSpec],
        *,
        budget: int = 6,
        recently_used: list[str] | None = None,
    ) -> ToolSelectionResult:
        """Select up to `budget` tools with CJK+Latin matching and MCP pair preservation."""
        if budget < 1:
            raise ValueError("Tool budget must be >= 1")
        if not all_tools:
            return ToolSelectionResult(
                selected_tool_names=[],
                scores={},
                full_schema_tokens=0,
                selected_schema_tokens=0,
                token_savings_ratio=0.0,
            )

        full_tokens = sum(t.schema_token_cost() for t in all_tools)
        if len(all_tools) <= budget:
            return ToolSelectionResult(
                selected_tool_names=[t.name for t in all_tools],
                scores={t.name: 1.0 for t in all_tools},
                full_schema_tokens=full_tokens,
                selected_schema_tokens=full_tokens,
                token_savings_ratio=0.0,
            )

        q_vec = self.embedder.embed_one(query)
        q_toks = set(multilingual_tokens(query))
        q_lower = query.lower()
        recent_set = set(recently_used or [])

        scored: dict[str, float] = {}
        by_name = {t.name: t for t in all_tools}
        namespace_groups: dict[str, list[ToolSchemaSpec]] = {}

        for spec in all_tools:
            namespace_groups.setdefault(spec.namespace, []).append(spec)
            t_vec, t_toks = self._index_tool(spec)
            vec_sim = max(0.0, float(np.dot(q_vec, t_vec)))
            overlap = len(q_toks & t_toks) / max(1.0, math.sqrt(len(q_toks) + 1.0))

            # Bilingual explicit keyword & exact tool-name boost
            kw_hits = sum(1 for kw in spec.keywords_bilingual if kw.lower() in q_lower)
            kw_boost = min(0.55, 0.22 * kw_hits)
            exact_boost = 0.65 if spec.name.lower() in q_lower else 0.0
            recent_boost = 0.15 if spec.name in recent_set else 0.0

            total = 0.40 * vec_sim + 0.35 * min(1.0, overlap) + kw_boost + exact_boost + recent_boost
            scored[spec.name] = round(total, 4)

        selected: list[str] = []
        for spec in all_tools:
            if spec.always_include and len(selected) < budget:
                selected.append(spec.name)

        ranked = sorted(all_tools, key=lambda item: scored[item.name], reverse=True)
        for spec in ranked:
            if len(selected) >= budget:
                break
            if spec.name not in selected:
                selected.append(spec.name)

            # Enforce MCP discover <-> act pairing inside the same MCP namespace
            if spec.namespace.startswith("mcp_") and spec.mcp_role in {"discover", "act"}:
                target_role = "act" if spec.mcp_role == "discover" else "discover"
                siblings = [
                    s for s in namespace_groups.get(spec.namespace, []) if s.mcp_role == target_role
                ]
                if siblings and len(selected) < budget:
                    best_sibling = max(siblings, key=lambda s: scored[s.name])
                    if best_sibling.name not in selected:
                        selected.append(best_sibling.name)

        selected_specs = [by_name[name] for name in selected]
        sel_tokens = sum(t.schema_token_cost() for t in selected_specs)
        savings = max(0.0, 1.0 - (sel_tokens / max(1, full_tokens)))

        return ToolSelectionResult(
            selected_tool_names=selected,
            scores=scored,
            full_schema_tokens=full_tokens,
            selected_schema_tokens=sel_tokens,
            token_savings_ratio=round(savings, 4),
        )
