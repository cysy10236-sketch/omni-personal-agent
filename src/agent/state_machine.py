"""Plan-Execute-Reflect (Audit) finite state machine and fair monolithic ReAct baseline."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import time
from typing import Any
from pydantic import BaseModel, Field

from src.agent.sandbox import (
    AuditVerdict,
    SandboxCheckResult,
    SandboxGuard,
    SandboxVerdict,
    WorkspaceSnapshotAuditor,
)
from src.agent.tools import PersonalToolRegistry, ToolExecutionResult
from src.config import AgentSettings
from src.llm.provider import LLMProviderProtocol, create_llm_provider
from src.memory.hybrid_store import HybridMemoryStore
from src.memory.models import MemorySearchResponse, MemoryTier
from src.observability.tracer import StructuredTracer, new_trace_id
from src.perception.multimodal_parser import MultimodalObservation, MultimodalPerceptionEngine
from src.router.model_router import DynamicModelRouter, ModelTier, RoutingDecision
from src.router.tool_selector import MultilingualToolSelector, ToolSelectionResult


class AgentState(str, Enum):
    """Deterministic finite state machine states for OmniPersonalAgent."""

    INIT = "INIT"
    PERCEIVE = "PERCEIVE"
    RETRIEVE_MEMORY = "RETRIEVE_MEMORY"
    ROUTE = "ROUTE"
    PLAN = "PLAN"
    SANDBOX_CHECK = "SANDBOX_CHECK"
    EXECUTE = "EXECUTE"
    AUDIT_REFLECT = "AUDIT_REFLECT"
    COMMIT_MEMORY = "COMMIT_MEMORY"
    COMPLETED = "COMPLETED"
    HALTED_APPROVAL = "HALTED_APPROVAL"
    BLOCKED_SECURITY = "BLOCKED_SECURITY"
    FAILED = "FAILED"


class AgentRunOutcome(BaseModel):
    """Complete structured trace and telemetry of a single agent run."""

    trace_id: str
    final_state: AgentState
    answer: str
    state_transitions: list[AgentState] = Field(default_factory=list)
    routing_decision: RoutingDecision
    tool_selection: ToolSelectionResult
    memory_response: MemorySearchResponse
    observations: list[MultimodalObservation] = Field(default_factory=list)
    executed_tools: list[ToolExecutionResult] = Field(default_factory=list)
    sandbox_events: list[SandboxCheckResult] = Field(default_factory=list)
    audit_verdict: AuditVerdict | None = None
    rounds_used: int = 1
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_input_tokens: int = 0
    latency_ms: float = 0.0


class OmniPersonalAgent:
    """Multimodal personal agent combining hybrid memory, zero-LLM routing, and audited execution."""

    def __init__(
        self,
        settings: AgentSettings,
        *,
        memory_store: HybridMemoryStore | None = None,
        tool_registry: PersonalToolRegistry | None = None,
        llm_provider: LLMProviderProtocol | None = None,
        tracer: StructuredTracer | None = None,
    ) -> None:
        self.settings = settings
        self.tracer = tracer or StructuredTracer(settings.trace_log_path)
        self.memory = memory_store or HybridMemoryStore(
            db_path=settings.memory_db_path,
            embedding_dim=settings.embedding_dim,
            base_half_life_hours=settings.memory_base_half_life_hours,
            abstention_threshold=settings.memory_abstention_threshold,
        )
        self.perception = MultimodalPerceptionEngine()
        self.tools = tool_registry or PersonalToolRegistry(settings.sandbox_workspace_dir)
        self.tool_selector = MultilingualToolSelector(embedder=self.memory.embedder)
        self.router = DynamicModelRouter(settings)
        self.sandbox = SandboxGuard(settings.sandbox_workspace_dir)
        self.auditor = WorkspaceSnapshotAuditor(settings.sandbox_workspace_dir)
        self.llm = llm_provider or create_llm_provider(settings)
        self._recent_tools: list[str] = []

    async def run(
        self,
        user_query: str,
        *,
        attachments: list[str | Path] | None = None,
        session_id: str = "default",
        approved_tokens: set[str] | None = None,
        persist_episodic_memory: bool = True,
    ) -> AgentRunOutcome:
        t_start = time.perf_counter_ns()
        trace_id = new_trace_id()
        transitions: list[AgentState] = [AgentState.INIT]
        in_tokens = 0
        out_tokens = 0
        cached_tokens = 0
        all_executed_tools: list[ToolExecutionResult] = []
        all_sandbox_events: list[SandboxCheckResult] = []

        # 1. PERCEIVE
        transitions.append(AgentState.PERCEIVE)
        observations: list[MultimodalObservation] = []
        with self.tracer.span(trace_id, "perception", "parse_attachments") as sp:
            for att in attachments or []:
                obs = self.perception.parse_file(att)
                observations.append(obs)
            sp.metadata["attachment_count"] = len(observations)
            sp.metadata["tainted"] = any(o.is_tainted_untrusted for o in observations)

        is_tainted = any(o.is_tainted_untrusted for o in observations)

        # 2. RETRIEVE_MEMORY
        transitions.append(AgentState.RETRIEVE_MEMORY)
        with self.tracer.span(trace_id, "memory", "hybrid_search") as sp:
            mem_resp = self.memory.search(user_query, top_k=4)
            sp.metadata["hits"] = len(mem_resp.hits)
            sp.metadata["abstained"] = mem_resp.abstained

        # 3. ROUTE (Zero-LLM Tool Pruning + Privacy/Complexity Tier Selection)
        transitions.append(AgentState.ROUTE)
        with self.tracer.span(trace_id, "router", "select_tools_and_model") as sp:
            tool_sel = self.tool_selector.select_tools(
                user_query,
                self.tools.all_specs(),
                budget=self.settings.router_tool_budget,
                recently_used=self._recent_tools,
            )
            selected_specs = self.tools.specs_by_names(tool_sel.selected_tool_names)
            routing_dec = self.router.route(
                user_query,
                observations=observations,
                recalled_hits=mem_resp.hits,
                candidate_tool_count=len(selected_specs),
                retry_attempt=0,
            )
            sp.model_id = routing_dec.resolved_model_id
            sp.metadata["tier"] = routing_dec.tier.value
            sp.metadata["selected_tools"] = tool_sel.selected_tool_names
            sp.metadata["schema_savings_ratio"] = tool_sel.token_savings_ratio

        auditor_feedback: str | None = None
        final_answer = ""
        audit_verdict: AuditVerdict | None = None
        max_rounds = 1 + self.settings.agent_max_reflect_retries
        rounds_used = 0

        for round_idx in range(max_rounds):
            rounds_used = round_idx + 1
            if round_idx > 0 and not routing_dec.privacy_triggered:
                routing_dec = self.router.route(
                    user_query,
                    observations=observations,
                    recalled_hits=mem_resp.hits,
                    candidate_tool_count=len(selected_specs),
                    retry_attempt=round_idx,
                )

            transitions.append(AgentState.PLAN)
            pre_snapshot = self.auditor.capture_snapshot()
            messages, _ = self.router.build_kv_cache_preserving_messages(
                user_query,
                recalled_hits=mem_resp.hits,
                observations=observations,
                auditor_feedback=auditor_feedback,
                include_native_images=(not self.settings.use_mock_llm),
            )

            round_tool_records: list[tuple[str, dict[str, Any], bool, str]] = []

            # Bounded inner tool execution loop
            for _step in range(8):
                with self.tracer.span(trace_id, "llm", "plan_or_respond") as sp:
                    completion = await self.llm.complete(
                        messages,
                        model_id=routing_dec.resolved_model_id,
                        tier=routing_dec.tier,
                        tools=selected_specs,
                    )
                    sp.model_id = completion.model_id
                    sp.input_tokens = completion.input_tokens
                    sp.output_tokens = completion.output_tokens
                    sp.cached_input_tokens = completion.cached_input_tokens

                in_tokens += completion.input_tokens
                out_tokens += completion.output_tokens
                cached_tokens += completion.cached_input_tokens

                if not completion.tool_calls:
                    final_answer = completion.content
                    break

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": completion.content or "",
                    "tool_calls": [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in completion.tool_calls
                    ],
                }
                if completion.reasoning_content is not None:
                    assistant_msg["reasoning_content"] = completion.reasoning_content
                messages.append(assistant_msg)

                for tc in completion.tool_calls:
                    transitions.append(AgentState.SANDBOX_CHECK)
                    sbx_res = self.sandbox.evaluate_tool_call(
                        tc.tool_name,
                        tc.arguments,
                        is_context_tainted=is_tainted,
                        approved_tokens=approved_tokens,
                    )
                    all_sandbox_events.append(sbx_res)

                    if sbx_res.verdict == SandboxVerdict.REQUIRE_APPROVAL:
                        transitions.append(AgentState.HALTED_APPROVAL)
                        elapsed = round((time.perf_counter_ns() - t_start) / 1_000_000.0, 3)
                        return AgentRunOutcome(
                            trace_id=trace_id,
                            final_state=AgentState.HALTED_APPROVAL,
                            answer=f"[APPROVAL REQUIRED] {sbx_res.reason}",
                            state_transitions=transitions,
                            routing_decision=routing_dec,
                            tool_selection=tool_sel,
                            memory_response=mem_resp,
                            observations=observations,
                            executed_tools=all_executed_tools,
                            sandbox_events=all_sandbox_events,
                            rounds_used=rounds_used,
                            total_input_tokens=in_tokens,
                            total_output_tokens=out_tokens,
                            total_cached_input_tokens=cached_tokens,
                            latency_ms=elapsed,
                        )

                    if sbx_res.verdict == SandboxVerdict.DENY:
                        # If denial was triggered by indirect prompt injection inside an untrusted attachment,
                        # block the injected call and allow the agent to finish the legitimate user request safely!
                        if is_tainted:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.call_id,
                                    "name": tc.tool_name,
                                    "content": (
                                        f'{{"blocked_by_sandbox": "{sbx_res.rule_id}", '
                                        f'"note": "Untrusted injection ignored; proceed with safe summary only."}}'
                                    ),
                                }
                            )
                            continue

                        self.auditor.restore_snapshot(pre_snapshot)
                        transitions.append(AgentState.BLOCKED_SECURITY)
                        elapsed = round((time.perf_counter_ns() - t_start) / 1_000_000.0, 3)
                        return AgentRunOutcome(
                            trace_id=trace_id,
                            final_state=AgentState.BLOCKED_SECURITY,
                            answer=f"[SECURITY BLOCKED: {sbx_res.rule_id}] {sbx_res.reason}",
                            state_transitions=transitions,
                            routing_decision=routing_dec,
                            tool_selection=tool_sel,
                            memory_response=mem_resp,
                            observations=observations,
                            executed_tools=all_executed_tools,
                            sandbox_events=all_sandbox_events,
                            rounds_used=rounds_used,
                            total_input_tokens=in_tokens,
                            total_output_tokens=out_tokens,
                            total_cached_input_tokens=cached_tokens,
                            latency_ms=elapsed,
                        )

                    # Verdict == ALLOW -> Execute tool with inline transient-error retry
                    transitions.append(AgentState.EXECUTE)
                    with self.tracer.span(trace_id, "executor", f"tool:{tc.tool_name}") as sp:
                        exec_res = self.tools.execute(tc.tool_name, tc.arguments)
                        if not exec_res.ok and exec_res.error_type == "TransientResourceTimeout":
                            exec_res = self.tools.execute(tc.tool_name, tc.arguments)
                        sp.status_code = 200 if exec_res.ok else 500
                        sp.metadata["ok"] = exec_res.ok
                        sp.metadata["error_type"] = exec_res.error_type

                    all_executed_tools.append(exec_res)
                    round_tool_records.append((tc.tool_name, tc.arguments, exec_res.ok, exec_res.output))
                    if exec_res.ok and tc.tool_name not in self._recent_tools:
                        self._recent_tools = (self._recent_tools + [tc.tool_name])[-4:]

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.call_id,
                            "name": tc.tool_name,
                            "content": exec_res.output,
                        }
                    )

            # 4. AUDIT_REFLECT (Read-Only Snapshot & Deliverable Ground-Truth Verification)
            transitions.append(AgentState.AUDIT_REFLECT)
            pre_audit_snap = self.auditor.capture_snapshot()
            audit_verdict = self.auditor.audit_round(
                user_query=user_query,
                final_answer=final_answer,
                executed_tools=round_tool_records,
                pre_round_snapshot=pre_snapshot,
            )
            post_audit_snap = self.auditor.capture_snapshot()
            if self.auditor.snapshot_hashes(pre_audit_snap) != self.auditor.snapshot_hashes(post_audit_snap):
                self.auditor.restore_snapshot(pre_audit_snap)
                audit_verdict.passed = False
                audit_verdict.integrity_status = "violation"
                audit_verdict.feedback_for_planner = "Read-only auditor workspace mutation detected and rolled back."

            if audit_verdict.passed:
                break
            auditor_feedback = audit_verdict.feedback_for_planner

        if audit_verdict is not None and not audit_verdict.passed:
            transitions.append(AgentState.FAILED)
            elapsed = round((time.perf_counter_ns() - t_start) / 1_000_000.0, 3)
            return AgentRunOutcome(
                trace_id=trace_id,
                final_state=AgentState.FAILED,
                answer=final_answer or f"Failed verification: {audit_verdict.missing_requirements}",
                state_transitions=transitions,
                routing_decision=routing_dec,
                tool_selection=tool_sel,
                memory_response=mem_resp,
                observations=observations,
                executed_tools=all_executed_tools,
                sandbox_events=all_sandbox_events,
                audit_verdict=audit_verdict,
                rounds_used=rounds_used,
                total_input_tokens=in_tokens,
                total_output_tokens=out_tokens,
                total_cached_input_tokens=cached_tokens,
                latency_ms=elapsed,
            )

        # 5. COMMIT_MEMORY (Async write-through episodic record)
        transitions.append(AgentState.COMMIT_MEMORY)
        if persist_episodic_memory and not is_tainted:
            summary_line = f"[Session:{session_id}] Q: {user_query[:100]} -> A: {final_answer[:140]}"
            await self.memory.add_memory_async(
                summary_line,
                tier=MemoryTier.EPISODIC,
                session_id=session_id,
                importance=0.45,
                source_provenance="agent_episode",
            )

        transitions.append(AgentState.COMPLETED)
        elapsed = round((time.perf_counter_ns() - t_start) / 1_000_000.0, 3)
        return AgentRunOutcome(
            trace_id=trace_id,
            final_state=AgentState.COMPLETED,
            answer=final_answer,
            state_transitions=transitions,
            routing_decision=routing_dec,
            tool_selection=tool_sel,
            memory_response=mem_resp,
            observations=observations,
            executed_tools=all_executed_tools,
            sandbox_events=all_sandbox_events,
            audit_verdict=audit_verdict,
            rounds_used=rounds_used,
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            total_cached_input_tokens=cached_tokens,
            latency_ms=elapsed,
        )


class MonolithicReActBaseline:
    """Fair unpruned ReAct baseline sharing the same LLM provider, tools, and database.

    Architectural differences vs OmniPersonalAgent:
    - Always invokes `FRONTIER_LLM` (no privacy SLM routing, no fast SLM offload).
    - Injects all 15 tool schemas every turn (no zero-LLM multilingual schema pruning).
    - Uses flat vector+BM25 top-K memory without Adamic-Adar graph expansion, without
      slot-conflict supersession filtering, and without abstention gate.
    - Mutates the system prompt per turn with dynamic timestamps and memories (invalidating KV-Cache).
    - Single-pass tool execution without SandboxGuard taint/opaque-wrapper gates or Auditor reflection.
    """

    def __init__(
        self,
        settings: AgentSettings,
        *,
        memory_store: HybridMemoryStore,
        tool_registry: PersonalToolRegistry,
        llm_provider: LLMProviderProtocol,
    ) -> None:
        self.settings = settings
        self.memory = memory_store
        self.tools = tool_registry
        self.perception = MultimodalPerceptionEngine()
        self.llm = llm_provider

    async def run(
        self,
        user_query: str,
        *,
        attachments: list[str | Path] | None = None,
    ) -> AgentRunOutcome:
        t_start = time.perf_counter_ns()
        trace_id = new_trace_id()
        observations = [self.perception.parse_file(a) for a in (attachments or [])]
        mem_resp = self.memory.search_flat_baseline(user_query, top_k=4)
        all_specs = self.tools.all_specs()
        full_tokens = sum(t.schema_token_cost() for t in all_specs)

        # Dynamic system prompt mutation (standard naive pattern that breaks Prefix KV-Cache)
        mem_xml = "\n".join(
            f'<memory id="{h.record.memory_id}">{h.record.content}</memory>'
            for h in mem_resp.hits
        )
        obs_xml = "\n".join(
            f"<observation><extracted_text>{o.extracted_text}</extracted_text></observation>"
            for o in observations
        )
        mutated_system = (
            f"You are a helpful ReAct assistant. Trace={trace_id}.\n"
            f"<recalled_memories>{mem_xml}</recalled_memories>\n"
        )
        user_payload = (
            f"<recalled_memories>{mem_xml}</recalled_memories>\n"
            f"<multimodal_observations>{obs_xml}</multimodal_observations>\n"
            f"<user_request>{user_query}</user_request>"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": mutated_system},
            {"role": "user", "content": user_payload},
        ]

        in_tokens = 0
        out_tokens = 0
        cached_tokens = 0
        executed: list[ToolExecutionResult] = []
        final_answer = ""

        for _step in range(8):
            comp = await self.llm.complete(
                messages,
                model_id=self.settings.llm_model,
                tier=ModelTier.FRONTIER_LLM,
                tools=all_specs,
            )
            in_tokens += comp.input_tokens
            out_tokens += comp.output_tokens
            cached_tokens += comp.cached_input_tokens

            if not comp.tool_calls:
                final_answer = comp.content
                break

            base_assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": comp.content or "",
                "tool_calls": [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in comp.tool_calls
                ],
            }
            if comp.reasoning_content is not None:
                base_assistant_msg["reasoning_content"] = comp.reasoning_content
            messages.append(base_assistant_msg)
            aborted = False
            for tc in comp.tool_calls:
                res = self.tools.execute(tc.tool_name, tc.arguments)
                executed.append(res)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "name": tc.tool_name,
                        "content": res.output,
                    }
                )
                # Without reflection/retry loop, a transient tool error aborts the chain
                if not res.ok:
                    aborted = True
                    break
            if aborted:
                final_answer = f"Tool execution aborted due to error: {executed[-1].output}"
                break

        elapsed = round((time.perf_counter_ns() - t_start) / 1_000_000.0, 3)
        return AgentRunOutcome(
            trace_id=trace_id,
            final_state=AgentState.COMPLETED if final_answer else AgentState.FAILED,
            answer=final_answer,
            state_transitions=[AgentState.INIT, AgentState.EXECUTE, AgentState.COMPLETED],
            routing_decision=RoutingDecision(
                tier=ModelTier.FRONTIER_LLM,
                resolved_model_id=self.settings.llm_model,
                complexity_score=1.0,
                estimated_hops=1,
                privacy_triggered=False,
                rationale="Baseline always uses static FRONTIER_LLM.",
            ),
            tool_selection=ToolSelectionResult(
                selected_tool_names=[s.name for s in all_specs],
                scores={},
                full_schema_tokens=full_tokens,
                selected_schema_tokens=full_tokens,
                token_savings_ratio=0.0,
            ),
            memory_response=mem_resp,
            observations=observations,
            executed_tools=executed,
            latency_ms=elapsed,
            total_input_tokens=in_tokens,
            total_output_tokens=out_tokens,
            total_cached_input_tokens=cached_tokens,
        )
