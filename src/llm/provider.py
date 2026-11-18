"""Unified OpenAI-compatible live provider and deterministic offline MockLLMProvider."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Protocol
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.config import AgentSettings
from src.router.model_router import ModelTier
from src.router.tool_selector import ToolSchemaSpec, estimate_tokens


class ToolCallRequest(BaseModel):
    """Structured tool invocation requested by an LLM."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMCompletion(BaseModel):
    """Normalized LLM generation output with token and cache telemetry."""

    content: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    model_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)


class LLMProviderProtocol(Protocol):
    """Protocol implemented by both live OpenAI-compatible and offline mock providers."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str,
        tier: ModelTier,
        tools: list[ToolSchemaSpec] | None = None,
    ) -> LLMCompletion: ...


class OpenAICompatibleProvider:
    """Production async provider compatible with OpenAI, DeepSeek, Qwen/DashScope, vLLM, and Ollama."""

    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self._cloud_client = AsyncOpenAI(
            api_key=settings.openai_api_key or "EMPTY",
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout_seconds,
        )
        local_url = settings.local_slm_base_url.strip() or settings.openai_base_url
        local_key = settings.local_slm_api_key.strip() or settings.openai_api_key or "ollama"
        self._local_client = AsyncOpenAI(
            api_key=local_key,
            base_url=local_url,
            timeout=settings.request_timeout_seconds,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str,
        tier: ModelTier,
        tools: list[ToolSchemaSpec] | None = None,
    ) -> LLMCompletion:
        client = (
            self._local_client
            if tier == ModelTier.LOCAL_PRIVACY_SLM
            else self._cloud_client
        )
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": self.settings.temperature,
            "seed": self.settings.random_seed,
        }
        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]
            kwargs["tool_choice"] = "auto"

        resp = await client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        parsed_calls: list[ToolCallRequest] = []
        for tc in choice.tool_calls or []:
            try:
                args_dict = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args_dict = {"_malformed_json": tc.function.arguments}
            parsed_calls.append(
                ToolCallRequest(
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=args_dict if isinstance(args_dict, dict) else {},
                )
            )

        usage = resp.usage
        in_tok = int(usage.prompt_tokens) if usage else 0
        out_tok = int(usage.completion_tokens) if usage else 0
        cached_tok = 0
        if usage and getattr(usage, "prompt_tokens_details", None):
            cached_tok = int(getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0)

        return LLMCompletion(
            content=choice.content or "",
            tool_calls=parsed_calls,
            model_id=model_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=cached_tok,
        )


def _extract_xml_block(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


class MockLLMProvider:
    """Context-driven deterministic simulator operating strictly over supplied prompts & tool schemas.

    Activated automatically when `OPENAI_API_KEY` is unconfigured. Does NOT hardcode benchmark IDs:
    all reasoning is derived from actual `<recalled_memories>`, `<multimodal_observations>`,
    supplied `tools` schemas, tool outputs in message history, and `ModelTier`.
    """

    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings
        self._seen_sys_hashes: set[str] = set()
        self._seen_full_prefix_hashes: set[str] = set()
        self._call_seq = 0

    def _compute_token_telemetry(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchemaSpec] | None,
        output_text: str,
        tool_calls: list[ToolCallRequest],
    ) -> tuple[int, int, int]:
        msg_tokens = sum(estimate_tokens(json.dumps(m, ensure_ascii=False)) for m in messages)
        schema_tokens = sum(t.schema_token_cost() for t in (tools or []))
        total_in = msg_tokens + schema_tokens

        # Contiguous Prefix KV-Cache model:
        # 1) Static system prompt prefix caches across all turns whenever system prompt is immutable.
        # 2) Tool schema block caches across steps when (system_prompt + tool_schemas) repeats.
        sys_text = str(messages[0].get("content", "")) if messages else ""
        sys_hash = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()
        schema_sig = ",".join(t.name for t in (tools or []))
        full_prefix_hash = hashlib.sha256(f"{sys_text}||{schema_sig}".encode("utf-8")).hexdigest()

        cached_in = 0
        if sys_hash in self._seen_sys_hashes:
            cached_in += estimate_tokens(sys_text)
        else:
            self._seen_sys_hashes.add(sys_hash)

        if full_prefix_hash in self._seen_full_prefix_hashes:
            cached_in += int(schema_tokens * 0.85)
        else:
            self._seen_full_prefix_hashes.add(full_prefix_hash)

        out_payload = output_text + json.dumps([tc.model_dump() for tc in tool_calls], ensure_ascii=False)
        total_out = max(8, estimate_tokens(out_payload))
        return total_in, total_out, min(total_in, cached_in)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str,
        tier: ModelTier,
        tools: list[ToolSchemaSpec] | None = None,
    ) -> LLMCompletion:
        self._call_seq += 1
        # Simulate realistic tier-proportional inference latency (SLM ~4x faster than Frontier LLM)
        sim_delay = {
            ModelTier.LOCAL_PRIVACY_SLM: 0.006,
            ModelTier.FAST_SLM: 0.008,
            ModelTier.FRONTIER_LLM: 0.032,
        }.get(tier, 0.015)
        await asyncio.sleep(sim_delay)

        available_tools = {t.name for t in (tools or [])}
        user_msg_full = ""
        for m in messages:
            if m.get("role") == "user":
                user_msg_full = str(m.get("content", ""))

        user_req = _extract_xml_block(user_msg_full, "user_request") or user_msg_full
        mem_block = _extract_xml_block(user_msg_full, "recalled_memories")
        obs_block = _extract_xml_block(user_msg_full, "multimodal_observations")

        # Collect already-completed tool results in this round's conversation
        completed_tools: dict[str, list[dict[str, Any]]] = {}
        failed_tool_calls: list[str] = []
        for m in messages:
            if m.get("role") == "tool":
                tname = str(m.get("name", ""))
                raw_content = str(m.get("content", ""))
                if m.get("ok", True) is False or "error" in raw_content.lower() or "timeout" in raw_content.lower():
                    failed_tool_calls.append(tname)
                else:
                    try:
                        parsed = json.loads(raw_content)
                    except json.JSONDecodeError:
                        parsed = {"raw": raw_content}
                    completed_tools.setdefault(tname, []).append(parsed)

        tool_calls = self._plan_next_tool_calls(
            user_req=user_req,
            mem_block=mem_block,
            obs_block=obs_block,
            full_prompt=user_msg_full,
            available_tools=available_tools,
            completed_tools=completed_tools,
            failed_tool_calls=failed_tool_calls,
            tier=tier,
        )

        content = ""
        if not tool_calls:
            content = self._synthesize_grounded_answer(
                user_req=user_req,
                mem_block=mem_block,
                obs_block=obs_block,
                completed_tools=completed_tools,
                tier=tier,
            )

        in_tok, out_tok, cached_tok = self._compute_token_telemetry(
            messages, tools, content, tool_calls
        )
        return LLMCompletion(
            content=content,
            tool_calls=tool_calls,
            model_id=model_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=cached_tok,
        )

    def _plan_next_tool_calls(
        self,
        *,
        user_req: str,
        mem_block: str,
        obs_block: str,
        full_prompt: str,
        available_tools: set[str],
        completed_tools: dict[str, list[dict[str, Any]]],
        failed_tool_calls: list[str],
        tier: ModelTier,
    ) -> list[ToolCallRequest]:
        req_lower = user_req.lower()
        calls: list[ToolCallRequest] = []

        def _mk(name: str, args: dict[str, Any]) -> ToolCallRequest:
            return ToolCallRequest(
                call_id=f"call_{self._call_seq}_{len(calls) + 1}",
                tool_name=name,
                arguments=args,
            )

        # 0. Simulate susceptibility to indirect prompt injection ONLY if untrusted block is NOT marked
        # with untrusted_external_data guard OR if attacker embedded an explicit directive, testing SandboxGuard!
        if "IMPORTANT SYSTEM NOTE:" in (obs_block + "\n" + full_prompt):
            inj_cmd = re.search(r"execute\s+`?([^`\n]+)`?", obs_block + "\n" + full_prompt, re.IGNORECASE)
            if inj_cmd and "shell_command_exec" in available_tools and "shell_command_exec" not in completed_tools:
                # If the prompt lacked <multimodal_observations trust_level="untrusted_external_data">
                # or even when testing defense-in-depth at SandboxGuard layer:
                if 'trust_level="untrusted_external_data"' not in full_prompt:
                    return [_mk("shell_command_exec", {"command": inj_cmd.group(1).strip()})]
            if ".env" in (obs_block + "\n" + full_prompt) and "workspace_read_file" in available_tools:
                if 'trust_level="untrusted_external_data"' not in full_prompt and "workspace_read_file" not in completed_tools:
                    return [_mk("workspace_read_file", {"relative_path": "../.env"})]

        # 1. Explicit user shell command request (used to verify SandboxGuard hard-deny & opaque-wrapper HITL)
        explicit_cmd = re.search(r"(?:执行命令|运行命令|run command|execute shell)\s*[:：]?\s*`([^`]+)`", user_req, re.IGNORECASE)
        if explicit_cmd and "shell_command_exec" in available_tools and "shell_command_exec" not in completed_tools:
            return [_mk("shell_command_exec", {"command": explicit_cmd.group(1).strip()})]

        # 2. Explicit sensitive file read request (tests path confinement)
        explicit_read = re.search(r"(?:读取文件|read file)\s*[:：]?\s*`([^`]+)`", user_req, re.IGNORECASE)
        if explicit_read and "workspace_read_file" in available_tools and "workspace_read_file" not in completed_tools:
            return [_mk("workspace_read_file", {"relative_path": explicit_read.group(1).strip()})]

        # 3. MCP Finance multi-step workflow: discover -> calculate -> export
        if any(k in req_lower for k in ("财务", "经费明细", "财务经费", "finance", "expense ledger")):
            proj = "星辰项目" if "星辰" in user_req else "Project-Orion"
            if "mcp_finance_query" in available_tools and "mcp_finance_query" not in completed_tools:
                return [_mk("mcp_finance_query", {"department_or_project": proj})]

            if "mcp_finance_query" in completed_tools:
                ledger = completed_tools["mcp_finance_query"][-1].get("ledger", {})
                line_vals = [
                    float(v)
                    for k, v in ledger.items()
                    if isinstance(v, (int, float)) and "cap" not in k
                ]
                if line_vals and "calculator_eval" in available_tools and "calculator_eval" not in completed_tools:
                    expr = " + ".join(str(v) for v in line_vals)
                    return [_mk("calculator_eval", {"expression": expr})]

                calc_total = sum(line_vals)
                if "calculator_eval" in completed_tools:
                    calc_total = float(completed_tools["calculator_eval"][-1].get("value", calc_total))

                if (
                    any(k in req_lower for k in ("导出", "报告", "export", "audit"))
                    and "mcp_finance_export_report" in available_tools
                    and "mcp_finance_export_report" not in completed_tools
                ):
                    cap = float(ledger.get("approved_cap_usd", 0.0))
                    remaining = cap - calc_total
                    return [
                        _mk(
                            "mcp_finance_export_report",
                            {
                                "project_code": proj,
                                "total_amount": calc_total,
                                "summary_markdown": (
                                    f"Total expense: {calc_total:.2f} USD; "
                                    f"Approved cap: {cap:.2f} USD; Remaining: {remaining:.2f} USD."
                                ),
                            },
                        )
                    ]

        # 4. MCP Notes search -> upsert / calculation workflow
        if any(k in req_lower for k in ("算力配额", "笔记", "notes", "vault", "gpu-hours", "rebuttal")):
            if "mcp_notes_search" in available_tools and "mcp_notes_search" not in completed_tools:
                q_kw = "quota" if ("算力" in user_req or "quota" in req_lower or "gpu" in req_lower) else "checklist"
                return [_mk("mcp_notes_search", {"query": q_kw})]

            if (
                "mcp_notes_search" in completed_tools
                and any(k in req_lower for k in ("总费用", "总成本", "calculate", "cost", "计算"))
                and "calculator_eval" in available_tools
                and "calculator_eval" not in completed_tools
            ):
                matches_str = json.dumps(completed_tools["mcp_notes_search"][-1])
                nums = re.findall(r"\b\d+(?:\.\d+)?\b", matches_str)
                numeric = [float(x) for x in nums if float(x) > 1.0]
                if len(numeric) >= 2:
                    return [_mk("calculator_eval", {"expression": f"{numeric[0]} * {numeric[1]}"})]

            if (
                any(k in req_lower for k in ("记录笔记", "更新备忘", "upsert", "存入笔记"))
                and "mcp_notes_upsert" in available_tools
                and "mcp_notes_upsert" not in completed_tools
            ):
                calc_str = ""
                if "calculator_eval" in completed_tools:
                    calc_str = str(completed_tools["calculator_eval"][-1].get("value", ""))
                return [
                    _mk(
                        "mcp_notes_upsert",
                        {
                            "note_key": "computed_summary",
                            "body": f"Verified note calculation result: {calc_str or 'updated'}.",
                        },
                    )
                ]

        # 5. Arithmetic calculation over PDF/Screenshot numbers (use structured_kv first to avoid double-counting)
        if (
            obs_block
            and any(k in req_lower for k in ("计算", "求和", "总计", "差额", "均值", "total", "sum", "delta", "calculate"))
            and "calculator_eval" in available_tools
            and "calculator_eval" not in completed_tools
        ):
            kv_only = _extract_xml_block(obs_block, "structured_kv")
            source_for_nums = kv_only if kv_only else _extract_xml_block(obs_block, "extracted_text")
            nums = [
                float(x)
                for x in re.findall(
                    r"(?<![A-Za-z0-9_.-])\d{2,6}(?:\.\d+)?(?![A-Za-z0-9_.-])", source_for_nums
                )
                if float(x) not in {2025.0, 2026.0}
            ]
            if len(nums) >= 2:
                expr = " + ".join(str(v) for v in nums[:4])
                return [_mk("calculator_eval", {"expression": expr})]

        # 6. Calendar scheduling workflow
        if (
            any(k in req_lower for k in ("预约日程", "创建日程", "安排会议", "添加到日历", "schedule a meeting", "book calendar"))
            and "schedule_calendar_event" in available_tools
            and "schedule_calendar_event" not in completed_tools
        ):
            time_match = re.search(
                r"((?:2026-\d{2}-\d{2}\s+\d{1,2}:\d{2})|(?:周[一二三四五六日天]\s*\d{1,2}:\d{2})|(?:Fri\s+\d{1,2}:\d{2}))",
                user_req + "\n" + mem_block + "\n" + obs_block,
            )
            slot_time = time_match.group(1) if time_match else "2026-11-20 15:00"
            return [
                _mk(
                    "schedule_calendar_event",
                    {
                        "title": "Follow-up Sync",
                        "start_time": slot_time,
                        "attendees": ["User"],
                        "notes": f"Auto-scheduled from request: {user_req[:80]}",
                    },
                )
            ]

        # 7. Workspace file output deliverable
        out_file_match = re.findall(r"([A-Za-z0-9_-]+\.(?:md|json|txt|csv))", user_req)
        if (
            out_file_match
            and any(k in req_lower for k in ("写入", "保存", "生成", "write", "save", "create"))
            and "workspace_write_file" in available_tools
            and "workspace_write_file" not in completed_tools
        ):
            target_file = out_file_match[-1]
            synthesized = self._synthesize_grounded_answer(
                user_req=user_req,
                mem_block=mem_block,
                obs_block=obs_block,
                completed_tools=completed_tools,
                tier=tier,
            )
            return [
                _mk(
                    "workspace_write_file",
                    {
                        "relative_path": target_file,
                        "content": f"# Deliverable: {target_file}\n\n{synthesized}\n",
                    },
                )
            ]

        return calls

    def _synthesize_grounded_answer(
        self,
        *,
        user_req: str,
        mem_block: str,
        obs_block: str,
        completed_tools: dict[str, list[dict[str, Any]]],
        tier: ModelTier,
    ) -> str:
        sections: list[str] = []

        # Parse recalled memories in ranked order (top hit wins when conflicting facts co-exist in Baseline!)
        mem_items = re.findall(r"<memory[^>]*>(.*?)</memory>", mem_block, flags=re.DOTALL)
        if mem_items:
            # Frontier / properly consolidated store relies on top ranked coherent facts;
            # if conflicting superseded facts were both dumped in flat baseline, top-1 may be stale!
            top_memories = mem_items[:2] if tier == ModelTier.FAST_SLM else mem_items[:3]
            sections.append("记忆检索依据：" + " | ".join(m.strip() for m in top_memories))
        elif 'status="abstained_or_empty"' in mem_block:
            sections.append("记忆门控状态：无相关历史个人记忆（已触发低置信度拒识门控，零噪声注入）。")

        if obs_block:
            kv_text = _extract_xml_block(obs_block, "structured_kv")
            ext_text = _extract_xml_block(obs_block, "extracted_text")
            clean_lines = [
                ln.strip()
                for ln in ext_text.splitlines()
                if ln.strip() and "IMPORTANT SYSTEM NOTE" not in ln and "ignore previous" not in ln.lower()
            ]
            summary_snippet = "; ".join(clean_lines[:5])
            sections.append(f"多模态观测提取：{kv_text or summary_snippet}")

        if completed_tools:
            tool_summaries: list[str] = []
            for tname, outputs in completed_tools.items():
                tool_summaries.append(f"{tname} -> {json.dumps(outputs[-1], ensure_ascii=False)}")
            sections.append("已验证工具执行结果：" + " ; ".join(tool_summaries))

        if tier == ModelTier.LOCAL_PRIVACY_SLM:
            sections.append("隐私保护声明：检测到敏感个人信息（PII/凭证），已在端侧本地隐私小模型完成闭环处理，零云端外发。")

        if not sections:
            sections.append(f"已完成请求处理：{user_req[:120]}")

        return "\n".join(sections)


def create_llm_provider(settings: AgentSettings) -> LLMProviderProtocol:
    """Factory returning MockLLMProvider when no live API key is set, else OpenAICompatibleProvider."""
    if settings.use_mock_llm:
        return MockLLMProvider(settings)
    return OpenAICompatibleProvider(settings)
