"""CLI entrypoint, end-to-end interactive demo, and optional FastAPI service for OmniPersonalAgent."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from src.agent.state_machine import OmniPersonalAgent
from src.config import PROJECT_ROOT, AgentSettings, load_settings
from src.memory.models import EntityRelation, MemoryTier
from src.perception.multimodal_parser import build_synthetic_screenshot, build_valid_pdf


class AgentRunRequest(BaseModel):
    query: str = Field(min_length=1)
    attachments: list[str] = Field(default_factory=list)
    session_id: str = "api_session"


def create_fastapi_app(settings: AgentSettings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    app = FastAPI(title="Omni-Personal-Agent API", version="0.1.0")
    agent = OmniPersonalAgent(cfg)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mock_mode": cfg.use_mock_llm,
            "active_memories": agent.memory.active_count,
            "router_stats": agent.router.stats(),
        }

    @app.post("/v1/agent/run")
    async def run_agent(req: AgentRunRequest) -> dict[str, Any]:
        outcome = await agent.run(
            req.query,
            attachments=req.attachments,
            session_id=req.session_id,
        )
        await agent.memory.flush()
        return outcome.model_dump(mode="json")

    @app.get("/v1/memory/search")
    async def search_memory(q: str, top_k: int = 5) -> dict[str, Any]:
        return agent.memory.search(q, top_k=top_k).model_dump(mode="json")

    return app


async def run_end_to_end_demo() -> int:
    """Execute a 4-stage self-contained demonstration across memory, multimodal, MCP, and sandbox."""
    # Ensure UTF-8 stdout on Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    demo_dir = PROJECT_ROOT / "runtime_data" / "demo_session"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(
        MEMORY_DB_PATH=demo_dir / "demo_memory.sqlite3",
        SANDBOX_WORKSPACE_DIR=demo_dir / "workspace",
        TRACE_LOG_PATH=demo_dir / "demo_traces.jsonl",
    )
    agent = OmniPersonalAgent(settings)

    print("=" * 76)
    print("OMNI-PERSONAL-AGENT END-TO-END AUTONOMOUS DEMO")
    print(f"Provider Mode : {'MockLLMProvider (Offline High-Fidelity)' if settings.use_mock_llm else 'Live OpenAI-Compatible API'}")
    print(f"Workspace Dir : {settings.sandbox_workspace_dir}")
    print("=" * 76)

    # Stage 1: Seed cross-session memories with temporal contradiction + 2-hop entity graph
    await agent.memory.add_memory_async(
        "我目前居住在深圳南山区。",
        tier=MemoryTier.SEMANTIC,
        subject="user",
        slot="city",
        slot_value="深圳",
        timestamp=1_790_000_000.0,
    )
    await agent.memory.add_memory_async(
        "我已经搬到了北京海淀区中关村软件园。",
        tier=MemoryTier.SEMANTIC,
        subject="user",
        slot="city",
        slot_value="北京",
        timestamp=1_794_000_000.0,
    )
    await agent.memory.add_memory_async(
        "Project-Orion 核心代码库由研究员 ChenYu 维护。",
        entities=["Project-Orion", "ChenYu"],
        relations=[EntityRelation(source="Project-Orion", relation="maintained_by", target="ChenYu")],
    )
    await agent.memory.add_memory_async(
        "研究员 ChenYu 要求所有 PR 合并前必须附带 P95 延迟回归测试表。",
        entities=["ChenYu", "PR-Merge-Rule"],
        relations=[EntityRelation(source="ChenYu", relation="enforces", target="PR-Merge-Rule")],
    )
    await agent.memory.flush()

    out1 = await agent.run("我现在居住在哪个城市？给 Project-Orion 提 PR 需要遵守哪位维护者的什么规范？")
    print("\n[Demo 1/4] Hybrid Memory (Slot Supersession + 2-Hop Adamic-Adar Graph)")
    print(f"  State Path   : {' -> '.join(s.value for s in out1.state_transitions)}")
    print(f"  Routed Model : {out1.routing_decision.tier.value} ({out1.routing_decision.resolved_model_id})")
    print(f"  Answer       : {out1.answer}")

    # Stage 2: Multimodal PDF + MCP Finance multi-step tool loop + Independent Auditor check
    invoice_pdf = build_valid_pdf(
        demo_dir / "orion_supplement.pdf",
        pages_lines=[
            [
                "Supplement Report: Project-Orion Hardware Addon",
                "GPU_Node_Fee_USD: 900",
                "NVMe_Storage_Fee_USD: 250",
            ]
        ],
    )
    screenshot_png = build_synthetic_screenshot(
        demo_dir / "cluster_panel.png",
        window_title="Orion Cluster Status",
        metrics_kv={"Active_Nodes": "8", "P95_Latency_ms": "142"},
        ui_elements=["button:ExportAudit"],
    )
    out2 = await agent.run(
        "结合附件 PDF 与监控截图，查询 Project-Orion 的财务经费明细，计算总开销，导出财务审计报告，并写入文件 orion_full_brief.md。",
        attachments=[invoice_pdf, screenshot_png],
    )
    print("\n[Demo 2/4] Multimodal Perception + MCP Finance Workflow + Auditor Verification")
    print(f"  Routed Model : {out2.routing_decision.tier.value} (schema savings: {out2.tool_selection.token_savings_ratio * 100:.1f}%)")
    print(f"  Tools Fired  : {[t.tool_name for t in out2.executed_tools]}")
    print(f"  Audit Passed : {out2.audit_verdict.passed if out2.audit_verdict else None} | Verified Artifacts: {out2.audit_verdict.verified_artifacts if out2.audit_verdict else []}")

    # Stage 3: Privacy PII routing + Opaque Shell HITL interception
    out3 = await agent.run("整理我的证件备忘：身份证号 110108199908153219，手机 13900112233，切勿上传云端。")
    out4 = await agent.run("请执行命令：`bash -c 'echo check_env && uname -a'`")
    print("\n[Demo 3/4] Edge-Cloud Privacy Routing & Security Sandbox Gate")
    print(f"  PII Request  : Tier={out3.routing_decision.tier.value}, PII={out3.routing_decision.pii_categories}")
    print(f"  Opaque Shell : FinalState={out4.final_state.value} -> {out4.answer}")

    # Stage 4: Structured JSONL observability telemetry summary
    await agent.memory.aclose()
    trace_lines = settings.trace_log_path.read_text(encoding="utf-8").strip().splitlines()
    sample_event = json.loads(trace_lines[-1])
    print("\n[Demo 4/4] Structured JSONL Trace Telemetry")
    print(f"  Logged Spans : {len(trace_lines)} JSONL records in {settings.trace_log_path.name}")
    print(f"  Sample Span  : trace_id={sample_event['trace_id']} | component={sample_event['component']} | latency_ms={sample_event['latency_ms']} | status={sample_event['status_code']}")
    print("=" * 76)
    return 0


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omni-Personal-Agent CLI & Local Daemon")
    parser.add_argument("--demo", action="store_true", help="Run self-contained 4-stage end-to-end verification demo")
    parser.add_argument("--query", type=str, default="", help="Execute a single agent request")
    parser.add_argument("--attach", nargs="*", default=[], help="Optional PDF or image file paths")
    parser.add_argument("--serve", action="store_true", help="Launch local FastAPI HTTP service")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host for --serve")
    parser.add_argument("--port", type=int, default=8080, help="Bind port for --serve")
    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()
    if args.demo:
        return asyncio.run(run_end_to_end_demo())
    if args.serve:
        app = create_fastapi_app()
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.query:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        async def _single() -> int:
            cfg = load_settings()
            agent = OmniPersonalAgent(cfg)
            try:
                outcome = await agent.run(args.query, attachments=args.attach)
                await agent.memory.flush()
                print(json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False, indent=2))
                return 0
            finally:
                await agent.memory.aclose()

        return asyncio.run(_single())

    # Default help message if no flags passed
    print("Usage: python main.py --demo | python main.py --query '...' | python main.py --serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
