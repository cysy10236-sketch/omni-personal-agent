"""Executable tool registry with strict Pydantic v2 JSON Schema contracts."""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any, Callable
from pydantic import BaseModel, Field, ValidationError

from src.router.tool_selector import ToolSchemaSpec


_SAFE_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_SAFE_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a pure mathematical expression via Python AST without `eval()`."""
    tree = ast.parse(expression.strip(), mode="eval")

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent magnitude exceeds safety cap 100")
            return float(_SAFE_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
            return float(_SAFE_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
        raise ValueError(f"Unsupported AST construct in arithmetic expression: {type(node).__name__}")

    return round(_eval_node(tree), 6)


# --- Pydantic v2 Argument Schemas ---


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, description="Pure arithmetic expression, e.g. '(1250 + 480) * 0.85'")


class WorkspaceWriteArgs(BaseModel):
    relative_path: str = Field(min_length=1, description="Relative file path inside sandbox workspace")
    content: str = Field(description="UTF-8 text content to write")


class WorkspaceReadArgs(BaseModel):
    relative_path: str = Field(min_length=1, description="Relative file path inside sandbox workspace")


class ScheduleEventArgs(BaseModel):
    title: str = Field(min_length=1, description="Calendar event title")
    start_time: str = Field(min_length=3, description="ISO or structured time slot, e.g. '2026-11-20 15:00'")
    attendees: list[str] = Field(default_factory=list, description="List of participant names")
    notes: str = Field(default="", description="Agenda or context notes")


class McpNotesSearchArgs(BaseModel):
    query: str = Field(min_length=1, description="Keyword or topic to search in local notes vault")


class McpNotesUpsertArgs(BaseModel):
    note_key: str = Field(min_length=1, description="Unique slug/key for the note")
    body: str = Field(min_length=1, description="Markdown content of the note")


class McpFinanceQueryArgs(BaseModel):
    department_or_project: str = Field(min_length=1, description="Project code or cost center name")


class McpFinanceExportArgs(BaseModel):
    project_code: str = Field(min_length=1, description="Project code being exported")
    total_amount: float = Field(ge=0.0, description="Verified numeric total")
    summary_markdown: str = Field(min_length=5, description="Markdown audit summary")


class ShellCommandArgs(BaseModel):
    command: str = Field(min_length=1, description="Shell command string subject to SandboxGuard policy")


class McpCrmLookupArgs(BaseModel):
    contact_name: str = Field(min_length=1, description="Client or colleague name")


class McpCrmFollowupArgs(BaseModel):
    contact_name: str = Field(min_length=1, description="Client or colleague name")
    action_item: str = Field(min_length=1, description="Follow-up action description")


class McpWeatherArgs(BaseModel):
    city: str = Field(min_length=1, description="City name for weather forecast")


class McpMusicArgs(BaseModel):
    track_or_playlist: str = Field(min_length=1, description="Track or playlist title")


class McpThermostatArgs(BaseModel):
    room: str = Field(min_length=1)
    target_celsius: float = Field(ge=16.0, le=30.0)


class McpFlightSearchArgs(BaseModel):
    origin: str = Field(min_length=2)
    destination: str = Field(min_length=2)
    date: str = Field(min_length=6)


class McpGitStatusArgs(BaseModel):
    repo_subpath: str = Field(default=".")


class ToolExecutionResult(BaseModel):
    """Structured outcome returned from executing a validated tool."""

    tool_name: str
    ok: bool
    output: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None


class RegisteredTool:
    """Binds a ToolSchemaSpec, Pydantic validator model, and Python handler."""

    def __init__(
        self,
        spec: ToolSchemaSpec,
        args_model: type[BaseModel],
        handler: Callable[[BaseModel], dict[str, Any]],
    ) -> None:
        self.spec = spec
        self.args_model = args_model
        self.handler = handler

    def validate_and_run(self, raw_args: dict[str, Any]) -> ToolExecutionResult:
        try:
            validated = self.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return ToolExecutionResult(
                tool_name=self.spec.name,
                ok=False,
                output=f"JSON Schema validation failed for `{self.spec.name}`: {exc}",
                error_type="SchemaValidationError",
            )
        try:
            result_data = self.handler(validated)
            return ToolExecutionResult(
                tool_name=self.spec.name,
                ok=True,
                output=json.dumps(result_data, ensure_ascii=False),
                structured_data=result_data,
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=self.spec.name,
                ok=False,
                output=f"Runtime tool error in `{self.spec.name}`: {exc}",
                error_type=type(exc).__name__,
            )


class PersonalToolRegistry:
    """Workspace-scoped personal agent tool registry with 15 core + MCP tools."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, RegisteredTool] = {}
        self._transient_fail_once_tools: set[str] = set()
        self._seed_default_ledgers()
        self._register_builtin_and_mcp_tools()

    def inject_transient_failure_once(self, tool_name: str) -> None:
        """Deterministically fail the next invocation of `tool_name` once to test retry logic."""
        self._transient_fail_once_tools.add(tool_name)

    def _seed_default_ledgers(self) -> None:
        notes_file = self.workspace_dir / "_ledgers" / "notes_vault.json"
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        if not notes_file.exists():
            notes_file.write_text(
                json.dumps(
                    {
                        "q4_compute_quota": "Project-Orion Q4 GPU cluster quota is 3200 GPU-hours at $2.50/hour.",
                        "paper_submission_checklist": "NeurIPS rebuttal deadline is Friday 18:00 UTC; attach ablation table.",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        finance_file = self.workspace_dir / "_ledgers" / "finance_records.json"
        if not finance_file.exists():
            finance_file.write_text(
                json.dumps(
                    {
                        "Project-Orion": {
                            "cloud_inference_usd": 1240.0,
                            "dataset_licensing_usd": 560.0,
                            "hardware_rental_usd": 2200.0,
                            "approved_cap_usd": 5000.0,
                        },
                        "星辰项目": {
                            "cloud_inference_usd": 850.0,
                            "dataset_licensing_usd": 350.0,
                            "hardware_rental_usd": 1800.0,
                            "approved_cap_usd": 3500.0,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _register(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[[Any], dict[str, Any]],
        *,
        namespace: str = "core",
        mcp_role: str = "standalone",
        always_include: bool = False,
        keywords_bilingual: list[str] | None = None,
    ) -> None:
        spec = ToolSchemaSpec(
            name=name,
            description=description,
            parameters_schema=args_model.model_json_schema(),
            namespace=namespace,
            mcp_role=mcp_role,
            always_include=always_include,
            keywords_bilingual=keywords_bilingual or [],
        )
        self._tools[name] = RegisteredTool(spec=spec, args_model=args_model, handler=handler)

    def _register_builtin_and_mcp_tools(self) -> None:
        self._register(
            "calculator_eval",
            "Evaluate exact arithmetic formulas for budgets, totals, percentages, and metrics.",
            CalculatorArgs,
            lambda a: {"expression": a.expression, "value": safe_eval_arithmetic(a.expression)},
            always_include=True,
            keywords_bilingual=["计算", "求和", "总额", "比例", "预算", "calculate", "sum", "total", "math"],
        )

        def _write_file(args: WorkspaceWriteArgs) -> dict[str, Any]:
            target = (self.workspace_dir / args.relative_path).resolve()
            if not target.is_relative_to(self.workspace_dir):
                raise PermissionError(f"Path escapes sandbox workspace: {args.relative_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args.content, encoding="utf-8")
            return {
                "written_path": args.relative_path,
                "bytes_written": len(args.content.encode("utf-8")),
            }

        self._register(
            "workspace_write_file",
            "Write markdown, JSON, or report files inside the confined personal workspace.",
            WorkspaceWriteArgs,
            _write_file,
            always_include=True,
            keywords_bilingual=["写入", "保存", "生成文件", "报告", "write", "save", "file", "report", "markdown"],
        )

        def _read_file(args: WorkspaceReadArgs) -> dict[str, Any]:
            target = (self.workspace_dir / args.relative_path).resolve()
            if not target.is_relative_to(self.workspace_dir):
                raise PermissionError(f"Path escapes sandbox workspace: {args.relative_path}")
            if not target.exists():
                raise FileNotFoundError(f"Workspace file not found: {args.relative_path}")
            return {"relative_path": args.relative_path, "content": target.read_text(encoding="utf-8")}

        self._register(
            "workspace_read_file",
            "Read a text, JSON, or markdown file from the confined personal workspace.",
            WorkspaceReadArgs,
            _read_file,
            keywords_bilingual=["读取", "查看文件", "打开文件", "read", "cat", "open", "file"],
        )

        def _schedule_event(args: ScheduleEventArgs) -> dict[str, Any]:
            cal_path = self.workspace_dir / "_ledgers" / "calendar_events.json"
            events = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else []
            entry = {
                "event_id": f"evt_{len(events) + 1:03d}",
                "title": args.title,
                "start_time": args.start_time,
                "attendees": args.attendees,
                "notes": args.notes,
            }
            events.append(entry)
            cal_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
            return entry

        self._register(
            "schedule_calendar_event",
            "Create or reschedule a personal calendar meeting or reminder entry.",
            ScheduleEventArgs,
            _schedule_event,
            keywords_bilingual=["日程", "会议", "预约", "排期", "日历", "calendar", "schedule", "meeting", "book"],
        )

        def _notes_search(args: McpNotesSearchArgs) -> dict[str, Any]:
            vault_path = self.workspace_dir / "_ledgers" / "notes_vault.json"
            vault = json.loads(vault_path.read_text(encoding="utf-8"))
            q = args.query.lower()
            matches = {
                k: v for k, v in vault.items() if q in k.lower() or q in v.lower() or any(tok in v.lower() for tok in q.split())
            }
            return {"matches": matches, "count": len(matches)}

        self._register(
            "mcp_notes_search",
            "MCP Notes Server [discover]: search personal research and project notes in vault.",
            McpNotesSearchArgs,
            _notes_search,
            namespace="mcp_notes",
            mcp_role="discover",
            keywords_bilingual=["笔记", "备忘", "算力配额", "知识库", "notes", "vault", "quota", "checklist"],
        )

        def _notes_upsert(args: McpNotesUpsertArgs) -> dict[str, Any]:
            vault_path = self.workspace_dir / "_ledgers" / "notes_vault.json"
            vault = json.loads(vault_path.read_text(encoding="utf-8"))
            vault[args.note_key] = args.body
            vault_path.write_text(json.dumps(vault, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"note_key": args.note_key, "status": "upserted"}

        self._register(
            "mcp_notes_upsert",
            "MCP Notes Server [act]: create or update a structured note in the personal vault.",
            McpNotesUpsertArgs,
            _notes_upsert,
            namespace="mcp_notes",
            mcp_role="act",
            keywords_bilingual=["笔记", "记录笔记", "更新备忘", "notes", "upsert", "vault"],
        )

        def _finance_query(args: McpFinanceQueryArgs) -> dict[str, Any]:
            fin_path = self.workspace_dir / "_ledgers" / "finance_records.json"
            fin = json.loads(fin_path.read_text(encoding="utf-8"))
            for key, record in fin.items():
                if key.lower() == args.department_or_project.lower() or args.department_or_project.lower() in key.lower():
                    return {"project": key, "ledger": record}
            return {"project": args.department_or_project, "ledger": {}}

        self._register(
            "mcp_finance_query",
            "MCP Finance Server [discover]: query project expense line items and budget cap.",
            McpFinanceQueryArgs,
            _finance_query,
            namespace="mcp_finance",
            mcp_role="discover",
            keywords_bilingual=["财务", "报销", "经费", "账单", "开销", "finance", "expense", "budget", "orion", "星辰"],
        )

        def _finance_export(args: McpFinanceExportArgs) -> dict[str, Any]:
            out_path = self.workspace_dir / f"finance_audit_{args.project_code}.md"
            body = (
                f"# Finance Audit: {args.project_code}\n\n"
                f"- Verified Total USD: {args.total_amount:.2f}\n\n"
                f"{args.summary_markdown}\n"
            )
            out_path.write_text(body, encoding="utf-8")
            return {
                "exported_file": out_path.name,
                "project_code": args.project_code,
                "total_amount": args.total_amount,
            }

        self._register(
            "mcp_finance_export_report",
            "MCP Finance Server [act]: export verified project expense audit report to workspace.",
            McpFinanceExportArgs,
            _finance_export,
            namespace="mcp_finance",
            mcp_role="act",
            keywords_bilingual=["财务报告", "导出报销", "审计导出", "finance", "export", "audit"],
        )

        self._register(
            "shell_command_exec",
            "Execute a whitelisted diagnostic command inside the sandboxed workspace.",
            ShellCommandArgs,
            lambda a: {"command": a.command, "simulated_exit_code": 0, "stdout": f"executed: {a.command}"},
            keywords_bilingual=["终端", "命令行", "脚本", "shell", "bash", "terminal", "command"],
        )

        # Distractor / specialized MCP tools exercising zero-LLM tool pruning from 15 -> budget
        self._register(
            "mcp_crm_contact_lookup",
            "MCP CRM Server [discover]: look up client contact email and organization role.",
            McpCrmLookupArgs,
            lambda a: {"contact_name": a.contact_name, "email": f"{a.contact_name.lower().replace(' ', '.')}@acme.example"},
            namespace="mcp_crm",
            mcp_role="discover",
            keywords_bilingual=["客户", "联系人", "邮箱", "crm", "contact", "client"],
        )
        self._register(
            "mcp_crm_log_followup",
            "MCP CRM Server [act]: log a client follow-up activity item into CRM.",
            McpCrmFollowupArgs,
            lambda a: {"contact_name": a.contact_name, "logged_action": a.action_item},
            namespace="mcp_crm",
            mcp_role="act",
            keywords_bilingual=["客户跟进", "crm", "followup"],
        )
        self._register(
            "mcp_weather_forecast",
            "MCP Weather Server: query 3-day meteorological temperature and precipitation forecast.",
            McpWeatherArgs,
            lambda a: {"city": a.city, "forecast": "Sunny, 21C"},
            namespace="mcp_weather",
            keywords_bilingual=["天气", "气温", "降雨", "weather", "forecast"],
        )
        self._register(
            "mcp_music_playback",
            "MCP Media Server: play or queue local music tracks and focus playlists.",
            McpMusicArgs,
            lambda a: {"playing": a.track_or_playlist},
            namespace="mcp_media",
            keywords_bilingual=["音乐", "播放", "歌单", "music", "playlist", "song"],
        )
        self._register(
            "mcp_iot_thermostat_set",
            "MCP SmartHome Server: set room HVAC thermostat target temperature in Celsius.",
            McpThermostatArgs,
            lambda a: {"room": a.room, "target_celsius": a.target_celsius},
            namespace="mcp_smarthome",
            keywords_bilingual=["空调", "温度", "智能家居", "thermostat", "room temperature"],
        )
        self._register(
            "mcp_travel_flight_search",
            "MCP Travel Server: search commercial flight schedules between airport cities.",
            McpFlightSearchArgs,
            lambda a: {"route": f"{a.origin}->{a.destination}", "date": a.date, "lowest_fare_usd": 340},
            namespace="mcp_travel",
            keywords_bilingual=["机票", "航班", "差旅出行", "flight", "airline"],
        )
        self._register(
            "mcp_git_repo_status",
            "MCP Git Server: inspect branch name and modified file count in a code repository.",
            McpGitStatusArgs,
            lambda a: {"repo_subpath": a.repo_subpath, "branch": "main", "clean": True},
            namespace="mcp_git",
            keywords_bilingual=["代码仓", "分支", "git", "branch", "commit"],
        )

    def all_specs(self) -> list[ToolSchemaSpec]:
        return [t.spec for t in self._tools.values()]

    def specs_by_names(self, names: list[str]) -> list[ToolSchemaSpec]:
        return [self._tools[n].spec for n in names if n in self._tools]

    def execute(self, tool_name: str, raw_args: dict[str, Any]) -> ToolExecutionResult:
        if tool_name not in self._tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                output=f"Unknown or unselected tool `{tool_name}`.",
                error_type="UnknownToolError",
            )
        if tool_name in self._transient_fail_once_tools:
            self._transient_fail_once_tools.remove(tool_name)
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                output=f"Transient lock timeout while executing `{tool_name}`; retryable.",
                error_type="TransientResourceTimeout",
            )
        return self._tools[tool_name].validate_and_run(raw_args)
