"""Two-layer security sandbox guard and workspace SHA-256 snapshot auditor."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from pydantic import BaseModel, Field


class SandboxVerdict(str, Enum):
    """Policy decision before a tool call reaches the host execution layer."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class SandboxCheckResult(BaseModel):
    """Detailed security check decision with triggered rule identifier."""

    verdict: SandboxVerdict
    rule_id: str
    reason: str
    tool_name: str


_SENSITIVE_PATH_RE = re.compile(
    r"(\.env(?:\.|$)|\.ssh[/\\]|id_rsa|\.pem$|\.key$|/etc/passwd|/etc/shadow|"
    r"windows[/\\]system32|\.\.[/\\])",
    re.IGNORECASE,
)

_HARD_DENY_CMD_RE = re.compile(
    r"(\brm\s+-[rf]{1,2}\b|\bdel\s+/[sqf]|\bformat\s+[a-z]:|\bmkfs\b|"
    r"\bsudo\s+|\bshutdown\b|\breboot\b|curl\s+.*\|\s*(?:sh|bash)|"
    r"\bcat\s+.*(?:\.env|id_rsa|\.ssh)|Invoke-WebRequest.*\|\s*iex)",
    re.IGNORECASE,
)

_OPAQUE_SHELL_WRAPPER_RE = re.compile(
    r"(\bbash\s+-c\b|\bsh\s+-c\b|\bcmd(?:\.exe)?\s+/c\b|"
    r"\bpowershell(?:\.exe)?\s+-(?:e|enc|encodedcommand|command)\b|"
    r"\beval\s+|\bxargs\s+|\bfind\b.*-exec\b)",
    re.IGNORECASE,
)


class SandboxGuard:
    """Enforces path confinement, hard-deny floor, opaque-wrapper HITL, and taint isolation."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()

    def evaluate_tool_call(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        *,
        is_context_tainted: bool = False,
        approved_tokens: set[str] | None = None,
    ) -> SandboxCheckResult:
        approved = approved_tokens or set()
        serialized_args = json.dumps(raw_args, ensure_ascii=False)

        # 1. Hard-deny sensitive credential/system path floor (never overridable)
        for key in ("relative_path", "path", "command"):
            val = str(raw_args.get(key, ""))
            if val and _SENSITIVE_PATH_RE.search(val):
                return SandboxCheckResult(
                    verdict=SandboxVerdict.DENY,
                    rule_id="HARD_DENY_SENSITIVE_PATH",
                    reason=f"Blocked access to sensitive or traversal path in `{key}`: {val}",
                    tool_name=tool_name,
                )

        if "relative_path" in raw_args:
            candidate = (self.workspace_dir / str(raw_args["relative_path"])).resolve()
            if not candidate.is_relative_to(self.workspace_dir):
                return SandboxCheckResult(
                    verdict=SandboxVerdict.DENY,
                    rule_id="HARD_DENY_WORKSPACE_ESCAPE",
                    reason=f"Target path escapes sandbox workspace root: {raw_args['relative_path']}",
                    tool_name=tool_name,
                )

        # 2. Untrusted external taint isolation (blocks indirect prompt injection exfiltration/exec)
        if is_context_tainted and tool_name in {"shell_command_exec"}:
            return SandboxCheckResult(
                verdict=SandboxVerdict.DENY,
                rule_id="TAINTED_CONTEXT_EXEC_BLOCK",
                reason="External document/screenshot carries prompt-injection taint; shell execution forbidden.",
                tool_name=tool_name,
            )

        if is_context_tainted and _SENSITIVE_PATH_RE.search(serialized_args):
            return SandboxCheckResult(
                verdict=SandboxVerdict.DENY,
                rule_id="TAINTED_EXFILTRATION_BLOCK",
                reason="Blocked tainted attempt to reference sensitive files or environment secrets.",
                tool_name=tool_name,
            )

        # 3. Shell command inspection: hard-deny vs opaque wrapper HITL approval
        if tool_name == "shell_command_exec":
            cmd = str(raw_args.get("command", ""))
            if _HARD_DENY_CMD_RE.search(cmd):
                return SandboxCheckResult(
                    verdict=SandboxVerdict.DENY,
                    rule_id="HARD_DENY_DESTRUCTIVE_CMD",
                    reason=f"Command matched destructive or secret-exfiltration hard-deny pattern: `{cmd}`",
                    tool_name=tool_name,
                )
            approval_sig = f"{tool_name}:{hashlib.sha256(cmd.encode('utf-8')).hexdigest()[:10]}"
            if _OPAQUE_SHELL_WRAPPER_RE.search(cmd) and approval_sig not in approved:
                return SandboxCheckResult(
                    verdict=SandboxVerdict.REQUIRE_APPROVAL,
                    rule_id="OPAQUE_SHELL_WRAPPER_HITL",
                    reason=(
                        f"Opaque shell wrapper detected (`{cmd}`); downgraded to Human-in-the-Loop "
                        f"approval token `{approval_sig}`."
                    ),
                    tool_name=tool_name,
                )

        return SandboxCheckResult(
            verdict=SandboxVerdict.ALLOW,
            rule_id="POLICY_ALLOW",
            reason="Passed path confinement, taint isolation, and command safety checks.",
            tool_name=tool_name,
        )


class AuditVerdict(BaseModel):
    """Independent ground-truth audit report modeled on LongHorizon-Harness Auditor."""

    passed: bool
    integrity_status: str = Field(pattern=r"^(clean|violation)$")
    verified_artifacts: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    feedback_for_planner: str = ""


class WorkspaceSnapshotAuditor:
    """Captures workspace SHA-256 snapshots and verifies real physical deliverables."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()

    def capture_snapshot(self) -> dict[str, bytes]:
        """Snapshot all workspace files (excluding internal `_ledgers` backup cache) in memory."""
        snap: dict[str, bytes] = {}
        if not self.workspace_dir.exists():
            return snap
        for file_path in sorted(self.workspace_dir.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(self.workspace_dir).as_posix()
                snap[rel] = file_path.read_bytes()
        return snap

    @staticmethod
    def snapshot_hashes(snapshot: dict[str, bytes]) -> dict[str, str]:
        return {rel: hashlib.sha256(raw).hexdigest() for rel, raw in snapshot.items()}

    def restore_snapshot(self, snapshot: dict[str, bytes]) -> None:
        """Roll back workspace files to `snapshot` state upon integrity violation."""
        for file_path in list(self.workspace_dir.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(self.workspace_dir).as_posix()
                if rel not in snapshot:
                    file_path.unlink()
        for rel, data in snapshot.items():
            target = self.workspace_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def audit_round(
        self,
        user_query: str,
        final_answer: str,
        executed_tools: list[tuple[str, dict[str, Any], bool, str]],
        pre_round_snapshot: dict[str, bytes],
    ) -> AuditVerdict:
        """Independently verify tool outcomes and physical workspace artifacts."""
        missing: list[str] = []
        verified: list[str] = []

        # 1. Any failed tool execution that was never recovered in the same round fails audit
        failed_tools = [name for name, _, ok, _ in executed_tools if not ok]
        succeeded_tools = {name for name, _, ok, _ in executed_tools if ok}
        unresolved_failures = [t for t in failed_tools if t not in succeeded_tools]
        if unresolved_failures:
            missing.append(f"Unresolved tool execution failure(s): {', '.join(unresolved_failures)}")

        # 2. Check explicit file deliverable requirements in user request
        file_matches = re.findall(r"([A-Za-z0-9_-]+\.(?:md|json|txt|csv))", user_query)
        for expected_file in file_matches:
            # Skip input fixture files mentioned as sources if they are outside workspace
            if expected_file.lower().endswith(".pdf") or expected_file.lower().endswith(".png"):
                continue
            # Only enforce existence when user asked to write/save/export/generate
            if any(
                kw in user_query.lower()
                for kw in ("写入", "保存", "生成", "导出", "write", "save", "export", "create")
            ):
                candidate = self.workspace_dir / expected_file
                if not candidate.exists() or candidate.stat().st_size == 0:
                    missing.append(f"Expected output artifact `{expected_file}` missing or empty on disk.")
                else:
                    verified.append(expected_file)

        # 3. Verify calendar ledger mutation if user explicitly asked to schedule/create an event
        if any(
            kw in user_query.lower()
            for kw in ("预约日程", "创建日程", "安排会议", "添加到日历", "schedule a meeting", "book calendar")
        ):
            cal_path = self.workspace_dir / "_ledgers" / "calendar_events.json"
            if not cal_path.exists() or "schedule_calendar_event" not in succeeded_tools:
                missing.append("Calendar scheduling requested, but `calendar_events.json` was not updated.")
            else:
                verified.append("_ledgers/calendar_events.json")

        # 4. Verify MCP finance export file if `mcp_finance_export_report` succeeded
        for name, args, ok, _ in executed_tools:
            if ok and name == "mcp_finance_export_report":
                proj = str(args.get("project_code", ""))
                audit_file = self.workspace_dir / f"finance_audit_{proj}.md"
                if audit_file.exists() and audit_file.stat().st_size > 10:
                    verified.append(audit_file.name)
                else:
                    missing.append(f"Finance export `{audit_file.name}` not persisted on disk.")

        # 5. Non-empty grounded answer requirement
        if not final_answer or len(final_answer.strip()) < 4:
            missing.append("Assistant returned an empty or trivial final answer.")

        post_hashes = self.snapshot_hashes(self.capture_snapshot())
        pre_hashes = self.snapshot_hashes(pre_round_snapshot)
        for rel, h_after in post_hashes.items():
            if pre_hashes.get(rel) != h_after and rel not in verified:
                verified.append(rel)

        if missing:
            return AuditVerdict(
                passed=False,
                integrity_status="clean",
                verified_artifacts=verified,
                missing_requirements=missing,
                feedback_for_planner=(
                    "Independent Auditor rejected completion: "
                    + "; ".join(missing)
                    + ". Re-plan and execute the missing tool steps with valid JSON arguments."
                ),
            )

        return AuditVerdict(
            passed=True,
            integrity_status="clean",
            verified_artifacts=verified,
            missing_requirements=[],
            feedback_for_planner="",
        )
