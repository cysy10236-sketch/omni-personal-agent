"""Centralized strongly-typed configuration with contract assertions."""

from __future__ import annotations

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AgentSettings(BaseSettings):
    """Validated runtime configuration loaded from `.env` or environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    local_slm_base_url: str = Field(default="", alias="LOCAL_SLM_BASE_URL")
    local_slm_api_key: str = Field(default="ollama", alias="LOCAL_SLM_API_KEY")

    llm_model: str = Field(default="gpt-4o-2024-11-20", alias="LLM_MODEL")
    slm_model: str = Field(default="gpt-4o-mini-2024-07-18", alias="SLM_MODEL")
    local_privacy_model: str = Field(default="qwen2.5-7b-instruct", alias="LOCAL_PRIVACY_MODEL")
    vlm_model: str = Field(default="gpt-4o-2024-11-20", alias="VLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="TEMPERATURE")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    request_timeout_seconds: float = Field(
        default=45.0, gt=0.0, le=300.0, alias="REQUEST_TIMEOUT_SECONDS"
    )

    memory_db_path: Path = Field(
        default=PROJECT_ROOT / "runtime_data" / "memory" / "omni_memory.sqlite3",
        alias="MEMORY_DB_PATH",
    )
    embedding_dim: int = Field(default=256, ge=64, le=4096, alias="EMBEDDING_DIM")
    memory_base_half_life_hours: float = Field(
        default=168.0, gt=0.1, alias="MEMORY_BASE_HALF_LIFE_HOURS"
    )
    memory_abstention_threshold: float = Field(
        default=0.18, ge=0.0, le=1.0, alias="MEMORY_ABSTENTION_THRESHOLD"
    )

    router_complexity_threshold: float = Field(
        default=0.42, ge=0.0, le=1.0, alias="ROUTER_COMPLEXITY_THRESHOLD"
    )
    router_tool_budget: int = Field(default=6, ge=2, le=64, alias="ROUTER_TOOL_BUDGET")
    agent_max_reflect_retries: int = Field(
        default=2, ge=0, le=5, alias="AGENT_MAX_REFLECT_RETRIES"
    )

    trace_log_path: Path = Field(
        default=PROJECT_ROOT / "logs" / "trace_events.jsonl", alias="TRACE_LOG_PATH"
    )
    sandbox_workspace_dir: Path = Field(
        default=PROJECT_ROOT / "runtime_data" / "workspace", alias="SANDBOX_WORKSPACE_DIR"
    )

    @field_validator("memory_db_path", "trace_log_path", "sandbox_workspace_dir", mode="before")
    @classmethod
    def _resolve_relative_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            return (PROJECT_ROOT / path).resolve()
        return path.resolve()

    @property
    def use_mock_llm(self) -> bool:
        """Returns True when no live API key is configured, activating MockLLMProvider."""
        cleaned = self.openai_api_key.strip()
        return (not cleaned) or cleaned.lower() in {"mock", "your_api_key_here", "sk-xxx"}


def load_settings(**overrides: object) -> AgentSettings:
    """Instantiate validated AgentSettings and ensure runtime directories exist."""
    settings = AgentSettings(**overrides)
    settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
    settings.sandbox_workspace_dir.mkdir(parents=True, exist_ok=True)
    return settings
