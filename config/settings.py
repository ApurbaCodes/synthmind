"""
Centralized configuration management using Pydantic BaseSettings.

All application configuration is validated at startup via environment variables
and .env file. This ensures fail-fast behavior for missing or invalid config.

SOLID: Single Responsibility — this module's only job is config validation/loading.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve project root dynamically (no hardcoded paths)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMSettings(BaseSettings):
    """LLM provider configuration (Google AI Studio)."""

    api_key: str = Field(
        ...,
        alias="GOOGLE_API_KEY",
        description="Google AI Studio API key",
    )
    chat_model: str = Field(
        default="gemma-4-31b-it",
        alias="LLM_CHAT_MODEL",
        description="Chat model identifier",
    )
    embedding_model: str = Field(
        default="embeddinggemma:latest",
        alias="LLM_EMBEDDING_MODEL",
        description="Embedding model identifier",
    )
    temperature: float = Field(
        default=0.0,
        alias="LLM_TEMPERATURE",
        ge=0.0,
        le=2.0,
    )
    max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES", ge=0)
    request_timeout: int = Field(default=60, alias="LLM_REQUEST_TIMEOUT", ge=1)

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "extra": "ignore"}


class VectorStoreSettings(BaseSettings):
    """FAISS vector store configuration."""

    documents_dir: Path = Field(
        default=_PROJECT_ROOT / "documents",
        alias="DOCUMENTS_DIR",
    )
    index_dir: Path = Field(
        default=_PROJECT_ROOT / "faiss_index",
        alias="FAISS_INDEX_DIR",
    )
    manifest_path: Path = Field(
        default=_PROJECT_ROOT / "faiss_index" / "manifest.json",
        alias="FAISS_MANIFEST_PATH",
    )
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE", ge=100)
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP", ge=0)
    search_k: int = Field(default=5, alias="SEARCH_K", ge=1)
    score_threshold: Optional[float] = Field(
        default=None,
        alias="SCORE_THRESHOLD",
        description="Minimum similarity score for results (None = no threshold)",
    )

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "extra": "ignore"}

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_size(cls, v: int, info) -> int:  # noqa: N805
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v


class AppSettings(BaseSettings):
    """Top-level application settings."""

    app_name: str = Field(default="GenAI Multi-Agent Assistant", alias="APP_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    debug: bool = Field(default=False, alias="DEBUG")
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")

    model_config = {"env_file": str(_PROJECT_ROOT / ".env"), "extra": "ignore"}

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:  # noqa: N805
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper


class Settings:
    """Aggregated settings facade — single access point for all config."""

    def __init__(self) -> None:
        self.llm = LLMSettings()
        self.vectorstore = VectorStoreSettings()
        self.app = AppSettings()
        self.project_root = _PROJECT_ROOT
        self._configure_logging()

    # ------------------------------------------------------------------
    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.app.log_level),
            format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def __repr__(self) -> str:
        return (
            f"Settings(model={self.llm.chat_model!r}, "
            f"embedding={self.llm.embedding_model!r}, "
            f"docs_dir={self.vectorstore.documents_dir!r})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached)."""
    return Settings()
