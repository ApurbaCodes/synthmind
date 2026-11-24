"""
Custom exception hierarchy for the AI Assistant.

Every exception carries structured context so that callers can log,
display, or retry intelligently without parsing exception messages.

Design:
    AIAssistantError          (base)
    ├── ConfigurationError    — missing/invalid env vars or config
    ├── VectorStoreError      — FAISS load/save/search failures
    ├── LLMError              — API call failures, rate limits, timeouts
    ├── AgentError            — agent execution failures
    └── DocumentLoadError     — PDF parsing or file-system failures
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class AIAssistantError(Exception):
    """Base exception for all AI Assistant errors.

    Attributes:
        message:  Human-readable description.
        context:  Structured metadata for logging / telemetry.
    """

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.context: Dict[str, Any] = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        base = self.message
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            base += f" [{details}]"
        return base


class ConfigurationError(AIAssistantError):
    """Raised when required configuration is missing or invalid."""

    def __init__(
        self,
        message: str = "Configuration error",
        *,
        setting_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        if setting_name:
            ctx["setting"] = setting_name
        super().__init__(message, context=ctx)


class VectorStoreError(AIAssistantError):
    """Raised on FAISS index load/save/search failures."""

    def __init__(
        self,
        message: str = "Vector store error",
        *,
        operation: Optional[str] = None,
        path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        if operation:
            ctx["operation"] = operation
        if path:
            ctx["path"] = path
        super().__init__(message, context=ctx)


class LLMError(AIAssistantError):
    """Raised when an LLM API call fails (timeout, rate-limit, etc.)."""

    def __init__(
        self,
        message: str = "LLM API error",
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retries: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        if provider:
            ctx["provider"] = provider
        if model:
            ctx["model"] = model
        if retries:
            ctx["retries_attempted"] = retries
        super().__init__(message, context=ctx)


class AgentError(AIAssistantError):
    """Raised when an agent fails to produce a result."""

    def __init__(
        self,
        message: str = "Agent execution error",
        *,
        agent_name: Optional[str] = None,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        if agent_name:
            ctx["agent"] = agent_name
        if query:
            ctx["query"] = query[:200]  # truncate long queries
        super().__init__(message, context=ctx)


class DocumentLoadError(AIAssistantError):
    """Raised when a document (PDF) fails to load or parse."""

    def __init__(
        self,
        message: str = "Document load error",
        *,
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context or {}
        if file_path:
            ctx["file"] = file_path
        super().__init__(message, context=ctx)
