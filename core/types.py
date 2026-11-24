"""
Shared type definitions used across all modules.

Centralizes TypedDicts, Pydantic models and enums so that every module
imports from a single source of truth — no inline re-definitions.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, List, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AgentSource(str, Enum):
    """Supported knowledge sources."""

    RAG = "RAG"
    ARXIV = "Arxiv"
    WIKIPEDIA = "Wikipedia"
    TAVILY = "Tavily"


# ---------------------------------------------------------------------------
# TypedDicts (used as LangGraph state schemas)
# ---------------------------------------------------------------------------
class AgentInput(TypedDict):
    """Input schema passed to individual agent nodes."""

    query: str


class AgentOutput(TypedDict):
    """Output schema returned by individual agent nodes."""

    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision mapping a source to a sub-question."""

    source: Literal["RAG", "Arxiv", "Wikipedia", "Tavily"]
    query: str


class RouterState(TypedDict):
    """
    Top-level LangGraph state for the multi-agent workflow.

    • ``results`` uses ``operator.add`` as reducer so that parallel
      agent outputs are automatically merged into a single list.
    """

    query: str
    classifications: List[Classification]
    results: Annotated[List[AgentOutput], operator.add]
    final_answer: str


# ---------------------------------------------------------------------------
# Pydantic models (for structured LLM output parsing)
# ---------------------------------------------------------------------------
class ClassificationResult(BaseModel):
    """Structured output model for the router LLM."""

    classifications: List[Classification] = Field(
        description="List of knowledge sources to query, each with an optimized sub-question",
    )


# ---------------------------------------------------------------------------
# Content extraction utility
# ---------------------------------------------------------------------------
def extract_text_content(content) -> str:
    """Extract plain text from LLM response content.

    Google/Gemma models may return content as a list of structured blocks
    like ``[{'type': 'text', 'text': '...'}]`` instead of a plain string.
    This function normalizes all formats to a single string.

    Args:
        content: The ``.content`` attribute from an LLM response message.

    Returns:
        A plain text string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)

