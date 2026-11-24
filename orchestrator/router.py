"""
Query Router — classifies user queries and maps them to knowledge sources.

Uses structured LLM output to determine which agents should handle a query
and generates optimized sub-questions for each source.

DSA Highlight:
    Graph-based routing — the router sits at the "classify" node of a
    LangGraph ``StateGraph``.  Conditional edges fan-out to parallel
    agent nodes, forming a DAG (Directed Acyclic Graph).

SOLID:
    • Single Responsibility — only classifies; does not execute agents.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langgraph.types import Send

from core.exceptions import LLMError
from core.types import AgentSource, Classification, ClassificationResult, RouterState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for query classification
# ---------------------------------------------------------------------------
_CLASSIFICATION_SYSTEM_PROMPT = """\
Analyze the user's query to identify the best knowledge sources and generate optimized sub-questions.

Sources & Logic:
1. RAG: Use for querying any context or information from the user's explicitly uploaded documents and files.
2. Arxiv: Use ONLY if the query asks for "latest," "recent," "2024-2026," or "current research" trends.
3. Wikipedia: Use for historical context, biographies, or broad non-technical overviews.
4. Tavily: Use for real-time web search, general internet queries, or recent events not covered by Wikipedia or Arxiv.

For each relevant source, provide a targeted sub-question.

Example:
  Query: "Summarize the key points from the document I just uploaded and what is the current weather?"
  → RAG:  "Extract the key points and summary from the uploaded documents."
  → Tavily: "What is the current weather today?"
"""


class QueryRouter:
    """Classifies user queries and routes them to appropriate agents.

    Args:
        llm: Chat model instance for structured output generation.
    """

    _VALID_SOURCES = {src.value for src in AgentSource}

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm
        self._structured_llm = llm.with_structured_output(ClassificationResult)

    # ------------------------------------------------------------------
    # LangGraph node functions
    # ------------------------------------------------------------------
    def classify(self, state: RouterState) -> Dict:
        """Classify a query into one or more agent sources.

        This is a LangGraph node function — it receives and returns state.

        Raises:
            LLMError: If the classification call fails.
        """
        query = state["query"]
        try:
            result: ClassificationResult = self._structured_llm.invoke([
                {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ])
            # Validate: filter out unknown sources
            valid_cls = [
                c for c in result.classifications
                if c["source"] in self._VALID_SOURCES
            ]
            if not valid_cls:
                logger.warning(
                    "No valid sources classified for query: %s — falling back to RAG",
                    query[:100],
                )
                valid_cls = [Classification(source="RAG", query=query)]

            logger.info(
                "Classified query → %s",
                [c["source"] for c in valid_cls],
            )
            return {"classifications": valid_cls}

        except Exception as exc:
            raise LLMError(
                "Query classification failed",
                model=getattr(self._llm, 'model_name', str(self._llm)),
                context={"query": query[:200], "error": str(exc)},
            ) from exc

    @staticmethod
    def route(state: RouterState) -> List[Send]:
        """Fan-out: dispatch sub-queries to the appropriate agent nodes.

        This is a LangGraph conditional edge function — returns ``Send``
        objects that target named nodes in the graph.
        """
        return [
            Send(c["source"], {"query": c["query"]})
            for c in state["classifications"]
        ]
        
