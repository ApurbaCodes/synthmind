"""
Result Synthesizer — merges multi-agent results into a coherent answer.

Handles edge cases:
    • Single result — returns it directly (no overhead).
    • Empty results — returns a helpful fallback.
    • Conflicting sources — notes discrepancies.

SOLID:
    • Single Responsibility — only synthesizes; does not route or retrieve.
"""

from __future__ import annotations

import logging
from typing import Dict

from langchain_core.language_models import BaseChatModel

from core.exceptions import LLMError
from core.types import RouterState, extract_text_content

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthesis system prompt
# ---------------------------------------------------------------------------
_SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert GenAI Assistant that synthesizes information from multiple knowledge sources.

Instructions:
1. Combine the information into a concise, technically accurate, and non-redundant answer.
2. If sources provide conflicting information, note the discrepancy explicitly.
3. Attribute key facts to their source using [Source] notation.
4. Use markdown formatting for readability.
5. If results are empty, politely indicate that no relevant information was found.
"""


class ResultSynthesizer:
    """Merges results from multiple agents into a single coherent response.

    Args:
        llm: Chat model instance for synthesis.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def synthesize(self, state: RouterState) -> Dict:
        """Synthesize multi-agent results into a final answer.

        This is a LangGraph node function.

        Returns:
            Dict with ``final_answer`` key.

        Raises:
            LLMError: If the synthesis LLM call fails.
        """
        results = state.get("results", [])
        query = state["query"]

        # ---  Edge case: no results ---
        if not results:
            logger.warning("No agent results to synthesize for query: %s", query[:100])
            return {
                "final_answer": (
                    "I wasn't able to find relevant information from any of my "
                    "knowledge sources. Could you rephrase your question?"
                )
            }

        # ---  Edge case: single result — skip LLM overhead ---
        if len(results) == 1:
            r = results[0]
            logger.info("Single-source answer from %s — skipping synthesis.", r["source"])
            return {"final_answer": r["result"]}

        # ---  Multi-source synthesis ---
        context = "\n\n".join(
            f"[{r['source']}]:\n{r['result']}" for r in results
        )
        sources_used = ", ".join(r["source"] for r in results)
        logger.info("Synthesizing results from: %s", sources_used)

        try:
            response = self._llm.invoke([
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"**User Query:** {query}\n\n"
                        f"**Source Results:**\n{context}"
                    ),
                },
            ])
            return {"final_answer": extract_text_content(response.content)}
        except Exception as exc:
            raise LLMError(
                "Result synthesis failed",
                model=getattr(self._llm, 'model_name', str(self._llm)),
                context={"query": query[:200], "error": str(exc)},
            ) from exc
