"""
Arxiv Agent — searches academic papers on arXiv.

Extracted from the monolithic ``other_agents.py`` into its own module.
Uses the modern ``langchain.agents.create_agent`` API.

SOLID:
    * Single Responsibility — only handles arXiv queries.
    * Liskov Substitution — interchangeable with any other BaseAgent.
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

from agents.base import BaseAgent, with_retry
from core.exceptions import AgentError
from core.types import AgentOutput, extract_text_content

logger = logging.getLogger(__name__)


class ArxivAgent(BaseAgent):
    """Agent that queries the arXiv API for academic research papers.

    Args:
        llm: Chat LLM instance (injected via DI).
        max_results: Maximum number of arXiv results to fetch.
    """

    def __init__(self, llm, *, max_results: int = 3) -> None:
        self._llm = llm
        self._max_results = max_results
        self._agent = self._build_agent()

    @property
    def name(self) -> str:
        return "Arxiv"

    def _build_agent(self):
        api_wrapper = ArxivAPIWrapper(
            top_k_results=self._max_results,
            doc_content_chars_max=4000,
        )
        tools = [ArxivQueryRun(api_wrapper=api_wrapper)]
        return create_agent(
            model=self._llm,
            tools=tools,
            system_prompt=(
                "You are a research assistant specializing in finding and summarizing "
                "academic papers from arXiv. Use the arxiv tool to search for relevant "
                "papers and provide concise summaries."
            ),
        )

    @with_retry(max_retries=2, base_delay=1.5)
    def execute(self, query: str) -> AgentOutput:
        """Query arXiv for research papers.

        Raises:
            AgentError: If the arXiv query fails.
        """
        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            messages = result.get("messages", [])
            final_content = extract_text_content(messages[-1].content) if messages else "No results from arXiv."
            return AgentOutput(
                source="Arxiv",
                result=final_content,
            )
        except Exception as exc:
            raise AgentError(
                "Arxiv agent execution failed",
                agent_name=self.name,
                query=query,
                context={"error": str(exc)},
            ) from exc
