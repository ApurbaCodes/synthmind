"""
Tavily Agent — searches the web for real-time information.

Uses the modern ``langchain.agents.create_agent`` API with TavilySearchResults.
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from agents.base import BaseAgent, with_retry
from config.settings import get_settings
from core.exceptions import AgentError
from core.types import AgentOutput, extract_text_content

logger = logging.getLogger(__name__)


class TavilyAgent(BaseAgent):
    """Agent that queries the Tavily API for general internet search.

    Args:
        llm: Chat LLM instance (injected via DI).
        max_results: Maximum number of search results to fetch.
    """

    def __init__(self, llm, *, max_results: int = 3) -> None:
        self._llm = llm
        self._max_results = max_results
        settings = get_settings()
        self._has_key = bool(settings.app.tavily_api_key and settings.app.tavily_api_key.strip())
        self._agent = self._build_agent() if self._has_key else None

    @property
    def name(self) -> str:
        return "Tavily"

    def health_check(self) -> bool:
        return self._has_key

    def _build_agent(self):
        settings = get_settings()
        api_wrapper = TavilySearchAPIWrapper(tavily_api_key=settings.app.tavily_api_key)
        tool = TavilySearchResults(
            api_wrapper=api_wrapper,
            max_results=self._max_results,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        tools = [tool]
        return create_agent(
            model=self._llm,
            tools=tools,
            system_prompt=(
                "You are a knowledgeable assistant specializing in real-time "
                "internet search using Tavily. Use the tavily_search_results_json tool "
                "to find relevant information online and provide accurate, well-structured answers."
            ),
        )

    @with_retry(max_retries=2, base_delay=1.5)
    def execute(self, query: str) -> AgentOutput:
        """Query Tavily.

        Raises:
            AgentError: If the Tavily query fails.
        """
        if not self._has_key or self._agent is None:
            return AgentOutput(
                source="Tavily",
                result="Tavily live search is not configured. Add a TAVILY_API_KEY to your .env file to enable web search.",
            )

        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            messages = result.get("messages", [])
            final_content = extract_text_content(messages[-1].content) if messages else "No results from Tavily."
            return AgentOutput(
                source="Tavily",
                result=final_content,
            )
        except Exception as exc:
            raise AgentError(
                "Tavily agent execution failed",
                agent_name=self.name,
                query=query,
                context={"error": str(exc)},
            ) from exc
