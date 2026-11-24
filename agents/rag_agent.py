"""
RAG Agent — retrieves answers from the local PDF knowledge base.

Uses the modern ``langchain.agents.create_agent`` API which returns a
LangGraph ``CompiledStateGraph`` for tool-calling agents.

SOLID:
    * Dependency Inversion — receives ``FAISSManager`` and LLM
      via constructor injection, not created internally.
    * Single Responsibility — retrieval + LLM answer generation.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool

from agents.base import BaseAgent, with_retry
from core.exceptions import AgentError
from core.types import AgentOutput, extract_text_content
from vectorstore.faiss_store import FAISSManager

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Retrieval-Augmented Generation agent backed by a FAISS vector store.

    Args:
        faiss_manager: Initialized FAISSManager instance (injected).
        llm: Chat LLM instance (injected).
        k: Number of documents to retrieve per query.
    """

    def __init__(
        self,
        faiss_manager: FAISSManager,
        llm,
        *,
        k: int = 5,
    ) -> None:
        self._faiss_manager = faiss_manager
        self._llm = llm
        self._k = k
        self._agent = self._build_agent()

    @property
    def name(self) -> str:
        return "RAG"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build_agent(self):
        """Construct the LangGraph agent with a retriever tool."""
        retriever = self._faiss_manager.as_retriever(
            search_kwargs={"k": self._k},
        )

        @tool
        def retriever_tool(query: str) -> str:
            """Search local PDF textbooks for NLP, GenAI, and ML technical information."""
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant documents found in the knowledge base."
            return "\n\n".join(
                f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
                for d in docs
            )

        return create_agent(
            model=self._llm,
            tools=[retriever_tool],
            system_prompt=(
                "You are a helpful AI assistant with access to a local knowledge base "
                "of PDF textbooks on NLP, GenAI, and Machine Learning. "
                "Use the retriever tool to search for relevant information before answering. "
                "Always base your answers on the retrieved content."
            ),
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    @with_retry(max_retries=2, base_delay=1.0)
    def execute(self, query: str) -> AgentOutput:
        """Run the RAG pipeline.

        Raises:
            AgentError: If the retrieval or LLM call fails.
        """
        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            # Extract the final assistant message
            messages = result.get("messages", [])
            final_content = extract_text_content(messages[-1].content) if messages else "No output generated."
            return AgentOutput(
                source="RAG",
                result=final_content,
            )
        except Exception as exc:
            raise AgentError(
                "RAG agent execution failed",
                agent_name=self.name,
                query=query,
                context={"error": str(exc)},
            ) from exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Verify the FAISS index is loaded."""
        return self._faiss_manager.stats.get("index_exists", False)
