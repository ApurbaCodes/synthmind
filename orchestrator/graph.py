"""
LangGraph Workflow Builder — constructs the multi-agent state graph.

DSA Highlight:
    The workflow is a Directed Acyclic Graph (DAG):
        START → classify → [RAG | Arxiv | Wikipedia | Tavily] → synthesize → END

    Conditional edges allow fan-out to parallel agent execution —
    graph topology is determined at runtime by the router's classification.

SOLID:
    • Open/Closed — register new agents dynamically without modifying
      the builder's core logic.
    • Dependency Inversion — agents are provided as instances, not
      created internally.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.base import BaseAgent
from core.exceptions import AgentError, AIAssistantError
from core.types import AgentInput, AgentOutput, RouterState
from orchestrator.router import QueryRouter
from orchestrator.synthesizer import ResultSynthesizer

logger = logging.getLogger(__name__)


class WorkflowBuilder:
    """Builds and manages the LangGraph workflow.

    Usage::

        builder = WorkflowBuilder(router, synthesizer)
        builder.register_agent(rag_agent)
        builder.register_agent(arxiv_agent)
        builder.register_agent(wiki_agent)
        builder.build()

        answer = builder.ask("What is attention?", config)

    Args:
        router: QueryRouter instance.
        synthesizer: ResultSynthesizer instance.
    """

    def __init__(
        self,
        router: QueryRouter,
        synthesizer: ResultSynthesizer,
    ) -> None:
        self._router = router
        self._synthesizer = synthesizer
        self._agents: Dict[str, BaseAgent] = {}
        self._workflow = None

    # ------------------------------------------------------------------
    # Agent registration (Open/Closed principle)
    # ------------------------------------------------------------------
    def register_agent(self, agent: BaseAgent) -> "WorkflowBuilder":
        """Register an agent to be included in the workflow graph.

        Args:
            agent: Any ``BaseAgent`` subclass instance.

        Returns:
            self — for fluent chaining.
        """
        self._agents[agent.name] = agent
        logger.info("Registered agent: %s", agent.name)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> "WorkflowBuilder":
        """Compile the LangGraph state graph.

        Must be called after all agents are registered.

        Returns:
            self — for fluent chaining.
        """
        if not self._agents:
            raise AIAssistantError("No agents registered — cannot build workflow.")

        builder = StateGraph(RouterState)

        # --- Add nodes ---
        builder.add_node("classify", self._router.classify)

        for agent_name, agent in self._agents.items():
            # Create a closure that captures the specific agent
            def _make_node(a: BaseAgent):
                def _node(state: AgentInput) -> dict:
                    try:
                        output = a.safe_execute(state["query"])
                        return {"results": [output]}
                    except AgentError as exc:
                        logger.error("Agent %s failed: %s", a.name, exc)
                        return {
                            "results": [
                                AgentOutput(
                                    source=a.name,
                                    result=f"[Error] {a.name} agent could not process the query.",
                                )
                            ]
                        }
                return _node

            builder.add_node(agent_name, _make_node(agent))

        builder.add_node("synthesize", self._synthesizer.synthesize)

        # --- Add edges ---
        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            self._router.route,
            list(self._agents.keys()),
        )
        for agent_name in self._agents:
            builder.add_edge(agent_name, "synthesize")
        builder.add_edge("synthesize", END)

        # --- Compile with checkpointer ---
        self._workflow = builder.compile(checkpointer=InMemorySaver())
        logger.info(
            "Workflow compiled with %d agents: %s",
            len(self._agents),
            list(self._agents.keys()),
        )
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, query: str, config: Optional[Dict] = None) -> str:
        """Run a query through the complete multi-agent pipeline.

        Args:
            query:  The user's question.
            config: LangGraph runtime config (e.g. thread_id).

        Returns:
            The synthesized answer string.

        Raises:
            AIAssistantError: If the workflow hasn't been built.
        """
        if self._workflow is None:
            raise AIAssistantError(
                "Workflow not built. Call build() before ask()."
            )

        config = config or {}
        try:
            result = self._workflow.invoke({"query": query}, config=config)
            return result.get(
                "final_answer",
                "Sorry, I couldn't generate an answer. Please try again.",
            )
        except AIAssistantError:
            raise
        except Exception as exc:
            logger.exception("Workflow execution failed for query: %s", query[:200])
            return f"⚠️ System Error: {exc}"

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def health_report(self) -> Dict[str, bool]:
        """Run health checks on all registered agents."""
        return {name: agent.health_check() for name, agent in self._agents.items()}
