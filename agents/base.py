"""
Abstract base agent with retry logic.

SOLID:
    • Interface Segregation — every agent exposes only ``execute()``
      and ``health_check()``.
    • Liskov Substitution — all concrete agents are interchangeable.
    • Open/Closed — add new agents by subclassing, not modifying.
"""

from __future__ import annotations

import functools
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeVar

from core.exceptions import AgentError
from core.types import AgentOutput

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry decorator with exponential back-off
# ---------------------------------------------------------------------------
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator: retries a function with exponential back-off.

    DSA note: Exponential back-off is a classic strategy for handling
    transient failures in distributed systems — delay doubles each
    attempt to avoid thundering-herd effects.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        logger.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.1fs",
                            func.__qualname__,
                            attempt,
                            max_retries,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__qualname__,
                            max_retries,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class BaseAgent(ABC):
    """Contract that every knowledge-source agent must implement.

    Subclasses must override:
        • ``execute(query)`` — perform the actual work.
        • ``name``          — a unique identifier for the agent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable name for this agent."""
        ...

    @abstractmethod
    def execute(self, query: str) -> AgentOutput:
        """Run the agent against a query.

        Args:
            query: The user's (or router's) question.

        Returns:
            AgentOutput dict with ``source`` and ``result``.

        Raises:
            AgentError: On unrecoverable failure.
        """
        ...

    def health_check(self) -> bool:
        """Verify that the agent is operational.

        Default implementation returns True.  Override for agents that
        depend on external resources (e.g., FAISS index, API keys).
        """
        return True

    # ------------------------------------------------------------------
    # Convenience: safe execution with error wrapping
    # ------------------------------------------------------------------
    def safe_execute(self, query: str) -> AgentOutput:
        """Execute with automatic error wrapping.

        Never raises — returns an error result instead.
        """
        try:
            return self.execute(query)
        except AgentError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in agent %s", self.name)
            raise AgentError(
                f"Agent '{self.name}' failed unexpectedly",
                agent_name=self.name,
                query=query,
                context={"error": str(exc)},
            ) from exc

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
