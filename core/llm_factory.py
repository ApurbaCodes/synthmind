"""
LLM Factory — single point of creation for all LLM and embedding instances.

Uses Google AI Studio (Gemini API) with ``langchain-google-genai``.

SOLID:
  * Single Responsibility — only creates LLM/embedding objects.
  * Open/Closed — add new providers by extending this factory
    without modifying existing code.
  * Dependency Inversion — consumers depend on this factory's interface,
    not on concrete constructors.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings

from config.settings import Settings, get_settings
from core.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMFactory:
    """Creates and caches LLM / embedding instances for Google AI Studio.

    Usage::

        factory = LLMFactory(settings)
        chat_llm = factory.create_chat_llm()
        embeddings = factory.create_embeddings()

    Args:
        settings: Application settings (optional, uses singleton).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_chat_llm(
        self,
        *,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
    ) -> ChatGoogleGenerativeAI:
        """Create a ``ChatGoogleGenerativeAI`` instance.

        Args:
            temperature: Override the default temperature.
            model: Override the default chat model name.

        Raises:
            LLMError: If instantiation fails.
        """
        llm_cfg = self._settings.llm
        model_name = model or llm_cfg.chat_model
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=llm_cfg.api_key,
                temperature=temperature if temperature is not None else llm_cfg.temperature,
                max_retries=llm_cfg.max_retries,
                timeout=llm_cfg.request_timeout,
            )
            logger.debug("Created ChatGoogleGenerativeAI  model=%s", model_name)
            return llm
        except Exception as exc:
            raise LLMError(
                "Failed to create ChatGoogleGenerativeAI instance",
                provider="google-ai-studio",
                model=model_name,
                context={"error": str(exc)},
            ) from exc

    def create_embeddings(self) -> OllamaEmbeddings:
        """Create a ``OllamaEmbeddings`` instance.

        Raises:
            LLMError: If instantiation fails.
        """
        llm_cfg = self._settings.llm
        try:
            embeddings = OllamaEmbeddings(
                model=llm_cfg.embedding_model
            )
            logger.debug(
                "Created OllamaEmbeddings  model=%s",
                llm_cfg.embedding_model,
            )
            return embeddings
        except Exception as exc:
            raise LLMError(
                "Failed to create OllamaEmbeddings instance",
                model=llm_cfg.embedding_model,
                context={"error": str(exc)},
            ) from exc


@lru_cache(maxsize=1)
def get_llm_factory() -> LLMFactory:
    """Return a singleton ``LLMFactory`` instance (cached)."""
    return LLMFactory()
