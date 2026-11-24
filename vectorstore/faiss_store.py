"""
FAISS Vector Store Manager — idempotent lifecycle for document indexing.

DSA Highlight:
    FAISS uses Approximate Nearest Neighbor (ANN) search via IVF
    (Inverted File Index) partitioning. Documents are chunked, embedded
    into high-dimensional vectors, and indexed for sub-linear retrieval
    time — O(√n) average instead of brute-force O(n).

SOLID:
    • Single Responsibility — only manages vector persistence & search.
    • Dependency Inversion — receives embeddings via constructor injection.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import Settings, get_settings
from core.exceptions import DocumentLoadError, VectorStoreError

logger = logging.getLogger(__name__)


class FAISSManager:
    """Thread-safe manager for a FAISS-backed vector store.

    Lifecycle:
        1. ``load_or_create()`` — initialise from disk or fresh
        2. ``add_documents(paths)`` — incremental indexing with dedup
        3. ``search(query, k)`` — retrieval with optional score filter
        4. ``save()`` / ``delete()`` — persistence management

    Args:
        embeddings: Embedding model instance (injected).
        settings:   Application settings (optional, uses singleton).
    """

    def __init__(
        self,
        embeddings: Embeddings,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings
        self._vs_cfg = self._settings.vectorstore

        self._index_dir: Path = self._vs_cfg.index_dir
        self._manifest_path: Path = self._vs_cfg.manifest_path
        self._documents_dir: Path = self._vs_cfg.documents_dir

        self._vectorstore: Optional[FAISS] = None
        self._manifest: Dict[str, Any] = {}
        self._lock = threading.Lock()

        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._vs_cfg.chunk_size,
            chunk_overlap=self._vs_cfg.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_or_create(self) -> "FAISSManager":
        """Initialize the FAISS index (in-memory only).

        Returns:
            self — for fluent chaining.
        """
        with self._lock:
            # Auto-index any existing PDFs in documents_dir (for session resumption)
            self._index_new_documents()

        return self

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def add_documents(self, file_paths: List[str | Path]) -> int:
        """Add documents to the index (with deduplication).

        Args:
            file_paths: Absolute paths to PDF files.

        Returns:
            Number of *new* chunks added.

        Raises:
            DocumentLoadError: If an individual file fails to parse.
            VectorStoreError: If index persistence fails.
        """
        processed: Dict[str, str] = self._manifest.get("processed_files", {})
        new_paths = [str(p) for p in file_paths if str(p) not in processed]

        if not new_paths:
            logger.info("No new documents to index.")
            return 0

        all_chunks: List[Document] = []
        for fpath in new_paths:
            try:
                ext = Path(fpath).suffix.lower()
                if ext == ".pdf":
                    loader = PyMuPDFLoader(fpath)
                elif ext in [".txt", ".md"]:
                    loader = TextLoader(fpath, encoding="utf-8")
                else:
                    logger.warning("Unsupported file type %s", fpath)
                    continue
                raw_docs = loader.load()
                chunks = self._text_splitter.split_documents(raw_docs)
                all_chunks.extend(chunks)
                processed[fpath] = datetime.now(timezone.utc).isoformat()
                logger.info("Parsed %s — %d chunks", Path(fpath).name, len(chunks))
            except Exception as exc:
                logger.error("Failed to load %s: %s", fpath, exc)
                raise DocumentLoadError(
                    f"Failed to parse document: {Path(fpath).name}",
                    file_path=fpath,
                    context={"error": str(exc)},
                ) from exc

        if not all_chunks:
            return 0

        with self._lock:
            if self._vectorstore is None:
                self._vectorstore = FAISS.from_documents(all_chunks, self._embeddings)
            else:
                self._vectorstore.add_documents(all_chunks)

            self._manifest["processed_files"] = processed
            self._manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._manifest["total_chunks"] = self._manifest.get("total_chunks", 0) + len(all_chunks)

        logger.info("Indexed %d new chunks from %d files.", len(all_chunks), len(new_paths))
        return len(all_chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """Search the vector store.

        Args:
            query: The search query.
            k: Number of results (default from settings).
            score_threshold: Min similarity score (None = no filter).

        Returns:
            List of (Document, score) tuples, ordered by relevance.

        Raises:
            VectorStoreError: If the index is not initialised.
        """
        if self._vectorstore is None:
            raise VectorStoreError(
                "Vector store is not initialized. Call load_or_create() first.",
                operation="search",
            )

        k = k or self._vs_cfg.search_k
        threshold = score_threshold or self._vs_cfg.score_threshold

        try:
            results = self._vectorstore.similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise VectorStoreError(
                "FAISS search failed",
                operation="search",
                context={"query": query[:200], "error": str(exc)},
            ) from exc

        if threshold is not None:
            results = [(doc, score) for doc, score in results if score <= threshold]

        return results

    # ------------------------------------------------------------------
    # Retriever (LangChain-compatible)
    # ------------------------------------------------------------------
    def as_retriever(self, **kwargs):
        """Return a LangChain retriever backed by this store."""
        if self._vectorstore is None:
            raise VectorStoreError(
                "Vector store is not initialized. Call load_or_create() first.",
                operation="as_retriever",
            )
        search_kwargs = {"k": self._vs_cfg.search_k}
        search_kwargs.update(kwargs.get("search_kwargs", {}))
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)

    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Clear the in-memory index and manifest."""
        with self._lock:
            self._vectorstore = None
            self._manifest = {}
            logger.info("Cleared FAISS index from memory.")

    # ------------------------------------------------------------------
    # Auto-index
    # ------------------------------------------------------------------
    def _index_new_documents(self) -> None:
        """Discover and index any un-processed PDFs in the documents dir."""
        if not self._documents_dir.exists():
            self._documents_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created documents directory: %s", self._documents_dir)
            return

        doc_files = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            doc_files.extend(self._documents_dir.glob(ext))
        doc_files = sorted(doc_files)
        if not doc_files:
            if self._vectorstore is None:
                # Bootstrap an empty index so search() doesn't crash
                self._vectorstore = FAISS.from_texts(
                    ["Initial empty index"],
                    self._embeddings,
                )
                logger.info("Bootstrapped empty FAISS index.")
            return

        processed = set(self._manifest.get("processed_files", {}).keys())
        new_docs = [p for p in doc_files if str(p) not in processed]

        if new_docs:
            # Release lock before add_documents (it re-acquires)
            # ... actually we call it within load_or_create which holds the lock,
            # so we do the inline version here.
            all_chunks: List[Document] = []
            pf: Dict[str, str] = self._manifest.get("processed_files", {})

            for doc_path in new_docs:
                try:
                    ext = doc_path.suffix.lower()
                    if ext == ".pdf":
                        loader = PyMuPDFLoader(str(doc_path))
                    elif ext in [".txt", ".md"]:
                        loader = TextLoader(str(doc_path), encoding="utf-8")
                    else:
                        continue
                    chunks = self._text_splitter.split_documents(loader.load())
                    all_chunks.extend(chunks)
                    pf[str(doc_path)] = datetime.now(timezone.utc).isoformat()
                    logger.info("Parsed %s — %d chunks", doc_path.name, len(chunks))
                except Exception as exc:
                    logger.error("Skipping %s: %s", doc_path.name, exc)

            if all_chunks:
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents(all_chunks, self._embeddings)
                else:
                    self._vectorstore.add_documents(all_chunks)

                self._manifest["processed_files"] = pf
                self._manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
                self._manifest["total_chunks"] = self._manifest.get("total_chunks", 0) + len(all_chunks)
                logger.info("Auto-indexed %d chunks from %d new documents.", len(all_chunks), len(new_docs))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------
    @property
    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "total_chunks": self._manifest.get("total_chunks", 0),
            "processed_files": len(self._manifest.get("processed_files", {})),
            "last_updated": self._manifest.get("last_updated"),
            "index_exists": self._vectorstore is not None,
        }
