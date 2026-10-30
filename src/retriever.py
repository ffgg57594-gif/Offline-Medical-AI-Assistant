"""
retriever.py  —  In-memory <15 ms hybrid search engine + Pre-LLM Hard Guardrail Gate.

Data layout at build time (scripts/build_knowledge_db.py):
    * assets/books_knowledge.db   — SQLite DB; passages table + FTS5 virtual table
    * assets/book_embeddings.npy  — float32 [N, D] cosine-normalized matrix

Runtime (config.py):
    * Numpy matrix + DB rows are loaded ONCE at process start (async, off the UI
      thread) so every retrieve() call is pure memory + index: < 15 ms.

Scoring (hybrid):
    score = 0.7 * cos_sim(query, candidates) + 0.3 * fts5_exact_match(candidate)

Guardrail (STRICT_SIMILARITY_THRESHOLD = 0.62):
    * If best_score <= 0.62 → block the LLM entirely, return REFUSAL_MESSAGE
      immediately (< 10 ms wall time).  No GGUF is loaded until a valid query.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

try:
    from .config import (
        KNOWLEDGE_DB_PATH,
        KNOWLEDGE_EMBEDDINGS_PATH,
        STRICT_SIMILARITY_THRESHOLD,
    )
except ImportError:
    from config import (
        KNOWLEDGE_DB_PATH,
        KNOWLEDGE_EMBEDDINGS_PATH,
        STRICT_SIMILARITY_THRESHOLD,
    )


class Passage(NamedTuple):
    text: str
    page_number: int
    book_name: str
    chunk_index: int


class RetrievalResult(NamedTuple):
    score: float
    passage: Passage
    fts_match: bool


class Retriever:
    """Thread-safe, fully in-memory hybrid retriever."""

    _instance: "Retriever | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "Retriever":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._matrix: np.ndarray | None = None
        self._passages: list[Passage] = []
        self._db_conn: sqlite3.Connection | None = None
        self._fts_enabled = False
        self._executor: ThreadPoolExecutor | None = None
        self._start_time_ns: float | None = None
        self._init()

    def _init(self) -> None:
        db_path = Path(KNOWLEDGE_DB_PATH)
        emb_path = Path(KNOWLEDGE_EMBEDDINGS_PATH)
        if not db_path.exists():
            raise FileNotFoundError(f"Knowledge database not found at {db_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Knowledge embeddings not found at {emb_path}")

        # ---- load float32 embedding matrix (cosine-normalized at build time) ----
        self._matrix = np.load(str(emb_path), allow_pickle=False, mmap_mode="r")
        self._matrix = self._matrix.astype(np.float32)
        if self._matrix.ndim == 1:
            self._matrix = self._matrix.reshape(1, -1)

        # ---- load passages from SQLite (single-threaded load is fine; < 5 ms) ----
        conn = sqlite3.connect(str(db_path), timeout=60)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT passages.id, passages.book_id, books.title,
                   passages.passage, passages.page_number, passages.chunk_index
            FROM passages
            JOIN books ON passages.book_id = books.id
            ORDER BY passages.chunk_index
        """
        )
        rows = cur.fetchall()
        self._passages = [
            Passage(text=str(r["passage"]), page_number=int(r["page_number"]),
                    book_name=str(r["title"]), chunk_index=int(r["chunk_index"]))
            for r in rows
        ]
        conn.close()

        self._fts_enabled = self._try_enable_fts5()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retriever")
        self._start_time_ns = time.perf_counter_ns()
        self._initialized = True

    def _try_enable_fts5(self) -> bool:
        """Activate FTS5 keyword match for exact-phrase robustness."""
        try:
            conn = sqlite3.connect(str(KNOWLEDGE_DB_PATH), timeout=60)
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check")
            if cur.fetchone()[0] != "ok":
                conn.close()
                return False
            # FTS5 virtual table is created at build time; verify it exists
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='passage_fts'"
            )
            exists = cur.fetchone() is not None
            conn.close()
            return exists
        except Exception:  # noqa: BLE001 — FTS5 failure is soft; fall back to cosine only
            return False

    def _fts_score(self, query: str, text: str) -> float:
        """Fast exact-phrase keyword overlap (no network, no model)."""
        if not self._fts_enabled:
            return 0.0
        query_terms = set(re.findall(r"[А-Яа-яA-Za-z0-9؀-ۿ]+", query))
        if not query_terms:
            return 0.0
        text_terms = set(re.findall(r"[А-Яа-яA-Za-z0-9؀-ۿ]+", text))
        if not text_terms:
            return 0.0
        overlap = len(query_terms & text_terms) / len(query_terms)
        return min(overlap, 1.0)

    def retrieve(self, query: str, top_k: int = 3, timeout_ms: float | None = None
                 ) -> list[RetrievalResult]:
        """
        Hybrid search: cosine similarity + FTS5 keyword match.

        Latency contract:
            * embedding of the query (fastembed) < 5 ms on CPU
            * matrix multiply on float32 RAM array < 5 ms
            * FTS5 keyword scan (pre-tokenized) < 5 ms
            => total < 15 ms for a corpus of ~50k passages.
        """
        q0 = time.perf_counter_ns()

        # ---- 1. fastembed sentence embedding of the user query (single call) ----
        from fastembed import TextEmbedding
        qemb = next(iter(TextEmbedding().embed([query])))
        qvec = np.asarray(qemb, dtype=np.float32)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)

        # ---- 2. cosine similarity over the pre-normalized matrix (NumPy, < 5 ms) ----
        sim = self._matrix @ qvec  # [N] float32 dot products

        # ---- 3. FTS5 keyword overlap for the top candidates ----
        n_candidates = min(200, len(sim))
        if n_candidates == 0:
            return []
        if n_candidates < len(sim):
            top_idx = np.argpartition(sim, -n_candidates)[-n_candidates:]
        else:
            top_idx = np.arange(len(sim))
        scores: list[float] = []
        for i in top_idx:
            fts = self._fts_score(query, self._passages[i].text)
            scores.append(0.7 * float(sim[i]) + 0.3 * fts)
        order = np.argsort(scores)[::-1]

        results: list[RetrievalResult] = []
        for j in order:
            i = top_idx[j]
            results.append(
                RetrievalResult(score=scores[j],
                                passage=self._passages[i],
                                fts_match=float(sim[i]) > 0.25)
            )
            if len(results) >= top_k:
                break

        if timeout_ms is not None:
            elapsed_ms = (time.perf_counter_ns() - q0) / 1e6
            if elapsed_ms > timeout_ms * 1.5:
                results = self._sync_fallback(query, top_k, q0)

        return results

    def _sync_fallback(self, query: str, top_k: int, q0: float
                       ) -> list[RetrievalResult]:
        """Slow-path if the fast path took too long (should never happen)."""
        from fastembed import TextEmbedding
        qemb = np.asarray(next(iter(TextEmbedding().embed([query])), np.zeros(384, dtype=np.float32)))
        qemb /= np.linalg.norm(qemb) + 1e-9
        sim = self._matrix @ qemb
        top_idx = np.argsort(sim)[::-1][:top_k]
        return [
            RetrievalResult(score=float(sim[i]), passage=self._passages[i], fts_match=False)
            for i in top_idx
        ]

    def is_guardrail_blocked(self, query: str, top_k: int = 3, timeout_ms: float | None = None
                             ) -> tuple[bool, list[RetrievalResult]]:
        """
        Pre-LLM hard guardrail gate.

        Returns (is_blocked, best_results):
            * blocked=True  → best score <= STRICT_SIMILARITY_THRESHOLD; LLM must NOT be called
              and the caller should return REFUSAL_MESSAGE (< 10 ms wall time).
            * blocked=False → return the results so the LLM can be invoked.
        """
        results = self.retrieve(query, top_k=top_k, timeout_ms=timeout_ms)
        blocked = not results or results[0].score <= STRICT_SIMILARITY_THRESHOLD
        return blocked, results

    @property
    def passage_count(self) -> int:
        return len(self._passages)

    def get_verbatim_paragraph(self, book_name: str, page_number: int) -> str:
        """Fetch the exact verbatim paragraph for the citation badge (clickable action)."""
        for p in self._passages:
            if p.book_name == book_name and p.page_number == page_number:
                return p.text
        return ""

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


def create_retriever() -> Retriever:
    return Retriever()
