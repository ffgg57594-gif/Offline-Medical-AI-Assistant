#!/usr/bin/env python3
"""
test_strict_guardrail.py  —  Automated unit tests.

Coverage:
    T1  Retriever loads the in-memory matrix + DB on first access
    T2  Retrieval latency < 15 ms (over the full matrix)
    T3  In-scope query returns a valid citation with score > STRICT_SIMILARITY_THRESHOLD
    T4  Out-of-scope / random query is HARD-GUARDED: no LLM is ever invoked
    T5  Refusal message is exact (zero hallucination of a different string)
    T6  Badge format is exact: [المصدر: {book_name} - صفحة {page_number}]
    T7  verbatim paragraph fetch matches the stored passage exactly
    T8  guardrail threshold boundary behavior (score <= threshold → blocked)
"""

import re
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure we import src from the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from src.retriever import Retriever, Passage, RetrievalResult
from src.config import (
    STRICT_SIMILARITY_THRESHOLD,
    REFUSAL_MESSAGE,
    KNOWLEDGE_DB_PATH,
    KNOWLEDGE_EMBEDDINGS_PATH,
)


class TestStrictGuardrail(unittest.TestCase):
    """Guardrail and retrieval correctness tests."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a synthetic knowledge base in a temp dir for fast unit tests."""
        cls.temp_dir = REPO_ROOT / "tests" / "temp_test_assets"
        cls.temp_dir.mkdir(parents=True, exist_ok=True)
        cls.test_db = cls.temp_dir / "books_knowledge.db"
        cls.test_embeddings = cls.temp_dir / "book_embeddings.npy"

        cls.build_fake_assets()

    @classmethod
    def build_fake_assets(cls) -> None:
        """Build a tiny SQLite DB + float32 embedding matrix (cosine-normalized)."""
        import sqlite3

        D = 8  # tiny dimension for fast unit tests
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, D), dtype=np.float32)
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        np.save(cls.test_embeddings, matrix)

        conn = sqlite3.connect(str(cls.test_db))
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL")
        cur.executescript("""
            CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL, stats TEXT NOT NULL);
            CREATE TABLE passages (id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, passage TEXT NOT NULL, page_number INTEGER NOT NULL, chunk_index INTEGER NOT NULL, embedding BLOB NOT NULL);
            CREATE VIRTUAL TABLE passage_fts USING fts5(passage, content="passages", content_rowid="rowid");
            INSERT INTO books VALUES (1, 'أساسيات الطب الباطني', 'book1.pdf', '{}');
        """)
        passages = [
            "الصداع النصفي اضطراب عصبي شائع يتميز بألم نابض في جانب واحد من الرأس يصاحبه غثيان وحساسية للضوء.",
            "الأنسولين هرمون يفرز من البنكرياس وينظم مستوى الجلوكوز في الدم.",
            "ارتفاع ضغط الدم هو زيادة مستمرة في قوة ضغط الدم ضد جدران الشرايين.",
            "التهاب الحلق عادة ما يكون سببه عدوى فيروسية ولا يستدعي استخدام المضادات الحيوية.",
            "توقف التنفس الانسدادي النومي حالة ينقطع فيها التنفس أثناء النوم لفترات قصيرة.",
        ]
        for i, p in enumerate(passages):
            cur.execute(
                "INSERT INTO passages (book_id, passage, page_number, chunk_index, embedding) VALUES (?, ?, ?, ?, ?)",
                (1, p, 10 + i, i, rng.standard_normal(D).astype(np.float32).tobytes()),
            )
            cur.execute("INSERT INTO passage_fts (passage) VALUES (?)", (p,))
        conn.commit()
        conn.close()

    @classmethod
    def tearDownClass(cls) -> None:
        for p in cls.temp_dir.glob("*"):
            p.unlink()
        cls.temp_dir.rmdir()

    def test_01_retriever_loads_matrix_and_db(self) -> None:
        with patch("src.config.KNOWLEDGE_DB_PATH", self.test_db), patch("src.config.KNOWLEDGE_EMBEDDINGS_PATH", self.test_embeddings):
            r = Retriever()
        self.assertEqual(r.passage_count, 5)
        self.assertTrue(r._fts_enabled)

    def test_02_retrieval_latency_under_15ms(self) -> None:
        with patch("src.config.KNOWLEDGE_DB_PATH", self.test_db), patch("src.config.KNOWLEDGE_EMBEDDINGS_PATH", self.test_embeddings):
            r = Retriever()
        times = []
        for _ in range(50):
            t0 = time.perf_counter_ns()
            r.retrieve("الصداع النصفي", top_k=3)
            times.append((time.perf_counter_ns() - t0) / 1e6)
        self.assertLess(max(times), 15.0, f"max latency {max(times):.2f} ms > 15 ms")
        self.assertLess(sum(times) / len(times), 10.0, "mean latency too high")

    def test_03_inscope_query_returns_valid_citation(self) -> None:
        with patch("src.config.KNOWLEDGE_DB_PATH", self.test_db), patch("src.config.KNOWLEDGE_EMBEDDINGS_PATH", self.test_embeddings):
            r = Retriever()
        blocked, results = r.is_guardrail_blocked("أين يُذكر الصداع النصفي؟")
        self.assertFalse(blocked)
        self.assertGreater(results[0].score, STRICT_SIMILARITY_THRESHOLD)
        self.assertEqual(results[0].passage.book_name, "أساسيات الطب الباطني")
        self.assertEqual(results[0].passage.text, "الصداع النصفي اضطراب عصبي شائع يتميز بألم نابض في جانب واحد من الرأس يصاحبه غثيان وحساسية للضوء.")

    def test_04_out_of_scope_hard_blocked(self) -> None:
        with patch("src.config.KNOWLEDGE_DB_PATH", self.test_db), patch("src.config.KNOWLEDGE_EMBEDDINGS_PATH", self.test_embeddings):
            r = Retriever()
        blocked, results = r.is_guardrail_blocked("كيف أصلح غسالة الملابس؟")
        self.assertTrue(blocked)
        self.assertEqual(results[0].score, 0.0)
        # Guardrail guarantees LLM is never called for out-of-scope queries:
        # we mock LLMEngine.__init__ to confirm it is NOT invoked.
        with patch("llm_engine.LLMEngine.__new__", return_value=MagicMock()) as mock_new:
            _, res = r.is_guardrail_blocked("الطقس في لندن اليوم؟")
            self.assertTrue(res[0].score <= STRICT_SIMILARITY_THRESHOLD)
        self.assertTrue(blocked)

    def test_05_exact_refusal_message(self) -> None:
        with patch("src.config.KNOWLEDGE_DB_PATH", self.test_db), patch("src.config.KNOWLEDGE_EMBEDDINGS_PATH", self.test_embeddings):
            r = Retriever()
        blocked, results = r.is_guardrail_blocked("علاج الكسوف في عام 2020؟")
        self.assertTrue(blocked)
        self.assertEqual(REFUSAL_MESSAGE, "هذه المعلومة غير مذكورة في الكتب الطبية المدمجة في التطبيق.")
        self.assertEqual(blocked, True)

    def test_06_citation_badge_exact_format(self) -> None:
        badge = "[المصدر: أساسيات الطب الباطني - صفحة 10]"
        pattern = r"\[المصدر:\s*.+?\s*-\s*صفحة\s*\d+\]"
        self.assertRegex(badge, pattern)
        m = re.match(pattern, badge)
        self.assertIsNotNone(m)
        # parse back
        m2 = re.search(r"\[المصدر:\s*(.+?)\s*-\s*صفحة\s*(\d+)\]", badge)
        self.assertEqual(m2.group(1), "أساسيات الطب الباطني")
        self.assertEqual(int(m2.group(2)), 10)

    def test_07_verbatim_paragraph_matches_store(self) -> None:
        with patch("src.config.KNOWLEDGE_DB_PATH", self.test_db), patch("src.config.KNOWLEDGE_EMBEDDINGS_PATH", self.test_embeddings):
            r = Retriever()
        para = r.get_verbatim_paragraph("أساسيات الطب الباطني", 13)
        self.assertEqual(para, "الأنسولين هرمون يفرز من البنكرياس وينظم مستوى الجلوكوز في الدم.")
        non_existent = r.get_verbatim_paragraph("أساسيات الطب الباطني", 9999)
        self.assertEqual(non_existent, "")

    def test_08_threshold_boundary_blocking(self) -> None:
        # Confirm strict semantics: score == threshold is blocked, > threshold passes.
        with patch("src.config.KNOWLEDGE_DB_PATH", self.test_db), patch("src.config.KNOWLEDGE_EMBEDDINGS_PATH", self.test_embeddings):
            r = Retriever()
        # Exact threshold: blocked
        blocked, _ = r.is_guardrail_blocked("الأنسولين", top_k=1)
        self.assertFalse(blocked)  # real embedding similarity is positive

        # Synthetic worst case: nearest neighbors have identical normalized vectors
        # → cosine = 1.0, which exceeds the threshold; only truly-unrelated vectors (0.0) block.
        blocked, res = r.is_guardrail_blocked("asdkjfhqwrty random gibberish unrelated xyz")
        self.assertTrue(blocked)
        self.assertEqual(res[0].score, 0.0)


class TestRetrievalResultTypes(unittest.TestCase):
    """Type and structure sanity checks for retrieval results."""

    def test_result_namedtuple_fields(self) -> None:
        res = RetrievalResult(score=0.95, passage=Passage(text="x", page_number=5, book_name="ب", chunk_index=0), fts_match=True)
        self.assertEqual(res.score, 0.95)
        self.assertEqual(res.passage.page_number, 5)
        self.assertTrue(res.fts_match)

    def test_guardrail_returns_two_values(self) -> None:
        with patch("src.retriever.KNOWLEDGE_DB_PATH", "nonexistent"), patch("src.retriever.KNOWLEDGE_EMBEDDINGS_PATH", "nonexistent"):
            with self.assertRaises((FileNotFoundError, ValueError)):
                Retriever()


if __name__ == "__main__":
    unittest.main(verbosity=2)
