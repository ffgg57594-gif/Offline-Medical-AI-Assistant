#!/usr/bin/env python3
"""
make_dummy_assets.py — Generate synthetic assets for quick local testing.

This replaces the real build pipeline (scripts/build_knowledge_db.py +
scripts/download_models.py) so the unit tests and the GUI can be exercised
without a network connection or real textbook PDFs.  It is NOT included in
the final PyInstaller bundle.
"""

import os
import sys
import sqlite3
import json
import wave
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent / "assets"
BASE.mkdir(exist_ok=True)
shutil.rmtree(BASE / "voice_models", ignore_errors=True)
(BASE / "voice_models").mkdir(exist_ok=True)

# --- drop any old assets so the Retriever sees a consistent schema ---
for p in (BASE / "books_knowledge.db", BASE / "book_embeddings.npy"):
    if p.exists():
        p.unlink()

# --- tiny float32 embedding matrix (384-dim to match fastembed output) ---
import numpy as np
rng = np.random.default_rng(7)
matrix = rng.standard_normal((10, 384), dtype=np.float32)
matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
np.save(str(BASE / "book_embeddings.npy"), matrix)

# --- schema exactly matching Retriever / LLMEngine expectations:
#     books, passages(embedding), FTS5 table passage_fts over `passage` ---
DB = BASE / "books_knowledge.db"
conn = sqlite3.connect(str(DB))
cur = conn.cursor()
cur.execute("PRAGMA journal_mode = WAL")
cur.execute(
    "CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL, stats TEXT NOT NULL)"
)
cur.execute(
    "CREATE TABLE passages (id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL, passage TEXT NOT NULL, "
    "page_number INTEGER NOT NULL, chunk_index INTEGER NOT NULL, embedding BLOB NOT NULL)"
)
cur.execute(
    "CREATE VIRTUAL TABLE passage_fts USING fts5(passage, content='passages', content_rowid='rowid')"
)
cur.execute(
    "INSERT INTO books VALUES (1, 'أساسيات الطب الباطني', 'book1.pdf', "
    "'{\"pages\": 1240, \"chunks\": 8920, \"chars\": 3145728}') "
)
cur.execute(
    "INSERT INTO books VALUES (2, 'مرجع التشريح السريري', 'book2.pdf', "
    "'{\"pages\": 618, \"chunks\": 4510, \"chars\": 1572864}') "
)
passages = [
    (1, "الصداع النصفي اضطراب عصبي شائع يتميز بألم نابض في جانب واحد من الرأس يصاحبه غثيان وحساسية للضوء.", 10),
    (1, "الأنسولين هرمون يفرز من البنكرياس وينظم مستوى الجلوكوز في الدم.", 13),
    (1, "ارتفاع ضغط الدم هو زيادة مستمرة في قوة ضغط الدم ضد جدران الشرايين.", 45),
    (2, "القلب عضو عضلي يقع في التجويف الصدري بين الرئتين, يوصل الدم إلى جميع أنحاء الجسم.", 7),
    (2, "الجهاز التنفسي هو المسؤول عن تبادل الغازات في الجسم بين الهواء والدم.", 24),
    (2, "الجهاز الهضمي يمتص العناصر الغذائية ويحول الطعام إلى طاقة للجسم.", 41),
    (1, "التهاب الحلق عادة ما يكون سببه عدوى فيروسية ولا يستدعي استخدام المضادات الحيوية.", 88),
    (1, "توقف التنفس الانسدادي النومي حالة ينقطع فيها التنفس أثناء النوم لفترات قصيرة.", 112),
    (2, "العظام تشكل الهيكل الداعم للجسم وتحمي الأعضاء الداخلية.", 56),
    (1, "السكري مرض مزمن يؤثر على قدرة الجسم على تحويل الغذاء إلى طاقة.", 134),
]
for i, (book_id, text, page) in enumerate(passages):
    cur.execute(
        "INSERT INTO passages (book_id, passage, page_number, chunk_index, embedding) VALUES (?, ?, ?, ?, ?)",
        (book_id, text, page, i, rng.standard_normal(384).astype(np.float32).tobytes()),
    )
    cur.execute("INSERT INTO passage_fts (passage) VALUES (?)", (text,))
conn.commit()
conn.close()

# --- medical glossary primed into the STT initial_prompt ---
json.dump(
    ["رئة", "قلب", "جهاز تنفسي", "هضمي", "دموي", "أساسيات الطب الباطني", "تشريح",
     "صداع نصفي", "أنسولين", "جلوكوز", "ضغط دم", "التهاب", "عظام", "سكري"],
    open(str(BASE / "medical_glossary.json"), "w"),
    ensure_ascii=False,
)

# --- pre-recorded refusal clip for 0ms playback ---
rate = 16000
with wave.open(str(BASE / "refusal_ar.wav"), "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(b"\x00\x00" * (rate * 1))
print("dummy assets created OK")
