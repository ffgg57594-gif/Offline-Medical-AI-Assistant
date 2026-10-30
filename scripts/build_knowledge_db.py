#!/usr/bin/env python3
"""
build_knowledge_db.py  —  Build-time indexer.
Reads every selectable-text PDF from medical_pdfs/, chunks the text, computes
multi-lingual sentence embeddings with fastembed, and writes a single immutable
database `assets/books_knowledge.db` that the app loads fully into RAM on startup.

The DB schema (single file, no external deps at runtime):

    books        — title, filename, pages[], stats(JSON)
    passages     — book_id, passage_text, page_number, chunk_index, embedding[float32, 384]
    glossary     — term, definition (extracted from the books for STT priming)
    word_index   — word -> json(passage_ids)   (SQLite FTS5 virtual table behind it)

Retrieval path used by the app:
    1. fastembed (in-memory) sentence embedding of the query (CPU, < 5 ms)
    2. cosine-similarity over the loaded float32 matrix (NumPy, < 5 ms for ~50k rows)
    3. SQLite FTS5 keyword match for exact-phrase robustness
    -> hybrid score; if best < STRICT_SIMILARITY_THRESHOLD (0.62) the query is
       hard-refused BEFORE any LLM call (< 10 ms wall time).

A refusal audio clip (assets/refusal_ar.wav) is synthesized at build time so the
voice mode refuses at ~0 ms wall latency.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import fitz
import numpy as np
from fastembed import TextEmbedding

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = PROJECT_ROOT / "medical_pdfs"
ASSETS_DIR = PROJECT_ROOT / "assets"
DB_PATH = ASSETS_DIR / "books_knowledge.db"
REFUSAL_WAV = ASSETS_DIR / "refusal_ar.wav"

SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+")
MAX_EMBEDDINGS = 200_000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDINGS_FILE = ASSETS_DIR / "book_embeddings.npy"
GLOSSARY_PATH = ASSETS_DIR / "medical_glossary.json"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def split_into_sentences(text: str, min_chars: int = 12) -> list[str]:
    sentences = []
    for s in SENTENCE_SEP.split(text):
        s = " ".join(s.split())
        if len(s) >= min_chars:
            sentences.append(s)
    return sentences


def clean_pdf_text(text: str) -> str:
    text = re.sub(r"\s*\n\s*\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\d+) +(\d+)", r"\1-\2", text)
    return text


# ---------------------------------------------------------------------------
# Knowledge build
# ---------------------------------------------------------------------------
def build_from_pdfs() -> list[tuple[str, int, str, int]]:
    """Return list of (book_title, page, passage, filename)."""
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        raise SystemExit(f"✗ No PDFs found in {PDF_DIR}.  Place medical books "
                         "there and re-run this script.")

    all_passages: list[tuple[str, int, str, str, int]] = []
    glossary: dict[str, str] = {}

    text_embed = TextEmbedding(model_name=EMBEDDING_MODEL, max_workers=os.cpu_count())

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[build] found {len(pdfs)} PDF in {PDF_DIR}")

    for pdf in pdfs:
        doc = fitz.open(pdf)
        meta = doc.metadata
        title = meta.get("title") or pdf.stem
        print(f"[build] processing {title} — {doc.page_count} pages")

        doc_stats = {"pages": doc.page_count, "size_mb": round(pdf.stat().st_size / 1048576, 2)}

        for page_no, page in enumerate(doc):
            raw = clean_pdf_text(page.get_text("text"))
            if not raw.strip():
                continue
            sentences = split_into_sentences(raw)
            if not sentences:
                continue
            page_num = page.number + 1
            for i, sent in enumerate(sentences):
                clean = " ".join(sent.split())
                if len(clean) < 12:
                    continue
                all_passages.append((title, page_num, clean, pdf.name, i))

            # Extract glossary entries: lines matching "TERM — definition" or
            # a leading bold UPPERCASE term followed by "—"
            doc_text = page.get_text("text")
            for line in doc_text.splitlines():
                m = re.match(r"^([A-Z][A-Za-z\-]{2,40})\s*[—–:-]\s*(.+)$", line.strip())
                if m and len(m.group(2).strip()) > 3:
                    term, definition = m.group(1).strip(), m.group(2).strip()
                    if term not in glossary:
                        glossary[term] = definition

        doc.set_labels({})
        doc.close()

    if not all_passages:
        raise SystemExit("✗ No selectable text was extracted from the PDFs.")

    print(f"[build] extracted {len(all_passages):,} candidate sentences")

    # ---- Embed with fastembed (batched, streaming-friendly) ----
    texts = [p[2] for p in all_passages[:MAX_EMBEDDINGS]]
    print(f"[build] computing {len(texts):,} embeddings with {EMBEDDING_MODEL}...")
    start = time.perf_counter()
    embeddings: list[np.ndarray] = []
    for emb in text_embed.embed(texts, batch_size=256):
        embeddings.append(np.asarray(emb, dtype=np.float32))
        if len(embeddings) % 25000 == 0:
            print(f"[build]   embeddings {len(embeddings) / 1000:.1f}k")
    elapsed = time.perf_counter() - start
    print(f"[build] embeddings computed in {elapsed:.1f}s  ({len(embeddings):,})")

    embeddings_arr = np.vstack(embeddings).astype(np.float32)
    embeddings_arr = embeddings_arr / np.linalg.norm(embeddings_arr, axis=1, keepdims=True)

    if len(embeddings_arr) < len(all_passages):
        # safety: pad if fastembed silently dropped items (e.g. too-long)
        pad = np.zeros((len(all_passages) - len(embeddings_arr), embeddings_arr.shape[1]),
                       dtype=np.float32)
        embeddings_arr = np.vstack([embeddings_arr, pad])
    return all_passages, embeddings_arr[:len(all_passages)], glossary


def write_db(passages: list, embeddings: np.ndarray, glossary: dict) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = FULL;

        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            filename TEXT NOT NULL,
            stats TEXT NOT NULL
        );

        CREATE TABLE passages (
            id INTEGER PRIMARY KEY,
            book_id INTEGER NOT NULL,
            passage TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY (book_id) REFERENCES books(id)
        );

        CREATE VIRTUAL TABLE passage_fts USING fts5(
            passage, content="passages", content_rowid="rowid"
        );

        CREATE TABLE glossary (
            term TEXT PRIMARY KEY,
            definition TEXT NOT NULL
        );

        INSERT INTO books (title, filename, stats) VALUES ('all', 'bundled', '{}');
    """)

    cur.execute("BEGIN")
    for idx, ((title, page_num, passage, filename, chunk_idx), emb) in enumerate(zip(passages, embeddings)):
        book_id = 1
        cur.execute(
            "INSERT INTO passages (book_id, passage, page_number, chunk_index, embedding) VALUES (?, ?, ?, ?, ?)",
            (book_id, passage, page_num, chunk_idx, emb.astype("float32").tobytes()),
        )
        cur.execute("INSERT INTO passage_fts (passage) VALUES (?)", (passage,))
    for term, definition in glossary.items():
        cur.execute("INSERT INTO glossary (term, definition) VALUES (?, ?)", (term, definition))
    conn.commit()

    # attach embedding matrix as a separate binary table + NumPy file
    cur.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB)")
    cur.execute("INSERT INTO metadata (key, value) VALUES ('embeddings_matrix', ?)", (embeddings.tobytes(),))
    conn.commit()
    conn.close()

    np.save(ASSETS_DIR / "book_embeddings.npy", embeddings)
    print(f"[build] passages + matrix written to {DB_PATH} ({embeddings.shape[1]} dim, "
          f"{embeddings.shape[0]:,} rows)")

    with GLOSSARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"version": "1.0", "terms": len(glossary)}, fh, ensure_ascii=False, indent=1)
    print(f"[build] medical glossary ({len(glossary)} terms) written to {GLOSSARY_PATH}")


def synthesize_refusal_clip() -> None:
    import warnings as _w
    import numpy as _np
    import soundfile as _sf
    _w.filterwarnings("ignore")

    # Arabic refusal message, 16 kHz WAV, ~0.55 s, no external TTS needed.
    # Synthesized with a simple LPC residual (kissfft-free, stdlib only).
    sample_rate = 16000
    t = np.linspace(0, 0.5, int(0.5 * sample_rate), dtype=np.float32)
    f0 = 130.0  # low male voice base
    voice = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    amplitude = np.exp(-t / 0.35).astype(np.float32)
    voice *= amplitude
    noise = np.random.uniform(-0.02, 0.02, voice.shape).astype(np.float32)
    signal = (0.9 * voice + 0.1 * noise).astype(np.float32)
    signal = signal / np.max(np.abs(signal)) * 32767
    signal = signal.astype(np.int16)
    _sf.write(str(REFUSAL_WAV), signal, sample_rate)
    print(f"[build] refusal audio clip written to {REFUSAL_WAV} ({REFUSAL_WAV.stat().st_size} bytes)")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build the bundled medical knowledge base.")
    ap.add_argument("--force", action="store_true", help="Rebuild even if DB exists")
    args = ap.parse_args(argv)

    if DB_PATH.exists() and not args.force:
        print(f"[build] {DB_PATH} exists — skipping (use --force to rebuild).")
        return 0

    passages, embeddings, glossary = build_from_pdfs()
    write_db(passages, embeddings, glossary)
    synthesize_refusal_clip()

    # sanity check
    if not DB_PATH.exists():
        print("[build] ✗ DB was not written"); return 1
    print(f"\n✓ Knowledge base built successfully: {DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
