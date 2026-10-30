"""
config.py  —  PyInstaller-safe resource paths, thresholds, and strict system prompts.

All asset paths resolve correctly whether the app runs from source (DEBUG=1) or
from inside a PyInstaller bundle (sys._MEIPASS).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Strict RAG thresholds (non-negotiable)
# ---------------------------------------------------------------------------
STRICT_SIMILARITY_THRESHOLD: Final[float] = 0.62  # best hybrid score must exceed this

# Strict generation lock (hard guardrail against hallucination)
TEMPERATURE: Final[float] = 0.0
TOP_P: Final[float] = 0.1
REPEAT_PENALTY: Final[float] = 1.1

# llama-cpp-python pinning (model stays in RAM for the whole session)
N_CTX: Final[int] = 2048
N_BATCH: Final[int] = 512

# Latency budgets (enforced by the live_voice module, not asserted here)
GOAL_FIRST_WORD_MS: Final[int] = 600
GOAL_AUDIO_END_TO_END_MS: Final[int] = 800

# ---------------------------------------------------------------------------
# Strict system prompt (Arabic, because the bundled corpus is Arabic)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: Final[str] = (
    "أنت مساعد طبي دقيق يعتمد على الكتب الطبية المدمجة فقط. "
    "لا تجب إلا من المصادر الواردة في قاعدة الكتب المدمجة في التطبيق. "
    "إذا لم تكن المعلومة مذكورة صراحة في الكتب، قل: "
    "\"هذه المعلومة غير مذكورة في الكتب الطبية المدمجة في التطبيق.\" "
    "بدون أي تأويل أو إضافة. "
    "إذا كانت المعلومة موجودة، أجب بدقة وبأسلوب طبي موجز، وانهِ إجابتك دائمًا بالشارة التالية: "
    "[المصدر: {book_name} - صفحة {page_number}] "
    "مع الاستشهاد بأول نتيجة تطابق تحقق. "
    "لا تتحدث أبدًا عن معلومات خارج نطاق الكتب. "
    "اجعل إجابتك مقسمة إلى عبارات قصيرة (جمل أو شروط مفصولة بنقطة أو فاصلة) لخدمة نظام نطق متدفق. "
)

# Strict guardrail — returned immediately when hybrid retrieval score < threshold,
# BEFORE any LLM call.  Must be under ~10 ms wall time.
REFUSAL_MESSAGE: Final[str] = (
    "هذه المعلومة غير مذكورة في الكتب الطبية المدمجة في التطبيق."
)

# Voice mode defaults
SAMPLE_RATE: Final[int] = 16000
FRAME_LENGTH_MS: Final[int] = 30
SILENCE_THRESHOLD_MS: Final[int] = 300
BARGE_IN_LATENCY_MS: Final[int] = 50
VAD_SILENCE_FRAMES: Final[int] = 10
VAD_SPEECH_FRAMES: Final[int] = 3
VAD_THRESHOLD: Final[float] = 0.5

# STT (faster-whisper) — primed with the medical glossary at build time
STT_MODEL: Final[str] = "tiny"  # Q: 2.
STT_BATCH: Final[int] = 8
STT_LANGUAGE: Final[str] = "ar"  # Q: 2.
STT_INITIAL_PROMPT: Final[str] = (
    "مساعد طبي. الكلمات: تشريح، فسيولوجيا، صيدلة، تشخيص، علاج، أعراض، تشخيص، "
    "دواء، جرعة، بروتوكول، مخطط، مخبر، أشعة، قسطرة، جراحة، أمراض، قلب، رئوي، عصبي، "
    "غدد، أطفال، نساء، جلدي، أذن، أنف، عيون، عظام، مسالك، أورام، طوارئ."
)

# TTS (piper) streaming
TTS_MODEL: Final[str] = "arabic-standard.onnx"
TTS_STREAMING: Final[str] = True
TTS_SAMPLE_RATE: Final[int] = 24000
TTS_SPEAKER: Final[int] = 0
TTS_SPEED: Final[float] = 1.0

# Clause splitter (first clause < 40 ms)
CLAUSE_MIN_CHARS: Final[int] = 12
CLAUSE_MAX_CHARS: Final[int] = 90
CLAUSE_SPLITTERS: Final[re.Pattern] = re.compile(r"(?<=[.!?.,;:])\s+")

# ---------------------------------------------------------------------------
# Asset paths — PyInstaller-aware; walk up the tree to locate the repo root
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    ROOT_DIR: Final[Path] = Path(sys._MEIPASS)
else:
    _FILE_DIR = Path(__file__).resolve().parent
    _ROOT_DIR: Path | None = None
    for _candidate in (_FILE_DIR.parents):
        if (_candidate / "assets").exists():
            _ROOT_DIR = _candidate
            break
    if _ROOT_DIR is None:  # project not under a root dir; fall back to project root
        _ROOT_DIR = _FILE_DIR.parent
    ROOT_DIR = _ROOT_DIR

ASSETS_DIR: Final[Path] = ROOT_DIR / "assets"
if not ASSETS_DIR.exists():  # fallback when running from unpacked PyInstaller dir
    ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
    if not ASSETS_DIR.exists():
        ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"

VOICE_MODELS_DIR: Final[Path] = ASSETS_DIR / "voice_models"
KNOWLEDGE_DB_PATH: Final[Path] = ASSETS_DIR / "books_knowledge.db"
KNOWLEDGE_EMBEDDINGS_PATH: Final[Path] = ASSETS_DIR / "book_embeddings.npy"
GLOSSARY_PATH: Final[Path] = ASSETS_DIR / "medical_glossary.json"
REFUSAL_WAV_PATH: Final[Path] = ASSETS_DIR / "refusal_ar.wav"

# sanity check on startup — missing asset means build pipeline was never run
if not hasattr(sys, "_MEIPASS") and not os.environ.get("TESTING"):
    _MISSING = []
    for p in (KNOWLEDGE_DB_PATH, KNOWLEDGE_EMBEDDINGS_PATH, VOICE_MODELS_DIR):
        if not p.exists():
            _MISSING.append(str(p))
    if _MISSING:
        raise FileNotFoundError(
            f"Missing pre-built assets — run the GitHub Actions pipeline or "
            f"`python scripts/download_models.py && python scripts/build_knowledge_db.py --force`:\n  " + "\n  ".join(_MISSING)
        )
