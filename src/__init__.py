"""
Offline Strict Medical Textbooks Chat & Live Voice GUI Application.

100% closed-book — answers come ONLY from the bundled textbook knowledge base
(assets/books_knowledge.db + float32 embedding matrix).  A pre-LLM hybrid search
guardrail blocks out-of-scope queries at < 10 ms.
"""

from .config import (
    STRICT_SIMILARITY_THRESHOLD,
    TEMPERATURE,
    TOP_P,
    REPEAT_PENALTY,
    N_CTX,
    N_BATCH,
    REFUSAL_MESSAGE,
)

__version__ = "1.0.0"
__all__ = [
    "STRICT_SIMILARITY_THRESHOLD",
    "TEMPERATURE",
    "TOP_P",
    "REPEAT_PENALTY",
    "N_CTX",
    "N_BATCH",
    "REFUSAL_MESSAGE",
]
