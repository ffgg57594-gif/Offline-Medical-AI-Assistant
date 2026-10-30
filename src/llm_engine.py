"""
llm_engine.py  —  Offline GGUF inference via llama-cpp-python with clause streaming.

Pinning contract:
    * The LLM model is loaded ONCE at process start and stays pinned in RAM/VRAM
      for the entire app session (no reload on each turn).
    * Generation is locked to temperature=0.0, top_p=0.1, repeat_penalty=1.1.
    * n_ctx=2048, n_batch=512, threads = CPU/GPU-optimized.

Streaming contract:
    * generate_stream() yields (full_text, completion_block, token) per token.
    * The caller can forward tokens to the GUI chat bubble AND to the live_voice
      streaming TTS pipeline simultaneously → first word rendered/speaks < 0.6 s.

Architecture:
    * Single threaded inference (llama-cpp-python blocks the thread) — the caller
      runs it on a background thread so the UI stays buttery smooth.
    * Pre-computed system prompt + citation metadata are injected per query.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator, NamedTuple

from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequest

from .config import (
    N_BATCH,
    N_CTX,
    REPEAT_PENALTY,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    VOICE_MODELS_DIR,
)


class Citation(NamedTuple):
    book_name: str
    page_number: int


class LLMEngine:
    """Singleton offline GGUF engine, pinned in memory for the app session."""

    _instance: "LLMEngine | None" = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "LLMEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_path: Path | None = None,
        gpu_layers: int | None = None,
        threads: int | None = None,
    ) -> None:
        if self._initialized:
            return
        self._model_path = model_path or self._resolve_model()
        self._llm: Llama | None = None
        self._loaded_at_ns: float | None = None
        self._lock = threading.Lock()

        cpu_count = __import__("os").cpu_count()
        self._threads = threads or max(2, cpu_count // 2)
        self._gpu_layers = gpu_layers or self._detect_gpu_layers()

        self._load_model()
        self._initialized = True

    def _resolve_model(self) -> Path:
        """Resolve the bundled GGUF (CI download place it under assets/voice_models/)."""
        candidates = [
            Path(VOICE_MODELS_DIR) / "phi3mini4k.Q4_K_M.gguf",
            Path(VOICE_MODELS_DIR) / "phi-3-mini-4k-instruct.Q4_K_M.gguf",
            list(Path(VOICE_MODELS_DIR).glob("*.gguf")),
        ]
        for c in candidates:
            if isinstance(c, list) and c:
                return c[0]
            if isinstance(c, Path) and c.exists():
                return c
        raise FileNotFoundError(
            "No bundled GGUF found.  Run `python scripts/download_models.py` first."
        )

    def _detect_gpu_layers(self) -> int:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            n_gpu = min(int(has_cuda), 99)
            return n_gpu
        except Exception:  # noqa: BLE001
            return 0

    def _load_model(self) -> None:
        """Load the model once; pinned in RAM/VRAM for the session."""
        t0 = time.perf_counter_ns()
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_threads=self._threads,
            n_gpu_layers=self._gpu_layers,
            use_mlock=True if self._gpu_layers == 0 else False,
            verbose=False,
        )
        self._loaded_at_ns = time.perf_counter_ns() - t0
        print(f"[llm] loaded {self._model_path.name} ({self._gpu_layers} GPU layers, "
              f"{self._threads} threads) in {self._loaded_at_ns / 1e6:.0f} ms")

    def generate_stream(
        self,
        query: str,
        system_prompt: str = SYSTEM_PROMPT,
        on_token: Callable[[str], None] | None = None,
        cite_result: Citation | None = None,
        max_tokens: int = 512,
    ) -> Generator[tuple[str, str, str], None, None]:
        """
        Stream LLM tokens.

        Yields (full_text_so_far, token, timestamp_ms):
            * full_text_so_far — the complete answer accumulated so far (GUI chat bubble)
            * token — the raw generated token (can be split into words)
            * timestamp_ms — wall time since first token

        The citation badge is appended once generation completes (or immediately if
        cite_result is provided by the guardrail layer).
        """
        t0 = time.perf_counter_ns()

        messages: ChatCompletionRequest["ChatCompletionMessageParam"] = [  # type: ignore
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        first_token_at: float | None = None

        for tok in self._llm.create_chat_completion(  # type: ignore
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
            max_tokens=max_tokens,
            stream=True,
        ):
            chunk = tok.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not chunk:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter_ns()
            full = getattr(self, "_stream_so_far", "") + chunk
            self._stream_so_far = full
            if on_token is not None:
                on_token(chunk)
            ms = (time.perf_counter_ns() - t0) / 1e6 if first_token_at is not None else 0.0
            yield full, chunk, ms

        # ---- append citation badge at the end ----
        badge = self._build_badge(cite_result)
        full = getattr(self, "_stream_so_far", "") + badge
        self._stream_so_far = full
        if on_token is not None:
            on_token(badge)
        yield full, badge, (time.perf_counter_ns() - t0) / 1e6

    def _build_badge(self, cite_result: Citation | None) -> str:
        if cite_result is None:
            return ""
        return f"\n[المصدر: {cite_result.book_name} - صفحة {cite_result.page_number}]"

    def get_first_token_latency_ms(self) -> float | None:
        """How long after load did the first token stream? (diagnostic)."""
        return getattr(self, "_first_token_latency_ms", None)

    def shutdown(self) -> None:
        if self._llm is not None:
            self._llm.close()
            self._llm = None
            self._initialized = False
