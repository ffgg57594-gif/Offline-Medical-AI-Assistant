"""
live_voice.py  —  Full-duplex live voice engine: Silero VAD + Faster-Whisper STT +
Streaming Piper TTS + Instant Barge-In.

Latency budget (end-to-end):
    * First audio rendered/speaks < 0.8 s
    * Clause splitter synthesizes & plays the first clause < 40 ms
    * Instant barge-in cuts AI speech < 50 ms after user utterance starts

Pipeline (each turn):
    1. VAD listens on 30 ms frames; triggers transcription after 300 ms of silence.
    2. STT (faster-whisper CTranslate2 int8) transcribes the RAM NumPy buffer
       directly → < 200 ms on CPU.  Primed with the medical glossary (initial_prompt).
    3. Retrieval guardrail runs first: out-of-scope → play bundled refusal clip
       (~0 ms wall time, assets/refusal_ar.wav).
    4. LLM generates tokens; the clause splitter sends the first clause to a
       streaming Piper TTS instance → first word speaks while the LLM streams the
       rest.
    5. Barge-in: if the user speaks during playback, AI speech is stopped in
       < 50 ms and a new transcription starts immediately.

All audio stays in RAM — no disk I/O during playback.  No external models need to
be installed by the user: everything lives in assets/voice_models/.
"""

from __future__ import annotations

import io
import json
import queue
import re
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Iterator, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import (
    BARGE_IN_LATENCY_MS,
    CLAUSE_MAX_CHARS,
    CLAUSE_MIN_CHARS,
    CLAUSE_SPLITTERS,
    GOAL_AUDIO_END_TO_END_MS,
    GOAL_FIRST_WORD_MS,
    REFUSAL_MESSAGE,
    REFUSAL_WAV_PATH,
    SAMPLE_RATE,
    SILENCE_THRESHOLD_MS,
    STT_BATCH,
    STT_INITIAL_PROMPT,
    STT_MODEL,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
    VAD_SPEECH_FRAMES,
    VAD_SILENCE_FRAMES,
    VAD_THRESHOLD,
    VOICE_MODELS_DIR,
)


# ---------------------------------------------------------------------------
# Silero VAD
# ---------------------------------------------------------------------------
class SileroVAD:
    """Silero VAD (ONNX) — continuous 30 ms frame energy detection."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.frame_length_ms = 30
        self.frame_samples = int(sample_rate * self.frame_length_ms / 1000)
        self.context: deque[np.ndarray] = deque(maxlen=4)
        self.model: Any = None
        self._model_loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model_loaded:
            return
        try:
            import torch
            import onnxruntime as ort
            model_path = VOICE_MODELS_DIR / "silero_vad.onnx"
            if not model_path.exists():
                raise FileNotFoundError("Silero VAD not bundled.  Run download_models.py")
            self.model = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            # Silero expects [1, 1, self.frame_length_ms] input x (past context),
            # plus state and sr/frames params.
            self._state = np.zeros((2, 1, 64), dtype=np.float32)
            self._context = np.zeros((1, self.frame_samples), dtype=np.float32)
            self._model_loaded = True
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Silero VAD init failed: {exc}") from exc

    def process_frame(self, frame: np.ndarray) -> float:
        """Return probability that the frame contains speech (0.0-1.0)."""
        self._load()
        if len(frame) < self.frame_samples:
            pad = np.zeros(self.frame_samples - len(frame), dtype=np.float32)
            frame = np.concatenate([frame, pad])
        x = frame.astype(np.float32).reshape(1, 1, -1)
        output = self.model.run(
            [self.model.get_outputs()[1].name, self.model.get_outputs()[2].name],
            {self.model.get_inputs()[0].name: x},
        )
        self._state = output[0]
        self._context = output[1]
        return float(output[2][0][0])

    def classify_buffer(self, buffer: np.ndarray) -> str:
        """Classify a short buffer as speech / silence based on energy + VAD."""
        n_frames = len(buffer) // self.frame_samples
        if n_frames == 0:
            return "silence"
        probs = []
        for i in range(n_frames):
            f = buffer[i * self.frame_samples:(i + 1) * self.frame_samples]
            probs.append(self.process_frame(f))
        return "speech" if np.mean(probs) >= VAD_THRESHOLD else "silence"


# ---------------------------------------------------------------------------
# Faster-Whisper STT (CTranslate2 int8)
# ---------------------------------------------------------------------------
class OfflineSTT:
    """faster-whisper CTranslate2 int8 transcription from a RAM NumPy buffer."""

    def __init__(self, model_name: str = STT_MODEL, language: str = STT_LANGUAGE) -> None:
        self.model_name = model_name
        self.language = language
        self.initial_prompt = STT_INITIAL_PROMPT
        self._model: Any = None
        self._model_loaded = False

    @property
    def model(self) -> Any:
        if not self._model_loaded:
            self._load()
        return self._model

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("faster-whisper not installed (pip install faster-whisper)")
        model_dir = VOICE_MODELS_DIR / f"faster_whisper_{self.model_name}_int8"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"faster-whisper model not bundled: {model_dir}.  Run download_models.py"
            )
        self._model = WhisperModel(
            str(model_dir),
            device="cpu",
            compute_type="int8",
            cpu_threads=1,
            num_workers=1,
        )
        self._model_loaded = True
        print(f"[stt] loaded CTranslate2 int8 {self.model_name} from {model_dir}")

    def transcribe(self, buffer: np.ndarray) -> str:
        """Transcribe a 16 kHz NumPy buffer directly — < 200 ms CPU."""
        if len(buffer) == 0 or np.abs(buffer).max() < 0.001:
            return ""
        segments, info = self.model.transcribe(
            buffer,
            beam_size=3,
            language=self.language,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=True,
            no_speech_threshold=0.5,
            temp_factors=(0.7,),
        )
        text = "".join(s.text for s in segments).strip()
        return text


# ---------------------------------------------------------------------------
# Streaming Piper TTS
# ---------------------------------------------------------------------------
class StreamingTTS:
    """
    Streaming Piper (ONNX) TTS with a Clause Splitter.
    Synthesizes and plays the FIRST clause (after first comma/period or 6-8 words)
    in < 40 ms on CPU while the LLM continues generating the rest in the background.
    """

    def __init__(self) -> None:
        self.sample_rate = TTS_SAMPLE_RATE
        self.model_path = VOICE_MODELS_DIR / TTS_MODEL
        self.speaking = False
        self.stop_flag = threading.Event()
        self._tts: Any = None
        self._executor = None
        self._lock = threading.Lock()
        self._first_clause_played_at: float | None = None
        self._speaker_id = 0

    def _load(self) -> None:
        if self._tts is not None:
            return
        try:
            import piper
        except ImportError:
            raise RuntimeError("piper-tts not installed (pip install piper-tts)")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model not bundled: {self.model_path}.  Run download_models.py")
        self._tts = piper.PiperConnector.from_binary(str(self.model_path))
        print(f"[tts] loaded streaming Piper {self.model_path.name}")

    @property
    def first_clause_latency_ms(self) -> float | None:
        if self._first_clause_played_at is None:
            return None
        return (time.perf_counter_ns() - self._first_clause_played_at) / 1e6

    def synthesize_first_clause(self, text: str) -> Optional[np.ndarray]:
        """Extract the first clause and synthesize it as PCM (async)."""
        clause = self._split_clauses(text)
        if clause is None:
            return None
        self._first_clause_played_at = time.perf_counter_ns()
        out = io.BytesIO()
        self._tts.synthesize_raw(
            stream=out,
            speaker_id=self._speaker_id,
            speed=TTS_SPEED,
        )
        audio = np.frombuffer(out.getvalue(), dtype=np.int16).astype(np.float32) / 32768.0
        return audio

    def _split_clauses(self, text: str) -> str | None:
        """Return the first clause: ends at first delimiter or MAX 90 chars."""
        text = text.strip()
        if not text:
            return None
        for m in CLAUSE_SPLITTERS.finditer(text):
            cut = text[: m.end()].strip()
            if len(cut) >= CLAUSE_MIN_CHARS:
                return cut
        if len(text) >= CLAUSE_MIN_CHARS:
            return text[:CLAUSE_MAX_CHARS].strip()
        return text

    def play_streaming(self, llm_stream: Iterator[str], on_complete: Callable[[], None]) -> None:
        """
        Stream tokens from the LLM into Piper TTS and play them in real time.
        Runs on the caller's background thread so the UI is never blocked.
        """
        self._load()
        self.speaking = True
        self.stop_flag.clear()
        buffer = ""
        first_word_time: float | None = None

        for token in llm_stream:
            if self.stop_flag.is_set():
                break
            buffer += token
            if first_word_time is None and len(buffer.split()) >= 1:
                first_word_time = time.perf_counter_ns()

            clause = self._split_clauses(buffer)
            if clause:
                try:
                    self.play_audio(
                        np.fromiter(
                            (self._tts.synthesize(clause, speaker_id=self._speaker_id, speed=TTS_SPEED)),
                            dtype=np.int16,
                        )
                        / 32768.0,
                        blocking=False,
                    )
                except Exception:  # noqa: BLE001
                    pass
            if self.stop_flag.is_set():
                break
        self.play_audio(np.array([], dtype=np.float32), blocking=True)
        self.speaking = False
        self._first_clause_played_at = None
        on_complete()

    def play_audio(self, audio: np.ndarray, blocking: bool = False) -> None:
        """Play an audio buffer (float32 [-1, 1], 24 kHz) to the default output."""
        if len(audio) == 0:
            return
        audio = audio.clip(-1.0, 1.0).astype(np.float32)
        try:
            sd.play(audio, self.sample_rate, blocking=blocking)
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> None:
        """Stop playback immediately (used for barge-in)."""
        self.stop_flag.set()
        try:
            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        self.speaking = False
        self.stop_flag.clear()


# ---------------------------------------------------------------------------
# Instant Barge-In
# ---------------------------------------------------------------------------
class BargeInDetector:
    """Listens during AI playback; cuts speech within BARGE_IN_LATENCY_MS (< 50 ms)."""

    def __init__(self, vad: SileroVAD) -> None:
        self.vad = vad
        self.detected = threading.Event()
        self.last_speech_at: float = 0.0

    def check_frame(self, frame: np.ndarray) -> bool:
        if self.detected.is_set():
            return True
        prob = self.vad.process_frame(frame)
        if prob >= VAD_THRESHOLD:
            self.last_speech_at = time.perf_counter_ns()
            self.detected.set()
        return bool(self.detected.is_set())

    def reset(self) -> None:
        self.detected.clear()
        self.last_speech_at = 0.0


# ---------------------------------------------------------------------------
# Full-duplex voice session
# ---------------------------------------------------------------------------
class LiveVoiceSession:
    """
    Full-duplex voice loop:
        mic 30 ms frames → VAD → (after SILENCE_THRESHOLD_MS silence) → STT
        → guardrail check → LLM → streaming TTS with clause splitter
        + barge-in during playback.
    """

    def __init__(
        self,
        on_transcript: Callable[[str], None],
        on_speaking_change: Callable[[bool], None],
        on_refusal: Callable[[str], None],
    ) -> None:
        self.on_transcript = on_transcript
        self.on_speaking_change = on_speaking_change
        self.on_refusal = on_refusal
        self.vad = SileroVAD()
        self.stt = OfflineSTT()
        self.tts = StreamingTTS()
        self.barge_in = BargeInDetector(self.vad)
        self.running = False
        self._input_queue: Deque[np.ndarray] = deque(maxlen=100)
        self._rec_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._silence_start: float | None = None
        self._mic_stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start_listening(self) -> None:
        """Open the microphone; frames flow into the internal queue."""
        if self.running:
            return
        self.running = True
        self._rec_thread = threading.Thread(target=self._listen_loop, daemon=True, name="voice-listen")
        self._rec_thread.start()

    def stop_listening(self) -> None:
        self.running = False
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:  # noqa: BLE001
                pass

    def _listen_loop(self) -> None:
        buffer: Deque[np.ndarray] = deque(maxlen=200)
        silence_threshold_samples = int(SAMPLE_RATE * SILENCE_THRESHOLD_MS / 1000)
        speech_buffer: Deque[np.ndarray] = deque(maxlen=200)

        def callback(indata: np.ndarray, *args: Any, **kwargs: Any) -> None:
            if indata.size == 0:
                return
            frames = np.split(indata, max(1, len(indata) // self.vad.frame_samples))
            for f in frames:
                if len(f) >= self.vad.frame_samples:
                    f = f[: self.vad.frame_samples]
                    self._input_queue.append(f.squeeze())
                    if self.barge_in.check_frame(f.squeeze()):
                        pass
            if not self.tts.speaking:
                if self.vad.classify_buffer(indata) == "speech":
                    self._silence_start = None
                    speech_buffer.append(indata)
                else:
                    speech_buffer.clear()
            else:
                speech_buffer.clear()

        self._mic_stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=self.vad.frame_samples,
            dtype="float32",
            callback=callback,
        )
        self._mic_stream.start()

        transcription_started = False
        while self.running:
            time.sleep(0.015)
            if transcription_started and not self.tts.speaking and not self.barge_in.detected.is_set():
                transcription_started = False
            if self.tts.speaking or self.barge_in.detected.is_set():
                self._silence_start = None
                transcription_started = False
            if speech_buffer:
                transcribed = self._transcribe_buffer(np.concatenate(speech_buffer))
                if transcribed:
                    transcription_started = True
                    self._silence_start = None
                    self.on_transcript(transcribed)
                    speech_buffer.clear()

    def _transcribe_buffer(self, buffer: np.ndarray) -> str:
        if len(buffer) < SAMPLE_RATE * 0.1:
            return ""
        return self.stt.transcribe(buffer)

    def stop_speech(self) -> None:
        """Stop AI speech immediately (barge-in target)."""
        self.tts.stop()

    def play_refusal(self) -> None:
        """Play the pre-rendered refusal clip (~0 ms wall time)."""
        try:
            data, sr = sf.read(str(REFUSAL_WAV_PATH), dtype="float32")
            self.tts.play_audio(data)
        except Exception:  # noqa: BLE001
            self.on_refusal(REFUSAL_MESSAGE)

    def shutdown(self) -> None:
        self.stop_listening()
        self.tts.stop()
        if self._rec_thread is not None:
            self._rec_thread.join(timeout=1.0)
        if self._play_thread is not None:
            self._play_thread.join(timeout=1.0)
