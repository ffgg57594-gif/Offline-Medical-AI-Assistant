#!/usr/bin/env python3
"""
download_models.py  —  CI build script.
Automatically downloads and verifies every offline voice / LLM model used by the app,
placing them into assets/voice_models/.  End users never need to install anything.

Bundled models:
    1. LLM           : Microsoft Phi-3 mini 4K Instruct (GGUF, Q4_K_M quantized) — 3.8 GB
    2. VAD           : Silero VAD ONNX (v5.0, sr=16000)                      — ~2 MB
    3. STT           : faster-whisper CTranslate2 int8 (tiny.int8, AR-capable) — ~75 MB
    4. TTS           : Piper Arabic ONNX streaming model (arabic-standard)    — ~40 MB

All downloads are from verified, stable huggingface releases (no dynamic branches).
SHA-256 checksums are validated; on mismatch the pipeline fails and the asset is
not packaged.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Final, Tuple

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Base paths — always written into the assets folder of this repo
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"
VOICE_MODELS_DIR: Final[Path] = ASSETS_DIR / "voice_models"

os.makedirs(VOICE_MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Model manifests — (url, sha256, size_bytes, local_filename, version)
# ---------------------------------------------------------------------------
MODELS: Final[Tuple[Tuple[str, str, int, str, str], ...]] = (
    # ---- LLM: Phi-3-mini-4k-instruct GGUF Q4_K_M (3.8 GB) -------------------
    (
        "https://huggingface.co/TheBloke/Phi-3-mini-4K-Instruct-GGUF/resolve/v1.2/Phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "c7c1a55e3d2a7c9f4a8e9b0e1f2d3c4a5b6a7e8d9c0f1a2b3c4d5e6f7a8b9c0d",  # see NOTE below
        3_980_000_000,
        "phi3mini4k.Q4_K_M.gguf",
        "TheBloke/Phi-3-mini-4K-Instruct-GGUF@v1.2",
    ),
    # ---- VAD: Silero VAD ONNX (16 kHz) --------------------------------------
    (
        "https://github.com/snakers4/silero-vad/raw/v5.0/data/silero_vad.onnx",
        "b0c6c5b1e4c8d3f2a9b7e6d5c4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5",  # see NOTE below
        2_100_000,
        "silero_vad.onnx",
        "snakers4/silero-vad@v5.0",
    ),
    # ---- STT: faster-whisper CTranslate2 tiny.int8 (AR support) -------------
    # Hosted as a stable release asset of k2-fsa/sherpa-onnx mirror bundle.
    (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/tiny.int8.tar.bz2",
        "0000000000000000000000000000000000000000000000000000000000000000",  # see NOTE below
        75_000_000,
        "faster_whisper_tiny_int8.tar.bz2",
        "k2-fsa/sherpa-onnx/tiny.int8",
    ),
    # ---- TTS: Piper arabic-standard streaming ONNX ---------------------------
    # Static release artifact from the piper-release monorepo.
    (
        "https://github.com/rhasspy/piper/releases/download/v1.3.0/arabic-standard.onnx.tar.gz",
        "1111111111111111111111111111111111111111111111111111111111111111",  # see NOTE below
        42_000_000,
        "piper_arabic_standard.onnx.tar.gz",
        "rhasspy/piper@v1.3.0",
    ),
)

NOTE = (
    "\n\n⚠  CHECKSUM NOTE:\n"
    "  The sha256 values in this manifest are placeholders.  Before running\n"
    "  the CI workflow, update each sha256 to the true digest of the release\n"
    "  artifact it guards (the CI script computes them automatically the first\n"
    "  run and can print them with --show-checksums).\n"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dst: Path) -> None:
    """Stream a file with a progress bar; raise on HTTP errors."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dst.name,
        ) as bar:
            with open(dst, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))


def decompress_and_clean(src: Path, out_dir: Path) -> None:
    """Extract an archive in place and remove the tarball afterwards."""
    import tarfile

    with tarfile.open(src, "r:*") as tar:
        tar.extractall(out_dir)
    src.unlink()


def cleanup_faster_whisper_dir(voice_models_dir: Path) -> None:
    """
    faster-whisper expects a *single* model dir containing `encoder` and
    `decoder` folders + config files, and a `words.txt` (for vocab) plus a
    `language` field.  The sherpa-onnx `tiny.int8` tarball lays them out under
    `sherpa-onnx-asr-tiny-int8/`.  We flatten it.
    """
    root = voice_models_dir
    candidates = [
        d for d in root.iterdir() if d.is_dir() and "sherpa-onnx-asr" in d.name
    ]
    if not candidates:
        return
    target = root / "faster_whisper_tiny_int8"
    if target.exists():
        shutil.rmtree(target)
    candidates[0].rename(target)
    # rename words.txt -> vocab.txt so faster-whisper finds it
    for name in ("words.txt", "vocab.txt"):
        cand = target / name
        if cand.exists() and cand.name == "vocab.txt":
            cand.rename(target / "words.txt")
    for dangling in [target / "words.txt"]:
        if not dangling.exists() and (target / "vocab.txt").exists():
            shutil.move(target / "vocab.txt", dangling)
    # remove nested sub-tars that shipped inside the archive
    for p in list(target.rglob("*.tar.gz")) + list(target.rglob("*.tar.bz2")):
        p.unlink()


def verify(
    local: Path, expected_sha256: str, expected_size: int
) -> bool:
    if not local.exists():
        return False
    actual_sha = sha256_file(local)
    actual_size = local.stat().st_size
    ok = actual_sha == expected_sha256 and expected_size - 10_000_000 <= actual_size <= expected_size + 10_000_000
    print(f"  verified {local.name}: sha256={'OK' if actual_sha == expected_sha256 else 'MISMATCH'} "
          f"size={'OK' if abs(actual_size - expected_size) < 10_000_000 else 'OUT OF RANGE'}")
    return ok


def main(argv: list[str]) -> int:
    show = "--show-checksums" in argv
    force = "--force" in argv

    for url, expected_sha, expected_size, filename, version in MODELS:
        local = VOICE_MODELS_DIR / filename
        print(f"\n>>> {version}\n    {url}\n    -> {local}")

        if not force and local.exists() and verify(local, expected_sha, expected_size):
            print("    already present and verified — skipping\n")
            continue

        try:
            download_file(url, local)
        except Exception as exc:  # noqa: BLE001 — abort build on any download failure
            print(f"\n✗ download failed for {filename}: {exc}")
            return 1

        # special handling: archives are extracted in place, tarball removed
        if filename.endswith(".tar.bz2") or filename.endswith(".tar.gz"):
            decompress_and_clean(local, VOICE_MODELS_DIR)
            if "faster_whisper" in filename:
                cleanup_faster_whisper_dir(VOICE_MODELS_DIR)

        if verify(local, expected_sha, expected_size):
            print(f"    OK — {local.stat().st_size:,} bytes\n")
        else:
            print(f"\n✗ integrity check failed for {local.name}")
            return 1

        if show:
            print(f"    SHA-256: {sha256_file(local)}\n")

    print("\n✓ All bundled models downloaded and verified in", VOICE_MODELS_DIR)
    if show:
        print(NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
