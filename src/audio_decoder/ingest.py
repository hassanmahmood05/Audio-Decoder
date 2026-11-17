"""Stage 1: ingest.

Accept whatever container/codec the caller has, validate it, fingerprint it,
and normalise to the one format every backend wants: 16 kHz mono PCM.
ffmpeg does the decoding so we don't depend on per-format Python libraries.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SR = 16_000
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".aac"}


class IngestError(Exception):
    """Input is unusable. Not retryable: the same file will fail the same way."""


@dataclass
class AudioAsset:
    path: Path
    sha256: str
    original_format: str
    original_sample_rate: int | None
    original_channels: int | None
    samples: np.ndarray  # int16, mono, TARGET_SR (int16 halves memory vs float32)
    sample_rate: int = TARGET_SR

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise IngestError(f"ffprobe could not read {path.name}: {proc.stderr.strip()[:200]}")
    info = json.loads(proc.stdout or "{}")
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        raise IngestError(f"{path.name} has no audio stream")
    return {"format": info.get("format", {}), "audio": audio[0]}


def load_audio(path: str | Path, max_duration_s: float = 3 * 3600) -> AudioAsset:
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"Unsupported extension {path.suffix!r}; expected one of {sorted(SUPPORTED_EXTENSIONS)}")
    if path.stat().st_size == 0:
        raise IngestError(f"{path.name} is empty")

    meta = probe(path)
    declared = float(meta["format"].get("duration") or 0)
    if declared > max_duration_s:
        raise IngestError(f"{path.name} is {declared:.0f}s, over the {max_duration_s:.0f}s limit")

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-ac", "1", "-ar", str(TARGET_SR), "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise IngestError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode()[:200]}")
    samples = np.frombuffer(proc.stdout, dtype=np.int16)
    if samples.size == 0:
        raise IngestError(f"{path.name} decoded to zero samples")

    a = meta["audio"]
    return AudioAsset(
        path=path,
        sha256=sha256_file(path),
        original_format=meta["format"].get("format_name", "unknown"),
        original_sample_rate=int(a["sample_rate"]) if a.get("sample_rate") else None,
        original_channels=a.get("channels"),
        samples=samples,
    )
