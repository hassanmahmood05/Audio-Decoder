"""Configuration from environment variables (12-factor), with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(f"AUDIO_DECODER_{name}", default)


@dataclass(frozen=True)
class ChunkingConfig:
    target_s: float = 30.0       # preferred chunk length
    min_s: float = 10.0          # never cut earlier than this
    max_s: float = 45.0          # always cut by this point
    min_silence_s: float = 0.3   # a pause must be this long to be a cut point
    overlap_s: float = 2.0       # overlap used only for forced (hard) cuts
    silence_db: float = -40.0    # dBFS floor for "silence"


@dataclass(frozen=True)
class Settings:
    backend: str = field(default_factory=lambda: _env("BACKEND", "whisper"))       # whisper | mock
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "cpu"))
    whisper_compute: str = field(default_factory=lambda: _env("WHISPER_COMPUTE", "int8"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")))
    workers: int = field(default_factory=lambda: int(_env("WORKERS", "2")))          # concurrent jobs
    chunk_workers: int = field(default_factory=lambda: int(_env("CHUNK_WORKERS", "2")))  # chunks per job
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "200")))
    max_duration_s: float = field(default_factory=lambda: float(_env("MAX_DURATION_S", str(3 * 3600))))
    # Back-pressure: queued + processing jobs allowed before new uploads get 429.
    # Also bounds disk use for pending uploads to roughly max_pending_jobs * max_upload_mb.
    max_pending_jobs: int = field(default_factory=lambda: int(_env("MAX_PENDING_JOBS", "100")))
    max_retries: int = field(default_factory=lambda: int(_env("MAX_RETRIES", "3")))
    retry_base_s: float = field(default_factory=lambda: float(_env("RETRY_BASE_S", "0.5")))
    redact_pii: bool = field(default_factory=lambda: _env("REDACT_PII", "true").lower() == "true")
    api_key: str | None = field(default_factory=lambda: os.getenv("AUDIO_DECODER_API_KEY") or None)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
