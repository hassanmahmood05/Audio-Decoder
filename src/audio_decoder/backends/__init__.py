from __future__ import annotations

import os

from ..config import Settings
from .base import AudioChunk, ChunkTranscript, RawSegment, RawWord, TranscriptionBackend, TransientBackendError


def create_backend(settings: Settings) -> TranscriptionBackend:
    if settings.backend == "mock":
        from .mock import MockBackend
        return MockBackend(os.getenv("AUDIO_DECODER_MOCK_SCRIPT") or None)
    if settings.backend == "whisper":
        from .whisper import WhisperBackend
        return WhisperBackend(settings.whisper_model, settings.whisper_device,
                              settings.whisper_compute, num_workers=settings.chunk_workers)
    raise ValueError(f"Unknown backend {settings.backend!r} (expected 'whisper' or 'mock')")


__all__ = ["AudioChunk", "ChunkTranscript", "RawSegment", "RawWord",
           "TranscriptionBackend", "TransientBackendError", "create_backend"]
