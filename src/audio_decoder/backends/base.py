"""Backend interface.

The pipeline depends only on this protocol, so the speech-to-text engine
(local faster-whisper, a hosted API, a mock) is swappable via configuration.
Backends return timestamps *relative to the chunk*; the pipeline offsets them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class TransientBackendError(Exception):
    """Retryable failure (timeout, OOM, rate limit, worker hiccup)."""


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float              # absolute start of the window in the source file
    end: float
    samples: np.ndarray       # float32 mono, -1..1
    sample_rate: int


@dataclass
class RawWord:
    start: float
    end: float
    text: str                 # Whisper convention: includes the leading space
    probability: float | None = None


@dataclass
class RawSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[RawWord] = field(default_factory=list)


@dataclass
class ChunkTranscript:
    segments: list[RawSegment]
    language: str | None


class TranscriptionBackend(Protocol):
    name: str

    def fingerprint(self) -> str:
        """Identifies model + settings; part of the cache key."""
        ...

    def transcribe(self, chunk: AudioChunk, language: str | None) -> ChunkTranscript:
        ...
