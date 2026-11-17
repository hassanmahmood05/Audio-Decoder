"""Data contracts.

`TranscriptResult` is the contract with downstream consumers (search, warehouse
loaders, subtitle tooling), so it is versioned and everything a consumer may
rely on is explicit: coverage gaps, confidence flags, model fingerprint.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float | None = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    chunk_index: int
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    flags: list[str] = Field(default_factory=list)


class ChunkReport(BaseModel):
    index: int
    start: float
    end: float
    cut: Literal["silence", "hard", "end"]
    status: Literal["ok", "cached", "skipped_silence", "failed"]
    attempts: int = 0
    error: str | None = None
    duration_ms: float = 0.0


class Gap(BaseModel):
    """Time range with no transcript coverage; downstream must not assume silence."""
    start: float
    end: float
    reason: str


class SourceInfo(BaseModel):
    filename: str
    sha256: str
    duration_s: float
    original_format: str
    original_sample_rate: int | None = None
    original_channels: int | None = None


class BackendInfo(BaseModel):
    name: str
    fingerprint: str


class TranscriptResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    status: Literal["completed", "partial", "failed"]
    source: SourceInfo
    backend: BackendInfo
    language: str | None
    redacted: bool
    text: str
    segments: list[Segment]
    gaps: list[Gap]
    chunks: list[ChunkReport]
    stats: dict
    created_at: datetime = Field(default_factory=utcnow)


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    partial = "partial"
    failed = "failed"


class Job(BaseModel):
    id: str
    status: JobStatus
    filename: str
    audio_sha256: str
    language: str | None = None
    error: str | None = None
    attempts: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
