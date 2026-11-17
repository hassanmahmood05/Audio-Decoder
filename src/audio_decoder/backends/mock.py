"""Deterministic mock backend for tests, demos and offline development.

Given a "script" (list of {start, end, text, ...} in absolute seconds), it
returns the words whose midpoint falls inside the chunk window, with word
timings spread evenly across each line. This behaves like a real model at
chunk boundaries (edge words appear in both overlapping chunks), so the
assembly/de-duplication logic is exercised for real.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .base import AudioChunk, ChunkTranscript, RawSegment, RawWord, TransientBackendError


class MockBackend:
    name = "mock"

    def __init__(self, script: list[dict] | str | Path | None = None,
                 fail_attempts: dict[int, int] | None = None):
        if isinstance(script, (str, Path)):
            script = json.loads(Path(script).read_text(encoding="utf-8"))
        self.script = script or []
        # chunk index -> number of attempts that should fail (simulates transient errors)
        self.fail_attempts = dict(fail_attempts or {})
        self.calls: dict[int, int] = {}
        self._lock = threading.Lock()

    def fingerprint(self) -> str:
        return f"mock:{len(self.script)}"

    def transcribe(self, chunk: AudioChunk, language: str | None) -> ChunkTranscript:
        with self._lock:
            n = self.calls[chunk.index] = self.calls.get(chunk.index, 0) + 1
        if n <= self.fail_attempts.get(chunk.index, 0):
            raise TransientBackendError(f"simulated failure on chunk {chunk.index}, attempt {n}")

        if not self.script:  # no script: one placeholder segment per chunk
            dur = chunk.end - chunk.start
            return ChunkTranscript([RawSegment(0.0, dur, " [mock speech]", -0.2, 0.01,
                                               [RawWord(0.0, dur, " [mock speech]", 0.9)])], "en")

        segments = []
        for line in self.script:
            tokens = line["text"].split()
            step = (line["end"] - line["start"]) / max(1, len(tokens))
            words = []
            for i, tok in enumerate(tokens):
                ws, we = line["start"] + i * step, line["start"] + (i + 1) * step
                if chunk.start <= (ws + we) / 2 < chunk.end:
                    words.append(RawWord(round(ws - chunk.start, 3), round(we - chunk.start, 3), " " + tok, 0.9))
            if words:
                segments.append(RawSegment(
                    start=words[0].start, end=words[-1].end,
                    text="".join(w.text for w in words),
                    avg_logprob=line.get("avg_logprob", -0.25),
                    no_speech_prob=line.get("no_speech_prob", 0.02),
                    words=words,
                ))
        return ChunkTranscript(segments, language or "en")
