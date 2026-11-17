"""Stage 3 + orchestration: transcribe chunks (parallel, retried, cached),
assemble them on one timeline, post-process, and build the result.
"""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np

from .backends import AudioChunk, ChunkTranscript, RawSegment, RawWord, TranscriptionBackend, TransientBackendError
from .chunking import ChunkPlan, plan_chunks
from .config import Settings
from .ingest import AudioAsset
from .models import BackendInfo, ChunkReport, Gap, Segment, SourceInfo, TranscriptResult
from .postprocess import postprocess
from .storage import BlobStore

log = logging.getLogger("audio_decoder.pipeline")

RETRYABLE = (TransientBackendError, TimeoutError, ConnectionError)


def _to_payload(ct: ChunkTranscript) -> dict:
    return {"language": ct.language, "segments": [asdict(s) for s in ct.segments]}


def _from_payload(p: dict) -> ChunkTranscript:
    segs = [RawSegment(**{**s, "words": [RawWord(**w) for w in s["words"]]}) for s in p["segments"]]
    return ChunkTranscript(segs, p["language"])


class Pipeline:
    def __init__(self, backend: TranscriptionBackend, settings: Settings, store: BlobStore | None = None):
        self.backend, self.settings, self.store = backend, settings, store

    # -- one chunk, with cache + retry ---------------------------------
    def _run_chunk(self, asset: AudioAsset, plan: ChunkPlan, language: str | None) -> tuple[ChunkReport, ChunkTranscript | None]:
        fp = self.backend.fingerprint()
        t0 = time.perf_counter()
        report = ChunkReport(index=plan.index, start=plan.start, end=plan.end, cut=plan.cut, status="ok")

        if self.store and (cached := self.store.get_chunk(asset.sha256, fp, plan.start, plan.end)):
            report.status = "cached"
            return report, _from_payload(cached)

        sr = asset.sample_rate
        samples = asset.samples[int(plan.start * sr):int(plan.end * sr)].astype(np.float32) / 32768.0
        chunk = AudioChunk(plan.index, plan.start, plan.end, samples, sr)

        for attempt in range(1, self.settings.max_retries + 2):
            report.attempts = attempt
            try:
                ct = self.backend.transcribe(chunk, language)
                if self.store:
                    self.store.put_chunk(asset.sha256, fp, plan.start, plan.end, _to_payload(ct))
                report.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
                return report, ct
            except RETRYABLE as e:
                report.error = f"{type(e).__name__}: {e}"
                if attempt > self.settings.max_retries:
                    break
                delay = self.settings.retry_base_s * 2 ** (attempt - 1) * (0.5 + random.random())
                log.warning("chunk %d attempt %d failed (%s); retrying in %.2fs", plan.index, attempt, e, delay)
                time.sleep(delay)
            except Exception as e:  # non-retryable: a bug or bad input won't fix itself
                report.error = f"{type(e).__name__}: {e}"
                break
        report.status = "failed"
        report.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.error("chunk %d failed permanently: %s", plan.index, report.error)
        return report, None

    # -- assembly ------------------------------------------------------
    @staticmethod
    def _assemble(plan: ChunkPlan, ct: ChunkTranscript, is_last: bool) -> list[Segment]:
        """Shift to absolute time and keep only words whose midpoint lies in the
        chunk's owned range. Owned ranges partition the timeline, so every word
        from an overlap is kept exactly once."""
        def owned(a: float, b: float) -> bool:
            mid = (a + b) / 2
            return plan.own_start <= mid < plan.own_end or (is_last and mid >= plan.own_end)

        out = []
        for s in ct.segments:
            if s.words:
                words = [w for w in s.words if owned(plan.start + w.start, plan.start + w.end)]
                if not words:
                    continue
                text = "".join(w.text for w in words)
                start, end = plan.start + words[0].start, plan.start + words[-1].end
            else:  # backend without word timestamps: fall back to segment midpoint
                if not owned(plan.start + s.start, plan.start + s.end):
                    continue
                text, start, end = s.text, plan.start + s.start, plan.start + s.end
            out.append(Segment(id=0, start=round(start, 3), end=round(end, 3), text=text,
                               chunk_index=plan.index, avg_logprob=s.avg_logprob,
                               no_speech_prob=s.no_speech_prob))
        return out

    @staticmethod
    def _merge_hard_cuts(segments: list[Segment], hard_cut_chunks: set[int]) -> list[Segment]:
        """A hard cut can split one sentence across two chunks; join it back."""
        out: list[Segment] = []
        for s in segments:
            prev = out[-1] if out else None
            if (prev and prev.chunk_index in hard_cut_chunks and s.chunk_index == prev.chunk_index + 1
                    and s.start - prev.end < 0.3 and not prev.text.rstrip().endswith((".", "?", "!"))):
                prev.text = prev.text.rstrip() + " " + s.text.lstrip()
                prev.end = s.end
                prev.chunk_index = s.chunk_index  # join only across the boundary, once
                if s.avg_logprob is not None and prev.avg_logprob is not None:
                    prev.avg_logprob = min(prev.avg_logprob, s.avg_logprob)
                continue
            out.append(s)
        return out

    # -- full run ------------------------------------------------------
    def run(self, asset: AudioAsset, language: str | None = None, filename: str | None = None) -> TranscriptResult:
        t0 = time.perf_counter()
        plans = plan_chunks(asset.samples, asset.sample_rate, self.settings.chunking)
        reports: dict[int, ChunkReport] = {}
        results: dict[int, ChunkTranscript] = {}

        for p in plans:
            if p.is_silent:
                reports[p.index] = ChunkReport(index=p.index, start=p.start, end=p.end, cut=p.cut,
                                               status="skipped_silence")
        speech = [p for p in plans if not p.is_silent]

        # Pin the language from the first speech chunk so all chunks agree.
        if speech and language is None:
            rep, ct = self._run_chunk(asset, speech[0], None)
            reports[speech[0].index] = rep
            if ct:
                results[speech[0].index] = ct
                language = ct.language
            speech = speech[1:]

        with ThreadPoolExecutor(max_workers=max(1, self.settings.chunk_workers)) as pool:
            for p, (rep, ct) in zip(speech, pool.map(lambda p: self._run_chunk(asset, p, language), speech)):
                reports[p.index] = rep
                if ct:
                    results[p.index] = ct

        segments: list[Segment] = []
        for p in plans:
            if p.index in results:
                segments += self._assemble(p, results[p.index], p.index == plans[-1].index)
        segments = self._merge_hard_cuts(segments, {p.index for p in plans if p.cut == "hard"})
        segments = postprocess(segments, self.settings.redact_pii)

        gaps = [Gap(start=p.own_start, end=p.own_end, reason=reports[p.index].error or "failed")
                for p in plans if reports[p.index].status == "failed"]
        speech_total = sum(1 for p in plans if not p.is_silent)
        failed = len(gaps)
        status = "completed" if failed == 0 else ("failed" if failed == speech_total else "partial")

        elapsed = time.perf_counter() - t0
        words = sum(len(s.text.split()) for s in segments)
        duration = asset.duration_s
        return TranscriptResult(
            status=status,
            source=SourceInfo(filename=filename or asset.path.name, sha256=asset.sha256,
                              duration_s=round(duration, 3), original_format=asset.original_format,
                              original_sample_rate=asset.original_sample_rate,
                              original_channels=asset.original_channels),
            backend=BackendInfo(name=self.backend.name, fingerprint=self.backend.fingerprint()),
            language=language,
            redacted=self.settings.redact_pii,
            text=" ".join(s.text for s in segments),
            segments=segments,
            gaps=gaps,
            chunks=[reports[p.index] for p in plans],
            stats={
                "chunks": len(plans),
                "chunks_skipped_silence": sum(p.is_silent for p in plans),
                "chunks_failed": failed,
                "chunks_cached": sum(r.status == "cached" for r in reports.values()),
                "segments": len(segments),
                "words": words,
                "words_per_minute": round(words / (duration / 60), 1) if duration else 0,
                "low_confidence_segments": sum("low_confidence" in s.flags for s in segments),
                "processing_s": round(elapsed, 3),
                "real_time_factor": round(elapsed / duration, 4) if duration else None,
            },
        )
