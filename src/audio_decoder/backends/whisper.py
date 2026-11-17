"""faster-whisper backend (CTranslate2 re-implementation of OpenAI Whisper).

Chosen over openai-whisper because it is ~4x faster on CPU with int8
quantisation, uses less memory, and supports word timestamps, which the
pipeline uses to de-duplicate chunk overlaps precisely.
"""
from __future__ import annotations

import threading

from .base import AudioChunk, ChunkTranscript, RawSegment, RawWord, TransientBackendError


class WhisperBackend:
    name = "faster-whisper"

    def __init__(self, model_size: str = "small", device: str = "cpu",
                 compute_type: str = "int8", num_workers: int = 2, beam_size: int = 5):
        self.model_size, self.device, self.compute_type = model_size, device, compute_type
        self.num_workers, self.beam_size = num_workers, beam_size
        self._model = None
        self._lock = threading.Lock()

    def fingerprint(self) -> str:
        return f"{self.name}:{self.model_size}:{self.compute_type}:beam{self.beam_size}"

    def _get_model(self):
        # Lazy, thread-safe, loaded once per process and shared by all workers.
        # num_workers lets CTranslate2 serve that many transcriptions in parallel.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from faster_whisper import WhisperModel
                    self._model = WhisperModel(self.model_size, device=self.device,
                                               compute_type=self.compute_type,
                                               num_workers=self.num_workers)
        return self._model

    def transcribe(self, chunk: AudioChunk, language: str | None) -> ChunkTranscript:
        try:
            segments, info = self._get_model().transcribe(
                chunk.samples,
                language=language,
                beam_size=self.beam_size,
                word_timestamps=True,
                vad_filter=False,                   # silence handled by our chunker
                condition_on_previous_text=False,   # avoids repetition loops
            )
            out = [
                RawSegment(
                    start=s.start, end=s.end, text=s.text,
                    avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
                    words=[RawWord(w.start, w.end, w.word, w.probability) for w in (s.words or [])],
                )
                for s in segments  # generator: decoding happens here
            ]
            return ChunkTranscript(out, info.language)
        except (MemoryError, RuntimeError) as e:  # CUDA OOM / CTranslate2 runtime errors
            raise TransientBackendError(str(e)) from e
