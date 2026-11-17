import dataclasses
import re

from audio_decoder.backends.mock import MockBackend
from audio_decoder.ingest import load_audio
from audio_decoder.pipeline import Pipeline
from audio_decoder.storage import BlobStore


def _words(text):
    return re.findall(r"[\w'@.+-]+", text.lower())


def test_every_spoken_word_appears_exactly_once(demo, settings):
    """Core invariant of chunking + overlap de-duplication."""
    wav, script = demo
    clean = [l for l in script if "avg_logprob" not in l and "no_speech_prob" not in l]
    settings = dataclasses.replace(settings, redact_pii=False)
    result = Pipeline(MockBackend(clean), settings).run(load_audio(wav))

    assert result.status == "completed"
    assert any(c.cut == "hard" for c in result.chunks), "demo should exercise a hard cut"
    assert any(c.status == "skipped_silence" for c in result.chunks)
    expected = [w for line in clean for w in _words(line["text"])]
    assert _words(result.text) == expected


def test_timestamps_are_absolute_and_ordered(demo, settings):
    wav, script = demo
    result = Pipeline(MockBackend(script), settings).run(load_audio(wav))
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    assert all(s.start < s.end for s in result.segments)
    assert result.segments[-1].end <= result.source.duration_s
    first = script[0]
    assert abs(result.segments[0].start - first["start"]) < 0.01


def test_language_detected_once_and_pinned(demo, settings):
    wav, script = demo

    class Recording(MockBackend):
        seen = []

        def transcribe(self, chunk, language):
            self.seen.append(language)
            return super().transcribe(chunk, language)

    b = Recording(script)
    result = Pipeline(b, settings).run(load_audio(wav))
    assert result.language == "en"
    assert b.seen[0] is None and all(l == "en" for l in b.seen[1:])


def test_transient_failure_is_retried(demo, settings):
    wav, script = demo
    result = Pipeline(MockBackend(script, fail_attempts={1: 2}), settings).run(load_audio(wav))
    assert result.status == "completed"
    assert result.chunks[1].attempts == 3


def test_permanent_failure_gives_partial_with_gap_then_resumes_from_cache(demo, settings):
    wav, script = demo
    asset = load_audio(wav)
    store = BlobStore(settings.data_dir)

    failing = MockBackend(script, fail_attempts={1: 99})
    r1 = Pipeline(failing, settings, store).run(asset)
    assert r1.status == "partial"
    assert len(r1.gaps) == 1 and r1.gaps[0].start == r1.chunks[1].start
    assert r1.chunks[1].attempts == settings.max_retries + 1

    healthy = MockBackend(script)
    r2 = Pipeline(healthy, settings, store).run(asset)
    assert r2.status == "completed"
    assert list(healthy.calls) == [1], "only the failed chunk should be re-transcribed"
    assert r2.stats["chunks_cached"] >= 3


def test_non_retryable_error_fails_fast(demo, settings):
    wav, script = demo

    class Broken(MockBackend):
        def transcribe(self, chunk, language):
            raise ValueError("bug")

    r = Pipeline(Broken(script), settings).run(load_audio(wav))
    assert r.status == "failed"
    assert all(c.attempts == 1 for c in r.chunks if c.status == "failed")


def test_redaction_applied_end_to_end(demo, settings):
    wav, script = demo
    r = Pipeline(MockBackend(script), settings).run(load_audio(wav))
    assert "sarah.khan@example.com" not in r.text and "[EMAIL]" in r.text
    assert "4111" not in r.text and "[CARD]" in r.text
    assert "CLM-2026-004417" in r.text
