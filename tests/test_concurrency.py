"""Races and back-pressure under concurrent uploads."""
import dataclasses
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from audio_decoder.api import create_app
from audio_decoder.backends.mock import MockBackend
from audio_decoder.jobs import JobManager
from audio_decoder.postprocess import redact
from audio_decoder.storage import BlobStore


def _run_together(n, fn):
    """Start n threads behind a barrier so they hit the critical section at once."""
    barrier, errors, results = threading.Barrier(n), [], [None] * n

    def body(i):
        barrier.wait()
        try:
            results[i] = fn(i)
        except Exception as e:  # collected so the test reports them all
            errors.append(e)

    threads = [threading.Thread(target=body, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def _upload(tmp_path, i, data=b"same audio bytes"):
    p = tmp_path / f"upload-{i}.tmp"
    p.write_bytes(data)
    return p


def test_identical_concurrent_submits_create_one_job(settings, tmp_path, monkeypatch):
    manager = JobManager(settings, MockBackend([]))   # workers not started: jobs stay queued
    original = manager.repo.find_reusable

    def slow_find(*a, **kw):                          # widen the check-then-create window
        r = original(*a, **kw)
        time.sleep(0.05)
        return r

    monkeypatch.setattr(manager.repo, "find_reusable", slow_find)
    results, errors = _run_together(
        8, lambda i: manager.submit(_upload(tmp_path, i), "abc123", ".wav", "call.wav", "en"))
    assert not errors
    assert len({job.id for job, _ in results}) == 1
    assert sum(created for _, created in results) == 1
    assert manager.store.audio_path("abc123", ".wav").read_bytes() == b"same audio bytes"
    assert not list(tmp_path.glob("upload-*.tmp"))    # every temp upload consumed or removed


def test_adopting_audio_while_a_reader_holds_it_open(settings, tmp_path):
    store = BlobStore(settings.data_dir)
    dest = store.adopt_audio(_upload(tmp_path, 0), "abc123", ".wav")
    with dest.open("rb"):                             # a worker's ffmpeg is reading it
        results, errors = _run_together(6, lambda i: store.adopt_audio(_upload(tmp_path, i + 1), "abc123", ".wav"))
    assert not errors
    assert all(r == dest for r in results)
    assert dest.read_bytes() == b"same audio bytes"
    assert not list(tmp_path.glob("upload-*.tmp"))


def test_concurrent_writers_of_the_same_chunk(settings):
    store = BlobStore(settings.data_dir)
    payload = {"language": "en", "segments": [{"text": "x" * 5000}]}
    _, errors = _run_together(16, lambda i: store.put_chunk("abc", "fp", 0.0, 30.0, payload))
    assert not errors
    assert store.get_chunk("abc", "fp", 0.0, 30.0) == payload
    assert not list(settings.data_dir.rglob("*.tmp"))


@pytest.fixture
def idle_client(settings, demo):
    """API whose workers never start, so submitted jobs stay queued."""
    s = dataclasses.replace(settings, max_pending_jobs=2)
    manager = JobManager(s, MockBackend(demo[1]))
    manager.start = lambda: None
    with TestClient(create_app(s, manager)) as c:
        yield c


def _post(client, data, name="a.wav"):
    return client.post("/v1/transcriptions", files={"file": (name, data)})


def test_queue_full_returns_429_but_duplicates_still_resolve(idle_client, demo, tmp_path):
    from conftest import speech_like, write_wav
    wavs = [write_wav(tmp_path / f"w{i}.wav", speech_like(1, f0=150 + 30 * i)).read_bytes() for i in range(3)]

    first, second = _post(idle_client, wavs[0]), _post(idle_client, wavs[1])
    assert first.status_code == second.status_code == 202
    assert idle_client.get(first.json()["status_url"]).json()["job"]["queue_position"] == 1
    assert idle_client.get(second.json()["status_url"]).json()["job"]["queue_position"] == 2

    full = _post(idle_client, wavs[2])
    assert full.status_code == 429 and int(full.headers["Retry-After"]) > 0

    dup = _post(idle_client, wavs[0])                 # already queued: no new work, so not rejected
    assert dup.status_code == 200 and dup.json()["job"]["id"] == first.json()["job"]["id"]

    data_dir = idle_client.app.state.jobs.settings.data_dir
    assert len(list((data_dir / "audio").iterdir())) == 2   # rejected upload not kept
    assert not list((data_dir / "tmp").iterdir())


def test_spoken_email_regex_is_linear_on_adversarial_text():
    text = "a dot " * 20_000 + "end"                  # long run that never reaches " at "
    t0 = time.perf_counter()
    out, found = redact(text)
    assert time.perf_counter() - t0 < 0.5 and found == [] and out == text
