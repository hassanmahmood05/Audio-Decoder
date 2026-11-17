import dataclasses
import time

import pytest
from fastapi.testclient import TestClient

from audio_decoder.api import create_app
from audio_decoder.backends.mock import MockBackend
from audio_decoder.jobs import JobManager


@pytest.fixture
def client(settings, demo):
    _, script = demo
    manager = JobManager(settings, MockBackend(script))
    with TestClient(create_app(settings, manager)) as c:
        c.manager = manager
        yield c


def _wait(client, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/v1/transcriptions/{job_id}").json()
        if body["job"]["status"] not in ("queued", "processing"):
            return body
        time.sleep(0.05)
    raise TimeoutError


def test_upload_returns_202_then_timestamped_result(client, demo):
    wav, _ = demo
    r = client.post("/v1/transcriptions", files={"file": ("call.wav", wav.read_bytes(), "audio/wav")})
    assert r.status_code == 202
    job_id = r.json()["job"]["id"]

    body = _wait(client, job_id)
    assert body["job"]["status"] == "completed"
    segs = body["result"]["segments"]
    assert segs and {"id", "start", "end", "text"} <= set(segs[0])

    srt = client.get(f"/v1/transcriptions/{job_id}/result", params={"format": "srt"})
    assert srt.status_code == 200 and "-->" in srt.text


def test_duplicate_upload_is_idempotent(client, demo):
    wav, _ = demo
    files = {"file": ("call.wav", wav.read_bytes(), "audio/wav")}
    first = client.post("/v1/transcriptions", files=files).json()["job"]["id"]
    second = client.post("/v1/transcriptions", files=files)
    assert second.status_code == 200 and second.json()["job"]["id"] == first


@pytest.mark.parametrize("name,data,code", [
    ("notes.txt", b"hello", 415),
    ("empty.wav", b"", 400),
    ("fake.mp3", b"definitely not audio", 422),
])
def test_validation_errors(client, name, data, code):
    assert client.post("/v1/transcriptions", files={"file": (name, data)}).status_code == code


def test_unknown_job_404_and_result_not_ready_409(client, settings):
    assert client.get("/v1/transcriptions/nope").status_code == 404


def test_api_key_enforced(settings, demo):
    s = dataclasses.replace(settings, api_key="secret")
    with TestClient(create_app(s, JobManager(s, MockBackend(demo[1])))) as c:
        assert c.get("/v1/transcriptions/x").status_code == 401
        assert c.get("/v1/transcriptions/x", headers={"X-API-Key": "secret"}).status_code == 404


def test_concurrent_uploads_all_complete(client, tmp_path):
    from conftest import silence, speech_like, write_wav
    import numpy as np
    ids = []
    for i in range(5):
        a = np.concatenate([speech_like(3, f0=150 + i * 20), silence(1)])
        p = write_wav(tmp_path / f"u{i}.wav", a)
        ids.append(client.post("/v1/transcriptions", files={"file": (p.name, p.read_bytes())}).json()["job"]["id"])
    assert len(set(ids)) == 5
    assert all(_wait(client, j)["job"]["status"] == "completed" for j in ids)
