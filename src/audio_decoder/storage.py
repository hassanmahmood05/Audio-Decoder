"""Persistence.

Locally: a content-addressed file store (stands in for Azure Blob / S3) and
SQLite (stands in for SQL Server / Postgres). Both sit behind small classes so
swapping to cloud services changes only this module.

Layout under DATA_DIR:
  audio/<sha256><ext>                     original upload, immutable
  transcripts/<job_id>.json               full result (schema + model version inside)
  cache/<sha256>/<model>/<start>-<end>.json  per-chunk results (resumability)
  jobs.db                                 job table + segment table
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from .models import Job, JobStatus, TranscriptResult, utcnow


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _atomic_write(path: Path, text: str, identical_ok: bool = False) -> None:
    """Write via a temp file unique to this writer, then rename into place, so
    readers never see a half-written file and concurrent writers never share a
    temp file. On Windows the rename fails while another thread has the target
    open (a reader, or a concurrent writer's rename), so retry briefly. When the
    content is deterministic (`identical_ok`), a target that already exists means
    another writer won with the same bytes, and this copy can be dropped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if identical_ok and path.exists():
                    return
                if attempt == 9:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class BlobStore:
    def __init__(self, root: Path):
        self.root = root
        for d in ("audio", "transcripts", "cache"):
            (root / d).mkdir(parents=True, exist_ok=True)

    # audio ------------------------------------------------------------
    def audio_path(self, sha: str, ext: str) -> Path:
        return self.root / "audio" / f"{sha}{ext.lower()}"

    def adopt_audio(self, upload: Path, sha: str, ext: str) -> Path:
        """Move an upload into the content-addressed store. Never overwrites: if the
        same content is already stored (possibly open by a worker), the upload is
        discarded. rename() fails on Windows when the target exists, and on POSIX
        it atomically swaps in identical bytes, so both are safe under races."""
        dest = self.audio_path(sha, ext)
        if dest.exists():
            upload.unlink(missing_ok=True)
            return dest
        try:
            upload.rename(dest)
        except (FileExistsError, PermissionError):
            if not dest.exists():
                raise
            upload.unlink(missing_ok=True)
        return dest

    # transcripts ------------------------------------------------------
    def transcript_path(self, job_id: str) -> Path:
        return self.root / "transcripts" / f"{_safe(job_id)}.json"

    def save_transcript(self, job_id: str, result: TranscriptResult) -> Path:
        p = self.transcript_path(job_id)
        _atomic_write(p, result.model_dump_json(indent=2))
        return p

    def load_transcript(self, job_id: str) -> TranscriptResult | None:
        p = self.transcript_path(job_id)
        return TranscriptResult.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else None

    # chunk cache ------------------------------------------------------
    def _chunk_path(self, sha: str, fingerprint: str, start: float, end: float) -> Path:
        return self.root / "cache" / sha / _safe(fingerprint) / f"{start:.3f}-{end:.3f}.json"

    def get_chunk(self, sha, fingerprint, start, end) -> dict | None:
        p = self._chunk_path(sha, fingerprint, start, end)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put_chunk(self, sha, fingerprint, start, end, payload: dict) -> None:
        # Same audio + model + window gives the same result, so a concurrent winner is fine.
        _atomic_write(self._chunk_path(sha, fingerprint, start, end), json.dumps(payload), identical_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, status TEXT NOT NULL, filename TEXT NOT NULL,
    audio_sha256 TEXT NOT NULL, language TEXT, error TEXT, attempts INTEGER DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_sha ON jobs(audio_sha256);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS segments (
    job_id TEXT NOT NULL REFERENCES jobs(id), segment_id INTEGER NOT NULL,
    start_s REAL NOT NULL, end_s REAL NOT NULL, text TEXT NOT NULL,
    avg_logprob REAL, flags TEXT, PRIMARY KEY (job_id, segment_id)
);
"""


class JobRepository:
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    def _row(self, r) -> Job | None:
        return Job(**dict(r)) if r else None

    def create(self, job: Job) -> Job:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
                (job.id, job.status.value, job.filename, job.audio_sha256, job.language,
                 job.error, job.attempts, job.created_at.isoformat(), job.updated_at.isoformat()))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._row(self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def find_reusable(self, sha: str, language: str | None) -> Job | None:
        """An existing job for the same audio that is done or still in flight (idempotency)."""
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM jobs WHERE audio_sha256=? AND language IS ? AND status IN "
                "('queued','processing','completed') ORDER BY created_at DESC LIMIT 1",
                (sha, language)).fetchone()
        return self._row(r)

    def update(self, job_id: str, status: JobStatus, error: str | None = None,
               bump_attempts: bool = False) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE jobs SET status=?, error=?, attempts=attempts+?, updated_at=? WHERE id=?",
                (status.value, error, 1 if bump_attempts else 0, utcnow().isoformat(), job_id))

    def count_active(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')").fetchone()[0]

    def queue_position(self, job_id: str) -> int | None:
        """1-based position among queued jobs (FIFO by insertion order), or None if not queued."""
        with self._lock:
            r = self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='queued' AND rowid <= "
                "(SELECT rowid FROM jobs WHERE id=? AND status='queued')", (job_id,)).fetchone()
        return r[0] or None

    def unfinished(self) -> list[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','processing') ORDER BY created_at").fetchall()
        return [self._row(r) for r in rows]

    def save_segments(self, job_id: str, result: TranscriptResult) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
            self._conn.executemany(
                "INSERT INTO segments VALUES (?,?,?,?,?,?,?)",
                [(job_id, s.id, s.start, s.end, s.text, s.avg_logprob, ",".join(s.flags))
                 for s in result.segments])
