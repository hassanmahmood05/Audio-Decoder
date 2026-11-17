"""Asynchronous job processing.

The API never transcribes inside the request: it stores the upload, records a
job and returns 202 immediately. A bounded pool of workers consumes the queue,
so concurrent uploads queue up instead of overloading the CPU/GPU.

Locally the queue is in-process; in production the same JobManager interface
would publish to Azure Service Bus / SQS / Redis and workers would run as a
separate, independently scaled deployment.
"""
from __future__ import annotations

import logging
import queue
import threading
import uuid
from pathlib import Path

from .backends import TranscriptionBackend
from .config import Settings
from .ingest import IngestError, load_audio
from .models import Job, JobStatus, TranscriptResult
from .pipeline import Pipeline
from .storage import BlobStore, JobRepository

log = logging.getLogger("audio_decoder.jobs")


class QueueFull(Exception):
    """Too many pending jobs; the client should retry later (HTTP 429)."""

    def __init__(self, pending: int, retry_after_s: int):
        super().__init__(f"{pending} jobs pending; try again later")
        self.pending, self.retry_after_s = pending, retry_after_s


class JobManager:
    def __init__(self, settings: Settings, backend: TranscriptionBackend):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.store = BlobStore(settings.data_dir)
        self.repo = JobRepository(settings.data_dir / "jobs.db")
        self.pipeline = Pipeline(backend, settings, self.store)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._threads: list[threading.Thread] = []
        # Serialises dedup check + capacity check + create, so identical concurrent
        # uploads yield one job. (Multi-instance: enforce with a DB unique constraint.)
        self._submit_lock = threading.Lock()

    # lifecycle -----------------------------------------------------------
    def start(self) -> None:
        # Recovery: jobs left queued/processing by a crash or restart are re-queued.
        # Chunk-level caching means they resume rather than start over.
        for job in self.repo.unfinished():
            log.info("re-queueing unfinished job %s", job.id)
            self._queue.put(job.id)
        for i in range(max(1, self.settings.workers)):
            t = threading.Thread(target=self._worker, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        for _ in self._threads:
            self._queue.put(None)
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()

    # submission ----------------------------------------------------------
    def submit(self, upload: Path, sha: str, ext: str, filename: str,
               language: str | None) -> tuple[Job, bool]:
        """Take ownership of an uploaded file and return (job, created).

        Same audio + language returns the existing job (and discards the upload),
        even when the queue is full, since that costs no new work. Otherwise raises
        QueueFull once `max_pending_jobs` are queued or processing.
        """
        with self._submit_lock:
            if existing := self.repo.find_reusable(sha, language):
                upload.unlink(missing_ok=True)
                return existing, False
            pending = self.repo.count_active()
            if pending >= self.settings.max_pending_jobs:
                upload.unlink(missing_ok=True)
                raise QueueFull(pending, retry_after_s=30)
            self.store.adopt_audio(upload, sha, ext)
            job = self.repo.create(Job(id=uuid.uuid4().hex, status=JobStatus.queued,
                                       filename=filename, audio_sha256=sha, language=language))
        self._queue.put(job.id)
        return job, True

    def get(self, job_id: str) -> Job | None:
        return self.repo.get(job_id)

    def queue_position(self, job_id: str) -> int | None:
        return self.repo.queue_position(job_id)

    def result(self, job: Job) -> TranscriptResult | None:
        return self.store.load_transcript(job.id)

    def wait_idle(self) -> None:
        self._queue.join()

    # worker --------------------------------------------------------------
    def _worker(self) -> None:
        while (job_id := self._queue.get()) is not None:
            try:
                self.process(job_id)
            except Exception:  # never let one job kill the worker
                log.exception("unexpected error in job %s", job_id)
            finally:
                self._queue.task_done()
        self._queue.task_done()

    def process(self, job_id: str) -> None:
        job = self.repo.get(job_id)
        if job is None:
            return
        self.repo.update(job_id, JobStatus.processing, bump_attempts=True)
        audio = next(self.store.root.joinpath("audio").glob(f"{job.audio_sha256}.*"), None)
        try:
            if audio is None:
                raise IngestError("stored audio not found")
            asset = load_audio(audio, self.settings.max_duration_s)
            result = self.pipeline.run(asset, job.language, filename=job.filename)
        except IngestError as e:
            self.repo.update(job_id, JobStatus.failed, error=str(e))
            return
        except Exception as e:
            self.repo.update(job_id, JobStatus.failed, error=f"{type(e).__name__}: {e}")
            raise

        self.store.save_transcript(job_id, result)
        self.repo.save_segments(job_id, result)
        error = "; ".join(f"{g.start:.1f}-{g.end:.1f}s: {g.reason}" for g in result.gaps) or None
        self.repo.update(job_id, JobStatus(result.status), error=error)
        log.info("job %s %s: %d segments, rtf=%s", job_id, result.status,
                 result.stats["segments"], result.stats["real_time_factor"])
