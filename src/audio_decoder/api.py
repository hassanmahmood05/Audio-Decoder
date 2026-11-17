"""REST API.

POST /v1/transcriptions          upload audio -> 202 + job (or 200 + existing job)
GET  /v1/transcriptions/{id}     job status (+ result when finished)
GET  /v1/transcriptions/{id}/result?format=json|txt|srt|jsonl
GET  /health
"""
from __future__ import annotations

import hashlib
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from .backends import create_backend
from .config import Settings
from .exporters import EXPORTERS
from .ingest import SUPPORTED_EXTENSIONS, IngestError, probe
from .jobs import JobManager, QueueFull
from .models import JobStatus

log = logging.getLogger("audio_decoder.api")
FINISHED = {JobStatus.completed, JobStatus.partial, JobStatus.failed}


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mgr = manager or JobManager(settings, create_backend(settings))
        app.state.jobs = mgr
        mgr.start()
        yield
        mgr.stop()

    app = FastAPI(title="Audio Decoder", version="1.0.0", lifespan=lifespan,
                  description="Asynchronous speech-to-text with timestamped segments.")

    def require_key(x_api_key: str | None = Header(default=None)):
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(401, "Invalid or missing X-API-Key")

    def jobs() -> JobManager:
        return app.state.jobs

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": settings.backend}

    @app.post("/v1/transcriptions", status_code=202, dependencies=[Depends(require_key)])
    async def create_transcription(file: UploadFile = File(...),
                                   language: str | None = Form(default=None)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type {ext!r}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}")

        # Stream to disk in 1 MB pieces: hash and enforce the size limit without
        # holding the whole upload in memory.
        limit = settings.max_upload_mb * 1024 * 1024
        h, size = hashlib.sha256(), 0
        tmp_dir = settings.data_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=ext, delete=False) as tmp:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    tmp.close()
                    Path(tmp.name).unlink(missing_ok=True)
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
                h.update(chunk)
                tmp.write(chunk)
        tmp_path = Path(tmp.name)
        if size == 0:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(400, "Empty file")

        # Validate by content, not just extension, so clients get an immediate error.
        try:
            meta = await run_in_threadpool(probe, tmp_path)
            duration = float(meta["format"].get("duration") or 0)
            if duration > settings.max_duration_s:
                raise IngestError(f"Audio is {duration:.0f}s; limit is {settings.max_duration_s:.0f}s")
        except IngestError as e:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(422, str(e))

        # submit() owns tmp_path from here: it stores it, or deletes it for a duplicate or a full queue.
        try:
            job, created = await run_in_threadpool(
                jobs().submit, tmp_path, h.hexdigest(), ext, file.filename or tmp_path.name, language)
        except QueueFull as e:
            return JSONResponse({"detail": str(e)}, status_code=429,
                                headers={"Retry-After": str(e.retry_after_s)})
        body = {"job": job.model_dump(mode="json"), "created": created,
                "status_url": f"/v1/transcriptions/{job.id}"}
        return JSONResponse(body, status_code=202 if created else 200)

    @app.get("/v1/transcriptions/{job_id}", dependencies=[Depends(require_key)])
    def get_transcription(job_id: str):
        job = jobs().get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        body = {"job": job.model_dump(mode="json")}
        body["job"]["queue_position"] = jobs().queue_position(job_id) if job.status == JobStatus.queued else None
        if job.status in FINISHED and (result := jobs().result(job)):
            body["result"] = result.model_dump(mode="json")
        return body

    @app.get("/v1/transcriptions/{job_id}/result", dependencies=[Depends(require_key)])
    def get_result(job_id: str, format: str = Query("json", pattern="^(json|txt|srt|jsonl)$")):
        job = jobs().get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        if job.status not in FINISHED:
            raise HTTPException(409, f"Job is {job.status.value}; result not ready")
        result = jobs().result(job)
        if result is None:
            raise HTTPException(404, job.error or "No result available")
        render, media = EXPORTERS[format]
        return Response(render(result), media_type=media)

    return app


app = create_app()  # uvicorn audio_decoder.api:app
