"""Output formats for different consumers."""
from __future__ import annotations

import json

from .models import TranscriptResult


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_json(result: TranscriptResult) -> str:
    return result.model_dump_json(indent=2)


def to_txt(result: TranscriptResult) -> str:
    return result.text + "\n"


def to_srt(result: TranscriptResult) -> str:
    blocks = [f"{i}\n{_ts(s.start)} --> {_ts(s.end)}\n{s.text}\n"
              for i, s in enumerate(result.segments, 1)]
    return "\n".join(blocks)


def to_jsonl(result: TranscriptResult) -> str:
    """One row per segment, ready for a warehouse / search index bulk load."""
    rows = [json.dumps({
        "audio_sha256": result.source.sha256, "segment_id": s.id, "start": s.start, "end": s.end,
        "text": s.text, "language": result.language, "flags": s.flags,
        "model": result.backend.fingerprint, "schema_version": result.schema_version,
    }) for s in result.segments]
    return "\n".join(rows) + ("\n" if rows else "")


EXPORTERS = {"json": (to_json, "application/json"), "txt": (to_txt, "text/plain"),
             "srt": (to_srt, "application/x-subrip"), "jsonl": (to_jsonl, "application/x-ndjson")}
