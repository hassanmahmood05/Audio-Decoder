"""Command-line entry point: transcribe a file without running the API.

    audio-decoder transcribe call.mp3 --out out/ --formats json,srt,txt
    audio-decoder serve --port 8000
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from .backends import create_backend
from .config import Settings
from .exporters import EXPORTERS
from .ingest import IngestError, load_audio
from .pipeline import Pipeline
from .storage import BlobStore


def _transcribe(args) -> int:
    overrides = {k: v for k, v in {"backend": args.backend, "whisper_model": args.model}.items() if v}
    if args.no_redact:
        overrides["redact_pii"] = False
    settings = dataclasses.replace(Settings(), **overrides)
    if args.mock_script:
        from .backends.mock import MockBackend
        backend = MockBackend(args.mock_script)
    else:
        backend = create_backend(settings)

    try:
        asset = load_audio(args.input, settings.max_duration_s)
    except IngestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    store = BlobStore(settings.data_dir) if not args.no_cache else None
    result = Pipeline(backend, settings, store).run(asset, args.language)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats.split(","):
        render, _ = EXPORTERS[fmt.strip()]
        path = out / f"{Path(args.input).stem}.{fmt.strip()}"
        path.write_text(render(result), encoding="utf-8")
        print(f"wrote {path}")
    print(f"status={result.status} segments={result.stats['segments']} "
          f"chunks={result.stats['chunks']} rtf={result.stats['real_time_factor']}")
    return 0 if result.status == "completed" else 1


def _serve(args) -> int:
    import uvicorn
    uvicorn.run("audio_decoder.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="audio-decoder")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="transcribe one file")
    t.add_argument("input")
    t.add_argument("--out", default="output")
    t.add_argument("--formats", default="json,srt,txt")
    t.add_argument("--language", default=None, help="e.g. en; default auto-detect")
    t.add_argument("--backend", choices=["whisper", "mock"])
    t.add_argument("--model", help="whisper model size: tiny|base|small|medium|large-v3")
    t.add_argument("--mock-script", help="JSON script for the mock backend")
    t.add_argument("--no-redact", action="store_true")
    t.add_argument("--no-cache", action="store_true")
    t.set_defaults(func=_transcribe)

    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
