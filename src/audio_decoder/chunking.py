"""Stage 2: split long audio into chunks at natural pauses.

Why chunk ourselves instead of handing the whole file to Whisper?
  * parallelism: chunks can be transcribed concurrently;
  * recoverability: a failure costs one chunk, and chunk results are cached;
  * bounded memory/latency per unit of work;
  * silent chunks are skipped entirely (cheaper, and Whisper tends to
    hallucinate text on silence).

Cuts prefer silences close to the target length. If no pause exists before
`max_s`, we make a hard cut and the neighbouring chunks overlap by `overlap_s`.
Each chunk "owns" a half-open time range [own_start, own_end); owned ranges
partition the timeline, which is what lets assembly de-duplicate the overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import ChunkingConfig

FRAME_S = 0.03


@dataclass(frozen=True)
class ChunkPlan:
    index: int
    start: float       # window sent to the backend (includes overlap)
    end: float
    own_start: float   # range this chunk is authoritative for
    own_end: float
    cut: Literal["silence", "hard", "end"]
    is_silent: bool


def frame_db(samples: np.ndarray, sr: int) -> np.ndarray:
    """RMS level per 30 ms frame, in dBFS."""
    n = max(1, int(sr * FRAME_S))
    usable = len(samples) - len(samples) % n
    if usable == 0:
        return np.array([-120.0])
    frames = samples[:usable].astype(np.float32).reshape(-1, n) / 32768.0
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def silence_mask(db: np.ndarray, floor_db: float) -> np.ndarray:
    # Adaptive threshold: above the noise floor, but always well below speech
    # level so a file that is all speech is never classified as silent.
    p10, p90 = np.percentile(db, 10), np.percentile(db, 90)
    thr = min(max(floor_db, p10 + 10), p90 - 15)
    return db < thr


def silence_runs(mask: np.ndarray, min_len_s: float) -> list[tuple[float, float]]:
    runs, start = [], None
    for i, s in enumerate(np.append(mask, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * FRAME_S >= min_len_s:
                runs.append((start * FRAME_S, i * FRAME_S))
            start = None
    return runs


def plan_chunks(samples: np.ndarray, sr: int, cfg: ChunkingConfig) -> list[ChunkPlan]:
    duration = len(samples) / sr
    db = frame_db(samples, sr)
    mask = silence_mask(db, cfg.silence_db)
    runs = silence_runs(mask, cfg.min_silence_s)
    half = cfg.overlap_s / 2

    def voiced_seconds(a: float, b: float) -> float:
        i, j = int(a / FRAME_S), int(np.ceil(b / FRAME_S))
        return float((~mask[i:j]).sum()) * FRAME_S

    plans: list[ChunkPlan] = []
    cursor, prev_hard = 0.0, False
    while cursor < duration - 1e-6:
        # Budget so the window, including any overlap on either side, never exceeds max_s.
        budget = cfg.max_s - (half if prev_hard else 0.0)
        if duration - cursor <= budget:
            cut, kind = duration, "end"
        else:
            lo, hi = cursor + cfg.min_s, cursor + budget - half
            # Any point inside a pause is a valid cut: take the point of each
            # pause closest to the target, so long pauses become silent chunks.
            want = cursor + cfg.target_s
            candidates = [c for a, b in runs
                          if lo <= (c := min(max(want, a + 0.1), b - 0.1)) <= hi]
            if candidates:
                cut = min(candidates, key=lambda c: abs(c - (cursor + cfg.target_s)))
                kind = "silence"
            else:
                cut, kind = hi, "hard"

        win_start = max(0.0, cursor - half) if prev_hard else cursor
        win_end = min(duration, cut + half) if kind == "hard" else cut
        plans.append(ChunkPlan(
            index=len(plans), start=round(win_start, 3), end=round(win_end, 3),
            own_start=round(cursor, 3), own_end=round(cut, 3), cut=kind,
            is_silent=voiced_seconds(cursor, cut) < 0.25,
        ))
        cursor, prev_hard = cut, kind == "hard"
    return plans
