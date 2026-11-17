from __future__ import annotations

import dataclasses
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_decoder.config import Settings

SR = 16_000


def write_wav(path: Path, audio: np.ndarray, sr: int = SR, channels: int = 1) -> Path:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def speech_like(seconds: float, sr: int = SR, f0: float = 200) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return 0.3 * np.sin(2 * np.pi * f0 * t).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.random.default_rng(0).normal(0, 0.0005, int(seconds * sr)).astype(np.float32)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return dataclasses.replace(Settings(), backend="mock", data_dir=tmp_path / "data",
                               retry_base_s=0.0, max_retries=2, workers=2, chunk_workers=3)


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    """The full mock demo call: (wav path, script list)."""
    from make_demo_audio import build
    wav, script = build(tmp_path_factory.mktemp("demo"))
    return wav, json.loads(script.read_text())
