import numpy as np
import pytest
from conftest import SR, silence, speech_like, write_wav

from audio_decoder.chunking import plan_chunks
from audio_decoder.config import ChunkingConfig
from audio_decoder.ingest import TARGET_SR, IngestError, load_audio


def test_stereo_44k_is_normalised_to_16k_mono(tmp_path):
    audio = speech_like(2.0, sr=44_100)
    stereo = np.stack([audio, audio], axis=1).ravel()
    asset = load_audio(write_wav(tmp_path / "a.wav", stereo, sr=44_100, channels=2))
    assert asset.sample_rate == TARGET_SR
    assert asset.original_sample_rate == 44_100 and asset.original_channels == 2
    assert abs(asset.duration_s - 2.0) < 0.05
    assert len(asset.sha256) == 64


def test_mp3_is_accepted(tmp_path):
    import subprocess
    wav = write_wav(tmp_path / "a.wav", speech_like(1.0))
    mp3 = tmp_path / "a.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    assert abs(load_audio(mp3).duration_s - 1.0) < 0.1


@pytest.mark.parametrize("name,content,msg", [
    ("a.txt", b"hello", "Unsupported"),
    ("a.wav", b"", "empty"),
    ("a.wav", b"not really audio at all", "ffprobe"),
])
def test_bad_inputs_are_rejected(tmp_path, name, content, msg):
    p = tmp_path / name
    p.write_bytes(content)
    with pytest.raises(IngestError, match=msg):
        load_audio(p)


def _int16(a):
    return (a * 32767).astype(np.int16)


def test_cuts_at_pauses():
    # 20s speech, 1s pause, 20s speech, 1s pause, 20s speech
    a = np.concatenate([speech_like(20), silence(1), speech_like(20), silence(1), speech_like(20)])
    plans = plan_chunks(_int16(a), SR, ChunkingConfig(target_s=25, min_s=10, max_s=45))
    assert [p.cut for p in plans[:-1]] == ["silence"] * (len(plans) - 1)
    assert 20.0 <= plans[0].own_end <= 21.0  # inside the first pause
    assert all(p.start == p.own_start and p.end == p.own_end for p in plans)  # no overlap needed


def test_hard_cut_with_overlap_when_no_pause():
    plans = plan_chunks(_int16(speech_like(100)), SR, ChunkingConfig(max_s=45, overlap_s=2))
    assert plans[0].cut == "hard"
    assert plans[0].end - plans[0].own_end == pytest.approx(1.0)
    assert plans[1].own_start - plans[1].start == pytest.approx(1.0)
    # owned ranges partition the timeline exactly
    assert plans[0].own_start == 0
    for a, b in zip(plans, plans[1:]):
        assert a.own_end == b.own_start
    assert plans[-1].own_end == pytest.approx(100, abs=0.01)
    assert all(p.end - p.start <= 45 + 1e-6 for p in plans)


def test_long_silence_becomes_skipped_chunk():
    a = np.concatenate([speech_like(15), silence(60), speech_like(15)])
    plans = plan_chunks(_int16(a), SR, ChunkingConfig())
    assert any(p.is_silent for p in plans)
    assert not plans[0].is_silent and not plans[-1].is_silent
