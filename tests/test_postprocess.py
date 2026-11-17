import pytest

from audio_decoder.exporters import to_srt
from audio_decoder.models import Segment
from audio_decoder.postprocess import clean_text, postprocess, redact


def seg(text, lp=-0.2, nsp=0.02, i=0):
    return Segment(id=i, start=i, end=i + 1, text=text, chunk_index=0, avg_logprob=lp, no_speech_prob=nsp)


def test_clean_text():
    assert clean_text("  hello   world , ok ?  ") == "hello world, ok?"


@pytest.mark.parametrize("text,expected,types", [
    ("mail me at a.b+c@ex-ample.co.uk please", "mail me at [EMAIL] please", ["EMAIL"]),
    ("my email is Sarah.com at example.com", "my email is [EMAIL]", ["EMAIL"]),           # ASR spoken form
    ("it's sarah dot khan at example dot co dot uk thanks", "it's [EMAIL] thanks", ["EMAIL"]),
    ("I'm at home. Then I left at 5.30", "I'm at home. Then I left at 5.30", []),        # not an email
    ("call +1 415 555 0132 now", "call [PHONE] now", ["PHONE"]),
    ("call 0300-1234567", "call [PHONE]", ["PHONE"]),
    ("card 4111 1111 1111 1111.", "card [CARD].", ["CARD"]),
    ("ref 1234 5678 9012 3456", "ref 1234 5678 9012 3456", []),          # fails Luhn: not a card
    ("claim CLM-2026-004417 and POL-48213377", "claim CLM-2026-004417 and POL-48213377", []),  # business ids kept
])
def test_redaction(text, expected, types):
    out, found = redact(text)
    assert out == expected and found == types


def test_hallucination_and_repetition_dropped_low_confidence_flagged():
    segs = [
        seg("Hello there.", i=0),
        seg("Thank you for watching.", lp=-1.2, nsp=0.9, i=1),   # hallucination on silence
        seg("I think so", lp=-1.3, i=2),                          # kept but flagged
        seg("I think so", lp=-1.4, i=3),                          # repetition loop
    ]
    out = postprocess(segs, do_redact=True)
    assert [s.text for s in out] == ["Hello there.", "I think so"]
    assert out[1].flags == ["low_confidence"]
    assert [s.id for s in out] == [0, 1]


def test_srt_format():
    from audio_decoder.models import BackendInfo, SourceInfo, TranscriptResult
    r = TranscriptResult(status="completed", source=SourceInfo(filename="a", sha256="x", duration_s=4000,
                         original_format="wav"), backend=BackendInfo(name="m", fingerprint="m"),
                         language="en", redacted=True, text="hi",
                         segments=[Segment(id=0, start=3661.5, end=3662.25, text="hi", chunk_index=0)],
                         gaps=[], chunks=[], stats={})
    assert to_srt(r) == "1\n01:01:01,500 --> 01:01:02,250\nhi\n"
