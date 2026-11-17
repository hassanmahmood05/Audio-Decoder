"""Stage 4: make raw model output safe and useful for downstream systems.

  * normalise whitespace/punctuation spacing;
  * drop likely hallucinations (Whisper invents text such as "Thank you for
    watching." on silence, and can loop on the same sentence);
  * flag low-confidence segments instead of silently trusting them;
  * redact contact/payment PII before text reaches search or a warehouse.
"""
from __future__ import annotations

import re

from .models import Segment

LOW_LOGPROB = -1.0
HIGH_NO_SPEECH = 0.6
KNOWN_HALLUCINATIONS = {
    "thank you for watching.", "thanks for watching!", "thanks for watching.",
    "please subscribe.", "subtitles by the amara.org community", "you",
}

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# ASR usually writes a spoken address without "@": "sarah dot khan at example dot com",
# "sarah.khan at example.com". A literal "." must not be followed by a space, so sentence
# ends ("I'm at home. Then ...") don't match. Every repetition is bounded (RFC 5321 sizes,
# a few dotted parts) so each start position does constant work: linear, not quadratic,
# on long runs like "a dot a dot ..." that never reach " at ".
_DOT = r"(?:\.|\s{1,3}dot\s{1,3})"
_SPOKEN_EMAIL = re.compile(
    rf"\b[\w+-]{{1,64}}(?:{_DOT}[\w+-]{{1,64}}){{0,4}}\s{{1,3}}at\s{{1,3}}"
    rf"[\w-]{{1,63}}(?:{_DOT}[\w-]{{1,63}}){{0,3}}{_DOT}[a-z]{{2,24}}\b", re.IGNORECASE)
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?\(?\d{3,4}\)?[\s.-]?\d{3}[\s.-]?\d{3,4}(?![\w-])")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", text)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total, alt = total + n, not alt
    return total % 10 == 0


def redact(text: str) -> tuple[str, list[str]]:
    """Replace PII with typed placeholders. Returns (text, found types)."""
    found: list[str] = []

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append("CARD")
            return "[CARD]"
        return m.group()

    text = _CARD.sub(card, text)  # cards first: their digit runs also look like phones

    def sub(pattern: re.Pattern, label: str, s: str) -> str:
        def repl(_m):
            found.append(label)
            return f"[{label}]"
        return pattern.sub(repl, s)

    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_SPOKEN_EMAIL, "EMAIL", text)
    text = sub(_PHONE, "PHONE", text)
    return text, found


def is_hallucination(seg: Segment) -> bool:
    t = seg.text.strip().lower()
    nsp, lp = seg.no_speech_prob or 0.0, seg.avg_logprob or 0.0
    if t in KNOWN_HALLUCINATIONS and nsp > 0.5:
        return True
    return nsp > 0.8 and lp < LOW_LOGPROB


def postprocess(segments: list[Segment], do_redact: bool = True) -> list[Segment]:
    out: list[Segment] = []
    for seg in segments:
        seg.text = clean_text(seg.text)
        if not seg.text or is_hallucination(seg):
            continue
        # Whisper repetition loop: identical consecutive text with low confidence.
        if out and seg.text.lower() == out[-1].text.lower() and (seg.avg_logprob or 0) < LOW_LOGPROB:
            continue
        if (seg.avg_logprob is not None and seg.avg_logprob < LOW_LOGPROB) or \
           (seg.no_speech_prob is not None and seg.no_speech_prob > HIGH_NO_SPEECH):
            seg.flags.append("low_confidence")
        if do_redact:
            seg.text, found = redact(seg.text)
            seg.flags += sorted({f"redacted_{f.lower()}" for f in found})
        out.append(seg)
    for i, seg in enumerate(out):
        seg.id = i
    return out
