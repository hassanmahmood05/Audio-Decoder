"""Generate mock data: a ~3 minute synthetic "insurance claim call".

Writes:
  samples/demo_call.wav          44.1 kHz stereo (exercises normalisation)
  samples/demo_call.script.json  timed transcript for the mock backend

The audio is tone bursts (one per word) where the script has speech and
near-silence elsewhere, so the real chunker finds real pauses. It includes:
  * normal turns separated by pauses (silence cuts),
  * a ~55 s stretch with no pause (forces a hard cut + overlap),
  * a long hold (fully silent chunk, skipped),
  * a hallucination on silence, a low-confidence line, a repetition loop,
  * PII (email, phone, card) to exercise redaction.

    python scripts/make_demo_audio.py [--out samples]
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 44_100
WPS = 2.6  # words per second

# (speaker text, pause after in seconds, optional overrides)
LINES = [
    ("Good afternoon, thank you for calling Northwind Insurance claims. My name is Ayesha. How can I help you today?", 0.8),
    ("Hi Ayesha, I need to report an accident. Someone hit my car yesterday evening.", 0.7),
    ("I'm sorry to hear that. Is everyone okay?", 0.6),
    ("Yes, thankfully nobody was hurt, but the rear bumper and the tail light are badly damaged.", 0.9),
    ("Can I have your policy number please?", 0.6),
    ("Sure, it's POL-48213377.", 0.8),
    ("Thank you. And the best email to reach you?", 0.5),
    ("It's sarah.khan@example.com, and my phone number is +1 415 555 0132.", 1.0),
    ("Let me put you on a short hold while I pull up the policy.", 0.5),
    # long hold: silent chunk
    ("__HOLD__", 50.0),
    ("Thank you for waiting. I have your policy open now.", 0.2),
    # no-pause monologue (~55 s) -> forces a hard cut
    *[(t, 0.1) for t in [
        "So it happened around six thirty on the main road near the market and traffic was moving slowly because of the rain",
        "and the car behind me did not stop in time and it went straight into the back of my car and pushed it forward a little",
        "the other driver got out and apologised and we exchanged numbers and took photos of both cars and the number plates",
        "then a police officer came over and wrote down what happened and gave me a reference slip which I still have with me",
        "the bumper is cracked the left tail light is broken and the boot does not close properly anymore so I cannot drive it far",
        "I took it to a garage this morning and they said the estimate would be around four thousand eight hundred and fifty dollars",
    ]],
    ("uh it was the uh other driver's fault I think", 0.7, {"avg_logprob": -1.35}),
    ("Understood. I've opened claim CLM-2026-004417 for you.", 0.5),
    ("Understood. I've opened claim CLM-2026-004417 for you.", 0.6, {"avg_logprob": -1.4}),  # repetition loop
    ("For the deductible, can I take a card number?", 0.6),
    ("Yes, it's 4111 1111 1111 1111.", 3.5),
    ("Thank you for watching.", 0.8, {"no_speech_prob": 0.92, "avg_logprob": -1.3, "silent": True}),
    ("Great, that's all done. An adjuster will call you within two business days. Anything else?", 0.7),
    ("No, that's everything. Thanks for your help.", 0.6),
    ("You're welcome. Have a good day.", 1.0),
]


def build(out: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    script, bursts, t = [], [], 1.0
    for entry in LINES:
        text, pause, extra = (*entry, {})[:3]
        if text == "__HOLD__":
            t += pause
            continue
        n = len(text.split())
        dur = n / WPS
        line = {"start": round(t, 3), "end": round(t + dur, 3), "text": text}
        line.update({k: v for k, v in extra.items() if k != "silent"})
        script.append(line)
        if not extra.get("silent"):
            step = dur / n
            for i in range(n):
                bursts.append((t + i * step, t + (i + 1) * step - 0.04))
        t += dur + pause
    total = t + 1.0

    audio = rng.normal(0, 0.0005, int(total * SR)).astype(np.float32)  # faint room noise
    for a, b in bursts:
        i, j = int(a * SR), int(b * SR)
        tt = np.arange(j - i) / SR
        f0 = rng.uniform(140, 260)
        env = np.sin(np.pi * np.linspace(0, 1, j - i)) ** 0.5
        audio[i:j] += 0.3 * env * (np.sin(2 * np.pi * f0 * tt) + 0.4 * np.sin(4 * np.pi * f0 * tt))
    stereo = np.stack([audio, audio * 0.9], axis=1)
    pcm = (np.clip(stereo, -1, 1) * 32767).astype("<i2")

    out.mkdir(parents=True, exist_ok=True)
    wav_path, script_path = out / "demo_call.wav", out / "demo_call.script.json"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    script_path.write_text(json.dumps(script, indent=2), encoding="utf-8")
    return wav_path, script_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    wav, scr = build(Path(ap.parse_args().out))
    print(f"wrote {wav} and {scr}")
