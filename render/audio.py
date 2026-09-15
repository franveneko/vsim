"""A small procedural chiptune generator.

No sample packs, no licensing questions: every sound here is synthesised from
numpy arrays and written out as a WAV that ffmpeg muxes onto the finished video.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
A4 = 69


def midi(n: int) -> float:
    return 440.0 * 2 ** ((n - A4) / 12)


# chord progressions, as MIDI note numbers of the root plus chord shape
MINOR = (0, 3, 7, 10)
MAJOR = (0, 4, 7, 11)

MOODS = {
    # name:        bpm  roots                      shape  drums  arp
    "intro":      (88,  (45, 41, 43, 40), MINOR, 0.25, 0.55),
    "build":      (112, (45, 50, 43, 48), MINOR, 0.55, 0.85),
    "drive":      (128, (45, 45, 48, 43), MINOR, 0.80, 1.00),
    "triumph":    (124, (48, 55, 53, 50), MAJOR, 0.75, 0.95),
    "calm":       (80,  (45, 48, 41, 43), MINOR, 0.10, 0.40),
}


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    a = max(1, int(attack * n))
    e = np.ones(n, np.float32)
    e[:a] = np.linspace(0, 1, a)
    d = np.exp(-np.linspace(0, decay, n)).astype(np.float32)
    return e * d


def _osc(freq: float, n: int, kind: str) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    phase = 2 * np.pi * freq * t
    if kind == "square":
        return np.sign(np.sin(phase)).astype(np.float32)
    if kind == "pulse":
        return np.where((phase % (2 * np.pi)) < np.pi * 0.5, 1.0, -1.0).astype(np.float32)
    if kind == "tri":
        return (2 / np.pi * np.arcsin(np.sin(phase))).astype(np.float32)
    return np.sin(phase).astype(np.float32)


def _add(buf: np.ndarray, start: int, wave_: np.ndarray, gain: float) -> None:
    end = min(len(buf), start + len(wave_))
    if end > start:
        buf[start:end] += wave_[: end - start] * gain


def section(mood: str, seconds: float, seed: int = 0) -> np.ndarray:
    bpm, roots, shape, drum_mix, arp_mix = MOODS[mood]
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(seconds * SR)
    buf = np.zeros(n + SR, np.float32)

    bar = 4 * beat
    n_bars = int(np.ceil(seconds / bar))
    for b in range(n_bars):
        root = roots[b % len(roots)]
        t0 = b * bar

        # bass: root on every beat, with an octave lift on the last one
        for k in range(4):
            note = root - 12 + (12 if k == 3 else 0)
            ln = int(beat * 0.9 * SR)
            _add(buf, int((t0 + k * beat) * SR),
                 _osc(midi(note), ln, "square") * _env(ln, 0.01, 3.0), 0.22)

        # pad: the chord, held across the bar
        ln = int(bar * SR)
        for iv in shape[:3]:
            _add(buf, int(t0 * SR),
                 _osc(midi(root + 12 + iv), ln, "tri") * _env(ln, 0.25, 1.2), 0.055)

        # arpeggio: sixteenths climbing the chord
        steps = 16
        for k in range(steps):
            iv = shape[k % len(shape)]
            octv = 24 + 12 * ((k // len(shape)) % 2)
            ln = int(beat / 4 * 0.95 * SR)
            _add(buf, int((t0 + k * beat / 4) * SR),
                 _osc(midi(root + octv + iv), ln, "pulse") * _env(ln, 0.005, 7.0),
                 0.085 * arp_mix)

        # drums
        for k in range(8):
            t = t0 + k * beat / 2
            if k % 4 == 0:                                  # kick
                ln = int(0.16 * SR)
                sweep = np.exp(-np.linspace(0, 9, ln)).astype(np.float32)
                ph = 2 * np.pi * np.cumsum(50 + 90 * sweep) / SR
                _add(buf, int(t * SR), np.sin(ph).astype(np.float32) * sweep,
                     0.5 * drum_mix)
            ln = int(0.05 * SR)                             # hat
            noise = rng.standard_normal(ln).astype(np.float32)
            _add(buf, int(t * SR), noise * _env(ln, 0.001, 9.0),
                 0.045 * drum_mix * (1.0 if k % 2 else 0.6))
        if b % 4 == 3:                                      # snare accent
            ln = int(0.18 * SR)
            noise = rng.standard_normal(ln).astype(np.float32)
            _add(buf, int((t0 + 3 * beat) * SR), noise * _env(ln, 0.002, 6.0),
                 0.22 * drum_mix)

    out = buf[:n]
    # gentle fades so sections butt together without clicks
    f = int(0.05 * SR)
    out[:f] *= np.linspace(0, 1, f)
    out[-f:] *= np.linspace(1, 0, f)
    return out


def build(plan: list[tuple[str, float]], path: Path, seed: int = 0,
          peak: float = 0.72) -> Path:
    """plan: [(mood, seconds), ...] -> a 16-bit mono WAV."""
    track = np.concatenate([section(m, s, seed + i) for i, (m, s) in enumerate(plan)])
    track = np.tanh(track * 1.4)                       # soft clip instead of hard
    track = track / (np.abs(track).max() + 1e-9) * peak
    data = (track * 32767).astype(np.int16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(data.tobytes())
    return path
