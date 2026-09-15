"""Drawing helpers shared by every scene: palette, fonts, panels, text."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (13, 15, 20)
PANEL = (21, 25, 33)
PANEL_EDGE = (44, 52, 66)
INK = (232, 238, 246)
DIM = (128, 140, 158)
ACCENT = (108, 214, 126)      # mushroom-body green
WARN = (240, 176, 64)
HOT = (232, 84, 96)
COOL = (96, 186, 240)

POP_COLOURS = {
    "VPN": (96, 186, 240),
    "KC": (198, 146, 246),
    "MBON": (108, 214, 126),
    "DAN": (240, 176, 64),
    "CX": (244, 132, 196),
    "DN": (232, 84, 96),
}

_FONT_DIR = Path("/usr/share/fonts/truetype")
_MONO = _FONT_DIR / "dejavu" / "DejaVuSansMono-Bold.ttf"
_SANS = _FONT_DIR / "dejavu" / "DejaVuSans-Bold.ttf"
_SANS_REG = _FONT_DIR / "dejavu" / "DejaVuSans.ttf"
_cache: dict = {}


def font(size: int, kind: str = "sans") -> ImageFont.FreeTypeFont:
    key = (size, kind)
    if key not in _cache:
        path = {"mono": _MONO, "sans": _SANS, "reg": _SANS_REG}[kind]
        _cache[key] = ImageFont.truetype(str(path), size)
    return _cache[key]


def canvas(w: int, h: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    return img


def panel(draw: ImageDraw.ImageDraw, box, radius: int = 10, fill=PANEL, edge=PANEL_EDGE):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=edge, width=2)


def text(draw, xy, s, size=20, kind="sans", colour=INK, anchor="la"):
    draw.text(xy, s, font=font(size, kind), fill=colour, anchor=anchor)


def bar(draw, box, frac, colour=ACCENT, back=(38, 44, 56), radius=4):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=radius, fill=back)
    w = max(0, min(1.0, float(frac))) * (x1 - x0)
    if w > 1:
        draw.rounded_rectangle((x0, y0, x0 + w, y1), radius=radius, fill=colour)


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def fit_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((w, h), Image.NEAREST))


def paste(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape[:2]
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(dst.shape[0], top + h), min(dst.shape[1], left + w)
    if y0 < y1 and x0 < x1:
        dst[y0:y1, x0:x1] = src[y0 - top:y1 - top, x0 - left:x1 - left]


def format_time(ticks: int, tps: int = 20) -> str:
    """MineSim ticks as a speedrun clock (20 ticks per second, as in Minecraft)."""
    total = ticks / tps
    return f"{int(total // 60):d}:{total % 60:05.2f}"


def wrap(draw, text_: str, fnt, max_width: int) -> list[str]:
    words, lines, cur = text_.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=fnt) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def caption_over(frame: np.ndarray, box, text_: str, size: int = 30,
                 alpha: float = 0.72, colour=INK) -> np.ndarray:
    """A readable caption band drawn over live footage."""
    if not text_:
        return frame
    x0, y0, x1, y1 = box
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil)
    fnt = font(size, "sans")
    lines = wrap(d, text_, fnt, x1 - x0 - 56)
    lh = int(size * 1.35)
    h = lh * len(lines) + 34
    top = y1 - h
    band = frame[top:y1, x0:x1].astype(np.float32)
    frame = frame.copy()
    frame[top:y1, x0:x1] = (band * (1 - alpha) + np.array(BG) * alpha).astype(np.uint8)
    pil = Image.fromarray(frame)
    d = ImageDraw.Draw(pil)
    d.line((x0, top, x1, top), fill=ACCENT, width=3)
    for i, line in enumerate(lines):
        d.text((x0 + 28, top + 20 + i * lh), line, font=fnt, fill=colour)
    return np.asarray(pil).copy()
