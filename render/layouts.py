"""Full-frame layouts: one landscape (YouTube) and one portrait (Reels/TikTok)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from minesim.world import MILESTONES, N_MILESTONES, MineSim
from render import scene, tiles, ui

LANDSCAPE = (1920, 1080)
PORTRAIT = (1080, 1920)


def _draw(img: np.ndarray):
    pil = Image.fromarray(img)
    return pil, ImageDraw.Draw(pil)


def _kc_raster(rates: np.ndarray, rows: int = 64, cols: int | None = None) -> np.ndarray:
    """Kenyon-cell activity as a sparse-looking raster block."""
    cols = cols or rows
    n = rows * cols
    v = np.zeros(n, np.float32)
    v[:min(n, rates.size)] = rates[:n]
    v = np.clip(v / (v.max() + 1e-6), 0, 1).reshape(rows, cols)
    col = np.array(ui.POP_COLOURS["KC"], np.float32)
    base = np.array((26, 28, 38), np.float32)
    return (base + (col - base) * v[..., None]).astype(np.uint8)


def _header(draw, w: int, title: str, right: str = "", sub: str = "", y: int = 0,
            h: int = 84):
    draw.rectangle((0, y, w, y + h), fill=ui.PANEL)
    ui.text(draw, (34, y + h // 2 - (10 if sub else 0)), title, size=30, anchor="lm")
    if sub:
        ui.text(draw, (34, y + h // 2 + 18), sub, size=17, kind="reg",
                colour=ui.DIM, anchor="lm")
    if right:
        ui.text(draw, (w - 34, y + h // 2), right, size=26, kind="mono",
                colour=ui.ACCENT, anchor="rm")


# --------------------------------------------------------------- hero frames
def hero_landscape(env: MineSim, rates: dict, kc: np.ndarray, mb_gain: float,
                   agent: int, title: str, right: str = "", sub: str = "",
                   hud_label: str = "SPEEDRUN TIMER", hud_sub: str = "") -> np.ndarray:
    W, H = LANDSCAPE
    img = ui.canvas(W, H)
    view = scene.world_image(env, agent, view=20)          # 320 x 320
    ui.paste(img, ui.upscale(view, 3), 130, 40)            # 960 x 960
    pil, d = _draw(img)
    _header(d, W, title, right, sub)
    d.rectangle((38, 128, 1002, 1092), outline=ui.PANEL_EDGE, width=2)

    scene.draw_hud(d, (1030, 130, 1880, 268), env, agent, hud_label, hud_sub)

    ui.panel(d, (1030, 288, 1450, 1040))
    ui.text(d, (1050, 302), "ANY% ROUTE", size=17, colour=ui.DIM)
    scene.draw_route(d, (1050, 336, 1430, 1020), int(env.stage()[agent]))

    scene.draw_brain(d, pil, (1474, 288, 1880, 1040), rates,
                     int((kc > 0).sum()), mb_gain, _kc_raster(kc))
    return np.asarray(pil).copy()


def hero_portrait(env: MineSim, rates: dict, kc: np.ndarray, mb_gain: float,
                  agent: int, title: str, right: str = "", sub: str = "",
                  hud_label: str = "SPEEDRUN TIMER", hud_sub: str = "") -> np.ndarray:
    W, H = PORTRAIT
    img = ui.canvas(W, H)
    view = scene.world_image(env, agent, view=20)          # 320
    ui.paste(img, ui.upscale(view, 3), 210, 60)            # 960 x 960
    pil, d = _draw(img)
    _header(d, W, title, right, sub, h=140)
    d.rectangle((58, 208, 1022, 1172), outline=ui.PANEL_EDGE, width=2)
    scene.draw_hud(d, (60, 1200, 1020, 1340), env, agent, hud_label, hud_sub)
    ui.panel(d, (60, 1360, 1020, 1470))
    ui.text(d, (80, 1378), "ANY% ROUTE", size=16, colour=ui.DIM)
    scene.draw_route(d, (80, 1402, 1000, 1452), int(env.stage()[agent]), compact=True)
    scene.draw_brain(d, pil, (60, 1494, 1020, 1880), rates,
                     int((kc > 0).sum()), mb_gain, _kc_raster(kc, rows=22, cols=182))
    return np.asarray(pil).copy()


# --------------------------------------------------------- population frames
def _grid_block(env: MineSim, order: np.ndarray, cols: int, rows: int, cell: int,
                gap: int) -> np.ndarray:
    w = cols * (cell + gap) - gap
    h = rows * (cell + gap) - gap
    block = np.empty((h, w, 3), np.uint8)
    block[:] = ui.BG
    for i, a in enumerate(order[: cols * rows]):
        r, c = divmod(i, cols)
        thumb = scene.minimap(env, int(a), cell)
        if env.won[a]:
            thumb[:3] = thumb[-3:] = thumb[:, :3] = thumb[:, -3:] = ui.ACCENT
        elif not env.alive[a]:
            thumb = (thumb * 0.35).astype(np.uint8)
        ui_top, ui_left = r * (cell + gap), c * (cell + gap)
        block[ui_top:ui_top + cell, ui_left:ui_left + cell] = thumb
    return block


def population_landscape(env: MineSim, title: str, right: str = "", sub: str = "",
                         cols: int = 11, rows: int = 7, stats: list[str] | None = None
                         ) -> np.ndarray:
    W, H = LANDSCAPE
    img = ui.canvas(W, H)
    order = np.argsort(-env.progress())
    cell, gap = 128, 8
    block = _grid_block(env, order, cols, rows, cell, gap)
    ui.paste(img, block, 120, 40)
    pil, d = _draw(img)
    _header(d, W, title, right, sub)
    if stats:
        x = 40 + block.shape[1] + 24
        ui.panel(d, (x, 120, W - 40, 120 + block.shape[0]))
        for i, line in enumerate(stats):
            ui.text(d, (x + 22, 150 + i * 32), line, size=17, kind="mono",
                    colour=ui.INK if i == 0 else ui.DIM)
    return np.asarray(pil).copy()


def population_portrait(env: MineSim, title: str, right: str = "", sub: str = "",
                        cols: int = 7, rows: int = 10,
                        stats: list[str] | None = None) -> np.ndarray:
    W, H = PORTRAIT
    img = ui.canvas(W, H)
    order = np.argsort(-env.progress())
    cell, gap = 140, 8
    block = _grid_block(env, order, cols, rows, cell, gap)
    ui.paste(img, block, 220, (W - block.shape[1]) // 2)
    pil, d = _draw(img)
    _header(d, W, title, right, sub, h=180)
    if stats:
        y = 220 + block.shape[0] + 30
        ui.panel(d, (60, y, W - 60, H - 60))
        for i, line in enumerate(stats):
            ui.text(d, (90, y + 26 + i * 36), line, size=22, kind="mono",
                    colour=ui.INK if i == 0 else ui.DIM)
    return np.asarray(pil).copy()


# ------------------------------------------------------------------- cards
def title_card(size, lines: list[tuple[str, int, tuple]], align: str = "center",
               foot: str = "") -> np.ndarray:
    W, H = size
    img = ui.canvas(W, H)
    pil, d = _draw(img)
    total = sum(s + 22 for _, s, _ in lines)
    y = (H - total) // 2
    for txt, s, col in lines:
        x = W // 2 if align == "center" else 90
        ui.text(d, (x, y), txt, size=s, colour=col,
                anchor=("ma" if align == "center" else "la"))
        y += s + 22
    if foot:
        ui.text(d, (W // 2, H - 70), foot, size=20, kind="reg", colour=ui.DIM, anchor="ma")
    return np.asarray(pil).copy()
