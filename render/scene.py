"""Compose one video frame from the live state of a MineSim population."""
from __future__ import annotations

import numpy as np
from PIL import Image

from minesim.world import ITEM_IX, MILESTONES, N_MILESTONES, N_MOBS, MineSim
from render import tiles, ui

TILE = tiles.TILE


# --------------------------------------------------------------------- views
def world_image(env: MineSim, agent: int, view: int | None = None) -> np.ndarray:
    """The agent's world, drawn with mobs and the fly itself.

    `view` crops a square window of that many blocks around the agent; None
    draws the whole map.
    """
    grid = env.grid[env.dim[agent], agent]
    img = tiles.render_world(grid).copy()

    for m in range(N_MOBS):
        if env.mob_hp[agent, m] > 0 and env.mob_dim[agent, m] == env.dim[agent]:
            tiles.blit(img, tiles.MOBS[env.mob_kind[agent, m]],
                       int(env.mob_y[agent, m]) * TILE, int(env.mob_x[agent, m]) * TILE)
    tiles.blit(img, tiles.FLY, int(env.y[agent]) * TILE, int(env.x[agent]) * TILE)

    if view is not None:
        s = env.SIZE
        half = view // 2
        cx = int(np.clip(env.x[agent], half, s - half))
        cy = int(np.clip(env.y[agent], half, s - half))
        img = img[(cy - half) * TILE:(cy + half) * TILE,
                  (cx - half) * TILE:(cx + half) * TILE]
    return img


def minimap(env: MineSim, agent: int, px: int) -> np.ndarray:
    """A tiny thumbnail of one agent's world, for the population grid."""
    grid = env.grid[env.dim[agent], agent]
    small = tiles.ATLAS[grid][:, :, 8, 8]                 # one pixel per block
    img = ui.fit_nearest(small, px, px).copy()
    s = env.SIZE
    k = max(2, px // s)
    y = int(env.y[agent]) * px // s
    x = int(env.x[agent]) * px // s
    img[max(0, y - k):y + k, max(0, x - k):x + k] = (255, 96, 96)
    return img


# -------------------------------------------------------------------- panels
def _step_colour(i: int, stage: int):
    return ui.ACCENT if i < stage else (ui.WARN if i == stage else ui.DIM)


def draw_route(draw, box, stage: int, *, compact: bool = False):
    """The speedrun checklist: a labelled column, or a row of pips when compact."""
    x0, y0, x1, y1 = box
    n = N_MILESTONES
    if compact:
        step = (x1 - x0) / n
        r = min(step * 0.26, (y1 - y0) * 0.3)
        cy = (y0 + y1) / 2
        for i in range(n):
            cx = x0 + step * (i + 0.5)
            colour = _step_colour(i, stage)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                         fill=colour if i <= stage else None, outline=colour, width=2)
            if i and i % 6 == 0:
                draw.line((cx - step / 2, cy - r * 1.9, cx - step / 2, cy + r * 1.9),
                          fill=ui.PANEL_EDGE, width=1)
        ui.text(draw, (x1, y0 - 4),
                MILESTONES[min(stage, n - 1)].replace("_", " "),
                size=int((y1 - y0) * 0.42), kind="mono", colour=ui.WARN, anchor="rb")
        return
    step = (y1 - y0) / n
    for i, name in enumerate(MILESTONES):
        y = y0 + i * step
        colour = _step_colour(i, stage)
        r = step * 0.22
        cy = y + step / 2
        draw.ellipse((x0, cy - r, x0 + 2 * r, cy + r),
                     fill=colour if i <= stage else None, outline=colour, width=2)
        ui.text(draw, (x0 + 2 * r + 10, cy), name.replace("_", " "),
                size=int(step * 0.62), kind="reg", colour=colour, anchor="lm")


def draw_brain(draw, pil, box, rates: dict, kc_active: int, mb_gain: float,
               kc_raster: np.ndarray | None = None, title: str = "MaleCNS v1.0"):
    x0, y0, x1, y1 = box
    ui.panel(draw, box)
    ui.text(draw, (x0 + 16, y0 + 12), title, size=17, colour=ui.DIM)
    ui.text(draw, (x0 + 16, y0 + 34), "connectome activity", size=22)

    top = y0 + 70
    row = 30
    for i, (name, value) in enumerate(rates.items()):
        y = top + i * row
        ui.text(draw, (x0 + 16, y + 9), name, size=16, kind="mono",
                colour=ui.POP_COLOURS[name], anchor="lm")
        ui.bar(draw, (x0 + 74, y, x1 - 90, y + 16),
               min(1.0, value / 1.2), colour=ui.POP_COLOURS[name])
        ui.text(draw, (x1 - 16, y + 9), f"{value:5.3f}", size=15, kind="mono",
                colour=ui.DIM, anchor="rm")

    y = top + len(rates) * row + 10
    ui.text(draw, (x0 + 16, y), f"Kenyon cells active   {kc_active:4d}", size=15,
            kind="mono", colour=ui.DIM)
    ui.text(draw, (x0 + 16, y + 20), f"KC->MBON efficacy   {mb_gain:5.3f}", size=15,
            kind="mono", colour=ui.ACCENT)

    if kc_raster is not None and kc_raster.size:
        ry = y + 46
        rh = max(10, y1 - ry - 14)
        rw = x1 - x0 - 32
        strip = Image.fromarray(kc_raster).resize((rw, rh), Image.NEAREST)
        pil.paste(strip, (x0 + 16, ry))


def draw_hud(draw, box, env: MineSim, agent: int, label: str, subtitle: str = ""):
    x0, y0, x1, y1 = box
    ui.panel(draw, box)
    ui.text(draw, (x0 + 18, y0 + 12), label, size=17, colour=ui.DIM)
    ui.text(draw, (x0 + 18, y0 + 34), ui.format_time(int(env.tick[agent])),
            size=44, kind="mono", colour=ui.INK)
    if subtitle:
        ui.text(draw, (x0 + 18, y0 + 88), subtitle, size=17, kind="reg", colour=ui.DIM)

    inv_y = y1 - 34
    items = ("log", "cobble", "iron", "diamond", "obsidian", "blaze_rod", "pearl", "eye")
    cols = {"log": (150, 110, 60), "cobble": (140, 140, 140), "iron": (206, 166, 120),
            "diamond": (94, 226, 224), "obsidian": (120, 96, 180),
            "blaze_rod": (240, 176, 64), "pearl": (96, 200, 168), "eye": (168, 240, 120)}
    x = x0 + 18
    for it in items:
        c = int(env.inv[agent, ITEM_IX[it]])
        if c:
            draw.rectangle((x, inv_y, x + 16, inv_y + 16), fill=cols[it])
            ui.text(draw, (x + 20, inv_y + 8), str(c), size=15, kind="mono",
                    colour=ui.INK, anchor="lm")
            x += 46
    tier_names = ("bare hands", "wood", "stone", "iron", "diamond")
    ui.text(draw, (x1 - 18, inv_y + 8), tier_names[int(env.tier[agent])],
            size=15, kind="mono", colour=ui.WARN, anchor="rm")
