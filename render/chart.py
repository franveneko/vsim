"""Charts drawn straight into video frames.

Two series of one measure on one axis, thin marks, recessive grid, direct labels
at the line ends plus a compact legend so identity is never carried by colour
alone.  The palette is the validated dark-surface categorical pair.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from minesim.world import MILESTONES, N_MILESTONES
from render import ui

SERIES_1 = (0x39, 0x87, 0xE5)     # blue  - best fly
SERIES_2 = (0xD9, 0x59, 0x26)     # orange - population mean
GRID = (40, 46, 58)
AXIS = (86, 96, 112)

# route landmarks worth a reference line
LANDMARKS = (
    (MILESTONES.index("wooden_pickaxe"), "wooden pickaxe"),
    (MILESTONES.index("iron_pickaxe"), "iron pickaxe"),
    (MILESTONES.index("enter_nether"), "the nether"),
    (N_MILESTONES, "dragon down"),
)


def learning_curve(history: list[dict], upto: int, size: tuple[int, int],
                   title: str = "Milestones reached, generation by generation",
                   total_gens: int | None = None) -> np.ndarray:
    W, H = size
    img = ui.canvas(W, H)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)

    n_total = total_gens or max(len(history), 1)
    left, right = 96, W - 210
    top, bottom = 130, H - 90

    ui.text(d, (56, 48), title, size=30)
    ui.text(d, (56, 88), "one point per generation  ·  19 milestones from bare hands "
                         "to the Ender Dragon", size=17, kind="reg", colour=ui.DIM)

    def px(g):
        return left + (right - left) * (g / max(n_total - 1, 1))

    def py(v):
        return bottom - (bottom - top) * (v / N_MILESTONES)

    # recessive grid + landmark reference lines
    for v in range(0, N_MILESTONES + 1, 5):
        y = py(v)
        d.line((left, y, right, y), fill=GRID, width=1)
        ui.text(d, (left - 14, y), str(v), size=16, kind="mono", colour=AXIS, anchor="rm")
    for v, name in LANDMARKS:
        y = py(v)
        for x in range(left, right, 12):
            d.line((x, y, x + 6, y), fill=(58, 66, 82), width=1)
        ui.text(d, (right + 12, y), name, size=15, kind="reg", colour=AXIS, anchor="lm")
    d.line((left, top, left, bottom), fill=AXIS, width=2)
    d.line((left, bottom, right, bottom), fill=AXIS, width=2)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        g = int(round(frac * (n_total - 1)))
        x = px(g)
        d.line((x, bottom, x, bottom + 7), fill=AXIS, width=2)
        ui.text(d, (x, bottom + 14), str(g), size=16, kind="mono", colour=AXIS,
                anchor="ma")
    ui.text(d, ((left + right) // 2, bottom + 46), "generation", size=17,
            kind="reg", colour=AXIS, anchor="ma")

    n = max(1, min(upto, len(history)))
    gens = [h["gen"] for h in history[:n]]
    best = [h["best_progress"] for h in history[:n]]
    mean = [h["mean_progress"] for h in history[:n]]

    for series, colour in ((mean, SERIES_2), (best, SERIES_1)):
        pts = [(px(g), py(v)) for g, v in zip(gens, series)]
        if len(pts) > 1:
            d.line(pts, fill=colour, width=2, joint="curve")
        if pts:
            x, y = pts[-1]
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colour)

    # direct labels at the line ends, nudged apart when they collide
    if gens:
        ends = [(py(best[-1]), "best fly", SERIES_1), (py(mean[-1]), "population mean", SERIES_2)]
        ends.sort()
        if ends[1][0] - ends[0][0] < 24:
            ends = [(ends[0][0] - 12, *ends[0][1:]), (ends[1][0] + 12, *ends[1][1:])]
        for y, label, colour in ends:
            ui.text(d, (px(gens[-1]) + 12, y), label, size=17, kind="reg",
                    colour=colour, anchor="lm")

    # compact legend
    lx, ly = 56, H - 40
    for label, colour in (("best fly", SERIES_1), ("population mean", SERIES_2)):
        d.rectangle((lx, ly - 6, lx + 22, ly + 2), fill=colour)
        ui.text(d, (lx + 32, ly - 2), label, size=16, kind="reg", colour=ui.DIM, anchor="lm")
        lx += 40 + 9 * len(label)
    return np.asarray(pil).copy()


def connectome_portrait(conn, size: tuple[int, int], block: int = 36) -> np.ndarray:
    """The wiring diagram itself: |W| coarse-grained into a heatmap.

    One hue, light to dark, with the population blocks labelled - the visible
    structure (the mushroom-body band, the central-complex block) is real.
    """
    from flybrain.connectome import POPULATIONS
    W, H = size
    n = conn.n
    k = int(np.ceil(n / block))
    dense = np.zeros((k, k), np.float32)
    coo = abs(conn.W).tocoo()
    np.add.at(dense, (coo.row // block, coo.col // block), coo.data)
    v = np.log1p(dense)
    v /= v.max() + 1e-9

    base = np.array(ui.BG, np.float32)
    hot = np.array((110, 200, 255), np.float32)
    heat = (base + (hot - base) * v[..., None] ** 0.55).astype(np.uint8)

    side = min(W, H) - 260
    img = ui.canvas(W, H)
    grid = ui.fit_nearest(heat, side, side)
    x0 = (W - side) // 2
    y0 = 150
    ui.paste(img, grid, y0, x0)

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    ui.text(d, (x0, 52), "The wiring diagram", size=34)
    ui.text(d, (x0, 96), f"{n:,} neurons  ·  {conn.W.nnz:,} measured connections  ·  "
                         "rows are targets, columns are sources", size=17,
            kind="reg", colour=ui.DIM)
    d.rectangle((x0, y0, x0 + side, y0 + side), outline=ui.PANEL_EDGE, width=2)

    # population boundaries, labelled
    for i, name in enumerate(POPULATIONS[:-1]):
        idx = conn.idx(name)
        if not idx.size:
            continue
        a = x0 + int(side * idx[0] / n)
        b = x0 + int(side * (idx[-1] + 1) / n)
        ya = y0 + int(side * idx[0] / n)
        yb = y0 + int(side * (idx[-1] + 1) / n)
        d.line((a, y0, a, y0 + side), fill=(70, 80, 98), width=1)
        d.line((x0, ya, x0 + side, ya), fill=(70, 80, 98), width=1)
        if b - a > 26:
            ui.text(d, ((a + b) // 2, y0 + side + 14), name, size=16, kind="mono",
                    colour=ui.DIM, anchor="ma")
        if yb - ya > 26:
            ui.text(d, (x0 - 12, (ya + yb) // 2), name, size=16, kind="mono",
                    colour=ui.DIM, anchor="rm")
    return np.asarray(pil).copy()
