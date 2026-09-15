"""Procedural 16x16 block textures, assembled into a world image with numpy.

Sixteen-pixel tiles are the native resolution of Minecraft's own textures, and
building them procedurally keeps the repository free of any borrowed art while
still reading, at a glance, as the game the flies are playing.
"""
from __future__ import annotations

import numpy as np

from minesim.world import (AIR, BEDROCK, BENCH, DIAMOND_ORE, END_PORTAL, END_STONE,
                           FURNACE, GRASS, IRON_ORE, LAVA, LEAVES, LOG, NETHERRACK,
                           OBSIDIAN, PORTAL, SAND, SPAWNER, STONE, WATER)

TILE = 16
N_BLOCKS = 19


def _noise(rng, shape, amount):
    return (rng.integers(-amount, amount + 1, shape)).astype(np.int16)


def _flat(rng, base, grain=10):
    t = np.zeros((TILE, TILE, 3), np.int16)
    t[:] = base
    t += _noise(rng, (TILE, TILE, 1), grain)
    return t


def build_atlas(seed: int = 7) -> np.ndarray:
    """(N_BLOCKS, TILE, TILE, 3) uint8 texture atlas."""
    rng = np.random.default_rng(seed)
    atlas = np.zeros((N_BLOCKS, TILE, TILE, 3), np.int16)

    atlas[AIR] = _flat(rng, (24, 26, 34), 3)
    atlas[GRASS] = _flat(rng, (98, 152, 74), 14)
    atlas[SAND] = _flat(rng, (214, 203, 154), 10)
    atlas[STONE] = _flat(rng, (126, 126, 126), 14)
    atlas[BEDROCK] = _flat(rng, (54, 54, 58), 22)
    atlas[NETHERRACK] = _flat(rng, (124, 44, 44), 18)
    atlas[END_STONE] = _flat(rng, (221, 223, 165), 10)
    atlas[OBSIDIAN] = _flat(rng, (30, 22, 46), 8)

    # wood: vertical grain with a ring
    log = _flat(rng, (108, 78, 46), 8)
    log[:, ::5] -= 18
    log[6:10, 6:10] = np.array([150, 116, 70]) + _noise(rng, (4, 4, 1), 6)
    atlas[LOG] = log

    leaves = _flat(rng, (58, 112, 48), 18)
    leaves[rng.random((TILE, TILE)) < 0.12] = (34, 78, 32)
    atlas[LEAVES] = leaves

    water = _flat(rng, (56, 92, 190), 8)
    water[::4] += 14
    atlas[WATER] = water

    lava = _flat(rng, (214, 96, 20), 16)
    lava[rng.random((TILE, TILE)) < 0.18] = (252, 206, 64)
    atlas[LAVA] = lava

    iron = atlas[STONE].copy()
    for _ in range(7):
        r, c = rng.integers(1, TILE - 2, 2)
        iron[r:r + 2, c:c + 2] = np.array([206, 166, 120])
    atlas[IRON_ORE] = iron

    dia = atlas[STONE].copy()
    for _ in range(6):
        r, c = rng.integers(1, TILE - 2, 2)
        dia[r:r + 2, c:c + 2] = np.array([94, 226, 224])
    atlas[DIAMOND_ORE] = dia

    bench = _flat(rng, (142, 104, 62), 8)
    bench[:4] = (168, 126, 74)
    bench[7:9, :] = (96, 70, 42)
    bench[:, 7:9] = (96, 70, 42)
    atlas[BENCH] = bench

    furnace = _flat(rng, (110, 110, 110), 10)
    furnace[5:12, 4:12] = (44, 44, 46)
    furnace[8:11, 6:10] = (232, 150, 44)
    atlas[FURNACE] = furnace

    portal = _flat(rng, (126, 52, 206), 20)
    portal[rng.random((TILE, TILE)) < 0.25] = (188, 122, 246)
    portal[0] = portal[-1] = (30, 22, 46)
    atlas[PORTAL] = portal

    endp = _flat(rng, (18, 18, 28), 6)
    endp[rng.random((TILE, TILE)) < 0.3] = (128, 232, 196)
    atlas[END_PORTAL] = endp

    spawner = _flat(rng, (62, 62, 68), 10)
    spawner[::3, :] = (36, 36, 40)
    spawner[:, ::3] = (36, 36, 40)
    spawner[6:10, 6:10] = (240, 160, 40)
    atlas[SPAWNER] = spawner

    # bevel every solid tile so the world reads as blocks rather than a bitmap
    for b in range(N_BLOCKS):
        if b == AIR:
            continue
        atlas[b, 0, :] += 26
        atlas[b, :, 0] += 18
        atlas[b, -1, :] -= 26
        atlas[b, :, -1] -= 18
    return np.clip(atlas, 0, 255).astype(np.uint8)


ATLAS = build_atlas()


def render_world(grid: np.ndarray) -> np.ndarray:
    """(S, S) block ids -> (S*TILE, S*TILE, 3) uint8 image."""
    tiles = ATLAS[grid]                              # (S, S, T, T, 3)
    s = grid.shape[0]
    return tiles.transpose(0, 2, 1, 3, 4).reshape(s * TILE, s * TILE, 3)


def _sprite(pixels: dict[tuple[int, int], tuple[int, int, int]], size: int = 16):
    """Build a small RGBA sprite from a {(row, col): colour} map."""
    spr = np.zeros((size, size, 4), np.uint8)
    for (r, c), col in pixels.items():
        spr[r, c, :3] = col
        spr[r, c, 3] = 255
    return spr


def _blob(cx, cy, rx, ry, colour, size=16):
    out = {}
    for r in range(size):
        for c in range(size):
            if ((c - cx) / rx) ** 2 + ((r - cy) / ry) ** 2 <= 1.0:
                out[(r, c)] = colour
    return out


def make_fly_sprite() -> np.ndarray:
    """A Drosophila, from above: red eyes, grey thorax, striped abdomen, wings."""
    px = {}
    px.update(_blob(7.5, 11.5, 2.6, 3.6, (188, 188, 196)))     # wings (spread)
    for (r, c) in list(px):
        px[(r, c)] = (196, 200, 210)
    px.update(_blob(4.0, 10.5, 3.2, 2.2, (150, 154, 166)))
    px.update(_blob(11.0, 10.5, 3.2, 2.2, (150, 154, 166)))
    px.update(_blob(7.5, 8.0, 2.4, 3.0, (104, 82, 46)))        # abdomen
    for r in (7, 9, 11):
        for c in range(5, 11):
            if (r, c) in px:
                px[(r, c)] = (54, 42, 26)
    px.update(_blob(7.5, 5.0, 2.6, 2.2, (72, 64, 54)))         # thorax
    px.update(_blob(7.5, 2.5, 2.8, 2.0, (36, 32, 30)))         # head
    px[(6, 2)] = px[(6, 1)] = (214, 46, 40)                    # eyes
    px[(9, 2)] = px[(9, 1)] = (214, 46, 40)
    return _sprite(px)


def make_mob_sprite(kind: int) -> np.ndarray:
    palettes = {
        0: ((58, 122, 74), (44, 90, 58), (28, 52, 36)),       # zombie
        1: ((76, 176, 84), (58, 140, 66), (18, 40, 22)),      # creeper
        2: ((236, 176, 44), (214, 130, 26), (90, 50, 12)),    # blaze
        3: ((58, 34, 78), (96, 54, 128), (232, 78, 168)),     # ender dragon
    }
    body, shade, dark = palettes[kind]
    px = {}
    if kind == 3:
        px.update(_blob(7.5, 7.5, 6.0, 4.0, body))
        px.update(_blob(7.5, 7.5, 2.4, 6.0, shade))
        px[(6, 4)] = px[(9, 4)] = dark
    elif kind == 2:
        px.update(_blob(7.5, 7.5, 4.0, 4.0, body))
        px.update(_blob(7.5, 7.5, 2.0, 2.0, shade))
        px[(6, 6)] = px[(9, 6)] = dark
    else:
        for r in range(3, 13):
            for c in range(4, 12):
                px[(r, c)] = body if r < 8 else shade
        px[(5, 6)] = px[(5, 9)] = dark
        px[(7, 7)] = px[(7, 8)] = dark
    return _sprite(px)


FLY = make_fly_sprite()
MOBS = [make_mob_sprite(k) for k in range(4)]


def blit(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    """Alpha-composite a sprite into an RGB image, clipped at the edges."""
    h, w = sprite.shape[:2]
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(dst.shape[0], top + h), min(dst.shape[1], left + w)
    if y0 >= y1 or x0 >= x1:
        return
    sub = sprite[y0 - top:y1 - top, x0 - left:x1 - left]
    alpha = (sub[..., 3:4] / 255.0)
    region = dst[y0:y1, x0:x1]
    dst[y0:y1, x0:x1] = (region * (1 - alpha) + sub[..., :3] * alpha).astype(np.uint8)
