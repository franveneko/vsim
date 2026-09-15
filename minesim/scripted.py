"""A hand-written speedrunner, used as the reference route and as a baseline.

This policy knows the route by construction.  It is not learning anything - it
exists to prove the world is completable, and to give the evolved flies a human
benchmark to be measured against.
"""
from __future__ import annotations

import numpy as np

from .world import (DX, DY, HAZARD, ITEM_IX, LOG, MILESTONES, SOLID, TIER_REQ,
                    MineSim)

M = MILESTONES.index
CRAFT_STAGES = np.array([M("craft_planks"), M("wooden_pickaxe"), M("stone_pickaxe"),
                         M("smelt_iron"), M("iron_pickaxe"), M("diamond_pickaxe"),
                         M("craft_eyes")])
PLACE_STAGES = np.array([M("place_bench"), M("light_portal"), M("enter_end")])
COMBAT_STAGES = np.array([M("blaze_rods"), M("ender_pearls"), M("slay_dragon")])


def _can_craft(e: MineSim, st: np.ndarray) -> np.ndarray:
    inv, I = e.inv, ITEM_IX
    log, planks, stick = inv[:, I["log"]], inv[:, I["planks"]], inv[:, I["stick"]]
    wood = (log >= 1) | (planks >= 2) | (stick >= 2)
    cond = np.zeros(e.n, bool)
    cond |= (st == M("craft_planks")) & (log >= 1)
    cond |= (st == M("wooden_pickaxe")) & ((planks >= 3) & (stick >= 2) | (log >= 1) | (planks >= 5))
    cond |= (st == M("stone_pickaxe")) & (inv[:, I["cobble"]] >= 3) & wood
    cond |= (st == M("smelt_iron")) & (inv[:, I["iron_ore"]] >= 1)
    cond |= (st == M("iron_pickaxe")) & (
        ((inv[:, I["iron"]] >= 3) & ((stick >= 2) | wood)) | (inv[:, I["iron_ore"]] >= 1))
    cond |= (st == M("diamond_pickaxe")) & (inv[:, I["diamond"]] >= 3) & ((stick >= 2) | wood)
    cond |= (st == M("craft_eyes")) & (inv[:, I["blaze_rod"]] >= 2) & (inv[:, I["pearl"]] >= 2)
    return cond


def _can_place(e: MineSim, st: np.ndarray) -> np.ndarray:
    inv, I = e.inv, ITEM_IX
    return (((st == M("place_bench")) & (inv[:, I["planks"]] >= 4))
            | ((st == M("light_portal")) & (inv[:, I["obsidian"]] >= 10))
            | ((st == M("enter_end")) & (inv[:, I["eye"]] >= 2)))


# A speedrunner grabs spare wood on the way down: every pickaxe costs sticks, and
# there are no trees at diamond level.
WOOD_STOCK = 6


def act(e: MineSim) -> np.ndarray:
    rows = np.arange(e.n)
    st = e.stage()
    gx, gy, gd = e._goal_position()
    # detour for wood when a tool is due and there is none left
    inv, I = e.inv, ITEM_IX
    wood_left = inv[:, I["log"]] * 4 + inv[:, I["planks"]] * 2 + inv[:, I["stick"]]
    need_wood = np.isin(st, CRAFT_STAGES) & (wood_left < 2)
    if need_wood.any():
        wgx, wgy, wgd = e._nearest_block(LOG)
        gx = np.where(need_wood, wgx, gx)
        gy = np.where(need_wood, wgy, gy)
        gd = np.where(need_wood, wgd, gd)
    dx, dy = gx - e.x, gy - e.y

    # preferred step: close the larger axis first, avoiding lava
    prefer_x = np.abs(dx) >= np.abs(dy)
    dir_x = np.where(dx > 0, 2, 3)
    dir_y = np.where(dy > 0, 1, 0)
    first = np.where(prefer_x & (dx != 0), dir_x, dir_y)
    second = np.where(prefer_x & (dx != 0), dir_y, dir_x)

    def cell(d):
        nx = np.clip(e.x + DX[d], 0, e.SIZE - 1)
        ny = np.clip(e.y + DY[d], 0, e.SIZE - 1)
        return e.grid[e.dim, rows, ny, nx]

    b1, b2 = cell(first), cell(second)
    step = np.where(HAZARD[b1] & ~HAZARD[b2], second, first)
    blocking = cell(step)

    a = step.copy()
    # if the chosen step is into a block we may mine, face it then mine it
    minable = SOLID[blocking] & (e.tier >= TIER_REQ[blocking])
    facing = e.face == step
    a = np.where(minable & facing, 4, a)

    # arrived at the objective: mine it if it is a block, walk onto it if it is
    # a portal
    at_goal = gd <= 1
    goal_dir = np.where(np.abs(dx) >= np.abs(dy), dir_x, dir_y)
    goal_block = e.grid[e.dim, rows, np.clip(gy, 0, e.SIZE - 1), np.clip(gx, 0, e.SIZE - 1)]
    solid_goal = SOLID[goal_block]
    a = np.where(at_goal & solid_goal & (e.face == goal_dir), 4, a)
    a = np.where(at_goal & solid_goal & (e.face != goal_dir), goal_dir, a)
    a = np.where(at_goal & ~solid_goal, goal_dir, a)

    # combat
    mdx, mdy, md, mkind, _ = e._nearest_mob()
    in_combat = np.isin(st, COMBAT_STAGES)
    a = np.where((md <= 1.5) & (in_combat | (md <= 1.0)), 7, a)

    a = np.where(_can_place(e, st), 5, a)
    a = np.where(_can_craft(e, st), 6, a)
    return a.astype(np.int64)


def run(n: int = 64, seed: int = 0, max_ticks: int = 4000) -> MineSim:
    e = MineSim(n, seed=seed, max_ticks=max_ticks)
    for _ in range(max_ticks):
        _, done = e.step(act(e))
        if done.all():
            break
    return e


if __name__ == "__main__":
    import collections
    e = run()
    print(f"won {int(e.won.sum())}/{e.n}   alive {int(e.alive.sum())}")
    print("milestones:", collections.Counter(e.progress().tolist()))
    if e.won.any():
        print("fastest completion:", int(e.ms_tick[e.won, -1].min()), "ticks")
    else:
        best = int(e.progress().max())
        print("furthest stage:", MILESTONES[min(best, len(MILESTONES) - 1)])
