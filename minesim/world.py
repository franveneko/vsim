"""A vectorised Minecraft-shaped sandbox.

This is not Minecraft.  Minecraft is a proprietary Java game that needs an
account, a GPU and a desktop session; none of that exists in this container, and
no headless Minecraft bridge (MineRL / MineDojo) would run hundreds of instances
on four CPU cores anyway.  So MineSim reimplements the parts of Minecraft that
make it a *learning problem*:

  * a blocky world you can mine and build in,
  * a tool tech-tree where each tier gates the next material,
  * hostile mobs that kill you,
  * three dimensions reached through portals,
  * and the actual Any%-glitchless win condition: kill the Ender Dragon.

Everything is stored as arrays with a leading agent axis, so `n_agents` worlds
advance on every call to `step()` - which is what makes it possible to watch a
few hundred flies attempt the run simultaneously.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- block types
AIR, GRASS, STONE, LOG, LEAVES, WATER, LAVA = 0, 1, 2, 3, 4, 5, 6
IRON_ORE, DIAMOND_ORE, OBSIDIAN, BENCH, FURNACE = 7, 8, 9, 10, 11
PORTAL, NETHERRACK, SPAWNER, END_PORTAL, END_STONE, BEDROCK, SAND = 12, 13, 14, 15, 16, 17, 18

BLOCKS = {
    AIR: "air", GRASS: "grass", STONE: "stone", LOG: "log", LEAVES: "leaves",
    WATER: "water", LAVA: "lava", IRON_ORE: "iron_ore", DIAMOND_ORE: "diamond_ore",
    OBSIDIAN: "obsidian", BENCH: "crafting_table", FURNACE: "furnace",
    PORTAL: "nether_portal", NETHERRACK: "netherrack", SPAWNER: "blaze_spawner",
    END_PORTAL: "end_portal", END_STONE: "end_stone", BEDROCK: "bedrock", SAND: "sand",
}
# The world is a top-down slice: grass, sand and end stone are the floor you
# walk on, everything else is a block that stands in your way until you mine it.
SOLID = np.zeros(19, bool)
SOLID[[STONE, LOG, LEAVES, IRON_ORE, DIAMOND_ORE, OBSIDIAN, BENCH,
       FURNACE, NETHERRACK, SPAWNER, BEDROCK]] = True
HAZARD = np.zeros(19, bool)
HAZARD[[LAVA]] = True
WALKABLE = np.zeros(19, bool)
WALKABLE[[AIR, GRASS, SAND, WATER, PORTAL, END_PORTAL, END_STONE]] = True

# hardness in ticks, and the pickaxe tier required (0 = bare hands)
HARDNESS = np.ones(19, np.int16)
HARDNESS[[LOG, LEAVES]] = 3
HARDNESS[[STONE, NETHERRACK, END_STONE]] = 4
HARDNESS[[IRON_ORE]] = 6
HARDNESS[[DIAMOND_ORE]] = 8
HARDNESS[[OBSIDIAN]] = 14
HARDNESS[BEDROCK] = 30000
TIER_REQ = np.zeros(19, np.int8)
TIER_REQ[[STONE, NETHERRACK, END_STONE]] = 1       # wooden pickaxe
TIER_REQ[[IRON_ORE]] = 2                           # stone pickaxe
TIER_REQ[[DIAMOND_ORE]] = 3                        # iron pickaxe
TIER_REQ[[OBSIDIAN]] = 4                           # diamond pickaxe
TIER_REQ[BEDROCK] = 99

# ------------------------------------------------------------------ inventory
ITEMS = ("log", "planks", "stick", "cobble", "iron_ore", "iron", "diamond",
         "obsidian", "blaze_rod", "pearl", "eye")
ITEM_IX = {name: i for i, name in enumerate(ITEMS)}
N_ITEMS = len(ITEMS)

OVERWORLD, NETHER, END = 0, 1, 2

# ------------------------------------------------------------- speedrun route
# The canonical Any% route, compressed to the steps that matter.
MILESTONES = (
    "punch_wood", "craft_planks", "place_bench", "wooden_pickaxe",
    "mine_stone", "stone_pickaxe", "mine_iron", "smelt_iron",
    "iron_pickaxe", "mine_diamond", "diamond_pickaxe", "mine_obsidian",
    "light_portal", "enter_nether", "blaze_rods", "ender_pearls",
    "craft_eyes", "enter_end", "slay_dragon",
)
N_MILESTONES = len(MILESTONES)

ACTION_NAMES = ("north", "south", "east", "west", "mine", "place", "craft", "attack")
DX = np.array([0, 0, 1, -1], np.int16)
DY = np.array([-1, 1, 0, 0], np.int16)

# Starting a run part-way along the route: what a player would already be
# carrying at that point.  Used for checkpoint practice during training - final
# results are always measured from a standing start.
CHECKPOINTS: dict[int, dict] = {}


def _checkpoint_table():
    M = MILESTONES.index
    tiers = {M("wooden_pickaxe") + 1: 1, M("stone_pickaxe") + 1: 2,
             M("iron_pickaxe") + 1: 3, M("diamond_pickaxe") + 1: 4}
    table = {}
    tier = 0
    for st in range(len(MILESTONES)):
        tier = tiers.get(st, tier)
        items: dict[str, int] = {"log": 4} if st >= M("wooden_pickaxe") else {}
        if st >= M("ender_pearls"):
            items["blaze_rod"] = 2
        if st >= M("craft_eyes"):
            items["pearl"] = 2
        if st == M("enter_end"):
            items["eye"] = 2
        if M("blaze_rods") <= st <= M("craft_eyes"):
            dim = 1
        elif st >= M("slay_dragon"):
            dim = 2
        else:
            dim = 0
        table[st] = {"tier": tier, "dim": dim, "items": items}
    return table


ZOMBIE, CREEPER, BLAZE, DRAGON = 0, 1, 2, 3
MOB_COOLDOWN = 12            # ticks between a mob's attacks
MAX_HP = 20.0                # ten hearts, as in the game
COMBAT_STAGES = np.array([MILESTONES.index("blaze_rods"),
                          MILESTONES.index("ender_pearls"),
                          MILESTONES.index("slay_dragon")])
# which mob each combat milestone is actually about
COMBAT_TARGET = {MILESTONES.index("blaze_rods"): 4,
                 MILESTONES.index("ender_pearls"): 3,
                 MILESTONES.index("slay_dragon"): 5}
N_MOBS = 6


class MineSim:
    """`n` independent worlds advanced in lockstep."""

    SIZE = 32
    OBS_DIM = 32

    def __init__(self, n_agents: int, seed: int = 0, max_ticks: int = 1500,
                 shared_world: bool = True):
        """`shared_world` gives every agent the same terrain, spawn and mobs.

        That matters for evolution: comparing genomes that each played a
        different map measures luck, not policy.
        """
        self.n = n_agents
        self.max_ticks = max_ticks
        self.seed = seed
        self.shared_world = shared_world
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((3, n_agents, self.SIZE, self.SIZE), np.uint8)
        self.reset()

    # ------------------------------------------------------------- generation
    _spawn_xy = (np.zeros(1, int), np.zeros(1, int))

    def _generate(self, who: np.ndarray) -> None:
        if not CHECKPOINTS:
            CHECKPOINTS.update(_checkpoint_table())
        S, rng = self.SIZE, self.rng
        k = 1 if self.shared_world else who.size

        # --- overworld: grass plain, trees, a stone belt hiding ores ---------
        ow = np.full((k, S, S), GRASS, np.uint8)
        noise = rng.random((k, S, S))
        ow[noise < 0.06] = SAND
        ow[(noise > 0.94)] = WATER
        yy = np.arange(S)[None, :, None]
        stone_belt = (yy >= S // 2 + 2)
        ow = np.where(stone_belt & (rng.random((k, S, S)) < 0.70), STONE, ow)
        # trees in the top half
        tree = (rng.random((k, S, S)) < 0.05) & (yy < S // 2)
        ow[tree] = LOG
        ow[(rng.random((k, S, S)) < 0.04) & (yy < S // 2) & (ow == GRASS)] = LEAVES
        # ores inside the stone belt
        deep = (yy >= S // 2 + 4)
        ow = np.where(deep & (rng.random((k, S, S)) < 0.045), IRON_ORE, ow)
        deeper = (yy >= S - 8)
        ow = np.where(deeper & (rng.random((k, S, S)) < 0.055), DIAMOND_ORE, ow)
        ow = np.where(deeper & (rng.random((k, S, S)) < 0.03), LAVA, ow)
        # a lava lake, with the obsidian shelf the portal is built from
        roll = rng.random((k, 4, S))
        ow[:, -5:-1, :] = np.where(roll < 0.35, LAVA,
                                   np.where(roll < 0.80, OBSIDIAN, STONE))
        ow[:, -1, :] = BEDROCK
        ow[:, 0, :] = BEDROCK
        ow[:, :, 0] = BEDROCK
        ow[:, :, -1] = BEDROCK

        # --- nether: netherrack, lava, a blaze spawner ----------------------
        roll = rng.random((k, S, S))
        nz = np.where(roll < 0.12, LAVA,
                      np.where(roll < 0.40, NETHERRACK, AIR)).astype(np.uint8)
        nz[:, 1:4, 1:4] = AIR
        nz[:, 1, 1] = PORTAL                       # return portal
        # a cleared fortress chamber, so the blaze has somewhere to stand and the
        # player has somewhere to fight it
        sx = rng.integers(S - 9, S - 4, k)
        sy = rng.integers(S - 9, S - 4, k)
        for j in range(k):
            nz[j, sy[j] - 3:sy[j] + 4, sx[j] - 3:sx[j] + 4] = AIR
            nz[j, sy[j], sx[j]] = SPAWNER
        self._spawn_xy = (sx, sy)
        nz[:, 0, :] = nz[:, -1, :] = nz[:, :, 0] = nz[:, :, -1] = BEDROCK

        # --- the end: end stone island, dragon ------------------------------
        ez = np.full((k, S, S), END_STONE, np.uint8)
        ez[:, 0, :] = ez[:, -1, :] = ez[:, :, 0] = ez[:, :, -1] = BEDROCK

        self.grid[OVERWORLD, who] = ow
        self.grid[NETHER, who] = nz
        self.grid[END, who] = ez

        self.x[who] = np.broadcast_to(rng.integers(2, S - 2, k), (who.size,)).astype(np.int16)
        self.y[who] = np.broadcast_to(rng.integers(2, S // 2 - 2, k), (who.size,)).astype(np.int16)
        self.grid[OVERWORLD, who, self.y[who], self.x[who]] = AIR

        # --- mobs -----------------------------------------------------------
        self.mob_x[who] = np.broadcast_to(rng.integers(2, S - 2, (k, N_MOBS)),
                                          (who.size, N_MOBS)).astype(np.int16)
        self.mob_y[who] = np.broadcast_to(rng.integers(2, S - 2, (k, N_MOBS)),
                                          (who.size, N_MOBS)).astype(np.int16)
        # the nether pair lives in the fortress chamber
        sx, sy = self._spawn_xy
        for mob, (dx_, dy_) in ((3, (-2, 0)), (4, (2, 0))):
            self.mob_x[who, mob] = np.broadcast_to(sx + dx_, (who.size,)).astype(np.int16)
            self.mob_y[who, mob] = np.broadcast_to(sy + dy_, (who.size,)).astype(np.int16)
        self.mob_kind[who] = ZOMBIE
        self.mob_kind[who[:, None], np.array([2])[None, :]] = CREEPER
        self.mob_kind[who[:, None], np.array([4])[None, :]] = BLAZE
        self.mob_kind[who[:, None], np.array([5])[None, :]] = DRAGON
        self.mob_hp[who] = 3
        self.mob_hp[who[:, None], np.array([4])[None, :]] = 4      # blaze
        self.mob_hp[who[:, None], np.array([5])[None, :]] = 12     # dragon
        self.mob_dim[who] = OVERWORLD
        # mob 3 is the pearl source and lives in the nether alongside the blaze
        self.mob_dim[who[:, None], np.array([3, 4])[None, :]] = NETHER
        self.mob_dim[who[:, None], np.array([5])[None, :]] = END

    # ------------------------------------------------------------------ reset
    def reset(self, who: np.ndarray | None = None, seed: int | None = None,
              start_stage: int = 0) -> None:
        """Restart worlds.  Passing `seed` regenerates terrain from a fresh draw,
        which is how the population is stopped from memorising one map."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        n = self.n
        if who is None:
            who = np.arange(n)
            self.x = np.zeros(n, np.int16); self.y = np.zeros(n, np.int16)
            self.face = np.zeros(n, np.int8)
            self.dim = np.zeros(n, np.int8)
            self.hp = np.full(n, MAX_HP, np.float32)
            self.inv = np.zeros((n, N_ITEMS), np.int32)
            self.tier = np.zeros(n, np.int8)
            self.done_ms = np.zeros((n, N_MILESTONES), bool)
            self.ms_tick = np.full((n, N_MILESTONES), -1, np.int32)
            self.tick = np.zeros(n, np.int32)
            self.alive = np.ones(n, bool)
            self.won = np.zeros(n, bool)
            self.mine_target = np.full(n, -1, np.int32)
            self.mine_prog = np.zeros(n, np.int16)
            self.mob_x = np.zeros((n, N_MOBS), np.int16)
            self.mob_y = np.zeros((n, N_MOBS), np.int16)
            self.mob_kind = np.zeros((n, N_MOBS), np.int8)
            self.mob_hp = np.zeros((n, N_MOBS), np.int16)
            self.mob_dim = np.zeros((n, N_MOBS), np.int8)
            self.prev_mob_dist = np.full(n, 99.0, np.float32)
        else:
            self.face[who] = 0; self.dim[who] = 0; self.hp[who] = 10.0
            self.inv[who] = 0; self.tier[who] = 0
            self.done_ms[who] = False; self.ms_tick[who] = -1
            self.tick[who] = 0; self.alive[who] = True; self.won[who] = False
            self.mine_target[who] = -1; self.mine_prog[who] = 0
            self.prev_mob_dist[who] = 99.0
        self._generate(np.asarray(who))
        if start_stage:
            self._apply_checkpoint(np.asarray(who), start_stage)
        self.prev_goal_dist = self._goal_distance()

    def _apply_checkpoint(self, who: np.ndarray, stage: int) -> None:
        cp = CHECKPOINTS[min(stage, N_MILESTONES - 1)]
        self.done_ms[who, :stage] = True
        self.ms_tick[who, :stage] = 0
        self.tier[who] = cp["tier"]
        self.dim[who] = cp["dim"]
        for item, count in cp["items"].items():
            self.inv[who, ITEM_IX[item]] = count
        # a run that has already lit the portal starts with one standing
        if stage > MILESTONES.index("light_portal"):
            self.grid[OVERWORLD, who, self.SIZE // 2, self.SIZE // 2] = PORTAL
        if cp["dim"] != OVERWORLD:
            self.x[who], self.y[who] = 3, 3
            self.grid[cp["dim"], who, 3, 3] = AIR

    # --------------------------------------------------------------- helpers
    def _at(self, x, y, dim=None):
        dim = self.dim if dim is None else dim
        return self.grid[dim, np.arange(self.n), y, x]

    def _set(self, x, y, value, mask):
        idx = np.flatnonzero(mask)
        if idx.size:
            self.grid[self.dim[idx], idx, y[idx], x[idx]] = value if np.isscalar(value) else value[idx]

    def stage(self) -> np.ndarray:
        """Index of the next milestone each agent is working on."""
        return self.done_ms.argmin(axis=1) * (~self.done_ms.all(axis=1))

    def _goal_block(self) -> np.ndarray:
        """Which block type each agent currently needs to reach."""
        st = self.stage()
        goal = np.full(self.n, AIR, np.uint8)
        table = {
            0: LOG, 1: LOG, 2: LOG, 3: LOG, 4: STONE, 5: STONE, 6: IRON_ORE,
            7: IRON_ORE, 8: IRON_ORE, 9: DIAMOND_ORE, 10: DIAMOND_ORE,
            11: OBSIDIAN, 12: OBSIDIAN, 13: PORTAL, 14: SPAWNER, 15: SPAWNER,
            16: SPAWNER, 17: END_PORTAL, 18: END_STONE,
        }
        for s, b in table.items():
            goal[st == s] = b
        return goal

    def _goal_distance(self) -> np.ndarray:
        gx, gy, d = self._goal_position()
        return d


    def _goal_position(self):
        """Where the agent should be heading right now: the nearest instance of
        the block its current milestone needs, or the dragon during the fight."""
        goal = self._goal_block()
        S, n = self.SIZE, self.n
        rows = np.arange(n)
        g = self.grid[self.dim, rows]                             # (n, S, S)
        match = g == goal[:, None, None]
        ys, xs = np.mgrid[0:S, 0:S]
        dist = (np.abs(xs[None] - self.x[:, None, None])
                + np.abs(ys[None] - self.y[:, None, None]))
        masked = np.where(match, dist, 9999).reshape(n, -1)
        flat = masked.argmin(axis=1)
        d = masked[rows, flat].astype(np.float32)
        gy, gx = np.divmod(flat, S)
        gx, gy = gx.astype(np.int16), gy.astype(np.int16)

        # combat stages chase the specific mob that drops what the route needs
        st = self.stage()
        for stage_ix, mob in COMBAT_TARGET.items():
            fight = (st == stage_ix) & (self.mob_hp[:, mob] > 0)
            if not fight.any():
                continue
            gx = np.where(fight, self.mob_x[:, mob], gx)
            gy = np.where(fight, self.mob_y[:, mob], gy)
            dd = (np.abs(self.mob_x[:, mob] - self.x)
                  + np.abs(self.mob_y[:, mob] - self.y)).astype(np.float32)
            d = np.where(fight, dd, d)

        unreachable = d > 5000
        d = np.where(unreachable, 40.0, d)
        gx = np.where(unreachable, self.x, gx)
        gy = np.where(unreachable, self.y, gy)
        return gx, gy, d

    def _nearest_block(self, block_id: int):
        """Nearest instance of one block type in the agent's current dimension."""
        S, n, rows = self.SIZE, self.n, np.arange(self.n)
        g = self.grid[self.dim, rows]
        ys, xs = np.mgrid[0:S, 0:S]
        dist = (np.abs(xs[None] - self.x[:, None, None])
                + np.abs(ys[None] - self.y[:, None, None]))
        masked = np.where(g == block_id, dist, 9999).reshape(n, -1)
        flat = masked.argmin(axis=1)
        d = masked[rows, flat].astype(np.float32)
        gy, gx = np.divmod(flat, S)
        miss = d > 5000
        return (np.where(miss, self.x, gx).astype(np.int16),
                np.where(miss, self.y, gy).astype(np.int16),
                np.where(miss, 40.0, d))

    def _nearest_mob(self):
        """Nearest living, same-dimension mob: offset, distance, kind."""
        rows = np.arange(self.n)
        live = (self.mob_hp > 0) & (self.mob_dim == self.dim[:, None])
        dx = (self.mob_x - self.x[:, None]).astype(np.float32)
        dy = (self.mob_y - self.y[:, None]).astype(np.float32)
        d = np.abs(dx) + np.abs(dy)
        d = np.where(live, d, 999.0)
        k = d.argmin(axis=1)
        return dx[rows, k], dy[rows, k], d[rows, k], self.mob_kind[rows, k], k

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        """(OBS_DIM, n) - what the fly's sensory neurons receive."""
        n, S = self.n, self.SIZE
        rows = np.arange(n)
        goal = self._goal_block()
        out = np.zeros((self.OBS_DIM, n), np.float32)

        # 4 neighbours x 3 features: solid, hazard, is-what-I-need
        for d in range(4):
            nx = np.clip(self.x + DX[d], 0, S - 1)
            ny = np.clip(self.y + DY[d], 0, S - 1)
            b = self.grid[self.dim, rows, ny, nx]
            out[d * 3 + 0] = SOLID[b]
            out[d * 3 + 1] = HAZARD[b]
            out[d * 3 + 2] = (b == goal)

        gx, gy, gd = self._goal_position()
        out[12] = np.clip((gx - self.x) / 8.0, -1, 1)
        out[13] = np.clip((gy - self.y) / 8.0, -1, 1)
        out[14] = np.clip(1.0 - gd / 32.0, 0, 1)

        mdx, mdy, md, mkind, _ = self._nearest_mob()
        near = md < 40
        out[15] = np.where(near, np.clip(mdx / 8.0, -1, 1), 0)
        out[16] = np.where(near, np.clip(mdy / 8.0, -1, 1), 0)
        out[17] = np.where(near, np.clip(1.0 - md / 12.0, 0, 1), 0)
        # looming: how fast the threat is closing.  This is the signal LPLC2/LC4
        # actually encode, and the reason they get their own sensory channel.
        out[18] = np.clip(self.prev_mob_dist - md, 0, 3) / 3.0
        self.prev_mob_dist = md

        out[19] = self.tier / 4.0
        out[20] = self.stage() / N_MILESTONES
        for j, item in enumerate(("log", "cobble", "iron", "diamond")):
            out[21 + j] = np.clip(self.inv[:, ITEM_IX[item]] / 4.0, 0, 1)
        out[25] = self.hp / MAX_HP
        for d in range(3):
            out[26 + d] = self.dim == d
        out[29] = self.mine_prog / 10.0
        out[30] = np.clip(self.tick / self.max_ticks, 0, 1)
        out[31] = 1.0                                  # tonic bias
        return out

    def bearing(self) -> np.ndarray:
        """Direction of the current objective, in radians."""
        gx, gy, _ = self._goal_position()
        return np.arctan2((gy - self.y).astype(np.float32),
                          (gx - self.x).astype(np.float32))

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray):
        """actions: (n,) ints.  Returns (reward, done)."""
        n, S = self.n, self.SIZE
        rows = np.arange(n)
        live = self.alive & ~self.won
        reward = np.zeros(n, np.float32)
        actions = np.where(live, actions, -1)

        # ---- movement -------------------------------------------------------
        mv = (actions >= 0) & (actions < 4)
        d = np.clip(actions, 0, 3)
        self.face = np.where(mv, d, self.face).astype(np.int8)
        nx = np.clip(self.x + np.where(mv, DX[d], 0), 0, S - 1)
        ny = np.clip(self.y + np.where(mv, DY[d], 0), 0, S - 1)
        target = self.grid[self.dim, rows, ny, nx]
        can = mv & WALKABLE[target]
        self.x = np.where(can, nx, self.x).astype(np.int16)
        self.y = np.where(can, ny, self.y).astype(np.int16)
        # Walking away abandons a half-mined block; bumping into one does not.
        self.mine_prog = np.where(can, 0, self.mine_prog).astype(np.int16)

        # stepping into a portal changes dimension
        standing = self.grid[self.dim, rows, self.y, self.x]
        to_nether = can & (standing == PORTAL) & (self.dim == OVERWORLD)
        to_over = can & (standing == PORTAL) & (self.dim == NETHER)
        to_end = can & (standing == END_PORTAL)
        self.dim = np.where(to_nether, NETHER, self.dim).astype(np.int8)
        self.dim = np.where(to_over, OVERWORLD, self.dim).astype(np.int8)
        self.dim = np.where(to_end, END, self.dim).astype(np.int8)
        moved_dim = to_nether | to_over | to_end
        if moved_dim.any():
            i = np.flatnonzero(moved_dim)
            self.x[i], self.y[i] = 3, 3
            self.grid[self.dim[i], i, 3, 3] = AIR

        # ---- mining ---------------------------------------------------------
        # You swing at the block you face.  If that is not something you can
        # break, the swing goes to an adjacent block instead, preferring the one
        # the current milestone needs - which of four neighbours to hit is a menu
        # problem, not a behaviour problem.
        mine = actions == 4
        fx = np.clip(self.x + DX[self.face], 0, S - 1)
        fy = np.clip(self.y + DY[self.face], 0, S - 1)
        block = self.grid[self.dim, rows, fy, fx]
        breakable = SOLID[block] & (self.tier >= TIER_REQ[block])
        want = self._goal_block()
        for prefer in (True, False):
            for d in range(4):
                ax = np.clip(self.x + DX[d], 0, S - 1)
                ay = np.clip(self.y + DY[d], 0, S - 1)
                ab = self.grid[self.dim, rows, ay, ax]
                alt = SOLID[ab] & (self.tier >= TIER_REQ[ab])
                if prefer:
                    alt &= ab == want
                alt &= ~breakable
                fx = np.where(alt, ax, fx)
                fy = np.where(alt, ay, fy)
                block = np.where(alt, ab, block)
                breakable = breakable | alt
        ok = mine & breakable
        key = (fy.astype(np.int32) * S + fx).astype(np.int32)
        restart = ok & (self.mine_target != key)
        self.mine_prog = np.where(restart, 0, self.mine_prog).astype(np.int16)
        self.mine_target = np.where(ok, key, self.mine_target)
        self.mine_prog = (self.mine_prog + ok).astype(np.int16)
        broke = ok & (self.mine_prog >= HARDNESS[block])
        if broke.any():
            i = np.flatnonzero(broke)
            b = block[i]
            self.grid[self.dim[i], i, fy[i], fx[i]] = AIR
            self.mine_prog[i] = 0
            bulk = {LOG: 2}
            drop = {LOG: "log", STONE: "cobble", IRON_ORE: "iron_ore",
                    DIAMOND_ORE: "diamond", OBSIDIAN: "obsidian",
                    NETHERRACK: "cobble", END_STONE: "cobble"}
            for bt, item in drop.items():
                sel = i[b == bt]
                if sel.size:
                    self.inv[sel, ITEM_IX[item]] += bulk.get(bt, 1)
                    reward[sel] += 0.3

        # ---- place ----------------------------------------------------------
        # What you place is dictated by the route; the agent chooses *when*.
        # Blocks go in the cell you face, or the nearest free one if it is taken.
        px, py = fx.copy(), fy.copy()
        free = WALKABLE[block]
        for d in range(3, -1, -1):
            ax = np.clip(self.x + DX[d], 0, S - 1)
            ay = np.clip(self.y + DY[d], 0, S - 1)
            alt_ok = ~free & WALKABLE[self.grid[self.dim, rows, ay, ax]]
            px = np.where(alt_ok, ax, px)
            py = np.where(alt_ok, ay, py)
            free = free | alt_ok
        place = (actions == 5) & free
        st = self.stage()
        recipes = (
            (MILESTONES.index("place_bench"), BENCH, "planks", 4, 1.0),
            (MILESTONES.index("light_portal"), PORTAL, "obsidian", 10, 3.0),
            (MILESTONES.index("enter_end"), END_PORTAL, "eye", 2, 3.0),
        )
        for stage_ix, block_id, item, cost, bonus in recipes:
            sel = place & (st == stage_ix) & (self.inv[:, ITEM_IX[item]] >= cost)
            if sel.any():
                i_ = np.flatnonzero(sel)
                self.grid[self.dim[i_], i_, py[i_], px[i_]] = block_id
                self.inv[i_, ITEM_IX[item]] -= cost
                reward[i_] += bonus

        # ---- craft (context-sensitive: applies the next possible recipe) ----
        craft = actions == 6
        if craft.any():
            reward += self._craft(craft)

        # ---- attack ---------------------------------------------------------
        atk = actions == 7
        if atk.any():
            mdx, mdy, md, mkind, mk = self._nearest_mob()
            hit = atk & (md <= 1.5) & (self.mob_hp[rows, mk] > 0)
            dmg = 1 + (self.tier >= 2) + (self.tier >= 4)
            if hit.any():
                i = np.flatnonzero(hit)
                self.mob_hp[i, mk[i]] -= dmg[i]
                killed = self.mob_hp[i, mk[i]] <= 0
                ki = i[killed]
                if ki.size:
                    kinds = self.mob_kind[ki, mk[ki]]
                    self.inv[ki[kinds == BLAZE], ITEM_IX["blaze_rod"]] += 2
                    self.inv[ki[kinds == ZOMBIE], ITEM_IX["pearl"]] += 2
                    reward[ki] += 1.5
                    dragon = ki[kinds == DRAGON]
                    if dragon.size:
                        self.won[dragon] = True
                        reward[dragon] += 50.0

        # ---- blaze spawner yields a blaze to fight --------------------------
        near_spawn = (self.grid[self.dim, rows, fy, fx] == SPAWNER) & (self.dim == NETHER)
        if near_spawn.any():
            i = np.flatnonzero(near_spawn & (self.mob_hp[:, 4] <= 0))
            if i.size:
                self.mob_hp[i, 4] = 4
                self.mob_x[i, 4], self.mob_y[i, 4] = fx[i], fy[i]

        # ---- mobs move and attack ------------------------------------------
        self._step_mobs(live)

        # ---- hazards --------------------------------------------------------
        here = self.grid[self.dim, rows, self.y, self.x]
        self.hp -= np.where(live & HAZARD[here], 8.0, 0.0)
        self.hp -= np.where(live & (self.dim == NETHER), 0.004, 0.0)
        self.hp = np.minimum(self.hp + 0.02 * live, MAX_HP)    # slow regeneration

        # ---- milestones and shaping ----------------------------------------
        reward += self._update_milestones()
        gd = self._goal_distance()
        progress = np.clip(self.prev_goal_dist - gd, -3, 3)
        reward += 0.08 * progress * live
        self.prev_goal_dist = gd
        reward -= 0.004 * live                     # time is the enemy of a speedrun

        self.tick += live
        dead = live & (self.hp <= 0)
        reward[dead] -= 5.0
        self.alive &= ~dead
        done = ~self.alive | self.won | (self.tick >= self.max_ticks)
        return reward, done

    # ------------------------------------------------------------------ craft
    def _craft(self, mask: np.ndarray) -> np.ndarray:
        """One craft action applies the recipe the current milestone calls for.

        Choosing *which* of Minecraft's hundreds of recipes to use is a menu
        problem, not a behaviour problem, so the route fixes the recipe and the
        agent only has to work out when it is worth pressing craft.  Missing
        intermediates (planks, sticks) are produced first, one step per press.
        """
        inv, I = self.inv, ITEM_IX
        r = np.zeros(self.n, np.float32)
        st = self.stage()

        def fire(sel, costs, gains, tier=None, bonus=1.0):
            for item, k in costs.items():
                sel = sel & (inv[:, I[item]] >= k)
            i = np.flatnonzero(sel)
            if i.size:
                for item, k in costs.items():
                    inv[i, I[item]] -= k
                for item, k in gains.items():
                    inv[i, I[item]] += k
                if tier is not None:
                    self.tier[i] = np.maximum(self.tier[i], tier)
                r[i] += bonus
                mask[i] = False
            return sel

        M = MILESTONES.index
        # tools for the current stage
        fire(mask & (st == M("iron_pickaxe")), {"iron": 3, "stick": 2}, {}, 3, 3.0)
        fire(mask & (st == M("diamond_pickaxe")), {"diamond": 3, "stick": 2}, {}, 4, 4.0)
        fire(mask & (st == M("stone_pickaxe")), {"cobble": 3, "stick": 2}, {}, 2, 2.0)
        fire(mask & (st == M("wooden_pickaxe")), {"planks": 3, "stick": 2}, {}, 1, 2.0)
        fire(mask & (st == M("craft_eyes")), {"blaze_rod": 2, "pearl": 2}, {"eye": 2},
             bonus=3.0)
        # intermediates, available from the stage that unlocks them onward
        fire(mask & (st >= M("smelt_iron")), {"iron_ore": 1}, {"iron": 1}, bonus=1.0)
        needs_stick = np.isin(st, [M("wooden_pickaxe"), M("stone_pickaxe"),
                                   M("iron_pickaxe"), M("diamond_pickaxe")])
        fire(mask & needs_stick & (inv[:, I["stick"]] < 2), {"planks": 2},
             {"stick": 4}, bonus=0.3)
        fire(mask, {"log": 1}, {"planks": 4}, bonus=0.4)
        return r

    # ------------------------------------------------------------------- mobs
    def _step_mobs(self, live: np.ndarray) -> None:
        rows = np.arange(self.n)
        same = (self.mob_dim == self.dim[:, None]) & (self.mob_hp > 0)
        dx = self.x[:, None] - self.mob_x
        dy = self.y[:, None] - self.mob_y
        step_x = np.sign(dx) * (np.abs(dx) >= np.abs(dy))
        step_y = np.sign(dy) * (np.abs(dy) > np.abs(dx))
        slow = (self.tick[:, None] % 2) == 0        # mobs are slower than the player
        go = same & slow & live[:, None]
        nx = np.clip(self.mob_x + np.where(go, step_x, 0), 1, self.SIZE - 2)
        ny = np.clip(self.mob_y + np.where(go, step_y, 0), 1, self.SIZE - 2)
        free = WALKABLE[self.grid[self.mob_dim.ravel(),
                                  np.repeat(rows, N_MOBS),
                                  ny.ravel(), nx.ravel()]].reshape(self.n, N_MOBS)
        self.mob_x = np.where(free, nx, self.mob_x).astype(np.int16)
        self.mob_y = np.where(free, ny, self.mob_y).astype(np.int16)

        # Mobs hit on a cooldown rather than every tick.  At 20 ticks per second
        # a per-tick hit is 30+ damage a second, which makes standing still to
        # craft instantly fatal and drowns out everything there is to learn.
        swing = (self.tick[:, None] % MOB_COOLDOWN) == 0
        touching = same & swing & (np.abs(self.x[:, None] - self.mob_x)
                                   + np.abs(self.y[:, None] - self.mob_y) <= 1)
        power = np.array([2.0, 5.0, 3.0, 4.0], np.float32)[self.mob_kind]
        self.hp -= (touching * power).sum(axis=1) * live

    # -------------------------------------------------------------- milestones
    def _update_milestones(self) -> np.ndarray:
        inv, I = self.inv, ITEM_IX
        S = self.SIZE
        rows = np.arange(self.n)
        g = self.grid[self.dim, rows]
        has_bench = (self.grid[OVERWORLD] == BENCH).reshape(self.n, -1).any(axis=1)
        has_portal = (self.grid[OVERWORLD] == PORTAL).reshape(self.n, -1).any(axis=1)
        reached = np.stack([
            inv[:, I["log"]] >= 1,
            inv[:, I["planks"]] >= 1,
            has_bench,
            self.tier >= 1,
            inv[:, I["cobble"]] >= 1,
            self.tier >= 2,
            inv[:, I["iron_ore"]] >= 1,
            inv[:, I["iron"]] >= 1,
            self.tier >= 3,
            inv[:, I["diamond"]] >= 1,
            self.tier >= 4,
            inv[:, I["obsidian"]] >= 10,
            has_portal,
            self.dim == NETHER,
            inv[:, I["blaze_rod"]] >= 2,
            inv[:, I["pearl"]] >= 2,
            inv[:, I["eye"]] >= 2,
            self.dim == END,
            self.won,
        ], axis=1)
        # milestones only count in order - you cannot skip the tech tree.
        # Already-earned steps stay earned even after you spend the materials.
        ordered = np.minimum.accumulate(reached | self.done_ms, axis=1)
        fresh = ordered & ~self.done_ms
        self.done_ms = ordered
        if fresh.any():
            self.ms_tick[fresh] = np.repeat(self.tick, N_MILESTONES).reshape(
                self.n, N_MILESTONES)[fresh]
        weight = np.linspace(2.0, 12.0, N_MILESTONES, dtype=np.float32)
        return (fresh * weight).sum(axis=1)

    # ------------------------------------------------------------------ stats
    def progress(self) -> np.ndarray:
        return self.done_ms.sum(axis=1)
