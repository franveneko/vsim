"""Worker-side evaluation of a slice of the population.

Each worker owns its own MineSim batch and brain batch, built once and reused;
only genomes and the segment plan cross the process boundary.

A generation is scored over several *segments*.  The first runs the whole route
from a standing start - that is the number that matters, and the one reported.
The rest drop the population in at later checkpoints so the circuits for stone,
iron, the nether and the dragon fight get selection pressure too; without them a
population that never survives past wood would never be scored on anything else.
"""
from __future__ import annotations

import numpy as np

from flybrain.connectome import build
from flybrain.network import Population
from minesim.world import MineSim

_STATE: dict = {}

# (start milestone, ticks, weight)
DEFAULT_SEGMENTS = ((0, 1100, 1.2), (5, 200, 0.7), (9, 200, 0.7),
                    (13, 200, 0.7), (17, 200, 0.9))


def init(chunk: int, episode: int, seed: int) -> None:
    conn = build()
    _STATE["env"] = MineSim(chunk, seed=seed, max_ticks=episode)
    _STATE["brain"] = Population(conn, chunk, obs_dim=MineSim.OBS_DIM, seed=seed)


def _run_segment(genomes, world_seed, start_stage, ticks):
    env, brain = _STATE["env"], _STATE["brain"]
    env.max_ticks = ticks
    env.reset(seed=world_seed, start_stage=start_stage)
    brain.reset()
    brain.set_genomes(genomes)
    fitness = np.zeros(env.n, np.float32)
    for _ in range(ticks):
        actions = np.argmax(brain.step(env.observe(), env.bearing()), axis=0)
        r, done = env.step(actions)
        fitness += r
        if done.all():
            break
    win_ticks = env.ms_tick[:, -1]
    fitness += np.where(env.won, 60.0 + 40.0 * (1.0 - win_ticks / max(ticks, 1)), 0.0)
    return fitness, env.progress().copy(), env.won.copy(), win_ticks.copy()


def evaluate(args):
    genomes, world_seed, segments = args
    total = None
    head = None
    dragon_kills = None
    for i, (stage, ticks, weight) in enumerate(segments):
        f, prog, won, wt = _run_segment(genomes, world_seed + 31 * i, stage, ticks)
        total = f * weight if total is None else total + f * weight
        if i == 0:
            head = (prog, won, wt)
        dragon_kills = won if dragon_kills is None else (dragon_kills | won)
    prog, won, wt = head
    return total, prog, won, wt, dragon_kills
