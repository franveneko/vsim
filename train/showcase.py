"""Run the evolved fly until it beats the game, and keep the best attempt.

The population here is one genome copied many times.  The copies still diverge,
because the brains are noisy - so this is one fly attempting the run hundreds of
times in parallel, which is exactly what a speedrunner does, minus the patience.

Everything is deterministic given the seeds, so the winning attempt can be
replayed frame by frame afterwards instead of being stored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flybrain.connectome import build
from flybrain.network import Population
from minesim.world import MILESTONES, N_MILESTONES, MineSim

OUT = Path(__file__).resolve().parent.parent / "out"


@dataclass
class Attempt:
    world_seed: int
    agent: int
    won: bool
    ticks: int
    progress: int

    @property
    def milestone(self) -> str:
        return ("COMPLETE" if self.progress >= N_MILESTONES
                else MILESTONES[min(self.progress, N_MILESTONES - 1)])

    def key(self):
        """Sort key: a win beats no win; faster beats slower; further beats less far."""
        return (not self.won, self.ticks if self.won else 10 ** 9, -self.progress)


def make(genome: np.ndarray, n_agents: int, ticks: int, world_seed: int,
         brain_seed: int = 0):
    conn = build()
    env = MineSim(n_agents, seed=world_seed, max_ticks=ticks)
    brain = Population(conn, n_agents, obs_dim=MineSim.OBS_DIM, seed=brain_seed)
    brain.reset()
    brain.set_genomes(np.repeat(genome[None, :], n_agents, axis=0))
    env.reset(seed=world_seed)
    return env, brain


def run(genome: np.ndarray, n_agents: int, ticks: int, world_seed: int,
        brain_seed: int = 0, on_step=None):
    env, brain = make(genome, n_agents, ticks, world_seed, brain_seed)
    for t in range(ticks):
        actions = np.argmax(brain.step(env.observe(), env.bearing()), axis=0)
        _, done = env.step(actions)
        if on_step is not None:
            on_step(t, env, brain, actions)
        if done.all():
            break
    return env, brain


def best_attempt(env: MineSim, world_seed: int) -> Attempt:
    order = sorted(
        (Attempt(world_seed, i, bool(env.won[i]), int(env.ms_tick[i, -1]),
                 int(env.progress()[i])) for i in range(env.n)),
        key=lambda a: a.key())
    return order[0]


def search(genome: np.ndarray, world_seeds, n_agents: int, ticks: int,
           brain_seed: int = 0, verbose: bool = True):
    """Try the fly on several worlds; return every attempt, best first."""
    found = []
    for ws in world_seeds:
        env, _ = run(genome, n_agents, ticks, ws, brain_seed)
        a = best_attempt(env, ws)
        found.append(a)
        if verbose:
            wins = int(env.won.sum())
            print(f"world {ws:5d}: best {a.milestone:14s} "
                  f"{'won in ' + str(a.ticks) + ' ticks' if a.won else ''} "
                  f"({wins}/{n_agents} completed)", flush=True)
    return sorted(found, key=lambda a: a.key())


def score_candidates(candidates: np.ndarray, world_seeds, ticks: int,
                     brain_seed: int = 0, verbose: bool = True) -> np.ndarray:
    """Rank genomes by how far they get on the *full* route, averaged over worlds.

    Training fitness mixes the full run with checkpoint practice, so the genome
    with the best training score is not necessarily the best end-to-end runner.
    This picks the one that actually plays the whole game.
    """
    conn = build()
    n = candidates.shape[0]
    env = MineSim(n, seed=world_seeds[0], max_ticks=ticks)
    brain = Population(conn, n, obs_dim=MineSim.OBS_DIM, seed=brain_seed)
    score = np.zeros(n, np.float64)
    for ws in world_seeds:
        env.max_ticks = ticks
        env.reset(seed=ws)
        brain.reset()
        brain.set_genomes(candidates)
        for _ in range(ticks):
            actions = np.argmax(brain.step(env.observe(), env.bearing()), axis=0)
            _, done = env.step(actions)
            if done.all():
                break
        prog = env.progress().astype(np.float64)
        speed = np.where(env.won, 1.0 - env.ms_tick[:, -1] / ticks, 0.0)
        score += prog + 5.0 * speed
        if verbose:
            print(f"  world {ws}: best {int(prog.max())} milestones, "
                  f"{int(env.won.sum())} completions", flush=True)
    return score / len(world_seeds)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--genomes", default=str(OUT / "evolution.npz"))
    ap.add_argument("--agents", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=1400)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--out", default=str(OUT / "showcase.json"))
    ap.add_argument("--select-worlds", type=int, default=3)
    ap.add_argument("--select-ticks", type=int, default=1400)
    a = ap.parse_args()

    z = np.load(a.genomes)
    pool = np.concatenate([z["gen_best"], z["genomes"]], axis=0).astype(np.float32)
    # de-duplicate: elites survive unchanged, so the pool repeats itself a lot
    _, uniq = np.unique(np.round(pool, 4), axis=0, return_index=True)
    pool = pool[np.sort(uniq)]
    print(f"selecting a champion from {len(pool)} distinct genomes", flush=True)
    score = score_candidates(pool, [7000 + i for i in range(a.select_worlds)],
                             a.select_ticks)
    genome = pool[int(np.argmax(score))]
    print(f"champion score {score.max():.2f} "
          f"(pool mean {score.mean():.2f}, runner-up {np.sort(score)[-2]:.2f})")
    np.save(OUT / "champion.npy", genome)
    seeds = [4000 + i for i in range(a.worlds)]
    results = search(genome, seeds, a.agents, a.ticks)
    best = results[0]
    print(f"\nbest attempt: world {best.world_seed} agent {best.agent} "
          f"{best.milestone} ticks {best.ticks}")
    Path(a.out).write_text(json.dumps(
        {"best": best.__dict__,
         "champion_score": float(score.max()),
         "pool_size": int(len(pool)),
         "all": [r.__dict__ for r in results],
         "agents": a.agents, "ticks": a.ticks}, indent=1))


if __name__ == "__main__":
    main()
