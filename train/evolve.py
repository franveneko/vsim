"""Evolve a population of connectome-constrained flies on the MineSim speedrun.

The wiring diagram never changes - it is measured data.  Evolution only touches
the interface: how strongly each sensory channel drives its receptive cell types,
and how descending-neuron activity is decoded into body commands.  Within a
single life, the mushroom body additionally re-weights its own KC->MBON synapses
under dopaminergic control, exactly as in `flybrain.network`.

Selection is a plain truncation ES: evaluate the whole population in parallel,
keep the best, and repopulate from them with Gaussian mutation.  With ~1k free
parameters that beats anything fancier, and it is trivially parallel - which is
the whole point of running hundreds of flies at once.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from flybrain.connectome import build
from flybrain.network import Population
from minesim.world import MILESTONES, N_MILESTONES, MineSim
from . import worker

OUT = Path(__file__).resolve().parent.parent / "out"


@dataclass
class GenStats:
    gen: int
    best_fitness: float
    mean_fitness: float
    best_progress: int
    mean_progress: float
    wins: int
    dragon_kills: int
    best_win_ticks: int
    elapsed: float

    @property
    def best_milestone(self) -> str:
        i = min(self.best_progress, N_MILESTONES - 1)
        return "COMPLETE" if self.best_progress >= N_MILESTONES else MILESTONES[i]


class Evolution:
    def __init__(self, pop_size: int = 128, episode: int = 1000, seed: int = 0,
                 elite_frac: float = 0.15, sigma: float = 0.35,
                 sigma_decay: float = 0.985, sigma_min: float = 0.06,
                 workers: int = 1, world_every: int = 3,
                 segments=worker.DEFAULT_SEGMENTS):
        self.conn = build()
        self.P = pop_size
        self.episode = episode
        self.workers = workers
        # Hold a map fixed for a few generations so an elite has time to be
        # consolidated, then move on so nothing memorises one layout.
        self.world_every = world_every
        self.segments = tuple(segments)
        self.pool = None
        if workers > 1:
            assert pop_size % workers == 0, "population must divide evenly over workers"
            self.chunk = pop_size // workers
            ctx = mp.get_context("fork")
            self.pool = ctx.Pool(workers, initializer=worker.init,
                                 initargs=(self.chunk, episode, seed))
        # a local batch is kept regardless: it runs the recorded showcase runs
        self.env = MineSim(pop_size, seed=seed, max_ticks=episode)
        self.brain = Population(self.conn, pop_size, obs_dim=MineSim.OBS_DIM, seed=seed)
        self.rng = np.random.default_rng(seed + 1)
        self.n_elite = max(2, int(elite_frac * pop_size))
        self.sigma, self.sigma_decay, self.sigma_min = sigma, sigma_decay, sigma_min
        self.genomes = self.rng.normal(0, 0.4, (pop_size, self.brain.genome_size)).astype(np.float32)
        self.history: list[GenStats] = []
        self.best_genome = self.genomes[0].copy()
        self.best_fitness = -1e9
        # kept for the video: the best fly of every generation, and the whole
        # population at a few points along the way
        self.gen_best: list[np.ndarray] = []
        self.snapshots: dict[int, np.ndarray] = {}
        self.snapshot_every = 20

    # ------------------------------------------------------------------ rollout
    def rollout(self, genomes: np.ndarray, world_seed: int, record: bool = False,
                start_stage: int = 0):
        env, brain = self.env, self.brain
        env.reset(seed=world_seed, start_stage=start_stage)
        brain.reset()
        brain.set_genomes(genomes)
        fitness = np.zeros(self.P, np.float32)
        frames = [] if record else None
        for _ in range(self.episode):
            obs = env.observe()
            logits = brain.step(obs, env.bearing())  # noqa: E501
            actions = np.argmax(logits, axis=0)
            r, done = env.step(actions)
            fitness += r
            if record:
                frames.append(self.snapshot(actions))
            if done.all():
                break
        # a completed run is worth more the faster it was
        win_ticks = env.ms_tick[:, -1]
        fitness += np.where(env.won, 60.0 + 40.0 * (1.0 - win_ticks / self.episode), 0.0)
        return fitness, frames

    def snapshot(self, actions: np.ndarray) -> dict:
        env, brain = self.env, self.brain
        return {
            "x": env.x.copy(), "y": env.y.copy(), "dim": env.dim.copy(),
            "face": env.face.copy(), "hp": env.hp.copy(),
            "stage": env.stage().copy(), "tick": env.tick.copy(),
            "inv": env.inv.copy(), "tier": env.tier.copy(),
            "won": env.won.copy(), "alive": env.alive.copy(),
            "actions": actions.copy(),
            "mob_x": env.mob_x.copy(), "mob_y": env.mob_y.copy(),
            "mob_hp": env.mob_hp.copy(), "mob_dim": env.mob_dim.copy(),
            "mob_kind": env.mob_kind.copy(),
            "rates": {k: v.copy() for k, v in brain.population_rates().items()},
            "kc_active": (brain.r[brain.kc] > 0).sum(axis=0),
            "mb_gain": brain.g.mean(axis=0),
        }

    # ---------------------------------------------------------------- evolution
    def evaluate(self, genomes: np.ndarray, world_seed: int):
        """Fitness, milestones, wins and completion ticks for the whole population."""
        if self.pool is None:
            fitness, _ = self.rollout(genomes, world_seed)
            e = self.env
            return (fitness, e.progress().copy(), e.won.copy(),
                    e.ms_tick[:, -1].copy(), e.won.copy())
        chunks = [(genomes[i * self.chunk:(i + 1) * self.chunk], world_seed,
                   self.segments) for i in range(self.workers)]
        parts = self.pool.map(worker.evaluate, chunks)
        per_seg = np.concatenate([p[0] for p in parts], axis=1)   # (n_seg, P)
        rest = tuple(np.concatenate([p[j] for p in parts]) for j in range(1, 5))
        weights = np.array([w for _, _, w in self.segments], np.float32)
        fitness = (self._rank(per_seg) * weights[:, None]).sum(axis=0)
        self.raw_fitness = per_seg
        return (fitness,) + rest

    @staticmethod
    def _rank(x: np.ndarray) -> np.ndarray:
        """Map each row to evenly spaced ranks in [-0.5, 0.5].

        Selection then depends on the *order* a segment puts the population in,
        not on how many points that segment happens to hand out.
        """
        order = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float32)
        return order / max(x.shape[1] - 1, 1) - 0.5

    def step_generation(self, gen: int) -> GenStats:
        t0 = time.time()
        fitness, progress, won, win_ticks, any_win = self.evaluate(
            self.genomes, 1000 + gen // self.world_every)
        order = np.argsort(-fitness)
        elite = self.genomes[order[: self.n_elite]]

        self.gen_best.append(self.genomes[order[0]].copy())
        if gen % self.snapshot_every == 0:
            self.snapshots[gen] = self.genomes.copy()
        if fitness[order[0]] > self.best_fitness:
            self.best_fitness = float(fitness[order[0]])
            self.best_genome = self.genomes[order[0]].copy()

        # repopulate: elites survive untouched, the rest are mutated elites
        children = np.empty_like(self.genomes)
        children[: self.n_elite] = elite
        parents = elite[self.rng.integers(0, self.n_elite, self.P - self.n_elite)]
        children[self.n_elite:] = parents + self.rng.normal(
            0, self.sigma, (self.P - self.n_elite, self.brain.genome_size)).astype(np.float32)
        self.genomes = children
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

        wins = int(won.sum())
        st = GenStats(
            gen=gen,
            best_fitness=float(fitness.max()),
            mean_fitness=float(fitness.mean()),
            best_progress=int(progress.max()),
            mean_progress=float(progress.mean()),
            wins=wins,
            dragon_kills=int(any_win.sum()),
            best_win_ticks=int(win_ticks[won].min()) if wins else -1,
            elapsed=time.time() - t0,
        )
        self.history.append(st)
        return st

    def run(self, generations: int, log_path: Path | None = None) -> None:
        for g in range(generations):
            st = self.step_generation(g)
            print(f"gen {st.gen:3d}  best {st.best_fitness:7.3f}  mean {st.mean_fitness:7.3f}"
                  f"  stage {st.best_milestone:16s} mean_ms {st.mean_progress:5.2f}"
                  f"  full-run wins {st.wins:3d}  dragons {st.dragon_kills:3d}"
                  f"  sigma {self.sigma:.3f}  {st.elapsed:5.1f}s",
                  flush=True)
            if log_path and (g % 5 == 0 or g == generations - 1):
                self.save(log_path)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"),
                 best_genome=self.best_genome, genomes=self.genomes,
                 gen_best=np.array(self.gen_best, np.float32),
                 snapshot_gens=np.array(sorted(self.snapshots), np.int32),
                 snapshots=np.array([self.snapshots[g] for g in sorted(self.snapshots)],
                                    np.float32))
        with open(path.with_suffix(".json"), "w") as fh:
            json.dump({"history": [asdict(s) for s in self.history],
                       "pop_size": self.P, "episode": self.episode,
                       "genome_size": int(self.brain.genome_size),
                       "n_neurons": int(self.conn.n),
                       "n_connections": int(self.conn.W.nnz),
                       "segments": [list(x) for x in self.segments],
                       "best_fitness": float(self.best_fitness)}, fh, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--episode", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--world-every", type=int, default=3)
    ap.add_argument("--out", type=str, default=str(OUT / "evolution"))
    a = ap.parse_args()
    ev = Evolution(pop_size=a.pop, episode=a.episode, seed=a.seed, workers=a.workers,
                   world_every=a.world_every)
    print(ev.conn.summary())
    print(f"genome: {ev.brain.genome_size} free parameters per fly; "
          f"{ev.brain.n_plastic} plastic KC->MBON synapses")
    ev.run(a.generations, Path(a.out))
    ev.save(Path(a.out))


if __name__ == "__main__":
    main()
