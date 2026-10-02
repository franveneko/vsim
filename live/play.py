"""Put the evolved fly brain in charge of a real Minecraft bot.

The brain is the same one from `train.evolve`: the measured MaleCNS connectome,
the mushroom-body plasticity, and the genome selected in the sandbox.  Nothing
is retrained here - the sandbox's job was to produce a controller, and this is
that controller driving a real game over the real protocol.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from flybrain.connectome import build
from flybrain.network import Population
from live.client import BridgeClient, launch_bridge, launch_server

OUT = Path(__file__).resolve().parent.parent / "out"
ACTIONS = ("north", "south", "east", "west", "mine", "place", "craft", "attack")


def load_genome(path: Path | None) -> np.ndarray:
    for candidate in ([path] if path else []) + [OUT / "champion.npy", OUT / "evolution.npz"]:
        if candidate and candidate.exists():
            if candidate.suffix == ".npy":
                return np.load(candidate)
            return np.load(candidate)["best_genome"]
    raise FileNotFoundError("no evolved genome found; run train.evolve first")


def play(steps: int, genome_path: Path | None = None, host: str = "127.0.0.1",
         port: int = 25565, version: str = "1.18.2", seed: int = 42,
         own_server: bool = True, username: str = "fly0", auth: str = "offline",
         trace_path: Path | None = None, quiet: bool = False,
         seed_trees: int = 0) -> dict:
    procs: list[subprocess.Popen] = []
    try:
        if own_server:
            procs.append(launch_server(port=port, version=version, seed=seed,
                                       log=OUT / "mc_server.log"))
            time.sleep(6)
        procs.append(launch_bridge(host=host, port=port, username=username,
                                   version=version, auth=auth,
                                   log=OUT / "mc_bridge.log"))

        client = BridgeClient()
        client.connect()

        conn = build()
        brain = Population(conn, n_agents=1, obs_dim=32, seed=0)
        brain.reset()
        brain.set_genomes(load_genome(genome_path)[None, :])

        if seed_trees:
            note = client.seed_world(seed_trees).info["note"]
            print(f"[fixture] {note}", flush=True)
        step = client.reset()
        trace = []
        t0 = time.time()
        for i in range(steps):
            obs = step.obs[:, None].astype(np.float32)
            bearing = np.array([step.bearing], np.float32)
            logits = brain.step(obs, bearing)
            action = int(np.argmax(logits[:, 0]))
            step = client.step(action)
            info = step.info
            trace.append({
                "i": i, "action": ACTIONS[action], **{
                    k: info[k] for k in ("tick", "stage", "milestone",
                                         "milestones_done", "pos", "health",
                                         "tier", "note")}
            })
            note = str(info.get("note") or "")
            if not quiet and (i % 10 == 0 or note.startswith("dug") or
                              note.startswith("crafted") or note.startswith("placed")):
                rates = {k: round(float(v[0]), 3) for k, v in brain.population_rates().items()}
                print(f"{i:4d} {ACTIONS[action]:7s} {info['milestone']:15s} "
                      f"pos={info['pos']} hp={info['health']} "
                      f"inv={info['inventory'][:4]} DN={rates['DN']} note={info['note']}",
                      flush=True)
        elapsed = time.time() - t0
        result = {
            "steps": steps, "seconds": round(elapsed, 1),
            "milestones_done": step.info["milestones_done"],
            "milestone": step.info["milestone"],
            "inventory": step.info["inventory"],
            "position": step.info["pos"],
            "actions": {a: sum(1 for r in trace if r["action"] == a) for a in ACTIONS},
            "blocks_mined": sum(1 for r in trace if str(r["note"] or "").startswith("dug")),
        }
        if trace_path:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps({"result": result, "trace": trace}, indent=1))
        client.close()
        return result
    finally:
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--genome", type=str, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=25565)
    ap.add_argument("--version", default="1.18.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--username", default="fly0")
    ap.add_argument("--auth", choices=("offline", "microsoft"), default="offline",
                    help="microsoft signs in with a real account (online-mode servers)")
    ap.add_argument("--external-server", action="store_true",
                    help="connect to a server already running (e.g. your own)")
    ap.add_argument("--seed-trees", type=int, default=0,
                    help="plant oak logs at spawn (JS server has no tree generation)")
    ap.add_argument("--trace", default=str(OUT / "mc_trace.json"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    result = play(a.steps, Path(a.genome) if a.genome else None, a.host, a.port,
                  a.version, a.seed, own_server=not a.external_server,
                  username=a.username, auth=a.auth, trace_path=Path(a.trace),
                  seed_trees=a.seed_trees, quiet=a.quiet)
    print("\n" + json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
