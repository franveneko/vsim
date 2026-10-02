# vsim — a measured fly brain plays a Minecraft-shaped game

This repository downloads the **MaleCNS v1.0 connectome** (Janelia FlyEM +
Google Research, published in *Cell*, September 2026 — the complete central
nervous system of a male *Drosophila*, ~166,000 neurons and ~125 million
synapses), turns part of it into a runnable network, puts hundreds of copies of
that network into a Minecraft-shaped voxel sandbox at the same time, evolves the
interface between brain and body until the flies can play, and renders the whole
thing as three videos (one long landscape cut, two vertical ones).

Everything here runs on CPU, with no GPU and no game client.

---

## What is real, and what is a model

Being precise about this is the point of the project.

**Real, taken from the data and not modified**

| | |
|---|---|
| The graph | Every connection is a measured segment-to-segment connection from the proofread `traced-only` release table. |
| The signs | Excitatory / inhibitory comes from each neuron's predicted neurotransmitter (ACh +, GABA −, glutamate −, histamine −). |
| The modulatory circuit | Dopaminergic, octopaminergic and serotonergic outputs are kept separate: they gate plasticity instead of driving their targets. |
| The site of learning | Plasticity happens at Kenyon-cell → MBON synapses, under dopaminergic control. That is the site and sign reported for *Drosophila* mushroom-body learning. |
| The cell types | Looming detectors (LPLC2, LC4), small-object cells (LC11, LC18), the central-complex compass (EPG/ER/PEN), descending neurons (DNa02, MDN, DNp09…) are the actual named cells, selected by type. |

**Modelled, because the data does not reach that far**

- **Neurons are rate units**, not spiking compartmental models: a leaky rate with
  a saturating transfer function and spike-frequency adaptation.
- **Input weights are normalised per neuron.** Raw synapse counts leave the
  network in a saturated attractor where sensory input cannot compete with
  recurrent drive. Each neuron's total input is scaled to a fixed budget, which
  preserves the *relative* weights of its inputs — the measured quantity.
- **Projection-neuron drive onto Kenyon cells** is a sparse random projection.
  The antennal lobe is outside the simulated core, and PN→KC connectivity is
  close to random in real flies anyway.
- **APL's feedback inhibition** is implemented as k-winners-take-all over the
  Kenyon cells, which is what that inhibition accomplishes.
- **The sensory mapping** — which game feature drives which cell type — is a
  modelling choice, informed by what those cells are known to be tuned to.
- **The sandbox is not Minecraft.** See below.

**Scope.** The simulated core is the sensorimotor spine of the brain: visual
projection neurons, the mushroom body (Kenyon cells, MBONs, DANs), the central
complex, and the descending neurons — 17,966 neurons and 329,706 connections at
the usual ≥5-synapse significance threshold. The optic-lobe interior, the VNC and
most cb_intrinsic neurons are dropped, because simulating 166k neurons for
hundreds of agents at once does not fit on four CPU cores.

## The sandbox

Minecraft is proprietary, needs an account, a GPU and a desktop session, and no
headless bridge would run hundreds of instances here. `minesim` reimplements what
makes Minecraft a *learning problem*:

- a blocky world you mine and build in,
- a tool tech-tree where each tier gates the next material,
- hostile mobs that kill you,
- three dimensions reached through portals,
- and the real Any%-glitchless win condition: **kill the Ender Dragon**.

The route is 19 ordered milestones, from `punch_wood` to `slay_dragon`. Every
world is stored with a leading agent axis, so hundreds of worlds advance on every
`step()`.

Three of Minecraft's interfaces are collapsed on purpose, because they are menu
problems rather than behaviour problems — the agent still decides *when*, the
route decides *what*:

- **crafting** applies the recipe the current milestone calls for, producing
  missing planks or sticks first;
- **placing** puts down the block that milestone needs, in the cell you face or
  the nearest free one;
- **mining** swings at the block you face, or at an adjacent one — preferring
  what the milestone needs — if the faced cell is not breakable.

One balance number is worth stating because it decides whether anything can be
learned at all: mobs attack on a 12-tick cooldown. At 20 ticks per second, a
per-tick hit is over 30 damage a second, which makes standing still to craft
instantly fatal and drowns out the rest of the problem.

`minesim/scripted.py` is a hand-written speedrunner used as a reference: it
proves the world is completable and gives the evolved flies a benchmark.

## What learns

The wiring diagram never changes. Two things do:

1. **Between generations** — 1,009 free parameters per fly: how strongly each
   sensory channel drives its receptive cell types, how descending-neuron
   activity is decoded into body commands, and two input gains. Selection is
   truncation ES: evaluate the population in parallel, keep the best, repopulate
   from them with Gaussian mutation.
2. **Within a single life** — the mushroom body depresses its own KC→MBON
   synapses wherever Kenyon-cell activity coincides with dopaminergic input to
   the same MBON.

Every fly in a generation plays the **same world** — comparing genomes that each
played a different map measures luck, not policy. A generation is scored over
several segments: one full run from a standing start (the number that is
reported) plus shorter runs dropped in at later checkpoints, so the circuits for
stone, iron, the nether and the dragon fight get selection pressure too.

## Running it

```bash
pip install -r requirements.txt

python3 -m flybrain.download          # fetch the MaleCNS tables (~565 MB)
python3 -m flybrain.connectome        # build and cache the signed core graph
python3 -m minesim.scripted           # sanity-check: the reference speedrun

python3 -m train.evolve --pop 192 --workers 4 --generations 130
python3 -m train.showcase --agents 128 --worlds 8
python3 make_videos.py                # writes out/fly_minecraft_*.mp4
```

## Layout

```
flybrain/     download, build and simulate the connectome
minesim/      the vectorised sandbox + the scripted reference runner
train/        evolution, parallel workers, showcase search
render/       textures, scenes, charts, storyboards, audio, ffmpeg
make_videos.py
```

## What actually happened

90 generations, population 192, on a 4-core CPU.

| | |
|---|---|
| Simulated core | 17,966 neurons, 329,706 measured connections |
| Free parameters per fly | 1,009 (plus 11,587 plastic KC→MBON synapses) |
| Mean milestones, generation 0 → 90 | 0.13 → ~4.8 |
| Best full run, over 1,280 attempts | **11 of 19 milestones** (`mine_obsidian`) |
| Full-route completions | **none** |
| Dropped straight into the End | dragon killed in **4 of 4 worlds**, 92–126 of 160 flies each, fastest **23 ticks** |
| Hand-written reference route | completes in **457 ticks** |

So: the flies learn to chop wood, craft a bench, work up the whole pickaxe
tech-tree and mine obsidian, and they can fight and kill the Ender Dragon when
they start next to it. What they cannot do is chain all nineteen steps in one
continuous life — the hand-written route does that in 457 ticks and they do not
get close. The videos say so on screen.

Training is chunked and resumable (`--resume`), because a detached run does not
survive this environment going idle.

## Real Minecraft

`mcreal/` + `live/` put the same brain in charge of a **real Minecraft bot over
the real protocol** — no simulation of the game at all. `mcreal/bridge.js` is a
[mineflayer](https://github.com/PrismarineJS/mineflayer) bot that translates the
server's view into the same 32-number observation and the same eight actions the
sandbox used, so a brain evolved in the fast sandbox drops straight into the real
game without being retrained.

```bash
cd mcreal && npm install && cd ..
python3 -m live.play --steps 200 --seed-trees 16        # bundled JS server
python3 -m live.play --external-server --host 127.0.0.1 --port 25565 --auth microsoft
```

Nothing about the world, the physics, the block-breaking times or the inventory
is ours in this mode: it is whatever the server says.

### Pointing it at your own Minecraft

This is the mode worth using, and it has to run **on your machine** — a cloud
container cannot reach your computer.

1. Clone this repo locally, `pip install -r requirements.txt`, `cd mcreal && npm install`.
2. Start your own server (vanilla, Paper, or Open to LAN from a single-player
   world) and note its port.
3. Run the bot against it:

```bash
python3 -m live.play --external-server --host 127.0.0.1 --port 25565 \
        --version 1.18.2 --auth microsoft --steps 2000
```

`--auth microsoft` signs in with your real account (needed for an online-mode
server); `--auth offline` is enough for a LAN world. Match `--version` to your
server.

### What runs today, and what does not

Verified against the bundled JavaScript server (flying-squid, Minecraft 1.18.2):

| | |
|---|---|
| Real protocol, world, physics, block breaking, inventory, health | works |
| The connectome driving it | works — 10 blocks broken in a 150-step run, 5 of them oak logs |
| Milestones reached | `punch_wood` ✓, then stuck |

It stops at the first craft because **flying-squid implements no recipes, no
loot tables and no mob AI** — it is a protocol server, not the game. Everything
past `craft_planks` needs Mojang's own server jar.

This container cannot download it: the environment's network policy denies
`piston-meta.mojang.com`, `piston-data.mojang.com` and `libraries.minecraft.net`.
Allow those hosts in the environment's Network access setting and the whole
route becomes available here; on your own machine the restriction does not exist.

### Transfer, honestly

The brain was evolved in a 2D sandbox and is being asked to drive a 3D game, so
the mapping does real work: cardinal moves become look-and-walk, and mining is
restricted to blocks that are in the way or that the milestone wants — in the
sandbox the ground was floor you could not dig, and without that restriction the
bot spends the run digging a hole. Behaviour transfers in outline, not in detail.

## Data

`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome` — public release
bucket. `flybrain/download.py` fetches the annotation, neurotransmitter and
connection-weight tables directly.

The 16×16 block textures, the sprites and the soundtrack are all generated
procedurally by this repository. No game assets are used.
