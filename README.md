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

## Data

`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome` — public release
bucket. `flybrain/download.py` fetches the annotation, neurotransmitter and
connection-weight tables directly.

The 16×16 block textures, the sprites and the soundtrack are all generated
procedurally by this repository. No game assets are used.
