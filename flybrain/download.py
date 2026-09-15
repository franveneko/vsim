"""Fetch the MaleCNS v1.0 flat-connectome tables from the public release bucket.

Source: gs://flyem-male-cns/v1.0/connectome-data/flat-connectome
Dataset: "Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system" (Janelia FlyEM + Google Research, Cell, Sept 2026).
~166k neurons / ~125M synapses across brain + ventral nerve cord.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com/flyem-male-cns"
RELEASE = "v1.0/connectome-data/flat-connectome"

FILES = {
    # curated per-neuron annotations: class / superclass / type / side
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    # per-neuron neurotransmitter predictions -> synaptic sign
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    # proofread segment-to-segment connection strengths (the wiring diagram)
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "malecns"


def path_for(key: str) -> Path:
    return DATA_DIR / FILES[key]


def fetch(key: str, force: bool = False) -> Path:
    dest = path_for(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    url = f"{BUCKET}/{RELEASE}/{FILES[key]}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    return dest


def fetch_all(force: bool = False) -> dict[str, Path]:
    return {k: fetch(k, force) for k in FILES}


if __name__ == "__main__":
    for k, p in fetch_all("--force" in sys.argv).items():
        print(f"{k:18s} {p} ({p.stat().st_size / 1e6:.1f} MB)")
