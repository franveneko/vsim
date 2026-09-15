#!/usr/bin/env python3
"""Build all three videos from the artefacts in `out/`.

    python3 make_videos.py                 # all three
    python3 make_videos.py --only youtube  # just one
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from render import story
from render.video import mux, probe

OUT = Path(__file__).resolve().parent / "out"

BUILDERS = {
    "youtube": (story.youtube, "long"),
    "reels": (story.reels, "reels"),
    "tiktok": (story.tiktok, "short"),
}


def duration(path: Path) -> float:
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", probe(path))
    if not m:
        return 0.0
    h, mi, se = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(se)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(BUILDERS), default=None)
    ap.add_argument("--genomes", default=str(OUT / "evolution.npz"))
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    s = story.stats(OUT)
    print(f"run: {s['generations']} generations, best milestone "
          f"{s['milestone']}, {'WON' if s['won'] else 'not completed'}")

    targets = [a.only] if a.only else list(BUILDERS)
    for name in targets:
        builder, style = BUILDERS[name]
        final = OUT / f"fly_minecraft_{name}.mp4"
        print(f"\n=== {name} ===", flush=True)
        silent = builder(Path(a.genomes), final, s)
        secs = duration(silent)
        print(f"  video: {secs:.1f}s  {silent.stat().st_size / 1e6:.1f} MB")
        if a.no_audio:
            silent.rename(final)
        else:
            track = story.soundtrack(secs, OUT / f"track_{name}.wav", style)
            mux(silent, track, final)
            silent.unlink()
            track.unlink()
        print(f"  -> {final}  ({final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
