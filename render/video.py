"""Stream frames into ffmpeg.

imageio-ffmpeg ships a static ffmpeg binary, so nothing has to be installed at
the system level.  Frames are piped in as raw RGB, which avoids writing tens of
thousands of PNGs to disk.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


class VideoWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: int = 30,
                 crf: int = 19, audio: Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self.size = size
        cmd = [FFMPEG, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(fps), "-i", "-"]
        if audio is not None:
            cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += ["-an"] if audio is None else []
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.count = 0

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        assert frame.shape[1] == self.size[0] and frame.shape[0] == self.size[1], \
            f"frame {frame.shape} does not match {self.size}"
        data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(data)
            self.count += 1

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def mux(video: Path, audio: Path, out: Path) -> None:
    """Attach an audio track to a finished silent video."""
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)],
        check=True)


def probe(path: Path) -> str:
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return out.stderr.strip()
