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


class Subtitles:
    """Collects the on-screen caption per frame and writes an uploadable SRT."""

    def __init__(self, fps: int = 30):
        self.fps = fps
        self.spans: list[list] = []

    def mark(self, frame_index: int, text: str) -> None:
        text = (text or "").strip()
        if self.spans and self.spans[-1][2] == text and self.spans[-1][1] == frame_index:
            self.spans[-1][1] = frame_index + 1
            return
        if text:
            self.spans.append([frame_index, frame_index + 1, text])

    @staticmethod
    def _stamp(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def write(self, path: Path) -> Path:
        path = Path(path)
        lines = []
        for i, (a, b, text) in enumerate(self.spans, 1):
            lines += [str(i),
                      f"{self._stamp(a / self.fps)} --> {self._stamp(b / self.fps)}",
                      text, ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
