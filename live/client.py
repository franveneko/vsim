"""Client for the Node bridge that owns the real Minecraft bot.

The bridge speaks newline-delimited JSON over a local TCP socket.  Each step is
a request and a reply, so the Python side never has to guess whether the bot has
finished swinging.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MCREAL = Path(__file__).resolve().parent.parent / "mcreal"


@dataclass
class Step:
    obs: np.ndarray        # (32,) float32, same layout as MineSim.observe()
    bearing: float
    info: dict


class BridgeClient:
    def __init__(self, port: int = 8765, host: str = "127.0.0.1", timeout: float = 60.0):
        self.addr = (host, port)
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._buf = b""

    def connect(self, retries: int = 40, delay: float = 0.5) -> None:
        for _ in range(retries):
            try:
                s = socket.create_connection(self.addr, timeout=self.timeout)
                s.settimeout(self.timeout)
                self.sock = s
                return
            except OSError:
                time.sleep(delay)
        raise ConnectionError(f"bridge not reachable at {self.addr}")

    def _request(self, msg: dict) -> dict:
        assert self.sock is not None, "not connected"
        self.sock.sendall((json.dumps(msg) + "\n").encode())
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("bridge closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line)

    def reset(self) -> Step:
        return self._wrap(self._request({"cmd": "reset"}))

    def seed_world(self, trees: int = 10) -> Step:
        """Plant oak logs around spawn (the JS server has no tree generation)."""
        return self._wrap(self._request({"cmd": "seedworld", "trees": trees}))

    def step(self, action: int) -> Step:
        return self._wrap(self._request({"cmd": "step", "a": int(action)}))

    @staticmethod
    def _wrap(reply: dict) -> Step:
        return Step(obs=np.asarray(reply["obs"], np.float32),
                    bearing=float(reply["bearing"]), info=reply["info"])

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.sendall(b'{"cmd":"quit"}\n')
            except OSError:
                pass
            self.sock.close()
            self.sock = None


def launch_server(port: int = 25565, version: str = "1.18.2", seed: int = 42,
                  world: str = "world", log: Path | None = None) -> subprocess.Popen:
    out = open(log, "w") if log else subprocess.DEVNULL
    return subprocess.Popen(
        ["node", "server.js", f"port={port}", f"version={version}",
         f"seed={seed}", f"world={world}"],
        cwd=MCREAL, stdout=out, stderr=subprocess.STDOUT)


def launch_bridge(host: str = "127.0.0.1", port: int = 25565, bridge_port: int = 8765,
                  username: str = "fly0", version: str = "1.18.2",
                  auth: str = "offline", log: Path | None = None) -> subprocess.Popen:
    """`auth="microsoft"` signs in with a real Minecraft account, which an
    online-mode server requires; the first run prints a device-code prompt."""
    out = open(log, "w") if log else subprocess.DEVNULL
    return subprocess.Popen(
        ["node", "bridge.js", f"host={host}", f"port={port}",
         f"bridgePort={bridge_port}", f"username={username}", f"version={version}",
         f"auth={auth}"],
        cwd=MCREAL, stdout=out, stderr=subprocess.STDOUT)
