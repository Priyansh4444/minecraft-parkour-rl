"""Mineflayer transport for the local Q-learning and compact PPO experiments."""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace


class MineflayerBackend:
    """Read state and control a local Mineflayer bot over JSON RPC."""

    def __init__(self, host: str, port: int, username: str, version: str):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("This adapter only connects to a local Minecraft server")
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Install Node.js 22 or newer, then run npm ci in bridge/")
        bridge = Path(__file__).resolve().parents[2] / "bridge" / "headless.mjs"
        self.process = subprocess.Popen(
            [node, str(bridge)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        input_stream, output_stream = self.process.stdin, self.process.stdout
        assert input_stream is not None and output_stream is not None  # Both use subprocess.PIPE.
        self.input_stream, self.output_stream = input_stream, output_stream
        self.responses = queue.Queue()
        self.request_id = 0
        self.closed = False
        threading.Thread(target=self._read_responses, daemon=True).start()
        try:
            self._request("connect", host=host, port=port, username=username, version=version)
        except BaseException:
            self.close()
            raise

    def _read_responses(self) -> None:
        try:
            for line in self.output_stream:
                try:
                    self.responses.put(json.loads(line))
                except ValueError:
                    self.responses.put(RuntimeError(f"Invalid bridge response: {line.rstrip()}"))
        finally:
            self.responses.put(RuntimeError("Mineflayer bridge exited; inspect its error output"))

    def _request(self, operation: str, **fields):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("Mineflayer bridge is not running")
        self.request_id += 1
        request = {"id": self.request_id, "op": operation, **fields}
        try:
            self.input_stream.write(json.dumps(request) + "\n")
            self.input_stream.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("Lost connection to the Mineflayer bridge") from error
        try:
            response = self.responses.get(timeout=20)
        except queue.Empty as error:
            raise RuntimeError(f"Mineflayer operation timed out: {operation}") from error
        if isinstance(response, Exception):
            raise response
        if response.get("id") != self.request_id:
            raise RuntimeError("Mineflayer responses arrived out of order")
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def player(self):
        state = self._request("snapshot")
        return SimpleNamespace(**state)

    def player_position(self):
        return self.player().position

    def player_health(self):
        return self.player().health

    def world_info(self):
        return SimpleNamespace(game_ticks=self.player().tick)

    def get_block_list(self, positions):
        return self._request("blocks", positions=positions)

    def getblock(self, x, y, z):
        return self.get_block_list([[x, y, z]])[0]

    def player_press_forward(self, pressed):
        self._request("controls", keys={"forward": pressed})

    def player_press_sprint(self, pressed):
        self._request("controls", keys={"sprint": pressed})

    def player_press_jump(self, pressed):
        self._request("controls", keys={"jump": pressed})

    def player_set_orientation(self, yaw, pitch):
        self._request("look", yaw=yaw, pitch=pitch)

    def execute(self, command):
        self._request("command", command=command)

    @staticmethod
    def echo(message):
        print(message, flush=True)

    def close(self):
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                self._request("close")
        except Exception:
            pass
        finally:
            self.closed = True
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
