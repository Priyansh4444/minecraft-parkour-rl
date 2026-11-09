"""Mineflayer transport for the local Q-learning and compact PPO experiments."""

from __future__ import annotations

import json
import math
import queue
import shutil
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict, cast


class PhysicsFrame(TypedDict):
    """One measured bot state received from the TypeScript bridge."""

    position: list[float]
    velocity: list[float]
    on_ground: bool
    health: float
    tick: int
    start_tick: int
    elapsed_ticks: int
    correction_count: int
    jump_ticks: NotRequired[int]
    tick_frames: NotRequired[list[PhysicsFrame]]


def number(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def point(value: object) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(number(v) for v in value)


def validate_frame(frame: object) -> PhysicsFrame:
    if (
        not isinstance(frame, dict)
        or not point(frame.get("position"))
        or not point(frame.get("velocity"))
    ):
        raise ValueError("Missing or invalid player position/velocity")
    if not isinstance(frame.get("on_ground"), bool) or not number(frame.get("health")):
        raise ValueError("Invalid ground/health observation")
    for name in ("tick", "start_tick", "elapsed_ticks", "correction_count"):
        if type(frame.get(name)) is not int or frame[name] < 0:
            raise ValueError(f"Invalid {name}")
    if frame["tick"] - frame["start_tick"] != frame["elapsed_ticks"]:
        raise ValueError("Inconsistent client tick counters")
    return cast(PhysicsFrame, frame)


class BridgeError(RuntimeError):
    """The bot connection could not complete a request."""


@dataclass(frozen=True)
class BridgeReply:
    request_id: int
    result: object


def parse_reply(line: str) -> BridgeReply:
    reply = json.loads(line)
    if not isinstance(reply, dict) or ("result" in reply) == ("error" in reply):
        raise BridgeError("The bot reply needs exactly one result or error")
    if "error" in reply:
        if not isinstance(reply["error"], str):
            raise BridgeError("The bot error must be text")
        raise BridgeError(reply["error"])
    if type(reply.get("id")) is not int:
        raise BridgeError("The bot reply is missing an integer request ID")
    return BridgeReply(reply["id"], reply["result"])


class MineflayerBackend:
    """Read state and control a local Mineflayer bot over JSON RPC."""

    def __init__(self, host: str, port: int, username: str, version: str) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("This adapter only connects to a local Minecraft server")
        node = shutil.which("node")
        if node is None:
            raise FileNotFoundError("Install Node.js 22.18 or newer, then run npm ci in bridge/")
        bridge = Path(__file__).resolve().parents[2] / "bridge" / "headless.ts"
        self.process = subprocess.Popen(
            [node, str(bridge)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        input_stream, output_stream = self.process.stdin, self.process.stdout
        assert input_stream is not None and output_stream is not None  # both use subprocess.pipe.
        self.input_stream, self.output_stream = input_stream, output_stream
        self.responses: queue.Queue[str | OSError | None] = queue.Queue()
        self.request_id = 0
        self.closed = False
        self.reader = threading.Thread(target=self._read_responses, daemon=True)
        self.reader.start()
        try:
            self._request("connect", host=host, port=port, username=username, version=version)
        except BaseException:
            self.close()
            raise

    def _read_responses(self) -> None:
        try:
            for line in self.output_stream:
                self.responses.put(line)
        except OSError as error:
            self.responses.put(error)
        finally:
            self.responses.put(None)

    def _request(self, operation: str, **fields: object) -> object:
        if self.closed or self.process.poll() is not None:
            self.close()
            raise BridgeError(f"{operation}: the bot connection is closed")
        self.request_id += 1
        request = {"id": self.request_id, "op": operation, **fields}
        try:
            self.input_stream.write(json.dumps(request) + "\n")
            self.input_stream.flush()
            line = self.responses.get(timeout=20)
            if isinstance(line, OSError):
                raise line
            if line is None:
                raise BridgeError("The bot process exited; inspect its error output")
            response = parse_reply(line)
            if response.request_id != self.request_id:
                raise BridgeError(
                    f"Expected reply {self.request_id}, received {response.request_id}"
                )
            return response.result
        except (BridgeError, OSError, json.JSONDecodeError, queue.Empty) as error:
            self.close()
            reason = (
                "no bot reply after 20 seconds" if isinstance(error, queue.Empty) else str(error)
            )
            raise BridgeError(f"{operation}: {reason}") from error

    def reset(self, start: tuple[float, float, float], yaw: float) -> PhysicsFrame:
        return validate_frame(self._request("parkour_reset", start=list(start), yaw=yaw))

    def step(self, buttons: tuple[int, int, int, int, int], ticks: int) -> PhysicsFrame:
        return validate_frame(self._request("parkour_step", action=list(buttons), ticks=ticks))

    def execute(self, command: str) -> None:
        self._request("command", command=command)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        # closing stdin lets the bridge disconnect the bot, even after a failed request
        with suppress(OSError):
            self.input_stream.close()
        with suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=3)
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.reader.join()
        self.output_stream.close()
