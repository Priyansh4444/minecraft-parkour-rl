"""Run a local server, optionally attaching a Minecraft spectator to the bot."""

from __future__ import annotations

import argparse
import re
import signal
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path


def player_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]{1,16}", value):
        raise ValueError("Minecraft names must contain 1–16 letters, digits or underscores")
    return value


@dataclass(frozen=True)
class ServerOptions:
    directory: Path
    java: str = "java"
    port: int = 25566
    bot_name: str = "Watcher"
    watch_player: str | None = None


class Spectator:
    """Attach only the named viewer once both players have joined."""

    def __init__(self, viewer: str, bot: str):
        self.viewer, self.bot = player_name(viewer), player_name(bot)
        if viewer.casefold() == bot.casefold():
            raise ValueError("The spectator and training bot must have different names")
        self.online: set[str] = set()
        self.attached = False

    def on_log(self, line: str) -> tuple[str, ...]:
        event = re.search(
            r"\[Server thread/INFO\]: ([A-Za-z0-9_]{1,16}) (joined|left) the game$", line.strip()
        )
        if not event:
            return ()
        name, action = event.groups()
        name = name.casefold()
        if action == "joined":
            self.online.add(name)
        else:
            self.online.discard(name)
            if name in (self.viewer.casefold(), self.bot.casefold()):
                self.attached = False
        ready = {self.viewer.casefold(), self.bot.casefold()} <= self.online
        if ready and not self.attached:
            self.attached = True
            return (
                f"gamemode spectator {self.viewer}",
                f"tp {self.viewer} {self.bot}",
                f"spectate {self.bot} {self.viewer}",
            )
        return ()


def validate(options: ServerOptions) -> None:
    player_name(options.bot_name)
    if options.watch_player is not None:
        Spectator(options.watch_player, options.bot_name)
    if not 1 <= options.port <= 65535:
        raise ValueError("Server port must be in 1..65535")
    if not (options.directory / "server.jar").is_file():
        raise FileNotFoundError(f"No server.jar in {options.directory}")
    properties = {}
    for line in (options.directory / "server.properties").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            properties[key.strip()] = value.strip()
    if properties.get("server-ip") not in ("127.0.0.1", "localhost"):
        raise ValueError("Set server-ip=127.0.0.1 in server.properties for this local experiment")
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", options.port)) == 0:
            raise RuntimeError(
                f"Port {options.port} is occupied; leave the existing server running"
            )


def run_server(options: ServerOptions) -> int:
    validate(options)
    watcher = Spectator(options.watch_player, options.bot_name) if options.watch_player else None
    command = [
        options.java,
        "-Xms512M",
        "-Xmx2G",
        "-jar",
        "server.jar",
        "nogui",
        "--port",
        str(options.port),
    ]
    process = subprocess.Popen(
        command,
        cwd=options.directory,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    input_stream, output_stream = process.stdin, process.stdout
    assert input_stream is not None and output_stream is not None  # both use subprocess.pipe.

    def send(command: str) -> None:
        input_stream.write(command + "\n")
        input_stream.flush()

    ready = False
    try:
        print("Server starting. Ctrl+C saves and stops this server.", flush=True)
        for line in output_stream:
            print(line, end="", flush=True)
            if not ready and "Done (" in line:
                ready = True
                print(
                    f"Connect Minecraft Java to 127.0.0.1:{options.port} to watch."
                    if watcher
                    else "Headless server ready; start the learner in another terminal.",
                    flush=True,
                )
            # wait for the real login so offline names resolve to the correct profile
            joined = re.search(
                r"\[Server thread/INFO\]: ([A-Za-z0-9_]{1,16}) joined the game$", line.strip()
            )
            if joined and joined.group(1) == options.bot_name:
                send(f"op {options.bot_name}")
            if watcher:
                for instruction in watcher.on_log(line):
                    send(instruction)
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if process.poll() is None:
            try:
                send("stop")
                process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        input_stream.close()
        output_stream.close()


def stop_requested(signum, frame) -> None:
    """Let ordinary process termination use the same save/stop path as Ctrl+C."""
    raise KeyboardInterrupt()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, required=True, help="prepared server folder containing server.jar"
    )
    parser.add_argument("--java", default="java", help="Java 25 executable")
    parser.add_argument("--port", type=int, default=25566)
    parser.add_argument("--bot-name", default="Watcher")
    parser.add_argument(
        "--watch-player",
        help="Minecraft username to put in spectator mode; the game client joins separately",
    )
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, stop_requested)
    options = ServerOptions(
        args.directory.expanduser().resolve(),
        args.java,
        args.port,
        args.bot_name,
        args.watch_player,
    )
    try:
        code = run_server(options)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
