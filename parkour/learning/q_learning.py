"""Tabular Q-learning for a small Minecraft parkour course."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from ..minecraft.connection import MineflayerBackend

ACTIONS = ("idle", "walk", "run", "walk_jump", "run_jump")
BUTTONS = ((0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (1, 0, 0, 1, 0), (1, 0, 1, 0, 0), (1, 0, 1, 1, 0))
ALPHA, GAMMA = 0.1, 0.99
DIRECTIONS = {0: (0, 1), 90: (-1, 0), 180: (0, -1), 270: (1, 0)}
# Each complete tuple identifies one table row. Nearby rows share no weights.
BINS = {
    "forward_distance": (0.25, -2, 8),
    "lateral_distance": (0.5, -2, 2),
    "height": (0.5, -4, 4),
    "forward_velocity": (0.05, -0.4, 0.4),
    "vertical_velocity": (0.1, -1.2, 0.6),
}
GRID = {"radius": 0, "down": 0, "up": 0}  # Transport needs a grid; it is not a learning input.
SCHEMA = {
    "version": 3,
    "algorithm": "tabular-q",
    "actions": list(ACTIONS),
    "state_fields": [*BINS, "grounded", "highest_platform"],
    "gamma": GAMMA,
    "timeout_bootstrap": True,
    "bins": {key: list(value) for key, value in BINS.items()},
}


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    return isinstance(value, list) and len(value) == 3 and all(number(v) for v in value)


@dataclass(frozen=True)
class Profile:
    start: tuple
    yaw: float
    platforms: tuple
    fall_y: float
    max_steps: int
    script: tuple

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or not point(data.get("start")):
            raise ValueError("Course requires three finite start coordinates")
        platforms = data.get("platforms")
        if (
            not isinstance(platforms, list)
            or len(platforms) < 2
            or not all(point(p) for p in platforms)
        ):
            raise ValueError("Course requires start and landing platform centers at feet height")
        yaw, fall = data.get("yaw"), data.get("fall_y")
        if not number(yaw) or yaw % 90 != 0 or not number(fall):
            raise ValueError("Use finite cardinal yaw and fall_y")
        steps = data.get("max_steps", 80)
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("max_steps must be positive")
        script = data.get("script", [])
        if not isinstance(script, list) or any(
            type(a) is not int or not 0 <= a < len(ACTIONS) for a in script
        ):
            raise ValueError("script must contain action indices 0..4")
        dx, dz = DIRECTIONS[int(yaw) % 360]
        for before, after in pairwise(platforms):
            x, z = after[0] - before[0], after[2] - before[2]
            if x * dx + z * dz <= 0 or abs(-x * dz + z * dx) > 0.01:
                raise ValueError("Platforms must follow a straight lane in the facing direction")
        start = data["start"]
        if not on_platform(start, platforms[0]) or fall >= min(p[1] for p in platforms):
            raise ValueError("Start must be on platform zero and fall_y below all platforms")
        return cls(
            tuple(start),
            float(yaw),
            tuple(map(tuple, platforms)),
            float(fall),
            steps,
            tuple(script),
        )


def on_platform(position, platform):
    return (
        abs(position[0] - platform[0]) <= 0.5
        and abs(position[2] - platform[2]) <= 0.5
        and abs(position[1] - platform[1]) <= 0.15
    )


def landed_platform(profile, frame):
    if not frame["on_ground"]:
        return None
    return next(
        (
            i
            for i, platform in reversed(list(enumerate(profile.platforms)))
            if on_platform(frame["position"], platform)
        ),
        None,
    )


def state_key(profile, frame, highest):
    target = profile.platforms[min(highest + 1, len(profile.platforms) - 1)]
    x, y, z = frame["position"]
    vx, vy, vz = frame["velocity"]
    # Exact cardinal vectors keep roundoff such as cos(90°) from creating a false sideways bin.
    dx, dz = DIRECTIONS[int(profile.yaw) % 360]
    values = (
        (target[0] - x) * dx + (target[2] - z) * dz,
        -(target[0] - x) * dz + (target[2] - z) * dx,
        target[1] - y,
        vx * dx + vz * dz,
        vy,
    )
    bins = tuple(
        math.floor(min(high, max(low, value)) / width)
        for value, (width, low, high) in zip(values, BINS.values(), strict=True)
    )
    return bins + (int(frame["on_ground"]), highest)


def choose(table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.randrange(len(ACTIONS))
    row = table.get(state, [0.0] * len(ACTIONS))
    return rng.choice([i for i, value in enumerate(row) if value == max(row)])


def update_q(table, state, action, reward, next_state, terminated):
    future = 0.0 if terminated else max(table.get(next_state, [0.0] * len(ACTIONS)))
    row = table.setdefault(state, [0.0] * len(ACTIONS))
    row[action] += ALPHA * (reward + GAMMA * future - row[action])


def reward_for(outcome, landed, previous_highest, final_index, shaped=False):
    if outcome == "goal":
        return 1.0
    if outcome == "fall":
        return -1.0
    bonus = shaped and landed is not None and previous_highest < landed < final_index
    return -0.01 + (0.1 if bonus else 0.0)


def save_table(path, table, shaped, ticks=2):
    payload = {
        "schema": SCHEMA,
        "reward_mode": "platform-bonus" if shaped else "sparse",
        "ticks": ticks,
        "q": {",".join(map(str, state)): row for state, row in table.items()},
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_table(path, shaped, ticks=2):
    saved = json.loads(path.read_text())
    if (
        not isinstance(saved, dict)
        or saved.get("schema") != SCHEMA
        or saved.get("ticks") != ticks
        or saved.get("reward_mode") != ("platform-bonus" if shaped else "sparse")
    ):
        raise ValueError("Checkpoint is not a compatible tabular Q-table/reward condition")
    rows = saved.get("q")
    if not isinstance(rows, dict):
        raise ValueError("Checkpoint lacks Q-table rows")
    table = {}
    for key, row in rows.items():
        state = tuple(int(part) for part in key.split(","))
        if (
            len(state) != 7
            or not isinstance(row, list)
            or len(row) != len(ACTIONS)
            or not all(number(v) for v in row)
        ):
            raise ValueError("Invalid Q-table state or action row")
        table[state] = row
    return table


def validate_frame(frame):
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
    return frame


class TabularRunner:
    def __init__(self, profile, backend, *, ticks=2, shaped=False):
        self.profile, self.backend, self.ticks, self.shaped = profile, backend, ticks, shaped

    def episode(self, table, rng, epsilon, mode, log):
        p = self.profile
        frame = validate_frame(
            self.backend._request("parkour_reset", start=list(p.start), yaw=p.yaw, grid=GRID)
        )
        if landed_platform(p, frame) != 0 or math.dist(frame["position"], p.start) > 0.15:
            raise RuntimeError("Reset did not return to the configured starting platform")
        correction = frame["correction_count"]
        highest = 0
        state = state_key(p, frame, highest)
        total, tick_total, started = 0.0, 0, time.monotonic()
        for step in range(p.max_steps):
            action = (
                (p.script[step] if step < len(p.script) else 0)
                if mode == "scripted"
                else choose(table, state, epsilon, rng)
            )
            after = validate_frame(
                self.backend._request(
                    "parkour_step", action=list(BUTTONS[action]), ticks=self.ticks, grid=GRID
                )
            )
            if after["elapsed_ticks"] != self.ticks or after["correction_count"] != correction:
                raise RuntimeError(
                    "Unexpected action duration or server correction; transition discarded"
                )
            landed = landed_platform(p, after)
            old_highest = highest
            highest = max(highest, landed if landed is not None else 0)
            outcome = (
                "fall"
                if after["health"] <= 0 or after["position"][1] < p.fall_y
                else "goal"
                if landed == len(p.platforms) - 1
                else "timeout"
                if step + 1 == p.max_steps
                else "running"
            )
            terminated = outcome in ("goal", "fall")
            truncated = outcome == "timeout"
            episode_end = terminated or truncated
            reward = reward_for(outcome, landed, old_highest, len(p.platforms) - 1, self.shaped)
            next_state = state_key(p, after, highest)
            if mode == "train":
                update_q(table, state, action, reward, next_state, terminated)
            total += reward
            tick_total += after["elapsed_ticks"]
            log(
                event="transition",
                step=step,
                state=list(state),
                action=ACTIONS[action],
                before={key: frame[key] for key in ("position", "velocity", "on_ground")},
                after={key: after[key] for key in ("position", "velocity", "on_ground")},
                next_state=list(next_state),
                reward=reward,
                terminal=terminated,
                terminated=terminated,
                truncated=truncated,
                episode_end=episode_end,
                outcome=outcome,
                highest_platform=highest,
                elapsed_ticks=after["elapsed_ticks"],
            )
            if episode_end:
                return {
                    "outcome": outcome,
                    "success": outcome == "goal",
                    "return": total,
                    "max_platform": highest,
                    "progress": highest / (len(p.platforms) - 1),
                    "steps": step + 1,
                    "seconds": time.monotonic() - started,
                    "client_ticks": tick_total,
                }
            frame, state = after, next_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--shaped", action="store_true")
    parser.add_argument("--resume", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--eval", action="store_true")
    modes.add_argument("--random", action="store_true")
    modes.add_argument("--scripted", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25566)
    parser.add_argument("--username", default="Watcher")
    parser.add_argument("--version", default="26.1.2")
    args = parser.parse_args(argv)
    if (
        args.episodes < 1
        or args.seed < 0
        or not 1 <= args.ticks <= 20
        or not 1 <= args.port <= 65535
        or not all(math.isfinite(v) and 0 <= v <= 1 for v in (args.epsilon_start, args.epsilon_end))
    ):
        parser.error(
            "Use positive episodes, nonnegative seed, valid port/ticks and epsilon in [0,1]"
        )
    mode = (
        "eval"
        if args.eval
        else "random"
        if args.random
        else "scripted"
        if args.scripted
        else "train"
    )
    if args.resume and mode != "train":
        parser.error("--resume is only for training")
    profile = Profile.load(args.course)
    if mode == "scripted" and not profile.script:
        parser.error("Course has no preset script")
    checkpoint, logfile = args.output.with_suffix(".json"), args.output.with_suffix(".jsonl")
    if mode == "eval" or args.resume:
        table = load_table(checkpoint, args.shaped, args.ticks)
    else:
        if mode == "train" and checkpoint.exists():
            raise FileExistsError("Existing table: use --resume or a fresh output prefix")
        table = {}
    logfile.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with logfile.open("a") as handle:

        def log(**fields):
            handle.write(
                json.dumps(
                    {
                        "mode": mode,
                        "seed": args.seed,
                        "reward_mode": "platform-bonus" if args.shaped else "sparse",
                        **fields,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()

        log(
            event="run_config",
            course=str(args.course.resolve()),
            profile=profile.__dict__,
            episodes=args.episodes,
            ticks=args.ticks,
            alpha=ALPHA,
            gamma=GAMMA,
            schema=SCHEMA,
        )
        backend = MineflayerBackend(args.host, args.port, args.username, args.version)
        counts = {"goal": 0, "fall": 0, "timeout": 0}
        try:
            runner = TabularRunner(profile, backend, ticks=args.ticks, shaped=args.shaped)
            for episode in range(args.episodes):
                epsilon = (
                    args.epsilon_start
                    + (args.epsilon_end - args.epsilon_start) * episode / max(1, args.episodes - 1)
                    if mode == "train"
                    else 1.0
                    if mode == "random"
                    else 0.0
                )
                summary = runner.episode(
                    table,
                    rng,
                    epsilon,
                    mode,
                    lambda episode=episode, **fields: log(episode=episode + 1, **fields),
                )
                counts[summary["outcome"]] += 1
                log(
                    event="episode",
                    episode=episode + 1,
                    epsilon=epsilon,
                    states=len(table),
                    **summary,
                )
                if mode == "train":
                    save_table(checkpoint, table, args.shaped, args.ticks)
                if (episode + 1) % 10 == 0 or summary["success"]:
                    print(
                        f"{episode + 1}/{args.episodes} {summary['outcome']} | {counts} states {len(table)}",
                        flush=True,
                    )
            log(event="done", **counts)
            print(json.dumps(counts), flush=True)
        finally:
            try:
                if mode == "train":
                    save_table(checkpoint, table, args.shaped, args.ticks)
            finally:
                backend.close()


if __name__ == "__main__":
    main()
