"""Choose varied training routes without changing observations or actions."""

import json
import math
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from ...minecraft.connection import MineflayerBackend
from ...minecraft.server import player_name
from ..q_learning import DIRECTIONS, Profile
from .environment import CompactParkourEnv, EpisodeInfo, Observation

STAGES = ((0, 1), (8192, 2), (24576, 3), (49152, 4), (65536, 5))


class RouteTrainingEnv(CompactParkourEnv):
    def __init__(
        self,
        routes: list[Path],
        backend: MineflayerBackend,
        log_file: TextIO,
        output_folder: Path,
        seed: int,
        watch_player: str | None = None,
    ) -> None:
        self.route_files = routes
        self.courses = [Profile.load(path) for path in routes]
        self.random = random.Random(seed)
        self.training_steps = 0
        self.current_route = 0
        self.evaluation_course: Profile | None = None
        self.evaluation_name = ""
        self.log_file = log_file
        self.output_folder = output_folder
        self.watch_player = player_name(watch_player) if watch_player else None
        super().__init__(self.courses[0], backend=backend, log=self.record)

    def record(self, **fields: object) -> None:
        route = (
            self.evaluation_name
            if self.evaluation_course
            else self.route_files[self.current_route].stem
        )
        self.log_file.write(
            json.dumps(
                dict(mode="eval" if self.evaluation_course else "train", route=route, **fields),
                allow_nan=False,
            )
            + "\n"
        )
        self.log_file.flush()

    def status(self, state: str, **fields: object) -> None:
        temporary = self.output_folder / "status.tmp"
        temporary.write_text(
            json.dumps(
                dict(
                    state=state,
                    process_id=os.getpid(),
                    training_steps=self.training_steps,
                    **fields,
                ),
                indent=2,
            )
            + "\n"
        )
        temporary.replace(self.output_folder / "status.json")

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[Observation, EpisodeInfo]:
        if self.evaluation_course is not None:
            self.profile = self.evaluation_course
        else:
            maximum_jumps = max(jumps for steps, jumps in STAGES if self.training_steps >= steps)
            eligible = [
                index
                for index, course in enumerate(self.courses)
                if len(course.platforms) - 1 <= maximum_jumps
            ]
            self.current_route = self.random.choice(eligible)
            course = self.courses[self.current_route]
            x, y, z = course.start
            self.profile = replace(
                course,
                start=(
                    x + self.random.choice((-0.1, 0.0, 0.1)),
                    y,
                    z + self.random.choice((-0.1, 0.0, 0.1)),
                ),
            )
        observation, info = super().reset(seed=seed, options=options)
        if self.watch_player and self.backend:
            # a stationary side view makes the full route visible during each attempt
            first, last = self.profile.platforms[0], self.profile.platforms[-1]
            forward_x, forward_z = DIRECTIONS[int(self.profile.yaw) % 360]
            side_x, side_z = forward_z, -forward_x
            distance = max(12.0, math.dist(first, last) + 5)
            camera_x = (first[0] + last[0]) / 2 + side_x * distance
            camera_z = (first[2] + last[2]) / 2 + side_z * distance
            camera_y = max(position[1] for position in self.profile.platforms) + 5
            camera_yaw = math.degrees(math.atan2(side_x, -side_z)) % 360
            self.backend.execute(f"execute as {self.watch_player} run spectate")
            self.backend.execute(
                f"tp {self.watch_player} {camera_x} {camera_y} {camera_z} {camera_yaw} 28"
            )
            route_name = (
                self.evaluation_name
                if self.evaluation_course
                else self.route_files[self.current_route].stem
            )
            label = f"{'EVALUATION' if self.evaluation_course else 'TRAINING'} | {route_name} | {len(self.profile.platforms) - 1} jumps"
            self.backend.execute(
                f"title {self.watch_player} actionbar "
                + json.dumps({"text": label, "color": "gold"})
            )
        return observation, info

    def step(self, action: int) -> tuple[Observation, float, bool, bool, EpisodeInfo]:
        result = super().step(action)
        if self.evaluation_course is None:
            self.training_steps += 1
        _, _, terminated, truncated, info = result
        if terminated or truncated:
            summary = info["summary"]
            if not isinstance(summary, dict):
                raise RuntimeError("Completed episodes must include their results")
            self.status(
                "evaluating" if self.evaluation_course else "training",
                route=self.evaluation_name
                if self.evaluation_course
                else self.route_files[self.current_route].stem,
                episode=self.episode,
                **summary,
            )
        return result
