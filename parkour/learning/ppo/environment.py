"""The Minecraft episode interface used by PPO."""

import math
import time
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from ...minecraft.connection import MineflayerBackend
from ..q_learning import (
    ACTIONS,
    BINS,
    BUTTONS,
    GRID,
    Profile,
    landed_platform,
    reward_for,
    state_key,
    validate_frame,
)

Observation = NDArray[np.float32]
EpisodeInfo = dict[str, Any]

LOW = np.array([math.floor(low / width) for width, low, high in BINS.values()] + [0, 0])
HIGH = np.array([math.floor(high / width) for width, low, high in BINS.values()] + [1, 5])


def encode(state: tuple[int, ...]) -> Observation:
    """Invertible scaling of the seven integer bins; no additional information."""
    raw = np.asarray(state)
    if (
        raw.shape != (7,)
        or not np.isfinite(raw).all()
        or np.any(raw != np.floor(raw))
        or np.any(raw < LOW)
        or np.any(raw > HIGH)
    ):
        raise ValueError("State is outside the seven-bin observation bounds")
    return (2 * (raw - LOW) / (HIGH - LOW) - 1).astype(np.float32)


class CompactParkourEnv(gym.Env):
    """Gymnasium environment with the same states, actions and rewards as Q-learning."""

    def __init__(
        self,
        profile: Profile,
        *,
        backend: MineflayerBackend | None = None,
        backend_factory: Callable[[], MineflayerBackend] | None = None,
        ticks: int = 2,
        shaped: bool = False,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.backend = backend
        try:
            if not 2 <= len(profile.platforms) <= 6:
                raise ValueError("Compact PPO supports only one to five jumps")
            if type(ticks) is not int or not 1 <= ticks <= 20:
                raise ValueError("ticks must be an integer in 1..20")
            self.profile, self.ticks, self.shaped = profile, ticks, shaped
            self.backend_factory = backend_factory
            self.log = log or (lambda **fields: None)
            self.observation_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(len(BUTTONS))
            self.episode, self.active = 0, False
        except BaseException:
            self.close()
            raise

    def reset(
        self, *, seed: int | None = None, options: EpisodeInfo | None = None
    ) -> tuple[Observation, EpisodeInfo]:
        super().reset(seed=seed)
        try:
            if self.backend is None:
                if self.backend_factory is None:
                    raise ValueError("A backend or backend_factory is required")
                self.backend = self.backend_factory()
            course = self.profile
            frame = validate_frame(
                self.backend._request(
                    "parkour_reset", start=list(course.start), yaw=course.yaw, grid=GRID
                )
            )
            if (
                landed_platform(course, frame) != 0
                or math.dist(frame["position"], course.start) > 0.15
            ):
                raise RuntimeError("Reset did not return to the configured starting platform")
            self.frame, self.highest, self.steps = frame, 0, 0
            self.correction = frame["correction_count"]
            self.total, self.client_ticks, self.started = 0.0, 0, time.monotonic()
            self.episode += 1
            self.active = True
            self.state = state_key(course, frame, 0)
            return encode(self.state), {"state": self.state, "highest_platform": 0, "progress": 0.0}
        except BaseException:
            self.close()
            raise

    def step(self, action: int) -> tuple[Observation, float, bool, bool, EpisodeInfo]:
        if not self.active or not self.action_space.contains(action):
            raise ValueError("Reset before stepping and use an action index in 0..4")
        backend = self.backend
        if backend is None:
            raise RuntimeError("Reset before stepping to establish the Minecraft connection")
        try:
            course, action = self.profile, int(action)
            after = validate_frame(
                backend._request(
                    "parkour_step", action=list(BUTTONS[action]), ticks=self.ticks, grid=GRID
                )
            )
            if after["elapsed_ticks"] != self.ticks or after["correction_count"] != self.correction:
                raise RuntimeError(
                    "Unexpected action duration or server correction; transition discarded"
                )
            landed, previous = landed_platform(course, after), self.highest
            self.highest = max(previous, landed if landed is not None else 0)
            self.steps += 1
            outcome = (
                "fall"
                if after["health"] <= 0 or after["position"][1] < course.fall_y
                else "goal"
                if landed == len(course.platforms) - 1
                else "timeout"
                if self.steps == course.max_steps
                else "running"
            )
            terminated, truncated = outcome in ("goal", "fall"), outcome == "timeout"
            reward = reward_for(outcome, landed, previous, len(course.platforms) - 1, self.shaped)
            next_state = state_key(course, after, self.highest)
            self.total += reward
            self.client_ticks += after["elapsed_ticks"]
            info = dict(
                state=next_state,
                outcome=outcome,
                highest_platform=self.highest,
                progress=self.highest / (len(course.platforms) - 1),
            )
            self.log(
                event="transition",
                episode=self.episode,
                step=self.steps - 1,
                state=list(self.state),
                action=ACTIONS[action],
                next_state=list(next_state),
                before={k: self.frame[k] for k in ("position", "velocity", "on_ground")},
                after={k: after[k] for k in ("position", "velocity", "on_ground")},
                reward=reward,
                terminated=terminated,
                terminal=terminated,
                truncated=truncated,
                episode_end=terminated or truncated,
                outcome=outcome,
                highest_platform=self.highest,
                progress=info["progress"],
                elapsed_ticks=after["elapsed_ticks"],
            )
            if terminated or truncated:
                self.active = False
                summary = dict(
                    outcome=outcome,
                    success=outcome == "goal",
                    **{"return": self.total},
                    max_platform=self.highest,
                    progress=info["progress"],
                    steps=self.steps,
                    seconds=time.monotonic() - self.started,
                    client_ticks=self.client_ticks,
                )
                info["summary"] = summary
                self.log(event="episode", episode=self.episode, **summary)
            self.frame, self.state = after, next_state
            return encode(next_state), reward, terminated, truncated, info
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.active = False
        if self.backend is not None:
            self.backend.close()
            self.backend = None
