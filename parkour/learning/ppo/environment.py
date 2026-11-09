"""The Minecraft episode interface used by PPO."""

import math
import time
from collections.abc import Callable

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from ...minecraft.connection import MineflayerBackend
from ..contacts import action_frames, detect_contacts
from ..q_learning import (
    ACTIONS,
    BINS,
    BUTTONS,
    EpisodeSummary,
    Profile,
    landed_platform,
    reward_for,
    state_key,
)
from .observations import CONTEXT_FIELDS, context_observation

Observation = NDArray[np.float32]

EpisodeInfo = dict[str, object]


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
    """Minecraft episode loop, using the original seven-bin observations by default."""

    def __init__(
        self,
        profile: Profile,
        *,
        backend: MineflayerBackend | None = None,
        backend_factory: Callable[[], MineflayerBackend] | None = None,
        ticks: int = 2,
        shaped: bool = False,
        contextual: bool = False,
        track_contacts: bool = False,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.backend = backend
        try:
            if not 2 <= len(profile.platforms) <= 6:
                raise ValueError("Compact PPO supports only one to five jumps")
            if type(ticks) is not int or not 1 <= ticks <= 20:
                raise ValueError("ticks must be an integer in 1..20")
            self.profile, self.ticks, self.shaped = profile, ticks, shaped
            self.contextual, self.track_contacts = contextual, track_contacts
            self.backend_factory = backend_factory
            self.log = log or (lambda **fields: None)
            fields = len(CONTEXT_FIELDS) if contextual else 7
            self.observation_space = gym.spaces.Box(-1, 1, (fields,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(len(BUTTONS))
            self.episode, self.active = 0, False
        except BaseException:
            self.close()
            raise

    def observation(self) -> Observation:
        if self.contextual:
            return context_observation(self.profile, self.frame, self.highest, self.steps)
        return encode(state_key(self.profile, self.frame, self.highest))

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[Observation, EpisodeInfo]:
        super().reset(seed=seed)
        try:
            if self.backend is None:
                if self.backend_factory is None:
                    raise ValueError("A backend or backend_factory is required")
                self.backend = self.backend_factory()
            course = self.profile
            frame = self.backend.reset(course.start, course.yaw)
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
            return self.observation(), {"highest_platform": 0, "progress": 0.0}
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
            before_observation = self.observation()
            course, action = self.profile, int(action)
            after = backend.step(BUTTONS[action], self.ticks)
            if after["elapsed_ticks"] != self.ticks or after["correction_count"] != self.correction:
                raise RuntimeError(
                    "Unexpected action duration or server correction; transition discarded"
                )
            previous = self.highest
            frames = action_frames(
                after, self.ticks, self.correction, track_contacts=self.track_contacts
            )
            contacts = detect_contacts(course.platforms, course.fall_y, frames, self.highest)
            self.highest, landed, outcome = (
                contacts.highest_platform,
                contacts.landed_platform,
                contacts.outcome,
            )
            self.steps += 1
            if outcome == "running" and self.steps == course.max_steps:
                outcome = "timeout"
            terminated, truncated = outcome in ("goal", "fall"), outcome == "timeout"
            reward = reward_for(outcome, landed, previous, len(course.platforms) - 1, self.shaped)
            self.frame = after
            next_observation = self.observation()
            self.total += reward
            self.client_ticks += after["elapsed_ticks"]
            progress = self.highest / (len(course.platforms) - 1)
            info: EpisodeInfo = dict(
                outcome=outcome,
                highest_platform=self.highest,
                progress=progress,
            )
            self.log(
                event="transition",
                episode=self.episode,
                step=self.steps - 1,
                action=ACTIONS[action],
                observation=before_observation.tolist(),
                next_observation=next_observation.tolist(),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                outcome=outcome,
            )
            if terminated or truncated:
                self.active = False
                summary: EpisodeSummary = {
                    "outcome": outcome,
                    "success": outcome == "goal",
                    "return": self.total,
                    "max_platform": self.highest,
                    "progress": progress,
                    "steps": self.steps,
                    "seconds": time.monotonic() - self.started,
                    "client_ticks": self.client_ticks,
                }
                info["summary"] = summary
                self.log(event="episode", episode=self.episode, **summary)
            return next_observation, reward, terminated, truncated, info
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.active = False
        if self.backend is not None:
            self.backend.close()
            self.backend = None
