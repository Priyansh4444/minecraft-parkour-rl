"""PPO training and evaluation using the same course contract as Q-learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from ..minecraft.connection import MineflayerBackend
from .q_learning import (
    ACTIONS,
    BINS,
    BUTTONS,
    GAMMA,
    GRID,
    SCHEMA,
    Profile,
    landed_platform,
    reward_for,
    state_key,
    validate_frame,
)

LOW = np.array([math.floor(low / width) for width, low, high in BINS.values()] + [0, 0])
HIGH = np.array([math.floor(high / width) for width, low, high in BINS.values()] + [1, 5])


class PPOOptions(TypedDict):
    """The PPO settings saved with each model and passed to Stable-Baselines3."""

    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    learning_rate: float
    clip_range: float
    ent_coef: float
    target_kl: float
    policy_kwargs: dict[str, list[int]]
    device: str


PPO_OPTIONS: PPOOptions = dict(
    n_steps=128,
    batch_size=64,
    n_epochs=4,
    gamma=GAMMA,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,
    target_kl=0.02,
    policy_kwargs={"net_arch": [32, 32]},
    device="cpu",
)


def encode(state):
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


def decode(observation):
    return tuple(np.rint((np.asarray(observation) + 1) * (HIGH - LOW) / 2 + LOW).astype(int))


class CompactParkourEnv(gym.Env):
    """Gymnasium environment with the same states, actions and rewards as Q-learning."""

    def __init__(
        self, profile, *, backend=None, backend_factory=None, ticks=2, shaped=False, log=None
    ):
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        try:
            if self.backend is None:
                if self.backend_factory is None:
                    raise ValueError("A backend or backend_factory is required")
                self.backend = self.backend_factory()
            p = self.profile
            frame = validate_frame(
                self.backend._request("parkour_reset", start=list(p.start), yaw=p.yaw, grid=GRID)
            )
            if landed_platform(p, frame) != 0 or math.dist(frame["position"], p.start) > 0.15:
                raise RuntimeError("Reset did not return to the configured starting platform")
            self.frame, self.highest, self.steps = frame, 0, 0
            self.correction = frame["correction_count"]
            self.total, self.client_ticks, self.started = 0.0, 0, time.monotonic()
            self.episode += 1
            self.active = True
            self.state = state_key(p, frame, 0)
            return encode(self.state), {"state": self.state, "highest_platform": 0, "progress": 0.0}
        except BaseException:
            self.close()
            raise

    def step(self, action):
        if not self.active or not self.action_space.contains(action):
            raise ValueError("Reset before stepping and use an action index in 0..4")
        backend = self.backend
        if backend is None:
            raise RuntimeError("Reset before stepping to establish the Minecraft connection")
        try:
            p, action = self.profile, int(action)
            after = validate_frame(
                backend._request(
                    "parkour_step", action=list(BUTTONS[action]), ticks=self.ticks, grid=GRID
                )
            )
            if after["elapsed_ticks"] != self.ticks or after["correction_count"] != self.correction:
                raise RuntimeError(
                    "Unexpected action duration or server correction; transition discarded"
                )
            landed, previous = landed_platform(p, after), self.highest
            self.highest = max(previous, landed if landed is not None else 0)
            self.steps += 1
            outcome = (
                "fall"
                if after["health"] <= 0 or after["position"][1] < p.fall_y
                else "goal"
                if landed == len(p.platforms) - 1
                else "timeout"
                if self.steps == p.max_steps
                else "running"
            )
            terminated, truncated = outcome in ("goal", "fall"), outcome == "timeout"
            reward = reward_for(outcome, landed, previous, len(p.platforms) - 1, self.shaped)
            next_state = state_key(p, after, self.highest)
            self.total += reward
            self.client_ticks += after["elapsed_ticks"]
            info = dict(
                state=next_state,
                outcome=outcome,
                highest_platform=self.highest,
                progress=self.highest / (len(p.platforms) - 1),
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

    def close(self):
        self.active = False
        if self.backend is not None:
            self.backend.close()
            self.backend = None


def metadata(profile, course, *, ticks=2, shaped=False, seed=11):
    return dict(
        algorithm="ppo-clip",
        version=1,
        tabular_schema=SCHEMA,
        actions=list(ACTIONS),
        state_fields=SCHEMA["state_fields"],
        raw_low=LOW.tolist(),
        raw_high=HIGH.tolist(),
        normalization="2*(integer_bin-low)/(high-low)-1; float32; round to invert",
        reward_mode="platform-bonus" if shaped else "sparse",
        ticks=ticks,
        seed=seed,
        profile=json.loads(json.dumps(asdict(profile))),
        course_sha256=hashlib.sha256(course.read_bytes()).hexdigest(),
        ppo=PPO_OPTIONS,
        torch_threads=1,
        packages={name: version(name) for name in ("gymnasium", "stable-baselines3", "torch")},
    )


def paths(prefix):
    return tuple(prefix.with_suffix(suffix) for suffix in (".zip", ".json", ".jsonl"))


def validate_checkpoint(prefix, expected, *, evaluation=False):
    model_path, meta_path, _ = paths(prefix)
    saved = json.loads(meta_path.read_text())
    ignored = {"profile", "course_sha256"} if evaluation else set()
    if (
        not isinstance(saved, dict)
        or set(saved) != set(expected)
        or {k: v for k, v in saved.items() if k not in ignored}
        != {k: v for k, v in expected.items() if k not in ignored}
        or not model_path.is_file()
    ):
        raise ValueError(
            "Checkpoint is not compatible with compact PPO, this course, seed or reward/tick condition"
        )
    return saved


def save_checkpoint(model, prefix, meta):
    model_path, meta_path, _ = paths(prefix)
    temporary = model_path.with_name(model_path.stem + ".tmp.zip")
    temporary_meta = meta_path.with_name(meta_path.name + ".tmp")
    try:
        model.save(temporary)
        temporary_meta.write_text(json.dumps(meta, allow_nan=False) + "\n")
        temporary.replace(model_path)
        temporary_meta.replace(meta_path)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)


class SaveEveryRollout(BaseCallback):
    def __init__(self, prefix, meta):
        super().__init__()
        self.prefix, self.meta = prefix, meta

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # This occurs after the previous rollout's optimizer update.
        save_checkpoint(self.model, self.prefix, self.meta)


def evaluate(model, env, episodes, seed):
    counts = dict(goal=0, fall=0, timeout=0)
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                counts[info["outcome"]] += 1
                break
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ticks", type=int, default=2)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shaped", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=25567)
    parser.add_argument("--username", default="Watcher")
    parser.add_argument("--version", default="26.1.2")
    args = parser.parse_args(argv)
    if (
        args.steps < 1
        or args.episodes < 1
        or args.seed < 0
        or not 1 <= args.ticks <= 20
        or not 1 <= args.port <= 65535
        or (args.eval and args.resume)
    ):
        parser.error(
            "Use positive steps/episodes, nonnegative seed, valid ticks/port and training-only --resume"
        )
    profile = Profile.load(args.course)
    if not 2 <= len(profile.platforms) <= 6:
        parser.error("Compact PPO supports only one to five jumps")
    meta = metadata(profile, args.course, ticks=args.ticks, shaped=args.shaped, seed=args.seed)
    model_path, _, log_path = paths(args.output)
    training_meta = meta
    if args.eval or args.resume:
        training_meta = validate_checkpoint(args.output, meta, evaluation=args.eval)
    elif any(path.exists() for path in paths(args.output)):
        raise FileExistsError("Existing PPO output: use --resume or a fresh prefix")
    torch.set_num_threads(1)
    handle, env, model = None, None, None
    try:
        if not args.eval:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a")

        def log(**fields):
            line = json.dumps(
                dict(mode="eval" if args.eval else "train", seed=args.seed, **fields),
                allow_nan=False,
            )
            if handle:
                handle.write(line + "\n")
                handle.flush()
            elif args.eval:
                print(line, flush=True)

        log(
            event="run_config",
            course=str(args.course.resolve()),
            configuration=training_meta,
            evaluation_profile=meta["profile"] if args.eval else None,
            evaluation_course_sha256=meta["course_sha256"] if args.eval else None,
            requested_steps=args.steps,
            evaluation_episodes=args.episodes,
        )
        env = CompactParkourEnv(
            profile,
            ticks=args.ticks,
            shaped=args.shaped,
            log=log,
            backend_factory=lambda: MineflayerBackend(
                args.host, args.port, args.username, args.version
            ),
        )
        model = (
            PPO.load(model_path, env=env, device="cpu")
            if args.eval or args.resume
            else PPO("MlpPolicy", env, seed=args.seed, **PPO_OPTIONS)
        )
        if args.check_env:
            check_env(env, warn=True)
        if args.eval:
            counts = evaluate(model, env, args.episodes, args.seed)
            log(event="done", episodes=args.episodes, **counts)
        else:
            model.learn(
                total_timesteps=args.steps,
                reset_num_timesteps=not args.resume,
                callback=SaveEveryRollout(args.output, meta),
            )
            log(event="done", trained_steps=model.num_timesteps)
    except KeyboardInterrupt:
        print(
            "Evaluation interrupted."
            if args.eval
            else "Interrupted; saving the current compact PPO checkpoint.",
            flush=True,
        )
    finally:
        try:
            if model is not None and not args.eval:
                save_checkpoint(model, args.output, meta)
        finally:
            if env is not None:
                env.close()
            if handle:
                handle.close()


if __name__ == "__main__":
    main()
