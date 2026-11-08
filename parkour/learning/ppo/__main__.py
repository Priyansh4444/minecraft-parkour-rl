"""Train or evaluate a PPO model on a configured Minecraft course."""

import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ...minecraft.connection import MineflayerBackend
from ..q_learning import Profile
from .checkpoints import (
    PPO_OPTIONS,
    CheckpointFiles,
    SaveEveryRollout,
    build_metadata,
    save_checkpoint,
    validate_checkpoint,
)
from .environment import CompactParkourEnv


def evaluate(
    model: PPO, environment: CompactParkourEnv, episodes: int, seed: int
) -> dict[str, int]:
    counts = dict(goal=0, fall=0, timeout=0)
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(int(action))
            if terminated or truncated:
                counts[info["outcome"]] += 1
                break
    return counts


def main(argv: list[str] | None = None) -> None:
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
    arguments = parser.parse_args(argv)
    if (
        arguments.steps < 1
        or arguments.episodes < 1
        or arguments.seed < 0
        or not 1 <= arguments.ticks <= 20
        or not 1 <= arguments.port <= 65535
        or (arguments.eval and arguments.resume)
    ):
        parser.error(
            "Use positive steps/episodes, nonnegative seed, valid ticks/port and training-only --resume"
        )
    profile = Profile.load(arguments.course)
    if not 2 <= len(profile.platforms) <= 6:
        parser.error("Compact PPO supports only one to five jumps")
    configuration = build_metadata(
        profile,
        arguments.course,
        ticks=arguments.ticks,
        shaped=arguments.shaped,
        seed=arguments.seed,
    )
    files = CheckpointFiles.from_prefix(arguments.output)
    training_configuration = configuration
    if arguments.eval or arguments.resume:
        training_configuration = validate_checkpoint(
            arguments.output, configuration, evaluation=arguments.eval
        )
    elif files.model_file.exists() or files.metadata_file.exists() or files.training_log.exists():
        raise FileExistsError("Existing PPO output: use --resume or a fresh prefix")
    torch.set_num_threads(1)
    log_file, environment, model = None, None, None
    try:
        if not arguments.eval:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            log_file = files.training_log.open("a")

        def log(**fields):
            line = json.dumps(
                dict(mode="eval" if arguments.eval else "train", seed=arguments.seed, **fields),
                allow_nan=False,
            )
            if log_file:
                log_file.write(line + "\n")
                log_file.flush()
            elif arguments.eval:
                print(line, flush=True)

        log(
            event="run_config",
            course=str(arguments.course.resolve()),
            configuration=training_configuration,
            evaluation_profile=configuration["profile"] if arguments.eval else None,
            evaluation_course_sha256=configuration["course_sha256"] if arguments.eval else None,
            requested_steps=arguments.steps,
            evaluation_episodes=arguments.episodes,
        )
        environment = CompactParkourEnv(
            profile,
            ticks=arguments.ticks,
            shaped=arguments.shaped,
            log=log,
            backend_factory=lambda: MineflayerBackend(
                arguments.host, arguments.port, arguments.username, arguments.version
            ),
        )
        model = (
            PPO.load(files.model_file, env=environment, device="cpu")
            if arguments.eval or arguments.resume
            else PPO("MlpPolicy", environment, seed=arguments.seed, **PPO_OPTIONS)
        )
        if arguments.check_env:
            check_env(environment, warn=True)
        if arguments.eval:
            counts = evaluate(model, environment, arguments.episodes, arguments.seed)
            log(event="done", episodes=arguments.episodes, **counts)
        else:
            model.learn(
                total_timesteps=arguments.steps,
                reset_num_timesteps=not arguments.resume,
                callback=SaveEveryRollout(arguments.output, configuration),
            )
            log(event="done", trained_steps=model.num_timesteps)
    except KeyboardInterrupt:
        print(
            "Evaluation interrupted."
            if arguments.eval
            else "Interrupted; saving the current compact PPO checkpoint.",
            flush=True,
        )
    finally:
        try:
            if model is not None and not arguments.eval:
                save_checkpoint(model, arguments.output, configuration)
        finally:
            if environment is not None:
                environment.close()
            if log_file:
                log_file.close()


if __name__ == "__main__":
    main()
