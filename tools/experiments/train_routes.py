"""Train one compact PPO policy across a frozen bank of straight parkour routes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from stable_baselines3 import PPO

from parkour.learning.ppo.checkpoints import PPO_OPTIONS, SaveEveryRollout, save_checkpoint
from parkour.learning.ppo.routes import RouteTrainingEnv
from parkour.learning.q_learning import SCHEMA, Profile
from parkour.minecraft.connection import MineflayerBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--port", type=int, default=25570)
    parser.add_argument("--watch-player")
    arguments = parser.parse_args()
    if arguments.steps < 1 or (arguments.resume and arguments.warm_start):
        parser.error("Use positive steps; choose resume or warm-start, not both")

    bank = json.loads(arguments.bank.read_text())
    bank_digest = hashlib.sha256(arguments.bank.read_bytes()).hexdigest()
    bank_folder = arguments.bank.parent
    training_routes = [bank_folder / entry["file"] for entry in bank["training"]]
    for entry in bank["training"] + bank["evaluation"]:
        if (
            hashlib.sha256((bank_folder / entry["file"]).read_bytes()).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError("A frozen course file changed")

    output = arguments.output
    prefix = output / "policy"
    if arguments.resume:
        configuration = json.loads(prefix.with_suffix(".json").read_text())
        if (
            configuration["route_bank_sha256"] != bank_digest
            or configuration["schema"] != SCHEMA
            or configuration["seed"] != arguments.seed
            or configuration["ppo"] != PPO_OPTIONS
        ):
            raise ValueError("This checkpoint belongs to a different training condition")
        model_file = prefix.with_suffix(".zip")
    else:
        output.mkdir(parents=True, exist_ok=False)
        configuration = dict(
            algorithm="ppo",
            schema=SCHEMA,
            reward_mode="sparse",
            ticks=2,
            seed=arguments.seed,
            ppo=PPO_OPTIONS,
            route_bank_sha256=bank_digest,
            start_offsets=bank["training_start_offsets"],
            requested_steps=arguments.steps,
        )
        model_file = None
        if arguments.warm_start:
            source = arguments.warm_start
            condition = json.loads(source.with_suffix(".json").read_text())
            if (
                condition.get("tabular_schema") != SCHEMA
                or condition.get("reward_mode") != "sparse"
                or condition.get("ticks") != 2
                or condition.get("ppo") != PPO_OPTIONS
            ):
                raise ValueError(
                    "Warm-start model has a different observation/action/reward condition"
                )
            model_file = output / "initial-model.zip"
            shutil.copyfile(source.with_suffix(".zip"), model_file)
            configuration["warm_start_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()

    torch.set_num_threads(1)
    environment = None
    model = None
    with prefix.with_suffix(".jsonl").open("a") as log_file:
        try:
            backend = MineflayerBackend("127.0.0.1", arguments.port, "Watcher", "26.1.2")
            environment = RouteTrainingEnv(
                training_routes, backend, log_file, output, arguments.seed, arguments.watch_player
            )
            model = (
                PPO.load(model_file, env=environment, device="cpu")
                if model_file
                else PPO("MlpPolicy", environment, seed=arguments.seed, **PPO_OPTIONS)
            )
            if not arguments.resume:
                configuration["starting_model_steps"] = model.num_timesteps
            starting_model_steps = configuration["starting_model_steps"]
            if not isinstance(starting_model_steps, int):
                raise ValueError("Checkpoint starting_model_steps must be an integer")
            environment.training_steps = model.num_timesteps - starting_model_steps
            environment.record(event="run_config", configuration=configuration)
            environment.status("training", checkpoint=str(prefix.with_suffix(".zip")))
            model.learn(
                arguments.steps,
                reset_num_timesteps=False,
                callback=SaveEveryRollout(prefix, configuration),
            )
            save_checkpoint(model, prefix, configuration)

            # evaluate only after the predeclared training budget; never update on these routes
            frozen_weights = hashlib.sha256(prefix.with_suffix(".zip").read_bytes()).hexdigest()
            for entry in bank["evaluation"]:
                environment.evaluation_course = Profile.load(bank_folder / entry["file"])
                environment.evaluation_name = entry["name"]
                for _episode in range(50):
                    observation, _ = environment.reset()
                    while True:
                        action, _ = model.predict(observation, deterministic=True)
                        observation, _, terminated, truncated, _ = environment.step(int(action))
                        if terminated or truncated:
                            break
            if (
                hashlib.sha256(prefix.with_suffix(".zip").read_bytes()).hexdigest()
                != frozen_weights
            ):
                raise RuntimeError("Evaluation changed the checkpoint")
            environment.status("completed", trained_model_steps=model.num_timesteps)
        except KeyboardInterrupt:
            if environment:
                environment.status("stopped", note="Checkpoint retained for resume")
        except Exception as error:
            if environment:
                environment.status("failed", error=str(error))
            raise
        finally:
            if (
                model is not None
                and environment is not None
                and environment.evaluation_course is None
            ):
                save_checkpoint(model, prefix, configuration)
            if environment is not None:
                environment.close()


if __name__ == "__main__":
    main()
