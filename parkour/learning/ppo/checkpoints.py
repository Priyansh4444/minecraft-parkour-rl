"""PPO settings and the files belonging to one saved run."""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Self, TypedDict

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from ..q_learning import ACTIONS, GAMMA, SCHEMA, Profile
from .environment import HIGH, LOW
from .observations import CONTEXT_FIELDS


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
BIN_NORMALIZATION = "2*(integer_bin-low)/(high-low)-1; float32; round to invert"


@dataclass(frozen=True)
class CheckpointFiles:
    model_file: Path
    metadata_file: Path
    training_log: Path

    @classmethod
    def from_prefix(cls, output_prefix: Path) -> Self:
        return cls(
            model_file=output_prefix.with_suffix(".zip"),
            metadata_file=output_prefix.with_suffix(".json"),
            training_log=output_prefix.with_suffix(".jsonl"),
        )


def build_metadata(
    profile: Profile,
    course: Path,
    *,
    ticks: int = 2,
    shaped: bool = False,
    seed: int = 11,
    contextual: bool = False,
    track_contacts: bool = False,
) -> dict[str, object]:
    configuration: dict[str, object] = dict(
        algorithm="ppo-clip",
        version=1,
        tabular_schema=SCHEMA,
        actions=list(ACTIONS),
        state_fields=SCHEMA["state_fields"],
        raw_low=LOW.tolist(),
        raw_high=HIGH.tolist(),
        normalization=BIN_NORMALIZATION,
        reward_mode="platform-bonus" if shaped else "sparse",
        ticks=ticks,
        seed=seed,
        profile=json.loads(json.dumps(asdict(profile))),
        course_sha256=hashlib.sha256(course.read_bytes()).hexdigest(),
        ppo=PPO_OPTIONS,
        torch_threads=1,
        packages={name: version(name) for name in ("gymnasium", "stable-baselines3", "torch")},
    )
    # retain the original metadata for existing seven-bin checkpoints
    if contextual:
        configuration.update(
            version=2,
            state_fields=CONTEXT_FIELDS,
            raw_low=[-1.0] * len(CONTEXT_FIELDS),
            raw_high=[1.0] * len(CONTEXT_FIELDS),
            normalization="context_observation in observations.py; float32 clipped to [-1, 1]",
        )
    if track_contacts:
        configuration.update(version=2, platform_detection="center-qualified contact at each tick")
    return configuration


def validate_checkpoint(
    output_prefix: Path, expected_configuration: dict[str, object], *, evaluation: bool = False
) -> dict[str, object]:
    files = CheckpointFiles.from_prefix(output_prefix)
    model_file, metadata_file = files.model_file, files.metadata_file
    saved_configuration = json.loads(metadata_file.read_text())
    # file formatting can change without changing the decoded course. training
    # still requires an identical profile and observation/action/reward contract.
    ignored = {"profile", "course_sha256"} if evaluation else {"course_sha256"}
    if (
        not isinstance(saved_configuration, dict)
        or set(saved_configuration) != set(expected_configuration)
        or {k: v for k, v in saved_configuration.items() if k not in ignored}
        != {k: v for k, v in expected_configuration.items() if k not in ignored}
        or not model_file.is_file()
    ):
        raise ValueError(
            "Checkpoint does not match the observations, contact tracking, course, seed or reward/tick settings"
        )
    return saved_configuration


def validate_warm_start(output_prefix: Path) -> Path:
    """Route training uses seven bins, sparse rewards and final-tick contacts."""
    files = CheckpointFiles.from_prefix(output_prefix)
    saved = json.loads(files.metadata_file.read_text())
    expected = dict(
        algorithm="ppo-clip",
        version=1,
        tabular_schema=SCHEMA,
        actions=list(ACTIONS),
        state_fields=SCHEMA["state_fields"],
        raw_low=LOW.tolist(),
        raw_high=HIGH.tolist(),
        normalization=BIN_NORMALIZATION,
        reward_mode="sparse",
        ticks=2,
        ppo=PPO_OPTIONS,
        packages={name: version(name) for name in ("gymnasium", "stable-baselines3", "torch")},
    )
    if (
        not isinstance(saved, dict)
        or any(saved.get(key) != value for key, value in expected.items())
        or "platform_detection" in saved
    ):
        raise ValueError(
            "Warm-start requires seven binned inputs, final-tick contacts and matching sparse-reward/PPO settings"
        )
    if not files.model_file.is_file():
        raise FileNotFoundError(files.model_file)
    return files.model_file


def save_checkpoint(
    model: BaseAlgorithm, output_prefix: Path, configuration: dict[str, object]
) -> None:
    files = CheckpointFiles.from_prefix(output_prefix)
    model_file, metadata_file = files.model_file, files.metadata_file
    temporary = model_file.with_name(model_file.stem + ".tmp.zip")
    temporary_meta = metadata_file.with_name(metadata_file.name + ".tmp")
    try:
        model.save(temporary)
        temporary_meta.write_text(json.dumps(configuration, allow_nan=False) + "\n")
        temporary.replace(model_file)
        temporary_meta.replace(metadata_file)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)


class SaveEveryRollout(BaseCallback):
    def __init__(self, output_prefix: Path, configuration: dict[str, object]) -> None:
        super().__init__()
        self.output_prefix, self.configuration = output_prefix, configuration

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        # this occurs after the previous rollout's optimizer update.
        save_checkpoint(self.model, self.output_prefix, self.configuration)
