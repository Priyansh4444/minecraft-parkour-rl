"""Read the intermediate movement frames needed to detect short contacts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..minecraft.connection import PhysicsFrame, validate_frame

Outcome = Literal["running", "goal", "fall", "timeout"]
Position = tuple[float, float, float]


@dataclass(frozen=True)
class ActionContacts:
    highest_platform: int
    landed_platform: int | None
    outcome: Outcome


def on_platform(position: Sequence[float], platform: Sequence[float]) -> bool:
    return (
        abs(position[0] - platform[0]) <= 0.5
        and abs(position[2] - platform[2]) <= 0.5
        and abs(position[1] - platform[1]) <= 0.15
    )


def platform_contact(platforms: tuple[Position, ...], frame: PhysicsFrame) -> int | None:
    if not frame["on_ground"]:
        return None
    return next(
        (
            index
            for index in range(len(platforms) - 1, -1, -1)
            if on_platform(frame["position"], platforms[index])
        ),
        None,
    )


def detect_contacts(
    platforms: tuple[Position, ...], fall_y: float, frames: list[PhysicsFrame], highest: int
) -> ActionContacts:
    landed = None
    outcome: Outcome = "running"
    for frame in frames:
        contact = platform_contact(platforms, frame)
        if contact is not None:
            highest = max(highest, contact)
            landed = max(landed if landed is not None else 0, contact)
        if frame["health"] <= 0 or frame["position"][1] < fall_y:
            outcome = "fall"
            break
        if contact == len(platforms) - 1:
            outcome = "goal"
            break
    return ActionContacts(highest, landed, outcome)


def action_frames(
    after: PhysicsFrame, ticks: int, correction: int, *, track_contacts: bool
) -> list[PhysicsFrame]:
    if not track_contacts:
        return [after]

    frames = after.get("tick_frames")
    if not isinstance(frames, list) or len(frames) != ticks:
        raise RuntimeError("Contact tracking requires one observation per physics tick")

    for elapsed, frame in enumerate(frames, start=1):
        validate_frame(frame)
        if (
            frame["start_tick"] != after["start_tick"]
            or frame["elapsed_ticks"] != elapsed
            or frame["correction_count"] != correction
        ):
            raise RuntimeError("Invalid intermediate tick or server correction; discard transition")

    last = frames[-1]
    if (
        last["position"] != after["position"]
        or last["velocity"] != after["velocity"]
        or last["on_ground"] != after["on_ground"]
    ):
        raise RuntimeError("The last physics tick does not match the action's final observation")
    return frames
