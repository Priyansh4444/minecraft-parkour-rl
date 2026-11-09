"""Generate a seeded straight course; CLI output is reserved for evaluation."""

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path


def simple_script(profile):
    """Hold forward, sprint and jump throughout; no geometry or timing lookup."""
    result = copy.deepcopy(profile)
    if "start_position" in result:
        result["scripted_actions"] = ["run_jump"] * result.get("max_decisions", 80)
    else:
        result["script"] = [4] * result.get("max_steps", 80)
    return result


def generate(seed=17501, jumps=3, *, purpose="evaluation"):
    if type(seed) is not int or seed < 0 or type(jumps) is not int or not 1 <= jumps <= 5:
        raise ValueError("Use a nonnegative integer seed and one to five jumps")
    if purpose not in ("training", "evaluation"):
        raise ValueError("purpose must be training or evaluation")
    if purpose == "training" and seed == 17501:
        raise ValueError("Seed 17501 is reserved for evaluation")
    rng = random.Random(seed)
    platforms, choices = [[1000.5, 80, 1000.5]], []
    for _ in range(jumps):
        distance = rng.choice((2, 3, 4))
        rise = rng.choice((0, 1)) if distance < 4 else 0
        x, y, z = platforms[-1]
        platforms.append([x - distance, y + rise, z])
        choices.append({"center_distance": distance, "height_change": rise})
    profile = simple_script(
        dict(
            start_position=dict(zip(("x", "y", "z"), platforms[0], strict=True)),
            facing_yaw_degrees=90,
            landing_positions=[
                dict(zip(("x", "y", "z"), position, strict=True)) for position in platforms
            ],
            fall_below_y=77,
            max_decisions=80,
        )
    )
    xmin, xmax = math.floor(platforms[-1][0]) - 3, 1003
    ymin, ymax, zmin, zmax = 77, max(p[1] for p in platforms) + 4, 997, 1003
    commands = [
        f"forceload add {xmin} {zmin} {xmax} {zmax}",
        f"fill {xmin} {ymin} {zmin} {xmax} {ymax} {zmax} minecraft:air",
    ]
    commands += [
        f"setblock {math.floor(x)} {y - 1} {math.floor(z)} minecraft:green_wool"
        for x, y, z in platforms
    ]
    provenance = dict(
        seed=seed,
        jumps=jumps,
        choices=choices,
        clear_bounds=[[xmin, ymin, zmin], [xmax, ymax, zmax]],
        purpose="training"
        if purpose == "training"
        else "evaluation only; do not train or tune on observed outcomes",
        baseline="always action 4 (run_jump); identical rule on every course",
        limitation="conservative geometry limits do not prove solvability",
    )
    return profile, provenance, commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17501)
    parser.add_argument("--jumps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile, manifest, commands = generate(args.seed, args.jumps)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / name for name in ("course.json", "build.mcfunction", "manifest.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Generated artifacts already exist; use a fresh output directory")
    course_text = json.dumps(profile, indent=2) + "\n"
    command_text = "\n".join(commands) + "\n"
    manifest["sha256"] = {
        "course.json": hashlib.sha256(course_text.encode()).hexdigest(),
        "build.mcfunction": hashlib.sha256(command_text.encode()).hexdigest(),
    }
    for path, contents in zip(
        paths, (course_text, command_text, json.dumps(manifest, indent=2) + "\n"), strict=True
    ):
        with path.open("x") as handle:
            handle.write(contents)
    print(f"Frozen evaluation course: {paths[0]}")


if __name__ == "__main__":
    main()
