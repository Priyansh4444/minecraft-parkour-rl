"""Prepare varied straight routes and freeze separate evaluation layouts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from parkour.learning.q_learning import DIRECTIONS
from tools.courses.prepare_random_course import generate

TRAINING_LENGTHS = (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5)
EVALUATION_LENGTHS = (3, 3, 4, 4, 5, 5)


def prepare_bank(output_folder: Path) -> Path:
    output_folder.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {"training": [], "evaluation": []}
    known_routes: set[tuple[tuple[int, int], ...]] = set()
    commands: list[str] = []
    entries: dict[str, list[dict[str, object]]] = {"training": [], "evaluation": []}

    for split, lengths, first_seed in (
        ("training", TRAINING_LENGTHS, 17600),
        ("evaluation", EVALUATION_LENGTHS, 17700),
    ):
        for index, jumps in enumerate(lengths):
            seed = first_seed + index
            while True:
                course, provenance, _ = generate(seed, jumps, purpose=split)
                signature = tuple(
                    (choice["center_distance"], choice["height_change"])
                    for choice in provenance["choices"]
                )
                if split == "training" or signature not in known_routes:
                    break
                seed += 100
            known_routes.add(signature)

            # keep every route in its own patch of the working world
            location = index if split == "training" else index + len(TRAINING_LENGTHS)
            origin_x = 1000.5 + (location % 4) * 64
            origin_z = 1000.5 + (location // 4) * 64
            yaw = (index % 4) * 90
            forward_x, forward_z = DIRECTIONS[yaw]
            positions = []
            for landing in course["landing_positions"]:
                distance = 1000.5 - landing["x"]
                positions.append(
                    {
                        "x": origin_x + forward_x * distance,
                        "y": landing["y"],
                        "z": origin_z + forward_z * distance,
                    }
                )
            course.update(
                start_position=positions[0].copy(),
                facing_yaw_degrees=yaw,
                landing_positions=positions,
            )
            course.pop("scripted_actions", None)
            path = output_folder / split / f"route-{index + 1:02}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(course, indent=2) + "\n")
            entries[split].append(
                {
                    "name": f"{split}-{index + 1:02}",
                    "file": str(path.relative_to(output_folder)),
                    "seed": seed,
                    "jumps": jumps,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

            minimum_x = math.floor(min(point["x"] for point in positions)) - 3
            maximum_x = math.floor(max(point["x"] for point in positions)) + 3
            minimum_z = math.floor(min(point["z"] for point in positions)) - 3
            maximum_z = math.floor(max(point["z"] for point in positions)) + 3
            maximum_y = max(point["y"] for point in positions) + 4
            commands += [
                f"forceload add {minimum_x} {minimum_z} {maximum_x} {maximum_z}",
                f"fill {minimum_x} 77 {minimum_z} {maximum_x} {maximum_y} {maximum_z} minecraft:air",
            ]
            commands += [
                f"setblock {math.floor(point['x'])} {point['y'] - 1} {math.floor(point['z'])} minecraft:green_wool"
                for point in positions
            ]

    manifest.update(entries)
    manifest.update(
        seed_note="17600-series training; 17700-series evaluation; seed 17501 untouched",
        training_start_offsets=[-0.1, 0.0, 0.1],
        scope="straight full-block jumps, gaps and rises; no turns, neos or headhitters",
    )
    (output_folder / "build.mcfunction").write_text("\n".join(commands) + "\n")
    manifest_path = output_folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(prepare_bank(arguments.output))


if __name__ == "__main__":
    main()
