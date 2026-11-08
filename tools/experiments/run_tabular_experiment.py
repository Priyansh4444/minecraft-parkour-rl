"""Run the proposal's independent Q-table seeds and matched comparison episodes.

Start the local Minecraft server first. This script never builds/changes a map.
"""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh results directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--held-out", nargs="*", type=Path, default=[])
    parser.add_argument("--port", type=int, default=25566)
    parser.add_argument("--shaped", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3 or min(args.seeds) < 0:
        parser.error("use at least three distinct nonnegative training seeds")
    if min(args.train_episodes, args.eval_episodes) < 1 or not 1 <= args.port <= 65535:
        parser.error("episode counts and port must be positive and valid")
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Results directory already has a run; choose a fresh --output")
    from parkour.learning.q_learning import Profile
    from tools.courses.prepare_random_course import simple_script

    profile = Profile.load(args.course)
    for course in args.held_out:
        held_out = Profile.load(course)
        if (held_out.start, held_out.platforms) == (profile.start, profile.platforms):
            raise ValueError("A held-out layout/start must differ from the training case")
    args.output.mkdir(parents=True, exist_ok=True)
    course_paths = [args.course.resolve(), *(p.resolve() for p in args.held_out)]
    if len({p.stem for p in course_paths}) != len(course_paths):
        raise ValueError("Course filenames must have distinct stems for separate logs")
    scripted_paths = {}
    for index, course in enumerate(course_paths):
        path = args.output / "baseline-courses" / f"{index}-{course.stem}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(simple_script(json.loads(course.read_text())), indent=2) + "\n")
        scripted_paths[course] = path.resolve()
    manifest = {
        "algorithm": "tabular-q",
        "reward_mode": "platform-bonus" if args.shaped else "sparse",
        "seeds": args.seeds,
        "training_episodes_per_seed": args.train_episodes,
        "evaluation_episodes_per_controller_seed_course": args.eval_episodes,
        "training_course": str(course_paths[0]),
        "held_out_courses": list(map(str, course_paths[1:])),
        "scripted_baseline": "simple-v1: always action 4 (forward+sprint+jump), no observations or tuned timing",
        "scripted_profiles": {str(k): str(v) for k, v in scripted_paths.items()},
        "course_definitions": {str(p): json.loads(p.read_text()) for p in course_paths},
        "source_sha256": {
            str(p): digest(p)
            for p in [
                Path("parkour/learning/q_learning.py"),
                Path("parkour/minecraft/connection.py"),
                Path("bridge/headless.mjs"),
                Path("bridge/parkour-state.mjs"),
                Path("bridge/package-lock.json"),
                Path("tools/courses/prepare_random_course.py"),
                Path("tools/experiments/run_tabular_experiment.py"),
            ]
        },
        "note": "One-jump runs are calibration; the main proposal course has 3–5 jumps. No cross-seed checkpoint reuse.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    current = None

    def status(state, **fields):
        temporary = args.output / "status.tmp"
        temporary.write_text(json.dumps({"state": state, **fields}, indent=2) + "\n")
        temporary.replace(args.output / "status.json")

    def run(seed, course, prefix, mode, episodes):
        nonlocal current
        prefix.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "parkour.learning.q_learning",
            "--course",
            str(course),
            "--output",
            str(prefix),
            "--seed",
            str(seed),
            "--episodes",
            str(episodes),
            "--port",
            str(args.port),
        ]
        if mode == "scripted":
            command[command.index("--course") + 1] = str(scripted_paths[course])
        if mode != "train":
            command.append("--" + mode)
        if args.shaped:
            command.append("--shaped")
        status("running", seed=seed, controller=mode, course=str(course), command=command)
        output = args.output / f"seed-{seed}" / f"{course.stem}-{mode}.log"
        checkpoint = prefix.with_suffix(".json")
        before = digest(checkpoint) if mode != "train" and checkpoint.exists() else None
        with output.open("w") as handle:
            current = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
            )
            # A finite bound includes every decision, reset overhead and a lag margin.
            current.wait(timeout=max(60, episodes * (Profile.load(course).max_steps * 0.1 + 1) * 2))
        if current.returncode:
            raise RuntimeError(f"{mode} failed; inspect {output}")
        current = None
        if before is not None and digest(checkpoint) != before:
            raise RuntimeError("Comparison/evaluation changed a training checkpoint")

    try:
        for seed in args.seeds:
            prefix = args.output / f"seed-{seed}" / "q"
            run(seed, course_paths[0], prefix, "train", args.train_episodes)
            for course in course_paths:
                run(seed, course, prefix, "eval", args.eval_episodes)
                for mode in ("random", "scripted"):
                    run(
                        seed,
                        course,
                        prefix.parent / (course.stem + "-" + mode),
                        mode,
                        args.eval_episodes,
                    )
        status(
            "completed",
            note="Inspect per-seed outcomes; do not pool repeated episodes as independent training runs",
        )
    except BaseException as error:
        status("stopped" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        if current is not None and current.poll() is None:
            current.send_signal(signal.SIGINT)
            try:
                current.wait(timeout=15)
            except subprocess.TimeoutExpired:
                current.kill()
                current.wait()


if __name__ == "__main__":
    main()
