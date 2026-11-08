# Simple baseline and randomized test course

The original Map 1 scripted result used a hand-timed sequence calibrated on that training layout. Keep those results as a **tuned preset baseline**. They are not results for the new simple baseline, and a baseline succeeding on an easy jump is not itself a defect.

The new `simple-v1` controller always chooses action 4: forward + sprint + jump. It reads no observations, uses no platform-specific timings and applies exactly the same rule on every layout. `tools.experiments.run_tabular_experiment` now creates separate baseline profiles with this constant sequence and records the controller rule in its manifest. Training and greedy evaluation still use the supplied course. Original profiles/results are never rewritten. The low-level `parkour.learning.q_learning --scripted` command still replays whatever sequence the supplied profile contains; use the generated simple profiles for this comparison.

Generate a reproducible test course before examining any policy results:

```sh
python -m tools.courses.prepare_random_course --seed 17501 --jumps 3 --output .local/random-course-test-17501
```

The directory contains `course.json`, `build.mcfunction` and a manifest with the seed, sampling rules and file hashes. Existing artifacts cannot be overwritten. The generator does not connect to Minecraft. Execute the construction commands only in a separate working world: they force-load and clear a small region near x/z=1000, then place one-block green-wool platforms. No turns, hunger, potions or special obstacles are introduced. Center distances are sampled from 2/3/4 blocks, rises from 0/1; four-block center distances stay level. These bounds do not prove the whole course is solvable.

The seed-17501 test is frozen before evaluation and must not be used for training or tuning. It has three jumps. The current Q-tables were trained only on a single jump, so applying them here is an **evaluation-only transfer stress test**, not evidence for or against matched-length generalization after three-jump training. Highest-platform states after the first landing were not learned in the one-jump task. Do not choose a replacement seed based on whether Q-learning beats the baseline. If these results guide tuning, this course becomes validation data; declare a new untouched test before the final experiment.

The completed finite evaluation used its own server/world on port 25568. It verified actual support blocks and first-gap air, then ran 50 greedy, random and simple-script episodes per trained seed (11/22/33) on the randomized layout. It also reran the simple script for 50 episodes per seed on the original one-jump layout, so changing controller and course are reported separately. Evaluation used copied Q-tables and checked their hashes; it never updated or resumed training.

Status: `.local/random-course-eval-job/status.json`. Results: `.local/random-course-evaluation-17501/`. The supervisor script in `.local/random-course-eval-job/run.py` preserves the original commands as run provenance; the server shut down after completion. Use the current module names in `tools/` for new experiments. Historical Q/PPO results, reserved start profiles and the submitted proposal are separate and unchanged.
