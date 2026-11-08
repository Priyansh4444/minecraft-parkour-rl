# Inspecting a run

Use `./scripts/check.sh` for formatting, linting and type checking. There are no automated test suites or CI workflows.

For behavior, watch the bot in Minecraft and inspect the run records:

- confirm the starting position and the complete landing route match the actual blocks
- distinguish training with exploration from evaluation with learning disabled
- follow the highest landing reached, falls, goals and decision counts
- check that a goal means the final landing, not just the first jump
- look for unexpected server corrections, missing blocks, disconnects or reset failures

A fixed reference controller can check whether the configured route and movement loop work. Its successes must never be reported as learned-policy results. Likewise, a successful exploratory training attempt does not prove a reliable evaluation policy.

Current records live in `.local/map1-tabular-v3-calibration-sparse/`, `.local/map1-compact-ppo-calibration-sparse/` and `.local/random-course-evaluation-17501/`. The visible three-jump run is recorded separately under `.local/headful-demo/three-jump-training/`.

Keep evaluation logs separate and do not update the model during evaluation. A frozen randomized course can reveal specialization to the training layout; do not train on that course after seeing its results and continue calling it unseen.

## Map 1 inspection — November 8

The three-jump route was inspected through actual server block reads. Its feet positions are `(19.5, -6, 13.5)`, `(17.5, -6, 13.5)`, `(13.5, -6, 13.5)` and `(10.5, -5, 13.5)`. Each has green wool below it and air at feet/head height. The final landing is one block higher.

A fixed, hand-timed reference completed five complete attempts, including all three jumps. This confirms the configured geometry and movement loop for that route. It is an interface check, not the simple comparison baseline and not learned-policy performance.

The visible Q-learning run completed 300 training attempts, followed by 50 attempts with exploration and learning disabled. Eight of the 50 evaluation attempts reached the final landing. The policy remains unreliable and this says nothing about unseen-course performance. No success on the first jump alone was counted as full completion.

Map reads and reference trajectories are in `.local/map-inspection/`. The learned run and evaluation are in `.local/headful-demo/three-jump-training/`. The graphical demonstration uses a fixed overview camera so resets do not repeatedly pull the viewer across the map. Reference and learned runs are labeled separately.

The inspection also exposed an operator-permission problem for a newly named bot. The launcher now grants permission after that bot actually joins, when its offline player profile is available.
