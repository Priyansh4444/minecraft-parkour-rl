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
