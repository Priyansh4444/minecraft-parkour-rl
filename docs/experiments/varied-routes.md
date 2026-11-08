# Learning across varied routes

The optional PPO learner can share one policy across several straight routes. This is an exploratory extension; it does not replace the independent-seed Q-learning, random and simple scripted comparison required by the proposal.

## What it learns

The bank contains 16 training routes with one to five jumps. Gap distances are two, three or four blocks between platform centers. Shorter gaps may rise by one block. Routes face all four cardinal directions, and training starts vary by ±0.1 blocks on each horizontal axis. The policy still receives the same seven numerical bins and chooses among the same five forward/jump actions. Rewards remain sparse.

The curriculum is fixed before training:

| New training decisions | Routes available |
| --- | --- |
| 0 | one jump |
| 8,192 | one or two jumps |
| 24,576 | one to three jumps |
| 49,152 | one to four jumps |
| 65,536 onward | one to five jumps |

Routes are sampled uniformly from those available, so earlier skills continue to appear. A route is selected when an episode resets. The curriculum counts decisions from this run, excluding earlier warm-start training.

Six different layouts are frozen for evaluation. Their geometry sequences differ from the training layouts. After the requested training budget, the learner runs 50 deterministic episodes per evaluation route without updating weights. Training successes and these evaluation results are logged separately. The existing evaluation seed 17501 and reserved starting profiles are never used for training.

## Files and commands

Prepare a new bank:

```sh
uv run python -m tools.courses.route_bank --output .local/my-routes
```

The folder contains readable course JSON, `manifest.json` with seeds and file hashes, and `build.mcfunction` with construction commands. Build these routes on a private working server before starting the learner. Issue all `forceload` commands first and let chunks load before placing blocks. Inspect the actual supports and clearance; conservative generation rules alone do not prove that a route is solvable.

With the built server running:

```sh
uv run --extra ppo python -m tools.experiments.train_routes \
  --bank .local/my-routes/manifest.json \
  --output .local/my-training \
  --steps 200000 --port 25570
```

Add `--watch-player YourName` to put a connected spectator beside the current route. Closing the spectator window leaves learning running. Omit it for headless operation. Both modes use the same Mineflayer environment and local server.

An optional `--warm-start path/to/ppo` copies an existing compatible compact PPO model into the new output folder; it never overwrites that source model. The saved metadata must match the observation, action, reward and PPO settings. A warm-start run is continued learning, not a new independent seed.

To continue an interrupted varied-route run, use its original bank and output with `--resume`, the same `--seed`, and a positive additional `--steps` budget. Do not pass `--warm-start` on resume. Checkpoint recovery restarts at an episode boundary; it does not restore the exact Minecraft or route sampler state.

`policy.zip` contains weights and optimizer state. `policy.json` records the training condition. Both save after every 128-decision rollout and on a clean interruption. `policy.jsonl` records transitions and episode summaries with `mode` and `route`. `status.json` reports the learner PID, completed decision count and latest episode outcome. A stopped or failed status is not successful training.

## Limits

This is practice across a small family of straight, full-block routes. The controls cannot turn or strafe. It does not train arbitrary Minecraft parkour, neos or headhitters, and it does not establish generalization until the separate evaluation completes. The final evaluation covers greedy PPO transfer; the primary random/scripted comparisons remain separate experiments.

The local run started on 2026-11-08 is under `.local/varied-ppo-20261108/`. It requests 200,000 additional decisions, warm-starts the predeclared seed-11 sparse calibration model, and keeps previous results untouched. `runtime-status.json` records its supervisor, server, spectator and learner PIDs. Closing Minecraft does not stop that supervisor. The computer and background server must stay running.
