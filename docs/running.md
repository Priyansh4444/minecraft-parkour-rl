# Training and evaluation

Start the [prepared server](setup.md) first. These commands use port 25566 and a fresh output prefix for each model.

## Q-learning

```sh
uv run python -m parkour.learning.q_learning --course parkour/courses/proposal-map1-single.json --output .local/runs/q --episodes 300 --seed 11
uv run python -m parkour.learning.q_learning --course parkour/courses/proposal-map1-single.json --output .local/runs/q --episodes 50 --seed 11 --eval
```

Use `proposal-map1-three.json` for the three-jump course. `--resume` restores a compatible table from the same course and uses the supplied exploration schedule. `--random` samples actions; `--scripted` replays the supplied course's action list. `--no-track-contacts` selects final-tick detection when evaluating an older table; resume requires saved course information.

For three independent seeds followed by learned, random and simple scripted evaluations:

```sh
uv run python -m tools.experiments.run_tabular_experiment --course parkour/courses/proposal-map1-three.json --output .local/comparison --train-episodes 300 --eval-episodes 50
```

This runner uses seeds 11, 22 and 33. Its simple script always runs and jumps. `--held-out PATHS...` adds evaluation courses without training on them. Keep evaluation courses out of training and tuning, and compare completion, progress, falls and time for each seed separately.

## PPO

```sh
uv run --extra ppo python -m parkour.learning.ppo --course parkour/courses/proposal-map1-single.json --output .local/runs/ppo --steps 4096 --seed 11 --port 25566 --track-contacts
uv run --extra ppo python -m parkour.learning.ppo --course parkour/courses/proposal-map1-single.json --output .local/runs/ppo --eval --episodes 50 --seed 11 --port 25566 --track-contacts
```

Optional settings:

| Flag | Effect |
| --- | --- |
| `--observations context` | Use the fifteen inputs in [the code guide](code.md#ppo) |
| `--track-contacts` | Detect platform contacts on every physics tick |
| `--shaped` | Add the one-time intermediate-platform reward |
| `--resume` | Continue a saved model |

Use matching observation, contact and reward settings when training, evaluating or resuming a model.

## Generated courses

Generate a single evaluation layout:

```sh
uv run python -m tools.courses.prepare_random_course --seed 17501 --jumps 3 --output .local/evaluation-course
```

Generate a bank of training and evaluation routes:

```sh
uv run python -m tools.courses.route_bank --output .local/routes
```

Both commands write course JSON and `build.mcfunction`. Run its construction commands on the working server: load the requested chunks with `forceload` before clearing and placing blocks. Generating files alone does not change the world.

After building the bank, train one PPO policy across it:

```sh
uv run --extra ppo python -m tools.experiments.train_routes --bank .local/routes/manifest.json --output .local/route-training --steps 200000 --port 25566
```

`routes.py` samples one-jump routes initially, then allows up to two, three, four and five jumps after 8,192, 24,576, 49,152 and 65,536 decisions. The bank has 16 training routes and six evaluation routes. After training, the runner evaluates 50 episodes on each evaluation route. All routes use straight, full-block jumps.

Add `--watch-player YourName` for a side view of the current route. `--warm-start path/to/ppo` requires seven binned inputs, sparse rewards and final-tick contacts; `--resume` continues this run. Outputs are `policy.zip`, `policy.json`, `policy.jsonl` and `status.json`.
