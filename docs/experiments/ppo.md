# Compact PPO comparison

PPO is a secondary comparison based on CS175 Lecture 13. Tabular Q-learning is the main experiment.

Both learners receive the **same seven discretized values**: forward/lateral/height offsets to the next platform, forward and vertical velocity, ground contact and highest platform index. PPO scales those integer bins into a bounded numerical vector; it receives no extra measurements. Both choose the same five idle/walk/run/jump actions and use the same platform definitions, reset rules, two-client-tick actions and rewards.

The default reward remains -0.01 per ordinary decision, +1 for the final goal and -1 for a fall. There is no distance shaping, terrain grid, turning or sideways control. The optional one-time intermediate-platform bonus is a separate reward condition for both learners, not enabled for the initial sparse-reward runs.

## Code and course concepts

`parkour/learning/ppo.py` adapts the proposal contract to Gymnasium and Stable-Baselines3 PPO. It imports the Q-learning implementation's state encoding, actions, reward and platform helpers. Tests replay identical frames/actions through both loops and check their states, rewards and episode outcomes. The learners share the observation/action/reward contract; their checkpoints and logs stay separate.

PPO uses a small CPU policy with two 32-unit hidden layers. The algorithm follows the concepts in Lectures 11–13: a stochastic policy, value baseline, advantage estimates and the clipped policy update. Stable-Baselines3 implements the optimizer/algorithm; our code supplies the Minecraft environment and experiment interface. It is a separate model, not a replacement Q-table or a claim of general Minecraft ability.

## Install and train

```sh
uv sync --extra ppo --python 3.12
npm ci --prefix bridge --ignore-scripts
```

Use the same [local server setup](../setup/server.md). If Q-learning is already running, use a separate server folder with a separate working copy of Map 1 and a different port, such as 25567. Do not connect a second bot named `Watcher` to the Q-learning server. The experiments must not interfere with one another's player/world state.

With the separate server running:

```sh
uv run --extra ppo python -m parkour.learning.ppo --course parkour/courses/proposal-map1-single.json --output .local/compact-ppo/seed-11/ppo --steps 4096 --seed 11 --port 25567
```

Evaluate the saved policy without updating it:

```sh
uv run --extra ppo python -m parkour.learning.ppo --course parkour/courses/proposal-map1-single.json --output .local/compact-ppo/seed-11/ppo --eval --episodes 50 --seed 11 --port 25567
```

Use fresh output prefixes for independent seeds 22 and 33. `--resume` continues a compatible compact PPO checkpoint. Model metadata verifies the observation/action/reward contract and action duration before loading. A checkpoint with a different observation/action/reward contract is rejected.

## Results and limits

Each output prefix has a `.zip` policy checkpoint, `.json` contract/configuration and `.jsonl` transition/episode log. Model saving is atomic. Evaluation does not change model weights. Checkpoints, conditions and seeds remain separate from Q-learning results.

Compare completion, falls, progress, decisions and wall time using the same fixed starts and 50 evaluation episodes per trained seed. PPO's training budget is measured in decisions; the current Q-learning calibration budget is measured in episodes. Record both actual decisions and elapsed time, and do not attribute differences to the algorithm alone when training budgets differ.

These first runs are single-jump calibration. The three-jump configuration and preset controller are available after calibration. Reserved starting positions remain out of training and tuning. A successful fixed-start policy does not establish transfer or full Map 1 completion. Live tests so far target Linux; other operating systems still need testing.

The graphical Minecraft client can remain closed. A local Java server and Mineflayer bot must keep running. Each action advances Mineflayer client physics by the requested tick count; the server world continues between requests. The current static-block course tests do not establish authoritative server-side lockstep or exact equivalence to a human client's movement.

The completed calibration batch records status in `.local/map1-compact-ppo-job/status.json` and results in `.local/map1-compact-ppo-calibration-sparse/`. It uses independent seeds 11/22/33, 4,096 training decisions and 50 deterministic evaluation episodes per seed. The supervisor owns only its separate server and stops it after the finite batch. The Q-learning batch on port 25566 completed normally during PPO startup; its results remain separate.
