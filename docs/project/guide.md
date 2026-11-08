# How the project works

## Learning loop

The Minecraft server hosts a working copy of the course. Mineflayer connects the bot. Python repeatedly reads its state, chooses an action, advances two client ticks, measures reward and updates the learner during training. Evaluation performs the same loop without updates.

`learning/q_learning.py` groups the relative next-platform position, forward/vertical velocity, grounded status and highest platform index into seven integer bins. That complete tuple selects one row of five Q-values. Epsilon-greedy exploration and the one-step Q-learning update change one state/action value at a time. `learning/ppo.py` uses the same bins/actions/rewards with a small PPO network.

The actions are idle, walk, run, walking jump and running jump. Base rewards are -0.01 per ordinary decision, +1 at the goal and -1 for a fall. A timeout ends an attempt but retains its next-state value. The optional one-time intermediate-platform bonus is a separate condition. Binning approximates the state; it does not guarantee a Markov representation.

## Code folders

| Folder | Responsibility |
| --- | --- |
| `parkour/learning/` | Q-learning and compact PPO, including checkpoints and training/evaluation commands. |
| `parkour/minecraft/` | JSON connection to Mineflayer and a server launcher with optional spectator setup. |
| `parkour/courses/` | Platform geometry, start positions and episode limits. JSON profiles describe courses; they do not build worlds. |
| `bridge/` | Node transport, raw state, controls and bounded client-physics stepping. |
| `tools/courses/` | Safe world import and reproducible randomized test construction commands. |
| `tools/experiments/` | Independent training seeds and matched greedy/random/simple-script comparisons. |
| `tests/` | Checks grouped by learning, Minecraft, course and experiment responsibilities. |

Both [headless](../setup/server.md) and [visible](../setup/watching.md) operation use this exact loop. The spectator affects only a separate player. The server world continues between requested actions; Mineflayer stepping is not authoritative server-side lockstep.

## Useful data

Runtime files belong under `.local/`, outside Git:

- `worlds/`: clean imported maps; keep the original ZIP too.
- `mc2612/server/`: canonical server and downloaded libraries. Other working servers may link to these libraries.
- `map1-tabular-v3-calibration-sparse/`: completed single-jump Q-tables and evaluation logs.
- `map1-compact-ppo-calibration-sparse/`: separate compact PPO results.
- `random-course-test-17501/`: frozen test definition and construction commands; never train or tune on it.
- `random-course-evaluation-17501/`: separate transfer-test results.
- `*-job/`: supervisor commands, status and server logs.
- `cs175-fixes/`, `compact-ppo-validation/`, `watch-mode-validation/`: focused live verification artifacts.

Q `.json` files contain schemas and tables. PPO `.zip` files contain network/optimizer state and have companion `.json` contracts. `.jsonl` files record transitions and episode summaries. Manifests record seeds, courses and source versions. These records make results interpretable; a saved policy alone does not show whether it learned or generalized.

Keep `.venv/` and `bridge/node_modules/` installed for offline use. They are dependencies, not results. The submitted experiment remains a short 3–5-jump task; the completed single-jump batches and randomized transfer test do not replace it. See [the experiment scope](proposal-scope.md) and [random-course protocol](../experiments/random-course.md).
