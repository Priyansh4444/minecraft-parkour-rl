# How the agent works

The course has three straight jumps, with the middle landing one block higher. The objective is to reach the gold goal without falling. This is an initial experiment; it does **not** solve arbitrary Minecraft parkour, and the current learner has not been shown to learn or generalize.

## Files and loop

- `parkour/build_course.py` sends commands to make the platforms, gaps, catch floor and goal. It overwrites blocks around the origin in the test world.
- `parkour/course.json` supplies the start, facing direction, goal and fall threshold. Keep it consistent with the builder.
- `parkour/agent.py` reads state through Minescript, chooses and performs an action, computes reward, updates weights and repeats. An episode ends at the goal, a fall, or an 80-decision limit.

## What the player can observe

Minecraft supplies numerical position `(x,y,z)`, velocity `(vx,vy,vz)`, and yaw. The agent also estimates whether it is grounded, reads a short line of blocks ahead in a single batch, and knows the configured goal. Its inputs include:

| Input | Why it matters | Current limitation |
| --- | --- | --- |
| Distance to the next edge and approximate gap width | Helps decide when and how to jump | Only a narrow line straight ahead is scanned. |
| Estimated landing height | An uphill landing changes a jump | Measured relative to the player's feet, so it changes in midair; unknown has its own category. |
| Forward speed, vertical velocity, lateral velocity | Momentum and whether the player rises or falls affect a landing | Speed is estimated from displacement; the controller cannot correct sideways movement yet. |
| Grounded estimate, lateral offset and yaw error | Jumps require contact and alignment | Grounded is inferred from nearby blocks and position, not guaranteed. Facing direction is fixed. |
| Distance to goal | Encodes the objective | Straight-line distance alone ignores obstacles and is not a route plan. |

Other available or derivable information includes health, hunger, nearby block types, the current world tick and heading. This experiment leaves health and hunger out of the learning state to keep the first experiment smaller. Hunger and damage can still change in the game; control them in the world settings, or include their effects and timing in the observations if the team studies them. No single vector captures every relevant piece of Minecraft state.

Position, velocity and yaw are continuous measurements, but the learner groups them into coarse **features** such as slow/fast or near/far. It also combines edge distance, speed and grounded status so that it can learn a takeoff timing. Unlike one entry for every full state, separate feature weights are shared across observations; learning can transfer between observations that share features. This is a lightweight **linear Q-function approximation**, not a neural network or proof of generalization. Binning loses information; the state may not be Markov, and coarse features may alias situations needing different actions. Linear off-policy bootstrapping can also be unstable.

The five actions are idle, walk, run, walk+jump and run+jump. There are no turns or left/right actions, so arbitrary 3D navigation, neos, and mazes are out of scope. The code sums weights for active features to estimate the value of each action, then picks a random action with probability epsilon or a highest-value action otherwise. It updates active weights toward a one-step Q-learning target. The weights are saved in `linear-q.json`.

## Reward for reaching the goal

The base reward is `-0.01` per decision, `+10` for reaching the goal, and `-10` for a fall. Goal means the player is near the configured point and estimated grounded; it is not strictly a gold-block collision. The code adds distance-based shaping:

```text
potential(s) = -0.1 × min(distance_to_goal(s), 40)
shaping = 0.95 × potential(next_state) - potential(state)
reward = base_reward + shaping
```

For a true goal or fall, the next potential is zero and the Q target does not bootstrap. An external time limit still bootstraps from its final observation. The potential is a heuristic: straight-line distance can point through obstacles, and it does not guarantee safe jumps or learning. The policy-preservation result for potential-based shaping assumes a suitable MDP and matching discount; approximated and partially observed state can violate those assumptions. Each step logs both shaped `reward` and `base_reward`, plus raw velocity, position and goal distance.

## Speed and evaluation

The code batches block reads and avoids re-sending button releases that are already off. It does **not** claim a measured speedup or exact two-tick action duration: Python-to-game calls and the game tick rate can dominate. Each episode's log records `steps` and `seconds`; measure steps per second on the actual machine and inspect timeouts. Increasing simulator speed or using asynchronous executors requires separate correctness testing.

Training saves `linear-q.json` in the game's `minescript` folder. A later run resumes it; move or delete the file to start a fresh run. Use `\agent 20 --eval --seed 8` to evaluate saved weights without updating them; `\agent 20 --random --seed 7` evaluates random actions without training. A seed controls Python's random choices, not all Minecraft timing or physics. Evaluation still needs multiple independent trained seeds, a scripted baseline, and held-out course layouts and start positions that were not used for tuning. Report success rate, falls, completion steps, and wall time. A successful single course is not general parkour.

## Sources

- The course's Lecture 09 covers function approximation, tile coding and instability with off-policy bootstrapping; Lecture 10 discusses evaluation.
- [Minescript documentation](https://minescript.net/docs) describes player data, batched block reads and control APIs.
- [Ng, Harada and Russell: Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) describes potential-based shaping.
- [Gymnasium: Handling Time Limits](https://gymnasium.farama.org/v1.1.1/tutorials/gymnasium_basics/handling_time_limits/) explains termination versus truncation.
- [Stable-Baselines3 RL tips](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) discusses evaluation, multiple runs and continuous observations.
