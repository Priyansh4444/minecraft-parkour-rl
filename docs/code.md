# Understanding the code

The project is an early, straight-line, one-gap parkour prototype. The official-launcher single-player workflow has not been tested end to end. Timing, resets, observations, and learning results require more testing before any claims of performance or generalization.

## Files and flow

- `parkour/build_course.py` sends Minecraft commands to build the start, gap, catch floor, and goal in a new Superflat world. It uses fixed coordinates.
- `parkour/course.json` defines the start, facing direction, goal, fall threshold, and goal radius for the agent. Keep it consistent with the builder's coordinates.
- `parkour/agent.py` observes, acts, computes rewards, updates the Q table, and repeats episodes. It runs as a Minescript command inside the game.

An episode resets the player, chooses and performs actions, and ends on a fall, goal, or time limit. A run is a batch of episodes. The external Minecraft Launcher starts the game; Fabric loads Minescript, which exposes game state and controls to the Python scripts.

## State and observation

`observe` reads position and vertical velocity through Minescript, estimates horizontal speed from position changes, and calls `scan` to sample blocks ahead. It groups values into six small categories, or **bins**, and joins them into a string for the Q-table key:

| Field | Meaning | Why it matters |
| --- | --- | --- |
| `lip` | Distance category to the near edge of the next gap | Timing takeoff |
| `gap` | Estimated number of empty columns ahead | Clearing the gap |
| `height` | Landing surface estimated lower, similar, or higher than the player | Height changes |
| `speed` | Horizontal speed category | Momentum at takeoff and landing |
| `vy` | Vertical velocity category | Rising or falling |
| `ground` | Estimated whether the player is on a block | Whether a jump can start |

For example, `4,2,1,0,1,1` describes six categories, not six coordinates. `scan` samples a narrow line in the configured direction and ignores the safety floor. It does not understand arbitrary block shapes, corners, sideways motion or full 3D geometry. Unknown landing height shares the similar-height bin. Ground contact is estimated using blocks, position and velocity. Two different situations can fall into one bin yet need different actions, so this representation is **not guaranteed Markov** and has not demonstrated generalization.

## Actions and learning

`choose` selects one of five actions: `idle`, `walk`, `run`, `walk_jump`, or `run_jump`. Each presses a combination of forward, sprint, and jump buttons. With probability epsilon it chooses a random action; otherwise it chooses a highest-valued action, breaking ties randomly. Unseen states also produce a random action.

A Q-table maps each state to five estimated action values. After an action, the agent updates the selected value:

```text
target = reward + discount × best value in the next state
Q[state, action] += learning_rate × (target - Q[state, action])
```

For a goal or fall, the target is just the reward: there is no future value after a terminal event. At the external step limit, the last update still bootstraps from the next observation. The current parameters are learning rate `ALPHA = 0.25`, discount `GAMMA = 0.95`, and at most 80 decisions per episode. Actual button timing needs checking: the game and script run asynchronously.

Ordinary reward is `2 × forward progress - 0.01`; reaching the goal replaces it with `+10` and falling with `-10`. A one-time `+3` may be added when the code estimates a gap was crossed. That crossing flag is not included in the Q-table state, and scan-based detection is approximate. The goal is a position near the target with estimated ground contact; it does not require touching the gold block itself. These details matter when evaluating what the agent actually learned.

`reset` releases controls, checks that the player is alive, teleports to the configured start, and waits for position and horizontal speed to settle. The catch floor is intended to help with falls, but resets are not yet proven reliable. The table and logs are local to the game directory. Training reuses the table; deliberately move or delete it to start fresh. There is no separate evaluation mode, fixed seed, or controlled evaluation harness yet.
