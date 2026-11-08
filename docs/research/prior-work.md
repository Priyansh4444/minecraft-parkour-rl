# Research beyond CS175

Reviewed November 7, 2026. This is a source review, not a reproduction of the external projects' results. Our code below uses library APIs and an original environment implementation; no external trained policy is imported.

## Minecraft-specific sources

### Marc Ducret: Imitation and Reinforcement Learning in Minecraft

This USC internship project includes a scientific report and an animated parkour experiment. Its parkour task uses randomly generated paths, a local block neighborhood, relative position and actor-critic learning. The report also discusses representing terrain spatially and separating Python decisions from environment stepping. Its parkour actions directly control horizontal velocity, so that task is not equivalent to our button-controlled jumps. Borrow local geometry, varied layouts and an explicit stepping interface; do not borrow its results as evidence for our courses.

- [Author's project and demonstrations](https://marc.ducret.io/project/minecraft_learning/)
- [Scientific report, sections 3.3.4, 4.1 and 4.2](https://marc.ducret.io/content/documents/minecraft_learning/report.pdf)

### MalmoRL: Minecraft-Reinforcement-Learning

The [parkour environment source](https://github.com/andrearante12/Minecraft-Reinforcement-Learning/blob/main/malmo/rl/envs/parkour_env.py) combines contact, directional motion, relative target offset and local voxels. Its [curriculum implementation guide](https://github.com/andrearante12/Minecraft-Reinforcement-Learning/blob/main/malmo/docs/framework/curriculum_training.md) describes sharing one observation/action contract across easier and harder sections. These are useful implementation references, not independently reproduced benchmarks. The source also falls back to zero terrain on missing observations; our current compact environment rejects malformed state and uses configured platform targets rather than a learned voxel input.

### Minecraft-RL-env

This [Mineflayer parkour framework](https://github.com/K3YB04RD-w4rri0r/Minecraft-RL-env) is relevant to our transport. Its README alone is not evidence of a successful general policy. We retain explicit observation validation, goal-based rewards and live tests instead of copying its placeholders.

## Parkour beyond Minecraft

### Extreme Parkour with Legged Robots

The [ICRA project's code](https://github.com/chengxuxin/extreme-parkour) explicitly combines motion/contact state, previous actions, terrain sensing, target direction and history. It trains with terrain variation and separates simulation training from deployment. These principles motivate checking motion/contact state and testing held-out layouts. Robot joint control, Isaac Gym parallel simulation and its reported training times do not transfer directly to Minecraft. We do not install its robotics stack or import its weights.

## Decisions used in this project

- Keep relative target position, signed velocity and actual grounded contact in the compact state. Both learners use the same seven bins; neither receives a 3D grid.
- Keep the five timed forward/jump actions from the proposal. Steering, vision and arbitrary-world navigation are outside the current task.
- Own the client step boundary and reject malformed observations, unexpected action durations and server corrections during an action. The server world continues running.
- Use a literal Q-table for the main experiment and a small Stable-Baselines3 PPO policy only as a comparison.
- Generate reproducible course variations, freeze test seeds before evaluation and separate exploratory training successes from no-update evaluation.
- Validate resets, learning updates, timeout handling, checkpoint reload and evaluation without updates in actual code and live server checks.

These sources supply ideas, not evidence that our learner succeeds. More features or a different algorithm do not guarantee fast learning. The [submitted scope](../project/proposal-scope.md) remains a short, straight course with ordinary platforms and numerical game-state inputs.
