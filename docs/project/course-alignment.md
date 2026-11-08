# Keep the implementation within CS175

This table connects the implementation to the CS175 lecture slides in `course_files_export.zip`.

| Project part | Course material | Implementation choice |
| --- | --- | --- |
| State, action, reward, episode | Lectures 01–03 | Small numerical state, five timed movement/jump actions and a goal/fall/time-limit outcome. |
| Tabular Q-learning | Lecture 07, slides 4, 9 and 12 | One row per complete binned state; epsilon-greedy behavior; a max-over-actions TD target. |
| Timeout handling | Lecture 07, slides 12 and 27 | Stop the episode on a timeout, but keep bootstrapping; zero the future value only for a true goal/fall terminal. |
| Approximate state | Lecture 09, slides 17–18 | Binning is an approximation. The main learner uses a literal table rather than adding a function approximator. |
| Optional PPO comparison | Lectures 11–13, especially Lecture 13 slides 11–17 | Permitted if useful, using a small policy and the same task/input/actions/rewards. No new navigation, vision or control problem. |

The timeout rule is covered by a focused regression. A one-step regression with reward -0.01, next-state max value 2, gamma 0.99 and alpha 0.1 previously updated a zero Q cell to -0.001. The lecture's target gives 0.197, because the cutoff is not a true terminal state. The fixed test exercises the complete episode-to-table-update path. Goal and fall rewards and their zero bootstrap are unchanged.

Checkpoint schema version 3 records this rule and rejects older tables. Independent training runs use fresh tables so results are not mixed across learning rules.

The three-jump scripted controller is calibrated only on the training layout, then frozen as preset actions. Reserved starting positions and layouts remain out of calibration and training. This is an environment/baseline check, not extra training data for the learned policy or proof of generalization.

Keep the initial work to a single jump, then the proposed 3–5-jump course, ordinary platforms, the specified sparse rewards and the optional one-time platform bonus comparison. No hunger/potions/neos/headhitters, pretrained general policies or extra algorithms are required for the main result. The headless bot/server is infrastructure and still needs live timing/reset observations.
