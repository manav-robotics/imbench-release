"""Run a uniform-random policy on a task and count its successes.

Use it to check that a task's success condition does not trigger by chance
when contributing a new task. An episode counts as a success when
``info["is_success"]`` becomes true; it runs for the task's own
``max_episode_steps`` unless ``--max-steps`` is given.

Usage::

    python scripts/run_random_policy.py --task ib-p2-cube-toss-v1 --no-render
    python scripts/run_random_policy.py --task ib-p7-packing-v1 --episodes 5

Press ``q`` in the dashboard to stop; the interrupted episode is not counted.
"""

import argparse
import os
import sys

import numpy as np
import imbench as ib
from imbench.policy.base_policy import PolicyInterface
from imbench.tasks.base_task import BaseTask


class RandomPolicy(PolicyInterface):
    """Uniform actions in ``[-1, 1]``; ``reset(seed=...)`` reseeds the RNG."""

    def __init__(self, action_dim=7, seed=None):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> np.ndarray:
        return self.rng.uniform(-1, 1, size=self.action_dim)


def main():
    parser = argparse.ArgumentParser(
        description="Run a uniform-random policy on a task and count its successes.",
    )
    parser.add_argument("--task", required=True, help="Task ID (see imbench.list_tasks())")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes (default: 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Episode i resets the task and the policy with seed + i (default: 0)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Per-episode step limit (default: the task YAML's max_episode_steps)")
    parser.add_argument("--obs-mode", choices=BaseTask.OBS_MODES, default="combo",
                        help="Observation mode (default: combo = proprio + cameras)")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless, without the observation dashboard")
    args = parser.parse_args()
    if args.task not in ib.list_tasks():
        parser.error(f"unknown task {args.task!r}; see imbench.list_tasks()")
    if not args.no_render and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("the dashboard needs a display, but DISPLAY is not set; pass --no-render")

    env = ib.make(args.task, obs_mode=args.obs_mode, render_mode=None,
                  max_episode_steps=args.max_steps)
    policy = RandomPolicy(action_dim=env.action_space.shape[0])
    dash = None
    if not args.no_render:
        from imbench.utils.obs_visualizer import ObsDashboard
        dash = ObsDashboard()

    successes = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            policy.reset(seed=args.seed + ep)
            if dash is not None:
                dash.reset()
                dash.update(obs, info)
            success = quit_requested = False
            done = False
            while not done:
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                success = success or info["is_success"]
                done = terminated or truncated
                if dash is not None and dash.update(obs, info, reward=reward) == ord("q"):
                    quit_requested = True
                    break
            if quit_requested:
                print(f"Episode {ep}  stopped at step {info['step']} (not counted)")
                break
            successes.append(success)
            print(f"Episode {ep}  steps={info['step']}  score={info['task_score']:.3f}  "
                  f"success={success}")
    finally:
        if dash is not None:
            dash.close()
        env.close()

    print(f"\n{args.task}: success in {sum(successes)}/{len(successes)} random episodes")


if __name__ == "__main__":
    main()
