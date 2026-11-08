"""Render the scripted PendulumGraspPolicy on the pendulum-grasp task.

Usage::

    python scripts/run_pendulum_grasp_policy.py
    python scripts/run_pendulum_grasp_policy.py --episodes 5 --seed 0
    python scripts/run_pendulum_grasp_policy.py --no-render --episodes 20

Success is ``info["is_success"]``.  Press ``q`` in the dashboard to quit;
the interrupted episode is not counted.
"""
import argparse
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import imbench as ib

from data_synthetic.single_arm_pipeline.pendulum_grasp_policy import (
    PendulumGraspPolicy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Per-episode step limit (default: the task YAML's max_episode_steps)")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless (no on-screen viewer / dashboard)")
    args = parser.parse_args()
    if not args.no_render and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("the viewer needs a display, but DISPLAY is not set; pass --no-render")

    env = ib.make(
        "ib-p2-pendulum-grasp-v1",
        privileged_obs=True,
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_steps,
    )
    policy = PendulumGraspPolicy(seed=args.seed)
    dash = None
    if not args.no_render:
        from imbench.utils.obs_visualizer import ObsDashboard
        dash = ObsDashboard()

    scores, successes = [], []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            policy.reset()
            if dash is not None:
                dash.reset()
                dash.update(obs, info)
            success = quit_requested = False
            done = False
            last_phase = None
            while not done:
                action = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                success = success or info["is_success"]
                if dash is not None:
                    env.render()
                if policy.phase != last_phase:
                    print(
                        f"  ep{ep} t={info['step']:4d} → {policy.phase}  "
                        f"side={policy._side:+d}  "
                        f"latched={policy._close_latched}"
                    )
                    last_phase = policy.phase
                if dash is not None and dash.update(obs, info, reward=reward) == ord("q"):
                    quit_requested = True
                    break
                done = terminated or truncated
            if quit_requested:
                print(f"Episode {ep:3d}  stopped at step {info['step']} (not counted)")
                break
            score = info.get("task_score", 0.0)
            scores.append(score)
            successes.append(success)
            print(
                f"Episode {ep:3d}  side={policy._side:+d}  "
                f"steps={info['step']:4d}  score={score:.2f}  "
                f"success={success}  phase={policy.phase}"
            )
    finally:
        if dash is not None:
            dash.close()
        env.close()

    if not scores:
        return
    print(
        f"\nMean: {np.mean(scores):.3f}  "
        f"Success: {np.mean(successes):.0%}  "
        f"Partial: {np.mean([s >= 0.4 for s in scores]):.0%}"
    )


if __name__ == "__main__":
    main()
