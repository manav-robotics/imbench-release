"""Render the scripted OccluderPushPolicy on the occluder-push task.

Usage::

    python scripts/run_occluder_push_policy.py
    python scripts/run_occluder_push_policy.py --episodes 20 --seed 0 --quiet
    python scripts/run_occluder_push_policy.py --no-render --quiet --episodes 20

Episode ``i`` resets the task and seeds the policy's RNG with ``--seed + i``,
as the collectors do, so runs are reproducible.  Success is
``info["is_success"]``.  Press ``q`` in the dashboard to quit; the
interrupted episode is not counted.
"""
import argparse
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import imbench as ib

from data_synthetic.single_arm_pipeline.occluder_push_policy import (
    OccluderPushPolicy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Per-episode step limit (default: the task YAML's max_episode_steps)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-phase policy logs and per-tick prints")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless (no on-screen viewer / dashboard)")
    args = parser.parse_args()
    if not args.no_render and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        parser.error("the viewer needs a display, but DISPLAY is not set; pass --no-render")

    env = ib.make(
        "ib-p5-occluder-push-v1",
        privileged_obs=True,
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_steps,
    )

    dash = None
    if not args.no_render:
        from imbench.utils.obs_visualizer import ObsDashboard
        dash = ObsDashboard()

    scores, successes = [], []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, info = env.reset(seed=seed)
            policy = OccluderPushPolicy(verbose=not args.quiet, rng=np.random.default_rng(seed))
            policy.reset()
            if dash is not None:
                dash.reset()
                dash.update(obs, info)
            success = quit_requested = False
            done = False
            last_phase = None
            while not done:
                action = policy.act(obs, env=env)
                obs, reward, terminated, truncated, info = env.step(action)
                success = success or info["is_success"]
                if dash is not None:
                    env.render()
                if not args.quiet and policy.phase != last_phase:
                    print(f"  ep{ep} t={info.get('step', 0):4d} → {policy.phase}")
                    last_phase = policy.phase
                if dash is not None:
                    key = dash.update(obs, info, reward=reward)
                    if key == ord("q"):
                        quit_requested = True
                        break
                done = terminated or truncated
            if quit_requested:
                print(f"Episode {ep:3d}  stopped at step {info.get('step', 0)} (not counted)")
                break
            score = info.get("task_score", 0.0)
            scores.append(score)
            successes.append(success)
            print(
                f"Episode {ep:3d}  steps={info.get('step', 0):4d}  "
                f"score={score:.2f}  success={success}  phase={policy.phase}"
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
        f"({sum(successes)}/{len(successes)})"
    )


if __name__ == "__main__":
    main()
