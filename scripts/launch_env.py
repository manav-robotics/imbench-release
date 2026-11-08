"""Launch an IMBench environment with an on-screen viewer.

Usage:
    # Launch a specific task with the MuJoCo viewer:
    python scripts/launch_env.py ib-p1-bowl-table-v1

    # Run for 500 steps instead of the default 200:
    python scripts/launch_env.py ib-p1-bowl-table-v1 --steps 500

    # List all available tasks:
    python scripts/launch_env.py --list

With ``--dashboard``, press q in the dashboard window to quit.
"""

import argparse
import numpy as np
import imbench as ib
from imbench.utils.obs_visualizer import ObsDashboard


def main():
    parser = argparse.ArgumentParser(
        description="Launch an IMBench environment with rendering",
    )
    parser.add_argument(
        "task_id", nargs="?", default=None,
        help="Task ID to launch (e.g. ib-p1-bowl-table-v1)",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of random-action steps per episode (default: 200)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available task IDs and exit",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Open the live observation dashboard (requires cv2)",
    )
    parser.add_argument(
        "--cameras-only", action="store_true",
        help="Dashboard shows only camera feeds (no F/T or tactile panels)",
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="Open the dashboard window in fullscreen mode",
    )
    parser.add_argument(
        "--render-camera", type=str, default=None,
        help="Camera name to use as the default MuJoCo viewer perspective "
             "(e.g. frontview, birdview, agentview). Omit for free camera.",
    )
    parser.add_argument(
        "--obs-mode", type=str, default="combo",
        choices=["proprio", "combo", "rgb"],
        help="Observation mode: 'combo' is proprio + cameras (default); "
             "'proprio' skips camera rendering and is faster",
    )
    args = parser.parse_args()

    if args.list or args.task_id is None:
        tasks = ib.list_tasks()
        print("Available tasks:")
        for tid in tasks:
            print(f"  {tid}")
        if args.task_id is None and not args.list:
            parser.error("task_id is required (or use --list)")
        return

    env = ib.make(args.task_id, render_mode="human", obs_mode=args.obs_mode,
                  config_overrides={"viewer_camera": args.render_camera} if args.render_camera else None)
    dash = ObsDashboard(cameras_only=args.cameras_only, fullscreen=args.fullscreen) if args.dashboard else None

    action_lo, action_hi = env.action_space.low, env.action_space.high
    act_dim = action_lo.shape[0]
    print(f"Task: {args.task_id}  |  action_dim={act_dim}")

    quit_requested = False
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        print(f"\nEpisode {ep}  |  task_score={info.get('task_score', 'N/A')}")
        env.render()  # create + maximize MuJoCo window before opening dashboard
        if dash:
            dash.reset()
            dash.update(obs, info)

        step_i = -1
        for step_i in range(args.steps):
            action = np.random.uniform(action_lo, action_hi)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if dash:
                key = dash.update(obs, info, reward=reward)
                if key == ord('q'):
                    print("  Dashboard closed by user.")
                    quit_requested = True
                    break
            if terminated or truncated:
                print(f"  Ended at step {step_i} "
                      f"(terminated={terminated}, truncated={truncated})")
                break

        score = info.get("task_score", 0.0)
        print(f"  Finished  |  steps={step_i + 1}  score={score:.3f}")
        if quit_requested:
            break

    if dash:
        dash.close()
    env.close()


if __name__ == "__main__":
    main()
