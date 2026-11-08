"""Visual test — spin up heterogeneous bimanual environments one-by-one.

For each selected task the script:
    1. Creates the env with has_renderer=True (on-screen MuJoCo viewer).
    2. Runs random actions for N steps so you can inspect the scene.
    3. Prints obs keys, shapes, and task-score each episode.

Exits 1 if any task fails.

Usage:
    # Test all het-bi tasks (50 steps each):
    python scripts/tests/test_hetbi_envs.py

    # Test a single task for more steps:
    python scripts/tests/test_hetbi_envs.py --tasks ib-p1-plate-shelf-hetbi-v1 --steps 200

    # Test a subset:
    python scripts/tests/test_hetbi_envs.py --tasks ib-p1-bowl-table-v1 ib-p1-cup-inversion-v1
"""

import argparse
import sys
import time
import numpy as np
import imbench as ib
from imbench.utils.obs_visualizer import ObsDashboard


# All het-bi task IDs shipped with IMBench
HETBI_TASKS = [tid for tid in ib.list_tasks() if "hetbi" in tid or tid in {
    "ib-p1-bowl-table-v1", "ib-p1-cup-inversion-v1",
    "ib-p1-cracker-box-jaw-v1", "ib-p1-cracker-box-suction-v1",
}]


def print_obs_summary(obs: dict):
    """Print a compact summary of observation keys and shapes."""
    print("  Obs keys:")
    for k, v in sorted(obs.items()):
        if isinstance(v, np.ndarray):
            print(f"    {k:30s}  shape={str(v.shape):15s}  dtype={v.dtype}")
        else:
            print(f"    {k:30s}  value={v}")


def test_task(task_id: str, n_steps: int = 50, render: bool = True,
              dashboard: bool = False):
    """Instantiate one task, run random actions, and tear down."""
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id}")
    print(f"{'='*60}")

    try:
        env = ib.make(task_id, privileged_obs=True, render_mode="human" if render else None)
    except Exception as e:
        print(f"  [FAIL] Could not create env: {e}")
        return False

    try:
        obs, info = env.reset(seed=42)
        print(f"  Reset OK  |  task_score={info.get('task_score', 'N/A')}")
        print_obs_summary(obs)

        dash = ObsDashboard(window_name=f"Obs — {task_id}") if dashboard else None
        if dash:
            dash.update(obs, info)

        # Determine action dimension from the robosuite env
        action_lo, action_hi = env.action_space.low, env.action_space.high
        act_dim = action_lo.shape[0]
        print(f"  Action dim: {act_dim}")

        for step_i in range(n_steps):
            # Random action in valid range
            action = np.random.uniform(action_lo, action_hi)
            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()
            if dash:
                key = dash.update(obs, info, reward=reward)
                if key == ord('q'):
                    print("  Dashboard closed by user.")
                    break

            if step_i % 10 == 0 or step_i == n_steps - 1:
                score = info.get("task_score", 0.0)
                print(f"    step {step_i:4d}  reward={reward:+.4f}  score={score:.3f}")

            if terminated or truncated:
                print(f"  Episode ended at step {step_i} "
                      f"(terminated={terminated}, truncated={truncated})")
                break

        print(f"  [OK] {task_id} — {step_i + 1} steps completed")
        success = True

    except Exception as e:
        print(f"  [FAIL] Runtime error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    finally:
        try:
            if dash is not None:
                dash.close()
        except (NameError, UnboundLocalError):
            pass
        try:
            env.close()
        except Exception:
            pass
        # Small delay so the MuJoCo window closes cleanly
        time.sleep(0.3)

    return success


def main():
    parser = argparse.ArgumentParser(description="Visual test for het-bi envs")
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Task IDs to test (default: all het-bi tasks)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Steps per episode (default: 50)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Run headless (no MuJoCo viewer)",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Open the live observation dashboard (requires cv2)",
    )
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else HETBI_TASKS
    render = not args.no_render

    print(f"IMBench het-bi visual test")
    print(f"  Tasks to test : {len(tasks)}")
    print(f"  Steps/episode : {args.steps}")
    print(f"  Render        : {render}")
    print(f"  Registered het-bi tasks: {HETBI_TASKS}")

    results = {}
    for tid in tasks:
        if tid not in ib.list_tasks():
            print(f"\n  [SKIP] Unknown task: {tid}")
            results[tid] = False
            continue
        results[tid] = test_task(tid, n_steps=args.steps, render=render,
                                  dashboard=args.dashboard)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    for tid, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] {tid}")
    print(f"\n  {passed} passed, {failed} failed out of {len(results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
