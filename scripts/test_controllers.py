"""Thorough controller compatibility and robustness test.

Tests all 6 robosuite controller types in both delta and absolute input
modes, for single-arm and bimanual (hetbi) tasks.

Tests performed:
    1. **Instantiation** — env creates + steps without error.
    2. **EEF drift** — zero-action hold, measures gravity-induced drift.
    3. **Hold-after-lift** — lift then hold, measures stability post-motion.

Usage
-----
    # Full headless sweep (all controllers × modes × tasks):
    python scripts/test_controllers.py

    # Only compatibility (fast — skip drift/hold tests):
    python scripts/test_controllers.py --compat-only

    # Run all tests with MuJoCo viewer:
    python scripts/test_controllers.py --render

    # Run all tests with MuJoCo viewer + observation dashboard:
    python scripts/test_controllers.py --render --dashboard

    # Test a single controller config:
    python scripts/test_controllers.py --controller OSC_POSE --mode delta

    # Customize tasks and steps:
    python scripts/test_controllers.py --single-task ib-p2-cube-toss-v1 \
                                       --bimanual-task ib-p1-bowl-table-v1 \
                                       --steps 300
"""

import argparse
import sys
import time
import numpy as np
import imbench as ib
from imbench.utils.obs_visualizer import ObsDashboard


# ── All supported controllers ────────────────────────────────────────
CONTROLLER_TYPES = [
    "OSC_POSE",
    "OSC_POSITION",
    "IK_POSE",
    "JOINT_POSITION",
    "JOINT_VELOCITY",
    "JOINT_TORQUE",
]

INPUT_MODES = ["delta", "absolute"]


# ── Helpers ──────────────────────────────────────────────────────────

def _make_overrides(ctrl_type, input_mode, n_robots=1, kp=None):
    """Build config_overrides dict for a controller type + input mode."""
    entry = {"type": ctrl_type, "robot": "Panda", "input_type": input_mode}
    if kp is not None:
        entry["kp"] = kp
    if n_robots == 1:
        return {"controller": entry}
    return {"controller": [dict(entry) for _ in range(n_robots)]}


def _make_env(task_id, overrides, render=False):
    """Create env, optionally with rendering."""
    return ib.make(
        task_id,
        privileged_obs=True,
        render_mode="human" if render else None,
        config_overrides=overrides,
    )


def _make_dash(ctrl_type, input_mode, dashboard=False):
    """Create an ObsDashboard if requested."""
    if not dashboard:
        return None
    return ObsDashboard(window_name=f"Obs — {ctrl_type} ({input_mode})")


def _step_with_viz(env, action, dash, render):
    """Step env + optional render + dashboard update.

    Returns (obs, reward, terminated, truncated, info, quit_requested).
    """
    obs, reward, term, trunc, info = env.step(action)
    if render:
        env.render()
    quit_requested = False
    if dash:
        key = dash.update(obs, info, reward=reward)
        if key == ord('q'):
            quit_requested = True
    return obs, reward, term, trunc, info, quit_requested


def _cleanup(env, dash):
    """Close env and dashboard."""
    if dash:
        dash.close()
    env.close()


# ── Test 1: Instantiation compat ────────────────────────────────────

def test_instantiation(task_id, ctrl_type, input_mode, n_robots=1,
                       n_steps=5, render=False, dashboard=False):
    """Create env, reset, step a few times. Returns (ok, act_dim, error)."""
    overrides = _make_overrides(ctrl_type, input_mode, n_robots)
    try:
        env = _make_env(task_id, overrides, render)
        obs, info = env.reset(seed=0)
        dash = _make_dash(ctrl_type, input_mode, dashboard)
        if dash:
            dash.update(obs, info)
        lo, hi = env.action_space.low, env.action_space.high
        for _ in range(n_steps):
            act = np.random.uniform(lo, hi)
            obs, rew, term, trunc, info, quit = _step_with_viz(
                env, act, dash, render,
            )
            if quit or term or trunc:
                break
        _cleanup(env, dash)
        return True, lo.shape[0], None
    except Exception as e:
        return False, 0, str(e).split("\n")[0][:100]


# ── Test 2: EEF drift ───────────────────────────────────────────────

def test_drift(task_id, ctrl_type, input_mode, kp=None, n_steps=200,
               render=False, dashboard=False):
    """Send zero actions and measure EEF drift from rest."""
    overrides = _make_overrides(ctrl_type, input_mode, n_robots=1, kp=kp)
    env = _make_env(task_id, overrides, render)
    obs, info = env.reset(seed=0)
    dash = _make_dash(ctrl_type, input_mode, dashboard)
    if dash:
        dash.update(obs, info)

    raw = env._env._get_observations()
    start_eef = raw["robot0_eef_pos"].copy()

    traj = [start_eef.copy()]
    act_dim = env.action_space.shape[0]
    for _ in range(n_steps):
        obs, rew, term, trunc, info, quit = _step_with_viz(
            env, np.zeros(act_dim), dash, render,
        )
        raw = env._env._get_observations()
        traj.append(raw["robot0_eef_pos"].copy())
        if quit or term or trunc:
            break

    traj = np.array(traj)
    end_eef = traj[-1]
    _cleanup(env, dash)

    return {
        "total_drift": np.linalg.norm(end_eef - start_eef),
        "z_drift": start_eef[2] - end_eef[2],
        "max_deviation": np.max(np.linalg.norm(traj - start_eef, axis=1)),
    }


# ── Test 3: Hold after lift ─────────────────────────────────────────

def test_hold_after_lift(task_id, ctrl_type, input_mode, kp=None,
                         lift_steps=100, hold_steps=200,
                         render=False, dashboard=False):
    """Lift EEF upward then hold — measures stability post-motion.

    Only meaningful for task-space controllers (OSC_POSE, OSC_POSITION,
    IK_POSE) where action[2] maps to Z movement.
    """
    overrides = _make_overrides(ctrl_type, input_mode, n_robots=1, kp=kp)
    env = _make_env(task_id, overrides, render)
    obs, info = env.reset(seed=0)
    dash = _make_dash(ctrl_type, input_mode, dashboard)
    if dash:
        dash.update(obs, info)

    act_dim = env.action_space.shape[0]

    # Phase 1: push Z up
    for _ in range(lift_steps):
        action = np.zeros(act_dim)
        action[2] = 0.5  # Z-up
        if act_dim >= 7:
            action[-1] = -1.0  # gripper open
        obs, rew, term, trunc, info, quit = _step_with_viz(
            env, action, dash, render,
        )
        if quit or term or trunc:
            break

    raw = env._env._get_observations()
    hold_start = raw["robot0_eef_pos"].copy()

    # Phase 2: hold (zero actions)
    traj = [hold_start.copy()]
    for _ in range(hold_steps):
        obs, rew, term, trunc, info, quit = _step_with_viz(
            env, np.zeros(act_dim), dash, render,
        )
        raw = env._env._get_observations()
        traj.append(raw["robot0_eef_pos"].copy())
        if quit or term or trunc:
            break

    traj = np.array(traj)
    hold_end = traj[-1]
    _cleanup(env, dash)

    return {
        "hold_drift": np.linalg.norm(hold_end - hold_start),
        "z_sag": hold_start[2] - hold_end[2],
        "max_deviation": np.max(np.linalg.norm(traj - hold_start, axis=1)),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Thorough controller compatibility and robustness test",
    )
    parser.add_argument(
        "--single-task", default="ib-p2-cube-toss-v1",
        help="Single-arm task for testing (default: ib-p2-cube-toss-v1)",
    )
    parser.add_argument(
        "--bimanual-task", default="ib-p1-bowl-table-v1",
        help="Bimanual task for testing (default: ib-p1-bowl-table-v1)",
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Steps per drift/hold test (default: 200)",
    )
    parser.add_argument(
        "--compat-steps", type=int, default=5,
        help="Steps per controller config in instantiation test (default: 5)",
    )
    parser.add_argument(
        "--compat-only", action="store_true",
        help="Only run instantiation tests (skip drift/hold)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Show MuJoCo viewer during all tests",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Open the live observation dashboard during all tests (requires cv2)",
    )
    parser.add_argument(
        "--kp", type=int, default=None,
        help="Override kp gain (for drift/hold tests)",
    )
    parser.add_argument(
        "--controller", default=None,
        choices=CONTROLLER_TYPES,
        help="Test only this controller type (default: all)",
    )
    parser.add_argument(
        "--mode", default=None,
        choices=INPUT_MODES,
        help="Test only this input mode (default: both delta & absolute)",
    )
    args = parser.parse_args()

    render = args.render
    dashboard = args.dashboard
    ctrl_filter = [args.controller] if args.controller else CONTROLLER_TYPES
    mode_filter = [args.mode] if args.mode else INPUT_MODES

    passed = 0
    failed = 0
    t0 = time.time()

    # ==================================================================
    #  TEST 1: Instantiation compatibility (all combos)
    # ==================================================================
    print("=" * 78)
    print("  TEST 1: Controller instantiation & stepping")
    print("=" * 78)

    setups = [
        ("Single-arm", args.single_task, 1),
        ("Bimanual",   args.bimanual_task, 2),
    ]

    for setup_label, task_id, n_robots in setups:
        print(f"\n  {setup_label}: {task_id}")
        print(f"  {'Controller':<20} {'Mode':<10} {'Status':<8} {'act_dim':<10} {'Note'}")
        print("  " + "-" * 70)

        for ctrl in ctrl_filter:
            for mode in mode_filter:
                ok, act_dim, err = test_instantiation(
                    task_id, ctrl, mode, n_robots,
                    n_steps=args.compat_steps,
                    render=render, dashboard=dashboard,
                )
                if ok:
                    passed += 1
                    print(f"  {ctrl:<20} {mode:<10} {'OK':<8} {act_dim:<10}")
                else:
                    failed += 1
                    print(f"  {ctrl:<20} {mode:<10} {'FAIL':<8} {'':<10} {err}")

    if args.compat_only:
        _print_summary(passed, failed, t0)
        return

    # ==================================================================
    #  TEST 2: EEF drift (single-arm, selected controllers)
    # ==================================================================
    # Drift test: zero actions. For task-space controllers this means
    # "hold position". For JOINT_TORQUE, zero torque + gravity comp
    # also holds. JOINT_VELOCITY zero means zero velocity = hold.
    DRIFT_CONTROLLERS = [
        ("OSC_POSE",       "delta",  None),
        ("OSC_POSE",       "delta",  200),
        ("OSC_POSE",       "delta",  300),
        ("OSC_POSITION",   "delta",  None),
        ("IK_POSE",        "delta",  None),
        ("JOINT_POSITION", "delta",  None),
    ]
    if args.controller:
        DRIFT_CONTROLLERS = [(c, m, kp) for c, m, kp in DRIFT_CONTROLLERS
                             if c == args.controller]
    if args.mode:
        DRIFT_CONTROLLERS = [(c, m, kp) for c, m, kp in DRIFT_CONTROLLERS
                             if m == args.mode]

    print()
    print("=" * 78)
    print(f"  TEST 2: EEF drift from rest — zero actions for {args.steps} steps")
    print(f"  Task: {args.single_task}")
    print("=" * 78)
    print(f"  {'Controller':<22} {'kp':>4} {'Total drift':>13} {'Z drift':>11} {'Max dev':>11}")
    print("  " + "-" * 70)

    for ctrl, mode, kp in DRIFT_CONTROLLERS:
        effective_kp = args.kp if args.kp is not None else kp
        kp_str = str(effective_kp) if effective_kp else "def"
        try:
            r = test_drift(args.single_task, ctrl, mode, kp=effective_kp,
                           n_steps=args.steps, render=render, dashboard=dashboard)
            print(f"  {ctrl:<22} {kp_str:>4} "
                  f"{r['total_drift']:>12.5f}m "
                  f"{r['z_drift']:>10.5f}m "
                  f"{r['max_deviation']:>10.5f}m")
        except Exception as e:
            err = str(e).split("\n")[0][:50]
            print(f"  {ctrl:<22} {kp_str:>4} {'ERROR':>13} — {err}")

    # ==================================================================
    #  TEST 3: Hold after lift (task-space controllers)
    # ==================================================================
    HOLD_CONTROLLERS = [
        ("OSC_POSE",     "delta",  200),
        ("OSC_POSITION", "delta",  None),
        ("IK_POSE",      "delta",  None),
    ]
    if args.controller:
        HOLD_CONTROLLERS = [(c, m, kp) for c, m, kp in HOLD_CONTROLLERS
                            if c == args.controller]
    if args.mode:
        HOLD_CONTROLLERS = [(c, m, kp) for c, m, kp in HOLD_CONTROLLERS
                            if m == args.mode]

    lift = args.steps // 2
    hold = args.steps

    print()
    print("=" * 78)
    print(f"  TEST 3: Hold stability after lift  (lift={lift} steps, hold={hold} steps)")
    print(f"  Task: {args.single_task}")
    print("=" * 78)
    print(f"  {'Controller':<22} {'kp':>4} {'Hold drift':>13} {'Z sag':>11} {'Max dev':>11}")
    print("  " + "-" * 70)

    for ctrl, mode, kp in HOLD_CONTROLLERS:
        effective_kp = args.kp if args.kp is not None else kp
        kp_str = str(effective_kp) if effective_kp else "def"
        try:
            r = test_hold_after_lift(args.single_task, ctrl, mode,
                                     kp=effective_kp,
                                     lift_steps=lift, hold_steps=hold,
                                     render=render, dashboard=dashboard)
            print(f"  {ctrl:<22} {kp_str:>4} "
                  f"{r['hold_drift']:>12.5f}m "
                  f"{r['z_sag']:>10.5f}m "
                  f"{r['max_deviation']:>10.5f}m")
        except Exception as e:
            err = str(e).split("\n")[0][:50]
            print(f"  {ctrl:<22} {kp_str:>4} {'ERROR':>13} — {err}")

    print()
    print("  Lower values = more robust.  Negative Z drift/sag = drifted upward.")
    _print_summary(passed, failed, t0)


def _print_summary(passed, failed, t0):
    elapsed = time.time() - t0
    total = passed + failed
    print()
    print("=" * 78)
    print(f"  SUMMARY:  {passed}/{total} passed,  {failed} failed  "
          f"({elapsed:.1f}s)")
    print("=" * 78)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
