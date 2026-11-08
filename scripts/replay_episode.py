#!/usr/bin/env python3
"""Replay a recorded demonstration in the simulator.

Loads an episode from a LeRobot v2.1 dataset and replays it through the
environment using the recorded actions (``env.step(action)``), which
gives full physics simulation including grasping and object interaction.
The scene is reset with the episode's recorded ``seed`` and the recorded
number of unrecorded zero-action ``warmup_steps`` is taken first, exactly
as ``collect_demos.py`` did, so the replay reproduces the recording.

Each stream runs at its recorded frequency:
  - Actions are stepped at the control rate (e.g. 100 Hz)
  - The viewer renders at the camera rate (e.g. 25 fps)
  - The obs dashboard updates at the camera rate

Falls back to joint-position replay (kinematics only, no grasping)
if the dataset was collected without actions.

Works on raw collection output (``demos/<task>/M<id>/``, files named
``episode_NNNNNN``) and on merged datasets (files named ``episode_MMCCCCCC``);
``--episode`` is the ``episode_index`` listed in ``meta/episodes.jsonl``.

Usage::

    # Replay every episode of a dataset
    python scripts/replay_episode.py demos/ib-p2-cube-toss-v1/M01/

    # Replay a specific episode
    python scripts/replay_episode.py demos/ib-p2-cube-toss-v1/M01/ --episode 3

    # Replay at half speed
    python scripts/replay_episode.py demos/ib-p2-cube-toss-v1/M01/ --speed 0.5

    # With the obs dashboard
    python scripts/replay_episode.py demos/ib-p2-cube-toss-v1/M01/ --dashboard

    # Headless: re-simulate and compare the final score with the recording
    python scripts/replay_episode.py ~/processed_runs/ib-p2-cube-toss-v1 --no-render --speed 0
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

import imbench as ib


def _read_dataset_meta(dataset_root):
    """Read dataset-level metadata.

    Returns
    -------
    info : dict
        Contents of meta/info.json (task_id, control_freq, save_freq, …).
    episodes : list[dict]
        One dict per episode from meta/episodes.jsonl.
    """
    info_path = os.path.join(dataset_root, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    ep_jsonl = os.path.join(dataset_root, "meta", "episodes.jsonl")
    episodes = []
    with open(ep_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    return info, episodes


def _find_episode_parquet(dataset_root, episode_index):
    """Return the parquet path for *episode_index* regardless of filename width.

    Collection writes ``episode_NNNNNN.parquet``; merged datasets use
    ``episode_MMCCCCCC.parquet``.  Files are matched on the integer index.
    """
    pattern = os.path.join(dataset_root, "data", "chunk-*", "episode_*.parquet")
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            if int(stem.split("_", 1)[1]) == episode_index:
                return path
        except (IndexError, ValueError):
            continue
    raise FileNotFoundError(f"no parquet for episode_index {episode_index} matching {pattern}")


def _episode_metadata(dataset_root, episode_index):
    """Return the per-episode ``meta/episode_*_metadata.json`` contents, or {}."""
    for width in (6, 8):
        path = os.path.join(
            dataset_root, "meta", f"episode_{episode_index:0{width}d}_metadata.json",
        )
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def _reset_protocol(dataset_root, ep_meta, episode_index):
    """Return ``(seed, warmup_steps)`` recorded for an episode, or ``(None, None)``."""
    seed, warmup = ep_meta.get("seed"), ep_meta.get("warmup_steps")
    if seed is None or warmup is None:
        extra = _episode_metadata(dataset_root, episode_index)
        seed = extra.get("seed") if seed is None else seed
        warmup = extra.get("warmup_steps") if warmup is None else warmup
    if seed is None or warmup is None:
        return None, None
    return int(seed), int(warmup)


def _replay_config(dataset_root, info, ep_meta):
    """Restore collection settings, including the vocabulary used for this episode."""
    config = ep_meta.get("task_config")
    if config is None:
        extra = _episode_metadata(dataset_root, ep_meta["episode_index"])
        config = extra.get("task_config") or extra.get("dataset_summary", {}).get("task_config")
    if config is None:
        config = info.get("task_config", {})
    return {**config, "max_episode_steps": 999_999}


def _extract_columns(table, prefix):
    """Extract sorted columns matching ``prefix.0``, ``prefix.1``, … into a 2-D array.

    Returns ``None`` if no matching columns exist. Rows where all values
    are null are excluded.
    """
    cols = sorted(
        [c for c in table.column_names if c.startswith(f"{prefix}.")],
        key=lambda c: int(c.rsplit(".", 1)[1]),
    )
    if not cols:
        return None

    col_lists = [table.column(c).to_pylist() for c in cols]
    n_rows = len(col_lists[0])
    valid = [i for i in range(n_rows) if col_lists[0][i] is not None]
    return np.array(
        [[col_lists[j][i] for j in range(len(cols))] for i in valid],
        dtype=np.float64,
    )


def _read_episode(dataset_root, episode_index):
    """Read actions and joint_positions from a LeRobot dataset episode.

    Returns
    -------
    actions : np.ndarray | None
        (T, action_dim) if the dataset contains actions, else None.
    joint_positions : np.ndarray | None
        (T, D) if the dataset contains joint_positions, else None.
    """
    table = pq.read_table(_find_episode_parquet(dataset_root, episode_index))

    actions = _extract_columns(table, "action")
    joint_positions = _extract_columns(table, "observation.joint_positions")

    return actions, joint_positions


def replay_one(dataset_root, info, ep_meta, speed, use_dashboard, render=True):
    """Replay a single episode from a LeRobot dataset.

    Returns the replayed task score at the first terminating step (where the
    recorded ``final_score`` is taken; teleop episodes record cooldown steps
    after it), else at the last step; ``None`` if nothing was replayed.
    """
    task_id = info["task_id"]
    episode_index = ep_meta["episode_index"]
    n_steps = ep_meta["length"]
    final_score = ep_meta.get("final_score")

    actions, joint_positions = _read_episode(dataset_root, episode_index)

    if actions is None and joint_positions is None:
        print(f"  episode {episode_index}: no actions or joint_positions found, skipping")
        return None

    use_actions = actions is not None
    n_frames = len(actions) if use_actions else len(joint_positions)

    seed, warmup_steps = _reset_protocol(dataset_root, ep_meta, episode_index)
    if seed is None:
        print(f"  WARNING: episode {episode_index} has no recorded seed / warmup_steps; "
              "replaying with seed=0 and no warm-up, which may not match the recording")
        seed, warmup_steps = 0, 0

    # ── Frequencies from dataset metadata ───────────────────────────
    save_freq = info.get("save_freq", {})
    control_freq = info.get("control_freq", 100)
    camera_freq = save_freq.get("cameras", control_freq)

    # How many control steps between each viewer/dashboard frame
    render_skip = max(1, control_freq // camera_freq)

    # Wall-clock delay per control step
    dt = 1.0 / control_freq
    frame_delay = dt / speed if speed > 0 else 0

    mode = "action replay (full physics)" if use_actions else "joint position replay (kinematics only)"

    print(f"  dataset : {dataset_root}")
    print(f"  episode : {episode_index}")
    print(f"  task    : {task_id}")
    print(f"  steps   : {n_steps}")
    print(f"  reset   : seed={seed}  warmup_steps={warmup_steps}")
    print(f"  speed   : {speed}x")
    print(f"  mode    : {mode}")
    print(f"  control : {control_freq} Hz")
    print(f"  viewer  : {camera_freq} Hz  (render every {render_skip} steps)")

    env = ib.make(
        task_id,
        privileged_obs=True,
        obs_mode="combo",
        render_mode="human" if render else None,
        config_overrides=_replay_config(dataset_root, info, ep_meta),
    )

    dashboard = None
    if use_dashboard:
        from imbench.utils.obs_visualizer import ObsDashboard
        dashboard = ObsDashboard(
            window_name=f"Replay: episode {episode_index}",
        )

    obs, info_env = env.reset(seed=seed)
    action_dim = env.action_space.shape[0]
    for _ in range(warmup_steps):
        obs, _, _, _, info_env = env.step(np.zeros(action_dim))
    if dashboard:
        dashboard.update(obs, info_env, reward=0.0)

    # Pre-compute robot joint index arrays for joint-position fallback
    if not use_actions:
        sim = env._env.sim
        robot_indices = []
        for robot in env._env.robots:
            arm_idx = np.array(robot._ref_joint_pos_indexes)
            grip_dict = robot._ref_gripper_joint_pos_indexes
            grip_idx = np.array([i for idxs in grip_dict.values() for i in idxs])
            robot_indices.append((arm_idx, grip_idx))

    reward = 0.0
    end_score = None
    for t in range(n_frames):
        t_start = time.perf_counter()

        if use_actions:
            # ── Action replay: full physics with grasping ──────────
            obs, reward, terminated, truncated, info_env = env.step(actions[t])
            if terminated and end_score is None:
                end_score = info_env["task_score"]
        else:
            # ── Joint position fallback: kinematics only ───────────
            offset = 0
            for arm_idx, grip_idx in robot_indices:
                n_arm = len(arm_idx)
                n_grip = len(grip_idx)
                sim.data.qpos[arm_idx] = joint_positions[t, offset:offset + n_arm]
                offset += n_arm
                if n_grip > 0:
                    sim.data.qpos[grip_idx] = joint_positions[t, offset:offset + n_grip]
                    offset += n_grip
            sim.forward()

        # ── Render viewer at camera rate ────────────────────────────
        if t % render_skip == 0:
            if render and use_actions:
                env.render()
            # robosuite render() is a no-op for mjviewer, use viewer directly
            if render and env._env.viewer is not None:
                env._env.viewer.update()

            if dashboard:
                if not use_actions:
                    # qpos was written directly: bypass the per-step obs cache
                    obs = env._rebuild_obs()
                dashboard.update(obs, info_env, reward=reward)

        # ── Pace to real-time ───────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        remaining = frame_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    if use_actions:
        replay_score = end_score if end_score is not None else info_env.get("task_score")
    else:
        replay_score = None
    recorded = f"{final_score:.3f}" if final_score is not None else "n/a"
    if replay_score is not None:
        print(f"  score   : replayed {replay_score:.3f}  (recorded {recorded})\n")
    else:
        print(f"  score   : recorded {recorded}\n")

    if dashboard:
        dashboard.close()
    env.close()
    return replay_score


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded demonstrations from a LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Path to a LeRobot v2.1 dataset directory (e.g. demos/<task>/M01)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="episode_index to replay, as listed in meta/episodes.jsonl "
             "(default: replay all episodes)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0, use 0.5 for slow-mo, 0 for unpaced)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show the real-time obs dashboard alongside the viewer",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run without the MuJoCo viewer (headless)",
    )

    args = parser.parse_args()

    dataset_root = args.path

    # Verify it's a LeRobot dataset
    info_path = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"Not a LeRobot dataset: {dataset_root} (missing meta/info.json)")
        sys.exit(1)

    info, episodes = _read_dataset_meta(dataset_root)

    if args.episode is not None:
        selected = [ep for ep in episodes if ep["episode_index"] == args.episode]
        if not selected:
            available = sorted(ep["episode_index"] for ep in episodes)
            print(f"episode_index {args.episode} not found in {dataset_root}; "
                  f"available: {available}")
            sys.exit(1)
    else:
        selected = sorted(episodes, key=lambda ep: ep["episode_index"])

    print(f"[replay] {len(selected)} episode(s) from {dataset_root}\n")

    for ep_meta in selected:
        replay_one(dataset_root, info, ep_meta, speed=args.speed,
                   use_dashboard=args.dashboard, render=not args.no_render)


if __name__ == "__main__":
    main()
