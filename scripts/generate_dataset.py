#!/usr/bin/env python3
"""Generate LeRobot datasets by replaying planned action trajectories through IMBench tasks.

Each ``.npy`` file in ``--traj-dir`` holds one episode as a ``(T, action_dim)``
array of cartesian-delta actions (e.g. from the bimanual planning pipeline).
Every trajectory is stepped through the task from ``reset(seed=--seed)`` and
the observations, rewards and task scores are recorded in the same layout
``collect_demos.py`` produces: a LeRobot v2.1 dataset in
``<out-dir>/M<machine_id>/`` with a ``meta/episode_NNNNNN_metadata.json`` per
episode carrying a global run_id, the seed and ``warmup_steps: 0``, so the
output feeds ``merge_annotated_runs.py`` and ``replay_episode.py`` directly.
Works for single-arm and bimanual tasks.

Usage::

    python scripts/generate_dataset.py \\
        --task ib-p1-plate-shelf-hetbi-v1 \\
        --traj-dir /path/to/trajectories/

    python scripts/generate_dataset.py \\
        --task ib-p1-bowl-table-v1 \\
        --traj-dir /path/to/bowl_trajs/ \\
        --out-dir datasets/ib-p1-bowl-table-v1 --seed 3
"""

import argparse
import glob
import os
import sys

import numpy as np

import imbench as ib
from imbench.tasks.base_task import load_task_config
from imbench.utils.dashboard_reporter import report_episode
from imbench.utils.lerobot_writer import DatasetLockedError, DatasetSchemaError
from imbench.utils.recording import (
    deferred_stop_signals,
    make_writer,
    require_machine_id,
    require_run_counter,
    write_episode_metadata,
)
from imbench.utils.robot_state import extract_robot_state
from imbench.utils.run_id import next_run_id
from imbench.utils.task_names import task_display_name

DEVICE_NAME = "trajectory"


class TrajectoryError(ValueError):
    """A trajectory file that cannot be replayed."""


def load_trajectories(traj_files, action_dim):
    """Load every trajectory before anything is written; empty ones are skipped.

    Returns ``[(path, actions)]``.  Raises :class:`TrajectoryError` for a file that is not
    a finite ``(T, action_dim)`` array.
    """
    trajectories = []
    for traj_path in traj_files:
        actions = np.load(traj_path)
        if actions.ndim != 2 or actions.shape[1] != action_dim:
            raise TrajectoryError(
                f"{traj_path}: expected shape (T, {action_dim}), got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise TrajectoryError(f"{traj_path}: contains non-finite actions")
        if len(actions) == 0:
            print(f"  skipping empty trajectory {traj_path}")
            continue
        trajectories.append((traj_path, actions))
    return trajectories


def generate_episodes(task_id, traj_files, out_root, machine_id, obs_mode="combo",
                      render=False, seed=0):
    """Replay action sequences through a task and save them as a LeRobot dataset.

    All trajectories are validated (see :func:`load_trajectories`) before the
    dataset is created.  Returns ``(scores, successes)``, one entry per saved
    episode.
    """
    out_dir = os.path.join(out_root, f"M{machine_id}")
    task_cfg = load_task_config(task_id)
    dt = 1.0 / task_cfg.get("control_freq", 20)

    env = ib.make(
        task_id,
        privileged_obs=True,
        obs_mode=obs_mode,
        render_mode="human" if render else None,
        config_overrides={"max_episode_steps": 999_999},
    )
    writer = None
    scores, successes = [], []
    try:
        action_dim = env.action_space.shape[0]
        trajectories = load_trajectories(traj_files, action_dim)
        print(f"Generating {len(trajectories)} episodes for {task_id} -> {out_dir}/")
        if not trajectories:
            return scores, successes
        writer = make_writer(out_dir, task_id, task_cfg)
        save_extras = env.save_extras

        for traj_path, actions in trajectories:
            obs, info = env.reset(seed=seed)
            writer.begin_episode()
            n_steps = 0
            for t, act in enumerate(actions):
                act = np.asarray(act, dtype=np.float32)
                obs_snapshot = {k: v.copy() for k, v in obs.items()}
                jpos, epose = extract_robot_state(env)
                obs_snapshot["joint_positions"] = jpos
                obs_snapshot["ee_poses"] = epose

                obs, reward, terminated, truncated, info = env.step(act)
                # Record what ran: keyboard-typing replaces arm 0 with its scripted arm.
                act = np.asarray(info.get("applied_action", act), dtype=np.float32)
                if render:
                    env.render()

                writer.add_step(
                    timestamp=t * dt,
                    obs=env.filter_obs_for_saving(obs_snapshot),
                    action=act if save_extras["action"] else None,
                    reward=reward,
                    task_score=info["task_score"],
                    save_reward=save_extras["reward"],
                    save_task_score=save_extras["task_score"],
                )
                n_steps = t + 1
                if terminated or truncated:
                    break

            score = info["task_score"]
            success = bool(info["is_success"])
            ep_idx = writer.episode_index
            with deferred_stop_signals():
                run_id = next_run_id(machine_id)
                writer.end_episode(metadata={
                    "final_score": score,
                    "teleop_device": DEVICE_NAME,
                    "run_id": run_id,
                    "seed": seed,
                    "warmup_steps": 0,
                    "traj_file": os.path.basename(traj_path),
                })
                write_episode_metadata(
                    out_dir, ep_idx,
                    task_id=task_id,
                    task_cfg=task_cfg,
                    display_name=task_display_name(task_id),
                    run_id=run_id,
                    n_steps=n_steps,
                    action_dim=action_dim,
                    final_score=score,
                    success=success,
                    seed=seed,
                    warmup_steps=0,
                    teleop_device=DEVICE_NAME,
                )
            report_episode(
                task_name=task_display_name(task_id),
                outcome="success" if success else "failure",
                score=score,
                episode_id=run_id,
                machine_id=machine_id,
            )
            scores.append(score)
            successes.append(success)
            print(f"  ep {ep_idx:04d}  run_id={run_id}  ({n_steps} steps, score={score:.3f})")
    finally:
        if writer is not None:
            writer.abort_episode()
            writer.finalize()
        env.close()
    return scores, successes


def main():
    parser = argparse.ArgumentParser(
        description="Generate LeRobot datasets from planned action trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", required=True, help="IMBench task ID")
    parser.add_argument("--traj-dir", required=True,
                        help="Directory containing .npy trajectory files")
    parser.add_argument("--out-dir", default=None,
                        help="Output root; episodes go to <out-dir>/M<machine_id>/ "
                             "(default: datasets/<task>/)")
    parser.add_argument("--obs-mode", default="combo",
                        choices=["combo", "proprio", "rgb"])
    parser.add_argument("--seed", type=int, default=0,
                        help="Reset seed of the scene the trajectories were planned in (default: 0)")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Maximum number of trajectories to convert")
    args = parser.parse_args()

    machine_id = require_machine_id("generate")
    require_run_counter("generate")

    out_root = args.out_dir or os.path.join("datasets", args.task)

    traj_files = sorted(glob.glob(os.path.join(args.traj_dir, "*.npy")))
    if not traj_files:
        print(f"No .npy files found in {args.traj_dir}")
        sys.exit(1)

    if args.max_episodes:
        traj_files = traj_files[: args.max_episodes]

    try:
        scores, successes = generate_episodes(
            args.task, traj_files, out_root, machine_id,
            obs_mode=args.obs_mode, render=args.render, seed=args.seed,
        )
    except (DatasetLockedError, DatasetSchemaError, TrajectoryError) as exc:
        sys.exit(f"Error: {exc}")
    if not scores:
        return

    print(f"\nDone. Mean score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"Success rate: {np.mean(successes):.1%}")


if __name__ == "__main__":
    main()
