#!/usr/bin/env python3
"""
Pipeline verification script.

Runs 3 episodes of every single-arm scripted policy in POLICY_REGISTRY, each
for the task's own max_episode_steps unless --max-steps is given, and checks:
  1. Policy executes without exception.
  2. Randomization: object/bob XYZ position differs across episodes.
  3. Camera frames: non-black, finite, correct shape.
  4. Success: info["is_success"] becomes true in at least one episode.
  5. LeRobot data format: Parquet files written with correct columns,
     MP4 files non-empty, metadata JSON valid.

Episode i resets the task with seed 37*i + 5 and, as the collectors do, seeds
the policy's rng with the same value, so runs are reproducible.  Exits 1 if
any task fails a check.

Usage (headless, 3 eps per task):
    python scripts/tests/verify_pipelines.py

Use --tasks to target specific tasks:
    python scripts/tests/verify_pipelines.py \
        --tasks ib-p2-pendulum-grasp-v1 ib-p1-edge-slide-v1 ib-p6-slip-recovery-v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

import imbench as ib
from imbench.tasks.base_task import load_task_config
from imbench.utils.robot_state import extract_robot_state

from data_synthetic.collection import build_policy

# ── Task → Policy map (mirrors singlearm_collect_demos._POLICY_REGISTRY) ──────
from data_synthetic.single_arm_pipeline.bounce_intercept_policy  import BounceInterceptPolicy
from data_synthetic.single_arm_pipeline.cube_toss_policy         import CubeTossPolicy
from data_synthetic.single_arm_pipeline.cup_extract_policy       import CupExtractPolicy
from data_synthetic.single_arm_pipeline.domino_select_policy     import DominoSelectPolicy
from data_synthetic.single_arm_pipeline.domino_single_policy     import DominoPolicy
from data_synthetic.single_arm_pipeline.edge_slide_policy        import EdgeSlidePushPolicy
from data_synthetic.single_arm_pipeline.galileo_ramp_drop_policy import GalileoRampDropPolicy
from data_synthetic.single_arm_pipeline.gap_funnel_policy        import GapFunnelPolicy
from data_synthetic.single_arm_pipeline.indirect_push_policy     import IndirectPushPolicy
from data_synthetic.single_arm_pipeline.linear_intercept_policy  import LinearInterceptPolicy
from data_synthetic.single_arm_pipeline.mass_sort_policy         import MassSortPolicy
from data_synthetic.single_arm_pipeline.mirror_pick_place_policy import MirrorPickPlacePolicy
from data_synthetic.single_arm_pipeline.occluder_push_policy     import OccluderPushPolicy
from data_synthetic.single_arm_pipeline.packing_policy          import PackingPolicy
from data_synthetic.single_arm_pipeline.peg_insert_policy        import PegInsertPolicy
from data_synthetic.single_arm_pipeline.pendulum_grasp_policy    import PendulumGraspPolicy
from data_synthetic.single_arm_pipeline.pick_place_ycb_policy    import PickPlaceYCBPolicy
from data_synthetic.single_arm_pipeline.ramp_sort_policy         import RampSortPolicy
from data_synthetic.single_arm_pipeline.seesaw_balance_policy    import SeesawBalancePolicy
from data_synthetic.single_arm_pipeline.shape_stack_policy       import ShapeStackPolicy
from data_synthetic.single_arm_pipeline.sheltered_pick_policy    import ShelteredGraspPolicy
from data_synthetic.single_arm_pipeline.slide_catch_policy       import SlideCatchPolicy
from data_synthetic.single_arm_pipeline.slip_recovery_policy     import SlipRecoveryPolicy
from data_synthetic.single_arm_pipeline.stack_collapse_recovery_policy import StackCollapseRecoveryPolicy
from data_synthetic.single_arm_pipeline.tool_retrieve_policy     import ToolRetrievePolicy

POLICY_REGISTRY: dict[str, type] = {
    "ib-p3-domino-select-v1":              DominoSelectPolicy,
    "ib-p3-domino-single-v1":              DominoPolicy,
    "ib-p1-edge-slide-v1":                 EdgeSlidePushPolicy,
    "ib-p2-cube-toss-v1":                  CubeTossPolicy,
    "ib-p2-galileo-ramp-drop-v1":          GalileoRampDropPolicy,
    "ib-p2-linear-intercept-v1":           LinearInterceptPolicy,
    "ib-p2-bounce-intercept-v1":           BounceInterceptPolicy,
    "ib-m-mirror-pick-place-v1":          MirrorPickPlacePolicy,
    "ib-p5-occluder-push-v1":              OccluderPushPolicy,
    "ib-p7-packing-v1":                    PackingPolicy,
    "ib-p2-pendulum-grasp-v1":             PendulumGraspPolicy,
    "ib-p2-slide-catch-v1":                SlideCatchPolicy,
    "ib-p6-pick-place-ycb-v1":         PickPlaceYCBPolicy,
    "ib-p6-stack-collapse-recovery-v1":StackCollapseRecoveryPolicy,
    "ib-p3-gap-funnel-v1":             GapFunnelPolicy,
    "ib-p7-shape-stack-v1":            ShapeStackPolicy,
    "ib-p4-cup-extract-v1":                CupExtractPolicy,
    "ib-p3-seesaw-balance-v1":             SeesawBalancePolicy,
    "ib-p4-tool-retrieve-v1":              ToolRetrievePolicy,
    "ib-p5-mass-sort-v1":                  MassSortPolicy,
    "ib-p1-recover-peg-insert-v1":         PegInsertPolicy,
    "ib-p1-sheltered-grasp-v1":            ShelteredGraspPolicy,
    "ib-p6-slip-recovery-v1":              SlipRecoveryPolicy,
    "ib-p3-indirect-push-v1":          IndirectPushPolicy,
    "ib-p2-ramp-sort-v1":              RampSortPolicy,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

PASS  = "\033[92m PASS\033[0m"
FAIL  = "\033[91m FAIL\033[0m"
WARN  = "\033[93m WARN\033[0m"
SKIP  = "\033[94m SKIP\033[0m"


def _policy_act_needs_env(policy) -> bool:
    import inspect
    return "env" in inspect.signature(policy.act).parameters


def _obj_pos_from_obs(obs: dict) -> Optional[np.ndarray]:
    """Return first available object position from obs dict."""
    for key in ("object_pos", "object0_pos"):
        if key in obs:
            return np.asarray(obs[key], dtype=float)
    # Fallback: find any *_pos key that isn't robot
    for key, val in obs.items():
        if key.endswith("_pos") and not key.startswith("robot"):
            try:
                arr = np.asarray(val, dtype=float).flatten()
                if arr.shape == (3,):
                    return arr
            except Exception:
                pass
    return None


def _check_camera_frames(obs: dict) -> list[str]:
    """Return list of issues found in camera frames."""
    issues = []
    cam_keys = [k for k in obs if "image" in k or "rgb" in k.lower() or "cam" in k.lower()]
    if not cam_keys:
        issues.append("no camera keys found in obs")
        return issues
    for k in cam_keys:
        frame = np.asarray(obs[k])
        if not np.isfinite(frame).all():
            issues.append(f"{k}: non-finite pixels")
        if frame.mean() < 1.0:
            issues.append(f"{k}: mean pixel {frame.mean():.1f} — near-black frame")
        if frame.max() < 5:
            issues.append(f"{k}: max pixel {frame.max()} — completely black")
        if frame.ndim != 3 or frame.shape[2] != 3:
            issues.append(f"{k}: unexpected shape {frame.shape}")
    return issues


def _check_lerobot_output(out_dir: str) -> list[str]:
    """Verify LeRobot v2.1 output: Parquet files + metadata JSON."""
    issues = []
    out = Path(out_dir)

    # Find episode parquet files
    parquet_files = sorted(out.rglob("*.parquet"))
    if not parquet_files:
        issues.append("no .parquet files written")
    else:
        try:
            import pyarrow.parquet as pq
            for pf in parquet_files[:3]:  # spot-check first 3
                tbl = pq.read_table(pf)
                if tbl.num_rows == 0:
                    issues.append(f"{pf.name}: 0 rows")
                # Actions are stored as action.0 ... action.N (flattened)
                has_ts = "timestamp" in tbl.schema.names
                has_action = any(c.startswith("action") for c in tbl.schema.names)
                if not has_ts:
                    issues.append(f"{pf.name}: missing 'timestamp' column")
                if not has_action:
                    issues.append(f"{pf.name}: missing 'action.*' columns")
        except ImportError:
            issues.append("pyarrow not installed — skipping parquet check")
        except Exception as e:
            issues.append(f"parquet read error: {e}")

    # Check metadata JSON files — LeRobot v2.1 stores them as meta/info.json
    meta_jsons = sorted(out.rglob("info.json")) or sorted(out.rglob("*metadata*.json"))
    if not meta_jsons:
        issues.append("no meta/info.json files found")
    else:
        for mj in meta_jsons[:2]:
            try:
                with open(mj) as f:
                    meta = json.load(f)
                # LeRobot v2.1 info.json fields
                required = {"codebase_version", "total_episodes", "total_frames"}
                missing = required - set(meta.keys())
                if missing:
                    issues.append(f"{mj.name}: missing keys {missing}")
            except Exception as e:
                issues.append(f"{mj.name}: JSON parse error: {e}")

    # Check video files (MP4)
    mp4_files = sorted(out.rglob("*.mp4"))
    for mf in mp4_files[:3]:
        size = mf.stat().st_size
        if size < 500:
            issues.append(f"{mf.name}: suspiciously small ({size} bytes)")

    return issues


# ── Per-task runner ────────────────────────────────────────────────────────────

def run_task_verification(
    task_id: str,
    PolicyClass: type,
    n_episodes: int = 3,
    max_steps: Optional[int] = None,
    out_dir: Optional[str] = None,
    save_lerobot: bool = True,
) -> dict:
    """Run n_episodes of task_id with PolicyClass and return a result dict.

    *max_steps* overrides the task's max_episode_steps.  The result fails
    (``ok`` False) when no episode reaches ``info["is_success"]``.
    """
    result = {
        "task_id": task_id,
        "policy": PolicyClass.__name__,
        "episodes": [],
        "issues": [],
        "ok": True,
    }

    env = None
    try:
        task_cfg = load_task_config(task_id)
        control_freq = task_cfg.get("control_freq", 100)
        dt = 1.0 / control_freq

        env = ib.make(
            task_id,
            privileged_obs=True,
            obs_mode="combo",
            render_mode=None,
            config_overrides={"max_episode_steps": max_steps} if max_steps is not None else None,
        )

        # ── LeRobot writer setup ───────────────────────────────────────────
        writer = None
        if save_lerobot and out_dir:
            from imbench.utils.lerobot_writer import LeRobotWriter
            rec_cfg   = task_cfg.get("recording", {})
            save_freq = rec_cfg.get("save_freq", {})
            for cat in ("proprio", "force_torque", "object_state", "cameras", "tactile"):
                save_freq.setdefault(cat, control_freq)
            ctrl_cfg = task_cfg.get("controller", {})
            if isinstance(ctrl_cfg, list):
                ctrl_cfg = ctrl_cfg[0] if ctrl_cfg else {}
            robot_type = ctrl_cfg.get("robot", "Panda") if isinstance(ctrl_cfg, dict) else "Panda"
            writer = LeRobotWriter(
                root=out_dir,
                task_id=task_id,
                control_freq=control_freq,
                save_freq=save_freq,
                features={},
                robot_type=robot_type,
                task_config=task_cfg,
            )

        obj_positions_per_ep: list[Optional[np.ndarray]] = []
        scores: list[float] = []
        successes: list[bool] = []

        for ep in range(n_episodes):
            seed = ep * 37 + 5
            obs, info = env.reset(seed=seed)

            policy = build_policy(PolicyClass, env, np.random.default_rng(seed))
            act_needs_env = _policy_act_needs_env(policy)

            # Record initial object position for randomization check
            obj_positions_per_ep.append(_obj_pos_from_obs(obs))

            # Camera check on first step of first episode
            if ep == 0:
                cam_issues = _check_camera_frames(obs)
                if cam_issues:
                    result["issues"].extend([f"[camera] {i}" for i in cam_issues])

            if writer:
                writer.begin_episode()

            save_extras = env.save_extras

            t = 0
            success = False
            done = False
            while not done:
                try:
                    action = (
                        policy.act(obs, env=env) if act_needs_env
                        else policy.act(obs)
                    )
                except Exception as e:
                    result["issues"].append(
                        f"ep{ep} step{t} policy.act() raised: {type(e).__name__}: {e}"
                    )
                    result["ok"] = False
                    break

                obs_snap = {k: v.copy() for k, v in obs.items()}
                try:
                    jpos, epose = extract_robot_state(env)
                    obs_snap["joint_positions"] = jpos
                    obs_snap["ee_poses"]        = epose
                except Exception:
                    pass

                obs, reward, terminated, truncated, info = env.step(action)
                success = success or bool(info["is_success"])

                if writer:
                    save_obs = env.filter_obs_for_saving(obs_snap)
                    writer.add_step(
                        timestamp=t * dt,
                        obs=save_obs,
                        action=action if save_extras.get("action") else None,
                        reward=reward,
                        task_score=info.get("task_score", 0.0),
                        save_reward=save_extras.get("reward", False),
                        save_task_score=save_extras.get("task_score", False),
                    )

                t += 1
                done = terminated or truncated or getattr(policy, "done", False)

            score = float(info.get("task_score", 0.0))
            scores.append(score)
            successes.append(success)
            phase = getattr(policy, "phase", "?")

            if writer:
                writer.end_episode()

            result["episodes"].append({
                "ep": ep, "seed": seed, "steps": t,
                "score": score, "success": success, "phase": str(phase),
            })

        # ── Randomization check ────────────────────────────────────────────
        valid_pos = [p for p in obj_positions_per_ep if p is not None]
        if len(valid_pos) >= 2:
            dists = [np.linalg.norm(valid_pos[i] - valid_pos[0])
                     for i in range(1, len(valid_pos))]
            max_dist = max(dists)
            if max_dist < 1e-4:
                result["issues"].append(
                    f"randomization: all {len(valid_pos)} initial object "
                    f"positions identical (max_dist={max_dist:.6f}) — "
                    f"seeding may be broken"
                )
            result["max_pos_spread"] = round(float(max_dist), 4)
        else:
            result["max_pos_spread"] = None

        # ── Score summary ──────────────────────────────────────────────────
        result["mean_score"]    = round(float(np.mean(scores)), 3)
        result["max_score"]     = round(float(np.max(scores)), 3)
        result["success_rate"]  = round(float(np.mean(successes)), 3)
        if not any(successes):
            result["issues"].append(
                f"[success] info['is_success'] never became true in {n_episodes} episode(s)"
            )

        # ── LeRobot format check ───────────────────────────────────────────
        if writer:
            writer.finalize()
            lero_issues = _check_lerobot_output(out_dir)
            if lero_issues:
                result["issues"].extend([f"[lerobot] {i}" for i in lero_issues])

    except Exception as e:
        result["issues"].append(f"FATAL: {type(e).__name__}: {e}")
        result["ok"] = False
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()

    if result["issues"]:
        result["ok"] = False

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Verify scripted motion pipelines")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Task IDs to test (default: every task in POLICY_REGISTRY).")
    ap.add_argument("--n-episodes", type=int, default=3,
                    help="Episodes per task (default 3).")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Per-episode step limit (default: each task's max_episode_steps).")
    ap.add_argument("--no-lerobot", action="store_true",
                    help="Skip LeRobot data writing.")
    ap.add_argument("--out-dir", default=None,
                    help="Root output directory for LeRobot data.")
    args = ap.parse_args()

    tasks = args.tasks or list(POLICY_REGISTRY.keys())

    # Resolve output directory
    use_tmp = args.out_dir is None and not args.no_lerobot
    tmp_dir = None
    if use_tmp:
        tmp_dir = tempfile.mkdtemp(prefix="ib_verify_")
        print(f"LeRobot output → {tmp_dir}")

    results: list[dict] = []
    total = len(tasks)

    print(f"\n{'═'*70}")
    print(f"  IMBench Pipeline Verification  —  {total} task(s)  ×  {args.n_episodes} eps each")
    print(f"{'═'*70}\n")

    for idx, task_id in enumerate(tasks):
        if task_id not in POLICY_REGISTRY:
            print(f"[{idx+1:02d}/{total}] {task_id}  {SKIP}  (not in registry)")
            continue

        PolicyClass = POLICY_REGISTRY[task_id]
        out = (
            None if args.no_lerobot
            else os.path.join(args.out_dir or tmp_dir, task_id)
        )

        print(f"[{idx+1:02d}/{total}] {task_id}  ({PolicyClass.__name__})")
        sys.stdout.flush()

        r = run_task_verification(
            task_id, PolicyClass,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            out_dir=out,
            save_lerobot=not args.no_lerobot,
        )
        results.append(r)

        # Per-episode summary
        for ep in r["episodes"]:
            print(f"    ep{ep['ep']:02d}  seed={ep['seed']}  steps={ep['steps']:5d}"
                  f"  score={ep['score']:.3f}  success={ep['success']}  phase={ep['phase']}")

        spread = r.get("max_pos_spread")
        spread_str = f"{spread:.4f}m" if spread is not None else "N/A"
        status = PASS if r["ok"] else FAIL
        print(f"  → mean_score={r.get('mean_score','?')}  "
              f"success_rate={r.get('success_rate','?')}  "
              f"pos_spread={spread_str}  {status}")
        if r["issues"]:
            for iss in r["issues"]:
                print(f"      ⚠  {iss}")
        print()

    # ── Final summary ──────────────────────────────────────────────────────────
    n_ok   = sum(1 for r in results if r["ok"])
    n_fail = len(results) - n_ok
    print(f"{'═'*70}")
    print(f"  RESULTS:  {n_ok}/{len(results)} PASSED   {n_fail} FAILED")
    print(f"{'═'*70}")

    if n_fail:
        print("\nFailed tasks:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['task_id']}")
                for iss in r["issues"]:
                    print(f"    • {iss}")

    # Cleanup
    if use_tmp and tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
