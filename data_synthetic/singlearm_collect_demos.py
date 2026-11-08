"""
singlearm_collect_demos.py
==========================
Scripted-policy data collection for the single-arm tasks listed in
``_POLICY_REGISTRY``.  Saves successful episodes in LeRobot v2.1 format
(Parquet + MP4) under ``<out-dir>/<task-id>/M<machine-id>/``; see
``data_synthetic/collection.py`` for the dataset layout, seeding and resume
rules.  Requires ``python scripts/setup_machine.py`` to have been run once.

Usage examples::

    # 40 successful domino-select episodes, headless
    python data_synthetic/singlearm_collect_demos.py --task-id ib-p3-domino-select-v1 --n-episodes 40

    # Watch the policy in the MuJoCo viewer + dashboard
    python data_synthetic/singlearm_collect_demos.py --task-id ib-p2-cube-toss-v1 --n-episodes 5 --render

    # Custom output root, also save raw .npy action trajectories
    python data_synthetic/singlearm_collect_demos.py --task-id ib-p1-edge-slide-v1 \\
        --n-episodes 10 --out-dir my_demos --save-npy

Per-stream save frequencies come from the task YAML (``recording.save_freq``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_synthetic.collection import add_collection_args, build_policy, collect, import_policy

_DEFAULT_TASK_ID = "ib-p3-domino-select-v1"

# Task id -> (module in data_synthetic.single_arm_pipeline, policy class).
_POLICY_REGISTRY: dict[str, tuple[str, str]] = {
    "ib-m-mirror-pick-place-v1":        ("mirror_pick_place_policy", "MirrorPickPlacePolicy"),
    "ib-p1-edge-slide-v1":              ("edge_slide_policy", "EdgeSlidePushPolicy"),
    "ib-p1-recover-peg-insert-v1":      ("peg_insert_policy", "PegInsertPolicy"),
    "ib-p1-sheltered-grasp-v1":         ("sheltered_pick_policy", "ShelteredGraspPolicy"),
    "ib-p2-bounce-intercept-v1":        ("bounce_intercept_policy", "BounceInterceptPolicy"),
    "ib-p2-cube-toss-v1":               ("cube_toss_policy", "CubeTossPolicy"),
    "ib-p2-galileo-ramp-drop-v1":       ("galileo_ramp_drop_policy", "GalileoRampDropPolicy"),
    "ib-p2-linear-intercept-v1":        ("linear_intercept_policy", "LinearInterceptPolicy"),
    "ib-p2-pendulum-grasp-v1":          ("pendulum_grasp_policy", "PendulumGraspPolicy"),
    "ib-p2-ramp-sort-v1":               ("ramp_sort_policy", "RampSortPolicy"),
    "ib-p2-slide-catch-v1":             ("slide_catch_policy", "SlideCatchPolicy"),
    "ib-p3-domino-select-v1":           ("domino_select_policy", "DominoSelectPolicy"),
    "ib-p3-domino-single-v1":           ("domino_single_policy", "DominoPolicy"),
    "ib-p3-gap-funnel-v1":              ("gap_funnel_policy", "GapFunnelPolicy"),
    "ib-p3-indirect-push-v1":           ("indirect_push_policy", "IndirectPushPolicy"),
    "ib-p3-seesaw-balance-v1":          ("seesaw_balance_policy", "SeesawBalancePolicy"),
    "ib-p4-cup-extract-v1":             ("cup_extract_policy", "CupExtractPolicy"),
    "ib-p4-tool-retrieve-v1":           ("tool_retrieve_policy", "ToolRetrievePolicy"),
    "ib-p5-mass-sort-v1":               ("mass_sort_policy", "MassSortPolicy"),
    "ib-p5-occluder-push-v1":           ("occluder_push_policy", "OccluderPushPolicy"),
    "ib-p6-pick-place-ycb-v1":          ("pick_place_ycb_policy", "PickPlaceYCBPolicy"),
    "ib-p6-slip-recovery-v1":           ("slip_recovery_policy", "SlipRecoveryPolicy"),
    "ib-p6-stack-collapse-recovery-v1": ("stack_collapse_recovery_policy", "StackCollapseRecoveryPolicy"),
    "ib-p7-packing-v1":                 ("packing_policy", "PackingPolicy"),
    "ib-p7-shape-stack-v1":             ("shape_stack_policy", "ShapeStackPolicy"),
}

# Policies that set ``failed`` once the episode can no longer succeed.
_SELF_TERMINATING = {"ib-p2-bounce-intercept-v1"}

_WARMUP_STEPS = 1


def resolve_policy(task_id: str) -> type:
    """Import and return the policy class registered for exactly *task_id*."""
    return import_policy(task_id, _POLICY_REGISTRY, "data_synthetic.single_arm_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted-policy data collection for single-arm tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task-id", type=str, default=_DEFAULT_TASK_ID,
        help=f"Task to collect (default: {_DEFAULT_TASK_ID}). "
             f"Registered: {', '.join(sorted(_POLICY_REGISTRY))}",
    )
    parser.add_argument(
        "--task-display-name", type=str, default=None,
        help="Human-readable task name for metadata (default: derived from --task-id)",
    )
    add_collection_args(parser, n_episodes_default=40)
    args = parser.parse_args()

    try:
        policy_cls = resolve_policy(args.task_id)
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")
    result = collect(
        task_id=args.task_id,
        policy_factory=lambda env, obs, rng: build_policy(policy_cls, env, rng),
        n_episodes=args.n_episodes,
        out_dir=args.out_dir,
        render=args.render,
        use_dashboard=not args.no_dashboard,
        obs_mode=args.obs_mode,
        max_steps=args.max_steps,
        seed=args.seed,
        save_npy=args.save_npy,
        warmup_steps=_WARMUP_STEPS,
        stop_when_policy_failed=args.task_id in _SELF_TERMINATING,
        display_name=args.task_display_name,
        log_prefix="singlearm",
    )
    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
