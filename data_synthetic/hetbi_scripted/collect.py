"""Scripted data collection for the heterogeneous-bimanual (het-bi) tasks.

This is the supported scripted path for het-bi tasks.  Each task has a
dedicated policy in ``data_synthetic/hetbi_scripted/`` (``TASK_POLICY_MAP``);
an episode ends when the task terminates, the horizon is reached or the policy
reports ``done``, and only successful episodes are saved, in LeRobot v2.1
format under ``<out-dir>/<task-id>/M<machine-id>/`` (see
``data_synthetic/collection.py``).  Requires
``python scripts/setup_machine.py`` to have been run once.

Usage::

    # Single task, 10 successful episodes, headless
    python -m data_synthetic.hetbi_scripted.collect \\
        --task ib-p1-plate-shelf-hetbi-v1 --n-episodes 10

    # All het-bi tasks, 5 episodes each, in the viewer
    python -m data_synthetic.hetbi_scripted.collect --all --n-episodes 5 --render

    # Custom output root + raw .npy actions
    python -m data_synthetic.hetbi_scripted.collect \\
        --task ib-p1-cup-inversion-v1 --n-episodes 20 --out-dir demos_hetbi --save-npy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_synthetic.collection import CollectionResult, add_collection_args, build_policy, collect
from data_synthetic.hetbi_scripted import TASK_POLICY_MAP

ALL_HETBI_TASKS = list(TASK_POLICY_MAP)


def collect_hetbi(
    task_id: str,
    n_episodes: int = 10,
    out_dir: Optional[str] = None,
    render: bool = False,
    use_dashboard: bool = True,
    max_steps: Optional[int] = None,
    save_lerobot: bool = True,
    save_npy: bool = False,
    seed: Optional[int] = None,
    obs_mode: str = "combo",
) -> CollectionResult:
    """Collect *n_episodes* successful scripted episodes of *task_id*.

    *max_steps* overrides the YAML horizon; see :func:`data_synthetic.collection.collect`.
    """
    if task_id not in TASK_POLICY_MAP:
        raise ValueError(f"Unknown het-bi task {task_id!r}. Available: {ALL_HETBI_TASKS}")
    policy_cls = TASK_POLICY_MAP[task_id]
    return collect(
        task_id=task_id,
        policy_factory=lambda env, obs, rng: build_policy(policy_cls, env, rng),
        n_episodes=n_episodes,
        out_dir=out_dir,
        render=render,
        use_dashboard=use_dashboard,
        obs_mode=obs_mode,
        max_steps=max_steps,
        seed=seed,
        save_lerobot=save_lerobot,
        save_npy=save_npy,
        stop_when_policy_done=True,
        log_prefix="hetbi",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect scripted het-bi demonstrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", metavar="TASK_ID", choices=ALL_HETBI_TASKS,
                       help=f"Task to collect: {', '.join(ALL_HETBI_TASKS)}")
    group.add_argument("--all", action="store_true", help="Collect every het-bi task in turn.")
    add_collection_args(parser, n_episodes_default=10, lerobot_toggle=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    results = {}
    for task_id in (ALL_HETBI_TASKS if args.all else [args.task]):
        results[task_id] = collect_hetbi(
            task_id,
            n_episodes=args.n_episodes,
            out_dir=args.out_dir,
            render=args.render,
            use_dashboard=not args.no_dashboard,
            max_steps=args.max_steps,
            save_lerobot=not args.no_lerobot,
            save_npy=args.save_npy,
            seed=args.seed,
            obs_mode=args.obs_mode,
        )
        if results[task_id].interrupted:
            break

    if len(results) > 1:
        print("\nSUMMARY")
        for task_id, res in results.items():
            print(f"  {task_id:40s}  saved {res.n_saved}/{res.n_requested} "
                  f"in {res.attempts} attempts")
    sys.exit(max(res.exit_code for res in results.values()))


if __name__ == "__main__":
    main()
