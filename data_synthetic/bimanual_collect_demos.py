"""
bimanual_collect_demos.py
=========================
Scripted-policy data collection for the homogeneous-bimanual (HoBi) tasks
listed in ``_POLICY_REGISTRY``.  Saves successful episodes in LeRobot v2.1
format (Parquet + MP4) under ``<out-dir>/<task-id>/M<machine-id>/``; see
``data_synthetic/collection.py`` for the dataset layout, seeding and resume
rules.  Requires ``python scripts/setup_machine.py`` to have been run once.

Usage examples::

    # 10 keyboard-typing episodes in the viewer
    python data_synthetic/bimanual_collect_demos.py \\
        --task-id ib-m-keyboard-typing-v1 --n-episodes 10 --render

    # Headless with a custom output root and raw .npy actions
    python data_synthetic/bimanual_collect_demos.py \\
        --task-id ib-m-keyboard-typing-v1 --n-episodes 50 --out-dir my_demos --save-npy

``ib-p1-plate-shelf-hobi-v1`` has its own collector,
``python -m data_synthetic.mp_hobi_plate_shelf``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_synthetic.collection import add_collection_args, build_policy, collect, import_policy

_DEFAULT_TASK_ID = "ib-m-keyboard-typing-v1"

# Task id -> (module in data_synthetic.bimanual_pipeline, policy class).
_POLICY_REGISTRY: dict[str, tuple[str, str]] = {
    "ib-m-keyboard-typing-v1": ("keyboard_typing_policy", "KeyboardTypingPolicy"),
}

_WARMUP_STEPS = 1
DEFAULT_WORDS_FILE = Path(__file__).with_name("keyboard_words.txt")


def load_words(path: str | Path = DEFAULT_WORDS_FILE) -> list[str]:
    """Read a vocabulary with one A-Z word per line (blank lines are ignored)."""
    words = [line.strip().upper() for line in Path(path).read_text().splitlines() if line.strip()]
    if not words or any(len(word) < 2 or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for ch in word)
                        for word in words):
        raise ValueError("word list must contain words of at least two A-Z letters, one per line")
    if len(set(words)) != len(words):
        raise ValueError("word list must not contain duplicate words")
    return words


def resolve_policy(task_id: str) -> type:
    """Import and return the policy class registered for exactly *task_id*."""
    return import_policy(task_id, _POLICY_REGISTRY, "data_synthetic.bimanual_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scripted-policy data collection for bimanual tasks.",
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
    parser.add_argument(
        "--words-file", type=Path, default=DEFAULT_WORDS_FILE,
        help="One word per line; a word is sampled randomly for each attempt using its seed. "
             "Defaults to the bundled 25-word keyboard_words.txt.",
    )
    add_collection_args(parser, n_episodes_default=10)
    args = parser.parse_args()

    try:
        policy_cls = resolve_policy(args.task_id)
        words = load_words(args.words_file)
    except (OSError, ValueError) as exc:
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
        config_overrides={"word": None, "words": words},
        seed=args.seed,
        save_npy=args.save_npy,
        warmup_steps=_WARMUP_STEPS,
        display_name=args.task_display_name,
        episode_info=lambda policy: {"word": policy._word},
        log_prefix="bimanual",
    )
    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
