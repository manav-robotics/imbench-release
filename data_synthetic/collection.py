"""Shared episode loop and bookkeeping for the scripted-demo collectors.

Every collector in ``data_synthetic/`` runs through :func:`collect`, so they
share one on-disk contract:

* Episodes go to ``<out-dir>/<task-id>/M<machine-id>/`` as a LeRobot v2.1
  dataset, plus ``meta/episode_NNNNNN_metadata.json`` per episode.  Parquet,
  MP4, metadata and ``npy/traj_NNNNNN.npy`` share the writer's episode index.
* Only episodes that end with ``info["is_success"]`` are saved; failed
  attempts and an episode cut short by Ctrl-C, SIGTERM or SIGHUP are
  discarded.
* Attempt ``k`` resets the task with seed ``first_seed + k`` and seeds the
  policy's ``rng`` (a ``numpy.random.Generator``) from the same value, so a
  saved episode is reproducible from its metadata ``seed``.  ``first_seed``
  defaults to ``machine_id * 1_000_000`` and a resumed dataset continues
  after the largest seed it already holds.
* Actions are clipped to ``env.action_space`` and cast to float32.  The
  recorded action is ``info["applied_action"]`` when the task reports one
  (arm 0 of ib-m-keyboard-typing-v1 is replaced by its scripted arm), else
  the array passed to ``env.step()``.  The frozen arm of
  ib-p1-cracker-box-suction/jaw-v1 still records the policy's values in its
  ignored columns; replay is unaffected because the override is deterministic.
* ``episode_length_stats.json`` in the dataset holds the step-count stats of
  every episode the dataset contains.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

import imbench as ib
from imbench.tasks.base_task import BaseTask
from imbench.utils.dashboard_reporter import report_episode
from imbench.utils.lerobot_writer import DatasetLockedError, DatasetSchemaError
from imbench.utils.recording import (
    deferred_stop_signals,
    make_writer,
    require_machine_id,
    require_run_counter,
    stop_signals_raise_interrupt,
    write_episode_metadata,
)
from imbench.utils.robot_state import extract_robot_state
from imbench.utils.run_id import next_run_id
from imbench.utils.task_names import task_display_name

from data_synthetic.episode_length_stats import record_run

DEFAULT_OUT_ROOT = "demos"
SEEDS_PER_MACHINE = 1_000_000

#: ``policy_factory(env, obs, rng)`` returns a policy ready to act on *obs*,
#: or ``None`` when no plan exists for the scene (the attempt then fails).
PolicyFactory = Callable[[Any, dict, np.random.Generator], Optional[Any]]


# ── Identity and layout ────────────────────────────────────────────

def dataset_dir(out_root: Optional[str], task_id: str, machine_id: str) -> str:
    """``<out_root>/<task_id>/M<machine_id>``; *out_root* may already end in ``<task_id>``."""
    root = out_root or DEFAULT_OUT_ROOT
    if os.path.basename(os.path.normpath(root)) != task_id:
        root = os.path.join(root, task_id)
    return os.path.join(root, f"M{machine_id}")


def largest_recorded_seed(dataset: str) -> Optional[int]:
    """Largest ``seed`` in the dataset's per-episode metadata, or ``None``."""
    seeds = []
    for path in Path(dataset, "meta").glob("episode_*_metadata.json"):
        try:
            with open(path) as f:
                seed = json.load(f).get("seed")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(seed, int) and not isinstance(seed, bool):
            seeds.append(seed)
    return max(seeds) if seeds else None


def dataset_episode_lengths(dataset: str) -> list[int]:
    """Step count of every episode in the dataset's ``meta/episodes.jsonl``."""
    try:
        with open(Path(dataset, "meta", "episodes.jsonl")) as f:
            return [int(json.loads(line)["length"]) for line in f if line.strip()]
    except OSError:
        return []


# ── Policies ───────────────────────────────────────────────────────

def import_policy(task_id: str, registry: dict[str, tuple[str, str]], package: str) -> type:
    """Import the policy class *registry* maps exactly *task_id* to.

    *registry* maps task ids to ``(module, class)`` inside *package*; modules
    are imported on demand so a broken policy only affects its own task.
    """
    entry = registry.get(task_id)
    if entry is None:
        if re.search(r"-v(?!1$)\d+$", task_id):
            raise ValueError(
                f"{task_id!r} is an out-of-distribution evaluation variant; scripted "
                "demonstrations are only provided for the -v1 training tasks."
            )
        raise ValueError(
            f"No scripted policy for {task_id!r} in this collector. "
            f"Registered tasks: {', '.join(sorted(registry))}"
        )
    module_name, class_name = entry
    return getattr(importlib.import_module(f"{package}.{module_name}"), class_name)


def build_policy(policy_cls: type, env, rng: np.random.Generator):
    """Instantiate *policy_cls* for one episode and ``reset()`` it.

    ``env`` and ``rng`` are passed to the constructor when it accepts them;
    ``policy.rng`` is set either way, so the policy draws only from *rng*.
    """
    params = inspect.signature(policy_cls).parameters
    kwargs = {name: value for name, value in (("env", env), ("rng", rng)) if name in params}
    policy = policy_cls(**kwargs)
    policy.rng = rng
    policy.reset()
    return policy


def clip_action(action, space) -> np.ndarray:
    """Clip *action* to the Box *space* and return it as float32."""
    action = np.asarray(action, dtype=np.float64).reshape(space.shape)
    return np.clip(action, space.low, space.high).astype(np.float32)


# ── Display ────────────────────────────────────────────────────────

def display_available() -> bool:
    """Whether an on-screen window can be opened (always true off Linux)."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


class _Dashboard:
    """Integrated viewer + camera-strip dashboard, throttled to *max_fps*."""

    def __init__(self, env, window_name: str, visualize_ft: bool, max_fps: int = 25):
        from imbench.utils.obs_visualizer import IntegratedDashboard, _screen_height, _screen_width

        env._viewer_target_geometry = (0, 0, _screen_width(), int(_screen_height() * 0.70))
        self._dash = IntegratedDashboard(
            window_name=window_name, max_fps=max_fps, visualize_ft=visualize_ft,
        )
        self._min_dt = 1.0 / max_fps
        self._last_update = 0.0

    def post(self, obs, info, reward) -> None:
        now = time.monotonic()
        if now - self._last_update >= self._min_dt:
            self._dash.update(obs, info, reward=reward)
            self._last_update = now

    def reset(self) -> None:
        """Clear the per-episode F/T history and tare."""
        self._dash.reset()

    def close(self) -> None:
        self._dash.close()


# ── Collection loop ────────────────────────────────────────────────

def attempt_budget(n_episodes: int) -> int:
    """Maximum attempts for *n_episodes* successes."""
    return max(10 * n_episodes, n_episodes + 20)


@dataclass
class CollectionResult:
    n_requested: int
    dataset: Optional[str] = None
    n_saved: int = 0
    attempts: int = 0
    scores: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    interrupted: bool = False

    @property
    def exit_code(self) -> int:
        """0 when every requested episode was saved, 130 on Ctrl-C, else 1."""
        if self.interrupted:
            return 130
        return 0 if self.n_saved >= self.n_requested else 1


@dataclass
class _Episode:
    seed: int
    score: float = 0.0
    success: bool = False
    steps: int = 0
    actions: list = field(default_factory=list)
    policy: Any = None


@stop_signals_raise_interrupt()
def collect(
    *,
    task_id: str,
    policy_factory: PolicyFactory,
    n_episodes: int,
    out_dir: Optional[str] = None,
    render: bool = False,
    use_dashboard: bool = True,
    obs_mode: str = "combo",
    max_steps: Optional[int] = None,
    config_overrides: Optional[dict] = None,
    seed: Optional[int] = None,
    save_lerobot: bool = True,
    save_npy: bool = False,
    warmup_steps: int = 0,
    stop_when_policy_done: bool = False,
    stop_when_policy_failed: bool = False,
    display_name: Optional[str] = None,
    generation_method: str = "scripted_policy",
    episode_info: Optional[Callable[[Any], dict]] = None,
    log_prefix: str = "collect",
    max_attempts: Optional[int] = None,
) -> CollectionResult:
    """Collect *n_episodes* successful episodes of *task_id*.

    Runs at most ``attempt_budget(n_episodes)`` (or *max_attempts*) episodes.
    With *render*, *use_dashboard* also opens the camera/observation
    dashboard. *config_overrides* customizes the task for collection and is
    preserved in each saved episode's effective task config for replay.
    *max_steps* overrides the configured ``max_episode_steps``.
    *warmup_steps* zero-action steps run after the policy is built and before
    recording.
    The episode ends on termination or truncation, and also when the policy
    sets ``done`` (*stop_when_policy_done*) or ``failed``
    (*stop_when_policy_failed*).  *episode_info(policy)* adds per-episode
    fields to the saved metadata.  With *save_lerobot* off nothing is written
    unless *save_npy* is set, in which case actions go to
    ``npy/seed_<seed>.npy``.  Exits with status 1 if another collector is
    recording into the dataset or it holds data with a different schema.
    """
    display_name = display_name or task_display_name(task_id)
    machine_id = require_machine_id(log_prefix)
    if save_lerobot:
        require_run_counter(log_prefix)

    if render and not display_available():
        print(f"[{log_prefix}] WARNING: no display available; running headless.")
        render = False
    use_dashboard = use_dashboard and render

    dataset = dataset_dir(out_dir, task_id, machine_id)
    config_overrides = deepcopy(config_overrides) if config_overrides else {}
    if max_steps is not None:
        config_overrides["max_episode_steps"] = max_steps
    env = ib.make(
        task_id,
        privileged_obs=True,
        obs_mode=obs_mode,
        render_mode="human" if render else None,
        config_overrides=config_overrides or None,
    )
    task_cfg = env.cfg
    control_freq = task_cfg.get("control_freq", 20)
    budget = max_attempts if max_attempts is not None else attempt_budget(n_episodes)
    result = CollectionResult(n_requested=n_episodes, dataset=dataset if save_lerobot else None)

    writer = None
    dashboard = None
    try:
        writer = make_writer(dataset, task_id, task_cfg) if save_lerobot else None
        # Read after the writer has locked the dataset, so every seed saved by
        # a collector that held it until now is seen.
        first_seed = seed if seed is not None else int(machine_id) * SEEDS_PER_MACHINE
        recorded = largest_recorded_seed(dataset) if writer is not None else None
        if recorded is not None and recorded >= first_seed:
            print(f"[{log_prefix}] {dataset} already holds seed {recorded}; continuing after it.")
            first_seed = recorded + 1
        if use_dashboard:
            dashboard = _Dashboard(
                env, f"{display_name}: {task_id}", bool(task_cfg.get("visualize_FT", False)),
            )
        start_index = writer.episode_index if writer is not None else 0
        npy_dir = os.path.join(dataset, "npy") if save_npy else None

        print(f"[{log_prefix}] task={task_id}  obs_mode={obs_mode}  control_freq={control_freq} Hz  "
              f"horizon={env.MAX_EPISODE_STEPS}")
        if writer is not None:
            print(f"[{log_prefix}] saving to {dataset}/  ({start_index} existing episodes)")
        else:
            print(f"[{log_prefix}] LeRobot output disabled")
        print(f"[{log_prefix}] collecting {n_episodes} successful episodes in at most "
              f"{budget} attempts, seeds from {first_seed}\n")

        while result.n_saved < n_episodes and result.attempts < budget:
            episode = _Episode(seed=first_seed + result.attempts)
            result.attempts += 1
            _run_episode(
                env, episode, policy_factory, writer, dashboard,
                render=render, warmup_steps=warmup_steps, control_freq=control_freq,
                stop_when_policy_done=stop_when_policy_done,
                stop_when_policy_failed=stop_when_policy_failed,
            )
            result.scores.append(episode.score)
            phase = getattr(episode.policy, "phase", None)
            summary = (f"seed={episode.seed}  score={episode.score:.3f}  steps={episode.steps}"
                       + (f"  phase={phase}" if phase is not None else "")
                       + ("  (no plan)" if episode.policy is None else ""))

            if not episode.success:
                if writer is not None:
                    writer.abort_episode()
                print(f"  attempt {result.attempts}/{budget}  {summary}  [FAILURE - discarded]")
                continue

            extra = episode_info(episode.policy) if episode_info is not None else {}
            if config_overrides:
                # Per-episode config survives resuming with another vocabulary
                # or merging datasets whose collection settings differ.
                extra = {**extra, "task_config": task_cfg}
            with deferred_stop_signals():
                label = _save_episode(
                    episode, writer, npy_dir,
                    dataset=dataset, task_id=task_id, task_cfg=task_cfg,
                    display_name=display_name, machine_id=machine_id,
                    action_dim=int(env.action_space.shape[0]), warmup_steps=warmup_steps,
                    generation_method=generation_method, extra=extra,
                )
                result.n_saved += 1
                result.lengths.append(episode.steps)
            print(f"  attempt {result.attempts}/{budget}  {summary}  "
                  f"[SUCCESS {result.n_saved}/{n_episodes}: {label}]")

    except KeyboardInterrupt:
        result.interrupted = True
        print(f"\n[{log_prefix}] interrupted; the episode in progress was discarded.")
    except (DatasetLockedError, DatasetSchemaError) as exc:
        print(f"\n[{log_prefix}] ERROR: {exc}")
        sys.exit(1)
    finally:
        if dashboard is not None:
            dashboard.close()
        env.close()
        if writer is not None:
            writer.abort_episode()  # an unfinished episode is never saved
            writer.finalize()

    _print_summary(result, log_prefix, writer)
    if writer is not None:
        record_run(out_dir=dataset, task_id=task_id, task_display_name=display_name,
                   lengths=dataset_episode_lengths(dataset))
    return result


def _save_episode(
    episode: _Episode,
    writer,
    npy_dir: Optional[str],
    *,
    dataset: str,
    task_id: str,
    task_cfg: dict,
    display_name: str,
    machine_id: str,
    action_dim: int,
    warmup_steps: int,
    generation_method: str,
    extra: dict,
) -> str:
    """Persist a successful episode and return a short label for the log."""
    if writer is None:
        label = f"seed {episode.seed}"
        npy_name = f"seed_{episode.seed}.npy"
    else:
        index = writer.episode_index
        run_id = next_run_id(machine_id)
        writer.end_episode(metadata={
            "final_score":       episode.score,
            "success":           True,
            "teleop_device":     "scripted_policy",
            "run_id":            run_id,
            "generation_method": generation_method,
            "seed":              episode.seed,
            "warmup_steps":      warmup_steps,
            **extra,
        })
        write_episode_metadata(
            dataset, index,
            task_id=task_id, task_cfg=task_cfg, display_name=display_name,
            run_id=run_id, n_steps=episode.steps, action_dim=action_dim,
            final_score=episode.score, success=True, seed=episode.seed,
            warmup_steps=warmup_steps, teleop_device="scripted_policy",
            extra={"generation_method": generation_method,
                   "policy": type(episode.policy).__name__, **extra},
        )
        report_episode(
            task_name=display_name, outcome="success", score=episode.score,
            episode_id=run_id, machine_id=machine_id,
        )
        label = f"episode {index:06d}  run_id={run_id}"
        npy_name = f"traj_{index:06d}.npy"
    if npy_dir is not None:
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, npy_name), np.asarray(episode.actions))
    return label


def _run_episode(
    env,
    episode: _Episode,
    policy_factory: PolicyFactory,
    writer,
    dashboard,
    *,
    render: bool,
    warmup_steps: int,
    control_freq: int,
    stop_when_policy_done: bool,
    stop_when_policy_failed: bool,
) -> None:
    space = env.action_space
    dt = 1.0 / control_freq

    obs, info = env.reset(seed=episode.seed)
    if dashboard is not None:
        dashboard.reset()
    policy = policy_factory(env, obs, np.random.default_rng(episode.seed))
    episode.policy = policy
    if policy is None:
        return
    act_takes_env = "env" in inspect.signature(policy.act).parameters

    env.align_camera_frames(warmup_steps)  # saved frames are rendered on their own step
    for _ in range(warmup_steps):
        obs, _, _, _, info = env.step(np.zeros(space.shape, dtype=np.float32))
    if render:
        env.render()
    if dashboard is not None:
        dashboard.post(obs, info, 0.0)

    if writer is not None:
        writer.begin_episode()
    save_extras = env.save_extras
    done = False
    while not done:
        t_loop = time.perf_counter()
        raw = policy.act(obs, env=env) if act_takes_env else policy.act(obs)
        action = clip_action(raw, space)

        if writer is not None:
            snapshot = {k: v.copy() for k, v in obs.items()}
            snapshot["joint_positions"], snapshot["ee_poses"] = extract_robot_state(env)

        obs, reward, terminated, truncated, info = env.step(action)
        action = np.asarray(info.get("applied_action", action), dtype=action.dtype)

        if writer is not None:
            writer.add_step(
                timestamp=episode.steps * dt,
                obs=env.filter_obs_for_saving(snapshot),
                action=action if save_extras["action"] else None,
                reward=reward,
                task_score=info["task_score"],
                save_reward=save_extras["reward"],
                save_task_score=save_extras["task_score"],
            )
        episode.actions.append(action)
        episode.steps += 1

        if render:
            env.render()
        if dashboard is not None:
            dashboard.post(obs, info, reward)

        done = (
            terminated or truncated
            or (stop_when_policy_done and bool(getattr(policy, "done", False)))
            or (stop_when_policy_failed and bool(getattr(policy, "failed", False)))
        )
        if render:
            remaining = dt - (time.perf_counter() - t_loop)
            if remaining > 0:
                time.sleep(remaining)

    episode.score = float(info["task_score"])
    episode.success = bool(info["is_success"])


def _print_summary(result: CollectionResult, log_prefix: str, writer) -> None:
    print(f"\n[{log_prefix}] done: {result.n_saved}/{result.n_requested} successful episodes "
          f"saved in {result.attempts} attempts.")
    if result.scores:
        scores = np.asarray(result.scores)
        print(f"  attempt success rate: {result.n_saved / result.attempts:.1%}   "
              f"mean final score: {scores.mean():.3f} ± {scores.std():.3f}")
    if writer is not None:
        print(f"  {writer.episode_index} episodes in {result.dataset}/")
    if not result.interrupted and result.n_saved < result.n_requested:
        print(f"[{log_prefix}] ERROR: attempt budget exhausted before reaching "
              f"{result.n_requested} successes.")


# ── CLI helpers ────────────────────────────────────────────────────

def _int_at_least(minimum: int):
    """argparse ``type`` accepting integers >= *minimum*."""
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}, got {value}")
        return value
    return parse


def add_collection_args(parser, *, n_episodes_default: int, lerobot_toggle: bool = False) -> None:
    """Add the flags shared by every scripted collector to *parser*."""
    parser.add_argument(
        "--n-episodes", type=_int_at_least(1), default=n_episodes_default,
        help=f"Successful episodes to save (default: {n_episodes_default}). Failed attempts are "
             "discarded; at most max(10*N, N+20) attempts are made and the exit status is "
             "non-zero if fewer than N succeed.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=f"Output root (default: {DEFAULT_OUT_ROOT}). Episodes go to "
             "<out-dir>/<task-id>/M<machine-id>/; existing episodes there are kept and "
             "collection resumes after them.",
    )
    parser.add_argument(
        "--max-steps", type=_int_at_least(1), default=None,
        help="Per-episode step limit (default: the task YAML's max_episode_steps).",
    )
    parser.add_argument(
        "--seed", type=_int_at_least(0), default=None,
        help=f"Seed of the first attempt; attempt k uses seed+k (default: machine_id * "
             f"{SEEDS_PER_MACHINE}). A resumed dataset continues after its largest recorded seed.",
    )
    parser.add_argument(
        "--obs-mode", choices=BaseTask.OBS_MODES, default="combo",
        help="Observation mode to record (default: combo = proprio + cameras).",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Open the MuJoCo viewer (needs a display; ignored with a warning when headless).",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="With --render, do not open the camera/observation dashboard.",
    )
    parser.add_argument(
        "--save-npy", action="store_true",
        help="Also save each saved episode's actions to <dataset>/npy/traj_NNNNNN.npy.",
    )
    if lerobot_toggle:
        parser.add_argument(
            "--no-lerobot", action="store_true",
            help="Do not write a dataset (dry run); with --save-npy only npy/seed_<seed>.npy "
                 "files are written.",
        )


def run_task_cli(
    task_id: str,
    policy_factory: PolicyFactory,
    *,
    description: str,
    epilog: Optional[str] = None,
    log_prefix: str,
    **collect_kwargs,
) -> None:
    """Command-line entry point for a collector dedicated to one task; exits with
    :attr:`CollectionResult.exit_code`."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    add_collection_args(parser, n_episodes_default=10, lerobot_toggle=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")
    result = collect(
        task_id=task_id,
        policy_factory=policy_factory,
        n_episodes=args.n_episodes,
        out_dir=args.out_dir,
        render=args.render,
        use_dashboard=not args.no_dashboard,
        obs_mode=args.obs_mode,
        max_steps=args.max_steps,
        seed=args.seed,
        save_lerobot=not args.no_lerobot,
        save_npy=args.save_npy,
        log_prefix=log_prefix,
        **collect_kwargs,
    )
    sys.exit(result.exit_code)
