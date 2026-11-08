"""Bookkeeping shared by the demonstration collectors.

``scripts/collect_demos.py`` (teleoperation), ``scripts/generate_dataset.py``
(trajectory replay) and ``data_synthetic/collection.py`` (scripted policies)
use these helpers to check the machine setup, create the LeRobot writer,
write ``meta/episode_NNNNNN_metadata.json`` and keep an episode save from
being cut short by a stop signal.
"""

from __future__ import annotations

import contextlib
import json
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from imbench.utils.fileio import atomic_write_json
from imbench.utils.lerobot_writer import LeRobotWriter
from imbench.utils.run_id import RunCounterError, imbench_dir, load_machine_id, read_counter

#: Signals that stop a collection run.  SIGINT raises KeyboardInterrupt by
#: default; :func:`stop_signals_raise_interrupt` makes the others do the same.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

_RATE_CATEGORIES = ("proprio", "force_torque", "object_state", "cameras", "tactile")


# ── Machine setup ──────────────────────────────────────────────────

def require_machine_id(log_prefix: str) -> str:
    """Return the validated machine ID, or exit with setup instructions."""
    try:
        return load_machine_id()
    except FileNotFoundError:
        print(f"[{log_prefix}] ERROR: machine not configured. "
              "Run:  python scripts/setup_machine.py")
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[{log_prefix}] ERROR: {exc}")
    sys.exit(1)


def require_run_counter(log_prefix: str) -> None:
    """Exit if the run counter cannot issue IDs."""
    try:
        read_counter()
    except RunCounterError as exc:
        print(f"[{log_prefix}] ERROR: {exc}")
        sys.exit(1)


# ── Dataset ────────────────────────────────────────────────────────

def make_writer(root: str | Path, task_id: str, task_cfg: dict) -> LeRobotWriter:
    """LeRobot writer for *task_id* at the task's control and per-stream save rates.

    Rate categories missing from ``recording.save_freq`` are saved at the
    control rate.
    """
    control_freq = task_cfg.get("control_freq", 20)
    save_freq = dict(task_cfg.get("recording", {}).get("save_freq", {}))
    for category in _RATE_CATEGORIES:
        save_freq.setdefault(category, control_freq)
    ctrl = task_cfg.get("controller", {})
    if isinstance(ctrl, list):
        ctrl = ctrl[0] if ctrl else {}
    robot_type = ctrl.get("robot", "Panda") if isinstance(ctrl, dict) else "Panda"
    return LeRobotWriter(
        root=root,
        task_id=task_id,
        control_freq=control_freq,
        save_freq=save_freq,
        features={},
        robot_type=robot_type,
        task_config=task_cfg,
    )


def session_config_path() -> Path:
    """``~/.imbench/session.json``: the teleop session written by ``scripts/dashboard.py``."""
    return imbench_dir() / "session.json"


def load_session() -> dict:
    """Return the dashboard's session config, or ``{}`` if absent or unreadable."""
    try:
        with open(session_config_path()) as f:
            session = json.load(f)
    except (OSError, ValueError):
        return {}
    return session if isinstance(session, dict) else {}


def write_episode_metadata(
    dataset: str | Path,
    episode_index: int,
    *,
    task_id: str,
    task_cfg: dict,
    display_name: str,
    run_id: str,
    n_steps: int,
    action_dim: int,
    final_score: float,
    success: bool,
    seed: Optional[int],
    warmup_steps: Optional[int],
    teleop_device: str,
    session: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write ``meta/episode_<episode_index>_metadata.json``.

    *session* (see :func:`load_session`) fills the task type and the
    operator, robot and version fields, which are ``null`` without it.
    *extra* is merged into ``dataset_summary``.
    """
    session = session or {}
    cameras = task_cfg.get("obs", {}).get("cameras", {})
    sensors = {
        name: f"{spec['width']}x{spec['height']}"
        for name, spec in cameras.items()
        if isinstance(spec, dict) and spec.get("enabled", False)
    }
    metadata = {
        "task_name":         display_name,
        "subtask_name":      None,
        "date":              datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "is_sim":            True,
        "run_id":            run_id,
        "task_id":           task_id,
        "task_type":         session.get("task_type") or task_cfg.get("task_type", "single_stage"),
        "outcome":           "success" if success else "failure",
        "failure_modes":     None if success else "task_incomplete",
        "recovery_actions":  None,
        "teleoperator_id":   session.get("teleoperator_id") or None,
        "safety_watcher_id": session.get("safety_watcher_id") or None,
        "supervisor_id":     session.get("supervisor_id") or None,
        "robot_model":       session.get("robot_model") or None,
        "robot_id":          session.get("robot_id") or None,
        "hardware_version":  session.get("hardware_version") or None,
        "software_version":  session.get("software_version") or None,
        "sim_id":            session.get("sim_id") or None,
        "datatype":          "lerobot v2.1",
        "sensors":           sensors,
        "seed":              seed,
        "warmup_steps":      warmup_steps,
        "dataset_summary": {
            "action":        [n_steps, action_dim],
            "episode":       episode_index,
            "teleop_device": teleop_device,
            "final_score":   round(float(final_score), 4),
            **(extra or {}),
        },
    }
    meta_dir = Path(dataset) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(meta_dir / f"episode_{episode_index:06d}_metadata.json", metadata, indent=4)


# ── Stop signals ───────────────────────────────────────────────────

def _in_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _honoured_stop_signals():
    """Stop signals the process has not been told to ignore (``nohup`` ignores SIGHUP)."""
    return [sig for sig in STOP_SIGNALS if signal.getsignal(sig) is not signal.SIG_IGN]


@contextlib.contextmanager
def stop_signals_raise_interrupt():
    """Deliver SIGTERM and SIGHUP as KeyboardInterrupt, like SIGINT; ignored signals stay ignored."""
    if not _in_main_thread():
        yield
        return

    def _raise(signum, frame):
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, _raise) for sig in _honoured_stop_signals() if sig != signal.SIGINT}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextlib.contextmanager
def deferred_stop_signals():
    """Hold stop signals until the block completes, then raise KeyboardInterrupt.

    Guards the write of an episode's parquet, videos and metadata so a stop
    signal cannot leave a partially saved episode.
    """
    if not _in_main_thread():
        yield
        return
    received = []
    previous = {sig: signal.signal(sig, lambda signum, frame: received.append(signum))
                for sig in _honoured_stop_signals()}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if received:
        raise KeyboardInterrupt
