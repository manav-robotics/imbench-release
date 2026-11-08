"""Multi-rate LeRobot v2.1 dataset writer.

Writes timestamped, multi-rate sensor data in the LeRobot v2.1 directory
layout:

    <dataset_root>/
        meta/
            info.json            # schema, per-feature fps, robot type
            episodes.jsonl       # one JSON line per episode
            tasks.jsonl          # task descriptions
        data/
            chunk-000/
                episode_000000.parquet   # scalar / vector streams
                episode_000001.parquet
                ...
        videos/
            chunk-000/
                <camera_label>/
                    episode_000000.mp4
                    ...

Each parquet row is one control step with an absolute ``timestamp``
(seconds since episode start) and an ``index`` (global frame counter).
Streams saved below the control rate are null on the steps where they are
not sampled.  Camera streams are saved as MP4 videos at their own fps and
referenced by frame index columns in the parquet
(``frame_index.<camera_label>``), which are null on steps without a frame.
Vectors are stored as one float64 column per element (``action.0``, ...),
not as the float32 vector features ``info.json`` declares, so upstream
LeRobot cannot load the parquet files.

An episode is committed by renaming its complete parquet file into place
before ``episodes.jsonl`` and ``info.json`` are updated, so a dataset
interrupted at any point reopens (a kill between the rename and the
``episodes.jsonl`` append leaves a parquet without its line, which merge
reports and skips).  Opening a
writer on an existing dataset resumes it: episode and frame indices continue
and the feature schema is seeded from the existing ``info.json``; an
unreadable parquet file is renamed aside with a warning.  A writer holds an
exclusive lock on the dataset until :meth:`LeRobotWriter.finalize`, so a
second writer on it raises :class:`DatasetLockedError`.

Usage
-----
    writer = LeRobotWriter(root="demos/lerobot", task_id="ib-p2-cube-toss-v1",
                           control_freq=100, save_freq={...}, features={...})
    writer.begin_episode()
    for step in range(T):
        writer.add_step(timestamp=..., obs=obs, reward=..., task_score=...)
    writer.end_episode(metadata={...})
    writer.finalize()
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import secrets
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from imbench.utils.fileio import atomic_write_json

logger = logging.getLogger(__name__)


class DatasetSchemaError(ValueError):
    """Recorded data does not match the schema of the dataset being resumed."""


class DatasetLockedError(RuntimeError):
    """Another writer, in this or another process, holds the dataset."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: str | Path) -> None:
    os.makedirs(p, exist_ok=True)


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    """Write *table* to *path* through a temporary file, so *path* is never partial."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "wb") as f:
            pq.write_table(table, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _same_json(a, b) -> bool:
    """Whether *a* and *b* are equal once stored as JSON (tuples become lists, etc.)."""
    return json.loads(json.dumps(a, default=str)) == json.loads(json.dumps(b, default=str))


def _stream_category(obs_key: str) -> str:
    """Map an observation key to a rate category used in save_freq config."""
    if obs_key.startswith("rgb_") or obs_key.startswith("depth_"):
        return "cameras"
    if obs_key.startswith("force_torque"):
        return "force_torque"
    if (obs_key in ("object_state", "object_pos", "object_quat",
                     "cube_pos", "cube_quat")
            or obs_key.startswith("obj_")):
        return "object_state"
    if "_tactile_" in obs_key:
        return "tactile"
    # Individual proprio keys (robot0_eef_pos, etc.) and any other
    # scalar streams default to the proprio save rate.
    return "proprio"


# ---------------------------------------------------------------------------
# Video encoder helper
# ---------------------------------------------------------------------------

class _VideoWriter:
    """Thin wrapper around OpenCV VideoWriter for one camera stream."""

    def __init__(self, path: str, fps: int, height: int, width: int):
        self.path = path
        _ensure_dir(os.path.dirname(path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer at {path}")
        self._frame_count = 0

    def write(self, rgb: np.ndarray) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)
        self._frame_count += 1

    def close(self) -> None:
        self._writer.release()

    @property
    def frame_count(self):
        return self._frame_count


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

class LeRobotWriter:
    """Multi-rate LeRobot v2.1 dataset writer.

    Parameters
    ----------
    root : str | Path
        Root directory for the dataset.
    task_id : str
        Identifier for the task (written into metadata).
    control_freq : int
        Rate at which the environment is stepped (Hz).
    save_freq : dict[str, int]
        Mapping of stream category -> save frequency in Hz.
        Categories: ``"proprio"``, ``"force_torque"``,
        ``"object_state"``, ``"cameras"``.
        Streams not listed default to ``control_freq``.
    features : dict
        Feature schema — mapping of feature name to dict with keys
        ``dtype`` (numpy dtype string) and ``shape`` (tuple).
        Camera features use ``dtype="video"``.
    robot_type : str
        Robot name for metadata (default ``"Panda"``).
    chunk_size : int
        Max episodes per chunk directory (default 1000).

    Raises :class:`DatasetLockedError` if another writer holds *root*.  The
    lock is released by :meth:`finalize`, garbage collection or process exit.
    """

    CODEBASE_VERSION = "v2.1"

    def __init__(
        self,
        root: str | Path,
        task_id: str,
        control_freq: int,
        save_freq: dict[str, int],
        features: dict[str, dict],
        robot_type: str = "Panda",
        chunk_size: int = 1000,
        task_config: dict | None = None,
    ) -> None:
        self.root = Path(root)
        self.task_id = task_id
        self.control_freq = control_freq
        self.save_freq = dict(save_freq)
        self.features = dict(features)
        self.robot_type = robot_type
        self.chunk_size = chunk_size
        # Dataset-level task config (a resumed dataset keeps the one it was
        # created with); episodes recorded with another one store their own.
        self.task_config = dict(task_config) if task_config else {}
        self._session_task_config = self.task_config

        # Shape of each obs key (first encounter wins) and the action size
        self._obs_shapes = {}   # obs_key -> list (shape)
        self._action_dim = None

        # Validate: every save_freq must divide control_freq evenly
        for cat, freq in self.save_freq.items():
            if control_freq % freq != 0:
                raise ValueError(
                    f"save_freq['{cat}']={freq} does not evenly divide "
                    f"control_freq={control_freq}"
                )

        # Compute decimation factors: step_skip[category] = N means save every N-th step
        self._step_skip = {
            cat: control_freq // freq for cat, freq in self.save_freq.items()
        }

        # State — resume from existing dataset if present
        self._episode_rows = []
        self._video_writers = {}   # camera_label -> _VideoWriter
        self._episode_started = False
        self._episode_start_index = 0

        # Prepare directory structure
        _ensure_dir(self.root / "meta")
        _ensure_dir(self.root / "data" / "chunk-000")
        self._lock_file = self._lock()

        # Detect existing episodes to support cumulative recording
        try:
            self._episode_index, self._global_index = self._detect_existing()
            self._load_existing_features()
        except BaseException:
            self._release_lock()
            raise
        # Schema of the dataset being resumed; new episodes must match it.
        self._resumed_obs_shapes = dict(self._obs_shapes)
        self._resumed_action_dim = self._action_dim

        # Write tasks.jsonl (single task — idempotent)
        tasks_path = self.root / "meta" / "tasks.jsonl"
        with open(tasks_path, "w") as f:
            f.write(json.dumps({"task_index": 0, "task": task_id}) + "\n")

    # ------------------------------------------------------------------
    # Dataset lock
    # ------------------------------------------------------------------

    def _lock(self):
        """Return the open ``meta/.writer.lock``, exclusively ``flock``-ed.

        The kernel drops the lock when the file is closed, including when the
        process dies, so a crashed collector never leaves a stale lock.
        """
        lock = open(self.root / "meta" / ".writer.lock", "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock.close()
            if isinstance(exc, BlockingIOError):
                raise DatasetLockedError(
                    f"{self.root} is in use: another collector is recording into it. "
                    "Wait for it to finish, or record into another directory."
                ) from None
            raise
        return lock

    def _release_lock(self) -> None:
        lock, self._lock_file = getattr(self, "_lock_file", None), None
        if lock is not None:
            lock.close()

    def __del__(self) -> None:
        self._release_lock()

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _detect_existing(self) -> tuple[int, int]:
        """Scan the dataset directory and return (next_episode_index, next_global_index).

        Reads existing parquet files to find the highest episode index and
        the total number of frames already recorded, so new episodes are
        appended rather than overwriting.  An unreadable file (e.g. one cut
        short by a crash of an older writer) is renamed aside and not counted.
        """
        next_episode = 0
        total_frames = 0

        # Scan all chunk dirs for parquet files
        data_root = self.root / "data"
        if data_root.exists():
            for pq_path in sorted(data_root.rglob("episode_*.parquet")):
                # Extract episode index from filename
                stem = pq_path.stem  # e.g. "episode_000003"
                try:
                    ep_idx = int(stem.split("_")[1])
                except (IndexError, ValueError):
                    continue

                # Count rows to get the global frame index
                try:
                    meta = pq.read_metadata(str(pq_path))
                except (pa.ArrowInvalid, OSError) as exc:
                    aside = pq_path.with_name(f"{pq_path.name}.{secrets.token_hex(4)}.unreadable")
                    os.rename(pq_path, aside)
                    logger.warning("[LeRobotWriter] %s is unreadable (%s); renamed it to %s",
                                   pq_path, exc, aside.name)
                    continue
                next_episode = max(next_episode, ep_idx + 1)
                total_frames += meta.num_rows

        if next_episode > 0:
            logger.info(
                "[LeRobotWriter] Resuming: found %d existing episodes, %d total frames",
                next_episode, total_frames,
            )

        return next_episode, total_frames

    def _load_existing_features(self) -> None:
        """Seed the schema, rates and task config from an existing ``meta/info.json``.

        Keeps the schema of a resumed dataset intact even when the session
        records no new episode.  Raises :class:`DatasetSchemaError` if the
        dataset was recorded at other rates.
        """
        info_path = self.root / "meta" / "info.json"
        try:
            with open(info_path) as f:
                info = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[LeRobotWriter] Ignoring unreadable %s: %s", info_path, exc)
            return
        rates = {"control_freq": self.control_freq, "save_freq": self.save_freq,
                 "chunk_size": self.chunk_size}
        changed = [f"{key} {info[key]} (not {value})" for key, value in rates.items()
                   if key in info and info[key] != value]
        if changed:
            raise DatasetSchemaError(
                f"{self.root} was recorded with {', '.join(changed)}. Record into a new "
                "directory, or use the task config it was recorded with."
            )
        if isinstance(info.get("task_config"), dict) and info["task_config"]:
            self.task_config = info["task_config"]
        for name, feat in info.get("features", {}).items():
            shape = feat.get("shape") if isinstance(feat, dict) else None
            if shape is None:
                continue
            if name == "action" and shape:
                self._action_dim = int(shape[0])
            elif name.startswith("observation."):
                self._obs_shapes[name[len("observation."):]] = list(shape)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    @property
    def episode_index(self) -> int:
        """Index the next saved episode will receive (equals episodes saved so far)."""
        return self._episode_index

    def begin_episode(self) -> None:
        """Start recording a new episode, discarding any unfinished one."""
        if self._lock_file is None:
            raise RuntimeError("finalize() was called; open a new writer to record more episodes")
        if self._episode_started:
            self.abort_episode()
        self._episode_rows = []
        self._video_writers = {}
        self._step_in_episode = 0
        self._episode_start_index = self._global_index
        self._episode_started = True

    def add_step(
        self,
        timestamp: float,
        obs: dict[str, np.ndarray],
        action: np.ndarray | None = None,
        reward: float = 0.0,
        task_score: float = 0.0,
        save_reward: bool = True,
        save_task_score: bool = True,
    ) -> None:
        """Record one control step.

        Parameters
        ----------
        timestamp : float
            Seconds since episode start.
        obs : dict
            Observation dict (already filtered to saveable keys).
        action : np.ndarray, optional
            Action array to save alongside observations.
        reward : float
            Reward received.
        task_score : float
            Normalised task completion score.
        save_reward : bool
            Whether to write the reward column.
        save_task_score : bool
            Whether to write the task_score column.
        """
        if not self._episode_started:
            raise RuntimeError("Call begin_episode() first")
        if self._step_in_episode == 0:
            self._check_resumed_schema(obs, action)

        step = self._step_in_episode
        row = {
            "index": self._global_index,
            "timestamp": timestamp,
            "episode_index": self._episode_index,
            "task_index": 0,
        }

        if save_reward:
            row["reward"] = float(reward)
        if save_task_score:
            row["task_score"] = float(task_score)

        # --- action ---
        if action is not None:
            action = np.asarray(action, dtype=np.float32).flatten()
            if self._action_dim is None:
                self._action_dim = len(action)
            for i, v in enumerate(action):
                row[f"action.{i}"] = float(v)

        # --- scalar observation streams ---
        for key, val in obs.items():
            cat = _stream_category(key)
            skip = self._step_skip.get(cat, 1)

            if key.startswith("rgb_") or key.startswith("depth_"):
                # handled below as video
                continue

            val = np.asarray(val, dtype=np.float32)
            # Record shape on first encounter
            if key not in self._obs_shapes:
                self._obs_shapes[key] = list(val.shape)

            if step % skip == 0:
                val_flat = val.flatten()
                for i, v in enumerate(val_flat):
                    row[f"observation.{key}.{i}"] = float(v)

        # --- camera streams (video) ---
        cam_skip = self._step_skip.get("cameras", 1)
        if step % cam_skip == 0:
            for key, val in obs.items():
                if not key.startswith("rgb_"):
                    continue
                cam_label = key  # e.g. "rgb_gripper_cam"
                val = np.asarray(val, dtype=np.uint8)
                h, w = val.shape[:2]
                # Record shape on first encounter
                if cam_label not in self._obs_shapes:
                    self._obs_shapes[cam_label] = [h, w, 3]

                if cam_label not in self._video_writers:
                    chunk = f"chunk-{self._episode_index // self.chunk_size:03d}"
                    vid_dir = self.root / "videos" / chunk / cam_label
                    vid_path = vid_dir / f"episode_{self._episode_index:06d}.mp4"
                    cam_fps = self.save_freq.get("cameras", self.control_freq)
                    self._video_writers[cam_label] = _VideoWriter(
                        str(vid_path), cam_fps, h, w,
                    )

                vw = self._video_writers[cam_label]
                vw.write(val)
                row[f"frame_index.{cam_label}"] = vw.frame_count - 1

            # Also handle depth cameras
            for key, val in obs.items():
                if not key.startswith("depth_"):
                    continue
                cam_label = key
                val = np.asarray(val, dtype=np.float32)
                if cam_label not in self._obs_shapes:
                    self._obs_shapes[cam_label] = list(val.shape)
                # Depth stored as 16-bit PNG-in-video is lossy; store as
                # separate scalar columns instead (flattened).
                val_flat = val.flatten()
                for i, v in enumerate(val_flat):
                    row[f"observation.{cam_label}.{i}"] = float(v)

        self._episode_rows.append(row)
        self._global_index += 1
        self._step_in_episode += 1

    def _check_resumed_schema(self, obs: dict, action) -> None:
        """Raise :class:`DatasetSchemaError` if *obs* or *action* differ from the resumed dataset."""
        if not self._resumed_obs_shapes:
            return
        shapes = {
            key: [*np.shape(val)[:2], 3] if key.startswith("rgb_") else list(np.shape(val))
            for key, val in obs.items()
        }
        problems = []
        missing = sorted(set(self._resumed_obs_shapes) - set(shapes))
        extra = sorted(set(shapes) - set(self._resumed_obs_shapes))
        if missing:
            problems.append(f"missing {missing}")
        if extra:
            problems.append(f"unexpected {extra}")
        problems += [
            f"{key} has shape {shapes[key]}, not {shape}"
            for key, shape in sorted(self._resumed_obs_shapes.items())
            if key in shapes and shapes[key] != list(shape)
        ]
        if action is not None and self._resumed_action_dim is not None \
                and np.size(action) != self._resumed_action_dim:
            problems.append(f"action has {np.size(action)} values, not {self._resumed_action_dim}")
        if problems:
            raise DatasetSchemaError(
                f"{self.root} holds data with a different observation schema ({'; '.join(problems)}). "
                "Record into a new directory, or use the obs mode and cameras it was recorded with."
            )

    def abort_episode(self) -> None:
        """Discard the in-progress episode: delete its partial videos and roll back indices."""
        if not self._episode_started:
            return
        for vw in self._video_writers.values():
            vw.close()
            with contextlib.suppress(FileNotFoundError):
                os.remove(vw.path)
        self._video_writers = {}
        self._episode_rows = []
        self._global_index = self._episode_start_index
        self._step_in_episode = 0
        self._episode_started = False

    def end_episode(self, metadata: dict | None = None) -> None:
        """Finalise the current episode: write parquet + close videos."""
        if not self._episode_started:
            return

        # Close video writers
        for vw in self._video_writers.values():
            vw.close()
        self._video_writers = {}

        # Write parquet.  Renaming it into place commits the episode: the
        # videos are closed before and episodes.jsonl / info.json updated
        # after, so an interrupted save never leaves metadata without data.
        chunk = f"chunk-{self._episode_index // self.chunk_size:03d}"
        data_dir = self.root / "data" / chunk
        _ensure_dir(data_dir)

        table = pa.Table.from_pylist(self._episode_rows)
        pq_path = data_dir / f"episode_{self._episode_index:06d}.parquet"
        _write_parquet_atomic(table, pq_path)

        # Append to episodes.jsonl
        ep_meta = {
            "episode_index": self._episode_index,
            "task_index": 0,
            "task": self.task_id,
            "length": self._step_in_episode,
            "timestamp_start": self._episode_rows[0]["timestamp"] if self._episode_rows else 0.0,
            "timestamp_end": self._episode_rows[-1]["timestamp"] if self._episode_rows else 0.0,
        }
        if metadata:
            ep_meta.update(metadata)
        if "task_config" not in ep_meta and not _same_json(self._session_task_config, self.task_config):
            ep_meta["task_config"] = self._session_task_config

        ep_jsonl = self.root / "meta" / "episodes.jsonl"
        with open(ep_jsonl, "a") as f:
            f.write(json.dumps(ep_meta) + "\n")

        self._episode_index += 1
        self._episode_rows = []
        self._episode_started = False
        self._write_info()

    def finalize(self) -> None:
        """Save any in-progress episode, write ``meta/info.json`` and release the dataset lock."""
        if self._lock_file is None:
            return
        try:
            if self._episode_started:
                self.end_episode()
            else:
                self._write_info()
        finally:
            self._release_lock()

    def _write_info(self) -> None:
        """Atomically (re)write ``meta/info.json`` for the episodes on disk."""
        # ── Build feature schema ────────────────────────────────────
        feature_info = {}

        # Action
        if self._action_dim is not None:
            feature_info["action"] = {
                "dtype": "float32",
                "shape": [self._action_dim],
                "fps": self.control_freq,
                "description": "Robot action vector",
            }

        # Reward / task_score — check if they exist in the first parquet
        chunk_dir = self.root / "data" / "chunk-000"
        pq_files = sorted(chunk_dir.glob("*.parquet"))
        first_schema_names = set()
        if pq_files:
            schema = pq.read_schema(str(pq_files[0]))
            first_schema_names = {f.name for f in schema}

        if "reward" in first_schema_names:
            feature_info["reward"] = {
                "dtype": "float32",
                "shape": [1],
                "fps": self.control_freq,
            }
        if "task_score" in first_schema_names:
            feature_info["task_score"] = {
                "dtype": "float32",
                "shape": [1],
                "fps": self.control_freq,
            }

        # Scalar observation features — use tracked shapes
        for obs_key, shape in sorted(self._obs_shapes.items()):
            if obs_key.startswith("rgb_") or obs_key.startswith("depth_"):
                continue  # handled below as video / depth
            cat = _stream_category(obs_key)
            feature_info[f"observation.{obs_key}"] = {
                "dtype": "float32",
                "shape": shape,
                "fps": self.save_freq.get(cat, self.control_freq),
            }

        # Video features
        vid_root = self.root / "videos"
        if vid_root.exists():
            for cam_dir in sorted(vid_root.rglob("chunk-*/*")):
                if cam_dir.is_dir() and any(cam_dir.glob("*.mp4")):
                    cam_label = cam_dir.name
                    shape = self._obs_shapes.get(cam_label, [])
                    feature_info[f"observation.{cam_label}"] = {
                        "dtype": "video",
                        "shape": shape,
                        "fps": self.save_freq.get("cameras", self.control_freq),
                        "video_path": f"videos/{{chunk}}/{cam_label}/{{episode}}.mp4",
                    }
                    # Frame index column
                    feature_info[f"frame_index.{cam_label}"] = {
                        "dtype": "int64",
                        "shape": [1],
                        "description": f"Video frame index for {cam_label}",
                    }

        # Depth features (stored as flattened scalars, not video)
        for obs_key, shape in sorted(self._obs_shapes.items()):
            if not obs_key.startswith("depth_"):
                continue
            feature_info[f"observation.{obs_key}"] = {
                "dtype": "float32",
                "shape": shape,
                "fps": self.save_freq.get("cameras", self.control_freq),
            }

        # ── Assemble info.json ──────────────────────────────────────
        info = {
            "codebase_version": self.CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "task_id": self.task_id,
            "total_episodes": self._episode_index,
            "total_frames": self._global_index,
            "fps": self.control_freq,
            "control_freq": self.control_freq,
            "save_freq": self.save_freq,
            "chunk_size": self.chunk_size,
            "features": feature_info,
            "task_config": self.task_config,
        }

        atomic_write_json(self.root / "meta" / "info.json", info)
