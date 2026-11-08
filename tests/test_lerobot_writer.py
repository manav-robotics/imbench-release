"""Tests for the multi-rate LeRobot dataset writer.

Run with::

    pytest tests/test_lerobot_writer.py -v

No simulator, teleop device, or GPU required — uses synthetic data only.
"""

import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from imbench.utils.lerobot_writer import (
    DatasetLockedError,
    DatasetSchemaError,
    LeRobotWriter,
    _stream_category,
)

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONTROL_FREQ = 100
SAVE_FREQ = {
    "proprio": 100,
    "force_torque": 100,
    "object_state": 50,
    "cameras": 10,
}
TASK_ID = "test-task"
# Individual proprio key dimensions (must sum to PROPRIO_TOTAL for column counting)
PROPRIO_KEYS = {
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_gripper_qpos": 2,
    "robot0_joint_pos": 7,
    "robot0_joint_vel": 7,
}
PROPRIO_TOTAL = sum(PROPRIO_KEYS.values())  # 23
FT_DIM = 6
OBJ_DIM = 14
CAM_H, CAM_W = 64, 64


@pytest.fixture()
def dataset_dir(tmp_path):
    """Return a fresh temporary directory for each test."""
    d = tmp_path / "test_dataset"
    yield d
    # cleanup is automatic with tmp_path


def _make_writer(root, **overrides):
    kwargs = dict(
        root=root,
        task_id=TASK_ID,
        control_freq=CONTROL_FREQ,
        save_freq=dict(SAVE_FREQ),
        features={},
        robot_type="Panda",
    )
    kwargs.update(overrides)
    return LeRobotWriter(**kwargs)


def _fake_obs(step):
    """Generate a synthetic observation dict for a given step index."""
    obs = {k: np.random.randn(dim).astype(np.float32) for k, dim in PROPRIO_KEYS.items()}
    obs["force_torque"] = np.random.randn(FT_DIM).astype(np.float32)
    obs["object_state"] = np.random.randn(OBJ_DIM).astype(np.float32)
    obs["rgb_front_cam"] = np.random.randint(0, 255, (CAM_H, CAM_W, 3), dtype=np.uint8)
    return obs


def _record_episode(writer, n_steps=50):
    """Record a single episode of synthetic data into the writer."""
    dt = 1.0 / CONTROL_FREQ
    writer.begin_episode()
    for step in range(n_steps):
        obs = _fake_obs(step)
        writer.add_step(
            timestamp=step * dt,
            obs=obs,
            reward=float(step) * 0.01,
            task_score=float(step) / n_steps,
        )
    writer.end_episode(metadata={"final_score": 0.95})


def _python(code, *args, **kwargs):
    """Start ``python -c code args`` on this checkout (it can import this module)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    return subprocess.Popen([sys.executable, "-c", code, *map(str, args)],
                            cwd=REPO, env=env, **kwargs)


# Holds a writer on argv[1] until killed.
_HOLD_LOCK = """
import sys, time
from tests.test_lerobot_writer import _make_writer
writer = _make_writer(sys.argv[1])
print("locked", flush=True)
time.sleep(600)
"""

# Saves one episode into argv[1], then is SIGKILLed halfway through writing
# the second episode's parquet.
_KILLED_DURING_SAVE = """
import os, signal, sys
import pyarrow as pa
import pyarrow.parquet as pq
from tests.test_lerobot_writer import _make_writer, _record_episode

def write_half_then_die(table, where, **kwargs):
    sink = pa.BufferOutputStream()
    write_table(table, sink, **kwargs)
    data = sink.getvalue().to_pybytes()
    f = open(where, "wb") if isinstance(where, (str, os.PathLike)) else where
    f.write(data[: len(data) // 2])
    f.flush()
    os.kill(os.getpid(), signal.SIGKILL)

writer = _make_writer(sys.argv[1])
_record_episode(writer, n_steps=20)
write_table, pq.write_table = pq.write_table, write_half_then_die
_record_episode(writer, n_steps=20)
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDirectoryLayout:
    """Verify the on-disk directory and file structure."""

    def test_creates_meta_directory(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        writer.finalize()

        assert (dataset_dir / "meta" / "info.json").exists()
        assert (dataset_dir / "meta" / "episodes.jsonl").exists()
        assert (dataset_dir / "meta" / "tasks.jsonl").exists()

    def test_creates_parquet_per_episode(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        _record_episode(writer, n_steps=30)
        writer.finalize()

        pq_files = sorted((dataset_dir / "data" / "chunk-000").glob("*.parquet"))
        assert len(pq_files) == 2
        assert pq_files[0].name == "episode_000000.parquet"
        assert pq_files[1].name == "episode_000001.parquet"

    def test_creates_video_per_camera_per_episode(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        writer.finalize()

        vid_files = list((dataset_dir / "videos").rglob("*.mp4"))
        assert len(vid_files) >= 1
        # Should have a video for rgb_front_cam
        names = [f.parent.name for f in vid_files]
        assert "rgb_front_cam" in names


class TestMultiRate:
    """Verify that streams are decimated to the correct save frequencies."""

    def test_proprio_saved_every_step(self, dataset_dir):
        """proprio at 100 Hz with control_freq=100 -> every row has it."""
        n_steps = 50
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=n_steps)
        writer.finalize()

        table = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        assert len(table) == n_steps

        # Every row should have individual proprio columns
        proprio_cols = []
        for key in PROPRIO_KEYS:
            cols = [c for c in table.column_names if c.startswith(f"observation.{key}.")]
            proprio_cols.extend(cols)
        assert len(proprio_cols) == PROPRIO_TOTAL
        for col_name in proprio_cols:
            col = table.column(col_name)
            non_null = col.drop_null()
            assert len(non_null) == n_steps, f"{col_name} should be present in every row"

    def test_object_state_decimated(self, dataset_dir):
        """object_state at 50 Hz with control_freq=100 -> every 2nd row."""
        n_steps = 100
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=n_steps)
        writer.finalize()

        table = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        obj_cols = [c for c in table.column_names if c.startswith("observation.object_state.")]
        assert len(obj_cols) == OBJ_DIM

        # Count non-null values — should be n_steps / 2
        col = table.column(obj_cols[0])
        non_null_count = len(col.drop_null())
        expected = n_steps // (CONTROL_FREQ // SAVE_FREQ["object_state"])  # 100 // 2 = 50
        assert non_null_count == expected, (
            f"object_state should have {expected} samples, got {non_null_count}"
        )

    def test_camera_video_frame_count(self, dataset_dir):
        """cameras at 10 Hz with control_freq=100 -> 1/10th the frames."""
        n_steps = 100
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=n_steps)
        writer.finalize()

        # Check video frame count via parquet frame_index column
        table = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        fi_cols = [c for c in table.column_names if c.startswith("frame_index.")]
        assert len(fi_cols) >= 1, "Should have at least one frame_index column"

        col = table.column(fi_cols[0])
        non_null = col.drop_null()
        expected_frames = n_steps // (CONTROL_FREQ // SAVE_FREQ["cameras"])  # 100 // 10 = 10
        assert len(non_null) == expected_frames

        # Last frame index should be expected_frames - 1
        last_frame_idx = non_null[-1].as_py()
        assert last_frame_idx == expected_frames - 1

    @pytest.mark.parametrize("with_action", [True, False])
    def test_action_columns_follow_recorded_actions(self, dataset_dir, with_action):
        """action.* columns and the action feature exist exactly when actions are passed."""
        writer = _make_writer(dataset_dir)
        writer.begin_episode()
        for step in range(5):
            writer.add_step(timestamp=step / CONTROL_FREQ, obs=_fake_obs(step),
                            action=np.full(7, step, np.float32) if with_action else None)
        writer.end_episode()
        writer.finalize()

        table = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        action_cols = [c for c in table.column_names if c.startswith("action.")]
        features = json.loads((dataset_dir / "meta" / "info.json").read_text())["features"]
        if with_action:
            assert action_cols == [f"action.{i}" for i in range(7)]
            assert table.column("action.6").to_pylist() == [0.0, 1.0, 2.0, 3.0, 4.0]
            assert features["action"]["shape"] == [7]
        else:
            assert action_cols == [] and "action" not in features


class TestTimestamps:
    """Verify timestamp correctness."""

    def test_timestamps_monotonically_increasing(self, dataset_dir):
        n_steps = 50
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=n_steps)
        writer.finalize()

        table = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        timestamps = table.column("timestamp").to_pylist()
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_timestamp_spacing_matches_control_freq(self, dataset_dir):
        n_steps = 20
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=n_steps)
        writer.finalize()

        table = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        timestamps = table.column("timestamp").to_pylist()
        dt = 1.0 / CONTROL_FREQ
        for i in range(1, len(timestamps)):
            assert abs(timestamps[i] - timestamps[i - 1] - dt) < 1e-9


class TestGlobalIndex:
    """Verify the global frame index spans episodes correctly."""

    def test_global_index_continuous_across_episodes(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        _record_episode(writer, n_steps=30)
        writer.finalize()

        # Read both parquet files and collect all indices
        all_indices = []
        for pq_file in sorted((dataset_dir / "data" / "chunk-000").glob("*.parquet")):
            table = pq.read_table(str(pq_file))
            all_indices.extend(table.column("index").to_pylist())

        assert all_indices == list(range(50)), "Global index should be 0..49 across both episodes"

    def test_episode_index_correct(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=10)
        _record_episode(writer, n_steps=15)
        writer.finalize()

        t0 = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        t1 = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000001.parquet"))

        assert all(v.as_py() == 0 for v in t0.column("episode_index"))
        assert all(v.as_py() == 1 for v in t1.column("episode_index"))


class TestCumulativeRecording:
    """Verify that opening a writer on an existing dataset appends correctly."""

    def test_resume_appends_episodes(self, dataset_dir):
        # Session 1: record 2 episodes
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=20)
        _record_episode(w1, n_steps=20)
        w1.finalize()

        pq_files_1 = sorted((dataset_dir / "data" / "chunk-000").glob("*.parquet"))
        assert len(pq_files_1) == 2

        # Session 2: open the same directory, record 2 more
        w2 = _make_writer(dataset_dir)
        assert w2.episode_index == 2, "Should resume from episode 2"
        assert w2._global_index == 40, "Should resume from global index 40"
        _record_episode(w2, n_steps=15)
        _record_episode(w2, n_steps=15)
        w2.finalize()

        pq_files_2 = sorted((dataset_dir / "data" / "chunk-000").glob("*.parquet"))
        assert len(pq_files_2) == 4
        assert pq_files_2[2].name == "episode_000002.parquet"
        assert pq_files_2[3].name == "episode_000003.parquet"

    @pytest.mark.parametrize("change", ["drop_camera", "resize_camera", "extra_key", "action_dim"])
    def test_resume_rejects_a_different_schema(self, dataset_dir, change):
        """Appending episodes with other observations would leave a dataset no tool can read."""
        w1 = _make_writer(dataset_dir)
        w1.begin_episode()
        for step in range(10):
            w1.add_step(timestamp=step / CONTROL_FREQ, obs=_fake_obs(step), action=np.zeros(7))
        w1.end_episode()
        w1.finalize()

        obs, action = _fake_obs(0), np.zeros(7)
        if change == "drop_camera":
            del obs["rgb_front_cam"]
        elif change == "resize_camera":
            obs["rgb_front_cam"] = np.zeros((CAM_H * 2, CAM_W, 3), dtype=np.uint8)
        elif change == "extra_key":
            obs["new_sensor"] = np.zeros(2, dtype=np.float32)
        else:
            action = np.zeros(8)
        w2 = _make_writer(dataset_dir)
        w2.begin_episode()
        with pytest.raises(DatasetSchemaError):
            w2.add_step(timestamp=0.0, obs=obs, action=action)
        w2.abort_episode()
        w2.finalize()
        assert len(list((dataset_dir / "data").rglob("*.parquet"))) == 1
        assert len(list((dataset_dir / "videos").rglob("*.mp4"))) == 1

    @pytest.mark.parametrize("change", [{"control_freq": CONTROL_FREQ * 2},
                                        {"save_freq": {**SAVE_FREQ, "cameras": 25}}])
    def test_resume_rejects_other_rates(self, dataset_dir, change):
        """Episodes already saved keep their rates; info.json must not relabel them."""
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=10)
        w1.finalize()
        with pytest.raises(DatasetSchemaError, match="was recorded with"):
            _make_writer(dataset_dir, **change)
        _make_writer(dataset_dir).finalize()  # the lock was released

    def test_resume_keeps_the_dataset_task_config(self, dataset_dir):
        w1 = _make_writer(dataset_dir, task_config={"words": ["A"]})
        _record_episode(w1, n_steps=5)
        w1.finalize()
        w2 = _make_writer(dataset_dir, task_config={"words": ["B"]})
        _record_episode(w2, n_steps=5)
        w2.finalize()
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        entries = [json.loads(line) for line in (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines()]
        assert info["task_config"] == {"words": ["A"]}
        assert "task_config" not in entries[0]
        assert entries[1]["task_config"] == {"words": ["B"]}

    def test_resume_keeps_scalar_observations(self, dataset_dir):
        for _ in range(2):
            w = _make_writer(dataset_dir)
            w.begin_episode()
            for step in range(3):
                w.add_step(timestamp=step / CONTROL_FREQ, obs={**_fake_obs(step), "scalar": np.float32(step)})
            w.end_episode()
            w.finalize()

    def test_resume_accepts_the_same_schema(self, dataset_dir):
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=10)
        w1.finalize()
        w2 = _make_writer(dataset_dir)
        _record_episode(w2, n_steps=10)
        w2.finalize()
        assert len(list((dataset_dir / "data").rglob("*.parquet"))) == 2

    def test_resume_global_index_continues(self, dataset_dir):
        # Session 1
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=10)
        w1.finalize()

        # Session 2
        w2 = _make_writer(dataset_dir)
        _record_episode(w2, n_steps=10)
        w2.finalize()

        # Read episode 1 (from session 2)
        t1 = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000001.parquet"))
        indices = t1.column("index").to_pylist()
        assert indices[0] == 10, "Second session should start at global index 10"
        assert indices[-1] == 19

    def test_resume_does_not_corrupt_existing_parquet(self, dataset_dir):
        # Session 1
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=25)
        w1.finalize()

        # Read original data
        orig = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        orig_rows = len(orig)
        orig_ts = orig.column("timestamp").to_pylist()

        # Session 2
        w2 = _make_writer(dataset_dir)
        _record_episode(w2, n_steps=10)
        w2.finalize()

        # Original episode should be unchanged
        after = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000000.parquet"))
        assert len(after) == orig_rows
        assert after.column("timestamp").to_pylist() == orig_ts

    def test_resume_episodes_jsonl_appends(self, dataset_dir):
        # Session 1
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=10)
        w1.finalize()

        # Session 2
        w2 = _make_writer(dataset_dir)
        _record_episode(w2, n_steps=10)
        w2.finalize()

        lines = (dataset_dir / "meta" / "episodes.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        ep0 = json.loads(lines[0])
        ep1 = json.loads(lines[1])
        assert ep0["episode_index"] == 0
        assert ep1["episode_index"] == 1

    def test_resume_info_json_reflects_total(self, dataset_dir):
        # Session 1
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=10)
        w1.finalize()

        # Session 2
        w2 = _make_writer(dataset_dir)
        _record_episode(w2, n_steps=10)
        w2.finalize()

        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 2
        assert info["total_frames"] == 20


    def test_resume_without_new_episodes_keeps_features(self, dataset_dir):
        w1 = _make_writer(dataset_dir)
        _record_episode(w1, n_steps=20)
        w1.finalize()
        before = json.loads((dataset_dir / "meta" / "info.json").read_text())

        w2 = _make_writer(dataset_dir)
        w2.finalize()
        after = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert after["features"] == before["features"]
        assert after["features"]["observation.rgb_front_cam"]["shape"] == [CAM_H, CAM_W, 3]
        assert after["total_episodes"] == 1 and after["total_frames"] == 20

    def test_resume_keeps_action_feature(self, dataset_dir):
        w1 = _make_writer(dataset_dir)
        w1.begin_episode()
        w1.add_step(timestamp=0.0, obs=_fake_obs(0), action=np.zeros(14))
        w1.end_episode()
        w1.finalize()

        _make_writer(dataset_dir).finalize()
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert info["features"]["action"]["shape"] == [14]


class TestAbortAndPersistence:
    """Aborted episodes leave no trace; info.json survives a hard kill."""

    def test_abort_rolls_back_indices_and_videos(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        writer.begin_episode()
        for step in range(30):
            writer.add_step(timestamp=step / CONTROL_FREQ, obs=_fake_obs(step))
        partial = dataset_dir / "videos" / "chunk-000" / "rgb_front_cam" / "episode_000001.mp4"
        assert partial.exists()
        writer.abort_episode()
        assert not partial.exists()

        _record_episode(writer, n_steps=10)
        writer.finalize()
        t1 = pq.read_table(str(dataset_dir / "data" / "chunk-000" / "episode_000001.parquet"))
        assert t1.column("index").to_pylist() == list(range(20, 30))
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert info["total_frames"] == 30
        assert info["total_episodes"] == 2

    def test_info_json_written_after_each_episode(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        # No finalize(): simulates a process killed between episodes.
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 20
        assert "observation.rgb_front_cam" in info["features"]
        assert not list((dataset_dir / "meta").glob("*.tmp"))

    def test_kill_during_parquet_write_leaves_a_dataset_that_reopens(self, dataset_dir):
        """A SIGKILL or OOM kill mid-save leaves no partial episode and no stale lock."""
        child = _python(_KILLED_DURING_SAVE, dataset_dir, stderr=subprocess.PIPE, text=True)
        _, stderr = child.communicate(timeout=300)
        assert child.returncode == -signal.SIGKILL, stderr
        chunk = dataset_dir / "data" / "chunk-000"
        assert not (chunk / "episode_000001.parquet").exists()

        writer = _make_writer(dataset_dir)
        assert (writer.episode_index, writer._global_index) == (1, 20)
        _record_episode(writer, n_steps=10)
        writer.finalize()
        assert sorted(p.name for p in chunk.iterdir()) == ["episode_000000.parquet",
                                                           "episode_000001.parquet"]
        lines = (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines()
        assert [(json.loads(line)["episode_index"], json.loads(line)["length"])
                for line in lines] == [(0, 20), (1, 10)]
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert (info["total_episodes"], info["total_frames"]) == (2, 30)

    def test_unreadable_parquet_is_renamed_aside_on_resume(self, dataset_dir, caplog):
        """A parquet cut short by an older writer neither blocks resuming nor is deleted."""
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        writer.finalize()
        chunk = dataset_dir / "data" / "chunk-000"
        data = (chunk / "episode_000000.parquet").read_bytes()
        truncated = data[: len(data) // 2]
        (chunk / "episode_000001.parquet").write_bytes(truncated)

        with caplog.at_level(logging.WARNING):
            writer = _make_writer(dataset_dir)
        assert str(chunk / "episode_000001.parquet") in caplog.text
        assert (writer.episode_index, writer._global_index) == (1, 20)
        (aside,) = [p for p in chunk.iterdir() if p.name.endswith(".unreadable")]
        assert aside.name.startswith("episode_000001.parquet.")
        assert aside.read_bytes() == truncated

        _record_episode(writer, n_steps=10)
        writer.finalize()
        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert (info["total_episodes"], info["total_frames"]) == (2, 30)


class TestDatasetLock:
    """One writer per dataset at a time; the lock never outlives its holder."""

    def test_second_writer_is_refused_until_the_first_finalizes(self, dataset_dir):
        first = _make_writer(dataset_dir)
        with pytest.raises(DatasetLockedError, match="another collector is recording") as exc:
            _make_writer(dataset_dir)
        assert str(dataset_dir) in str(exc.value)
        _record_episode(first, n_steps=10)
        first.finalize()

        second = _make_writer(dataset_dir)
        assert second.episode_index == 1
        second.finalize()

    def test_lock_is_released_when_the_writer_is_garbage_collected(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=10)
        del writer
        assert _make_writer(dataset_dir).episode_index == 1

    def test_finalized_writer_cannot_record(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        writer.finalize()
        with pytest.raises(RuntimeError, match="finalize"):
            writer.begin_episode()
        writer.finalize()  # no-op

    def test_lock_spans_processes_and_dies_with_its_holder(self, dataset_dir):
        with _python(_HOLD_LOCK, dataset_dir, stdout=subprocess.PIPE, text=True) as holder:
            try:
                assert holder.stdout.readline() == "locked\n"
                with pytest.raises(DatasetLockedError):
                    _make_writer(dataset_dir)
            finally:
                holder.kill()
        _make_writer(dataset_dir).finalize()


class TestMetadata:
    """Verify metadata files are well-formed."""

    def test_info_json_has_required_fields(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        writer.finalize()

        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        assert info["codebase_version"] == "v2.1"
        assert info["robot_type"] == "Panda"
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 20
        assert info["fps"] == CONTROL_FREQ
        assert info["control_freq"] == CONTROL_FREQ
        assert info["task_id"] == TASK_ID
        assert "save_freq" in info
        assert "features" in info

    def test_info_json_features_have_fps(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=20)
        writer.finalize()

        info = json.loads((dataset_dir / "meta" / "info.json").read_text())
        features = info["features"]

        assert "action" not in features, "no action was recorded"
        # Individual proprio keys should have the proprio save freq
        for key in PROPRIO_KEYS:
            feat_key = f"observation.{key}"
            if feat_key in features:
                assert features[feat_key]["fps"] == SAVE_FREQ["proprio"]
        if "observation.object_state" in features:
            assert features["observation.object_state"]["fps"] == SAVE_FREQ["object_state"]

    def test_tasks_jsonl_content(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=10)
        writer.finalize()

        lines = (dataset_dir / "meta" / "tasks.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        task = json.loads(lines[0])
        assert task["task_index"] == 0
        assert task["task"] == TASK_ID

    def test_episodes_jsonl_has_episode_metadata(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        _record_episode(writer, n_steps=30)
        writer.finalize()

        lines = (dataset_dir / "meta" / "episodes.jsonl").read_text().strip().split("\n")
        ep = json.loads(lines[0])
        assert ep["episode_index"] == 0
        assert ep["task"] == TASK_ID
        assert ep["length"] == 30
        assert ep["final_score"] == 0.95
        assert ep["timestamp_start"] == 0.0
        assert ep["timestamp_end"] == pytest.approx(29.0 / CONTROL_FREQ)


class TestValidation:
    """Verify input validation."""

    def test_save_freq_must_divide_control_freq(self, dataset_dir):
        with pytest.raises(ValueError, match="does not evenly divide"):
            _make_writer(dataset_dir, save_freq={
                "proprio": 100,
                "force_torque": 100,
                "object_state": 30,  # 100 % 30 != 0
                "cameras": 10,
            })

    def test_add_step_without_begin_raises(self, dataset_dir):
        writer = _make_writer(dataset_dir)
        with pytest.raises(RuntimeError, match="begin_episode"):
            writer.add_step(timestamp=0.0, obs={})


class TestStreamCategory:
    """Verify _stream_category helper."""

    def test_rgb_camera(self):
        assert _stream_category("rgb_front_cam") == "cameras"

    def test_depth_camera(self):
        assert _stream_category("depth_front_cam") == "cameras"

    def test_individual_proprio_keys(self):
        assert _stream_category("robot0_eef_pos") == "proprio"
        assert _stream_category("robot0_joint_vel") == "proprio"

    def test_force_torque(self):
        assert _stream_category("force_torque") == "force_torque"
        assert _stream_category("force_torque_world") == "force_torque"

    def test_object_state(self):
        assert _stream_category("object_state") == "object_state"

    def test_object_pos_quat(self):
        assert _stream_category("object_pos") == "object_state"
        assert _stream_category("object_quat") == "object_state"
        assert _stream_category("cube_pos") == "object_state"
        assert _stream_category("cube_quat") == "object_state"

    def test_unknown_defaults_to_proprio(self):
        assert _stream_category("custom_sensor") == "proprio"
