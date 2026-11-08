"""collect_demos.py and generate_dataset.py: real-sim smoke tests with a scripted
teleop or a fixed trajectory, plus config helpers."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

import imbench
from imbench.teleop import TeleopInterface, _TELEOP_REGISTRY

REPO = Path(__file__).resolve().parents[1]
TASK = "ib-p2-cube-toss-v1"
N_ACTIVE = 10


def _script(name):
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


def _collect_demos():
    return _script("collect_demos")


class _ScriptedTeleop(TeleopInterface):
    """Random actions; discards the first attempt after a few steps."""

    def __init__(self, num_arms=1):
        self._rng = np.random.RandomState(0)
        self._num_arms = num_arms
        self._tick = 0
        self._episode = 0
        self._end = False

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self._tick, self._end = 0, False
        self._episode += 1

    def get_action(self, obs):
        self._tick += 1
        if self._episode == 1 and self._tick == 5:
            self._end = True
        action = self._rng.uniform(-0.3, 0.3, 7 * self._num_arms)
        action[6::7] = 1.0 if self._tick > 4 else -1.0
        return {"action": action, "paused": self._tick == 2}

    def should_end_episode(self):
        return self._end


def test_action_filter_config_resolved_from_repo(tmp_path, monkeypatch):
    cd = _collect_demos()
    monkeypatch.chdir(tmp_path)
    expected = yaml.safe_load((REPO / "configs" / "teleop" / "gamepad.yaml").read_text())
    assert cd._load_action_filter_config(None) == expected["action_filter"]
    with pytest.raises(FileNotFoundError):
        cd._load_action_filter_config("missing.yaml")


def test_task_display_name_fallbacks():
    cd = _collect_demos()
    assert cd.task_display_name("ib-p2-cube-toss-v1") == "Cube Toss"
    assert cd.task_display_name("ib-p1-edge-slide-v2") == "Edge Slide and Grasp (v2)"
    assert cd.task_display_name("ib-p9-brand-new-task-v1") == "Brand New Task"
    for task_id in imbench.list_tasks():
        assert cd.task_display_name(task_id)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """HOME with machine ID '07' and a dashboard session for another task."""
    path = tmp_path / "home"
    (path / ".imbench").mkdir(parents=True)
    (path / ".imbench" / "machine.json").write_text(json.dumps({"machine_id": "07"}))
    (path / ".imbench" / "session.json").write_text(json.dumps(
        {"task_id": "ib-p1-edge-slide-v1", "teleoperator_id": "OP-003", "supervisor_id": "SV-001"}))
    monkeypatch.setenv("HOME", str(path))
    return path


def _patch_make(monkeypatch, cd, success_on_termination=False):
    """Make ``cd.ib.make`` return a proprio env that terminates after N_ACTIVE operator steps.

    With *success_on_termination*, ``is_success`` holds on the terminating
    step only and lapses during the cooldown.
    """
    real_make = imbench.make

    def make_proprio(task_id, **kwargs):
        kwargs["obs_mode"] = "proprio"
        env = real_make(task_id, **kwargs)
        step, reset = env.step, env.reset
        count = {"n": 0}

        def counted_reset(**kw):
            count["n"] = -cd.WARMUP_STEPS
            return reset(**kw)

        def counted_step(action):
            obs, reward, terminated, truncated, info = step(action)
            count["n"] += 1
            if success_on_termination:
                info = dict(info, is_success=count["n"] == N_ACTIVE)
            return obs, reward, terminated or count["n"] >= N_ACTIVE, truncated, info

        env.reset, env.step = counted_reset, counted_step
        return env

    monkeypatch.setattr(cd.ib, "make", make_proprio)
    return real_make


def test_collect_smoke_discard_and_replay(home, tmp_path, monkeypatch):
    cd = _collect_demos()
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    sent = []
    monkeypatch.setattr(cd, "report_episode", lambda **kw: sent.append(kw))
    real_make = _patch_make(monkeypatch, cd)

    out = tmp_path / "demos"
    cd.collect(TASK, "scripted", n_demos=1, out_dir=str(out), render=False, use_dashboard=False)

    root = out / "M07"
    (entry,) = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    expected_len = N_ACTIVE + cd.COOLDOWN_STEPS
    assert entry["episode_index"] == 0
    assert entry["length"] == expected_len
    assert entry["seed"] == 7_000_000 and entry["warmup_steps"] == cd.WARMUP_STEPS
    assert entry["run_id"].endswith("07000001")

    # The discarded attempt left no trace: indices start at 0, no extra files.
    table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
    assert table.column("index").to_pylist() == list(range(expected_len))
    assert sorted(p.name for p in (root / "data" / "chunk-000").iterdir()) == ["episode_000000.parquet"]

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1 and info["total_frames"] == expected_len
    assert info["features"]["action"]["shape"] == [7]
    meta = json.loads((root / "meta" / "episode_000000_metadata.json").read_text())
    assert meta["run_id"] == entry["run_id"] and meta["seed"] == entry["seed"]
    # The dashboard session supplies the crew; the task is the one recorded.
    assert meta["task_id"] == TASK
    assert (meta["teleoperator_id"], meta["supervisor_id"]) == ("OP-003", "SV-001")
    assert len(sent) == 1 and sent[0]["episode_id"] == entry["run_id"]

    # Re-simulating with the recorded seed / warm-up reproduces the recording.
    actions = np.stack([table.column(f"action.{i}").to_numpy() for i in range(7)], axis=1)
    eef = np.stack([table.column(f"observation.robot0_eef_pos.{i}").to_numpy() for i in range(3)], axis=1)
    env = real_make(TASK, obs_mode="proprio", config_overrides={"max_episode_steps": 999_999})
    try:
        obs, _ = env.reset(seed=entry["seed"])
        for _ in range(entry["warmup_steps"]):
            obs, *_ = env.step(np.zeros(7))
        for t, action in enumerate(actions):
            np.testing.assert_allclose(obs["robot0_eef_pos"], eef[t], atol=1e-5)
            obs, _, _, _, step_info = env.step(action)
            if t == N_ACTIVE - 1:
                terminal_score = step_info["task_score"]
    finally:
        env.close()
    assert terminal_score == pytest.approx(entry["final_score"], abs=1e-6)


def test_outcome_is_taken_at_termination(home, tmp_path, monkeypatch):
    cd = _collect_demos()
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    sent = []
    monkeypatch.setattr(cd, "report_episode", lambda **kw: sent.append(kw))
    _patch_make(monkeypatch, cd, success_on_termination=True)

    out = tmp_path / "demos"
    cd.collect(TASK, "scripted", n_demos=1, out_dir=str(out), render=False, use_dashboard=False)

    meta = json.loads((out / "M07" / "meta" / "episode_000000_metadata.json").read_text())
    assert (meta["outcome"], meta["failure_modes"]) == ("success", None)
    assert [kw["outcome"] for kw in sent] == ["success"]


def test_generated_episodes_ignore_the_teleop_session(home, tmp_path, monkeypatch):
    gd = _script("generate_dataset")
    monkeypatch.setattr(gd, "report_episode", lambda **kw: None)
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    np.save(traj_dir / "traj_000.npy", np.zeros((5, 7), np.float32))

    gd.generate_episodes(TASK, [str(traj_dir / "traj_000.npy")], str(tmp_path / "out"), "07",
                         obs_mode="proprio")

    meta = json.loads((tmp_path / "out" / "M07" / "meta" / "episode_000000_metadata.json").read_text())
    assert meta["task_id"] == TASK
    assert meta["teleoperator_id"] is None and meta["supervisor_id"] is None
    assert meta["dataset_summary"]["teleop_device"] == gd.DEVICE_NAME
