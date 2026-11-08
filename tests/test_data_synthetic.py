"""Scripted-demo collectors (data_synthetic/): headless import, policy registry,
dataset bookkeeping, action clipping and seeded determinism; also the dataset
lock check of the teleop collector (scripts/collect_demos.py)."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from gymnasium import spaces

import imbench as ib
from data_synthetic import collection
from data_synthetic import bimanual_collect_demos, singlearm_collect_demos
from data_synthetic.episode_length_stats import record_run
from imbench.utils.lerobot_writer import LeRobotWriter
from imbench.utils.run_id import read_counter

REPO = Path(__file__).resolve().parents[1]
TASK = "ib-p3-domino-single-v1"
HORIZON = 6


# ── Stubs ──────────────────────────────────────────────────────────

class _StubEnv:
    """Minimal task: every episode lasts HORIZON steps and succeeds per *outcomes*."""

    MAX_EPISODE_STEPS = HORIZON

    def __init__(self, outcomes):
        self.cfg = {"control_freq": 20, "recording": {}, "obs": {}, "task_type": "single_stage"}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.save_extras = {"action": True, "reward": True, "task_score": True}
        self.outcomes = list(outcomes)
        self.seeds = []
        self.stepped = []
        self.make_kwargs = None

    def _obs(self):
        return {"robot0_eef_pos": np.full(3, float(self._t))}

    def reset(self, *, seed=None):
        self.seeds.append(seed)
        self._success = self.outcomes.pop(0)
        self._t = 0
        return self._obs(), {"task_score": 0.0, "is_success": False}

    def step(self, action):
        self.stepped.append(np.array(action))
        self._t += 1
        last = self._t >= HORIZON
        success = last and self._success
        info = {"task_score": 1.0 if success else 0.25, "is_success": success}
        return self._obs(), info["task_score"], success, last and not success, info

    def filter_obs_for_saving(self, obs):
        return {k: v for k, v in obs.items() if k.startswith("robot0_")}

    def align_camera_frames(self, first_recorded_step):
        pass

    def render(self):
        pass

    def close(self):
        pass


class _StubPolicy:
    """Draws out-of-range actions from ``self.rng``; optionally interrupts."""

    phase = "act"

    def __init__(self, interrupt_at=None):
        self.interrupt_at = interrupt_at
        self.rng = None

    def reset(self):
        self.t = 0

    def act(self, obs):
        self.t += 1
        if self.interrupt_at is not None and self.t == self.interrupt_at:
            raise KeyboardInterrupt
        return self.rng.uniform(-3.0, 3.0, size=7)


@pytest.fixture
def stub_world(tmp_path, monkeypatch):
    """Configured machine '07', no dashboard endpoint, ``ib.make`` returning stubs."""
    home = tmp_path / "home"
    (home / ".imbench").mkdir(parents=True)
    (home / ".imbench" / "machine.json").write_text(json.dumps({"machine_id": "07"}))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DASHBOARD_API_URL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    reports = []
    monkeypatch.setattr(collection, "report_episode", lambda *a, **kw: reports.append(kw))
    monkeypatch.setattr(collection, "extract_robot_state", lambda env: (np.zeros(2), np.zeros(2)))
    envs = []

    def make_env(outcomes):
        env = _StubEnv(outcomes)
        envs.append(env)

        def make(task_id, **kwargs):
            env.make_kwargs = kwargs
            return env
        monkeypatch.setattr(ib, "make", make)
        return env

    return {"out": tmp_path / "demos", "make_env": make_env, "reports": reports}


def _factory(interrupt_at=None):
    def make(env, obs, rng):
        return collection.build_policy(lambda: _StubPolicy(interrupt_at), env, rng)
    return make


def _dataset(out):
    return out / TASK / "M07"


def _metadata(ds):
    return {
        int(p.name.split("_")[1]): json.loads(p.read_text())
        for p in sorted((ds / "meta").glob("episode_*_metadata.json"))
    }


# ── Headless import / registry ─────────────────────────────────────

def test_collectors_import_headless_without_pynput():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    code = (
        "import sys\n"
        "import data_synthetic.singlearm_collect_demos as sa\n"
        "import data_synthetic.bimanual_collect_demos as bi\n"
        "import data_synthetic.hetbi_scripted.collect\n"
        "for t in sa._POLICY_REGISTRY: sa.resolve_policy(t)\n"
        "for t in bi._POLICY_REGISTRY: bi.resolve_policy(t)\n"
        "assert 'pynput' not in sys.modules, 'pynput imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, check=True, timeout=300)
    for script in ("singlearm_collect_demos.py", "bimanual_collect_demos.py"):
        proc = subprocess.run(
            [sys.executable, f"data_synthetic/{script}", "--help"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--n-episodes" in proc.stdout


@pytest.mark.parametrize("module", [singlearm_collect_demos, bimanual_collect_demos])
def test_ood_variants_have_no_scripted_policy(module):
    v2 = [t for t in ib.list_tasks() if t.endswith("-v2")]
    assert v2
    for task_id in v2:
        with pytest.raises(ValueError, match="out-of-distribution"):
            module.resolve_policy(task_id)


def test_registry_matches_exact_task_ids():
    assert set(singlearm_collect_demos._POLICY_REGISTRY) <= set(ib.list_tasks())
    assert set(bimanual_collect_demos._POLICY_REGISTRY) <= set(ib.list_tasks())
    with pytest.raises(ValueError, match="No scripted policy"):
        singlearm_collect_demos.resolve_policy("ib-p2-pendulum-grasp")
    with pytest.raises(ValueError, match="No scripted policy"):
        singlearm_collect_demos.resolve_policy("ib-m-keyboard-typing-v1")


@pytest.mark.parametrize("flag,value", [("--seed", "-1"), ("--n-episodes", "0"), ("--max-steps", "0")])
def test_cli_rejects_out_of_range_values(stub_world, monkeypatch, capsys, flag, value):
    stub_world["make_env"]([True])
    monkeypatch.setattr(sys, "argv", ["collector", flag, value, "--out-dir", str(stub_world["out"])])
    with pytest.raises(SystemExit) as exc:
        collection.run_task_cli(TASK, _factory(), description="test", log_prefix="test")
    assert exc.value.code == 2 and "must be >=" in capsys.readouterr().err
    assert not stub_world["out"].exists()


def test_length_stats_cover_the_whole_dataset(stub_world):
    out = stub_world["out"]
    stub_world["make_env"]([True, True])
    collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=2, out_dir=str(out), seed=0)
    stub_world["make_env"]([True])
    collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1, out_dir=str(out))
    stats = json.loads((_dataset(out) / "episode_length_stats.json").read_text())[TASK]
    assert stats["n_episodes"] == 3


def test_scripted_and_teleop_collectors_name_tasks_alike():
    """Dashboards group episodes by task_name, so every collector must write the same one."""
    assert collection.task_display_name("ib-p3-domino-select-v1") == "Domino Chain Selection"
    assert collection.task_display_name("ib-m-keyboard-typing-v2") == "Keyboard Typing (Bimanual) (v2)"
    assert collection.task_display_name("ib-p9-unlisted-thing-v1") == "Unlisted Thing"


# ── Dataset bookkeeping ────────────────────────────────────────────

def test_saved_episodes_use_writer_index_after_discarded_attempts(stub_world):
    out = stub_world["out"]
    env = stub_world["make_env"]([False, True, False, True])
    result = collection.collect(
        task_id=TASK, policy_factory=_factory(), n_episodes=2,
        out_dir=str(out), seed=100, save_npy=True,
    )
    assert (result.n_saved, result.attempts, result.exit_code) == (2, 4, 0)
    assert env.seeds == [100, 101, 102, 103]

    ds = _dataset(out)
    assert sorted(p.name for p in ds.rglob("*.parquet")) == ["episode_000000.parquet",
                                                              "episode_000001.parquet"]
    assert sorted(p.name for p in (ds / "npy").iterdir()) == ["traj_000000.npy", "traj_000001.npy"]
    meta = _metadata(ds)
    assert sorted(meta) == [0, 1]
    assert [meta[i]["seed"] for i in (0, 1)] == [101, 103]
    assert [meta[i]["dataset_summary"]["episode"] for i in (0, 1)] == [0, 1]
    episodes = [json.loads(line) for line in (ds / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [e["episode_index"] for e in episodes] == [0, 1]
    assert [e["run_id"] for e in episodes] == [meta[0]["run_id"], meta[1]["run_id"]]
    assert [e["seed"] for e in episodes] == [101, 103]
    assert [r["episode_id"] for r in stub_world["reports"]] == [meta[0]["run_id"], meta[1]["run_id"]]

    # Resuming continues after the largest recorded seed and appends at index 2.
    env2 = stub_world["make_env"]([True])
    result = collection.collect(
        task_id=TASK, policy_factory=_factory(), n_episodes=1, out_dir=str(out), seed=100,
    )
    assert result.n_saved == 1
    assert env2.seeds == [104]
    meta = _metadata(ds)
    assert sorted(meta) == [0, 1, 2]
    assert meta[2]["seed"] == 104
    assert len({m["run_id"] for m in meta.values()}) == 3


def test_scripted_metadata_has_real_task_and_no_operator(stub_world):
    out = stub_world["out"]
    stub_world["make_env"]([True])
    collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1, out_dir=str(out), seed=0)
    meta = _metadata(_dataset(out))[0]
    assert meta["task_id"] == TASK
    assert meta["outcome"] == "success"
    for key in ("teleoperator_id", "safety_watcher_id", "supervisor_id", "robot_id", "sim_id"):
        assert meta[key] is None
    assert meta["seed"] == 0 and meta["warmup_steps"] == 0
    assert meta["dataset_summary"]["policy"] == "_StubPolicy"


def test_collection_records_effective_config_for_episode_replay(stub_world):
    env = stub_world["make_env"]([True])
    overrides = {"word": None, "words": ["JOINT", "MATRIX"]}
    env.cfg.update(overrides)  # the stub stands in for config loading in ib.make
    env.cfg["max_episode_steps"] = 10
    collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1,
                       out_dir=str(stub_world["out"]), config_overrides=overrides, max_steps=10)
    assert env.make_kwargs["config_overrides"] == {**overrides, "max_episode_steps": 10}
    assert "max_episode_steps" not in overrides  # the caller's config is not mutated
    ds = _dataset(stub_world["out"])
    entry = json.loads((ds / "meta" / "episodes.jsonl").read_text().splitlines()[0])
    assert entry["task_config"]["words"] == ["JOINT", "MATRIX"]
    assert _metadata(ds)[0]["dataset_summary"]["task_config"] == entry["task_config"]


@pytest.mark.parametrize("custom", [False, True])
def test_keyboard_collector_passes_word_list_to_collection(tmp_path, monkeypatch, custom):
    calls = []
    def fake_collect(**kwargs):
        calls.append(kwargs)
        return collection.CollectionResult(n_requested=1, n_saved=1)
    monkeypatch.setattr(bimanual_collect_demos, "collect", fake_collect)
    argv = ["collector", "--n-episodes", "1"]
    if custom:
        words_file = tmp_path / "words.txt"
        words_file.write_text("joint\n\nMATRIX\n")
        argv += ["--words-file", str(words_file)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        bimanual_collect_demos.main()
    assert exc.value.code == 0
    overrides = calls[0]["config_overrides"]
    assert overrides["word"] is None
    assert overrides["words"] == (["JOINT", "MATRIX"] if custom else bimanual_collect_demos.load_words())
    assert len(overrides["words"]) == (2 if custom else 25)


def test_actions_are_clipped_before_stepping_and_recording(stub_world):
    out = stub_world["out"]
    env = stub_world["make_env"]([True])
    collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1, out_dir=str(out),
                       seed=5, save_npy=True)
    stepped = np.stack(env.stepped)
    assert stepped.dtype == np.float32
    assert stepped.min() >= -1.0 and stepped.max() <= 1.0
    assert np.isclose(np.abs(stepped), 1.0).any(), "stub policy should have exceeded the bounds"

    ds = _dataset(out)
    np.testing.assert_array_equal(np.load(ds / "npy" / "traj_000000.npy"), stepped)
    table = pq.read_table(next(ds.rglob("episode_000000.parquet")))
    recorded = np.stack([table.column(f"action.{i}").to_numpy() for i in range(7)], axis=1)
    np.testing.assert_array_equal(recorded, stepped)

    expected = np.clip(np.random.default_rng(5).uniform(-3.0, 3.0, size=(HORIZON, 7)), -1, 1)
    np.testing.assert_array_equal(stepped, expected.astype(np.float32))


def test_interrupt_discards_episode_in_progress(stub_world):
    out = stub_world["out"]
    stub_world["make_env"]([True, True])
    calls = {"n": 0}

    def factory(env, obs, rng):
        calls["n"] += 1
        return collection.build_policy(
            lambda: _StubPolicy(interrupt_at=3 if calls["n"] == 2 else None), env, rng,
        )

    result = collection.collect(task_id=TASK, policy_factory=factory, n_episodes=2, out_dir=str(out), seed=0)
    assert result.interrupted and result.exit_code == 130 and result.n_saved == 1
    ds = _dataset(out)
    assert [p.name for p in ds.rglob("*.parquet")] == ["episode_000000.parquet"]
    assert sorted(_metadata(ds)) == [0]
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_stop_signal_during_save_completes_the_episode(stub_world, monkeypatch, signum):
    out = stub_world["out"]
    stub_world["make_env"]([True, True])
    write_metadata = collection.write_episode_metadata

    def signal_then_write(*args, **kwargs):
        signal.raise_signal(signum)
        write_metadata(*args, **kwargs)

    monkeypatch.setattr(collection, "write_episode_metadata", signal_then_write)
    escaped = []
    previous = signal.signal(signum, lambda *_: escaped.append(signum))
    try:
        result = collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=2,
                                    out_dir=str(out), seed=0)
    finally:
        signal.signal(signum, previous)
    assert not escaped, "signal reached the default handler during the save"
    assert result.interrupted and result.exit_code == 130 and result.n_saved == 1
    ds = _dataset(out)
    assert sorted(_metadata(ds)) == [0]
    assert len((ds / "meta" / "episodes.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("save_phase", [False, True], ids=["episode", "save"])
def test_ignored_hangup_stays_ignored(stub_world, monkeypatch, save_phase):
    """A run started under nohup (SIGHUP ignored) keeps collecting when the terminal closes."""
    out = stub_world["out"]
    stub_world["make_env"]([True, True])
    if save_phase:
        write_metadata = collection.write_episode_metadata

        def hangup_then_write(*args, **kwargs):
            signal.raise_signal(signal.SIGHUP)
            write_metadata(*args, **kwargs)

        monkeypatch.setattr(collection, "write_episode_metadata", hangup_then_write)
    else:
        def factory(env, obs, rng):
            signal.raise_signal(signal.SIGHUP)
            return _factory()(env, obs, rng)

    previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        result = collection.collect(task_id=TASK, policy_factory=_factory() if save_phase else factory,
                                    n_episodes=2, out_dir=str(out), seed=0)
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, previous)
    assert not result.interrupted and result.n_saved == 2


def test_locked_dataset_exits_before_any_attempt(stub_world, capsys):
    out = stub_world["out"]
    env = stub_world["make_env"]([True])
    other = LeRobotWriter(_dataset(out), TASK, 20, {}, {})  # another collector recording here
    try:
        with pytest.raises(SystemExit) as exc:
            collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1,
                               out_dir=str(out), seed=0)
    finally:
        other.finalize()
    assert exc.value.code == 1
    assert f"{_dataset(out)} is in use: another collector is recording into it" in capsys.readouterr().out
    assert env.seeds == []
    assert read_counter() is None, "a run ID was issued"


def test_resume_with_a_different_schema_exits_cleanly(stub_world, capsys):
    out = stub_world["out"]
    stub_world["make_env"]([True])
    collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1, out_dir=str(out), seed=0)
    env = stub_world["make_env"]([True])
    env._obs = lambda: {"robot0_eef_pos": np.zeros(3), "robot0_gripper_qpos": np.zeros(2)}
    with pytest.raises(SystemExit) as exc:
        collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1,
                           out_dir=str(out), seed=0)
    assert exc.value.code == 1
    printed = capsys.readouterr().out
    assert "different observation schema" in printed and "budget exhausted" not in printed
    assert len(list(_dataset(out).rglob("*.parquet"))) == 1


def test_resume_seed_is_read_after_the_dataset_is_locked(stub_world, monkeypatch):
    out = stub_world["out"]
    env = stub_world["make_env"]([True])
    make_writer = collection.make_writer

    def make_writer_after_another_save(root, *args):
        # Another collector saved seed 41 and released the dataset while this one started.
        meta = Path(root, "meta")
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "episode_000000_metadata.json").write_text(json.dumps({"seed": 41}))
        return make_writer(root, *args)

    monkeypatch.setattr(collection, "make_writer", make_writer_after_another_save)
    result = collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1,
                                out_dir=str(out), seed=0)
    assert result.n_saved == 1
    assert env.seeds == [42]


def test_teleop_collector_exits_on_a_locked_dataset_before_starting(stub_world, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    collect_demos = importlib.import_module("collect_demos")

    def make(*args, **kwargs):
        raise AssertionError("simulator started for a locked dataset")

    monkeypatch.setattr(ib, "make", make)
    monkeypatch.setattr(collect_demos, "get_teleop_class", lambda name: object)
    out = stub_world["out"] / TASK
    other = LeRobotWriter(out / "M07", TASK, 20, {}, {})  # another collector recording here
    try:
        with pytest.raises(SystemExit) as exc:
            collect_demos.collect(TASK, "keyboard", n_demos=1, out_dir=str(out),
                                  render=False, use_dashboard=False)
    finally:
        other.finalize()
    assert exc.value.code == 1
    assert "another collector is recording into it" in capsys.readouterr().out
    assert read_counter() is None, "a run ID was issued"


def test_exhausted_attempt_budget_fails_and_saves_nothing(stub_world):
    out = stub_world["out"]
    stub_world["make_env"]([False, False, False])
    result = collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1,
                                out_dir=str(out), seed=0, max_attempts=3)
    assert (result.n_saved, result.attempts, result.exit_code) == (0, 3, 1)
    ds = _dataset(out)
    assert not list(ds.rglob("*.parquet")) and not _metadata(ds)
    assert stub_world["reports"] == []


def test_render_request_without_display_runs_headless(stub_world, monkeypatch):
    def no_dashboard(*a, **kw):
        raise AssertionError("dashboard opened without a display")
    monkeypatch.setattr(collection, "_Dashboard", no_dashboard)
    env = stub_world["make_env"]([True])
    result = collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1,
                                out_dir=str(stub_world["out"]), seed=0, render=True)
    assert result.n_saved == 1
    assert env.make_kwargs["render_mode"] is None


@pytest.mark.parametrize("no_dashboard", [False, True])
def test_render_opens_dashboard_unless_disabled(stub_world, monkeypatch, no_dashboard):
    opened = []

    class Dashboard:
        def __init__(self, *args, **kwargs):
            opened.append(args)

        def post(self, obs, info, reward):
            pass

        def reset(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(collection, "_Dashboard", Dashboard)
    monkeypatch.setattr(collection, "display_available", lambda: True)
    env = stub_world["make_env"]([True])
    argv = ["collector", "--render", "--n-episodes", "1", "--seed", "0",
            "--out-dir", str(stub_world["out"])]
    monkeypatch.setattr(sys, "argv", argv + (["--no-dashboard"] if no_dashboard else []))
    with pytest.raises(SystemExit) as exc:
        collection.run_task_cli(TASK, _factory(), description="test", log_prefix="test")
    assert exc.value.code == 0
    assert env.make_kwargs["render_mode"] == "human"
    assert len(opened) == (0 if no_dashboard else 1)


def test_hetbi_collector_accepts_no_dashboard(monkeypatch):
    module = importlib.import_module("data_synthetic.hetbi_scripted.collect")
    calls = []

    def fake_collect(**kwargs):
        calls.append(kwargs)
        return collection.CollectionResult(n_requested=kwargs["n_episodes"],
                                           n_saved=kwargs["n_episodes"])

    monkeypatch.setattr(module, "collect", fake_collect)
    task = module.ALL_HETBI_TASKS[0]
    monkeypatch.setattr(sys, "argv", ["collector", "--task", task, "--render", "--no-dashboard"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    assert [(c["render"], c["use_dashboard"]) for c in calls] == [(True, False)]


def test_no_lerobot_writes_nothing(stub_world, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stub_world["make_env"]([True])
    result = collection.collect(task_id=TASK, policy_factory=_factory(), n_episodes=1,
                                seed=0, save_lerobot=False)
    assert result.n_saved == 1
    assert not (tmp_path / collection.DEFAULT_OUT_ROOT).exists()


def test_interrupted_length_log_update_keeps_previous_log(tmp_path, monkeypatch):
    record_run(str(tmp_path), "task-b", "B", [4, 6])
    record_run(str(tmp_path), "task-a", "A", [10])
    log_path = tmp_path / "episode_length_stats.json"
    saved = log_path.read_text()
    assert list(json.loads(saved)) == ["task-a", "task-b"]

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(json, "dump", interrupt)
    with pytest.raises(KeyboardInterrupt):
        record_run(str(tmp_path), "task-c", "C", [7])
    assert log_path.read_text() == saved
    assert [p.name for p in tmp_path.iterdir()] == ["episode_length_stats.json"]


# ── Determinism ────────────────────────────────────────────────────

def _rollout(env, policy_cls, env_seed, policy_seed, n_steps=150):
    obs, _ = env.reset(seed=env_seed)
    policy = collection.build_policy(policy_cls, env, np.random.default_rng(policy_seed))
    act_kwargs = {"env": env} if "env" in inspect.signature(policy.act).parameters else {}
    actions = []
    for _ in range(n_steps):
        action = collection.clip_action(policy.act(obs, **act_kwargs), env.action_space)
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.stack(actions)


def test_policy_actions_are_reproducible_from_seed():
    policy_cls = singlearm_collect_demos.resolve_policy("ib-p3-domino-select-v1")
    env = ib.make("ib-p3-domino-select-v1", obs_mode="proprio")
    try:
        first = _rollout(env, policy_cls, env_seed=3, policy_seed=3)
        again = _rollout(env, policy_cls, env_seed=3, policy_seed=3)
        other_policy_seed = _rollout(env, policy_cls, env_seed=3, policy_seed=4)
    finally:
        env.close()
    np.testing.assert_array_equal(first, again)
    assert first.shape != other_policy_seed.shape or not np.array_equal(first, other_policy_seed)

