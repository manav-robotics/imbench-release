"""Command-line scripts under scripts/: arguments, success counting, seeding,
quitting from the dashboard and headless behaviour.  Stand-in envs replace the
simulator except where a test says otherwise."""
from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

import imbench as ib
import imbench.utils.obs_visualizer as obs_visualizer
from data_synthetic.single_arm_pipeline import occluder_push_policy, pendulum_grasp_policy

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TASK = "ib-p2-cube-toss-v1"


class _FakeEnv:
    """Replays ``episodes[i]``, a list of ``(task_score, is_success)`` steps; truncates after the last."""

    def __init__(self, episodes, action_dim=7):
        self.episodes = episodes
        self.action_space = spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
        self.save_extras = {}
        self.reset_seeds = []
        self.actions = []
        self.closed = False

    def _obs(self):
        return {
            "agentview_image": np.full((4, 4, 3), 128, np.uint8),
            "object_pos": np.array([0.01 * len(self.reset_seeds), 0.0, 0.8]),
        }

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self._steps = self.episodes[(len(self.reset_seeds) - 1) % len(self.episodes)]
        self._t = 0
        return self._obs(), {"task_score": 0.0, "is_success": False, "step": 0}

    def step(self, action):
        self.actions.append(np.asarray(action))
        score, success = self._steps[self._t]
        self._t += 1
        info = {"task_score": score, "is_success": success, "step": self._t}
        return self._obs(), score, False, self._t == len(self._steps), info

    def render(self):
        return None

    def close(self):
        self.closed = True


def _no_make(task_id, **kwargs):
    raise AssertionError(f"a task was built: {task_id}")


def _patch_make(monkeypatch, env):
    """Make ``ib.make`` return *env*; returns the list of ``(task_id, kwargs)`` calls."""
    calls = []

    def make(task_id, **kwargs):
        calls.append((task_id, kwargs))
        return env

    monkeypatch.setattr(ib, "make", make)
    return calls


def _run_script(monkeypatch, path, *argv):
    monkeypatch.setattr(sys, "argv", [str(path), *argv])
    runpy.run_path(str(path), run_name="__main__")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Dashboard:
    """ObsDashboard stand-in; returns ``q`` on update number ``quit_on`` (never by default)."""

    quit_on = None

    def __init__(self, *args, **kwargs):
        self.updates = 0

    def reset(self):
        pass

    def update(self, obs, info=None, reward=None):
        self.updates += 1
        return ord("q") if self.updates == self.quit_on else -1

    def close(self):
        pass


class _QuitDashboard(_Dashboard):
    quit_on = 3  # the reset frame, then two steps


@pytest.fixture(autouse=True)
def _no_windows(monkeypatch):
    monkeypatch.setattr(obs_visualizer, "ObsDashboard", _Dashboard)


@pytest.fixture
def display(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")


@pytest.fixture
def no_display(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)


# ── run_random_policy.py ────────────────────────────────────────────


class TestRunRandomPolicy:
    SCRIPT = SCRIPTS / "run_random_policy.py"

    def test_help_exits_without_building_a_task(self, monkeypatch, capsys):
        monkeypatch.setattr(ib, "make", _no_make)
        with pytest.raises(SystemExit) as exc:
            _run_script(monkeypatch, self.SCRIPT, "--help")
        assert exc.value.code == 0
        assert "--task" in capsys.readouterr().out

    def test_counts_is_success_at_the_task_budget(self, monkeypatch, capsys):
        # Success mid-episode with a low final score; a high score without success.
        env = _FakeEnv([[(0.2, False), (0.5, True), (0.3, False)], [(0.99, False), (0.99, False)]])
        calls = _patch_make(monkeypatch, env)
        _run_script(monkeypatch, self.SCRIPT, "--task", TASK, "--episodes", "2", "--seed", "5",
                    "--no-render")
        assert [task_id for task_id, _ in calls] == [TASK]
        assert calls[0][1]["max_episode_steps"] is None
        assert env.reset_seeds == [5, 6]
        assert f"{TASK}: success in 1/2 random episodes" in capsys.readouterr().out

    def test_runs_are_reproducible(self, monkeypatch):
        actions = []
        for _ in range(2):
            env = _FakeEnv([[(0.0, False)] * 3])
            _patch_make(monkeypatch, env)
            _run_script(monkeypatch, self.SCRIPT, "--task", TASK, "--episodes", "2", "--no-render")
            actions.append(np.array(env.actions))
        np.testing.assert_array_equal(actions[0], actions[1])

    def test_without_display_asks_for_no_render(self, monkeypatch, no_display, capsys):
        monkeypatch.setattr(ib, "make", _no_make)
        with pytest.raises(SystemExit) as exc:
            _run_script(monkeypatch, self.SCRIPT, "--task", TASK)
        assert exc.value.code == 2
        assert "--no-render" in capsys.readouterr().err

    def test_q_stops_and_is_not_counted(self, monkeypatch, display, capsys):
        monkeypatch.setattr(obs_visualizer, "ObsDashboard", _QuitDashboard)
        env = _FakeEnv([[(0.0, False)] * 5])
        _patch_make(monkeypatch, env)
        _run_script(monkeypatch, self.SCRIPT, "--task", TASK, "--episodes", "3")
        assert env.reset_seeds == [0]
        assert "success in 0/0 random episodes" in capsys.readouterr().out


# ── run_occluder_push_policy.py / run_pendulum_grasp_policy.py ─────


class _FakeOccluderPolicy:
    built: list = []

    def __init__(self, verbose=True, rng=None):
        self.rng = rng
        self.phase = "push"
        self.built.append(self)

    def reset(self):
        pass

    def act(self, obs, env=None):
        return np.zeros(7)


class _FakePendulumPolicy:
    def __init__(self, seed=None, rng=None):
        self.phase = "grasp"
        self._side = 1
        self._close_latched = False

    def reset(self):
        pass

    def act(self, obs):
        return np.zeros(7)


# Final score 0.995 without success, then success at a lower final score.
_ORACLE_EPISODES = [[(0.995, False)], [(0.2, True), (0.995, False)]]


class TestRunOccluderPushPolicy:
    SCRIPT = SCRIPTS / "run_occluder_push_policy.py"

    @pytest.fixture(autouse=True)
    def policy(self, monkeypatch):
        monkeypatch.setattr(_FakeOccluderPolicy, "built", [])
        monkeypatch.setattr(occluder_push_policy, "OccluderPushPolicy", _FakeOccluderPolicy)

    def test_budget_seeding_and_success(self, monkeypatch, capsys):
        env = _FakeEnv(_ORACLE_EPISODES)
        calls = _patch_make(monkeypatch, env)
        _run_script(monkeypatch, self.SCRIPT, "--episodes", "2", "--seed", "3", "--no-render", "--quiet")
        assert calls[0][1]["max_episode_steps"] is None
        # Episode i seeds the policy's RNG with seed + i, as the collectors do.
        assert [p.rng.random() for p in _FakeOccluderPolicy.built] == [
            np.random.default_rng(3).random(), np.random.default_rng(4).random()]
        assert "Success: 50%  (1/2)" in capsys.readouterr().out

    def test_q_quits(self, monkeypatch, display):
        monkeypatch.setattr(obs_visualizer, "ObsDashboard", _QuitDashboard)
        env = _FakeEnv([[(0.0, False)] * 5])
        _patch_make(monkeypatch, env)
        _run_script(monkeypatch, self.SCRIPT, "--episodes", "3", "--quiet")
        assert env.reset_seeds == [0]

    def test_without_display_asks_for_no_render(self, monkeypatch, no_display):
        monkeypatch.setattr(ib, "make", _no_make)
        with pytest.raises(SystemExit) as exc:
            _run_script(monkeypatch, self.SCRIPT)
        assert exc.value.code == 2


class TestRunPendulumGraspPolicy:
    SCRIPT = SCRIPTS / "run_pendulum_grasp_policy.py"

    @pytest.fixture(autouse=True)
    def policy(self, monkeypatch):
        monkeypatch.setattr(pendulum_grasp_policy, "PendulumGraspPolicy", _FakePendulumPolicy)

    def test_headless_budget_and_success(self, monkeypatch, capsys):
        env = _FakeEnv(_ORACLE_EPISODES)
        calls = _patch_make(monkeypatch, env)
        _run_script(monkeypatch, self.SCRIPT, "--episodes", "2", "--no-render")
        assert calls[0][1]["max_episode_steps"] is None
        assert calls[0][1]["render_mode"] is None
        assert "Success: 50%" in capsys.readouterr().out

    def test_q_quits(self, monkeypatch, display):
        monkeypatch.setattr(obs_visualizer, "ObsDashboard", _QuitDashboard)
        env = _FakeEnv([[(0.0, False)] * 5])
        _patch_make(monkeypatch, env)
        _run_script(monkeypatch, self.SCRIPT, "--episodes", "3")
        assert env.reset_seeds == [0]


# ── scripts/tests/ helpers ──────────────────────────────────────────


class _NeverSucceeds:
    built: list = []

    def __init__(self, rng=None):
        self.rng = rng
        self.built.append(self)

    def reset(self):
        pass

    def act(self, obs):
        return np.zeros(7)


@pytest.fixture(scope="module")
def vp():
    return _load("verify_pipelines", SCRIPTS / "tests" / "verify_pipelines.py")


class TestVerifyPipelines:

    @pytest.fixture(autouse=True)
    def built(self, monkeypatch):
        monkeypatch.setattr(_NeverSucceeds, "built", [])

    def test_fails_when_no_episode_succeeds(self, vp, monkeypatch):
        _patch_make(monkeypatch, _FakeEnv([[(0.95, False), (0.95, False)]]))
        result = vp.run_task_verification(TASK, _NeverSucceeds, n_episodes=2, save_lerobot=False)
        assert not result["ok"]
        assert result["success_rate"] == 0.0
        assert any(issue.startswith("[success]") for issue in result["issues"])

    def test_success_budget_and_seeding(self, vp, monkeypatch):
        env = _FakeEnv([[(0.95, False)], [(0.95, False)], [(0.5, True), (0.3, False)]])
        calls = _patch_make(monkeypatch, env)
        result = vp.run_task_verification(TASK, _NeverSucceeds, n_episodes=3, save_lerobot=False)
        assert result["ok"], result["issues"]
        assert result["success_rate"] == pytest.approx(1 / 3, abs=1e-3)
        assert calls[0][1]["config_overrides"] is None
        seeds = [ep * 37 + 5 for ep in range(3)]
        assert env.reset_seeds == seeds
        assert [p.rng.random() for p in _NeverSucceeds.built] == [
            np.random.default_rng(s).random() for s in seeds]

    def test_env_closed_when_a_reset_raises(self, vp, monkeypatch):
        env = _FakeEnv([[(0.0, False)]])

        def broken_reset(seed=None):
            raise RuntimeError("reset failed")

        env.reset = broken_reset
        _patch_make(monkeypatch, env)
        result = vp.run_task_verification(TASK, _NeverSucceeds, n_episodes=1, save_lerobot=False)
        assert not result["ok"]
        assert env.closed


def test_hetbi_env_check_exits_nonzero_on_failure(monkeypatch):
    monkeypatch.setattr(ib, "make", _no_make)
    with pytest.raises(SystemExit) as exc:
        _run_script(monkeypatch, SCRIPTS / "tests" / "test_hetbi_envs.py",
                    "--no-render", "--tasks", "ib-nope-v1")
    assert exc.value.code == 1


# ── generate_dataset.py ─────────────────────────────────────────────


class TestGenerateDataset:

    @pytest.fixture
    def gd(self, tmp_path, monkeypatch):
        home = tmp_path / "home" / ".imbench"
        home.mkdir(parents=True)
        (home / "machine.json").write_text(json.dumps({"machine_id": "07"}))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        gd = _load("generate_dataset", SCRIPTS / "generate_dataset.py")
        monkeypatch.setattr(gd, "report_episode", lambda **kw: None)
        return gd

    def _main(self, gd, monkeypatch, traj_dir, out_dir):
        monkeypatch.setattr(sys, "argv", [
            "generate_dataset.py", "--task", TASK, "--traj-dir", str(traj_dir),
            "--out-dir", str(out_dir), "--obs-mode", "proprio"])
        gd.main()

    def test_bad_trajectory_rejected_before_writing(self, gd, tmp_path, monkeypatch):
        trajs = tmp_path / "trajs"
        trajs.mkdir()
        np.save(trajs / "a_bad.npy", np.zeros((3, 6)))
        np.save(trajs / "b_good.npy", np.zeros((3, 7)))
        env = _FakeEnv([[(0.0, False)] * 3])
        _patch_make(monkeypatch, env)
        with pytest.raises(SystemExit, match=r"a_bad\.npy: expected shape \(T, 7\)"):
            self._main(gd, monkeypatch, trajs, tmp_path / "out")
        assert not (tmp_path / "out").exists()
        assert env.closed

    def test_header_counts_replayed_trajectories(self, gd, tmp_path, monkeypatch, capsys):
        # Real sim: one 3-step trajectory and one empty one.
        trajs = tmp_path / "trajs"
        trajs.mkdir()
        np.save(trajs / "a.npy", np.zeros((3, 7), np.float32))
        np.save(trajs / "b_empty.npy", np.zeros((0, 7), np.float32))
        self._main(gd, monkeypatch, trajs, tmp_path / "out")
        out = capsys.readouterr().out
        assert "Generating 1 episodes" in out
        assert "skipping empty trajectory" in out
        assert (tmp_path / "out" / "M07" / "meta" / "episode_000000_metadata.json").exists()


# ── scripts/tests/test_pick_lift_release.py ─────────────────────────


def test_pick_lift_example_targets_the_object():
    policy = runpy.run_path(str(SCRIPTS / "tests" / "test_pick_lift_release.py"))["PickLiftPolicy"]()
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 1.0]), "object_pos": np.array([0.1, -0.05, 0.83])}
    policy.phase = "approach"
    action = policy.act(obs)
    assert action[0] > 0 and action[1] < 0  # towards the object, not the fixed fallback point
    with pytest.raises(KeyError):
        policy.act({"robot0_eef_pos": np.zeros(3)})
