"""collect_demos.py collection loop (seeds, pauses, cooldown, display handling,
dashboard keys, telemetry) and launch_env.py quitting.

The collect() tests run a real proprio sim driven by a scripted teleop device.
"""

import json
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest

from imbench.teleop import TeleopInterface, _TELEOP_REGISTRY
from tests.test_data_pipeline_collect import (  # noqa: F401  (home is a fixture)
    N_ACTIVE,
    TASK,
    _collect_demos,
    _patch_make,
    _script,
    _ScriptedTeleop,
    home,
)


def _plan_teleop(monkeypatch, plan, end_on_call=None):
    """Register teleop ``"plan"`` replaying *plan*, one ``(x, gripper, paused)`` per get_action().

    The end-episode input is latched inside get_action() call *end_on_call*,
    as a polled device (gamepad) does.  Once *plan* is used up get_action()
    raises KeyboardInterrupt (Ctrl-C).
    """
    calls = []

    class PlanTeleop(TeleopInterface):
        def __init__(self, num_arms=1):
            self._end = False

        def start(self):
            pass

        def stop(self):
            pass

        def reset(self):
            self._end = False

        def get_action(self, obs):
            if len(calls) == len(plan):
                raise KeyboardInterrupt
            x, gripper, paused = plan[len(calls)]
            calls.append(len(calls))
            if len(calls) == end_on_call:
                self._end = True
            return {"action": np.array([x, 0, 0, 0, 0, 0, gripper], dtype=float), "paused": paused}

        def should_end_episode(self):
            return self._end

    monkeypatch.setitem(_TELEOP_REGISTRY, "plan", PlanTeleop)


def _episodes(root):
    path = root / "meta" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _columns(root, prefix, n):
    table = pq.read_table(root / "data" / "chunk-000" / "episode_000000.parquet")
    return np.stack([table.column(f"{prefix}.{i}").to_numpy() for i in range(n)], axis=1)


@pytest.fixture
def headless(monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("the display check applies on Linux")
    monkeypatch.delenv("DISPLAY", raising=False)


def _no_windows(monkeypatch):
    """Make any cv2 window call fail the test."""
    import cv2

    def boom(*_args, **_kwargs):
        raise AssertionError("a GUI window was opened")

    for name in ("namedWindow", "imshow"):
        monkeypatch.setattr(cv2, name, boom)


# ── Seeds and telemetry ───────────────────────────────────────────

def test_seeds_are_per_machine_resume_and_survive_discard(home, tmp_path, monkeypatch):
    cd = _collect_demos()
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    monkeypatch.setattr(cd, "report_episode", lambda **kw: None)
    _patch_make(monkeypatch, cd)
    out = tmp_path / "demos"

    # The first attempt is discarded and restarts with the same seed.
    cd.collect(TASK, "scripted", n_demos=2, out_dir=str(out), render=False, use_dashboard=False)
    cd.collect(TASK, "scripted", n_demos=1, out_dir=str(out), render=False, use_dashboard=False)

    root = out / "M07"
    first = 7_000_000  # machine 07
    assert [e["seed"] for e in _episodes(root)] == [first, first + 1, first + 2]
    metas = [json.loads((root / "meta" / f"episode_{i:06d}_metadata.json").read_text()) for i in range(3)]
    assert [m["seed"] for m in metas] == [first, first + 1, first + 2]


def test_telemetry_sends_session_operator_id_per_episode(home, tmp_path, monkeypatch):
    cd = _collect_demos()
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    session = home / ".imbench" / "session.json"
    sent = []

    def report(**kw):
        sent.append(kw)
        # The operator changes in the dashboard after the first save.
        session.write_text(json.dumps({"teleoperator_id": "OP-009"}))

    monkeypatch.setattr(cd, "report_episode", report)
    _patch_make(monkeypatch, cd)

    cd.collect(TASK, "scripted", n_demos=2, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=False)

    root = tmp_path / "demos" / "M07"
    metas = [json.loads((root / "meta" / f"episode_{i:06d}_metadata.json").read_text()) for i in range(2)]
    assert [m["teleoperator_id"] for m in metas] == ["OP-003", "OP-009"]
    assert [kw["operator_id"] for kw in sent] == ["OP-003", "OP-009"]


# ── Pauses, gripper settling and cooldown ─────────────────────────

def test_action_filter_restarts_from_rest_after_pause(home, tmp_path, monkeypatch):
    cd = _collect_demos()
    n_move = 5
    plan = ([(0.5, -1.0, False)] * n_move + [(0.0, -1.0, True)]
            + [(0.0, -1.0, False)] * (N_ACTIVE - n_move) + [(0.0, -1.0, True)] * cd.COOLDOWN_STEPS)
    _plan_teleop(monkeypatch, plan)
    monkeypatch.setattr(cd, "report_episode", lambda **kw: None)
    _patch_make(monkeypatch, cd)

    cd.collect(TASK, "plan", n_demos=1, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=False)

    actions = _columns(tmp_path / "demos" / "M07", "action", 7)
    assert np.all(actions[:n_move, 0] > 0)
    # No leftover motion from before the pause.
    np.testing.assert_array_equal(actions[n_move:N_ACTIVE, :6], 0.0)


@pytest.mark.parametrize("n_move", [0, 1], ids=["first_command", "after_motion"])
def test_sim_keeps_stepping_until_gripper_settles(home, tmp_path, monkeypatch, n_move):
    cd = _collect_demos()
    # Snap-close, then idle; paused ticks report gripper 0 (as Quest does).
    plan = ([(0.5, -1.0, False)] * n_move + [(0.0, 1.0, False)]
            + [(0.0, 0.0, True)] * (N_ACTIVE + cd.COOLDOWN_STEPS))
    _plan_teleop(monkeypatch, plan)
    monkeypatch.setattr(cd, "report_episode", lambda **kw: None)
    _patch_make(monkeypatch, cd)

    cd.collect(TASK, "plan", n_demos=1, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=False)

    root = tmp_path / "demos" / "M07"
    (entry,) = _episodes(root)
    assert entry["length"] == N_ACTIVE + cd.COOLDOWN_STEPS
    actions = _columns(root, "action", 7)
    # Steps after the snap hold zero motion and the commanded gripper.
    np.testing.assert_array_equal(actions[n_move + 1:N_ACTIVE, :6], 0.0)
    np.testing.assert_array_equal(actions[n_move:N_ACTIVE, 6], 1.0)
    fingers = _columns(root, "observation.robot0_gripper_qpos", 2)
    assert fingers[N_ACTIVE - 1, 0] < fingers[n_move + 1, 0] - 1e-3  # the jaws kept closing


def test_end_episode_input_polled_during_cooldown(home, tmp_path, monkeypatch):
    cd = _collect_demos()
    n_cooldown = 3
    plan = [(0.2, -1.0, False)] * N_ACTIVE + [(0.0, -1.0, True)] * cd.COOLDOWN_STEPS
    _plan_teleop(monkeypatch, plan, end_on_call=N_ACTIVE + n_cooldown)
    monkeypatch.setattr(cd, "report_episode", lambda **kw: None)
    _patch_make(monkeypatch, cd)

    cd.collect(TASK, "plan", n_demos=1, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=False)

    # The press cuts the cooldown short; the episode is still saved.
    (entry,) = _episodes(tmp_path / "demos" / "M07")
    assert entry["length"] == N_ACTIVE + n_cooldown


# ── Display handling and dashboard keys ───────────────────────────

def test_headless_dashboard_falls_back_without_windows(home, headless, tmp_path, monkeypatch):
    cd = _collect_demos()
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    monkeypatch.setattr(cd, "report_episode", lambda **kw: None)
    _patch_make(monkeypatch, cd)
    _no_windows(monkeypatch)

    cd.collect(TASK, "scripted", n_demos=1, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=True)

    assert len(_episodes(tmp_path / "demos" / "M07")) == 1


def test_no_dashboard_opens_no_packing_preview(home, headless, tmp_path, monkeypatch):
    cd = _collect_demos()
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    monkeypatch.setattr(cd, "report_episode", lambda **kw: None)
    _patch_make(monkeypatch, cd)
    _no_windows(monkeypatch)

    cd.collect("ib-p7-packing-v1", "scripted", n_demos=1, out_dir=str(tmp_path / "demos"),
               render=False, use_dashboard=False)

    assert len(_episodes(tmp_path / "demos" / "M07")) == 1


def _forbid_env(monkeypatch, cd):
    def make(*_args, **_kwargs):
        raise AssertionError("env built")

    monkeypatch.setattr(cd.ib, "make", make)


def test_render_without_display_fails_before_building_the_env(home, headless, tmp_path, monkeypatch, capsys):
    cd = _collect_demos()
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    _forbid_env(monkeypatch, cd)

    with pytest.raises(SystemExit) as exc:
        cd.collect(TASK, "scripted", n_demos=1, out_dir=str(tmp_path / "demos"), render=True, use_dashboard=True)
    assert exc.value.code == 1
    assert "--render needs a display" in capsys.readouterr().out


def test_unavailable_teleop_device_exits_with_one_line(home, tmp_path, monkeypatch, capsys):
    import imbench.teleop as teleop

    cd = _collect_demos()
    monkeypatch.setitem(teleop._UNAVAILABLE, "absent", ImportError("No module named 'absentlib'"))
    _forbid_env(monkeypatch, cd)

    with pytest.raises(SystemExit) as exc:
        cd.collect(TASK, "absent", n_demos=1, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=False)
    assert exc.value.code == 1
    assert "[collect] ERROR: Teleop device 'absent' is unavailable" in capsys.readouterr().out
    assert not (tmp_path / "demos").exists()


def test_device_that_fails_to_start_exits_with_one_line(home, tmp_path, monkeypatch, capsys):
    cd = _collect_demos()

    class Missing(_ScriptedTeleop):
        def start(self):
            raise RuntimeError("No gamepad found.")

    monkeypatch.setitem(_TELEOP_REGISTRY, "missing", Missing)
    _patch_make(monkeypatch, cd)
    with pytest.raises(SystemExit) as exc:
        cd.collect(TASK, "missing", n_demos=1, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=False)
    assert exc.value.code == 1
    assert "[collect] ERROR: No gamepad found." in capsys.readouterr().out


def test_q_in_dashboard_stops_collection(home, tmp_path, monkeypatch):
    cd = _collect_demos()
    import imbench.utils.obs_visualizer as viz

    updates = []

    class FakeDashboard:
        def __init__(self, **_kwargs):
            pass

        def update(self, obs, info=None, reward=None):
            updates.append(reward)
            return ord("q") if len(updates) == 3 else 255

        def reset(self):
            pass

        def close(self):
            pass

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(viz, "ObsDashboard", FakeDashboard)
    monkeypatch.setitem(_TELEOP_REGISTRY, "scripted", _ScriptedTeleop)
    monkeypatch.setattr(cd, "report_episode", lambda **kw: None)
    _patch_make(monkeypatch, cd)

    cd.collect(TASK, "scripted", n_demos=3, out_dir=str(tmp_path / "demos"), render=False, use_dashboard=True)

    # Collection stops at the key press instead of running all episodes.
    assert len(updates) == 3
    assert len(_episodes(tmp_path / "demos" / "M07")) < 3


# ── launch_env.py ─────────────────────────────────────────────────

class _FakeEnv:
    class action_space:
        low, high = -np.ones(7), np.ones(7)

    def __init__(self):
        self.resets = 0
        self.steps = 0

    def reset(self, seed=None):
        self.resets += 1
        return {}, {"task_score": 0.0}

    def step(self, action):
        self.steps += 1
        return {}, 0.0, False, False, {"task_score": 0.0}

    def render(self):
        pass

    def close(self):
        pass


def _run_launch_env(monkeypatch, argv, dashboard_cls=None):
    le = _script("launch_env")
    env = _FakeEnv()
    monkeypatch.setattr(le.ib, "make", lambda *a, **k: env)
    if dashboard_cls is not None:
        monkeypatch.setattr(le, "ObsDashboard", dashboard_cls)
    monkeypatch.setattr(sys, "argv", ["launch_env.py", *argv])
    le.main()
    return env


def test_launch_env_q_quits_all_episodes(monkeypatch):
    class QuitDashboard:
        def __init__(self, **_kwargs):
            self.n = 0

        def reset(self):
            pass

        def update(self, obs, info=None, reward=None):
            self.n += 1
            return ord("q") if self.n == 3 else 255

        def close(self):
            pass

    env = _run_launch_env(monkeypatch, [TASK, "--episodes", "3", "--steps", "50", "--dashboard"], QuitDashboard)
    assert (env.resets, env.steps) == (1, 2)


def test_launch_env_zero_steps(monkeypatch, capsys):
    env = _run_launch_env(monkeypatch, [TASK, "--steps", "0"])
    assert env.steps == 0
    assert "steps=0" in capsys.readouterr().out
