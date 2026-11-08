"""Tests for heterogeneous bimanual (hetbi) tasks."""

import numpy as np
import pytest

import imbench as ib
from imbench.tasks.base_task import CONFIGS_DIR


# Collect all hetbi task IDs from YAML files
_HETBI_DIR = CONFIGS_DIR / "hetbi"
HETBI_TASK_IDS = sorted(p.stem for p in _HETBI_DIR.glob("*.yaml"))


# ── Single-variant deep tests (use bowl-table as canonical) ─────────


class TestHetBiBowlTable:
    """Deep tests on ib-p1-bowl-table-v1 as the canonical hetbi task."""

    TASK_ID = "ib-p1-bowl-table-v1"

    @pytest.fixture(scope="class")
    @classmethod
    def env(cls):
        env = ib.make(cls.TASK_ID, obs_mode="combo", render_mode=None, seed=0, privileged_obs=True)
        yield env
        env.close()

    def test_in_registry(self):
        assert self.TASK_ID in ib.list_tasks()

    def test_make_and_reset(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert info["task_score"] == 0.0
        assert info["step"] == 0

    def test_obs_keys_bimanual_proprio(self, env):
        obs, _ = env.reset()
        for robot_idx in [0, 1]:
            for suffix in ["eef_pos", "eef_quat", "gripper_qpos",
                           "joint_pos", "joint_vel"]:
                key = f"robot{robot_idx}_{suffix}"
                assert key in obs, f"Missing key: {key}"
                assert isinstance(obs[key], np.ndarray)

    def test_obs_has_object_state(self, env):
        obs, _ = env.reset()
        assert "object_pos" in obs
        assert obs["object_pos"].shape == (3,)

    def test_obs_combo_has_cameras(self, env):
        obs, _ = env.reset()
        rgb_keys = [k for k in obs if k.startswith("rgb_")]
        assert len(rgb_keys) > 0

    def test_step_returns_valid_types(self, env):
        env.reset()
        lo, hi = env.action_space.low, env.action_space.high
        action = np.zeros(lo.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "task_score" in info
        assert 0.0 <= info["task_score"] <= 1.0

    def test_action_space_14dof(self, env):
        lo, hi = env.action_space.low, env.action_space.high
        # 2 Panda arms: 6 OSC + 1 gripper each = 14
        assert lo.shape[0] == 14
        assert hi.shape[0] == 14

    def test_task_conditions_failure(self, env):
        assert hasattr(env, "_failure_floor_threshold")
        assert env._failure_floor_threshold == env.cfg["task_conditions"]["failure"]["floor_drop_below"]

    def test_table_config(self, env):
        cfg_table = env.cfg["scene"]["table"]
        assert env._table_height == cfg_table["height"]
        assert env._table_full_size == tuple(cfg_table["full_size"])
        # Verify MuJoCo env also received these values
        np.testing.assert_allclose(env._env.table_full_size, cfg_table["full_size"], atol=1e-6)

    def test_payload_fell_off_resets(self, env):
        """_payload_fell_off should be False after reset."""
        env.reset()
        assert env._payload_fell_off is False

    def test_suction_weld_exists(self, env):
        assert hasattr(env, "_suction_weld")

    def test_gripper_types(self, env):
        """Hetbi tasks pair a suction arm with a parallel-jaw arm."""
        assert env.GRIPPER_TYPES is not None
        assert len(env.GRIPPER_TYPES) == 2

    def test_randomization_config_parsed(self, env):
        """Randomization flags should be parsed from config."""
        assert hasattr(env, "_rand_payload_mass")
        assert hasattr(env, "_rand_payload_friction")
        assert hasattr(env, "_rand_table_friction")
        assert hasattr(env, "_rand_placement")
        assert hasattr(env, "_rand_object_color")
        assert hasattr(env, "_rand_table_color")
        assert hasattr(env, "_rand_light_position")
        assert hasattr(env, "_rand_light_color")
        assert hasattr(env, "_rand_action_noise")
        assert hasattr(env, "_rand_obs_noise")

    def test_compute_task_score_bounded(self, env):
        env.reset()
        raw_obs = env._get_sim_observations()
        score = env._compute_task_score(raw_obs)
        assert 0.0 <= score <= 1.0

    def test_termination_not_immediate(self, env):
        """Freshly reset env should not be terminated."""
        env.reset()
        raw_obs = env._get_sim_observations()
        assert env._check_termination(raw_obs) is False


# ── Color helpers tests ─────────────────────────────────────────────


class TestHetBiColorHelpers:
    """Test constrained color randomization methods."""

    @pytest.fixture
    def env(self):
        env = ib.make("ib-p1-bowl-table-v1", obs_mode="combo", render_mode=None, seed=0)
        yield env
        env.close()

    def test_random_object_color_plausible(self, env):
        """Object colors should be muted (avoid neon extremes)."""
        for seed in range(10):
            env._rng = np.random.RandomState(seed)
            rgba = env._random_object_color()
            assert rgba.shape == (4,)
            assert rgba[3] == 1.0  # alpha
            # RGB channels should be in the muted range
            for c in rgba[:3]:
                assert 0.1 <= c <= 0.9

    def test_random_table_color_neutral(self, env):
        """Table colors should be neutral (grays, off-whites, wood tones)."""
        for seed in range(10):
            env._rng = np.random.RandomState(seed)
            rgba = env._random_table_color()
            assert rgba.shape == (4,)
            assert rgba[3] == 1.0
            for c in rgba[:3]:
                assert 0.3 <= c <= 1.0

    def test_light_color_near_white(self, env):
        """When light color randomization is enabled, result should be near-white."""
        env._rand_light_color = True
        env.reset(seed=42)
        sim = env._env.sim
        for i in range(sim.model.nlight):
            diffuse = sim.model.light_diffuse[i]
            # Should be warm-to-cool white, not strongly colored
            for c in diffuse:
                assert 0.4 <= c <= 1.0, f"Diffuse {diffuse} out of range"


# ── Placement randomization ─────────────────────────────────────────


class TestHetBiPlacementRandomization:

    def test_placement_jitter_changes_pos(self):
        """With placement randomization on, payload pos should vary across resets."""
        env = ib.make(
            "ib-p1-bowl-table-v1",
            obs_mode="combo", render_mode=None, seed=0,
            config_overrides={
                "randomization": {
                    "payload_placement": {"enabled": True, "x_std": 0.05, "y_std": 0.05}
                }
            },
        )
        try:
            positions = []
            for seed in range(3):
                obs, _ = env.reset(seed=seed)
                if "object_pos" in obs:
                    positions.append(obs["object_pos"].copy())
            if len(positions) >= 2:
                # At least some positions should differ
                diffs = [np.linalg.norm(positions[i] - positions[0])
                         for i in range(1, len(positions))]
                assert any(d > 1e-6 for d in diffs), \
                    "Placement randomization did not change positions"
        finally:
            env.close()

    @pytest.mark.parametrize("task_id", ["ib-p1-bowl-table-v1", "ib-p1-plate-shelf-hetbi-v1"])
    def test_shipped_config_varies_scene_across_seeds(self, task_id):
        """Evaluation seeds must give different scenes, not just different lighting."""
        env = ib.make(task_id, obs_mode="proprio", render_mode=None, seed=0)
        try:
            joint = env._env._payload.joints[0]
            starts = []
            for seed in range(5):
                env.reset(seed=seed)
                starts.append(env._env.sim.data.get_joint_qpos(joint)[:2].copy())
            assert len(np.unique(np.round(starts, 5), axis=0)) == len(starts), \
                f"{task_id}: payload start xy repeats across seeds"
        finally:
            env.close()


# ── Cross-variant tests (parametrized over all hetbi tasks) ─────────


@pytest.mark.slow
class TestHetBiAllVariants:
    """Tests that apply to every registered hetbi task variant.

    Marked slow: spawns one hetbi env per task × 4 methods.
    The canonical task is exhaustively covered by :class:`TestHetBiBowlTable`.
    """

    @pytest.mark.parametrize("task_id", HETBI_TASK_IDS)
    def test_make_reset_step_close(self, task_id):
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=0)
        try:
            obs, info = env.reset()
            assert isinstance(obs, dict)
            assert info["task_score"] == 0.0
            lo, _ = env.action_space.low, env.action_space.high
            obs, reward, terminated, truncated, info = env.step(np.zeros(lo.shape[0]))
            assert isinstance(obs, dict)
            assert "task_score" in info
            assert 0.0 <= info["task_score"] <= 1.0
        finally:
            env.close()

    @pytest.mark.parametrize("task_id", HETBI_TASK_IDS)
    def test_has_bimanual_proprio(self, task_id):
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=0)
        try:
            obs, _ = env.reset()
            for robot_idx in [0, 1]:
                assert f"robot{robot_idx}_eef_pos" in obs
        finally:
            env.close()

    @pytest.mark.parametrize("task_id", HETBI_TASK_IDS)
    def test_task_conditions_present(self, task_id):
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=0)
        try:
            assert hasattr(env, "_failure_floor_threshold")
        finally:
            env.close()

    @pytest.mark.parametrize("task_id", HETBI_TASK_IDS)
    def test_table_config_present(self, task_id):
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=0)
        try:
            assert hasattr(env, "_table_height")
            assert hasattr(env, "_table_full_size")
            assert hasattr(env, "_table_offset")
            assert env._table_height > 0
        finally:
            env.close()


# =====================================================================
#  Programmatic success — teleport payload to verify scoring
# =====================================================================

# Fixture-goal tasks: payload posed inside the fixture, released and at rest = success
_FIXTURE_GOAL_IDS = [
    "ib-p1-cracker-box-suction-v1",
    "ib-p1-cracker-box-jaw-v1",
    "ib-p1-plate-shelf-hetbi-v1",
]

_CRACKER_BOX_IDS = [
    "ib-p1-cracker-box-suction-v1",
    "ib-p1-cracker-box-jaw-v1",
]

# Table-goal tasks: payload upright on the table, outside the fixture = success
_TABLE_GOAL_IDS = [
    "ib-p1-bowl-table-v1",
]

# Goal payload orientation (wxyz, MuJoCo qpos order): the cracker box lies flat
# (thin axis vertical, long axis along the slot channel); the plate stands
# on edge with its face normal along world y, between the shelf dividers.
_FLAT_BOX_WXYZ = np.array([0.7071068, 0.0, -0.7071068, 0.0])
_ON_EDGE_PLATE_WXYZ = np.array([0.7071068, 0.7071068, 0.0, 0.0])
_FIXTURE_GOAL_QUATS = {tid: _FLAT_BOX_WXYZ for tid in _CRACKER_BOX_IDS}
_FIXTURE_GOAL_QUATS["ib-p1-plate-shelf-hetbi-v1"] = _ON_EDGE_PLATE_WXYZ


def _teleport_payload(env, pos, quat_wxyz):
    sim = env._env.sim
    joint = env._env._payload.joints[0]
    sim.data.set_joint_qpos(joint, np.concatenate([pos, quat_wxyz]))
    sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()


def _fixture_target(env, task_id):
    """Slot interior centre for the cracker-box variants, else the fixture goal point."""
    if task_id in _CRACKER_BOX_IDS:
        bounds = env._env._fixture_bounds
        return np.array([0.5 * (lo + hi) for lo, hi in bounds.values()])
    return env._env._target_goal_pos.copy()


def _step_until_done(env, max_steps=60):
    zero = np.zeros(env.action_space.shape)
    for _ in range(max_steps):
        step = env.step(zero)
        if step[2]:
            break
    return step


class TestHetBiFixtureGoalSuccess:

    @pytest.mark.parametrize("task_id", _FIXTURE_GOAL_IDS)
    def test_teleport_payload_to_fixture(self, task_id, monkeypatch):
        """Goal pose inside the fixture scores 1 and terminates once at rest
        (stability stubbed so a single mid-air frame is enough)."""
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=42)
        try:
            env.reset()
            if task_id == "ib-p1-plate-shelf-hetbi-v1":
                env._handoff_complete = True  # isolate placement after a handover
            monkeypatch.setattr(env, "_payload_is_stable_multistep", lambda *a, **k: True)
            _teleport_payload(env, _fixture_target(env, task_id), _FIXTURE_GOAL_QUATS[task_id])
            raw_obs = env._get_sim_observations(force_update=True)
            score = env._compute_task_score(raw_obs)
            assert score == 1.0, f"{task_id}: expected 1.0, got {score}"
            assert env._check_termination(raw_obs) is True
        finally:
            env.close()

    @pytest.mark.parametrize("task_id", _CRACKER_BOX_IDS)
    def test_box_settled_in_slot_terminates(self, task_id):
        """A flat box dropped into the slot settles there and ends the episode as a success."""
        env = ib.make(task_id, obs_mode="proprio", render_mode=None, seed=42)
        try:
            env.reset()
            _teleport_payload(env, _fixture_target(env, task_id), _FLAT_BOX_WXYZ)
            _, _, terminated, _, info = _step_until_done(env)
            assert terminated is True
            assert info["is_success"] is True
            assert info["task_score"] == 1.0
        finally:
            env.close()

    def test_plate_settled_in_shelf_terminates(self):
        """A plate standing on edge in the shelf's centre slot settles into success."""
        env = ib.make("ib-p1-plate-shelf-hetbi-v1", obs_mode="proprio", render_mode=None, seed=42)
        try:
            env.reset()
            env._handoff_complete = True  # isolate physical placement after handover
            fb = env._env._fixture_bounds
            z = fb["z"][0] + env._env._payload.horizontal_radius + 0.003
            pos = np.array([np.mean(fb["x"]), np.mean(fb["y"]), z])
            _teleport_payload(env, pos, _ON_EDGE_PLATE_WXYZ)
            _, _, terminated, _, info = _step_until_done(env, max_steps=80)
            assert terminated is True
            assert info["is_success"] is True
            assert info["task_score"] == 1.0
        finally:
            env.close()


class TestHetBiTableGoalSuccess:

    @pytest.mark.parametrize("task_id", _TABLE_GOAL_IDS)
    def test_settled_on_table_terminates(self, task_id):
        """Payload resting upright on the table, clear of the fixture → success, score 1."""
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=42)
        try:
            env.reset()
            z = env._table_height + abs(env._env._payload.bottom_offset[2]) + 0.002
            _teleport_payload(env, np.array([0.1, -0.15, z]), np.array([1.0, 0.0, 0.0, 0.0]))
            _, _, terminated, _, info = _step_until_done(env)
            assert terminated is True
            assert info["is_success"] is True
            assert info["task_score"] == 1.0
        finally:
            env.close()


@pytest.mark.parametrize("task_id", ["ib-p1-cracker-box-suction-v1", "ib-p1-cracker-box-jaw-v1"])
def test_frozen_arm_stays_still_under_action_noise(task_id, monkeypatch):
    """The frozen arm's action slice is zeroed after the noise is added."""
    task = ib.make(task_id, obs_mode="proprio",
                   config_overrides={"randomization": {"action_noise": {"enabled": True, "std": 0.5}}})
    try:
        task.reset(seed=0)
        sent = []
        env_step = task._env.step

        def record(action):
            sent.append(np.array(action))
            return env_step(action)

        monkeypatch.setattr(task._env, "step", record)
        task.step(np.full(task.action_space.shape, 0.5))
        assert not sent[0][task._frozen_action_slice].any()
    finally:
        task.close()
