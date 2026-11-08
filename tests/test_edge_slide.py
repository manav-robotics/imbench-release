"""Unit tests for the Edge Slide and Grasp task (ib-p1-edge-slide-v1)."""

import numpy as np
import pytest
import imbench as ib


TASK_ID = "ib-p1-edge-slide-v1"


@pytest.fixture(scope="module")
def env():
    """Shared edge-slide env for all tests in this module."""
    env = ib.make(TASK_ID, obs_mode="combo", render_mode=None, seed=0, privileged_obs=True)
    yield env
    env.close()


# ── Registry & lifecycle ────────────────────────────────────────────


def test_task_in_registry():
    assert TASK_ID in ib.list_tasks()


def test_make_and_reset(env):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert info["task_score"] == 0.0
    assert info["step"] == 0


def test_step_returns_valid_types(env):
    env.reset()
    lo, hi = env.action_space.low, env.action_space.high
    obs, reward, terminated, truncated, info = env.step(np.zeros(lo.shape[0]))
    assert isinstance(obs, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "task_score" in info
    assert 0.0 <= info["task_score"] <= 1.0


# ── Action space ────────────────────────────────────────────────────


def test_action_space_7dof(env):
    """Single Panda arm: 6 OSC + 1 gripper = 7."""
    lo, hi = env.action_space.low, env.action_space.high
    assert lo.shape[0] == 7
    assert hi.shape[0] == 7


def test_max_episode_steps(env):
    assert env.MAX_EPISODE_STEPS == env.cfg["max_episode_steps"]


# ── Observation tests ───────────────────────────────────────────────


def test_obs_keys_proprio(env):
    obs, _ = env.reset()
    expected_keys = [
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
        "robot0_joint_pos", "robot0_joint_vel",
    ]
    for key in expected_keys:
        assert key in obs, f"Missing proprio key: {key}"
        assert isinstance(obs[key], np.ndarray)


def test_object_state_present(env):
    obs, _ = env.reset()
    assert "object_pos" in obs
    assert obs["object_pos"].shape == (3,)


def test_combo_mode_has_cameras(env):
    obs, _ = env.reset()
    rgb_keys = [k for k in obs if k.startswith("rgb_")]
    assert len(rgb_keys) > 0
    for key in rgb_keys:
        assert obs[key].ndim == 3


def test_proprio_mode_no_cameras():
    e = ib.make(TASK_ID, obs_mode="proprio", render_mode=None, seed=0)
    try:
        obs, _ = e.reset()
        assert not any(k.startswith("rgb_") for k in obs)
    finally:
        e.close()


# ── Task score & termination ───────────────────────────────────────


def test_score_starts_at_zero(env):
    env.reset()
    assert env.task_score() == 0.0


def test_task_score_bounds(env):
    env.reset()
    lo, hi = env.action_space.low, env.action_space.high
    for _ in range(10):
        _, _, terminated, truncated, info = env.step(np.random.uniform(lo, hi))
        assert 0.0 <= info["task_score"] <= 1.0
        if terminated or truncated:
            break


def test_book_fell_off_resets(env):
    env.reset()
    assert env._book_fell_off is False


def test_no_immediate_termination(env):
    env.reset()
    raw_obs = env._get_sim_observations()
    assert env._check_termination(raw_obs) is False


# ── Config ──────────────────────────────────────────────────────────


def test_table_height(env):
    assert env._table_height == env.cfg["scene"]["table"]["height"]


def test_table_full_size(env):
    cfg_table = env.cfg["scene"]["table"]
    assert env._table_full_size == tuple(cfg_table["full_size"])
    # Verify MuJoCo env also received these values
    np.testing.assert_allclose(env._env.table_full_size, cfg_table["full_size"], atol=1e-6)


def test_table_offset_from_height(env):
    assert env._table_offset[2] == env._table_height


def test_robot_offset_default_none(env):
    assert env._robot_offset is None


def test_lift_threshold(env):
    assert env._lift_threshold == env._table_height + env.cfg["scene"]["lift_height"]


def test_custom_table_override():
    e = ib.make(TASK_ID, obs_mode="combo", render_mode=None, seed=0,
                config_overrides={"scene": {"table": {"height": 1.0}}})
    try:
        assert e._table_height == 1.0
    finally:
        e.close()


def test_geometry_randomization_ranges(env):
    assert env._book_length_range is not None
    assert env._book_width_range is not None
    assert env._book_friction_range is not None


# ── Visual randomization ───────────────────────────────────────────


def test_object_color_plausible(env):
    for seed in range(10):
        env._rng = np.random.RandomState(seed)
        rgba = env._random_object_color()
        assert rgba.shape == (4,)
        assert rgba[3] == 1.0
        for c in rgba[:3]:
            assert 0.1 <= c <= 0.9


def test_table_color_neutral(env):
    for seed in range(10):
        env._rng = np.random.RandomState(seed)
        rgba = env._random_table_color()
        assert rgba.shape == (4,)
        assert rgba[3] == 1.0
        for c in rgba[:3]:
            assert 0.3 <= c <= 1.0


def test_light_color_constrained(env):
    env._rand_light_color = True
    env.reset(seed=42)
    sim = env._env.sim
    for i in range(sim.model.nlight):
        for c in sim.model.light_diffuse[i]:
            assert 0.4 <= c <= 1.0


# ── Reproducibility ────────────────────────────────────────────────


def test_reproducibility(env):
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    for key in obs1:
        if isinstance(obs1[key], np.ndarray):
            np.testing.assert_array_equal(
                obs1[key], obs2[key],
                err_msg=f"Mismatch on key '{key}' across identical seeds",
            )


def test_different_seeds_differ(env):
    obs1, _ = env.reset(seed=0)
    obs2, _ = env.reset(seed=99)
    any_diff = any(
        isinstance(obs1[k], np.ndarray) and not np.array_equal(obs1[k], obs2[k])
        for k in obs1
    )
    assert any_diff, "Different seeds produced identical observations"


# ── Custom reward ───────────────────────────────────────────────────


def test_custom_reward_fn():
    def my_reward(task, raw_obs):
        return 42.0

    e = ib.make(TASK_ID, obs_mode="combo", render_mode=None, seed=0,
                reward_fn=my_reward)
    try:
        e.reset()
        lo, _ = e.action_space.low, e.action_space.high
        _, reward, _, _, _ = e.step(np.zeros(lo.shape[0]))
        assert reward == 42.0
    finally:
        e.close()


def test_default_reward_is_task_score(env):
    env.reset()
    lo, _ = env.action_space.low, env.action_space.high
    _, reward, _, _, info = env.step(np.zeros(lo.shape[0]))
    assert reward == info["task_score"]
