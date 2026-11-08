"""Tests for the Gymnasium contract exposed by BaseTask.

Covers the public interface that every task must satisfy:
- reset() / step() return types and shapes
- observation_space / action_space bounds
- step counter exposed via info["step"]
- seed reproducibility
- truncation at max_episode_steps
- config and control_freq parsing
"""

import numpy as np
import pytest
from gymnasium import spaces

import imbench as ib
from imbench.tasks.base_task import BaseTask


# ── BaseTask Gymnasium-contract tests ───────────────────────────────


class TestBaseTask:
    """Verify the Gymnasium protocol through a concrete single-arm task."""

    @pytest.fixture(scope="class")
    @classmethod
    def env(cls):
        env = ib.make("ib-p2-cube-toss-v1", obs_mode="combo", render_mode=None, seed=0)
        yield env
        env.close()

    def test_is_base_task(self, env):
        assert isinstance(env, BaseTask)

    def test_reset_returns_obs_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert "task_score" in info
        assert "step" in info

    def test_step_returns_5tuple(self, env):
        env.reset()
        lo, hi = env.action_space.low, env.action_space.high
        result = env.step(np.zeros(lo.shape[0]))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_space_returns_shapes(self, env):
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict)
        for key, space in obs_space.items():
            assert isinstance(space, spaces.Box)
            assert isinstance(space.shape, tuple)
            assert all(isinstance(d, int) for d in space.shape)

    def test_action_space_returns_bounds(self, env):
        space = env.action_space
        assert isinstance(space, spaces.Box)
        assert isinstance(space.low, np.ndarray)
        assert isinstance(space.high, np.ndarray)
        assert space.low.shape == space.high.shape
        assert np.all(space.low <= space.high)

    def test_step_count_increments(self, env):
        env.reset()
        lo, _ = env.action_space.low, env.action_space.high
        for i in range(3):
            _, _, _, _, info = env.step(np.zeros(lo.shape[0]))
            assert info["step"] == i + 1

    def test_step_count_resets_on_reset(self, env):
        env.reset()
        lo, _ = env.action_space.low, env.action_space.high
        env.step(np.zeros(lo.shape[0]))
        env.step(np.zeros(lo.shape[0]))
        obs, info = env.reset()
        assert info["step"] == 0
        _, _, _, _, info = env.step(np.zeros(lo.shape[0]))
        assert info["step"] == 1

    def test_seed_works(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        for key in obs1:
            if isinstance(obs1[key], np.ndarray):
                np.testing.assert_array_equal(obs1[key], obs2[key])

    def test_truncation_at_max_steps(self):
        env = ib.make(
            "ib-p2-cube-toss-v1",
            obs_mode="proprio",
            render_mode=None,
            seed=0,
            config_overrides={"max_episode_steps": 3},
        )
        try:
            env.reset()
            lo, _ = env.action_space.low, env.action_space.high
            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(
                    np.zeros(lo.shape[0])
                )
            assert truncated is True
        finally:
            env.close()

    def test_config_loaded(self, env):
        assert hasattr(env, "cfg")
        assert isinstance(env.cfg, dict)

    def test_control_freq_parsed(self, env):
        assert env.control_freq > 0

    def test_obs_mode_stored(self, env):
        assert env.obs_mode == "combo"
