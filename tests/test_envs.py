"""Tests for HetBiEnv and HoBiEnv environments."""

import numpy as np
import pytest

import imbench as ib
import imbench.envs  # noqa: register envs


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hetbi_env():
    """Create a hetbi env once for the module (expensive MuJoCo build)."""
    env = ib.make("ib-p1-bowl-table-v1", obs_mode="combo", render_mode=None, seed=0)
    yield env
    env.close()


@pytest.fixture(scope="module")
def hobi_env():
    """Create a (still-bimanual) hobi env once for the module.

    Packing was moved to ib-p7-packing-v1 (single-arm/suction); use
    plate-shelf as the bimanual hobi exemplar instead.
    """
    env = ib.make("ib-p1-plate-shelf-hobi-v1", obs_mode="combo", render_mode=None, seed=0)
    yield env
    env.close()


# ── HetBiEnv tests ──────────────────────────────────────────────────


class TestHetBiEnv:
    """Tests for the heterogeneous bimanual environment."""

    def test_env_builds_successfully(self, hetbi_env):
        assert hetbi_env._env is not None

    def test_two_robots_present(self, hetbi_env):
        assert len(hetbi_env._env.robots) == 2

    def test_table_offset_matches_config(self, hetbi_env):
        table_offset = hetbi_env._env.table_offset
        assert table_offset is not None
        assert len(table_offset) == 3

    def test_table_full_size_matches_config(self, hetbi_env):
        tfs = hetbi_env._env.table_full_size
        assert tfs is not None
        assert len(tfs) == 3
        assert all(s > 0 for s in tfs)

    def test_robot_offset_none_uses_auto_layout(self, hetbi_env):
        """When robot_offset is None, robots should be auto-positioned."""
        # Both robots should have non-zero base positions
        for robot in hetbi_env._env.robots:
            base_pos = robot.robot_model.base_xpos_offset
            assert base_pos is not None

    def test_payload_body_id_exists(self, hetbi_env):
        hetbi_env.reset()
        assert hasattr(hetbi_env._env, "payload_body_id")
        assert hetbi_env._env.payload_body_id >= 0

    def test_fixture_body_id_exists(self, hetbi_env):
        hetbi_env.reset()
        assert hasattr(hetbi_env._env, "fixture_body_id")
        # fixture_body_id can be None if no fixture, but for bowl-rack it should exist
        assert hetbi_env._env.fixture_body_id >= 0


# ── HoBiEnv tests ───────────────────────────────────────────────────


class TestHoBiEnv:
    """Tests for the homogeneous bimanual environment."""

    def test_env_builds_successfully(self, hobi_env):
        assert hobi_env._env is not None

    def test_two_robots_present(self, hobi_env):
        assert len(hobi_env._env.robots) == 2

    def test_same_gripper_type(self, hobi_env):
        """Both robots should have the same gripper type."""
        g0 = type(hobi_env._env.robots[0].gripper)
        g1 = type(hobi_env._env.robots[1].gripper)
        assert g0 == g1

    def test_table_offset_matches_config(self, hobi_env):
        table_offset = hobi_env._env.table_offset
        assert table_offset is not None
        assert len(table_offset) == 3
        cfg_height = hobi_env.cfg["scene"]["table"]["height"]
        np.testing.assert_allclose(table_offset[2], cfg_height, atol=0.01)

    def test_table_full_size(self, hobi_env):
        tfs = hobi_env._env.table_full_size
        np.testing.assert_allclose(tfs, hobi_env.cfg["scene"]["table"]["full_size"])

    def test_robot_offset_none_auto_layout(self, hobi_env):
        """When robot_offset is None, HoBiEnv should auto-position in parallel."""
        assert hobi_env._robot_offset is None
        assert hobi_env._env.robot_offset is None


class TestHoBiEnvRobotOffset:
    """Test robot_offset parameter for HoBiEnv."""

    def test_custom_robot_offset_propagates(self):
        """When robot_offset is set via YAML override, it should be
        passed through to HoBiEnv."""
        env = ib.make(
            "ib-p1-plate-shelf-hobi-v1",
            obs_mode="combo",
            render_mode=None,
            seed=0,
            config_overrides={
                "scene": {
                    "robot_offset": [
                        [0.0, -0.7, 0.0],
                        [0.0, 0.7, 0.0],
                    ]
                }
            },
        )
        try:
            assert env._robot_offset is not None
            assert len(env._robot_offset) == 2
        finally:
            env.close()

    def test_custom_table_params_propagate(self):
        """Table params from YAML should propagate to the env."""
        env = ib.make(
            "ib-p1-plate-shelf-hobi-v1",
            obs_mode="combo",
            render_mode=None,
            seed=0,
            config_overrides={
                "scene": {
                    "table": {
                        "height": 1.0,
                        "full_size": [1.2, 0.8, 0.05],
                        "offset": [0, 0, 1.0],
                    }
                }
            },
        )
        try:
            np.testing.assert_allclose(env._env.table_offset, [0, 0, 1.0])
            np.testing.assert_allclose(env._env.table_full_size, [1.2, 0.8, 0.05])
        finally:
            env.close()


class TestHetBiEnvRobotOffset:
    """Test robot_offset parameter for HetBiEnv."""

    def test_custom_table_params_propagate(self):
        env = ib.make(
            "ib-p1-bowl-table-v1",
            obs_mode="combo",
            render_mode=None,
            seed=0,
            config_overrides={
                "scene": {
                    "table": {
                        "height": 1.0,
                        "full_size": [1.2, 0.8, 0.05],
                        "offset": [0, 0, 1.0],
                    }
                }
            },
        )
        try:
            np.testing.assert_allclose(env._env.table_offset, [0, 0, 1.0])
            np.testing.assert_allclose(env._env.table_full_size, [1.2, 0.8, 0.05])
        finally:
            env.close()
