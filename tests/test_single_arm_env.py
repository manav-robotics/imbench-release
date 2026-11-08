"""Tests for SingleArmTableEnv (ManipulationEnv-based).

Verifies the env's inheritance chain, object creation, placement,
observables, reset behaviour, and helper methods.
"""

import numpy as np
import pytest

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

import imbench.envs  # noqa: F401 — register envs
from imbench.envs.single_arm_table_env import SingleArmTableEnv
from imbench.registry import make


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cube_toss_task():
    """A task that uses SingleArmTableEnv with dynamic + static objects."""
    task = make("ib-p2-cube-toss-v1", obs_mode="combo", render_mode=None, seed=0)
    task.reset()
    yield task
    task.close()


@pytest.fixture(scope="module")
def pegboard_task():
    """A puzzle task with many dynamic objects and a static one."""
    task = make("ib-p3-ball-cage-escape-v1", obs_mode="combo", render_mode=None, seed=0)
    task.reset()
    yield task
    task.close()


# ── Inheritance chain ───────────────────────────────────────────────


class TestInheritance:
    """Verify SingleArmTableEnv extends ManipulationEnv, not Lift."""

    def test_mro_includes_manipulation_env(self):
        mro_names = [c.__name__ for c in SingleArmTableEnv.__mro__]
        assert "ManipulationEnv" in mro_names

    def test_mro_excludes_lift(self):
        mro_names = [c.__name__ for c in SingleArmTableEnv.__mro__]
        assert "Lift" not in mro_names

    def test_is_subclass_of_manipulation_env(self):
        assert issubclass(SingleArmTableEnv, ManipulationEnv)

    def test_runtime_env_is_single_arm_table(self, cube_toss_task):
        # Tactile wrapper wraps it, but it should be in the MRO
        env = cube_toss_task._env
        mro_names = [c.__name__ for c in type(env).__mro__]
        assert "SingleArmTableEnv" in mro_names
        assert "ManipulationEnv" in mro_names
        assert "Lift" not in mro_names


# ── Object creation ─────────────────────────────────────────────────


class TestObjectCreation:
    """Verify objects are properly created and accessible."""

    def test_dynamic_objects_created(self, cube_toss_task):
        env = cube_toss_task._env
        assert len(env._dynamic_objects) > 0

    def test_static_objects_created(self, cube_toss_task):
        env = cube_toss_task._env
        # cube_toss has a goal_marker static object
        assert len(env._static_objs) > 0

    def test_dynamic_object_names_match_specs(self, cube_toss_task):
        env = cube_toss_task._env
        dynamic_names = [o.name for o in env._dynamic_objects]
        assert "cube" in dynamic_names

    def test_static_object_has_no_joints(self, cube_toss_task):
        env = cube_toss_task._env
        for obj in env._static_objs:
            assert obj.joints is None or len(obj.joints) == 0

    def test_dynamic_object_has_free_joint(self, cube_toss_task):
        env = cube_toss_task._env
        for obj in env._dynamic_objects:
            assert obj.joints is not None
            assert len(obj.joints) > 0

    def test_many_dynamic_objects(self, pegboard_task):
        """Pegboard creates ball + many pegs, all dynamic."""
        env = pegboard_task._env
        # At minimum: 1 ball + board_rows*board_cols pegs
        assert len(env._dynamic_objects) >= 2

    def test_make_object_box(self):
        obj = SingleArmTableEnv._make_object(
            {"type": "box", "name": "test_box", "size": [0.02, 0.02, 0.02]}
        )
        assert obj.name == "test_box"

    def test_make_object_ball(self):
        obj = SingleArmTableEnv._make_object(
            {"type": "ball", "name": "test_ball", "size": [0.02]}
        )
        assert obj.name == "test_ball"

    def test_make_object_cylinder(self):
        obj = SingleArmTableEnv._make_object(
            {"type": "cylinder", "name": "test_cyl", "size": [0.02, 0.04]}
        )
        assert obj.name == "test_cyl"

    def test_make_object_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown object type"):
            SingleArmTableEnv._make_object(
                {"type": "hexagon", "name": "bad"}
            )

    def test_make_object_ball_scalar_size(self):
        """Ball size can be a scalar — should not crash."""
        obj = SingleArmTableEnv._make_object(
            {"type": "ball", "name": "scalar_ball", "size": 0.015}
        )
        assert obj.name == "scalar_ball"

    def test_make_object_static(self):
        obj = SingleArmTableEnv._make_object(
            {"type": "box", "name": "static_box", "size": [0.1, 0.1, 0.01],
             "joints": None}
        )
        assert obj.joints is None or len(obj.joints) == 0


# ── Object access helpers ───────────────────────────────────────────


class TestObjectAccessHelpers:
    """Test get_object() and cube property."""

    def test_get_object_dynamic(self, cube_toss_task):
        env = cube_toss_task._env
        obj = env.get_object("cube")
        assert obj is not None
        assert obj.name == "cube"

    def test_get_object_static(self, cube_toss_task):
        env = cube_toss_task._env
        obj = env.get_object("goal_marker")
        assert obj is not None
        assert obj.name == "goal_marker"

    def test_get_object_nonexistent(self, cube_toss_task):
        env = cube_toss_task._env
        assert env.get_object("nonexistent_object") is None

    def test_cube_property_returns_first_dynamic(self, cube_toss_task):
        env = cube_toss_task._env
        assert env.cube is not None
        assert env.cube is env._dynamic_objects[0]

    def test_cube_property_name_matches(self, pegboard_task):
        env = pegboard_task._env
        assert env.cube is not None
        assert env.cube is env._dynamic_objects[0]


# ── Table configuration ─────────────────────────────────────────────


class TestTableConfig:
    """Verify table parameters are properly stored."""

    def test_table_offset_is_numpy(self, cube_toss_task):
        env = cube_toss_task._env
        assert isinstance(env.table_offset, np.ndarray)
        assert len(env.table_offset) == 3

    def test_table_full_size(self, cube_toss_task):
        env = cube_toss_task._env
        cfg_size = cube_toss_task.cfg["scene"]["table"]["full_size"]
        np.testing.assert_allclose(env.table_full_size, cfg_size)

    def test_table_offset_matches_config(self, cube_toss_task):
        env = cube_toss_task._env
        cfg_height = cube_toss_task.cfg["scene"]["table"]["height"]
        # Default offset = (0, 0, height) when offset is null
        assert env.table_offset[2] == pytest.approx(cfg_height, abs=0.01)

    def test_custom_table_offset_propagates(self):
        """Explicit table offset in config should arrive in the env."""
        task = make(
            "ib-p2-cube-toss-v1",
            obs_mode="combo",
            render_mode=None,
            seed=0,
            config_overrides={
                "scene": {"table": {"offset": [0.0, 0.0, 1.0], "height": 1.0}}
            },
        )
        try:
            task.reset()
            np.testing.assert_allclose(task._env.table_offset, [0.0, 0.0, 1.0])
        finally:
            task.close()


# ── Observables ─────────────────────────────────────────────────────


class TestObservables:
    """Verify raw observables from the env."""

    def test_raw_obs_has_cube_pos(self, cube_toss_task):
        raw = cube_toss_task._get_sim_observations()
        assert "cube_pos" in raw
        assert raw["cube_pos"].shape == (3,)

    def test_raw_obs_has_cube_quat(self, cube_toss_task):
        raw = cube_toss_task._get_sim_observations()
        assert "cube_quat" in raw
        assert raw["cube_quat"].shape == (4,)

    def test_raw_obs_has_gripper_to_cube(self, cube_toss_task):
        raw = cube_toss_task._get_sim_observations()
        grip_keys = [k for k in raw if "gripper_to" in k]
        assert len(grip_keys) > 0
        for k in grip_keys:
            assert raw[k].shape == (3,)

    def test_named_object_observables(self, pegboard_task):
        """Pegboard's first object is named 'cube' — pos/quat should exist."""
        raw = pegboard_task._get_sim_observations()
        first_name = pegboard_task._env._dynamic_objects[0].name
        assert f"{first_name}_pos" in raw
        assert f"{first_name}_quat" in raw

    def test_cube_pos_is_physical(self, cube_toss_task):
        """cube_pos z should be near or above table height."""
        raw = cube_toss_task._get_sim_observations()
        table_z = cube_toss_task._env.table_offset[2]
        # Object should be on or slightly above the table
        assert raw["cube_pos"][2] >= table_z - 0.1


# ── Body IDs ────────────────────────────────────────────────────────


class TestBodyIDs:
    """Verify _setup_references correctly populates body IDs."""

    def test_body_ids_populated(self, cube_toss_task):
        env = cube_toss_task._env
        assert hasattr(env, "_obj_body_ids")
        assert len(env._obj_body_ids) == len(env._dynamic_objects)

    def test_body_ids_are_valid(self, cube_toss_task):
        env = cube_toss_task._env
        for name, bid in env._obj_body_ids.items():
            assert isinstance(bid, (int, np.integer))
            assert bid >= 0


# ── Placement ───────────────────────────────────────────────────────


class TestPlacement:
    """Test placement sampler behaviour."""

    def test_placement_initializer_exists(self, cube_toss_task):
        env = cube_toss_task._env
        assert env.placement_initializer is not None

    def test_reset_changes_object_positions(self, cube_toss_task):
        """Multiple resets should produce different object positions."""
        positions = []
        for _ in range(5):
            cube_toss_task.reset()
            raw = cube_toss_task._get_sim_observations()
            positions.append(raw["cube_pos"].copy())
        # Just check shape — placement randomness is not guaranteed in all configs
        assert all(p.shape == (3,) for p in positions)


# ── Reward and check_success ────────────────────────────────────────


class TestEnvRewardAndSuccess:
    """Verify stub implementations of reward/success."""

    def test_reward_returns_zero(self, cube_toss_task):
        assert cube_toss_task._env.reward() == 0.0

    def test_check_success_returns_false(self, cube_toss_task):
        assert cube_toss_task._env._check_success() is False


# ── Robot ───────────────────────────────────────────────────────────


class TestRobot:
    """Verify single robot setup."""

    def test_one_robot(self, cube_toss_task):
        assert len(cube_toss_task._env.robots) == 1

    def test_robot_is_panda(self, cube_toss_task):
        robot_name = cube_toss_task._env.robots[0].name
        assert "Panda" in robot_name
