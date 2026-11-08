"""Tests for imbench.envs.hetbi_cup_inversion_env.HetBiCupInversionEnv.

Exercised via the ``ib-p1-cup-inversion-v1`` task so the full env
construction path (including _load_model → _setup_references →
_reset_internal) is covered without duplicating robosuite boilerplate.

Skipped automatically when the YCB dataset isn't present on disk — the
hetbi tasks load YCB meshes for the payload, so without them the env
cannot be constructed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imbench.objects import YCB_DATASET_DIR
from imbench.registry import make

_YCB_AVAILABLE = Path(YCB_DATASET_DIR).exists() and any(Path(YCB_DATASET_DIR).iterdir())

pytestmark = pytest.mark.skipif(
    not _YCB_AVAILABLE,
    reason="YCB dataset not present at deps/ycb/ycb — required by hetbi tasks",
)


@pytest.fixture(scope="module")
def cup_task():
    task = make(
        "ib-p1-cup-inversion-v1",
        obs_mode="proprio",
        render_mode=None,
        seed=0,
        max_episode_steps=3,
    )
    task.reset()
    yield task
    task.close()


class TestHetBiCupInversionEnv:

    def test_goal_marker_created(self, cup_task):
        env = cup_task._env
        # The task config provides goal_pos → the env should have a marker.
        assert env._goal_marker is not None, \
            "Cup-inversion task config should set goal_pos, creating a marker"

    def test_goal_body_id_registered(self, cup_task):
        env = cup_task._env
        assert env._goal_body_id is not None
        # Should be a valid body id in the sim.
        assert env._goal_body_id >= 0
        assert env._goal_body_id < env.sim.model.nbody

    def test_goal_half_size_tuple(self, cup_task):
        env = cup_task._env
        assert isinstance(env._goal_half_size, tuple)
        assert len(env._goal_half_size) == 2
        assert all(x > 0 for x in env._goal_half_size)

    def test_goal_rgba_length(self, cup_task):
        env = cup_task._env
        assert len(env._goal_rgba) == 4

    def test_in_goal_region_center(self, cup_task):
        env = cup_task._env
        gp = env._goal_pos_cfg
        assert env.in_goal_region(np.array([gp[0], gp[1]])) is True

    def test_in_goal_region_edge_inside(self, cup_task):
        env = cup_task._env
        gp = env._goal_pos_cfg
        hx, hy = env._goal_half_size
        # Just inside the edge.
        assert env.in_goal_region(np.array([gp[0] + hx - 1e-4, gp[1]])) is True

    def test_in_goal_region_outside(self, cup_task):
        env = cup_task._env
        gp = env._goal_pos_cfg
        hx, _ = env._goal_half_size
        # Outside the x bound.
        assert env.in_goal_region(np.array([gp[0] + hx + 0.05, gp[1]])) is False

    def test_marker_body_placed_at_configured_xy(self, cup_task):
        """After reset, the marker body_pos should match goal_pos x,y."""
        cup_task.reset()
        env = cup_task._env
        body_pos = env.sim.model.body_pos[env._goal_body_id]
        gp = env._goal_pos_cfg
        np.testing.assert_allclose(body_pos[:2], gp[:2], atol=1e-6)

    def test_marker_quat_identity(self, cup_task):
        cup_task.reset()
        env = cup_task._env
        body_quat = env.sim.model.body_quat[env._goal_body_id]
        np.testing.assert_allclose(body_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_marker_is_visual_only(self, cup_task):
        """The goal marker must not collide with the cup or the robots."""
        model = cup_task._env.sim.model
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == cup_task._env._goal_body_id]
        assert geoms
        for g in geoms:
            assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0

    def test_task_goal_check_matches_env(self, cup_task):
        gp = cup_task._env._goal_pos_cfg
        hx, _ = cup_task._env._goal_half_size
        for xy in ([gp[0], gp[1]], [gp[0] + hx + 0.05, gp[1]]):
            pos = np.array([xy[0], xy[1], 1.0])
            assert cup_task._payload_in_goal_region(pos) == cup_task._env.in_goal_region(pos[:2])


class TestInGoalRegionWithoutMarker:
    """Check the in_goal_region fast-path when no goal was configured."""

    def test_unconstrained_when_goal_pos_cfg_none(self, cup_task):
        env = cup_task._env
        # Temporarily clear the goal config (restore after).
        original = env._goal_pos_cfg
        env._goal_pos_cfg = None
        try:
            assert env.in_goal_region(np.array([0.0, 0.0])) is True
            assert cup_task._payload_in_goal_region(np.array([0.0, 0.0, 1.0])) is True
        finally:
            env._goal_pos_cfg = original
