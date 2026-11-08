"""Tests for the custom freeform task.

Covers:
- imbench.tasks.custom.freeform_v1 (_object_spec, scene construction,
  score, termination, unsupported options)
"""

import numpy as np
import pytest

from imbench.registry import make
from imbench.tasks.custom.freeform_v1 import _object_spec

_MYOBJ = {"type": "box", "name": "myobj", "size": [0.02, 0.02, 0.02], "pos": [0.0, 0.0, 0.82]}


# =====================================================================
#  Freeform task
# =====================================================================

class TestFreeformTask:

    def test_instantiation_minimal(self):
        """Freeform with no objects/goals, single Panda."""
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5)
        try:
            obs, info = task.reset()
            assert isinstance(obs, dict)
        finally:
            task.close()

    def test_with_objects_and_goals(self):
        """Freeform with box objects and distance-based goals."""
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5,
                     objects=[
                         {"type": "box", "name": "red_cube",
                          "size": [0.02, 0.02, 0.02],
                          "pos": [0.0, 0.0, 0.82], "rgba": [1, 0, 0, 1]},
                     ],
                     goals=[
                         {"object": "red_cube", "target_pos": [0.3, 0.0, 0.82],
                          "threshold": 0.05, "weight": 1.0},
                     ])
        try:
            obs, info = task.reset()
            assert isinstance(obs, dict)
        finally:
            task.close()

    def test_score_no_goals(self):
        """Score should be 0.0 with no goals."""
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5)
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert score == 0.0
        finally:
            task.close()

    def test_score_with_goals_key_missing(self):
        """Score with goals but missing pos key in raw_obs."""
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5,
                     objects=[_MYOBJ],
                     goals=[{"object": "myobj", "target_pos": [0, 0, 0],
                             "threshold": 0.05, "weight": 1.0}])
        try:
            task.reset()
            score = task._compute_task_score({})
            assert score == 0.0
        finally:
            task.close()

    def test_score_with_pos_present(self):
        """Score when object position is in raw_obs."""
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5,
                     objects=[_MYOBJ],
                     goals=[{"object": "myobj", "target_pos": [0.3, 0.0, 0.82],
                             "threshold": 0.05, "weight": 1.0}])
        try:
            task.reset()
            raw = {"myobj_pos": np.array([0.3, 0.0, 0.82])}
            score = task._compute_task_score(raw)
            assert score >= 0.9
        finally:
            task.close()

    def test_termination_on_high_score(self):
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5,
                     objects=[_MYOBJ],
                     goals=[{"object": "myobj", "target_pos": [0.3, 0.0, 0.82],
                             "threshold": 0.05, "weight": 1.0}])
        try:
            task.reset()
            raw = {"myobj_pos": np.array([0.3, 0.0, 0.82])}
            assert task._check_termination(raw)
        finally:
            task.close()

    def test_step(self):
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5)
        try:
            task.reset()
            lo, _ = task.action_space.low, task.action_space.high
            obs, r, t, tr, info = task.step(np.zeros(lo.shape[0]))
            assert isinstance(obs, dict)
        finally:
            task.close()

    def test_get_obs(self):
        task = make("ib-m-freeform-v1",
                     obs_mode="combo", render_mode=None, seed=42,
                     max_episode_steps=5)
        try:
            task.reset()
            raw = task._get_sim_observations()
            obs = task._get_obs(raw)
            assert isinstance(obs, dict)
            assert any(k.startswith("robot0") for k in obs)
        finally:
            task.close()

    def test_object_spec_helpers(self):
        box = _object_spec({"type": "box", "name": "test", "size": [0.02, 0.02, 0.02],
                            "mass": 0.064})
        assert box["type"] == "box"
        assert box["density"] == pytest.approx(1000.0)
        cyl = _object_spec({"type": "cylinder", "name": "test2", "size": [0.01, 0.03]})
        assert cyl["type"] == "cylinder"
        sph = _object_spec({"type": "sphere", "name": "test3", "size": [0.02]})
        assert sph["type"] == "ball"

    def test_object_spec_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown object type"):
            _object_spec({"type": "alien", "name": "x"})

    def test_object_spec_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="scale"):
            _object_spec({"type": "box", "name": "x", "size": [0.02] * 3, "scale": 2})

    def test_object_spec_rejects_quat_without_pos(self):
        with pytest.raises(ValueError, match="quat"):
            _object_spec({"type": "box", "name": "x", "size": [0.02] * 3,
                          "quat": [0.7071, 0, 0, 0.7071]})


class TestFreeformScene:
    """Configured objects, robots and grippers end up in the simulation."""

    def test_objects_exist_and_goal_is_scored(self):
        task = make("ib-m-freeform-v1", obs_mode="proprio", render_mode=None, seed=0,
                    max_episode_steps=20,
                    objects=[
                        {"type": "box", "name": "red_cube", "size": [0.02, 0.02, 0.02],
                         "pos": [0.0, 0.0, 0.82], "mass": 0.05},
                        {"type": "sphere", "name": "ball", "size": [0.02]},
                    ],
                    goals=[{"object": "red_cube", "target_pos": [0.2, 0.0, 0.82],
                            "threshold": 0.05, "weight": 1.0}])
        try:
            task.reset(seed=0)
            sim = task._env.sim
            cube_bid = sim.model.body_name2id("red_cube_main")
            np.testing.assert_allclose(sim.data.body_xpos[cube_bid], [0.0, 0.0, 0.82], atol=1e-6)
            assert sim.model.body_mass[cube_bid] == pytest.approx(0.05)
            ball_xy = sim.data.body_xpos[sim.model.body_name2id("ball_main")][:2]
            assert np.linalg.norm(ball_xy) > 0.04, "sampled object overlaps the fixed one"

            zero = np.zeros(task.action_space.shape[0])
            _, _, term, _, info = task.step(zero)
            assert not info["is_success"] and not term

            obj = task._env.get_object("red_cube")
            sim.data.set_joint_qpos(obj.joints[0], np.r_[0.2, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0])
            sim.forward()
            assert task.task_score(recompute=True) == pytest.approx(1.0)
            _, _, term, _, info = task.step(zero)
            assert info["is_success"] and term
        finally:
            task.close()

    def test_goal_shaping_tracks_current_distance_from_reset(self):
        """An idle scene scores 0, and moving the object away again loses the credit."""
        task = make("ib-m-freeform-v1", obs_mode="proprio", render_mode=None, seed=0,
                    max_episode_steps=20, objects=[_MYOBJ],
                    goals=[{"object": "myobj", "target_pos": [0.2, 0.0, 0.82],
                            "threshold": 0.05, "weight": 1.0}])
        try:
            task.reset(seed=0)
            zero = np.zeros(task.action_space.shape[0])
            _, _, _, _, info = task.step(zero)
            assert info["task_score"] == pytest.approx(0.0, abs=1e-3)
            sim = task._env.sim
            joint = task._env.get_object("myobj").joints[0]
            sim.data.set_joint_qpos(joint, np.r_[0.1, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0])
            sim.forward()
            assert task.task_score(recompute=True) > 0.3
            sim.data.set_joint_qpos(joint, np.r_[0.0, 0.0, 0.82, 1.0, 0.0, 0.0, 0.0])
            sim.forward()
            assert task.task_score(recompute=True) == pytest.approx(0.0, abs=1e-3)
        finally:
            task.close()

    def test_default_gripper_is_robot_gripper(self):
        task = make("ib-m-freeform-v1", obs_mode="proprio", render_mode=None, seed=0)
        try:
            assert task.action_space.shape == (7,)
            assert type(task._env.robots[0].gripper["right"]).__name__ == "PandaGripper"
        finally:
            task.close()

    def test_second_robot_placed_at_pos(self):
        task = make("ib-m-freeform-v1", obs_mode="proprio", render_mode=None, seed=0,
                    robots=[{"name": "Panda", "pos": [-0.6, -0.35, 0.0]},
                            {"name": "Panda", "pos": [-0.6, 0.35, 0.0]}])
        try:
            obs, _ = task.reset(seed=0)
            sim = task._env.sim
            for i, y in enumerate((-0.35, 0.35)):
                base = sim.data.body_xpos[sim.model.body_name2id(f"robot{i}_base")]
                np.testing.assert_allclose(base[:2], [-0.6, y], atol=1e-6)
            assert task.action_space.shape == (14,)
            assert "robot1_eef_pos" in obs
        finally:
            task.close()

    @pytest.mark.parametrize("kwargs, exc", [
        (dict(objects=[{"type": "mesh", "name": "mug", "mesh_file": "/tmp/mug.stl"}]),
         NotImplementedError),
        (dict(arena_xml="/tmp/arena.xml"), NotImplementedError),
        (dict(objects=[{"type": "box", "name": "a", "size": [0.02] * 3}],
              goals=[{"object": "a", "target_pos": [0, 0, 0.82], "weight": 0.0}]), ValueError),
        (dict(goals=[{"object": "ghost", "target_pos": [0, 0, 0.82]}]), ValueError),
        (dict(robots=[{"name": "Panda"}, {"name": "Panda"}]), ValueError),
    ])
    def test_unsupported_options_raise(self, kwargs, exc):
        with pytest.raises(exc):
            make("ib-m-freeform-v1", obs_mode="proprio", render_mode=None, seed=0, **kwargs)
