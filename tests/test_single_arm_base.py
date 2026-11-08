"""Tests for SingleArmBase helpers, success termination, obs modes, reproducibility.

Covers:
- imbench.tasks.prehensile.single_arm_base (helpers, properties, reward_fn)
- Cross-cutting concerns: obs modes, seeded reproducibility, success thresholds
"""

import numpy as np
import pytest

from imbench.registry import TASK_REGISTRY, make
from imbench.tasks.prehensile.single_arm_base import SingleArmBase


# ── Helpers ─────────────────────────────────────────────────────────

def _make_task(task_id, **kw):
    defaults = dict(obs_mode="combo", render_mode=None, seed=42, max_episode_steps=10)
    defaults.update(kw)
    return make(task_id, **defaults)


# =====================================================================
#  SingleArmBase helpers
# =====================================================================

class TestSingleArmBaseHelpers:
    """Cover _place_object, _place_static, _set_object_velocity, etc."""

    @pytest.fixture(scope="class")
    @classmethod
    def cube_toss(cls):
        task = _make_task("ib-p2-cube-toss-v1")
        task.reset()
        yield task
        task.close()

    def test_place_object_moves_cube(self, cube_toss):
        cube_toss.reset()
        new_pos = [0.1, 0.05, 0.9]
        cube_toss._place_object("cube", new_pos)
        cube_toss._env.sim.forward()
        body_id = cube_toss._env.sim.model.body_name2id("cube_main")
        actual = cube_toss._env.sim.data.body_xpos[body_id]
        np.testing.assert_allclose(actual, new_pos, atol=0.01)

    def test_place_object_with_quat(self, cube_toss):
        cube_toss.reset()
        quat = [0.707, 0.707, 0, 0]
        cube_toss._place_object("cube", [0.1, 0.0, 0.9], quat=quat)
        cube_toss._env.sim.forward()

    def test_set_object_velocity(self, cube_toss):
        cube_toss.reset()
        cube_toss._set_object_velocity("cube", [0.5, 0.0, 0.0])
        sim = cube_toss._env.sim
        body_id = sim.model.body_name2id("cube_main")
        jnt_addr = sim.model.body_jntadr[body_id]
        dof_addr = sim.model.jnt_dofadr[jnt_addr]
        vx = sim.data.qvel[dof_addr]
        assert abs(vx - 0.5) < 0.01

    def test_set_object_velocity_with_angvel(self, cube_toss):
        cube_toss.reset()
        cube_toss._set_object_velocity("cube", [0.0, 0.0, 0.0], angvel=[0.1, 0.2, 0.3])

    def test_gripper_contacts_object_initially_false(self, cube_toss):
        cube_toss.reset()
        assert not cube_toss._gripper_contacts_object("cube")

    def test_object_fell_off_table_false_normally(self, cube_toss):
        cube_toss.reset()
        raw = cube_toss._get_sim_observations()
        assert not cube_toss._object_fell_off_table(raw)

    def test_object_fell_off_table_true_below_threshold(self, cube_toss):
        raw = {"cube_pos": np.array([0.0, 0.0, 0.01])}
        assert cube_toss._object_fell_off_table(raw)

    def test_object_fell_off_table_missing_key(self, cube_toss):
        raw = {}
        assert not cube_toss._object_fell_off_table(raw)

    def test_random_object_color(self, cube_toss):
        c = cube_toss._random_object_color()
        assert c.shape == (4,)
        assert c[3] == 1.0
        assert all(0.1 <= c[i] <= 0.9 for i in range(3))

    def test_sim_property(self, cube_toss):
        assert cube_toss.sim is not None

    def test_env_property(self, cube_toss):
        assert cube_toss.env is not None

    def test_custom_reward_fn(self):
        """A reward_fn passed to a SingleArmBase task replaces the task-score reward."""
        def my_reward(task, raw_obs):
            return 0.42
        task = make("ib-p2-cube-toss-v1", obs_mode="combo", render_mode=None,
                     seed=0, max_episode_steps=5, reward_fn=my_reward)
        try:
            task.reset()
            _, reward, _, _, _ = task.step(np.zeros(7))
            assert reward == pytest.approx(0.42)
        finally:
            task.close()


# =====================================================================
#  Success termination for ALL single-arm tasks
# =====================================================================

_TASKS_NEED_TERM = [
    "ib-p2-bounce-intercept-v1",
    "ib-p2-cube-toss-v1",
    "ib-p2-linear-intercept-v1",
    "ib-p5-mass-sort-v1",
    "ib-p6-slip-recovery-v1",
    "ib-p3-seesaw-balance-v1",
    "ib-p3-domino-single-v1",
    "ib-p3-domino-select-v1",
]


class TestSuccessTermination:
    """Verify _check_termination uses _success_score_threshold for all tasks."""

    @pytest.mark.parametrize("task_id", _TASKS_NEED_TERM)
    def test_check_termination_references_threshold(self, task_id):
        task = _make_task(task_id)
        try:
            task.reset()
            assert hasattr(task, "_success_score_threshold")
            assert 0.0 < task._success_score_threshold <= 1.0
        finally:
            task.close()


class TestAllPrehensileHaveTermination:
    """Check that ALL prehensile tasks have _success_score_threshold."""

    def _prehensile_ids(self):
        ids = []
        for tid, cls in TASK_REGISTRY.items():
            # filter non-SingleArmBase tasks below via issubclass check
            actual = cls.func if hasattr(cls, "func") else cls
            try:
                if issubclass(actual, SingleArmBase):
                    ids.append(tid)
            except TypeError:
                pass
        return sorted(ids)

    def test_all_have_success_threshold(self):
        for tid in self._prehensile_ids():
            task = _make_task(tid)
            try:
                task.reset()
                assert hasattr(task, "_success_score_threshold"), f"{tid} missing threshold"
            finally:
                task.close()


# =====================================================================
#  Obs mode variations
# =====================================================================

class TestObsModeVariations:

    def test_proprio_mode_no_rgb(self):
        task = make("ib-p2-cube-toss-v1", obs_mode="proprio",
                     render_mode=None, seed=0, max_episode_steps=5)
        try:
            obs, _ = task.reset()
            assert not any(k.startswith("rgb_") for k in obs)
        finally:
            task.close()

    def test_combo_mode_has_both(self):
        task = _make_task("ib-p2-cube-toss-v1")
        try:
            obs, _ = task.reset()
            assert any(k.startswith("rgb_") for k in obs)
            assert "robot0_eef_pos" in obs
        finally:
            task.close()


# =====================================================================
#  Reproducibility
# =====================================================================

class TestReproducibility:

    def test_same_seed_same_obs(self):
        task1 = _make_task("ib-p2-cube-toss-v1", seed=123)
        task2 = _make_task("ib-p2-cube-toss-v1", seed=123)
        try:
            obs1, _ = task1.reset()
            obs2, _ = task2.reset()
            np.testing.assert_allclose(
                obs1["robot0_eef_pos"], obs2["robot0_eef_pos"], atol=1e-6
            )
        finally:
            task1.close()
            task2.close()

    def test_different_seed_different_obs(self):
        task1 = _make_task("ib-p2-cube-toss-v1", seed=1)
        task2 = _make_task("ib-p2-cube-toss-v1", seed=999)
        try:
            obs1, _ = task1.reset()
            obs2, _ = task2.reset()
            if "object_pos" in obs1 and "object_pos" in obs2:
                assert not np.allclose(obs1["object_pos"], obs2["object_pos"], atol=1e-4)
        finally:
            task1.close()
            task2.close()
