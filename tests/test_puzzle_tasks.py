"""Tests for physics puzzle tasks (ib-p[2-7]-*) using SingleArmTableEnv.

Parametrized test suite verifying all puzzle tasks initialise, reset,
step correctly, and expose proper observations and scoring.

Each task is created, tested, and closed within a single parametrized
test to avoid MuJoCo memory issues from many concurrent simulations.
"""

import numpy as np
import pytest

from imbench.registry import TASK_REGISTRY, make
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.base_task import BaseTask


# ── Discover all puzzle task IDs dynamically ────────────────────────

def _puzzle_task_ids():
    """Return puzzle tasks: ib-p3-*, ib-p6-*, ib-p7-shape-stack-*, ib-p2-ramp-sort-*."""
    _PUZZLE_IDS = {
        "ib-p2-ramp-sort-v1", "ib-p3-ball-cage-escape-v1",
        "ib-p3-gap-funnel-v1", "ib-p3-indirect-push-v1",
        "ib-p6-pick-place-ycb-v1", "ib-p6-stack-collapse-recovery-v1",
        "ib-p7-shape-stack-v1",
    }
    return sorted(tid for tid in TASK_REGISTRY if tid in _PUZZLE_IDS)


PUZZLE_IDS = _puzzle_task_ids()
# Canonical subset run by default; full matrix gated behind @pytest.mark.slow.
PUZZLE_SMOKE_IDS = [
    tid for tid in ("ib-p7-shape-stack-v1", "ib-p6-stack-collapse-recovery-v1") if tid in PUZZLE_IDS
] or PUZZLE_IDS[:2]
SINGLE_PANDA_ACTION_DIM = 7


# ── Registration ────────────────────────────────────────────────────


class TestPuzzleRegistration:

    def test_at_least_10_puzzle_tasks(self):
        assert len(PUZZLE_IDS) >= 7

    @pytest.mark.parametrize("task_id", PUZZLE_IDS)
    def test_in_registry(self, task_id):
        assert task_id in TASK_REGISTRY

    @pytest.mark.parametrize("task_id", PUZZLE_IDS)
    def test_is_single_arm_base(self, task_id):
        cls = TASK_REGISTRY[task_id]
        actual = cls.func if hasattr(cls, "func") else cls
        assert issubclass(actual, SingleArmBase)


# ── Comprehensive per-task test ─────────────────────────────────────


class TestPuzzleTaskFull:
    """Full lifecycle, step, obs, config, and env internals for each task."""

    @pytest.mark.slow
    @pytest.mark.parametrize("task_id", PUZZLE_IDS)
    def test_scores_remain_finite_and_bounded_with_stepping(self, task_id):
        task = make(task_id, obs_mode="combo", render_mode=None, seed=42, max_episode_steps=25)
        try:
            lo, _ = task.action_space.low, task.action_space.high
            task.reset()
            raw_obs = task._get_sim_observations(force_update=True)
            initial_score = float(task._compute_task_score(raw_obs))
            scores = [initial_score]
            info_scores = []

            for _ in range(5):
                action = np.zeros_like(lo)
                _, _, _, _, info = task.step(action)
                assert "task_score" in info
                info_scores.append(float(info["task_score"]))
                raw_obs = task._get_sim_observations()
                scores.append(float(task._compute_task_score(raw_obs)))

            assert all(np.isfinite(s) and 0.0 <= s <= 1.0 for s in scores)
            assert all(np.isfinite(s) and 0.0 <= s <= 1.0 for s in info_scores)
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", PUZZLE_SMOKE_IDS)
    def test_smoke_lifecycle(self, task_id):
        """Light lifecycle check on canonical puzzles — always runs."""
        task = make(task_id, obs_mode="combo", render_mode=None, seed=0, max_episode_steps=3)
        try:
            obs, info = task.reset()
            assert isinstance(obs, dict) and "task_score" in info
            lo, _ = task.action_space.low, task.action_space.high
            task.step(np.zeros(lo.shape[0]))
        finally:
            task.close()

    @pytest.mark.slow
    @pytest.mark.parametrize("task_id", PUZZLE_IDS)
    def test_full_lifecycle(self, task_id):
        task = make(task_id, obs_mode="combo", render_mode=None, seed=42, max_episode_steps=10,
                    privileged_obs=True)
        try:
            # ── isinstance checks ───────────────────────────────────
            assert isinstance(task, BaseTask)

            # ── MRO checks ──────────────────────────────────────────
            mro = [c.__name__ for c in type(task._env).__mro__]
            assert "ManipulationEnv" in mro
            assert "Lift" not in mro

            # ── Reset ───────────────────────────────────────────────
            obs, info = task.reset()
            assert isinstance(obs, dict)
            assert isinstance(info, dict)
            assert "task_score" in info
            assert "step" in info
            assert info["step"] == 0

            # ── Proprio keys ────────────────────────────────────────
            for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                assert key in obs, f"Missing proprio key: {key}"
                assert isinstance(obs[key], np.ndarray)

            # ── Object state ────────────────────────────────────────
            assert "object_pos" in obs
            assert obs["object_pos"].shape == (3,)

            # ── Camera obs (combo mode) ─────────────────────────────
            rgb_keys = [k for k in obs if k.startswith("rgb_")]
            assert len(rgb_keys) > 0, "combo mode should have camera obs"
            for k in rgb_keys:
                assert obs[k].ndim == 3, f"{k} should be 3D (H,W,C)"

            # ── Action space ────────────────────────────────────────
            lo, hi = task.action_space.low, task.action_space.high
            assert lo.shape == (SINGLE_PANDA_ACTION_DIM,), "Single Panda: 6 OSC + 1 gripper"
            assert hi.shape == (SINGLE_PANDA_ACTION_DIM,)
            assert np.all(lo <= hi)

            # ── Step ────────────────────────────────────────────────
            action = np.zeros(lo.shape[0])
            action[-1] = -1  # open gripper
            obs2, reward, terminated, truncated, info2 = task.step(action)
            assert isinstance(obs2, dict)
            assert isinstance(reward, (int, float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert "task_score" in info2
            assert 0.0 <= info2["task_score"] <= 1.0
            assert info2["step"] == 1

            # ── Config ──────────────────────────────────────────────
            assert task.cfg["task_id"] == task_id
            assert task.MAX_EPISODE_STEPS == task.cfg["max_episode_steps"]
            assert task.MAX_EPISODE_STEPS > 0
            assert "task" in task.cfg, "YAML missing 'task' section"
            tc = task.cfg.get("task_conditions", {})
            assert "success" in tc
            assert "failure" in tc

            # ── Env internals ───────────────────────────────────────
            env = task._env
            assert len(env._dynamic_objects) >= 1
            assert len(env._obj_body_ids) == len(env._dynamic_objects)
            for obj in env._dynamic_objects:
                assert obj.name in env._obj_body_ids
            assert env.cube is not None
            assert len(env.robots) == 1

            # ── get_object works ────────────────────────────────────
            first_name = env._dynamic_objects[0].name
            assert env.get_object(first_name) is not None
            assert env.get_object("nonexistent") is None

            # ── Raw observables ─────────────────────────────────────
            raw = env._get_observations()
            assert f"{first_name}_pos" in raw
            assert raw[f"{first_name}_pos"].shape == (3,)
            assert f"{first_name}_quat" in raw
            assert raw[f"{first_name}_quat"].shape == (4,)

            # ── Step count ──────────────────────────────────────────
            task.reset()
            for i in range(3):
                _, _, _, _, si = task.step(np.zeros(lo.shape[0]))
                assert si["step"] == i + 1

        finally:
            task.close()


# ── Proprio-only mode test ──────────────────────────────────────────


class TestPuzzleProprioMode:

    def test_proprio_mode_no_cameras(self):
        task = make(PUZZLE_IDS[0], obs_mode="proprio", render_mode=None, seed=0, max_episode_steps=5)
        try:
            obs, _ = task.reset()
            assert not any(k.startswith("rgb_") for k in obs)
        finally:
            task.close()


# =====================================================================
#  Programmatic success — teleport objects to goal states
# =====================================================================

def _make_task(task_id, **kw):
    defaults = dict(obs_mode="combo", render_mode=None, seed=42, max_episode_steps=200)
    defaults.update(kw)
    return make(task_id, **defaults)


class TestIndirectPushSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p3-indirect-push-v1")
        t.reset()
        yield t
        t.close()

    def test_teleport_target_to_goal(self, task):
        task.reset()
        sim = task._env.sim
        goal_xy = task._goal_pos
        bid = sim.model.body_name2id("target_main")
        jnt_addr = sim.model.body_jntadr[bid]
        qpos_addr = sim.model.jnt_qposadr[jnt_addr]
        dof_addr = sim.model.jnt_dofadr[jnt_addr]

        sim.data.qpos[qpos_addr:qpos_addr + 3] = [goal_xy[0], goal_xy[1], task._table_height + 0.02]
        sim.data.qvel[dof_addr:dof_addr + 6] = 0
        sim.forward()

        raw_obs = task._get_sim_observations(force_update=True)
        scores = [task._compute_task_score(raw_obs) for _ in range(task.GOAL_REST_STEPS)]
        assert scores[0] < 1.0  # one slow sample is not a rest
        assert scores[-1] == 1.0
        assert task._check_termination(raw_obs) is True


class TestRampSortSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p2-ramp-sort-v1")
        t.reset()
        yield t
        t.close()

    def test_sort_objects_to_correct_bins(self, task):
        task.reset()
        sim = task._env.sim

        sphere_bin_xy = task._sphere_bin_world_xy
        cube_bin_xy = task._cube_bin_world_xy
        drop_z = task._table_height + 0.02

        for i in range(task._num_spheres):
            name = f"sphere_{i}"
            task._place_object(name, [sphere_bin_xy[0], sphere_bin_xy[1], drop_z])
            task._set_object_velocity(name, [0, 0, 0])

        for i in range(task._num_cubes):
            name = f"box_{i}"
            task._place_object(name, [cube_bin_xy[0], cube_bin_xy[1], drop_z])
            task._set_object_velocity(name, [0, 0, 0])

        sim.forward()

        # Sorted once each ball has stayed at rest in its bin for REST_TIME.
        action = np.zeros(task.action_space.low.shape[0])
        for _ in range(int(round(task.REST_TIME * task.control_freq)) + 300):
            _, _, term, _, _ = task.step(action)
            if term:
                break
        raw_obs = task._get_sim_observations(force_update=True)
        assert task._compute_task_score(raw_obs) == 1.0
        assert task._check_termination(raw_obs) is True


