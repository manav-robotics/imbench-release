"""Tests for hard manipulation tasks (ib-p7-balance-*).

Covers:
- imbench.tasks.hard.medium_base_balance_v1
  - Platform physics (ridge exists, rod falls when unsupported)
  - Scoring (upright rod scores high, tilted rod scores low)
  - Termination (sustained balance terminates, fall terminates)
"""

import numpy as np
import pytest

from imbench.registry import TASK_REGISTRY, make
from imbench.tasks.prehensile.contacts import robot_contacts


# ── Helpers ─────────────────────────────────────────────────────────

def _make_task(task_id, **kw):
    defaults = dict(obs_mode="combo", render_mode=None, seed=42, privileged_obs=True,
                    max_episode_steps=10)
    defaults.update(kw)
    return make(task_id, **defaults)


def _platform_clear_of_robot(task):
    """Move the platform where an upright rod cannot touch the parked gripper; return its pos."""
    sim = task._env.sim
    quat = sim.data.body_xquat[sim.model.body_name2id("platform_main")].copy()
    task._place_static("platform", [0.15, 0.0, task._table_height], quat=quat)
    sim.forward()
    return sim.data.body_xpos[sim.model.body_name2id("platform_main")].copy()


def _step_n(task, n=5, action=None):
    lo, _ = task.action_space.low, task.action_space.high
    if action is None:
        action = np.zeros(lo.shape[0])
    for _ in range(n):
        task.step(action)


# =====================================================================
#  Registration
# =====================================================================

class TestHardRegistration:

    def test_medium_balance_in_registry(self):
        assert "ib-p7-balance-medium-v1" in TASK_REGISTRY


# =====================================================================
#  Platform physics
# =====================================================================

class TestBalancePlatformPhysics:
    """Verify the balance platform geometry is correctly created."""

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p7-balance-medium-v1", max_episode_steps=200)
        t.reset()
        yield t
        t.close()

    def test_platform_body_exists(self, task):
        sim = task._env.sim
        body_names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
        assert "platform_main" in body_names

    def test_rod_body_exists(self, task):
        sim = task._env.sim
        body_names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
        assert "rod_main" in body_names

    def test_rod_has_free_joint(self, task):
        """The rod must have a free joint so it can move/tilt."""
        sim = task._env.sim
        rod_id = sim.model.body_name2id("rod_main")
        jnt_addr = sim.model.body_jntadr[rod_id]
        assert jnt_addr >= 0, "Rod should have a joint"

    def test_ridge_geom_exists(self, task):
        sim = task._env.sim
        geom_names = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
                      if sim.model.geom_id2name(i)]
        ridge_geoms = [n for n in geom_names if "ridge" in n]
        assert len(ridge_geoms) >= 1, "Should have at least one ridge geom"

    def test_rod_falls_when_unsupported(self, task):
        """If the rod starts in mid-air away from platform, it should fall."""
        task.reset()
        sim = task._env.sim

        # Place rod high in the air away from platform
        task._place_object("rod", [0.0, 0.3, task._table_height + 0.3])
        sim.forward()

        # Step to let physics take effect
        _step_n(task, n=50)

        rod_id = sim.model.body_name2id("rod_main")
        rod_z = sim.data.body_xpos[rod_id][2]
        # Rod should have fallen significantly
        assert rod_z < task._table_height + 0.2, (
            f"Rod should fall but z={rod_z:.3f}")


# =====================================================================
#  Observations
# =====================================================================

class TestMediumBalanceObservations:
    """Verify task-specific observation keys are present and well-shaped."""

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p7-balance-medium-v1", max_episode_steps=20)
        t.reset()
        yield t
        t.close()

    def test_obs_keys_after_reset(self, task):
        obs, _ = task.reset()
        for key in ("object_pos", "object_quat", "platform_pos",
                     "rod_tilt_deg", "rod_axis", "rod_to_platform"):
            assert key in obs, f"Missing obs key: {key}"

    def test_object_pos_shape(self, task):
        obs, _ = task.reset()
        assert obs["object_pos"].shape == (3,)

    def test_object_quat_shape(self, task):
        obs, _ = task.reset()
        assert obs["object_quat"].shape == (4,)

    def test_platform_pos_shape(self, task):
        obs, _ = task.reset()
        assert obs["platform_pos"].shape == (3,)

    def test_rod_tilt_deg_shape(self, task):
        obs, _ = task.reset()
        assert obs["rod_tilt_deg"].shape == (1,)
        assert obs["rod_tilt_deg"][0] >= 0.0

    def test_rod_axis_shape(self, task):
        obs, _ = task.reset()
        assert obs["rod_axis"].shape == (3,)
        # Should be approximately unit length
        assert np.linalg.norm(obs["rod_axis"]) == pytest.approx(1.0, abs=0.01)

    def test_rod_to_platform_shape(self, task):
        obs, _ = task.reset()
        assert obs["rod_to_platform"].shape == (1,)
        assert obs["rod_to_platform"][0] >= 0.0

    def test_obs_keys_after_step(self, task):
        task.reset()
        action = np.zeros(task.action_space.low.shape[0])
        obs, _, _, _, _ = task.step(action)
        for key in ("object_pos", "object_quat", "platform_pos",
                     "rod_tilt_deg", "rod_axis", "rod_to_platform"):
            assert key in obs, f"Missing obs key after step: {key}"

    def test_camera_obs_present(self, task):
        """combo mode should include camera observations."""
        obs, _ = task.reset()
        rgb_keys = [k for k in obs if k.startswith("rgb_")]
        assert len(rgb_keys) > 0, "combo mode should have camera obs"


# =====================================================================
#  Scoring
# =====================================================================

class TestMediumBalanceScoring:
    """Verify scoring logic for the medium-base balance task."""

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p7-balance-medium-v1", max_episode_steps=200)
        t.reset()
        yield t
        t.close()

    def test_rod_lying_flat_scores_low(self, task):
        """A rod lying flat on the table should score low."""
        task.reset()
        sim = task._env.sim

        # Rod lying flat (90° about Y so long axis along world X)
        rod_z = task._table_height + task._rod_radius + 0.005
        quat_flat = [0.7071068, 0.0, 0.7071068, 0.0]
        task._place_object("rod", [0.10, 0.0, rod_z], quat=quat_flat)
        task._set_object_velocity("rod", [0, 0, 0], [0, 0, 0])
        sim.forward()

        _step_n(task, n=10)
        raw_obs = task._get_sim_observations(force_update=True)
        score = task._compute_task_score(raw_obs)
        tilt = task._get_rod_tilt_deg()
        # Rod lying flat → tilt ≈ 90° → score should be very low
        assert tilt > 45.0, f"Rod should be tilted but tilt={tilt:.1f}°"
        assert score < 0.5, f"Score should be low for flat rod but score={score:.2f}"

    def test_rod_upright_on_platform_scores_high(self, task):
        """Teleport the rod upright onto the platform ridge — should score ≥ 0.8."""
        task.reset()
        sim = task._env.sim
        plat_pos = _platform_clear_of_robot(task)

        # Rod sits on the ridge (no airborne gap — "balanced" requires contact).
        bz = task._base_half_size[2]
        ridge_top = plat_pos[2] + bz * 2 + task._ridge_height
        rod_z = ridge_top + task._rod_half_length

        # MuJoCo cylinder's long axis is local Z, so identity quat
        # [1,0,0,0] places the rod upright along world Z.
        task._place_object("rod", [plat_pos[0], plat_pos[1], rod_z],
                           quat=[1, 0, 0, 0])
        task._set_object_velocity("rod", [0, 0, 0], [0, 0, 0])
        sim.forward()

        # A couple of steps so MuJoCo registers rod/ridge contact before
        # we query the balance predicate.
        _step_n(task, n=2)

        raw_obs = task._get_sim_observations(force_update=True)
        score = task._compute_task_score(raw_obs)
        tilt = task._get_rod_tilt_deg()
        assert tilt < task._tilt_threshold_deg, (
            f"Rod should be upright but tilt={tilt:.1f}°")
        assert score >= 0.8, f"Upright rod on platform should score ≥ 0.8 but got {score:.2f}"


# =====================================================================
#  Sustained balance and termination
# =====================================================================

class TestMediumBalanceTermination:
    """Verify termination conditions for the medium-base balance task."""

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p7-balance-medium-v1", max_episode_steps=200)
        t.reset()
        yield t
        t.close()

    def test_sustained_balance_terminates(self, task):
        """If we hold the rod perfectly upright for enough steps, task terminates."""
        task.reset()
        sim = task._env.sim
        plat_pos = _platform_clear_of_robot(task)
        bz = task._base_half_size[2]
        ridge_top = plat_pos[2] + bz * 2 + task._ridge_height
        # Rod sits on the ridge — no airborne gap so rod/ridge contact is
        # consistently present across physics substeps.
        rod_z = ridge_top + task._rod_half_length

        action = np.zeros(task.action_space.low.shape[0])
        terminated = False

        for _ in range(task._balance_hold_steps + 60):
            # Teleport rod upright each step (simulating perfect control)
            task._place_object("rod", [plat_pos[0], plat_pos[1], rod_z],
                               quat=[1, 0, 0, 0])
            task._set_object_velocity("rod", [0, 0, 0], [0, 0, 0])
            sim.forward()

            obs, r, term, trunc, info = task.step(action)
            if term:
                terminated = True
                break

        assert terminated, "Task should terminate after sustained balance"
        assert info["task_score"] == pytest.approx(1.0, abs=0.05)

    def test_fallen_rod_terminates(self, task):
        """If the rod falls off the table entirely, task should terminate."""
        task.reset()
        sim = task._env.sim

        # Teleport rod below the floor threshold
        task._place_object("rod", [0.0, 0.0, 0.01])
        sim.forward()

        action = np.zeros(task.action_space.low.shape[0])
        obs, r, term, trunc, info = task.step(action)
        assert term, "Rod below floor should trigger termination"


# =====================================================================
#  Balance predicate and reset regressions
# =====================================================================

def _place_rod_upright(task, xy, bottom_z):
    task._place_object("rod", [xy[0], xy[1], bottom_z + task._rod_half_length], quat=[1, 0, 0, 0])
    task._set_object_velocity("rod", [0, 0, 0], [0, 0, 0])
    task._env.sim.forward()


@pytest.mark.parametrize("task_id", ["ib-p7-balance-medium-v1", "ib-p7-balance-hard-v2"])
class TestBalanceRegressions:

    def test_reset_obs_matches_sim(self, task_id):
        task = _make_task(task_id, obs_mode="proprio")
        try:
            obs, _ = task.reset(seed=1)
            sim = task._env.sim
            np.testing.assert_allclose(
                obs["object_pos"], sim.data.body_xpos[sim.model.body_name2id("rod_main")])
            np.testing.assert_allclose(
                obs["platform_pos"], sim.data.body_xpos[sim.model.body_name2id("platform_main")])
            assert obs["rod_tilt_deg"][0] == pytest.approx(task._get_rod_tilt_deg())
        finally:
            task.close()

    def test_upright_rod_on_base_plate_is_not_balanced(self, task_id):
        """Standing the rod on the flat base plate beside the ridge must not count."""
        task = _make_task(task_id, obs_mode="proprio", max_episode_steps=400)
        try:
            task.reset(seed=0)
            sim = task._env.sim
            pid = sim.model.body_name2id("platform_main")
            plat = sim.data.body_xpos[pid].copy()
            across = sim.data.body_xmat[pid].reshape(3, 3)[:, 1]
            xy = plat[:2] + across[:2] * 0.025
            _place_rod_upright(task, xy, plat[2] + 2 * task._base_half_size[2] + 0.0005)
            zero = np.zeros(task.action_space.shape[0])
            for _ in range(task._balance_hold_steps + 20):
                _, _, term, _, info = task.step(zero)
                assert not info["is_success"]
            assert task._get_rod_tilt_deg() < task._tilt_threshold_deg
            assert info["task_score"] == 0.0 and not term
        finally:
            task.close()

    def test_rod_touching_finger_is_not_balanced(self, task_id):
        """An upright rod on the ridge that leans on a finger (no two-pad grasp) is not balanced."""
        task = _make_task(task_id, obs_mode="proprio")
        try:
            task.reset(seed=0)
            sim = task._env.sim
            pad = sim.data.geom_xpos[
                sim.model.geom_name2id("gripper0_right_finger1_pad_collision")].copy()
            quat = sim.data.body_xquat[sim.model.body_name2id("platform_main")].copy()
            task._place_static("platform", [pad[0], pad[1], task._table_height], quat=quat)
            ridge_top = task._table_height + 2 * task._base_half_size[2] + task._ridge_height
            _place_rod_upright(task, pad[:2], ridge_top)
            zero = np.zeros(task.action_space.shape[0])
            zero[-1] = -1.0
            task.step(zero)
            assert task._get_rod_tilt_deg() < task._tilt_threshold_deg
            assert not task._gripper_contacts_object("rod")
            assert not task._is_balanced()
        finally:
            task.close()

    def test_robot_contact_resets_hold_counter(self, task_id):
        """The grace window covers tilt/contact flicker, not a step the robot supports the rod."""
        task = _make_task(task_id, obs_mode="proprio")
        try:
            task.reset(seed=0)
            sim = task._env.sim
            pad = sim.data.geom_xpos[
                sim.model.geom_name2id("gripper0_right_finger1_pad_collision")].copy()
            quat = sim.data.body_xquat[sim.model.body_name2id("platform_main")].copy()
            task._place_static("platform", [pad[0], pad[1], task._table_height], quat=quat)
            ridge_top = task._table_height + 2 * task._base_half_size[2] + task._ridge_height
            _place_rod_upright(task, pad[:2], ridge_top)
            task._gripper_has_released = True
            task._balance_counter = task._balance_hold_steps - 1
            zero = np.zeros(task.action_space.shape[0])
            zero[-1] = -1.0
            task.step(zero)
            assert robot_contacts(sim, task._rod_geom_ids)
            assert not task._gripper_contacts_object("rod")
            assert task._balance_counter == 0
        finally:
            task.close()


@pytest.mark.parametrize("task_id", ["ib-p7-balance-hard-v1", "ib-p7-balance-medium-v1"])
class TestRidgeContact:
    """A rod can stand on the ridge, and only a still rod counts as balanced."""

    def _ridge_frame(self, task):
        sim = task._env.sim
        pid = sim.model.body_name2id("platform_main")
        plat = _platform_clear_of_robot(task)
        across = sim.data.body_xmat[pid].reshape(3, 3)[:, 1]
        ridge_top = plat[2] + 2 * task._base_half_size[2] + task._ridge_height
        return plat, across, ridge_top

    def test_rod_off_centre_by_2mm_stays_upright(self, task_id):
        task = _make_task(task_id, obs_mode="proprio", max_episode_steps=400)
        try:
            task.reset(seed=0)
            plat, across, ridge_top = self._ridge_frame(task)
            _place_rod_upright(task, plat[:2] + 0.002 * across[:2], ridge_top)
            _step_n(task, n=200)
            assert task._get_rod_tilt_deg() < 3.0
        finally:
            task.close()

    def test_rotating_rod_is_not_balanced(self, task_id):
        task = _make_task(task_id, obs_mode="proprio", max_episode_steps=400)
        try:
            task.reset(seed=0)
            plat, across, ridge_top = self._ridge_frame(task)
            _place_rod_upright(task, plat[:2], ridge_top)
            _step_n(task, n=5)
            assert task._is_balanced()
            task._set_object_velocity("rod", [0, 0, 0], 0.3 * np.cross([0, 0, 1], across))
            task._env.sim.forward()
            assert not task._is_balanced()
        finally:
            task.close()
