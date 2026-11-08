"""Tests for prehensile task scoring, termination, and physics.

Covers:
- imbench.tasks.prehensile.* (per-task _compute_task_score, _check_termination)
- Seesaw hinge physics (seesaw_balance_v1)
- Pendulum physics (pendulum_grasp_v1)
- Bounce/linear intercept, cube_toss, slip_recovery, tool_retrieve, cup_extract,
  occluder_push, sheltered_grasp, edge_slide_v1, domino, mass_sort, pendulum_grasp (v1/v2), slide_catch scoring
"""

import math
import numpy as np
import pytest

from imbench.registry import TASK_REGISTRY, make
from imbench.tasks.base_task import BaseTask
from imbench.tasks.prehensile.contacts import geoms_touch

PENDULUM_TASKS = ["ib-p2-pendulum-grasp-v1", "ib-p2-pendulum-grasp-v2"]


# ── Helpers ─────────────────────────────────────────────────────────

def _make_task(task_id, **kw):
    # Default to proprio (no cameras) for scoring/physics tests — cuts OpenGL
    # rendering cost per step. Tests that assert on rgb_* keys must pass
    # obs_mode="combo" explicitly.
    defaults = dict(obs_mode="proprio", render_mode=None, seed=42, privileged_obs=True,
                    max_episode_steps=10)
    defaults.update(kw)
    return make(task_id, **defaults)


def _tip_body_pos(task):
    """Local (parent-relative) position of the hook's tip sub-body."""
    sim = task._env.sim
    body_id = sim.model.body_name2id("hook_tool_tip_main")
    return sim.model.body_pos[body_id]


def _step_n(task, n=5, action=None):
    lo, _ = task.action_space.low, task.action_space.high
    if action is None:
        action = np.zeros(lo.shape[0])
    for _ in range(n):
        task.step(action)


def _servo(task, obs, target, gripper):
    """One step driving the end effector toward *target* with gripper command *gripper*."""
    action = np.zeros(task.action_space.shape)
    action[:3] = np.clip(10.0 * (np.asarray(target) - obs["robot0_eef_pos"]), -1.0, 1.0)
    action[-1] = gripper
    return task.step(action)


def _pendulum_angle(task):
    sim = task._env.sim
    return float(sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint_name2id("pendulum_hinge")]])


def _hold_pendulum(task, angle):
    """Teleport the pendulum to hinge *angle* at rest."""
    sim = task._env.sim
    jid = sim.model.joint_name2id("pendulum_hinge")
    sim.data.qpos[sim.model.jnt_qposadr[jid]] = angle
    sim.data.qvel[sim.model.jnt_dofadr[jid]] = 0.0
    sim.forward()


# =====================================================================
#  Seesaw hinge physics
# =====================================================================

class TestSeesawHinge:
    """Verify the seesaw plank actually tips via a hinge joint."""

    @pytest.fixture(scope="class")
    @classmethod
    def seesaw_task(cls):
        task = _make_task("ib-p3-seesaw-balance-v1", max_episode_steps=300)
        task.reset()
        yield task
        task.close()

    def test_seesaw_hinge_exists(self, seesaw_task):
        sim = seesaw_task._env.sim
        joint_names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
        assert "seesaw_hinge" in joint_names

    def test_seesaw_tips_from_cube_weight(self, seesaw_task):
        """Plank should tip >5 degrees from the cube sitting on the far end."""
        seesaw_task.reset()
        sim = seesaw_task._env.sim
        hinge_id = sim.model.joint_name2id("seesaw_hinge")
        qpos_addr = sim.model.jnt_qposadr[hinge_id]

        _step_n(seesaw_task, n=100)
        angle_deg = math.degrees(sim.data.qpos[qpos_addr])
        assert abs(angle_deg) > 5.0, f"Seesaw should tip but angle={angle_deg:.2f}°"

    def test_seesaw_has_stoppers(self, seesaw_task):
        sim = seesaw_task._env.sim
        geom_names = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
                      if sim.model.geom_id2name(i)]
        stopper_geoms = [n for n in geom_names if "stopper" in n]
        assert len(stopper_geoms) >= 2

    def test_seesaw_score_drops_after_tipping(self, seesaw_task):
        seesaw_task.reset()
        # After stepping, the cube weight tips the plank past the threshold
        _step_n(seesaw_task, n=100)
        score = seesaw_task.task_score()
        assert score < 1.0, f"Seesaw should be unbalanced but score={score:.2f}"


# =====================================================================
#  Pendulum physics
# =====================================================================

class TestPendulumPhysics:
    """Verify pendulum swings and bracket keeps rod out of bob trajectory."""

    @pytest.fixture(scope="class")
    @classmethod
    def pend_task(cls):
        task = _make_task("ib-p2-pendulum-grasp-v1", max_episode_steps=200)
        task.reset()
        yield task
        task.close()

    def test_pendulum_hinge_exists(self, pend_task):
        sim = pend_task._env.sim
        hinge_id = sim.model.joint_name2id("pendulum_hinge")
        axis = sim.model.jnt_axis[hinge_id]
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-5)

    def test_pendulum_swings(self, pend_task):
        """Bob y-position should oscillate (amplitude > 0.03m)."""
        pend_task.reset()
        sim = pend_task._env.sim
        bob_geom_id = sim.model.geom_name2id("pendulum_bob")

        ys = []
        _step_n(pend_task, n=100)
        for _ in range(50):
            pend_task.step(np.zeros(7))
            ys.append(sim.data.geom_xpos[bob_geom_id][1])

        assert max(ys) - min(ys) > 0.03, "Pendulum should show Y oscillation"

    def test_stand_far_from_robot(self, pend_task):
        sim = pend_task._env.sim
        stand_geom = sim.model.geom_name2id("pendulum_stand_post")
        stand_x = sim.data.geom_xpos[stand_geom][0]
        bob_geom = sim.model.geom_name2id("pendulum_bob")
        bob_x = sim.data.geom_xpos[bob_geom][0]
        assert stand_x > bob_x, "Stand should be farther from robot than bob"


# =====================================================================
#  Per-task score/termination logic
# =====================================================================

class TestScoreTerminationLogic:
    """Test _compute_task_score and _check_termination with mock obs."""

    def test_seesaw_crossing_score(self):
        task = _make_task("ib-p3-seesaw-balance-v1")
        try:
            task.reset()
            score = task.task_score(recompute=True)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()

    def test_pendulum_initial_score_zero(self):
        """The `approach_pendulum` stage gives small dense credit for eef
        proximity to the bob even before any grasp, so the initial score is
        not exactly 0, just far from success."""
        task = _make_task("ib-p2-pendulum-grasp-v1")
        try:
            task.reset()
            score = task.task_score(recompute=True)
            assert 0.0 <= score < 0.2, "No grasp at start"
        finally:
            task.close()

    def test_occluder_push_initial_score_partial(self):
        task = _make_task("ib-p5-occluder-push-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()

    def test_sheltered_grasp_initial_score_partial(self):
        task = _make_task("ib-p1-sheltered-grasp-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()

    def test_domino_single_cascade_scoring(self):
        task = _make_task("ib-p3-domino-single-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
            term = task._check_termination(raw)
            assert isinstance(term, bool)
        finally:
            task.close()

    def test_domino_select_scoring(self):
        task = _make_task("ib-p3-domino-select-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert score in (0.0, 1.0)
        finally:
            task.close()

    def test_mass_sort_initial_score(self):
        task = _make_task("ib-p5-mass-sort-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()


# =====================================================================
#  Task-specific physics checks
# =====================================================================

class TestBounceIntercept:

    def test_initial_ball_moves(self):
        task = _make_task("ib-p2-bounce-intercept-v1", max_episode_steps=50)
        try:
            task.reset()
            sim = task._env.sim
            body_id = sim.model.body_name2id("cube_main")
            pos0 = sim.data.body_xpos[body_id].copy()
            _step_n(task, n=20)
            pos1 = sim.data.body_xpos[body_id].copy()
            dist = np.linalg.norm(pos1 - pos0)
            assert dist > 0.01, "Ball should move after reset"
        finally:
            task.close()

    def test_ball_out_of_bounds_terminates(self):
        task = _make_task("ib-p2-bounce-intercept-v1", max_episode_steps=500)
        try:
            task.reset()
            raw = {"cube_pos": np.array([2.0, 0.0, 0.9])}
            assert task._check_termination(raw) is True
        finally:
            task.close()


class TestLinearIntercept:

    def test_ball_out_of_bounds_terminates(self):
        task = _make_task("ib-p2-linear-intercept-v1", max_episode_steps=50)
        try:
            task.reset()
            raw = {"cube_pos": np.array([1.0, 0.0, 0.9])}
            assert task._check_termination(raw) is True
        finally:
            task.close()


class TestCubeToss:

    def test_goal_region_exists(self):
        task = _make_task("ib-p2-cube-toss-v1")
        try:
            task.reset()
            assert hasattr(task, "_goal_pos")
            assert hasattr(task, "_goal_radius")
            assert task._goal_radius > 0
        finally:
            task.close()

    def test_fell_off_terminates(self):
        task = _make_task("ib-p2-cube-toss-v1")
        try:
            task.reset()
            raw = {"cube_pos": np.array([0.0, 0.0, 0.01])}
            assert task._check_termination(raw)
        finally:
            task.close()


class TestSlipRecovery:

    def test_initial_no_recovery(self):
        task = _make_task("ib-p6-slip-recovery-v1")
        try:
            task.reset()
            assert not task._slipped
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", ["ib-p6-slip-recovery-v1", "ib-p6-slip-recovery-v2"])
    @pytest.mark.parametrize("seed", [1, 7])
    def test_lemon_spawns_at_rest(self, task_id, seed):
        """The spawn pose (and the transport baseline) is where the lemon stays."""
        task = _make_task(task_id, max_episode_steps=200,
                          config_overrides={"task": {"ycb_pool": ["lemon"]}})
        try:
            task.reset(seed=seed)
            body = task._object_body_id(task._active_obj_name())
            sim = task._env.sim
            _step_n(task, 100)
            drift = sim.data.xpos[body][:2] - task._object_spawn_xy
            assert np.linalg.norm(drift) < 0.005
        finally:
            task.close()


class TestToolRetrieve:

    def test_hook_tool_object_exists(self):
        task = _make_task("ib-p4-tool-retrieve-v1")
        try:
            task.reset()
            obj = task._env.get_object("hook_tool")
            assert obj is not None
        finally:
            task.close()


class TestToolRetrieveV2:
    """OOD variant: hook tip mirror-flipped onto the opposite side of the
    shaft (paper's P4 OOD-shift spec). Covers imbench/tasks/prehensile/
    tool_retrieve_v2.py."""

    def test_hook_tool_object_exists(self):
        task = _make_task("ib-p4-tool-retrieve-v2")
        try:
            task.reset()
            obj = task._env.get_object("hook_tool")
            assert obj is not None
        finally:
            task.close()

    def test_tip_is_mirrored_relative_to_v1(self):
        v1 = _make_task("ib-p4-tool-retrieve-v1")
        v2 = _make_task("ib-p4-tool-retrieve-v2")
        try:
            v1.reset()
            v2.reset()
            tip_y_v1 = _tip_body_pos(v1)[1]
            tip_y_v2 = _tip_body_pos(v2)[1]
            assert tip_y_v1 != 0.0
            assert tip_y_v2 == pytest.approx(-tip_y_v1)
        finally:
            v1.close()
            v2.close()

    def test_shaft_position_unchanged(self):
        """Only the tip should move -- the shaft (and its length) must be
        identical between v1 and v2, matching the paper's "same shaft
        position/length" OOD spec (only the tip side changes)."""
        v1 = _make_task("ib-p4-tool-retrieve-v1")
        v2 = _make_task("ib-p4-tool-retrieve-v2")
        try:
            v1.reset()
            v2.reset()
            tip_x_v1, tip_z_v1 = _tip_body_pos(v1)[0], _tip_body_pos(v1)[2]
            tip_x_v2, tip_z_v2 = _tip_body_pos(v2)[0], _tip_body_pos(v2)[2]
            assert tip_x_v1 == pytest.approx(tip_x_v2)
            assert tip_z_v1 == pytest.approx(tip_z_v2)
        finally:
            v1.close()
            v2.close()

    def test_score_bounded(self):
        task = _make_task("ib-p4-tool-retrieve-v2")
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()


class TestCupExtract:

    def test_initial_score_below_one(self):
        task = _make_task("ib-p4-cup-extract-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert score < 1.0
        finally:
            task.close()


class TestOccluderPush:

    def test_lid_exists(self):
        task = _make_task("ib-p5-occluder-push-v1")
        try:
            task.reset()
            sim = task._env.sim
            geom_names = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
                          if sim.model.geom_id2name(i)]
            assert any("lid" in n for n in geom_names), \
                f"Expected lid geom, found: {geom_names}"
        finally:
            task.close()


class TestShelteredGrasp:

    def test_enclosure_geoms_exist(self):
        task = _make_task("ib-p1-sheltered-grasp-v1")
        try:
            task.reset()
            sim = task._env.sim
            geom_names = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)
                          if sim.model.geom_id2name(i)]
            assert any("wall" in n for n in geom_names), \
                f"Expected enclosure wall geoms, found: {geom_names}"
        finally:
            task.close()


class TestEdgeSlideV1:

    def test_reset_and_step(self):
        task = _make_task("ib-p1-edge-slide-v1", seed=1)
        try:
            obs, info = task.reset()
            assert isinstance(obs, dict)
            lo, _ = task.action_space.low, task.action_space.high
            task.step(np.zeros(lo.shape[0]))
        finally:
            task.close()

    def test_initial_score(self):
        task = _make_task("ib-p1-edge-slide-v1", seed=1)
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()


# =====================================================================
#  Pendulum grasp scoring
# =====================================================================

class TestPendulumGraspScoring:

    def test_grasped_starts_false(self):
        task = _make_task("ib-p2-pendulum-grasp-v1")
        try:
            task.reset()
            assert not task._grasped
            assert task._apex_z is not None
        finally:
            task.close()

    def test_get_bob_pos(self):
        task = _make_task("ib-p2-pendulum-grasp-v1")
        try:
            task.reset()
            pos = task._get_bob_pos()
            assert pos is not None
            assert pos.shape == (3,)
        finally:
            task.close()

    def test_obs_includes_object_pos(self):
        task = _make_task("ib-p2-pendulum-grasp-v1")
        try:
            obs, _ = task.reset()
            assert "object_pos" in obs
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", PENDULUM_TASKS)
    def test_geometry_drawn_per_reset_from_reset_seed(self, task_id):
        def geometry(task):
            sim = task._env.sim
            compiled_bob = sim.model.geom_size[sim.model.geom_name2id("pendulum_bob")][0]
            return task._arm_length, task._stand_height, task._bob_size, float(compiled_bob)

        tasks = [_make_task(task_id, seed=ctor_seed) for ctor_seed in (0, 123, 5)]
        a, b, c = tasks
        try:
            a.reset(seed=5)
            b.reset(seed=5)
            assert geometry(a) == geometry(b)
            assert geometry(a)[2] == geometry(a)[3]  # compiled into the model
            np.testing.assert_array_equal(a._get_bob_pos(), b._get_bob_pos())
            a.reset()
            assert geometry(a) != geometry(b)
            assert geometry(a)[2] == geometry(a)[3]
            # The constructor seed sets the first unseeded episode, even after
            # observation_space probed a reset.
            c.observation_space
            c.reset()
            assert geometry(c) == geometry(b)
        finally:
            for task in tasks:
                task.close()

    @pytest.mark.parametrize("task_id", PENDULUM_TASKS)
    def test_holding_still_earns_no_approach_credit(self, task_id):
        """The bob swings past the gripper's reset pose; that is no approach."""
        task = _make_task(task_id, max_episode_steps=300)
        try:
            task.reset(seed=0)
            _step_n(task, 250)  # longer than a full swing
            assert task._stage_debug["approach_pendulum"] < 0.02
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", PENDULUM_TASKS)
    def test_shut_jaws_touching_bob_are_not_a_grasp(self, task_id):
        """Shut jaws lowered onto the bob held at its apex touch it with both
        fingertips; that must not count as catching it."""
        task = _make_task(task_id, max_episode_steps=2000)
        try:
            obs, _ = task.reset(seed=0)
            apex = _pendulum_angle(task)
            bob = task._get_bob_pos()
            both_touched = False
            for i in range(300):
                _hold_pendulum(task, apex)
                above = 0.08 if i < 150 else 0.0
                obs, _, terminated, _, info = _servo(task, obs, bob + [0.0, 0.0, above], 1.0)
                sim = task._env.sim
                both_touched |= all(geoms_touch(sim, finger, task._bob_geoms)
                                    for finger in task._finger_geoms)
                assert not info["is_success"] and not terminated
                assert task._stage_debug["grasp_bob"] == 0.0
            assert both_touched
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", PENDULUM_TASKS)
    def test_pinch_counts_once_held(self, task_id):
        """Jaws closing around the bob held at its apex: success only after
        the pinch has held for PINCH_HOLD_STEPS consecutive steps."""
        task = _make_task(task_id, max_episode_steps=2000)
        try:
            obs, _ = task.reset(seed=0)
            apex = _pendulum_angle(task)
            bob = task._get_bob_pos()
            for i in range(300):  # open jaws straddling the bob
                _hold_pendulum(task, apex)
                obs, *_ = _servo(task, obs, bob + [0.0, 0.0, 0.08 if i < 150 else 0.0], -1.0)
            assert not task._robot_touches_bob()
            pinched = 0
            for _ in range(100):
                _hold_pendulum(task, apex)
                obs, _, terminated, _, info = _servo(task, obs, bob, 1.0)
                pinched = pinched + 1 if task._bob_pinched() else 0
                assert info["is_success"] == (pinched >= task.PINCH_HOLD_STEPS)
                if terminated:
                    break
            assert info["is_success"] and terminated
            assert task._stage_debug["approach_pendulum"] > 0.9
        finally:
            task.close()


class TestSlideCatchScoring:

    def test_step_a_cube_falls_scores_zero(self):
        task = _make_task("ib-p2-slide-catch-v1", max_episode_steps=300,
                          config_overrides={"task": {"num_cubes_range": [2, 2]}})
        try:
            task.reset(seed=0)
            sim = task._env.sim
            first, other = task._active_cubes
            bin_pos = sim.data.body_xpos[task._object_body_id("bin")].copy()
            # Catch the first cube: at rest on the floor of the held bin.
            task._place_object(first, bin_pos + [0.0, 0.0, task._bin_wall + task._cube_half_heights[first] + 0.002])
            task._set_object_velocity(first, [0, 0, 0], [0, 0, 0])
            _step_n(task, 20)
            assert task._stage_debug["cube1_caught"] == 1.0
            # The other cube falls to the floor.
            task._place_object(other, [0.0, 1.0, 0.02])
            sim.forward()
            *_, terminated, _, info = task.step(np.zeros(task.action_space.shape))
            assert terminated
            assert info["task_score"] == 0.0
            assert info["is_success"] is False
        finally:
            task.close()


class TestSeesawTermination:

    def test_counterweight_on_floor_ends_episode(self):
        task = _make_task("ib-p3-seesaw-balance-v1", max_episode_steps=300)
        try:
            task.reset(seed=0)
            task._place_object("weight_0", [0.0, 1.0, 0.021])
            task._set_object_velocity("weight_0", [0, 0, 0], [0, 0, 0])
            task._env.sim.forward()
            *_, terminated, _, info = task.step(np.zeros(task.action_space.shape))
            assert terminated
            assert info["is_success"] is False
        finally:
            task.close()


# =====================================================================
#  Step cycles for all prehensile tasks
# =====================================================================

_ALL_PREHENSILE = [
    "ib-p4-cup-extract-v1",
    "ib-p4-tool-retrieve-v1",
    "ib-p5-occluder-push-v1",
    "ib-p1-sheltered-grasp-v1",
    "ib-p1-edge-slide-v1",
    "ib-p2-pendulum-grasp-v1",
    "ib-p6-slip-recovery-v1",
    "ib-p5-mass-sort-v1",
    "ib-p3-domino-single-v1",
    "ib-p3-domino-select-v1",
]


class TestPrehensileStepCycles:

    @pytest.mark.parametrize("task_id", _ALL_PREHENSILE)
    def test_step_and_score_after_steps(self, task_id):
        task = _make_task(task_id, max_episode_steps=20)
        try:
            task.reset()
            lo, _ = task.action_space.low, task.action_space.high
            for _ in range(5):
                obs, r, t, tr, info = task.step(np.zeros(lo.shape[0]))
                assert 0.0 <= info["task_score"] <= 1.0
        finally:
            task.close()
