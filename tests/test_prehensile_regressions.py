"""Regression tests for scoring exploits and defects in the prehensile tasks.

Success requires the intended outcome and never fires at reset or under zero
actions.

- Dynamics, throwing and edge slide: a tossed cube must leave the hand well
  outside the goal and come to rest in it, and the cube-toss table stops at
  the robot pedestal; a bounce-intercept catch counts only after a wall
  bounce; a linear-intercept ball reaching the robot pedestal has left the
  workspace; the Galileo ball must reach the goal untouched; an attached
  slide-catch bin earns no grasp credit; an edge-slide object off the table
  scores zero, its grasp and lift credit need a two-finger grasp, and its
  randomised friction governs sliding on the table.
- Dominoes: success must come from the first domino's cascade, every
  consecutive pair from the first domino to the target touching in order.
  Displacing the target any other way (knocking it or the last domino over,
  throwing a domino at it, knocking the chain over backwards from it) or
  touching the target or any later chain domino with the robot (fingers or
  arm links) never succeeds. Unused dominoes are parked off-scene and hidden
  from ``object_state``. Every domino-single layout cascades from a push on
  its first domino; toppling a domino-select decoy chain fails the episode.
- Cup extract: the cuboid counts as extracted only once no part of it is
  inside the cup, whatever the cup's pose; transport and placement credit
  require extraction; placement needs the cuboid released and at rest on the
  goal; a cuboid off the table ends the episode.
- Mass sort: cubes never spawn inside a zone, a cube the robot is still
  touching is never placed, and cube inertia matches the shuffled mass; the
  scripted policy re-plans its approach when the cube moves.
- Packing: success needs every piece packed at the same time (a piece taken
  back out loses its placement credit); suction welds only the pieces;
  ``config_overrides`` reach the task's own config; the observation layout
  does not depend on the piece count.
- Tool retrieve (v1, v2): grasping and lifting the cube succeeds; idling
  does not.
- Pendulum, seesaw, slip recovery: success requires a real grasp of the
  pendulum bob caught at its apex, a released counterweight balancing the
  plank with the target block still where it spawned, and a slip that fired,
  dropped the object onto the table and was recovered before the object was
  released into the goal; slip-recovery scores, recomputed or not, track the
  active pool object, and a dropped pool object gains no energy on impact.
- Occluder push (v2; v1 has its own tests in test_occluder_push_v1.py):
  lids spawn side by side without overlapping; v2
  stacks two or three lid layers and parks the unused one; the red cube
  counts as placed only once it is at rest on the patch and the robot has
  let go.
- Recover peg insert (v1, v2): the peg counts as inserted only while
  ``insert_fraction`` of it lies inside the slot on the slot's centre line; a
  peg held past the exit or beside the slot earns no depth credit; nothing is
  credited at reset; a peg still standing on the table is not grasped.
- Sheltered grasp (v1, v2): the cube spawns resting on its compartment floor,
  clear of the rack; v2 reads the task YAML; the cube counts as placed only
  once it is at rest on the tray and the robot has let go.
"""

import mujoco
import numpy as np
import pytest

from data_synthetic import collection
from data_synthetic.single_arm_pipeline.mass_sort_policy import HOVER_ABOVE_OBJ, CubePickPlace
from data_synthetic.single_arm_pipeline.pendulum_grasp_policy import (
    SIDE_QUAT_MINUS_Y,
    PendulumGraspPolicy,
    _quat_err_axis_angle,
)
from imbench.registry import make
from imbench.tasks.base_task import load_task_config
from imbench.tasks.prehensile.contacts import (
    body_geom_ids,
    body_subtree_ids,
    geoms_touch,
    object_geom_ids,
    robot_contacts,
)
from imbench.tasks.prehensile.packing_v1 import (
    _BOX_BASE_THICK,
    _MAX_PIECES,
    _PIECE_SETTLED_CONTACT_STEPS,
    _PIECE_THICKNESS,
)
from imbench.tasks.prehensile.recover_peg_insert_v1 import _SLOT_LENGTH

GALILEO_TASKS = ["ib-p2-galileo-ramp-drop-v1", "ib-p2-galileo-ramp-drop-v2"]
EDGE_SLIDE_TASKS = ["ib-p1-edge-slide-v1", "ib-p1-edge-slide-v2"]
DOMINO_TASKS = [
    "ib-p3-domino-single-v1",
    "ib-p3-domino-single-v2",
    "ib-p3-domino-select-v1",
    "ib-p3-domino-select-v2",
]
TOOL_TASKS = ["ib-p4-tool-retrieve-v1", "ib-p4-tool-retrieve-v2"]
PENDULUM_TASKS = ["ib-p2-pendulum-grasp-v1", "ib-p2-pendulum-grasp-v2"]
SLIP_TASKS = ["ib-p6-slip-recovery-v1", "ib-p6-slip-recovery-v2"]
OCCLUDER_TASKS = ["ib-p5-occluder-push-v2"]
PEG_TASKS = ["ib-p1-recover-peg-insert-v1", "ib-p1-recover-peg-insert-v2"]
SHELTERED_TASKS = ["ib-p1-sheltered-grasp-v1", "ib-p1-sheltered-grasp-v2"]


# =====================================================================
#  Shared helpers
# =====================================================================

def _make(task_id, **kw):
    # These tests inspect simulator ground truth (object poses) in obs.
    return make(task_id, **{"obs_mode": "proprio", "render_mode": None, "seed": 0,
                            "privileged_obs": True, **kw})


def _action(task, gripper=-1.0, delta=(0.0, 0.0, 0.0)):
    """Action moving the end effector by *delta* with gripper command *gripper*."""
    action = np.zeros(task.action_space.shape)
    action[:3] = delta
    action[-1] = gripper
    return action


def _step(task, n, gripper=-1.0, before_step=None):
    """Step *n* times with the arm still, stopping when the episode ends.

    *before_step(i)*, if given, runs before step *i*. Returns the last step
    tuple and whether success held at any step.
    """
    ever = False
    for i in range(n):
        if before_step is not None:
            before_step(i)
        out = task.step(_action(task, gripper))
        ever |= out[4]["is_success"]
        if out[2] or out[3]:
            break
    return out, ever


def _move_eef(task, target, gripper, max_steps=300, tol=0.003):
    """Servo the end effector to the world-frame *target* holding command *gripper*.

    Returns the last step tuple and whether success held at any step.
    """
    sim = task._env.sim
    site = task._env.robots[0].eef_site_id["right"]
    base_rot = sim.data.body_xmat[sim.model.body_name2id("robot0_base")].reshape(3, 3)
    ever = False
    for _ in range(max_steps):
        err = np.asarray(target, dtype=float) - sim.data.site_xpos[site]
        out = task.step(_action(task, gripper, np.clip(8.0 * base_rot.T @ err, -1.0, 1.0)))
        ever |= out[4]["is_success"]
        if np.linalg.norm(err) < tol:
            break
    return out, ever


def _grasp(task, pos, grip_steps=40):
    """Grasp the object centred at *pos* from above and keep holding it there.

    Returns whether success held at any step.
    """
    pos = np.asarray(pos, dtype=float)
    ever = _move_eef(task, pos + [0.0, 0.0, 0.10], -1.0)[1]
    ever |= _move_eef(task, pos, -1.0)[1]
    for _ in range(grip_steps):
        ever |= _move_eef(task, pos, 1.0, max_steps=1)[1]
    return ever


def _body_pos(task, body):
    sim = task._env.sim
    return sim.data.body_xpos[sim.model.body_name2id(body)].copy()


def _grip_site(task):
    sim = task._env.sim
    return sim.data.site_xpos[sim.model.site_name2id("gripper0_right_grip_site")].copy()


def _place_at_rest(task, name, pos):
    task._place_object(name, pos)
    task._set_object_velocity(name, [0, 0, 0], [0, 0, 0])
    task._env.sim.forward()


# =====================================================================
#  Reset and idle
# =====================================================================

@pytest.mark.parametrize("task_id", [
    "ib-p2-cube-toss-v1",
    "ib-p2-bounce-intercept-v1",
    *GALILEO_TASKS,
    "ib-p2-linear-intercept-v1",
    "ib-p2-slide-catch-v1",
    "ib-m-mirror-pick-place-v1",
    *EDGE_SLIDE_TASKS,
    *PENDULUM_TASKS,
    "ib-p3-seesaw-balance-v1",
    *SLIP_TASKS,
])
def test_no_success_at_reset_or_under_zero_action(task_id):
    task = _make(task_id)
    try:
        _, info = task.reset(seed=0)
        assert info["is_success"] is False
        for _ in range(5):
            _, _, _, _, info = task.step(_action(task))
            assert info["is_success"] is False
            assert info["task_score"] < 1.0
    finally:
        task.close()


# =====================================================================
#  Dynamics, throwing and edge slide
# =====================================================================

def _hold_in_air(task, name, z, steps):
    """Teleport *name* to height *z* above its current xy every step (no robot)."""
    info = {}
    for _ in range(steps):
        pos = _body_pos(task, f"{name}_main")
        _place_at_rest(task, name, [pos[0], pos[1], z])
        _, _, terminated, _, info = task.step(_action(task))
        if terminated:
            break
    return info


# ── Cube toss ───────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def cube_toss():
    task = _make("ib-p2-cube-toss-v1", max_episode_steps=300)
    yield task
    task.close()


class TestCubeToss:

    def test_robot_pedestal_stops_at_table_edge(self, cube_toss):
        cube_toss.reset(seed=0)
        sim = cube_toss._env.sim
        pedestal = sim.model.geom_name2id("fixed_mount0_pedestal_col")
        table = sim.model.geom_name2id("table_collision")
        front = sim.data.geom_xpos[pedestal][0] + sim.model.geom_size[pedestal][0]
        near_edge = sim.data.geom_xpos[table][0] - sim.model.geom_size[table][0]
        assert front <= near_edge + 1e-6

    def test_goal_marker_is_visual_only(self, cube_toss):
        cube_toss.reset(seed=0)
        sim = cube_toss._env.sim
        body = sim.model.body_name2id("goal_marker_main")
        geoms = np.flatnonzero(sim.model.geom_bodyid == body)
        assert geoms.size > 0
        assert not sim.model.geom_contype[geoms].any()
        assert not sim.model.geom_conaffinity[geoms].any()

    def test_spinning_cube_landing_in_goal_comes_to_rest_and_succeeds(self, cube_toss):
        # A cube balanced on the goal disc's single contact would spin forever;
        # with a visual-only disc it settles flat on the table.
        cube_toss.reset(seed=3)
        goal = cube_toss._goal_pos
        cube_toss._place_object(
            "cube", [goal[0] + 0.03, goal[1] - 0.02, cube_toss._table_height + 0.03],
            quat=[np.cos(0.2), 0, 0, np.sin(0.2)])
        cube_toss._set_object_velocity("cube", [0.3, 0.1, -0.2], [0.5, -0.3, 1.0])
        cube_toss._env.sim.forward()
        for _ in range(100):
            _, _, terminated, _, info = cube_toss.step(_action(cube_toss))
            if terminated:
                break
        assert info["is_success"] is True
        assert terminated is True
        assert cube_toss._at_rest_count >= cube_toss._rest_steps

    def test_cube_sliding_through_goal_does_not_succeed_until_at_rest(self, cube_toss):
        cube_toss.reset(seed=0)
        goal, radius = cube_toss._goal_pos, cube_toss._goal_radius
        start = [goal[0] - 0.8 * radius, goal[1], cube_toss._table_height + 0.022]
        cube_toss._place_object("cube", start)
        for _ in range(40):
            cube_toss._set_object_velocity("cube", [0.5, 0.0, 0.0], [0, 0, 0])
            _, _, terminated, _, info = cube_toss.step(_action(cube_toss))
            assert info["is_success"] is False and not terminated
        assert np.linalg.norm(_body_pos(cube_toss, "cube_main")[:2] - goal) < radius
        assert cube_toss._at_rest_count == 0
        # Left alone, the cube stops inside the goal and the toss succeeds.
        for _ in range(100):
            _, _, terminated, _, info = cube_toss.step(_action(cube_toss))
            if terminated:
                break
        assert info["is_success"] is True

    @staticmethod
    def _grasp_at_grip_site(task, steps=20):
        _place_at_rest(task, "cube", _grip_site(task))
        for _ in range(steps):
            task.step(_action(task, gripper=1.0))

    def test_cube_released_inside_goal_does_not_succeed(self, cube_toss):
        # Goal moved under the hand: the robot carries the cube into the goal
        # and drops it there instead of tossing it.
        cube_toss.reset(seed=0)
        cube_toss._goal_pos = _grip_site(cube_toss)[:2]
        self._grasp_at_grip_site(cube_toss)
        assert np.linalg.norm(cube_toss._release_xy - cube_toss._goal_pos) < cube_toss._goal_radius
        for _ in range(150):
            _, _, terminated, _, info = cube_toss.step(_action(cube_toss))
            assert info["is_success"] is False and not terminated
        cube = _body_pos(cube_toss, "cube_main")
        assert np.linalg.norm(cube[:2] - cube_toss._goal_pos) < cube_toss._goal_radius
        assert cube[2] < cube_toss._table_height + 0.04
        assert cube_toss._at_rest_count >= cube_toss._rest_steps
        assert cube_toss._stage_debug["land_in_goal"] == 0.0
        assert cube_toss._stage_debug["settle_at_rest"] == 0.0

    def test_cube_released_far_from_goal_and_resting_in_it_succeeds(self, cube_toss):
        cube_toss.reset(seed=0)
        self._grasp_at_grip_site(cube_toss)
        release = cube_toss._release_xy.copy()
        assert np.allclose(release, _grip_site(cube_toss)[:2], atol=0.02)
        goal, radius = cube_toss._goal_pos, cube_toss._goal_radius
        assert np.linalg.norm(release - goal) - radius >= cube_toss._min_release_dist
        # Cube in flight: no robot contact from here on.
        cube_toss._place_object("cube", [goal[0], goal[1], cube_toss._table_height + 0.03])
        cube_toss._set_object_velocity("cube", [0.2, 0.0, -0.3], [0, 0, 0])
        cube_toss._env.sim.forward()
        for _ in range(100):
            _, _, terminated, _, info = cube_toss.step(_action(cube_toss, gripper=1.0))
            if terminated:
                break
        assert info["is_success"] is True
        assert np.array_equal(cube_toss._release_xy, release)

    def test_release_xy_boost_is_configurable(self):
        assert load_task_config("ib-p2-cube-toss-v1")["task"]["release_xy_boost"] == 3.0
        task = _make("ib-p2-cube-toss-v1", config_overrides={"task": {"release_xy_boost": 1.0}})
        try:
            assert task._release_xy_boost == 1.0
        finally:
            task.close()


# ── Bounce intercept ────────────────────────────────────────────────

@pytest.fixture(scope="class")
def bounce():
    task = _make("ib-p2-bounce-intercept-v1", max_episode_steps=300)
    yield task
    task.close()


class TestBounceIntercept:

    def test_ball_held_before_any_bounce_does_not_succeed(self, bounce):
        bounce.reset(seed=0)
        hold_z = bounce._table_height + 0.15
        info = _hold_in_air(bounce, "cube", hold_z, bounce._hold_steps_required + 10)
        assert bounce._has_bounced is False
        assert bounce._hold_count == 0
        assert info["is_success"] is False

    def test_robot_contact_before_bounce_fails_episode(self, bounce):
        bounce.reset(seed=0)
        _place_at_rest(bounce, "cube", _grip_site(bounce))
        _, _, terminated, _, info = bounce.step(_action(bounce, gripper=1.0))
        assert bounce._has_bounced is False
        assert bounce._failed is True
        assert terminated is True
        assert info["task_score"] == 0.0 and info["is_success"] is False

    def test_ball_held_after_wall_bounce_succeeds(self, bounce):
        bounce.reset(seed=0)
        for _ in range(150):
            bounce.step(_action(bounce))
            if bounce._has_bounced:
                break
        assert bounce._has_bounced is True
        info = _hold_in_air(bounce, "cube", bounce._table_height + 0.15,
                            bounce._hold_steps_required + 2)
        assert info["is_success"] is True
        assert info["task_score"] == 1.0


class TestBounceScriptedPolicy:

    @staticmethod
    def _tracking_policy(monkeypatch):
        from types import SimpleNamespace
        from data_synthetic.single_arm_pipeline.bounce_intercept_policy import BounceInterceptPolicy

        policy = BounceInterceptPolicy()
        policy.phase = "track"
        policy._ball_vel_est = np.array([0.5, 0.0, 0.0])
        ball = np.array([0.2, 0.0, 0.82])
        policy._planned_intercept_T = 1.1
        policy._planned_intercept_xy = policy._predict_path_at(
            ball[:2], policy._ball_vel_est[:2], policy._planned_intercept_T,
        )
        monkeypatch.setattr(policy, "_update_ball_velocity", lambda *args: None)
        obs = {"object_pos": ball,
               "robot0_eef_pos": np.r_[policy._planned_intercept_xy, 0.9],
               "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0])}
        return policy, obs, SimpleNamespace(_has_bounced=False)

    def test_future_bounce_does_not_exhaust_replans(self, monkeypatch):
        """A valid reflected-path target is behind the current velocity."""
        policy, obs, env = self._tracking_policy(monkeypatch)
        assert policy._planned_intercept_xy[0] < obs["object_pos"][0] - 0.15
        for _ in range(10):
            policy.act(obs, env=env)
        assert policy.phase == "track" and not policy.failed
        assert policy._replan_count == 0

    @pytest.mark.parametrize("exhausted", [False, True])
    def test_expired_plan_replans_or_stops_at_the_limit(self, monkeypatch, exhausted):
        from data_synthetic.single_arm_pipeline.bounce_intercept_policy import MAX_REPLANS

        policy, obs, env = self._tracking_policy(monkeypatch)
        policy._global_step = 150  # more than 0.3 s past the 1.1 s plan
        policy._replan_count = MAX_REPLANS if exhausted else 0
        policy.act(obs, env=env)
        if exhausted:
            assert policy.failed and policy.phase == "done"
        else:
            assert not policy.failed and policy._replan_count == 1
            policy.act(obs, env=env)
            assert policy._replan_count == 1  # the replacement gets its own budget

    def test_descent_waits_for_first_wall_bounce(self, monkeypatch):
        policy, obs, env = self._tracking_policy(monkeypatch)
        obs["object_pos"][:2] = policy._planned_intercept_xy - [0.05, 0]
        policy.act(obs, env=env)
        assert policy.phase == "track"
        env._has_bounced = True
        policy.act(obs, env=env)
        assert policy.phase == "descend"

    def test_off_table_ball_still_aborts(self, monkeypatch):
        policy, obs, env = self._tracking_policy(monkeypatch)
        obs["object_pos"][2] = 0.6
        policy.act(obs, env=env)
        assert policy.failed and policy.phase == "done"

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_scripted_catch_after_wall_bounce(self, seed):
        """Seeds 1 and 2 formerly stopped before reaching the planned bounce."""
        from data_synthetic.collection import clip_action
        from data_synthetic.single_arm_pipeline.bounce_intercept_policy import BounceInterceptPolicy

        task = _make("ib-p2-bounce-intercept-v1", privileged_obs=True, seed=seed)
        try:
            obs, _ = task.reset(seed=seed)
            policy = BounceInterceptPolicy()
            # Match the single-arm collector's one-step warm-up and clipping.
            obs, _, _, _, info = task.step(np.zeros(task.action_space.shape, dtype=np.float32))
            for _ in range(task.MAX_EPISODE_STEPS - 1):
                action = clip_action(policy.act(obs, env=task), task.action_space)
                obs, _, terminated, truncated, info = task.step(action)
                if terminated or truncated or policy.failed:
                    break
            assert info["is_success"] and task._has_bounced
            assert not policy.failed
        finally:
            task.close()


# ── Galileo ramp drop ───────────────────────────────────────────────

def _ramp_arc_point(task, theta):
    """World position of a ball resting on the ramp arc at angle *theta* (rad)."""
    R, r = task._ARC_RADIUS, task._ball_radius
    ly = (R - r) * np.sin(theta)
    lz = task._RAMP_TROUGH_Z + R - (R - r) * np.cos(theta) + 0.002
    c, s = np.cos(task._ramp_yaw_actual), np.sin(task._ramp_yaw_actual)
    rx, ry, rz = task._ramp_pos
    return np.array([rx - s * ly, ry + c * ly, rz + lz])


def _calibrated_release_point(task):
    """Open-arm rest point whose first goal-side apex lands on the goal (scripted-policy calibration)."""
    from data_synthetic.single_arm_pipeline.galileo_ramp_drop_policy import RELEASE_ANGLE_POLY
    phi = np.deg2rad(np.polyval(RELEASE_ANGLE_POLY, np.degrees(task._goal_theta)))
    return _ramp_arc_point(task, -phi)


def _in_goal_region(task, ball):
    local = task._goal_rot_world.T @ (ball - task._goal_center_world)
    return (abs(local[0]) < task._goal_patch_half_x
            and abs(local[1]) < task._goal_patch_half_y
            and abs(local[2]) < 4 * task._ball_radius)


@pytest.mark.parametrize("task_id", GALILEO_TASKS)
class TestGalileoRampDrop:

    def test_ball_held_still_at_goal_by_robot_does_not_succeed(self, task_id):
        task = _make(task_id, max_episode_steps=200)
        try:
            task.reset(seed=0)
            # The hand cannot reach the goal patch past the channel's side
            # rails; they play no part in the rule under test.
            model = task.sim.model
            for i in range(model.ngeom):
                if (model.geom_id2name(i) or "").startswith(("ramp_rail", "ramp_cover")):
                    model.geom_contype[i] = model.geom_conaffinity[i] = 0
            normal = task._goal_rot_world[:, 2]
            hold = task._goal_center_world + normal * (task._ball_radius + 0.035)
            still_in_goal = False
            for _ in range(80):
                delta = np.clip((hold - _grip_site(task)) * 10.0, -1.0, 1.0)
                _place_at_rest(task, "cube", _grip_site(task))
                _, _, terminated, _, info = task.step(_action(task, 1.0, delta))
                ball = _body_pos(task, "cube_main")
                speed = np.linalg.norm(task._ball_velocity())
                if _in_goal_region(task, ball) and speed < task._goal_velocity_threshold:
                    still_in_goal = True
                assert info["is_success"] is False and not terminated
            assert still_in_goal, "the held ball never sat still inside the goal region"
            assert task._untouched_steps == 0
        finally:
            task.close()

    def test_ball_released_untouched_on_opposite_arm_succeeds(self, task_id):
        task = _make(task_id, max_episode_steps=200)
        try:
            task.reset(seed=0)
            # Energy-matched release: the first goal-side apex lands on the goal.
            _place_at_rest(task, "cube", _calibrated_release_point(task))
            for _ in range(40):
                _, _, terminated, _, info = task.step(_action(task))
                if terminated:
                    break
            assert info["is_success"] is True
            assert terminated is True
            assert task._untouched_steps >= task._UNTOUCHED_STEPS
        finally:
            task.close()


@pytest.mark.parametrize("task_id, long_axis", [
    ("ib-p2-galileo-ramp-drop-v1", 1),
    ("ib-p2-galileo-ramp-drop-v2", 0),
])
def test_galileo_ramp_geometry_matches_description(task_id, long_axis):
    """v1 ramp runs along world Y, v2 along world X; the goal is on the goal-side endpoint's arm."""
    task = _make(task_id)
    try:
        task.reset(seed=0)
        span = task._far_endpoint[:2] - task._near_endpoint[:2]
        assert abs(span[long_axis]) > 10 * abs(span[1 - long_axis])
        assert np.dot(task._goal_center_world[:2] - task._ramp_pos[:2], span) > 0
    finally:
        task.close()


# ── Linear intercept / mirror pick-place ────────────────────────────

def test_linear_intercept_ball_held_above_table_succeeds():
    task = _make("ib-p2-linear-intercept-v1", max_episode_steps=100)
    try:
        task.reset(seed=0)
        info = _hold_in_air(task, "cube", task._table_height + 0.15,
                            task._hold_steps_required + 2)
        assert info["is_success"] is True
    finally:
        task.close()


def test_linear_intercept_ball_reaching_robot_pedestal_ends_episode():
    """The pedestal is flush with the robot-side table edge; a ball reaching it has left the workspace."""
    task = _make("ib-p2-linear-intercept-v1", max_episode_steps=200)
    try:
        task.reset(seed=0)
        _place_at_rest(task, "cube", [-0.2, 0.0, task._table_height + task._object_radius + 0.001])
        task._set_object_velocity("cube", [-0.3, 0.0, 0.0])
        (_, _, terminated, truncated, info), ever = _step(task, 200)
        assert terminated and not truncated, f"ball still in play at {_body_pos(task, 'cube_main')}"
        assert not ever
    finally:
        task.close()


def test_mirror_pick_place_lift_needs_a_grasp(monkeypatch):
    task = _make("ib-m-mirror-pick-place-v1", max_episode_steps=50)
    try:
        task.reset(seed=0)
        z = task._table_height + task._cube_size + task._lift_height + 0.005
        info = _hold_in_air(task, "cube", z, 1)
        assert info["is_success"] is False
        monkeypatch.setattr(task, "_gripper_contacts_object", lambda obj_name="cube": True)
        info = _hold_in_air(task, "cube", z, 1)
        assert info["is_success"] is True
    finally:
        task.close()


# ── Slide catch ─────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def slide_catch():
    task = _make("ib-p2-slide-catch-v1", max_episode_steps=100)
    yield task
    task.close()


class TestSlideCatch:

    def test_attached_bin_earns_no_grasp_credit(self, slide_catch):
        slide_catch.reset(seed=0)
        _, _, _, _, info = slide_catch.step(_action(slide_catch))
        assert "grasp_bin" not in slide_catch._stage_debug
        assert info["task_score"] <= 0.25
        assert info["is_success"] is False

    def test_free_standing_bin_grasp_stage_starts_at_zero(self):
        task = _make("ib-p2-slide-catch-v1",
                     config_overrides={"task": {"bin_attached_to_gripper": False}})
        try:
            task.reset(seed=0)
            task.step(_action(task))
            assert task._stage_debug["grasp_bin"] == 0.0
        finally:
            task.close()

    def test_all_cubes_in_bin_succeeds(self, slide_catch):
        slide_catch.reset(seed=0)
        bin_pos = _body_pos(slide_catch, "bin_root")
        offsets = np.linspace(-0.03, 0.03, len(slide_catch._active_cubes))
        for name, dy in zip(slide_catch._active_cubes, offsets):
            slide_catch._place_object(name, [bin_pos[0], bin_pos[1] + dy, bin_pos[2] + 0.02])
            slide_catch._set_object_velocity(name, [0, 0, 0], [0, 0, 0])
        slide_catch._env.sim.forward()
        for _ in range(50):
            _, _, terminated, _, info = slide_catch.step(_action(slide_catch))
            if terminated:
                break
        assert info["is_success"] is True
        assert terminated is True

    def test_cube_over_the_rim_is_not_caught(self, slide_catch):
        slide_catch.reset(seed=0)
        sim = slide_catch._env.sim
        bin_pos = _body_pos(slide_catch, "bin_root")
        rim = slide_catch._bin_wall + slide_catch._bin_wall_height
        slide_catch._place_object("cube", [bin_pos[0], bin_pos[1], bin_pos[2] + rim + 0.02])
        sim.forward()
        assert not slide_catch._cube_in_bin(sim, "cube")

    def test_cube_passing_through_the_bin_is_not_caught(self, slide_catch):
        slide_catch.reset(seed=0)
        sim = slide_catch._env.sim
        bin_pos = _body_pos(slide_catch, "bin_root")
        slide_catch._place_object("cube", [bin_pos[0], bin_pos[1], bin_pos[2] + 0.02])
        slide_catch._set_object_velocity("cube", [1.0, 0.0, 0.0], [0, 0, 0])
        sim.forward()
        assert not slide_catch._cube_in_bin(sim, "cube")


# ── Edge slide ──────────────────────────────────────────────────────

def test_edge_slide_v2_cameras_match_v1():
    def cameras(task_id):
        cams = load_task_config(task_id)["obs"]["cameras"]
        return {k: (v["robosuite_name"], v["enabled"]) for k, v in cams.items()}

    assert cameras("ib-p1-edge-slide-v2") == cameras("ib-p1-edge-slide-v1")


@pytest.mark.parametrize("task_id", EDGE_SLIDE_TASKS)
def test_edge_slide_object_off_table_scores_zero(task_id):
    task = _make(task_id, max_episode_steps=20)
    try:
        task.reset(seed=0)
        pos = _body_pos(task, "cube_main")
        task._teleport_cube(task._env.sim, pos[0], pos[1], task._table_height - 0.2, 0.0)
        task._env.sim.forward()
        _, _, terminated, _, info = task.step(_action(task))
        assert terminated is True
        assert info["task_score"] == 0.0
        assert info["is_success"] is False
    finally:
        task.close()


@pytest.mark.parametrize("task_id", EDGE_SLIDE_TASKS)
def test_edge_slide_object_friction_governs_table_contact(task_id):
    """MuJoCo takes the larger friction of a pair, so the table must sit below the sampled object friction."""
    task = _make(task_id, max_episode_steps=20)
    try:
        for seed in (0, 1):
            task.reset(seed=seed)
            _step(task, 2)
            sim = task._env.sim
            cube = {sim.model.geom_name2id(g) for g in task._env.cube.contact_geoms}
            table = sim.model.geom_name2id("table_collision")
            frictions = [c.friction[0] for c in sim.data.contact[:sim.data.ncon]
                         if {int(c.geom1), int(c.geom2)} & cube and table in (c.geom1, c.geom2)]
            assert frictions, "object not resting on the table"
            np.testing.assert_allclose(frictions, sim.model.geom_friction[min(cube)][0])
    finally:
        task.close()


@pytest.mark.parametrize("task_id", EDGE_SLIDE_TASKS)
def test_edge_slide_grasp_and_lift_credit_need_a_grasp(task_id):
    """Raising the object without a grasp, or pressing on it with the fingertips, earns no credit."""
    task = _make(task_id, max_episode_steps=500)
    try:
        task.reset(seed=0)
        sim = task._env.sim
        pos = _body_pos(task, "cube_main")
        # Raised clear of the robot, as when flicked or levered onto its edge.
        task._teleport_cube(sim, pos[0], pos[1], task._rest_obj_z + 0.03, 0.0)
        sim.forward()
        _step(task, 50)
        assert task._stage_debug["lift"] == 0.0
        # Open fingertips pressed down onto its top face.
        top = 2 * task._rest_obj_z - task._table_height
        _move_eef(task, [pos[0], pos[1], top - 0.01], -1.0)
        assert task._gripper_contacts_object(), "fingertips never reached the object"
        assert task._stage_debug["grasp"] == 0.0
        assert task._stage_debug["lift"] == 0.0
    finally:
        task.close()


@pytest.mark.parametrize("task_id", EDGE_SLIDE_TASKS)
def test_edge_slide_success_needs_a_grasp(task_id, monkeypatch):
    """A lifted object touched by one finger (a scoop) is not a grasp."""
    task = _make(task_id, max_episode_steps=50)
    try:
        task.reset(seed=0)
        monkeypatch.setattr(task, "_gripper_contacts_object", lambda: True)
        monkeypatch.setattr(task, "_is_grasping", lambda: False)
        lifted = {"cube_pos": np.array([0.0, 0.0, task._lift_threshold + 0.01])}
        assert not task._staged_reward._success_fn(task, lifted)
        monkeypatch.setattr(task, "_is_grasping", lambda: True)
        assert task._staged_reward._success_fn(task, lifted)
    finally:
        task.close()


# =====================================================================
#  Dominoes
# =====================================================================

# Straight chains cascade reliably; curved patterns can jam.
LINE_LAYOUT = {"randomization": {"pattern": {"enabled": False}}, "task": {"interconnected": False}}


@pytest.fixture(scope="class", params=DOMINO_TASKS)
def domino(request):
    task = _make(request.param, max_episode_steps=300)
    yield task
    task.close()


@pytest.fixture(scope="class", params=DOMINO_TASKS)
def domino_line(request):
    task = _make(request.param, max_episode_steps=300, config_overrides=LINE_LAYOUT)
    yield task
    task.close()


def _chain(task):
    """Names of the dominoes whose cascade must reach the target, first to last."""
    if hasattr(task, "_correct_chain"):
        if getattr(task, "_interconnected", False):
            return task._solution_route
        return [f"chain{task._correct_chain}_domino_{i}"
                for i in range(task._dominoes_per_chain)]
    return [f"domino_{i}" for i in range(task._active_num_dominoes)]


def _toppled(task, name):
    sim = task._env.sim
    body_z = sim.data.body_xmat[sim.model.body_name2id(f"{name}_main")].reshape(3, 3)[:, 2]
    return np.degrees(np.arccos(min(1.0, abs(body_z[2])))) >= task._topple_tilt_threshold_deg


def _teleport_onto_robot(task, name, robot_body):
    """Move *name* onto the grip site (``"grip_site"``) or the arm body *robot_body*."""
    pos = _grip_site(task) if robot_body == "grip_site" else _body_pos(task, robot_body)
    _place_at_rest(task, name, pos)


def _throw_at_target(task, name, direction):
    """Throw *name*, lying flat and pointing along the throw, at the target from 8 cm away.

    *direction* is the throw direction in the target's (heading, lateral)
    frame: ``(0, 1)`` hits it from the side, ``(-1, 0)`` from beyond the
    chain's end, back toward the chain.
    """
    sim = task._env.sim
    target = sim.model.body_name2id("red_block_main")
    axes = sim.data.body_xmat[target].reshape(3, 3)
    v = direction[0] * axes[:, 0] + direction[1] * axes[:, 1]
    pos = sim.data.body_xpos[target] - 0.08 * v
    pos[2] = task._table_height + 0.07
    # Laid flat (90° about y, long axis on world x), then yawed onto v.
    yaw = np.arctan2(v[1], v[0])
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.sqrt(0.5) * np.array([c, -s, c, s])
    task._place_object(name, pos.tolist(), quat=quat.tolist())
    task._set_object_velocity(name, (2.0 * v).tolist(), [0, 0, 0])
    sim.forward()


def _finger_push(task, names):
    """``before_step`` hook tipping each of *names* along its heading, like a fingertip."""
    sim = task._env.sim
    bodies = [sim.model.body_name2id(f"{name}_main") for name in names]

    def push(i):
        for body in bodies:
            heading = sim.data.body_xmat[body].reshape(3, 3)[:, 0]
            sim.data.xfrc_applied[body, 3:] = 0.05 * np.cross([0, 0, 1], heading) if i < 25 else 0.0

    return push


def _object_state_slices(task):
    """``{observable key: slice}`` into ``obs['object_state']``."""
    raw = task._get_sim_observations(force_update=True)
    slices, start = {}, 0
    for key, obs in task._env._observables.items():
        if obs.modality == "object" and obs.is_enabled() and obs.is_active() and key in raw:
            size = np.atleast_1d(raw[key]).shape[0]
            slices[key] = slice(start, start + size)
            start += size
    return slices


class TestDominoNoFreeSuccess:

    def test_reset_and_idle_score_zero(self, domino):
        _, info = domino.reset(seed=0)
        assert info["is_success"] is False and info["task_score"] == 0.0
        (_, _, terminated, _, info), ever = _step(domino, 30)
        assert not ever
        assert info["task_score"] == 0.0
        assert not terminated
        assert not domino._robot_touched_chain


class TestDominoCascadeRequired:

    @pytest.mark.parametrize("victim", ["last_domino", "target"])
    def test_displacing_target_without_cascade_scores_zero(self, domino, victim):
        domino.reset(seed=0)
        name = _chain(domino)[-1] if victim == "last_domino" else "red_block"
        # Free-joint angular velocity is body-local: +y tips the top along the heading.
        domino._set_object_velocity(name, [0, 0, 0], [0, 4.0, 0])
        if victim == "last_domino" and getattr(domino, "_interconnected", False):
            # The resized network tail can absorb the old angular kick;
            # use the same sustained physical push as the cascade probes.
            (_, _, _, _, info), ever = _step(domino, 150, before_step=_finger_push(domino, [name]))
            domino.sim.data.xfrc_applied[:] = 0
        else:
            (_, _, _, _, info), ever = _step(domino, 150)
        assert domino._target_displaced(), "the shortcut must actually displace the target"
        assert not ever
        assert info["task_score"] == 0.0
        assert not domino._first_domino_toppled

    def test_first_domino_thrown_at_target_scores_no_success(self, domino):
        domino.reset(seed=0)
        _throw_at_target(domino, _chain(domino)[0], (0, 1))
        (_, _, _, _, info), ever = _step(domino, 150)
        assert domino._target_displaced() and domino._first_domino_toppled
        assert not ever
        assert info["task_score"] < 1.0

    def test_chain_knocked_over_backwards_scores_no_success(self, domino_line):
        task = domino_line
        task.reset(seed=0)
        chain = _chain(task)
        sim = task._env.sim
        second = sim.model.body_name2id(f"{chain[1]}_main")
        heading = sim.data.body_xmat[second].reshape(3, 3)[:, 0]
        # Rest the first domino against the second so the first pair touches ...
        offset = 2 * task._domino_size[2] - 2e-4  # faces overlap by 0.2 mm
        task._place_object(chain[0], (sim.data.body_xpos[second] - offset * heading).tolist(),
                           quat=sim.data.body_xquat[second].tolist())
        sim.forward()
        _step(task, 2)
        assert task._link_touch_step[0] > 0
        # ... then throw it at the target from beyond so the chain falls backwards.
        _throw_at_target(task, chain[0], (-1, 0))
        (_, _, _, _, info), ever = _step(task, 300)
        assert task._target_displaced() and task._first_domino_toppled
        assert (task._link_touch_step > 0).all(), "the backward wave must touch every pair"
        assert not ever
        assert info["task_score"] < 1.0

    def test_first_domino_push_cascades_to_success(self, domino_line):
        task = domino_line
        task.reset(seed=0)
        chain = _chain(task)
        (_, _, terminated, _, info), ever = _step(task, 300, before_step=_finger_push(task, chain[:1]))
        task._env.sim.data.xfrc_applied[:] = 0.0
        assert ever and info["is_success"], (
            f"no success after {info['step']} steps: {task._stage_debug}")
        assert terminated
        assert info["task_score"] == 1.0
        assert task._stage_debug["_success"] is True
        assert _toppled(task, chain[0]) and _toppled(task, chain[-1])
        assert not task._robot_touched_chain
        assert not getattr(task, "_decoy_toppled", False)


@pytest.mark.parametrize("task_id", DOMINO_TASKS[:2])
def test_domino_single_sampled_layouts_cascade(task_id):
    """Every layout the task samples cascades from the first domino to the target."""
    task = _make(task_id, max_episode_steps=300)
    try:
        for seed in range(6):
            task.reset(seed=seed)
            (_, _, _, _, info), ever = _step(task, 300, before_step=_finger_push(task, ["domino_0"]))
            task._env.sim.data.xfrc_applied[:] = 0.0
            assert ever, f"seed {seed}: cascade stalled, {task._stage_debug}"
    finally:
        task.close()


@pytest.mark.parametrize("pushed", ["every_chain", "decoys_only"])
@pytest.mark.parametrize("task_id", DOMINO_TASKS[2:])
def test_domino_select_toppled_decoy_chain_fails(task_id, pushed):
    """Toppling a decoy chain ends the episode with zero score, even alongside the correct one."""
    task = _make(task_id, max_episode_steps=300)
    try:
        task.reset(seed=0)
        starts = list(task._decoy_starts)
        if pushed == "every_chain":
            starts.append(task._solution_route[0])
        push = _finger_push(task, starts)
        (_, _, terminated, _, info), ever = _step(task, 300, before_step=push)
        task._env.sim.data.xfrc_applied[:] = 0.0
        assert not ever
        assert terminated and info["task_score"] == 0.0
        assert task._decoy_toppled and not task._robot_touched_chain
    finally:
        task.close()


class TestDominoRobotContactDisqualifies:

    @pytest.mark.parametrize("victim,robot_body", [
        ("target", "grip_site"),
        ("last_domino", "robot0_link5"),
        ("second_domino", "robot0_link4"),
    ])
    def test_contact_terminates_with_zero_score(self, domino, victim, robot_body):
        domino.reset(seed=0)
        chain = _chain(domino)
        name = {"target": "red_block", "last_domino": chain[-1], "second_domino": chain[1]}[victim]
        _teleport_onto_robot(domino, name, robot_body)
        (_, _, terminated, _, info), ever = _step(domino, 5)
        assert not ever and info["task_score"] == 0.0
        assert terminated
        assert domino._robot_touched_chain

        # Latched: a solved-looking scene afterwards still scores nothing.
        domino._place_object(chain[0], _body_pos(domino, f"{chain[0]}_main"),
                             quat=[0.7071, 0.0, 0.7071, 0.0])
        domino._place_object("red_block", domino._initial_target_pos + [0.1, 0.0, 0.0])
        domino._env.sim.forward()
        assert domino.task_score(recompute=True) == 0.0
        assert domino._is_success(0.0) is False

    def test_touching_first_domino_is_allowed(self, domino):
        domino.reset(seed=0)
        first = _chain(domino)[0]
        _teleport_onto_robot(domino, first, "grip_site")
        sim = domino._env.sim
        first_geoms = body_geom_ids(sim, f"{first}_main")
        touched_first = False

        def check(_):
            nonlocal touched_first
            touched_first |= robot_contacts(sim, first_geoms)

        _step(domino, 5, before_step=check)
        assert touched_first
        assert not domino._robot_touched_chain


class TestDominoUnusedParked:

    def test_parked_objects_far_away_and_zeroed_in_obs(self, domino):
        n_parked = 0
        shapes = set()
        for seed in range(4):
            obs, _ = domino.reset(seed=seed)
            slices = _object_state_slices(domino)
            parked = domino._parked_objects()
            n_parked += len(parked)
            for name in parked:
                assert _body_pos(domino, f"{name}_main")[2] > 50.0
                for key in (f"{name}_pos", f"{name}_quat"):
                    assert not np.any(obs["object_state"][slices[key]]), f"{key} visible in obs"
            for name in _chain(domino):
                z = _body_pos(domino, f"{name}_main")[2] - domino._table_height
                assert abs(z - domino._domino_size[0]) < 0.01
            # Reset obs reflect the re-laid scene, not the pre-reset one.
            first = _chain(domino)[0]
            np.testing.assert_allclose(obs["object_state"][slices[f"{first}_pos"]],
                                       _body_pos(domino, f"{first}_main"), atol=1e-6)
            shapes.add(obs["object_state"].shape)
            _step(domino, 20)
            obs = domino._rebuild_obs()
            assert np.abs(obs["object_state"]).max() < 5.0
        assert n_parked > 0, "no seed exercised parking"
        assert len(shapes) == 1


# =====================================================================
#  Cup extract, mass sort, packing, tool retrieve
# =====================================================================

# ── Cup extract ─────────────────────────────────────────────────────

TIPPED = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]   # cup axis along +x


@pytest.fixture(scope="class")
def cup():
    task = _make("ib-p4-cup-extract-v1", max_episode_steps=2000)
    yield task
    task.close()


class TestCupExtract:

    def test_object_color_flag_disables_cuboid_recolour(self):
        def cuboid_rgba(enabled, seed):
            task = _make("ib-p4-cup-extract-v1", config_overrides={
                "randomization": {"object_color": {"enabled": enabled}}})
            try:
                task.reset(seed=seed)
                m = task._env.sim.model
                body = m.body_name2id("small_object_main")
                return m.geom_rgba[m.geom_bodyid == body].copy()
            finally:
                task.close()
        np.testing.assert_allclose(cuboid_rgba(False, 1), cuboid_rgba(False, 2))
        assert not np.allclose(cuboid_rgba(True, 1), cuboid_rgba(True, 2))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_no_credit_at_reset_or_zero_action(self, cup, seed):
        obs, _ = cup.reset(seed=seed)
        np.testing.assert_allclose(obs["object_pos"], _body_pos(cup, "cube_main"))
        (_, _, terminated, _, info), ever = _step(cup, 5)
        assert info["task_score"] == 0.0
        assert not ever and not terminated

    def test_cup_on_goal_with_cuboid_inside_scores_nothing(self, cup):
        cup.reset(seed=0)
        g, z = cup._goal_pos, cup._table_height
        cup._place_object("cube", [g[0], g[1], z + 0.002], quat=[0.0, 0.0, 0.0, 1.0])
        cup._place_object("small_object", [g[0], g[1], z + 0.04])
        (_, _, terminated, _, info), ever = _step(cup, 30)
        assert info["task_score"] == 0.0
        assert not ever and not terminated
        assert cup._small_object_in_cup()

    def test_tipped_cup_with_cuboid_inside_scores_nothing(self, cup):
        cup.reset(seed=0)
        g, z = cup._goal_pos, cup._table_height
        cup._place_object("cube", [g[0] - 0.07, g[1], z + 0.04], quat=TIPPED)
        cup._place_object("small_object", [g[0] - 0.01, g[1], z + 0.034], quat=TIPPED)
        (_, _, terminated, _, info), ever = _step(cup, 60)
        assert info["task_score"] == 0.0
        assert not ever and not terminated
        assert cup._small_object_in_cup()

    def test_half_extracted_cuboid_is_still_in_cup(self, cup):
        cup.reset(seed=0)
        g, z = cup._goal_pos, cup._table_height
        # Centre just past the rim, near end 2.5 cm inside the cup, both on the goal.
        cup._place_object("cube", [g[0] - 0.07, g[1], z + 0.04], quat=TIPPED)
        cup._place_object("small_object", [g[0] + 0.035, g[1], z + 0.03], quat=TIPPED)
        cup._env.sim.forward()
        assert cup.task_score(recompute=True) == 0.0
        assert cup._small_object_in_cup()

    def test_cuboid_placed_on_goal_succeeds(self, cup):
        cup.reset(seed=0)
        g, z = cup._goal_pos, cup._table_height
        cup._place_object("small_object", [g[0], g[1], z + 0.002 + cup._CUBE_HALF_Z + 0.001])
        (_, _, terminated, _, info), ever = _step(cup, 30)
        assert info["is_success"] and terminated
        assert info["task_score"] == 1.0
        assert not cup._small_object_in_cup()

    def test_unsupported_cuboid_above_goal_is_not_placed(self, cup):
        cup.reset(seed=0)
        g, z = cup._goal_pos, cup._table_height
        # Within the resting-height band, 6 mm clear of the goal marker.
        cup._place_object("small_object", [g[0], g[1], z + cup._CUBE_HALF_Z + 0.008])
        cup._env.sim.forward()
        assert cup.task_score(recompute=True) == pytest.approx(0.70)
        assert cup._stage_debug["place_resting"] == 0.0
        (_, _, terminated, _, info), _ = _step(cup, 30)
        assert info["is_success"] and terminated

    def test_cuboid_sliding_across_goal_is_not_placed(self, cup):
        cup.reset(seed=0)
        g, z = cup._goal_pos, cup._table_height
        away = g - _body_pos(cup, "cube_main")[:2]
        away /= np.linalg.norm(away)
        # Upright on the table 10 cm short of the goal centre, sliding across
        # the goal away from the cup.
        start = g - 0.10 * away
        cup._place_object("small_object", [start[0], start[1], z + cup._CUBE_HALF_Z + 0.001])
        cup._set_object_velocity("small_object", [*(1.6 * away), 0.0])
        cup._env.sim.forward()
        (_, _, _, _, info), ever = _step(cup, 150)
        assert cup._stage_debug["transport_to_goal_region"] == 1.0
        assert not ever and not info["is_success"]

    def test_cuboid_held_on_goal_is_not_placed(self, cup):
        cup.reset(seed=0)
        g, z = cup._goal_pos, cup._table_height
        start = np.array([g[0], g[1] - cup._GOAL_HALF - 0.05, z + cup._CUBE_HALF_Z])
        _place_at_rest(cup, "small_object", start)
        _grasp(cup, start)
        _move_eef(cup, start + [0.0, 0.0, 0.05], 1.0)

        # Press the grasped cuboid onto the goal and keep holding it.
        hold = np.array([g[0], g[1], z + 0.002 + cup._CUBE_HALF_Z])
        _move_eef(cup, hold + [0.0, 0.0, 0.05], 1.0)
        ever = False
        for _ in range(100):
            (_, _, terminated, _, _), held = _move_eef(cup, hold, 1.0, max_steps=1)
            ever |= held
            assert not terminated
        sim, cuboid = cup._env.sim, object_geom_ids(cup._env, "small_object")
        assert abs(_body_pos(cup, "small_object_main")[2] - z - cup._CUBE_HALF_Z) < 0.01
        assert robot_contacts(sim, cuboid)
        assert geoms_touch(sim, cuboid, object_geom_ids(cup._env, "goal_marker"))
        assert cup._stage_debug["transport_to_goal_region"] == 1.0
        assert cup._stage_debug["place_resting"] == 0.0
        assert not ever

        # Releasing it on the goal completes the task.
        for _ in range(100):
            (_, _, terminated, _, info), _ = _move_eef(
                cup, hold + [0.0, 0.0, 0.05], -1.0, max_steps=1)
            if terminated:
                break
        assert info["is_success"] and terminated

    def test_cuboid_off_table_terminates(self, cup):
        cup.reset(seed=0)
        cup._place_object("small_object", [0.0, 1.5, 0.03])
        (_, _, terminated, _, info), _ = _step(cup, 1)
        assert terminated and not info["is_success"]


# ── Mass sort ───────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def mass():
    task = _make("ib-p5-mass-sort-v1", max_episode_steps=2000)
    yield task
    task.close()


def _cube_names(task):
    return ["cube" if i == 0 else f"cube_{i}" for i in range(task._num_cubes)]


def _zone_xy(task):
    return np.array([_body_pos(task, f"zone_{z}_main")[:2] for z in range(task._num_cubes)])


def _target_zones(task):
    masses = [task._masses[task._mass_order[i]] for i in range(task._num_cubes)]
    return [sorted(masses).index(m) for m in masses]


class TestMassSort:

    def test_cube_sliding_across_its_zone_is_not_placed(self, mass):
        mass.reset(seed=0)
        names, zones, targets = _cube_names(mass), _zone_xy(mass), _target_zones(mass)
        rest_z = mass._table_height + mass._cube_size + 0.001
        mass._place_object(names[0], [*zones[targets[0]], rest_z])
        mass._set_object_velocity(names[0], [1.2, 0.0, 0.0], [0, 0, 0])
        mass._env.sim.forward()
        statuses, _ = mass._cube_zone_state()
        assert statuses[0] == "unplaced"

    # Seeds 8, 28 and 42 spawn a cube inside a zone before it is resampled clear.
    @pytest.mark.parametrize("seed", [8, 28, 42, 0, 1, 2, 3, 4])
    def test_cubes_spawn_clear_of_zones(self, mass, seed):
        obs, _ = mass.reset(seed=seed)
        np.testing.assert_allclose(obs["object_pos"], _body_pos(mass, "cube_main"))
        clearance = mass._zone_width / 2 + np.sqrt(2.0) * mass._cube_size
        zones = _zone_xy(mass)
        for name in _cube_names(mass):
            xy = _body_pos(mass, f"{name}_main")[:2]
            assert np.min(np.linalg.norm(zones - xy, axis=1)) > clearance
        (_, _, terminated, _, info), ever = _step(mass, 5)
        assert info["task_score"] == 0.0
        assert not ever and not terminated

    def test_resample_moves_cube_out_of_zone(self, mass):
        mass.reset(seed=0)
        zones = _zone_xy(mass)
        others = {n: _body_pos(mass, f"{n}_main") for n in _cube_names(mass)[1:]}
        mass._place_object("cube", [zones[1][0], zones[1][1], mass._table_height + 0.03])
        mass._move_cubes_clear_of_zones(zones)
        mass._env.sim.forward()
        xy = _body_pos(mass, "cube_main")[:2]
        clearance = mass._zone_width / 2 + np.sqrt(2.0) * mass._cube_size
        assert np.min(np.linalg.norm(zones - xy, axis=1)) > clearance
        for name, pos in others.items():
            np.testing.assert_allclose(_body_pos(mass, f"{name}_main"), pos)

    @pytest.mark.parametrize("seed", [0, 1])
    def test_cube_inertia_matches_shuffled_mass(self, mass, seed):
        mass.reset(seed=seed)
        model, side = mass._env.sim.model, 2 * mass._cube_size
        for i, name in enumerate(_cube_names(mass)):
            body = model.body_name2id(f"{name}_main")
            dof = model.jnt_dofadr[model.body_jntadr[body]]
            m = mass._masses[mass._mass_order[i]]
            # dof_M0 is what MuJoCo integrates single-geom free bodies with.
            np.testing.assert_allclose(model._model.dof_M0[dof:dof + 3], m)
            np.testing.assert_allclose(model._model.dof_M0[dof + 3:dof + 6], m * side ** 2 / 6, rtol=1e-3)

    def test_cube_held_above_its_zone_is_not_placed(self, mass):
        mass.reset(seed=0)
        names, zones, targets = _cube_names(mass), _zone_xy(mass), _target_zones(mass)
        rest_z = mass._table_height + mass._cube_size + 0.003
        for i in (0, 1):
            mass._place_object(names[i], [*zones[targets[i]], rest_z])
            mass._set_object_velocity(names[i], [0, 0, 0], [0, 0, 0])
        mass._env.sim.forward()

        # Carry the third cube over its zone and hold it 1.5 cm above the surface.
        cube = _body_pos(mass, f"{names[2]}_main")
        _grasp(mass, cube)
        _move_eef(mass, cube + [0.0, 0.0, 0.06], 1.0)
        above = np.array([*zones[targets[2]], cube[2]])
        _move_eef(mass, above + [0.0, 0.0, 0.06], 1.0)
        ever = False
        for _ in range(80):
            (_, _, terminated, _, info), held = _move_eef(
                mass, above + [0.0, 0.0, 0.015], 1.0, max_steps=1)
            ever |= held
            assert not terminated
        held_z = _body_pos(mass, f"{names[2]}_main")[2] - mass._table_height
        assert held_z > mass._cube_size + 0.01
        assert mass._cube_zone_state()[0] == ["correct", "correct", "unplaced"]
        assert info["task_score"] == pytest.approx(2 / 3)
        assert not ever

        # Releasing it into the zone completes the task.
        for _ in range(60):
            (_, _, terminated, _, info), _ = _move_eef(
                mass, above + [0.0, 0.0, 0.05], -1.0, max_steps=1)
            if terminated:
                break
        assert info["is_success"] and terminated
        assert info["outcome"] == "success"


def test_mass_sort_policy_align_follows_moved_cube(monkeypatch):
    """The scripted ALIGN_ABOVE path re-plans when the cube moves after it was built.

    A path kept aiming at the old hover point parks the arm there for the rest
    of the episode, since the phase exits only near the live cube.
    """
    class _Task:
        _table_height = 0.8

    pick = CubePickPlace(_Task(), {"color": "red", "cube_body_name": "cube_main",
                                   "target_pos": np.zeros(3)})
    cube = np.array([0.0, 0.0, 0.86])
    monkeypatch.setattr(pick, "_cube_pos", lambda: cube.copy())
    pick._advance("ALIGN_ABOVE")
    obs = {"robot0_eef_pos": np.array([-0.1, 0.0, 1.0])}
    pick.act(obs)
    cube = np.array([0.0, -0.03, 0.825])  # it was still dropping and rolled
    pick.act(obs)
    np.testing.assert_allclose(pick._tracker._waypoints[-1], cube + [0.0, 0.0, HOVER_ABOVE_OBJ])


# ── Packing ─────────────────────────────────────────────────────────

@pytest.fixture(scope="class")
def packing():
    task = _make("ib-p7-packing-v1", max_episode_steps=2000)
    yield task
    task.close()


def _bin_pos(task):
    return _body_pos(task, task._box.root_body)


def _set_piece(task, name, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    sim = task._env.sim
    block = task._block_by_name[name]
    sim.data.set_joint_qpos(block.joints[0], np.concatenate([pos, quat]))
    dof = sim.model.jnt_dofadr[sim.model.joint_name2id(block.joints[0])]
    sim.data.qvel[dof:dof + 6] = 0.0
    sim.forward()


def _spawn_qpos(task):
    sim = task._env.sim
    return {n: sim.data.get_joint_qpos(task._block_by_name[n].joints[0]).copy()
            for n in task._block_names}


def _pack_piece(task, name):
    """Teleport *name* flat onto the bin floor at its slot in the generated tiling."""
    piece = task._pieces[task._block_names.index(name)]
    box = _bin_pos(task)
    cx, cy = piece["centroid"]
    _set_piece(task, name, [box[0] + cx, box[1] + cy, box[2] + _BOX_BASE_THICK + 0.001])


def _suction_pick(task, name):
    """Weld *name* to the suction snout at its grasp anchor, then release it."""
    sim = task._env.sim
    gripper = task._env.robots[0].gripper["right"]
    snout = sim.data.geom_xpos[sim.model.geom_name2id(gripper.correct_naming("geo_snout"))]
    piece = task._pieces[task._block_names.index(name)]
    off = np.subtract(piece["grasp_anchor"], piece["centroid"])
    _set_piece(task, name, [snout[0] - off[0], snout[1] - off[1],
                            snout[2] - _PIECE_THICKNESS - 0.0005])
    _step(task, 2, gripper=1.0)
    assert task._piece_ever_picked[name]
    _step(task, 1, gripper=-1.0)


class TestPacking:

    @pytest.mark.parametrize("seed", [0, 1])
    def test_no_success_at_reset_or_zero_action(self, packing, seed):
        obs, info = packing.reset(seed=seed)
        assert not info["is_success"]
        for name in packing._block_names:
            np.testing.assert_allclose(obs[f"{name}_pos"], packing._get_block_positions()[name])
        (_, _, terminated, _, info), ever = _step(packing, 5)
        assert info["task_score"] == 0.0
        assert not ever and not terminated

    def test_all_pieces_packed_succeeds(self, packing):
        packing.reset(seed=0)
        spawn = _spawn_qpos(packing)
        for name in packing._block_names:
            _pack_piece(packing, name)
        (_, _, terminated, _, info), _ = _step(packing, _PIECE_SETTLED_CONTACT_STEPS + 10)
        assert info["is_success"] and terminated
        assert info["task_score"] == 1.0

        # Taking one piece back out ends the success.
        name = packing._block_names[0]
        _set_piece(packing, name, spawn[name][:3], spawn[name][3:])
        (_, _, _, _, info), _ = _step(packing, 2)
        assert not info["is_success"]
        assert info["task_score"] < 1.0

    def test_pieces_packed_one_at_a_time_and_removed_never_succeed(self, packing):
        packing.reset(seed=3)
        spawn = _spawn_qpos(packing)
        ever = False
        for name in packing._block_names:
            _suction_pick(packing, name)
            _pack_piece(packing, name)
            _, packed = _step(packing, _PIECE_SETTLED_CONTACT_STEPS + 5)
            ever |= packed
            assert packing._stage_debug[name]["_success"]
            _set_piece(packing, name, spawn[name][:3], spawn[name][3:])
            (_, _, terminated, _, info), packed = _step(packing, 2)
            ever |= packed
            assert not terminated
            assert packing._stage_debug[name]["placed_in_box_flat"] == 0.0
            assert packing._stage_debug[name]["settled_contact_streak"] == 0.0
        assert not ever
        assert info["task_score"] == pytest.approx(0.20)

    def test_success_scored_on_the_step_pieces_settle(self, packing):
        """Scoring sees this step's suction and bin-contact state, not the last step's."""
        packing.reset(seed=0)
        for name in packing._block_names:
            _pack_piece(packing, name)
        for _ in range(_PIECE_SETTLED_CONTACT_STEPS + 10):
            _, _, terminated, _, info = packing.step(_action(packing, -1.0))
            settled = min(packing._bin_contact_streak.values()) >= _PIECE_SETTLED_CONTACT_STEPS
            assert info["is_success"] == settled
            if terminated:
                break
        assert info["is_success"]

    def test_suction_welds_only_pieces(self, packing):
        for seed in (0, 1):   # three and four pieces
            packing.reset(seed=seed)
            sim = packing._env.sim
            pieces = set().union(*(body_subtree_ids(sim, b.root_body) for b in packing._blocks))
            weldable = set(range(sim.model.nbody)) - packing._suction_weld.excluded_body_ids
            assert weldable == pieces

        # Suction pressed onto the empty bin floor does not weld it.
        box = _bin_pos(packing)
        floor = box + [0.05, 0.0, _BOX_BASE_THICK]
        _move_eef(packing, floor + [0.0, 0.0, 0.08], -1.0)
        for _ in range(150):
            _move_eef(packing, floor, 1.0, max_steps=1)
        sim = packing._env.sim
        gripper = packing._env.robots[0].gripper["right"]
        snout = sim.model.geom_name2id(gripper.correct_naming("geo_snout"))
        assert geoms_touch(sim, {snout}, body_geom_ids(sim, packing._box.root_body))
        assert not packing._suction_weld.welded_body_ids()

    def test_observation_layout_is_fixed_across_piece_counts(self, packing):
        space = packing.observation_space
        for seed, n_pieces in ((0, 3), (1, 4)):
            obs, _ = packing.reset(seed=seed)
            assert len(packing._block_names) == n_pieces
            assert set(obs) == set(space.spaces)
            assert space.contains(obs)
            state, positions = obs["object_state"], packing._get_block_positions()
            for k in range(_MAX_PIECES):
                slot = state[7 * k:7 * k + 7]
                if k < n_pieces:
                    np.testing.assert_allclose(slot[:3], obs[f"lpiece_{k}_pos"])
                    np.testing.assert_allclose(slot[:3], positions[f"lpiece_{k}"])
                else:
                    assert not slot.any()
                    assert not obs[f"lpiece_{k}_pos"].any() and not obs[f"lpiece_{k}_quat"].any()
            assert space.contains(packing.step(_action(packing))[0])

    def test_config_overrides_reach_task_config(self):
        task = _make("ib-p7-packing-v1", max_episode_steps=2000,
                     config_overrides={"randomization": {"piece_count_range": [4, 4]}})
        try:
            for seed in range(3):
                task.reset(seed=seed)
                assert len(task._block_names) == 4
        finally:
            task.close()


# ── Tool retrieve ───────────────────────────────────────────────────

@pytest.fixture(scope="class", params=TOOL_TASKS)
def tool(request):
    task = _make(request.param, max_episode_steps=2000)
    yield task
    task.close()


class TestToolRetrieve:

    def test_no_success_at_reset_or_zero_action(self, tool):
        obs, _ = tool.reset(seed=0)
        np.testing.assert_allclose(obs["object_pos"], _body_pos(tool, "cube_main"))
        (_, _, terminated, _, info), ever = _step(tool, 20)
        assert not ever and not terminated

    def test_grasp_and_lift_cube_succeeds(self, tool):
        tool.reset(seed=0)
        # Put the cube within reach, clear of the hook, then grasp and lift it.
        hook_y = _body_pos(tool, "hook_tool_root")[1]
        cube = np.array([-0.05, hook_y - np.sign(hook_y or 1.0) * 0.13,
                         tool._table_height + tool._cube_size])
        _place_at_rest(tool, "cube", cube)
        _grasp(tool, cube)
        ever = False
        for _ in range(300):
            (_, _, terminated, _, info), held = _move_eef(
                tool, cube + [0.0, 0.0, tool._lift_height + 0.03], 1.0, max_steps=1)
            ever |= held
            if terminated:
                break
        assert ever and info["is_success"] and terminated
        assert _body_pos(tool, "cube_main")[2] - tool._table_height >= tool._lift_height


# =====================================================================
#  Pendulum, seesaw, slip recovery
# =====================================================================

def _move_toward(task, obs, target, gripper):
    """One step driving the end effector toward *target*."""
    delta = np.clip(10.0 * (np.asarray(target) - obs["robot0_eef_pos"]), -1.0, 1.0)
    return task.step(_action(task, gripper, delta))


def _set_hinge(task, joint, angle):
    """Teleport the 1-dof *joint* to *angle* at rest."""
    sim = task._env.sim
    jid = sim.model.joint_name2id(joint)
    sim.data.qpos[sim.model.jnt_qposadr[jid]] = angle
    sim.data.qvel[sim.model.jnt_dofadr[jid]] = 0.0
    sim.forward()


def _hinge_angle(task, joint):
    sim = task._env.sim
    return float(sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint_name2id(joint)]])


# ── Pendulum grasp ──────────────────────────────────────────────────

class TestPendulumGrasp:

    @pytest.mark.parametrize("task_id", PENDULUM_TASKS)
    def test_bob_stopped_above_open_gripper_without_contact_does_not_succeed(self, task_id):
        """The "resting on gripper" success branch requires gripper contact with the bob."""
        task = _make(task_id)
        try:
            obs, _ = task.reset(seed=0)
            apex_angle = _hinge_angle(task, "pendulum_hinge")
            # Just below the bob held at its apex, inside the branch's xy
            # window but clear of the open fingers.
            target = task._get_bob_pos() + [0.058, 0.0, -0.01]
            for _ in range(150):
                _set_hinge(task, "pendulum_hinge", apex_angle)
                obs, *_ = _move_toward(task, obs, target, -1.0)
            for _ in range(10):
                _set_hinge(task, "pendulum_hinge", apex_angle)
                score = task.task_score(recompute=True)
            bob, eef = task._get_bob_pos(), obs["robot0_eef_pos"]
            # Every geometric condition of the resting branch holds.
            assert bob[2] - eef[2] >= task.RESTING_Z_OVER_EEF_MIN
            assert np.linalg.norm(bob[:2] - eef[:2]) <= task.RESTING_XY_TOL
            assert task._hinge_velocity() <= task.RESTING_VEL_TOL
            assert abs(bob[2] - task._apex_z) <= task.APEX_TOLERANCE
            sim = task._env.sim
            assert not robot_contacts(sim, {sim.model.geom_name2id("pendulum_bob")})
            assert score < 1.0
            assert task._stage_debug["_success"] is False
            assert not task._grasped
        finally:
            task.close()

    @pytest.mark.parametrize("seed", [0, 1])  # catches from the +Y and the -Y side
    def test_scripted_policy_pinches_bob_at_apex(self, seed):
        task = _make("ib-p2-pendulum-grasp-v1")
        try:
            obs, _ = task.reset(seed=seed)
            policy = collection.build_policy(PendulumGraspPolicy, task, np.random.default_rng(seed))
            for _ in range(task.MAX_EPISODE_STEPS):
                action = collection.clip_action(policy.act(obs, env=task), task.action_space)
                obs, _, terminated, truncated, info = task.step(action)
                if terminated or truncated:
                    break
            assert info["is_success"] is True
            assert terminated
            assert task._bob_pinched()
            assert abs(task._get_bob_pos()[2] - task._apex_z) <= task.APEX_TOLERANCE
            assert task._contact_apex_err <= task.APEX_TOLERANCE
        finally:
            task.close()

    def test_bob_pinched_below_apex_and_carried_up_does_not_succeed(self):
        task = _make("ib-p2-pendulum-grasp-v1")
        try:
            obs, _ = task.reset(seed=0)
            apex_angle = _hinge_angle(task, "pendulum_hinge")
            _set_hinge(task, "pendulum_hinge", 0.0)
            bottom = task._get_bob_pos()
            radius = task._arm_length + task._bob_size
            pivot_z = task._table_height + task._stand_height

            def side_grasp_step(target, gripper):
                delta = np.clip(12.0 * (target - obs["robot0_eef_pos"]), -1.0, 1.0)
                action = _action(task, gripper, delta)
                rot_err = _quat_err_axis_angle(obs["robot0_eef_quat"], SIDE_QUAT_MINUS_Y)
                action[3:6] = np.clip(3.5 * rot_err, -0.5, 0.5)
                return task.step(action)

            # Pinch the bob at the bottom of its arc, held there from the -Y side.
            for t in range(410):
                _set_hinge(task, "pendulum_hinge", 0.0)
                target = bottom + [0.0, -0.12, 0.0] if t < 250 else bottom
                obs, _, _, _, info = side_grasp_step(target, -1.0 if t < 370 else 1.0)
                assert info["is_success"] is False
            assert task._bob_pinched()
            # Carry it along its arc up to the apex height.
            pinched_at_apex = False
            for t in range(250):
                theta = min(abs(apex_angle), 0.003 * (t + 1))
                target = np.array([bottom[0], bottom[1] - radius * np.sin(theta),
                                   pivot_z - radius * np.cos(theta)])
                obs, _, terminated, _, info = side_grasp_step(target, 1.0)
                at_apex = abs(task._get_bob_pos()[2] - task._apex_z) <= task.APEX_TOLERANCE
                pinched_at_apex |= at_apex and task._bob_pinched()
                assert info["is_success"] is False and not terminated
            assert pinched_at_apex
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", PENDULUM_TASKS)
    def test_bob_resting_on_gripper_counts_only_with_closed_jaws(self, task_id):
        task = _make(task_id)
        try:
            obs, _ = task.reset(seed=0)
            apex_angle = _hinge_angle(task, "pendulum_hinge")
            # The bob, held at its apex, touches the open gripper near the palm.
            target = task._get_bob_pos() - [0.0, 0.0, 0.06]
            for _ in range(210):
                _set_hinge(task, "pendulum_hinge", apex_angle)
                obs, _, _, _, info = _move_toward(task, obs, target, -1.0)
                assert info["is_success"] is False
            bob, eef = task._get_bob_pos(), obs["robot0_eef_pos"]
            assert task._robot_touches_bob() and not task._bob_pinched()
            assert bob[2] - eef[2] >= task.RESTING_Z_OVER_EEF_MIN
            assert np.linalg.norm(bob[:2] - eef[:2]) <= task.RESTING_XY_TOL
            assert task._hinge_velocity() <= task.RESTING_VEL_TOL
            assert task._contact_apex_err <= task.APEX_TOLERANCE
            assert task._resting_steps == 0
            # Closing the jaws under the bob makes it rest on the closed gripper.
            for _ in range(60):
                _set_hinge(task, "pendulum_hinge", apex_angle)
                obs, _, terminated, _, info = _move_toward(task, obs, target, 1.0)
                if terminated:
                    break
            assert info["is_success"] is True
            assert task._resting_steps >= task.RESTING_HOLD_STEPS and not task._bob_pinched()
        finally:
            task.close()

    def test_geometry_follows_constructor_seed(self):
        geometry = []
        for seed in (0, 2):
            task = _make("ib-p2-pendulum-grasp-v1", seed=seed)
            task.close()
            geometry.append((task._arm_length, task._stand_height, task._bob_size))
            lo, hi = task.BOB_CLEARANCE_RANGE
            assert lo <= task._stand_height - task._arm_length - 2 * task._bob_size <= hi
        assert geometry[0] != geometry[1]

    @pytest.mark.parametrize("task_id", PENDULUM_TASKS)
    def test_unused_swing_amplitude_randomization_key_removed(self, task_id):
        assert "swing_amplitude_range" not in load_task_config(task_id)["randomization"]["pendulum"]


# ── Seesaw balance ──────────────────────────────────────────────────

@pytest.fixture(scope="class")
def seesaw():
    task = _make("ib-p3-seesaw-balance-v1", max_episode_steps=600,
                 config_overrides={"task": {"balance_hold_steps": 50}})
    yield task
    task.close()


def _plank_top(task):
    return task._table_height + task._fulcrum_height + task._seesaw_thickness


def _ideal_counterweight_spot(task):
    return np.array([task._fulcrum_x - task._ideal_counter_arm, 0.0, _plank_top(task) + 0.021])


class TestSeesawBalance:

    def test_doing_nothing_earns_no_credit(self, seesaw):
        seesaw.reset(seed=0)
        for _ in range(30):
            _, _, _, _, info = seesaw.step(_action(seesaw))
            assert info["task_score"] == 0.0
            assert info["is_success"] is False

    @pytest.mark.parametrize("remove_target", [False, True])
    def test_level_plank_without_counterweight_does_not_count(self, seesaw, remove_target):
        seesaw.reset(seed=0)
        if remove_target:
            _place_at_rest(seesaw, "cube", [seesaw._fulcrum_x, 0.25, seesaw._table_height + 0.021])
        for _ in range(seesaw._balance_hold_steps + 20):
            _set_hinge(seesaw, "seesaw_hinge", 0.0)
            _, _, _, _, info = seesaw.step(_action(seesaw))
            assert seesaw._balance_counter == 0
            assert info["is_success"] is False

    def test_counterweight_at_ideal_arm_balances_and_succeeds(self, seesaw):
        seesaw.reset(seed=0)
        _place_at_rest(seesaw, "weight_0", _ideal_counterweight_spot(seesaw))
        (_, _, _, _, info), _ = _step(seesaw, seesaw._balance_hold_steps + 100)
        assert info["is_success"] is True
        assert seesaw._weight_placed and seesaw._target_on_plank

    def test_target_block_moved_toward_fulcrum_does_not_count(self, seesaw):
        seesaw.reset(seed=0)
        moved_arm = 0.03
        assert seesaw._target_arm - moved_arm > seesaw.TARGET_ARM_TOL
        top = _plank_top(seesaw) + 0.021
        _place_at_rest(seesaw, "cube", [seesaw._fulcrum_x + moved_arm, 0.0, top])
        # Counterweight at the arm that balances the moved block.
        counter_arm = seesaw._target_mass_actual * moved_arm / seesaw._counter_mass_actual
        _place_at_rest(seesaw, "weight_0", [seesaw._fulcrum_x - counter_arm, 0.0, top])
        for _ in range(seesaw._balance_hold_steps + 100):
            _, _, terminated, _, info = seesaw.step(_action(seesaw))
            assert seesaw._balance_counter == 0
            assert info["is_success"] is False and not terminated
        assert seesaw._get_seesaw_angle_deg() <= seesaw._tip_angle_threshold
        assert seesaw._weight_placed and not seesaw._target_on_plank

    def test_counterweight_held_by_robot_does_not_count(self, seesaw):
        """A pinch through the tactile taxels counts as holding the counterweight."""
        obs, _ = seesaw.reset(seed=0)
        spot = _ideal_counterweight_spot(seesaw)
        for _ in range(150):
            _set_hinge(seesaw, "seesaw_hinge", 0.0)
            obs, *_ = _move_toward(seesaw, obs, spot, -1.0)
        eef = obs["robot0_eef_pos"]
        _place_at_rest(seesaw, "weight_0", [eef[0], eef[1], spot[2]])
        for _ in range(seesaw._balance_hold_steps + 30):
            obs, _, _, _, info = seesaw.step(_action(seesaw, 1.0))
            assert info["is_success"] is False
        assert seesaw._robot_touches("weight_0") and seesaw._weight_on_plank
        assert seesaw._balance_counter == 0
        for _ in range(seesaw._balance_hold_steps + 60):
            obs, _, terminated, _, info = _move_toward(seesaw, obs, eef + [0.0, 0.0, 0.15], -1.0)
            if terminated:
                break
        assert not seesaw._robot_touches("weight_0")
        # This is a contact-gating probe, not the press-to-level controller.
        # With reduced hinge friction, merely releasing the cube can leave
        # the plank tipped. Initialize an accurately balanced, level state
        # once, then require the released objects to sustain it physically.
        _set_hinge(seesaw, "seesaw_hinge", 0.0)
        _place_at_rest(seesaw, "cube", [seesaw._fulcrum_x + seesaw._target_arm,
                       0.0, _plank_top(seesaw) + 0.021])
        _place_at_rest(seesaw, "weight_0", _ideal_counterweight_spot(seesaw))
        (_, _, _, _, info), _ = _step(seesaw, seesaw._balance_hold_steps + 60)
        assert info["is_success"] is True


# ── Slip recovery ───────────────────────────────────────────────────

#: Drops ``(xy, lowest-point height above the table, quat, angular velocity)``
#: that libccd's MPR collision launched off the table.
LAUNCH_DROPS = {
    "strawberry": (
        [-0.018531459049233368, -0.023827028324671484], 0.05762782980115408,
        [-0.6598070614025444, 0.5851551879952629, -0.013769175871055103, 0.47123079003965684],
        [0.38386994512884254, -1.8910894563108704, 2.8629820360681926]),
    "a_lego_duplo": (
        [0.05457910158592666, 0.0010463624797459325], 0.10113630885248617,
        [0.16851715683178775, 0.9195067899243345, 0.3167506742864787, 0.16055603842720323],
        [-0.8491612697628463, -0.38639824313341586, -0.8633341134699783]),
}


@pytest.fixture(scope="class", params=SLIP_TASKS)
def slip(request):
    # ``_grasp_in_air`` pins the object while the jaws shut, which wedges some
    # hulls between them; the lemon drops out once they open.
    task = _make(request.param, max_episode_steps=600, config_overrides={"task": {
        "ycb_pool": ["lemon"], "slippage": {"hold_steps_range": [5, 5]}}})
    yield task
    task.close()


def _slips(task):
    """``(fired, released, recovered)`` per planned slip."""
    if hasattr(task, "_slip_count"):
        fired = [i < task._slip_count for i in range(len(task._recovered_slips))]
        return fired, list(task._released_slips), list(task._recovered_slips)
    return [task._slipped], [task._released_after_slip], [task._recovered_after_slip]


def _grasp_in_air(task, steps=25):
    """Close the gripper on the object, holding it at the grip site until the jaws shut."""
    for _ in range(steps):
        _place_at_rest(task, task._active_obj_name(), _grip_site(task))
        task.step(_action(task, 1.0))


class TestSlipRecovery:

    def test_slip_dropping_object_into_goal_does_not_succeed(self, slip):
        slip.reset(seed=0)
        goal = slip._goal_pos
        _place_at_rest(slip, slip._active_obj_name(), [goal[0], goal[1], slip._table_height + 0.12])
        (_, _, terminated, _, info), _ = _step(slip, 80)
        assert info["is_success"] is False
        assert info["task_score"] < 1.0
        assert not terminated
        raw_obs = slip._get_sim_observations(force_update=True)
        slip._inject_active_pose(raw_obs)
        assert slip._in_goal_region(raw_obs) and slip._object_touches_table() and slip._was_lifted
        fired, _, recovered = _slips(slip)
        assert fired[0] and not any(recovered)

    def test_object_caught_before_landing_is_not_recovered(self, slip):
        slip.reset(seed=0)
        _grasp_in_air(slip, steps=slip._slip_open_window + 25)
        fired, released, recovered = _slips(slip)
        assert fired[0] and not released[0] and not recovered[0]
        held = robot_contacts(slip._env.sim, object_geom_ids(slip._env, slip._active_obj_name()))
        assert held and _grip_site(slip)[2] > slip._table_height + slip._slip_min_lift

    def test_every_slip_recovered_then_placed_in_goal_succeeds(self, slip):
        slip.reset(seed=0)
        n_slips = len(_slips(slip)[0])
        for i in range(n_slips):
            _grasp_in_air(slip)
            _step(slip, 60)
            fired, released, _ = _slips(slip)
            assert fired[i] and released[i]
        _grasp_in_air(slip)
        assert all(_slips(slip)[2])
        _step(slip, 3)
        goal = slip._goal_pos
        _place_at_rest(slip, slip._active_obj_name(), [goal[0], goal[1], slip._table_height + 0.02])
        (_, _, terminated, _, info), _ = _step(slip, 60)
        assert info["is_success"] is True
        assert terminated

    def test_object_hovering_over_goal_does_not_touch_table(self, slip):
        slip.reset(seed=0)
        name = slip._active_obj_name()
        goal = slip._goal_pos
        _place_at_rest(slip, name, [goal[0], goal[1], slip._table_height + 0.1])
        lift = slip._table_height + 0.012 - slip._lowest_collision_z(name)
        _place_at_rest(slip, name, [goal[0], goal[1], slip._table_height + 0.1 + lift])
        assert not slip._object_touches_table()
        _step(slip, 40)
        assert slip._object_touches_table()

    def test_object_held_over_goal_succeeds_only_after_release(self, slip):
        slip.reset(seed=0)
        for _ in _slips(slip)[0]:
            _grasp_in_air(slip)
            _step(slip, 60)
        _grasp_in_air(slip)
        assert all(_slips(slip)[2])
        name, goal = slip._active_obj_name(), slip._goal_pos
        obs = slip._get_sim_observations(force_update=True)
        hold_steps = 0
        for _ in range(400):
            # Carry the gripped object over the goal, its lowest point 1 cm above the table.
            eef = obs["robot0_eef_pos"]
            gap = slip._lowest_collision_z(name) - slip._table_height
            target = np.array([goal[0], goal[1], eef[2] + 0.01 - gap])
            if np.linalg.norm(eef[:2] - goal) > 0.02:
                target[2] = max(target[2], eef[2])
            obs, _, terminated, _, info = _move_toward(slip, obs, target, 1.0)
            assert info["is_success"] is False and not terminated
            settled = np.linalg.norm(eef[:2] - goal) < 0.01 and abs(gap - 0.01) < 0.002
            hold_steps = hold_steps + 1 if settled else 0
            if hold_steps >= 30:
                break
        assert hold_steps >= 30
        raw_obs = slip._get_sim_observations(force_update=True)
        slip._inject_active_pose(raw_obs)
        assert slip._in_goal_region(raw_obs)
        assert robot_contacts(slip._env.sim, object_geom_ids(slip._env, name))
        (_, _, terminated, _, info), _ = _step(slip, 80)
        assert info["is_success"] is True
        assert terminated

    @pytest.mark.parametrize("ycb", ["gelatin_box", "lemon"])
    def test_spawned_object_rests_on_table_without_falling(self, ycb):
        task = _make("ib-p6-slip-recovery-v1", config_overrides={"task": {"ycb_pool": [ycb]}})
        try:
            task.reset(seed=0)
            sim = task._env.sim
            body = sim.model.body_name2id(f"{task._active_obj_name()}_main")
            z0 = sim.data.body_xpos[body][2]
            (_, _, terminated, _, _), _ = _step(task, 60)
            assert not terminated
            assert task._object_touches_table()
            assert 0.0 < z0 - sim.data.body_xpos[body][2] < 0.012
        finally:
            task.close()

    def test_unused_objects_are_parked_off_scene(self):
        task = _make("ib-p6-slip-recovery-v1")
        try:
            task.reset(seed=0)
            parked = task._parked_objects()
            assert parked == {task._ycb_obj_name(y) for y in task._ycb_pool} - {task._active_obj_name()}
            _step(task, 10)
            sim = task._env.sim
            for name in parked:
                pos = sim.data.body_xpos[sim.model.body_name2id(f"{name}_main")]
                assert pos[2] > 50.0 and np.linalg.norm(pos[:2]) > 50.0
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", SLIP_TASKS)
    def test_recomputed_score_tracks_the_active_object(self, task_id):
        # robosuite aliases ``cube_pos`` to the first pool object, strawberry.
        task = _make(task_id, max_episode_steps=600, config_overrides={"task": {
            "ycb_pool": ["strawberry", "lemon"], "slippage": {"hold_steps_range": [5, 5]}}})
        try:
            seed = 0
            task.reset(seed=seed)
            while task._selected_ycb != "lemon":
                seed += 1
                task.reset(seed=seed)
            assert task._ycb_obj_name("strawberry") in task._parked_objects()
            _step(task, 3)
            assert task.task_score(recompute=True) == task.task_score() == 0.0
            assert not task._was_lifted
            for _ in _slips(task)[0]:
                _grasp_in_air(task)
                _step(task, 60)
            _grasp_in_air(task)
            _step(task, 3)
            goal = task._goal_pos
            _place_at_rest(task, task._active_obj_name(), [goal[0], goal[1], task._table_height + 0.02])
            (_, _, terminated, _, info), _ = _step(task, 60)
            assert info["is_success"] and terminated
            assert task.task_score(recompute=True) == task.task_score() == 1.0
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", SLIP_TASKS)
    @pytest.mark.parametrize("ycb", sorted(LAUNCH_DROPS))
    def test_dropped_object_gains_no_energy_on_impact(self, task_id, ycb):
        xy, height, quat, angvel = LAUNCH_DROPS[ycb]
        task = _make(task_id, config_overrides={"task": {
            "ycb_pool": [ycb], "slippage": {"enabled": False}, "goal_xy_range": [0.0, 0.0]}})
        try:
            task.reset(seed=0)
            sim, name, z = task._env.sim, task._active_obj_name(), task._table_height
            model, data = sim.model._model, sim.data._data
            body = model.body(f"{name}_main").id
            mass, inertia, g = model.body_mass[body], model.body_inertia[body], -model.opt.gravity[2]

            def energy():
                # Angular then linear velocity at the COM, in the principal-axis frame.
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 1)
                return (0.5 * (inertia @ vel[:3] ** 2 + mass * vel[3:] @ vel[3:])
                        + mass * g * (data.xipos[body][2] - z))

            # Release the object spinning, its lowest point ``height`` above the table.
            task._place_object(name, [xy[0], xy[1], z], quat=quat)
            sim.forward()
            lift = z + height - task._lowest_collision_z(name)
            task._place_object(name, [xy[0], xy[1], z + lift], quat=quat)
            task._set_object_velocity(name, [0, 0, 0], angvel)
            sim.forward()
            released, peak, landed = energy(), 0.0, False
            for _ in range(90):
                task.step(_action(task))
                landed |= task._object_touches_table()
                if landed:
                    peak = max(peak, energy())
            assert landed and peak <= 1.05 * released
            assert task._object_touches_table()
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", SLIP_TASKS)
    def test_slip_trigger_distance_must_clear_goal_square(self, task_id):
        with pytest.raises(ValueError, match="min_goal_dist"):
            _make(task_id, config_overrides={"task": {"slippage": {"min_goal_dist": 0.10}}})


# =====================================================================
#  Occluder push, recover peg insert, sheltered grasp
# =====================================================================

#: Peg orientation with its long axis (local +Z) along world +Y, the slot axis.
PEG_ALONG_SLOT = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
#: Yaw of 90 degrees: puts the sheltered cube's long side along world x.
YAW_90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]


def _hold_over(task, name, xy, z, steps=150):
    """Grasp object *name* where it rests, carry it over *xy* and hold it there.

    The end effector is held at height *z*. Returns whether success held at
    any step.
    """
    obj = _body_pos(task, f"{name}_main")
    ever = _grasp(task, obj)
    ever |= _move_eef(task, [obj[0], obj[1], task._table_height + 0.15], 1.0)[1]
    ever |= _move_eef(task, [xy[0], xy[1], task._table_height + 0.15], 1.0)[1]
    for _ in range(steps):
        ever |= _move_eef(task, [xy[0], xy[1], z], 1.0, max_steps=1)[1]
    return ever


def _slide(task, start, velocity, steps=150):
    """Launch the cube from *start* with linear *velocity*, the arm still and open.

    Returns whether success held at any step, whether the episode terminated
    and the cube's final position.
    """
    task._place_object("cube", start)
    task._set_object_velocity("cube", velocity)
    task._env.sim.forward()
    (_, _, terminated, _, _), ever = _step(task, steps)
    return ever, terminated, _body_pos(task, "cube_main")


def _contact_penetration(task, geoms_a, geoms_b=None):
    """Deepest current penetration (m, >= 0) between *geoms_a* and *geoms_b* (default: anything)."""
    sim = task._env.sim
    depth = 0.0
    for c in sim.data.contact[:sim.data.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        if geoms_b is None:
            hit = g1 in geoms_a or g2 in geoms_a
        else:
            hit = (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b)
        if hit:
            depth = max(depth, -float(c.dist))
    return depth


# ── Occluder push ───────────────────────────────────────────────────

@pytest.fixture(scope="class", params=OCCLUDER_TASKS)
def occluder(request):
    task = _make(request.param, max_episode_steps=3000)
    yield task
    task.close()


def _lid_names(task):
    return [o.name for o in task._env._dynamic_objects if o.name.startswith("lid")]


class TestOccluderPush:

    @pytest.mark.parametrize("seed", range(6))
    def test_lids_spawn_without_overlap(self, occluder, seed):
        obs, _ = occluder.reset(seed=seed)
        np.testing.assert_allclose(obs["object_pos"], _body_pos(occluder, "cube_main"))
        parked = occluder._parked_objects()
        stacks = {}
        for name in _lid_names(occluder):
            if name not in parked:
                xy = tuple(np.round(_body_pos(occluder, f"{name}_main")[:2], 6))
                stacks.setdefault(xy, []).append(name)
        assert len(stacks) == 1 + occluder._num_distractors
        centres = np.array(list(stacks))
        for i in range(len(centres)):
            for j in range(i + 1, len(centres)):
                gap = np.abs(centres[i] - centres[j]) - 2 * occluder._lid_half_size
                assert gap.max() > 0.0, "lid footprints overlap"
        lid_geoms = set().union(*(object_geom_ids(occluder._env, n) for n in _lid_names(occluder)))
        assert _contact_penetration(occluder, lid_geoms) < 1e-3

        (_, _, terminated, _, info), ever = _step(occluder, 5)
        assert info["task_score"] == 0.0
        assert not ever and not terminated

    def test_lids_that_do_not_fit_raise(self, occluder):
        with pytest.raises(ValueError, match="do not fit"):
            occluder._generate_cube_positions(5)

    def test_cube_held_over_patch_is_not_placed(self, occluder):
        occluder.reset(seed=0)
        patch, z0 = occluder._patch_pos, occluder._table_height
        rest_z = z0 + occluder._cube_size + 0.001
        occluder._place_object("cube", [patch[0], patch[1] - 0.13, rest_z])
        _step(occluder, 10)
        assert not _hold_over(occluder, "cube", patch, z0 + 0.035)
        # Held clear of the patch, centre below the 5 cm placement height limit.
        cube = _body_pos(occluder, "cube_main")
        assert np.linalg.norm(cube[:2] - patch) < occluder._patch_tolerance
        assert z0 + 0.002 < cube[2] - occluder._cube_size and cube[2] < z0 + 0.05
        assert robot_contacts(occluder._env.sim, object_geom_ids(occluder._env, "cube"))
        assert occluder._stage_debug["place_on_patch"] == 0.0

        (_, _, terminated, _, info), ever = _step(occluder, 80, gripper=-1.0)
        assert ever and terminated
        assert info["is_success"] and info["task_score"] == 1.0

    def test_cube_sliding_off_patch_is_not_placed(self, occluder):
        occluder.reset(seed=0)
        patch = occluder._patch_pos
        # On the patch, inside the placement band, moving outward.
        z = occluder._table_height + 0.002 + occluder._cube_size + 0.0003
        ever, terminated, cube = _slide(occluder, [patch[0], patch[1] + 0.02, z], [0.0, 1.2, 0.0])
        assert not ever and not terminated
        assert np.linalg.norm(cube[:2] - patch) > occluder._patch_tolerance


def test_occluder_v2_builds_and_parks_lid_layers():
    task = _make("ib-p5-occluder-push-v2", max_episode_steps=3000)
    try:
        seen = set()
        for seed in range(8):
            task.reset(seed=seed)
            n = task._num_lid_layers
            seen.add(n)
            stacks = ["lid_layer_{k}"] + [
                f"lid_{i}_layer_{{k}}" for i in range(task._num_distractors)]
            expected_parked = {s.format(k=k) for s in stacks for k in range(n, 3)}
            assert task._parked_objects() == expected_parked
            for name in expected_parked:
                assert _body_pos(task, f"{name}_main")[2] > 50.0
            cube = _body_pos(task, "cube_main")
            lids = [_body_pos(task, f"lid_layer_{k}_main") for k in range(n)]
            for k, lid in enumerate(lids):
                np.testing.assert_allclose(lid[:2], cube[:2])
                assert lid[2] > (lids[k - 1] if k else cube)[2]
        assert seen == {2, 3}
    finally:
        task.close()


# ── Recover peg insert ──────────────────────────────────────────────

@pytest.fixture(scope="class", params=PEG_TASKS)
def peg(request):
    task = _make(request.param, max_episode_steps=3000)
    yield task
    task.close()


def _place_peg(task, axial, lateral=0.0, z=None):
    """Teleport the peg, long axis along the slot, *axial* m past the entrance.

    *lateral* shifts it sideways (world x); *z* overrides its height.
    """
    pos = task._hole_center_world + axial * task._hole_normal_world + [lateral, 0.0, 0.0]
    if z is not None:
        pos[2] = z
    task._place_object("cube", pos.tolist(), quat=PEG_ALONG_SLOT)
    task._env.sim.forward()


class TestRecoverPegInsert:

    @pytest.mark.parametrize("seed", [0, 1])
    def test_no_credit_at_reset_or_zero_action(self, peg, seed):
        obs, _ = peg.reset(seed=seed)
        np.testing.assert_allclose(obs["object_pos"], _body_pos(peg, "cube_main"))
        (_, _, terminated, _, info), ever = _step(peg, 5)
        assert info["task_score"] == 0.0
        assert not ever and not terminated

    @pytest.mark.parametrize("axial", ["past_exit", 0.30, 0.45])
    def test_peg_past_slot_exit_is_not_inserted(self, peg, axial):
        peg.reset(seed=0)
        if axial == "past_exit":
            axial = peg._max_insert_depth + 0.005
        _place_peg(peg, axial)
        assert peg.task_score(recompute=True) < 1.0
        assert not peg._stage_debug["_success"]
        assert peg._stage_debug["insertion_depth"] == 0.0
        assert not peg._check_termination(peg._get_sim_observations(force_update=True))

    def test_peg_beside_slot_gets_no_depth_credit(self, peg):
        peg.reset(seed=0)
        _place_peg(peg, peg._insert_depth_threshold, lateral=0.03)
        assert peg.task_score(recompute=True) < 1.0
        assert peg._stage_debug["insertion_depth"] == 0.0

    @pytest.mark.parametrize("where", ["min", "mid", "max"])
    def test_peg_resting_in_slot_succeeds(self, peg, where):
        peg.reset(seed=0)
        lo, hi = peg._insert_depth_threshold, peg._max_insert_depth
        axial = {"min": lo + 0.002, "mid": 0.5 * (lo + hi), "max": hi - 0.002}[where]
        # Resting on the slot floor (fixture-local z 0.31).
        _place_peg(peg, axial, z=peg._table_height + 0.31 + peg._peg_half_cross + 0.001)
        (_, _, terminated, _, info), ever = _step(peg, 10)
        assert ever and terminated
        assert info["is_success"] and info["task_score"] == 1.0
        # The bounds keep insert_fraction of the peg between the slot's two end planes.
        hl = peg._peg_half_length
        inside = min(axial + hl, _SLOT_LENGTH) - max(axial - hl, 0.0)
        assert inside >= peg._insert_fraction * 2 * hl - 1e-9


def test_upright_peg_on_table_is_not_grasped():
    task = _make("ib-p1-recover-peg-insert-v1", max_episode_steps=3000)
    try:
        task.reset(seed=0)
        peg_pos = _body_pos(task, "cube_main")
        _grasp(task, peg_pos + [0.0, 0.0, 0.02])
        assert robot_contacts(task._env.sim, object_geom_ids(task._env, "cube"))
        assert not task._peg_is_grasped()
        assert task._stage_debug["grasp_peg"] == 0.0
        _move_eef(task, peg_pos + [0.0, 0.0, 0.08], 1.0)
        assert task._peg_is_grasped()
        assert task._stage_debug["grasp_peg"] == 1.0
    finally:
        task.close()


# ── Sheltered grasp ─────────────────────────────────────────────────

@pytest.fixture(scope="class", params=SHELTERED_TASKS)
def sheltered(request):
    task = _make(request.param, max_episode_steps=3000)
    yield task
    task.close()


class TestShelteredGrasp:

    @pytest.mark.parametrize("seed", range(6))
    def test_cube_spawns_resting_clear_of_rack(self, sheltered, seed):
        obs, _ = sheltered.reset(seed=seed)
        cube = _body_pos(sheltered, "cube_main")
        np.testing.assert_allclose(obs["object_pos"], cube)
        cube_geoms = object_geom_ids(sheltered._env, "cube")
        rack_geoms = object_geom_ids(sheltered._env, "rack")
        assert _contact_penetration(sheltered, cube_geoms, rack_geoms) < 1e-3
        floor = sheltered._STORY_FLOOR_Z[sheltered._cube_story]
        assert cube[2] - sheltered._table_height == pytest.approx(
            floor + sheltered._cube_size, abs=2e-3)

        (_, _, terminated, _, info), ever = _step(sheltered, 20)
        assert np.linalg.norm(_body_pos(sheltered, "cube_main") - cube) < 3e-3
        assert not ever and not terminated
        assert info["task_score"] < 1.0

    def test_cube_held_over_tray_is_not_placed(self, sheltered):
        sheltered.reset(seed=0)
        tray, z0 = sheltered._tray_pos, sheltered._table_height
        rest_z = z0 + sheltered._cube_size + 0.001
        sheltered._place_object("cube", [tray[0] + 0.13, tray[1], rest_z], quat=YAW_90)
        _step(sheltered, 10)
        assert not _hold_over(sheltered, "cube", tray, z0 + 0.045)
        # Held clear of the tray, centre below the 6 cm placement height limit.
        cube = _body_pos(sheltered, "cube_main")
        assert np.linalg.norm(cube[:2] - tray) < sheltered._tray_tolerance
        assert z0 + 0.002 < cube[2] - sheltered._cube_size and cube[2] < z0 + 0.06
        assert robot_contacts(sheltered._env.sim, object_geom_ids(sheltered._env, "cube"))

        (_, _, terminated, _, info), ever = _step(sheltered, 80, gripper=-1.0)
        assert ever and terminated
        assert info["is_success"] and info["task_score"] == 1.0

    @pytest.mark.parametrize("offset, velocity", [
        ([0.12, 0.0], [-2.0, 0.0, 0.0]),
        ([0.0, 0.02], [0.0, 1.2, 0.0]),
    ], ids=["through", "off"])
    def test_cube_sliding_across_tray_is_not_placed(self, sheltered, offset, velocity):
        sheltered.reset(seed=0)
        tray = sheltered._tray_pos
        z = sheltered._table_height + 0.002 + sheltered._cube_size + 0.0003
        start = [tray[0] + offset[0], tray[1] + offset[1], z]
        ever, terminated, cube = _slide(sheltered, start, velocity)
        assert not ever and not terminated
        assert np.linalg.norm(cube[:2] - tray) > sheltered._tray_tolerance


def test_sheltered_v2_reads_task_yaml_and_fixes_only_the_rack_pose():
    overrides = {
        "task": {"tray_pos": [-0.12, 0.05], "placement_threshold": 0.04},
        "randomization": {
            "rack_position": {"enabled": True},
            "rack_orientation": {"enabled": True},
            "cube_story": {"enabled": False, "default": "top"},
        },
    }
    task = _make("ib-p1-sheltered-grasp-v2", max_episode_steps=3000, config_overrides=overrides)
    try:
        task.reset(seed=0)
        np.testing.assert_allclose(task._tray_pos, [-0.12, 0.05])
        assert task._tray_tolerance == 0.04
        assert task._cube_story == "top"
        np.testing.assert_allclose(task._rack_pos, task._RACK_POS)
        assert task._rack_yaw == task._RACK_YAW
        np.testing.assert_allclose(_body_pos(task, "tray_main")[:2], [-0.12, 0.05])
    finally:
        task.close()


@pytest.mark.parametrize("num_chains", [5, 6, 8])
@pytest.mark.parametrize("task_id", ["ib-p3-domino-select-v1", "ib-p3-domino-select-v2"])
def test_domino_select_chains_do_not_touch_at_reset(task_id, num_chains):
    """No domino of one chain touches another chain at reset (the gap decoy stays in its lane)."""
    task = _make(task_id, config_overrides={"task": {"num_chains": num_chains}})
    try:
        for seed in (0, 1, 2, 6, 9, 11):
            task.reset(seed=seed)
            sim = task._env.sim
            sim.forward()
            for c in sim.data.contact[:sim.data.ncon]:
                names = [sim.model.body_id2name(sim.model.geom_bodyid[g]) or "" for g in (c.geom1, c.geom2)]
                chains = {n.split("_")[0] for n in names if n.startswith("chain")}
                assert len(chains) < 2, (seed, names)
    finally:
        task.close()


@pytest.mark.parametrize("task_id", GALILEO_TASKS)
@pytest.mark.parametrize("v_second, expected", [(-0.3, 1.0), (0.3, 0.0)])
def test_galileo_turning_point_inside_goal_counts_as_stopped(task_id, v_second, expected, monkeypatch):
    """The 20 Hz samples straddle the turning point; the apex estimated between them is scored."""
    task = _make(task_id)
    try:
        task.reset(seed=0)
        rot = task._goal_rot_world
        # Isolate the turning-point rule after a validated free roll. Real
        # right-side rollouts and rejected direct placements are tested below.
        task._entered_from_right = task._crossed_trough = True
        module = f"imbench.tasks.prehensile.galileo_ramp_drop_v{task_id[-1]}"
        monkeypatch.setattr(f"{module}.geoms_touch", lambda *args: True)
        _place_at_rest(task, "cube", task._goal_center_world + rot[:, 2] * task._ball_radius)
        task._untouched_steps = task._UNTOUCHED_STEPS
        raw = task._get_sim_observations(force_update=True)
        monkeypatch.setattr(task, "_ball_velocity", lambda: 0.3 * rot[:, 1])
        assert task._compute_task_score(raw) == 0.0
        monkeypatch.setattr(task, "_ball_velocity", lambda: v_second * rot[:, 1])
        assert task._compute_task_score(raw) == expected
    finally:
        task.close()
