"""Galileo Ramp Drop — drop a ball onto one end of a frictionless symmetric
ramp so it rolls to the opposite side and comes to rest on a goal region.

Task ID: ``ib-p2-galileo-ramp-drop-v2``

Setup
-----
A Franka Panda sits in front of a table with three items on it:

1. A symmetric Galileo ramp (the ``galileo_ramp`` extras asset), oriented so
   its long axis runs along world X. The trough is near-frictionless so a
   ball released at one endpoint will oscillate back and forth.
2. A small open-top container on the table holding a ball. The container sits
   on the robot's side of the ramp.
3. A translucent green goal region on one arm of the ramp: a 1 cm strip
   along the arc (``goal_patch_half_y``) across the channel.

Objective
---------
The robot must pick up the ball from the container and release it on the
ramp arm opposite the goal, high enough that the ball rolls down through
the trough and up the goal-side arm to the goal. Success requires the
ball's first goal-side turnaround (where it momentarily stops) to lie on the
goal patch, with the robot not having touched the ball for the last
``_UNTOUCHED_STEPS`` control steps (a ball carried or held there does not
count).

The goal-side arm is covered; the right (ramp-local -Y) arm is the only
drop opening. A free right-side entry followed by a roll through the trough
is required, and touching the ball again cancels that attempt. Only the first
swing counts: the ball's first turn-around on the goal side must happen inside
the goal region. The ramp's oscillation decays, so without this rule a single
high release would reach any goal on a later, lower swing.

Reward: **binary** — 1.0 once successful, 0.0 otherwise. No dense shaping.

Pacing: runs at a slow control frequency (20 Hz) so the full release-roll-
settle sequence plays out visibly for a human watching the viewer.
"""

import logging

import numpy as np

from imbench.tasks.prehensile.contacts import geoms_touch, object_geom_ids, robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase

logger = logging.getLogger(__name__)


class GalileoRampDropV2(SingleArmBase):
    TASK_ID = "ib-p2-galileo-ramp-drop-v2"
    DESCRIPTION = (
        "Single arm: a ball sits in an open-top container on the table. A "
        "symmetric Galileo ramp lies across the table; a green goal region "
        "is marked on one of its arms. Pick the ball, release it on the ramp "
        "arm opposite the goal, and let it roll to the goal side, arriving "
        "there with near-zero velocity."
    )

    # Galileo ramp geometry (from deps/extras/galileo_ramp.xml).
    # Ramp local frame: +Y = ball travel (long axis), +X = short axis, +Z = up.
    # Inner arc parametrised by angle theta (theta=0 at trough bottom, +/-
    # ARC_HALF_ANGLE at the endpoints). Local (y, z) on the arc:
    #   y(theta) = R * sin(theta)
    #   z(theta) = R * (1 - cos(theta)) + TROUGH_Z
    _RAMP_SPAN_HALF = 0.232   # |inner arc endpoint y| (ball-travel direction)
    _RAMP_ENDPOINT_Z = 0.193  # inner arc endpoint height above base top
    _RAMP_TROUGH_Z = 0.015    # inner arc lowest z above base top
    _RAMP_BASE_THICKNESS = 0.034  # feet+plinth total vertical extent below origin
    _ARC_RADIUS = 0.24
    _ARC_HALF_ANGLE = np.deg2rad(75.0)
    # Control steps the ball must go untouched by the robot before its speed
    # in the goal region counts towards success.
    _UNTOUCHED_STEPS = 5

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._ball_radius = task_cfg.get("ball_radius", 0.012)
        # Goal region half-extents in the patch's own tangent frame:
        #   X = across the channel, Y = along the arc tangent.
        # The visual patch is drawn at exactly these half-extents so what you
        # see IS what the scorer checks against.
        self._goal_patch_half_x = task_cfg.get("goal_patch_half_x", 0.025)
        self._goal_patch_half_y = task_cfg.get("goal_patch_half_y", 0.005)
        self._goal_velocity_threshold = task_cfg.get(
            "goal_velocity_threshold", 0.05   # m/s
        )
        # Goal is sampled along the far-side arc: an interval of theta in
        # (0, ARC_HALF_ANGLE]. Values near 0 = near the trough bottom (easy,
        # ball has lots of energy); values near ARC_HALF_ANGLE = near the top
        # (hard, ball must climb almost to the endpoint).
        goal_theta_deg = task_cfg.get("goal_theta_range_deg", [25.0, 65.0])
        self._goal_theta_range = (
            np.deg2rad(goal_theta_deg[0]),
            np.deg2rad(goal_theta_deg[1]),
        )
        # Hardcoded — intentionally not read from config.
        self._ramp_offset_x = 0.0
        self._ramp_offset_y = 0.15
        self._ramp_yaw = np.pi / 2
        # Container is placed at the robot's side of the table in world X
        # (nothing to do with the ramp axis). Tuned so the arm can comfortably
        # reach it from its rest pose.
        self._container_offset_x = task_cfg.get("container_offset_x", -0.15)
        self._container_offset_y = task_cfg.get("container_offset_y", -0.05)

        # Container dimensions (metres).
        self._container_half_xy = task_cfg.get("container_half_xy", 0.030)
        self._container_wall = task_cfg.get("container_wall", 0.004)
        self._container_depth = task_cfg.get("container_depth", 0.035)

        self._task_done = False

    def _make_objects(self):
        return [
            # The rolling ball — small, dense, low friction (the ramp
            # itself already has zero friction so the rolling is near-lossless).
            {"type": "ball", "name": "cube",
             "size": [self._ball_radius],
             "rgba": [0.85, 0.15, 0.15, 1.0],
             "density": 7800,                     # steel
             "friction": [0.0, 0.0, 0.0],
             "joints": "default"},
        ]

    def _make_static_objects(self):
        # The container is a thin hollow box (5 walls, open top) built as a
        # CompositeBodyObject via the env's "composite" type. But that path
        # requires joint args — simpler to use four box walls and a floor, all
        # welded as separate static objects.
        cw = self._container_wall
        chxy = self._container_half_xy
        cd = self._container_depth
        container_color = [0.55, 0.40, 0.25, 1.0]
        return [
            # Galileo ramp (static fixture — no joint, welded to world).
            {"type": "extras", "name": "ramp",
             "extras_name": "galileo_ramp",
             "joints": None},
            # Goal region — thin translucent green patch that sits tangent
            # to the arc at its sampled location, small enough for the tilted
            # patch to fit within the channel trough. It is exactly the
            # region the scorer checks, in the patch's own frame.
            {"type": "box", "name": "goal_marker",
             "size": [self._goal_patch_half_x,
                      self._goal_patch_half_y, 0.0015],
             "rgba": [0.0, 0.85, 0.0, 0.9],
             "density": 1,
             "friction": [0.0, 0.0, 0.0],
             "joints": None},
            # Container: floor + 4 walls, all static. Frictionless to
            # match the rest of the scene — keeps the ball gliding
            # uniformly on whatever it touches.
            {"type": "box", "name": "container_floor",
             "size": [chxy, chxy, cw / 2],
             "rgba": container_color, "density": 1000,
             "friction": [0.0, 0.0, 0.0], "joints": None},
            {"type": "box", "name": "container_wall_px",
             "size": [cw / 2, chxy, cd / 2],
             "rgba": container_color, "density": 1000,
             "friction": [0.0, 0.0, 0.0], "joints": None},
            {"type": "box", "name": "container_wall_nx",
             "size": [cw / 2, chxy, cd / 2],
             "rgba": container_color, "density": 1000,
             "friction": [0.0, 0.0, 0.0], "joints": None},
            {"type": "box", "name": "container_wall_py",
             "size": [chxy, cw / 2, cd / 2],
             "rgba": container_color, "density": 1000,
             "friction": [0.0, 0.0, 0.0], "joints": None},
            {"type": "box", "name": "container_wall_ny",
             "size": [chxy, cw / 2, cd / 2],
             "rgba": container_color, "density": 1000,
             "friction": [0.0, 0.0, 0.0], "joints": None},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._task_done = False
        self._untouched_steps = 0
        self._entered_from_right = False
        self._crossed_trough = False
        self._invalid_entry = False
        self._prev_ramp_y = None
        self._first_swing_spent = False
        self._prev_v_ramp_y = 0.0
        # Ball state at the previous control step, kept while the ball is on
        # a valid route, for estimating where it turns around between steps.
        self._prev_route_ball = None
        self._last_apex_local = None

        sim = self._env.sim
        self._ball_geoms = object_geom_ids(self._env, "cube")
        self._channel_geoms = {
            g for g in range(sim.model.ngeom)
            if (sim.model.geom_id2name(g) or "").startswith("ramp_channel_col_")
        }
        self._ramp_geoms = object_geom_ids(self._env, "ramp")

        # ── Place the ramp on the table. ──
        # Ramp origin is at the top of its base plate; lift it by the base's
        # total thickness so the plinth bottom rests on the table.
        ramp_x = self._ramp_offset_x
        ramp_y = self._ramp_offset_y
        ramp_z = self._table_height + self._RAMP_BASE_THICKNESS
        # Yaw about world Z so ramp's +Y axis stays along world +Y when yaw=0.
        yaw = self._ramp_yaw
        ramp_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self._place_static("ramp", [ramp_x, ramp_y, ramp_z], quat=ramp_quat)
        self._ramp_pos = np.array([ramp_x, ramp_y, ramp_z])
        self._ramp_yaw_actual = yaw

        # Helper: convert ramp-local (x, y, z) offset to world frame.
        def ramp_to_world(dx, dy, dz):
            cy_, sy_ = np.cos(yaw), np.sin(yaw)
            wx = ramp_x + cy_ * dx - sy_ * dy
            wy = ramp_y + sy_ * dx + cy_ * dy
            wz = ramp_z + dz
            return np.array([wx, wy, wz])

        # Release-side endpoint: ramp_local_y = -RAMP_SPAN_HALF
        # Goal-side endpoint:    ramp_local_y = +RAMP_SPAN_HALF
        near_endpoint = ramp_to_world(0.0, -self._RAMP_SPAN_HALF,
                                      self._RAMP_ENDPOINT_Z)
        far_endpoint = ramp_to_world(0.0, +self._RAMP_SPAN_HALF,
                                     self._RAMP_ENDPOINT_Z)
        self._near_endpoint = near_endpoint
        self._far_endpoint = far_endpoint

        # ── Goal region ON the ramp's far arc (tangent-aligned, randomised). ──
        # Sample a theta on the far (+Y local) side of the arc, compute the
        # point on the inner arc, offset outward by a tiny amount (so the
        # patch sits just above the surface), and orient the patch's +Z
        # normal along the arc's outward radial direction at that angle.
        theta = self._rng.uniform(*self._goal_theta_range)
        self._goal_theta = theta
        R = self._ARC_RADIUS

        # Point on inner arc, in ramp-local frame.
        local_y = R * np.sin(theta)
        local_z = R * (1.0 - np.cos(theta)) + self._RAMP_TROUGH_Z
        # Outward radial direction (up out of trough) in ramp-local frame.
        outward_local = np.array([0.0, -np.sin(theta), np.cos(theta)])
        # Offset the patch centre out along the outward normal so it sits
        # just above the arc surface (half its own thickness + small epsilon).
        PATCH_HALF_THICK = 0.001
        offset = PATCH_HALF_THICK + 0.0005
        local_pos = np.array([0.0, local_y, local_z]) + offset * outward_local
        goal_world = ramp_to_world(local_pos[0], local_pos[1], local_pos[2])

        # Orientation: start from ramp's yaw, then rotate about the ramp-local
        # X axis by angle -theta so the patch's +Z normal matches outward_local.
        # Quaternion multiplication (w, x, y, z) ordering.
        def _quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ])
        # Rotation about the ramp-local X axis by +theta sends world +Z to
        # the ramp-local outward-radial direction (0, -sin theta, cos theta),
        # so the patch's flat face rests tangent to the arc.
        tilt_quat = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])
        goal_quat = _quat_mul(ramp_quat, tilt_quat)

        self._goal_pos = np.array([goal_world[0], goal_world[1]])
        self._goal_center_world = np.array(goal_world, dtype=float)
        self._goal_z = float(goal_world[2])
        self._place_static("goal_marker", goal_world.tolist(), quat=goal_quat)
        # Cache the patch's world->local rotation matrix for scoring. The
        # scorer transforms the ball into the patch frame and checks it falls
        # inside the visible rectangle.
        sim = self._env.sim
        sim.forward()
        bid = sim.model.body_name2id("goal_marker_main")
        self._goal_rot_world = np.array(sim.data.xmat[bid]).reshape(3, 3)

        # Goal marker is visual-only: zero its contype/conaffinity so the
        # ball passes through the patch volume instead of bouncing off
        # the 3 mm-thick box. Scoring is fully geometric (the scorer
        # transforms the ball into the patch frame and checks
        # rectangle inclusion + z tolerance) so contacts are not needed.
        for g_idx in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g_idx] == bid:
                sim.model.geom_contype[g_idx] = 0
                sim.model.geom_conaffinity[g_idx] = 0

        # ── Container on the robot's side of the table. ──
        # Placed in world frame (not relative to the ramp) so it stays close
        # to the robot regardless of ramp yaw.
        table_z = self._table_height
        cx = float(self._container_offset_x)
        cy = float(self._container_offset_y)
        cw = self._container_wall
        cd = self._container_depth
        chxy = self._container_half_xy
        self._place_static("container_floor",
                           [cx, cy, table_z + cw / 2])
        self._place_static("container_wall_px",
                           [cx + chxy - cw / 2, cy,
                            table_z + cw + cd / 2])
        self._place_static("container_wall_nx",
                           [cx - chxy + cw / 2, cy,
                            table_z + cw + cd / 2])
        self._place_static("container_wall_py",
                           [cx, cy + chxy - cw / 2,
                            table_z + cw + cd / 2])
        self._place_static("container_wall_ny",
                           [cx, cy - chxy + cw / 2,
                            table_z + cw + cd / 2])
        self._container_center = np.array([cx, cy])

        # ── Ball sits inside the container, resting on its floor. ──
        ball_z = table_z + cw + self._ball_radius + 0.001
        self._place_object("cube", [cx, cy, ball_z])

        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ── Scoring ───────────────────────────────────────────────────────────

    def _ball_velocity(self):
        sim = self._env.sim
        body_id = sim.model.body_name2id("cube_main")
        jnt_addr = sim.model.body_jntadr[body_id]
        if jnt_addr < 0:
            return np.zeros(3)
        dof_addr = sim.model.jnt_dofadr[jnt_addr]
        return np.array(sim.data.qvel[dof_addr:dof_addr + 3])

    def _compute_task_score(self, raw_obs):
        if self._task_done:
            return 1.0
        if "cube_pos" not in raw_obs:
            return 0.0
        ball_world = np.array(raw_obs["cube_pos"], dtype=float)
        vel = self._ball_velocity()
        sim = self._env.sim
        touching_robot = robot_contacts(sim, self._ball_geoms)
        on_channel = geoms_touch(sim, self._ball_geoms, self._channel_geoms)
        touching_ramp = geoms_touch(sim, self._ball_geoms, self._ramp_geoms)
        c, s = np.cos(self._ramp_yaw_actual), np.sin(self._ramp_yaw_actual)
        ramp_local = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) @ (
            ball_world - self._ramp_pos)
        y = float(ramp_local[1])
        # Along-ramp velocity (ramp-local +Y points from the open arm to the goal arm).
        v_ramp_y = float(vel[1] * c - vel[0] * s)
        if touching_robot:
            self._untouched_steps = 0
            self._entered_from_right = False
            self._crossed_trough = False
            self._invalid_entry = False
            self._prev_ramp_y = None
            self._first_swing_spent = False
        else:
            self._untouched_steps += 1
            # The first free contact with the ramp must be on its open arm.
            # A goal-side drop cannot earn credit on a later oscillation.
            if touching_ramp and not self._entered_from_right:
                if on_channel and y < -2 * self._ball_radius and not self._invalid_entry:
                    self._entered_from_right = True
                elif y >= -2 * self._ball_radius:
                    self._invalid_entry = True
            if self._entered_from_right:
                # Leaving the channel or flying over the trough invalidates
                # the route; small normal excursions allow contact bounce.
                radial = np.hypot(y, self._ARC_RADIUS + self._RAMP_TROUGH_Z - ramp_local[2])
                near_channel = (
                    abs(ramp_local[0]) < 0.030
                    and abs(radial - (self._ARC_RADIUS - self._ball_radius)) < 0.012
                )
                if not near_channel:
                    self._entered_from_right = False
                    self._crossed_trough = False
                    self._invalid_entry = True
                    self._prev_ramp_y = None
                else:
                    if (on_channel and self._prev_ramp_y is not None
                            and self._prev_ramp_y < 0 <= y):
                        self._crossed_trough = True
                    self._prev_ramp_y = y
        # Success is decided at the ball's first goal-side turnaround: the
        # apex must lie on the visible patch with near-zero velocity. At
        # 20 Hz the samples either side of the apex can sit ~1 cm below it,
        # so the apex is estimated from the two steps around the reversal,
        # assuming constant acceleration between them (error < 1 mm).
        on_route = (self._crossed_trough and not self._first_swing_spent and on_channel
                    and self._untouched_steps >= self._UNTOUCHED_STEPS)
        prev = self._prev_route_ball
        self._prev_route_ball = (ball_world, vel) if on_route else None
        if (self._crossed_trough and not self._first_swing_spent
                and y > 0 and self._prev_v_ramp_y > 0 >= v_ramp_y):
            self._first_swing_spent = True
            self._last_apex_local = None
            if on_route and prev is not None and self._apex_in_goal(prev, vel, c, s):
                self._task_done = True
                logger.info("[galileo-ramp-drop] SUCCESS: first apex at patch-local %s",
                            np.round(self._last_apex_local, 4))
                return 1.0
            if self._last_apex_local is not None:
                logger.info("[galileo-ramp-drop] first apex missed the goal: patch-local %s",
                            np.round(self._last_apex_local, 4))
        self._prev_v_ramp_y = v_ramp_y
        return 0.0

    def _apex_in_goal(self, prev, vel, c, s):
        """Estimate the apex between the previous and current step; True if it is on the patch."""
        prev_pos, prev_vel = prev
        along = np.array([-s, c, 0.0])           # ramp-local +Y in world
        accel = (vel - prev_vel) * self.control_freq
        decel = -float(accel @ along)
        if decel <= 0:
            self._last_apex_local = None
            return False
        t = float(prev_vel @ along) / decel
        apex = prev_pos + prev_vel * t + 0.5 * accel * t * t
        apex_vel = prev_vel + accel * t
        local = self._goal_rot_world.T @ (apex - self._goal_center_world)
        self._last_apex_local = local
        return (abs(local[0]) < self._goal_patch_half_x
                and abs(local[1]) < self._goal_patch_half_y
                and abs(local[2] - (self._ball_radius - 0.0015)) < 0.008
                and float(np.linalg.norm(apex_vel)) < self._goal_velocity_threshold)

    def _check_termination(self, raw_obs):
        if self._task_done:
            return True
        return self._object_fell_off_table(raw_obs)