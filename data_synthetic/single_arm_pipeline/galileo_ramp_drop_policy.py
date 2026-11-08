"""Scripted policy for ``ib-p2-galileo-ramp-drop-v1``.

Strategy
--------
The task scores the ball's FIRST goal-side turnaround: it must lie on
the goal patch (±5 mm along the arc). The channel is frictionless but
the segmented collision strip bleeds energy, so the ball released at
rest at the mirror-image angle −θ turns around short of θ (e.g. a
release at 50° turns around at ~45°).

``RELEASE_ANGLE_POLY`` maps the goal angle θ to the release angle on
the open arm. It was fitted by sweeping release angles with this
policy doing the release (same grasp, clearance and opening) and
measuring the first apex, so it absorbs the channel losses and the
release itself. On 100 seeds the first apex lands within 2.6 mm of
the patch centre (mean −0.15 mm, std 0.75 mm).

Two details make the release repeatable to well under a millimetre:
the jaws close across the channel (along the slope, the uphill finger
hits the ramp before the ball reaches it), and a fine-positioning
phase drives the measured ball centre to the release point with a PI
loop before the jaws open.

Motion model
------------
Rectilinear single-axis legs (lift first, descend last). Each leg
uses a P-controller with a smoothstep gain ramp on setpoint changes,
acts only on the leg's active axis, and locks the non-active axes to
their values at leg start.

Eef→ball offset is measured live: at the end of close_gripper we
record (eef_z - ball_center_z) and use it to compute the eef-z that
puts the ball center exactly where we want it for the release.

Phase machine
-------------
  settle
  -> ascend_pre_grasp     — pure Z up to pre-grasp altitude
  -> translate_to_ball    — pure XY to over the ball in the container
  -> descend_to_ball      — pure Z down to ball-grasp z
  -> close_gripper        — hold; jaws close; latch eef→ball offset
  -> ascend_lift          — pure Z up to hover altitude over release
  -> translate_to_drop    — pure XY to directly over the release point
  -> descend_to_release   — pure Z down to release height (ball
                            2 mm above the open-arm channel)
  -> fine_position        — PI on the measured ball centre until it
                            holds within FINE_POS_TOL_M of the target
  -> release_ball         — hold; jaws open; ball slides down the arm
  -> ascend_retreat       — pure Z up clear of the ramp
  -> done                 — hold pose; env scores the first apex
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw — gripper Z down

PRE_GRASP_HEIGHT_ABOVE_TBL = 0.20
BALL_GRASP_Z_OFFSET        = 0.005   # added to live ball-center z
LIFT_HEIGHT_ABOVE_TBL      = 0.40

# Drop strategy: release at rest on the open arm (ramp-local −Y) at
# the angle RELEASE_ANGLE_POLY gives for the goal angle.
#
# Ramp arc geometry (must match deps/extras/galileo_ramp.xml — the
# task uses the same constants in imbench/tasks/prehensile/
# galileo_ramp_drop_v1.py):
RAMP_ARC_RADIUS    = 0.24
RAMP_TROUGH_Z      = 0.015   # arc lowest z above ramp base top
RAMP_BASE_THICK    = 0.034   # base plate thickness below the origin
# So the trough-bottom z in WORLD coords is:
#   trough_world_z = table_z + RAMP_BASE_THICK + RAMP_TROUGH_Z

# Release angle (deg, open arm) as a cubic in the goal angle (deg).
# Fitted from a 0.5° sweep of release angles 24–68° with this policy
# releasing (first apex measured at 20 Hz with parabolic interpolation).
# Refit if the release clearance, grasp or ramp collision strip change.
RELEASE_ANGLE_POLY = np.array([7.2538e-05, -3.57208e-03, 1.137056, -0.396335])

# Ball centre: on the arc (radius R − r) at the release angle, raised
# this much so the jaws open without the ball touching the channel.
BALL_RELEASE_CLEARANCE_M   = 0.002

# Fine positioning before release: PI on the measured ball centre.
FINE_POS_TOL_M     = 0.0007   # every one of the last FINE_POS_HOLD ticks
FINE_POS_HOLD      = 6
FINE_POS_GAIN      = 6.0
FINE_POS_INT_GAIN  = 0.6
FINE_POS_MAX_STEP  = 0.15

# Hover altitude above the release point during the carry leg. The
# arm carries the ball at this z above the release-point arc surface,
# then descends straight down for the final-descend leg.
HOVER_ABOVE_RELEASE_M      = 0.10


# ── Tolerances ──────────────────────────────────────────────────────
POS_TOL_XY = 0.008
POS_TOL_Z  = 0.006
ORI_TOL    = 0.10


# ── Position-control gains ──────────────────────────────────────────
GAIN_XY  = 3.0
GAIN_Z   = 3.0
ROT_GAIN = 1.8
MAX_ACTION_STEP = 0.4

# Setpoint-change ramp: when a new leg starts we hold the gain low for
# the first RAMP_TICKS ticks (smoothstep from RAMP_START_FRAC to 1.0)
# so OSC doesn't overshoot the freshly-published waypoint.
RAMP_TICKS        = 12
RAMP_START_FRAC   = 0.25

# Per-phase tick budgets (caps; phases exit early on tolerance hit).
SETTLE_BUDGET             = 8
ASCEND_PRE_GRASP_BUDGET   = 80
TRANSLATE_TO_BALL_BUDGET  = 80
DESCEND_TO_BALL_BUDGET    = 80
CLOSE_BUDGET              = 20
ASCEND_LIFT_BUDGET        = 80
TRANSLATE_TO_DROP_BUDGET  = 100
DESCEND_TO_RELEASE_BUDGET = 80
FINE_POSITION_BUDGET      = 160
RELEASE_BUDGET            = 15
ASCEND_RETREAT_BUDGET     = 60


# ── Quaternion helpers (xyzw) ───────────────────────────────────────

def _quat_err_axis_angle(cur_xyzw: np.ndarray, tgt_xyzw: np.ndarray) -> np.ndarray:
    cx, cy, cz, cw = cur_xyzw
    tx, ty, tz, tw = tgt_xyzw
    ccx, ccy, ccz, ccw = -cx, -cy, -cz, cw
    ex = tw * ccx + tx * ccw + ty * ccz - tz * ccy
    ey = tw * ccy - tx * ccz + ty * ccw + tz * ccx
    ez = tw * ccz + tx * ccy - ty * ccx + tz * ccw
    ew = tw * ccw - tx * ccx - ty * ccy - tz * ccz
    if ew < 0.0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    sin_half = float(np.sqrt(ex * ex + ey * ey + ez * ez))
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, ew)
    axis = np.array([ex, ey, ez]) / sin_half
    return axis * angle


# ── Policy ──────────────────────────────────────────────────────────

class GalileoRampDropPolicy(PolicyInterface):
    """Scripted pick-from-container, drop-on-ramp-endpoint solver."""

    def __init__(self, rng=None, verbose: bool = False):
        self.rng = np.random.default_rng() if rng is None else rng
        self.verbose = verbose  # print phase transitions and plan details
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8

        # Latched at end of settle.
        self._ball_xy: np.ndarray | None     = None
        self._ball_z: float | None           = None
        self._ball_radius: float             = 0.024
        # Desired BALL-CENTRE world positions at the hover-above-
        # release-point leg and at the release itself. Set in
        # _plan_drop. eef-frame versions of these are derived in
        # _finalise_release_target once the grip offset is known.
        self._hover_xyz_ball_center: np.ndarray | None   = None
        self._release_xyz_ball_center: np.ndarray | None = None
        self._hover_eef_xyz: np.ndarray | None    = None
        self._release_eef_xyz: np.ndarray | None  = None
        # Grip offset latched at the moment we close on the ball:
        # eef_z_at_close - ball_center_z_at_close. Used to convert
        # desired ball-center z back into eef z.
        self._grip_offset_z: float | None    = None
        # Gripper orientation: top-down, re-yawed in _plan_drop so the
        # jaws close across the ramp channel.
        self._grip_quat: np.ndarray = TOP_DOWN_QUAT.copy()
        # Fine-positioning state: integral of the ball-centre error and
        # the recent error norms.
        self._fine_integral: np.ndarray = np.zeros(3)
        self._fine_errors: list[float] = []

        # Each motion phase locks its waypoint when it starts (first
        # call to _leg_target) so the setpoint never wanders mid-leg.
        self._leg_target: np.ndarray | None = None
        # Tick counter since the current leg's setpoint was published,
        # used to ramp the gain back up smoothly.
        self._ticks_in_leg: int = 0

    def _advance(self, next_phase: str):
        if self.verbose:
            print(f"  [{self.step_in_phase:4d} ticks] {self.phase} -> {next_phase}",
                  flush=True)
        self.phase = next_phase
        self.step_in_phase = 0
        self._leg_target = None
        self._ticks_in_leg = 0

    def _ramp(self) -> float:
        t = min(self._ticks_in_leg / max(RAMP_TICKS, 1), 1.0)
        s = t * t * (3.0 - 2.0 * t)
        return RAMP_START_FRAC + (1.0 - RAMP_START_FRAC) * s

    def _wrist_correction(self, eef_quat: np.ndarray) -> np.ndarray:
        ori_err = _quat_err_axis_angle(eef_quat, self._grip_quat)
        return np.clip(ori_err * ROT_GAIN, -0.3, 0.3)

    def _move_axis_z(self, eef_pos: np.ndarray, eef_quat: np.ndarray,
                     target_z: float, gripper: float) -> np.ndarray:
        """Pure-Z leg: hold xy at the leg's locked xy (= eef.xy at leg
        start), drive z toward target_z. Returns the 7-D action."""
        if self._leg_target is None:
            self._leg_target = np.array([eef_pos[0], eef_pos[1], float(target_z)])
            self._ticks_in_leg = 0
        ramp = self._ramp()
        action = np.zeros(7)
        # XY: tiny correction back to the locked xy so we don't drift,
        # but no commanded motion along these axes.
        dxy = self._leg_target[:2] - eef_pos[:2]
        action[0] = float(np.clip(dxy[0] * GAIN_XY * ramp,
                                  -MAX_ACTION_STEP, MAX_ACTION_STEP))
        action[1] = float(np.clip(dxy[1] * GAIN_XY * ramp,
                                  -MAX_ACTION_STEP, MAX_ACTION_STEP))
        # Z: drive toward target.
        dz = self._leg_target[2] - eef_pos[2]
        action[2] = float(np.clip(dz * GAIN_Z * ramp,
                                  -MAX_ACTION_STEP, MAX_ACTION_STEP))
        action[3:6] = self._wrist_correction(eef_quat)
        action[6] = gripper
        self._ticks_in_leg += 1
        return action

    def _move_axis_xy(self, eef_pos: np.ndarray, eef_quat: np.ndarray,
                      target_xy: np.ndarray, gripper: float) -> np.ndarray:
        """Pure-XY leg at the leg's locked altitude (= eef.z at leg
        start). Returns the 7-D action."""
        if self._leg_target is None:
            self._leg_target = np.array([
                float(target_xy[0]), float(target_xy[1]), float(eef_pos[2])
            ])
            self._ticks_in_leg = 0
        ramp = self._ramp()
        action = np.zeros(7)
        dxy = self._leg_target[:2] - eef_pos[:2]
        action[0] = float(np.clip(dxy[0] * GAIN_XY * ramp,
                                  -MAX_ACTION_STEP, MAX_ACTION_STEP))
        action[1] = float(np.clip(dxy[1] * GAIN_XY * ramp,
                                  -MAX_ACTION_STEP, MAX_ACTION_STEP))
        # Z: hold locked altitude.
        dz = self._leg_target[2] - eef_pos[2]
        action[2] = float(np.clip(dz * GAIN_Z * ramp,
                                  -MAX_ACTION_STEP, MAX_ACTION_STEP))
        action[3:6] = self._wrist_correction(eef_quat)
        action[6] = gripper
        self._ticks_in_leg += 1
        return action

    def _z_reached(self, eef_pos: np.ndarray) -> bool:
        if self._leg_target is None:
            return False
        return abs(self._leg_target[2] - eef_pos[2]) < POS_TOL_Z

    def _xy_reached(self, eef_pos: np.ndarray) -> bool:
        if self._leg_target is None:
            return False
        return float(np.linalg.norm(self._leg_target[:2] - eef_pos[:2])) < POS_TOL_XY

    def _hold_pose(self, eef_quat: np.ndarray, gripper: float) -> np.ndarray:
        action = np.zeros(7)
        action[3:6] = self._wrist_correction(eef_quat)
        action[6] = gripper
        return action

    def _plan_drop(self, env) -> bool:
        """Plan the release on the open arm (ramp-local −Y).

        Reads from env:
          ``_goal_center_world`` — patch centre in world coords
          ``_goal_theta``        — patch angle (rad) along the +Y arc
          ``_ramp_pos``          — ramp origin in world (xy of arc
                                    centre at the trough top)
          ``_ramp_yaw_actual``   — ramp yaw about world Z (rad)
          ``_ball_radius``       — ball radius (m)

        Computes:
          The release angle phi = RELEASE_ANGLE_POLY(theta). The ball
          centre target sits on the arc of radius R − r at −phi, raised
          by BALL_RELEASE_CLEARANCE_M. The gripper is yawed so the jaws
          close across the channel.

        Caches:
          self._release_xyz_ball_center — desired ball-centre world
                                          position at release
          self._hover_xyz_ball_center   — same xy, z = release_z +
                                          HOVER_ABOVE_RELEASE_M

        eef-z setpoints are finalised in close_gripper once the
        eef→ball offset is measured."""
        gc = getattr(env, "_goal_center_world", None)
        theta = getattr(env, "_goal_theta", None)
        ramp_pos = getattr(env, "_ramp_pos", None)
        ramp_yaw = getattr(env, "_ramp_yaw_actual", 0.0)
        if gc is None or theta is None or ramp_pos is None:
            return False
        self._ball_radius = float(getattr(env, "_ball_radius",
                                          self._ball_radius))

        theta = float(theta)
        ramp_pos = np.asarray(ramp_pos, dtype=float)
        phi = np.deg2rad(float(np.polyval(RELEASE_ANGLE_POLY, np.degrees(theta))))
        # Ball centre on the arc of radius (R − r) at angle −phi, in
        # ramp-local coords (arc centre at y = 0, z = trough_z + R).
        r_centre = RAMP_ARC_RADIUS - self._ball_radius
        local_y = -r_centre * np.sin(phi)
        local_z = RAMP_TROUGH_Z + RAMP_ARC_RADIUS - r_centre * np.cos(phi)
        cy_, sy_ = np.cos(ramp_yaw), np.sin(ramp_yaw)
        ball_center_release = np.array([
            ramp_pos[0] - sy_ * local_y,
            ramp_pos[1] + cy_ * local_y,
            ramp_pos[2] + local_z + BALL_RELEASE_CLEARANCE_M,   # ramp_pos.z is base-top
        ])
        ball_center_hover = ball_center_release.copy()
        ball_center_hover[2] += HOVER_ABOVE_RELEASE_M

        self._release_xyz_ball_center = ball_center_release
        self._hover_xyz_ball_center   = ball_center_hover

        # Top-down with the jaws across the channel: yaw the default
        # (jaws along world Y) by ramp_yaw + 90 deg about world Z.
        half = 0.5 * (ramp_yaw + np.pi / 2)
        self._grip_quat = np.array([np.cos(half), np.sin(half), 0.0, 0.0])

        # Eef targets get finalised in _finalise_release_target once
        # we know the grip offset.
        self._hover_eef_xyz   = None
        self._release_eef_xyz = None

        if self.verbose:
            print(f"  [plan] goal_xyz={np.round(np.asarray(gc, dtype=float), 4)}  "
                  f"theta={np.degrees(theta):+.1f}deg  "
                  f"ramp_pos={np.round(ramp_pos, 3)}",
                  flush=True)
            print(f"  [plan] release angle {np.degrees(phi):.2f}deg on the open arm: "
                  f"ball centre xyz={np.round(ball_center_release, 4)}",
                  flush=True)
        return True

    def _finalise_release_target(self, eef_pos: np.ndarray,
                                 ball_pos: np.ndarray) -> None:
        """Called once at the end of close_gripper. Latches the grip
        offset (eef_z - ball_center_z) and converts the planned
        ball-centre release/hover positions into eef-frame targets
        the leg controller can drive toward (xy passes through
        unchanged; z is offset by the measured grip offset)."""
        self._grip_offset_z = float(eef_pos[2] - ball_pos[2])
        self._hover_eef_xyz = np.array([
            self._hover_xyz_ball_center[0],
            self._hover_xyz_ball_center[1],
            self._hover_xyz_ball_center[2] + self._grip_offset_z,
        ])
        self._release_eef_xyz = np.array([
            self._release_xyz_ball_center[0],
            self._release_xyz_ball_center[1],
            self._release_xyz_ball_center[2] + self._grip_offset_z,
        ])
        if self.verbose:
            print(f"  [grip] grip_offset_z={self._grip_offset_z:+.4f}  "
                  f"ball_center_release_xyz="
                  f"{np.round(self._release_xyz_ball_center, 4)}  "
                  f"EEF_RELEASE_XYZ={np.round(self._release_eef_xyz, 4)}",
                  flush=True)
            print(f"  [grip] EEF_HOVER_XYZ={np.round(self._hover_eef_xyz, 4)}",
                  flush=True)

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0.0, 0.0, 0.0, 1.0])))
        ball_pos = np.asarray(obs.get("object_pos", obs.get("cube_pos")),
                              dtype=float)

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))

        # ── 1. Settle: latch ball xy/z and ramp near endpoint ────────
        if self.phase == "settle":
            self.step_in_phase += 1
            if self.step_in_phase >= SETTLE_BUDGET:
                self._ball_xy = ball_pos[:2].copy()
                self._ball_z  = float(ball_pos[2])
                ok = self._plan_drop(env) if env is not None else False
                if not ok:
                    raise RuntimeError(
                        "GalileoRampDropPolicy needs env._goal_center_world; "
                        "is the task ib-p2-galileo-ramp-drop-v1?"
                    )
                if self.verbose:
                    print(f"  [plan] ball_xy={np.round(self._ball_xy, 3)}  "
                          f"ball_z={self._ball_z:.3f}",
                          flush=True)
                self._advance("ascend_pre_grasp")
            return self._hold_pose(eef_quat, gripper=-1.0)

        # ── 2. Ascend to pre-grasp altitude (pure Z) ─────────────────
        if self.phase == "ascend_pre_grasp":
            target_z = self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL
            action = self._move_axis_z(eef_pos, eef_quat, target_z,
                                       gripper=-1.0)
            self.step_in_phase += 1
            if self._z_reached(eef_pos) \
                    or self.step_in_phase >= ASCEND_PRE_GRASP_BUDGET:
                self._advance("translate_to_ball")
            return action

        # ── 3. Translate xy to over the ball (pure XY) ───────────────
        if self.phase == "translate_to_ball":
            action = self._move_axis_xy(eef_pos, eef_quat, self._ball_xy,
                                        gripper=-1.0)
            self.step_in_phase += 1
            if self._xy_reached(eef_pos) \
                    or self.step_in_phase >= TRANSLATE_TO_BALL_BUDGET:
                self._advance("descend_to_ball")
            return action

        # ── 4. Descend onto the ball (pure Z) ────────────────────────
        if self.phase == "descend_to_ball":
            target_z = self._ball_z + BALL_GRASP_Z_OFFSET
            action = self._move_axis_z(eef_pos, eef_quat, target_z,
                                       gripper=-1.0)
            self.step_in_phase += 1
            if self._z_reached(eef_pos) \
                    or self.step_in_phase >= DESCEND_TO_BALL_BUDGET:
                self._advance("close_gripper")
            return action

        # ── 5. Close jaws on the ball ────────────────────────────────
        # On the LAST tick of this phase, latch the grip offset
        # (eef_z - ball_center_z) and use it together with the patch
        # position to finalise the eef-z setpoints for the hover and
        # release legs. We do this here (not in settle) because the
        # offset only becomes accurate once the jaws have actually
        # closed on the ball — before then ball_pos is the resting
        # position in the container, not the held position.
        if self.phase == "close_gripper":
            self.step_in_phase += 1
            if self.step_in_phase >= CLOSE_BUDGET:
                self._finalise_release_target(eef_pos, ball_pos)
                self._advance("ascend_lift")
            return self._hold_pose(eef_quat, gripper=1.0)

        # ── 6. Lift to carry altitude (pure Z) ───────────────────────
        # Carry altitude is the release z + HOVER_ABOVE_RELEASE_M + grip
        # offset, computed in _finalise_release_target. This is the z
        # of self._hover_eef_xyz.
        if self.phase == "ascend_lift":
            target_z = float(self._hover_eef_xyz[2])
            action = self._move_axis_z(eef_pos, eef_quat, target_z,
                                       gripper=1.0)
            self.step_in_phase += 1
            if self._z_reached(eef_pos) \
                    or self.step_in_phase >= ASCEND_LIFT_BUDGET:
                self._advance("translate_to_drop")
            return action

        # ── 7. Translate xy to directly above the release point ──────
        # The ball, held by a top-down gripper, ends up suspended over
        # the open arm at the planned release angle.
        if self.phase == "translate_to_drop":
            action = self._move_axis_xy(eef_pos, eef_quat,
                                        self._hover_eef_xyz[:2],
                                        gripper=1.0)
            self.step_in_phase += 1
            if self._xy_reached(eef_pos) \
                    or self.step_in_phase >= TRANSLATE_TO_DROP_BUDGET:
                self._advance("descend_to_release")
            return action

        # ── 8. Descend to the release height (pure Z) ────────────────
        # eef-z target = planned ball centre (BALL_RELEASE_CLEARANCE_M
        # above the channel) + grip offset.
        if self.phase == "descend_to_release":
            target_z = float(self._release_eef_xyz[2])
            action = self._move_axis_z(eef_pos, eef_quat, target_z,
                                       gripper=1.0)
            self.step_in_phase += 1
            if self._z_reached(eef_pos) \
                    or self.step_in_phase >= DESCEND_TO_RELEASE_BUDGET:
                self._advance("fine_position")
            return action

        # ── 9. Fine-position the ball centre on the release point ────
        # The legs stop within POS_TOL_Z (6 mm), which moves the first
        # apex by up to ~1.5 cm on low goals. A PI loop on the measured
        # ball centre removes the OSC steady-state error.
        if self.phase == "fine_position":
            err = self._release_xyz_ball_center - ball_pos
            self._fine_integral += err
            self._fine_errors.append(float(np.linalg.norm(err)))
            action = np.zeros(7)
            action[:3] = np.clip(err * FINE_POS_GAIN
                                 + self._fine_integral * FINE_POS_INT_GAIN,
                                 -FINE_POS_MAX_STEP, FINE_POS_MAX_STEP)
            action[3:6] = self._wrist_correction(eef_quat)
            action[6] = 1.0
            self.step_in_phase += 1
            settled = (len(self._fine_errors) >= FINE_POS_HOLD
                       and max(self._fine_errors[-FINE_POS_HOLD:]) < FINE_POS_TOL_M)
            if settled or self.step_in_phase >= FINE_POSITION_BUDGET:
                if self.verbose:
                    print(f"  [fine] ball-centre error {np.round(err * 1000, 2)} mm",
                          flush=True)
                self._advance("release_ball")
            return action

        # ── 10. Open jaws; ball slides down the open arm ─────────────
        if self.phase == "release_ball":
            self.step_in_phase += 1
            if self.step_in_phase >= RELEASE_BUDGET:
                self._advance("ascend_retreat")
            return self._hold_pose(eef_quat, gripper=-1.0)

        # ── 11. Ascend so the rolling ball is unobstructed (pure Z) ──
        if self.phase == "ascend_retreat":
            target_z = self._table_z + LIFT_HEIGHT_ABOVE_TBL
            action = self._move_axis_z(eef_pos, eef_quat, target_z,
                                       gripper=-1.0)
            self.step_in_phase += 1
            if self._z_reached(eef_pos) \
                    or self.step_in_phase >= ASCEND_RETREAT_BUDGET:
                self._advance("done")
            return action

        # ── 12. Done: hold pose while the ball swings to the goal ────
        return self._hold_pose(eef_quat, gripper=-1.0)


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env    = ib.make("ib-p2-galileo-ramp-drop-v1", privileged_obs=True, render_mode=render_mode)
    policy = GalileoRampDropPolicy()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 17)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===  "
              f"ball_pos={np.round(obs.get('object_pos', obs.get('cube_pos')), 3)}",
              flush=True)
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            done = terminated or truncated
        print(f"Episode {ep}  score={info.get('task_score', 0.0):.3f}  "
              f"phase={policy.phase}", flush=True)
    env.close()
