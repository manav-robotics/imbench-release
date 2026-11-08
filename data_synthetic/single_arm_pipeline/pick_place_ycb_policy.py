"""Scripted pick-and-place policy for ``ib-p6-pick-place-ycb-v1``.

The task spawns a random YCB object and a 0.15 m green goal patch. The
policy picks the object and places it inside the patch. The patch
relocates 1-3 times mid-episode -- each time the object approaches it
-- so the transport phase has to chase the live goal continuously
rather than committing to a fixed waypoint.

Grasp strategy: a hand-tuned per-object catalogue of 2-3 grasp
templates. Each template is a yaw offset and grasp height in the
object's body frame; at pick time we sample one and rotate it into
the world by the object's measured yaw. This is far more reliable
than inferring a grasp axis from a runtime quaternion -- particularly
for ``large_marker``, whose spawn quat includes a y-axis 90 deg
rotation that confuses naive yaw extraction.

Phase machine:
  settle -> pre_grasp -> descend -> close_gripper -> lift
        -> transport (continuously retargets env._goal_pos)
        -> hover_settle (commits only once mg_remaining == 0)
        -> lower -> release -> retract -> done
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # (x, y, z, w) -- gripper Z down

PRE_GRASP_HEIGHT_ABOVE_TBL = 0.20   # hover here before / between phases
TRANSPORT_HEIGHT_ABOVE_TBL = 0.30   # transport altitude (>= mg_min_lift)
PLACE_HEIGHT_ABOVE_TBL     = 0.025  # final lower height; success window +/-0.05
HOVER_SETTLE_TICKS         = 25     # ticks held above goal before committing
# Lower in two rectilinear legs: (1) hold transport altitude, slide xy
# until centred above the goal; (2) drop straight down. Mixing xy and
# z together makes a diagonal sweep that snags the object on the patch
# edge when the goal moved late. The XY tolerance below gates the
# transition into the z-down leg.
PLACE_XY_ALIGN_TOL         = 0.012

# ── Phase budgets (ticks @ 100 Hz, 1000-step horizon) ───────────────
# Sized for the throttled global motion (lower gains + tighter action
# clip).
SETTLE_STEPS         = 15
PRE_GRASP_MAX_STEPS  = 180
# After pre_grasp's tolerance check fires, hold the hover pose for this
# many extra ticks before starting to descend. OSC tends to overshoot
# into the position when a setpoint is first reached, so giving it a
# short stabilisation window means the descent starts from a pose the
# controller has actually settled at -- not one it's still oscillating
# through.
PRE_GRASP_STABILISE_STEPS = 25
DESCEND_MAX_STEPS    = 150
CLOSE_STEPS          = 30
LIFT_MAX_STEPS       = 100
TRANSPORT_MAX_STEPS  = 290   # incl. up to 3 mid-flight goal teleports
LOWER_MAX_STEPS      = 110
RELEASE_STEPS        = 25
# After the jaws have opened, hold the gripper at the goal-centre pose
# for this many extra ticks before retracting. OSC tends to overshoot
# its lower setpoint, so without this pause the object often gets
# nudged off-centre as the arm starts retracting from a still-moving
# pose. The hold lets the controller settle and the object come to
# rest exactly under the gripper before we lift away.
RELEASE_SETTLE_STEPS = 50
RETRACT_MAX_STEPS    = 50

# ── Tolerances ──────────────────────────────────────────────────────
POS_TOL_XY = 0.012
POS_TOL_Z  = 0.010
HOVER_XY_TOL = 0.030

# ── Position-control gains ──────────────────────────────────────────
# Kept moderate so OSC has slack to catch up between ticks.
# Combined with MAX_ACTION_STEP this keeps EE deltas modest.
GAIN_XY  = 4.0
GAIN_Z   = 4.0
ROT_GAIN = 2.5

# Hard cap on per-axis action magnitude. The OSC interprets actions as
# desired Cartesian deltas in metres; clipping to 0.6 keeps each tick
# below a safe tracking threshold. Lower = slower + safer.
MAX_ACTION_STEP = 0.6

# ── Setpoint-change ramp ────────────────────────────────────────────
# Even with lower gains, OSC still overshoots when handed a freshly
# published waypoint. We reset a tick counter on every setpoint change
# and ramp the linear-axis gain back up to 1.0 over RAMP_TICKS ticks
# (smoothstep). RAMP_START_FRAC=0.2 means we start at 20% gain.
RAMP_TICKS       = 20
RAMP_START_FRAC  = 0.2
TARGET_CHANGE_TOL = 0.015   # m -- below this, treat as same setpoint


# ────────────────────────────────────────────────────────────────────
# Per-YCB grasp catalogue
# ────────────────────────────────────────────────────────────────────
# Each entry is a list of (grasp_yaw_offset_rad, grasp_z_above_table_m).
#
#   - ``grasp_yaw_offset`` is added to the object's measured world yaw
#     to form the gripper yaw. Use 0.0 to grasp across the object's
#     body-X axis, pi/2 to grasp across body-Y.
#   - ``grasp_z_above_table`` is the absolute z (above table top) the
#     gripper TCP descends to before closing. Set near the object's
#     mid-height; the task's table_height is added at runtime.
#
# The yaw component is the only part that varies for objects with
# rotational symmetry (cube, brick, marker). For spheres any yaw works
# so we still list two entries to keep the sampling uniform.
# ────────────────────────────────────────────────────────────────────

GRASP_CATALOG: dict[str, list[tuple[float, float]]] = {
    # All descent z values lowered by ~5 mm vs. the original tuning so
    # the fingertips bracket the object midline rather than skimming
    # the upper surface -- objects had been popping out on close when
    # the pinch was too high.

    # foam_brick: 0.077 x 0.051 x 0.052. 2-fold sym about Z; pinch
    # across the short body-Y axis. Yaw offsets walked through 90 deg
    # increments empirically: 0/pi closed along the long axis (slipped
    # out); +pi/2 closes across the short axis (secure pinch).
    "foam_brick":            [(np.pi / 2, 0.020),
                              (-np.pi / 2, 0.020)],

    # large_marker: cylinder, length 0.121 lying along the world XY
    # plane after the spawn-time y-90 rotation. yaw_offset = 0 aligns
    # the jaws-close axis perpendicular to the cylinder, so they
    # straddle the diameter. Empirically the marker is small enough
    # in diameter that the only reliable grip is fingertips almost
    # flush with the table -- 1 mm above. Higher z lets the cylinder
    # roll out from under the closing jaws.
    "large_marker":          [(0.0, 0.001),
                              (np.pi, 0.001)],

    # padlock: ~0.058 x 0.029 x 0.040; pinch across short axis.
    # Body z sits ~0.013 m above the table; descend to ~0.010 m so the
    # fingertips bracket the padlock body squarely.
    "padlock":               [(0.0, 0.010),
                              (np.pi, 0.010)],

    # racquetball: sphere 0.056 m. Yaw irrelevant -- always grasp at
    # 0 deg so the gripper approaches in a fixed, predictable pose.
    "racquetball":           [(0.0, 0.020)],

    # medium_clamp: 0.077 x 0.085 x 0.026. Asymmetric -- grip across
    # the narrow handle. body-X is along the longer dimension; yaw 0
    # closes across the thin direction. Descent z 2 mm above the
    # table -- the clamp body is thin enough that anything higher
    # leaves the jaws closing on empty air above the handle.
    "medium_clamp":          [(np.pi / 2, 0.002)],

    # plum: sphere 0.053. Yaw doesn't matter.
    "plum":                  [(0.0, 0.020),
                              (np.pi / 2, 0.020)],

    # golf_ball: sphere 0.043.
    "golf_ball":             [(0.0, 0.015),
                              (np.pi / 2, 0.015)],
}


# ────────────────────────────────────────────────────────────────────
# Quaternion helpers (xyzw convention, matching robosuite obs)
# ────────────────────────────────────────────────────────────────────

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


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _grasp_quat_for_yaw(world_yaw: float) -> np.ndarray:
    """Top-down grasp quat with the gripper rotated by ``world_yaw``
    about world Z (xyzw convention)."""
    half = 0.5 * world_yaw
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


def _object_world_yaw(env) -> float:
    """Extract the active object's yaw about world Z from its body
    rotation matrix.

    We project the object's body-X axis onto the world XY plane and
    take its angle. This is robust to spawn quats that include
    out-of-plane components (notably ``large_marker``, whose base quat
    rotates body-Z to point along world X before the yaw is applied).
    """
    sim = env.sim
    bid = sim.model.body_name2id(env._active_obj_name() + "_main")
    xmat = sim.data.body_xmat[bid].reshape(3, 3)
    # First column is the world coords of the object's body-X axis.
    bx_world = xmat[:, 0]
    # If body-X is nearly vertical, fall back to body-Y.
    if abs(bx_world[2]) > 0.9:
        bx_world = xmat[:, 1]
    return float(np.arctan2(bx_world[1], bx_world[0]))


# ────────────────────────────────────────────────────────────────────
# Policy
# ────────────────────────────────────────────────────────────────────

class PickPlaceYCBPolicy(PolicyInterface):
    """Scripted pick-and-place. Tracks a moving goal continuously."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._grasp_quat: np.ndarray | None = None
        self._grasp_z: float = 0.825          # absolute world z to descend to
        self._table_z: float = 0.8
        self._goal_xy: np.ndarray | None = None

        # Hover-above-goal book-keeping.
        self._hover_ticks_above_goal: int = 0
        self._tp_replans: int = 0
        self._last_goal_xy: np.ndarray | None = None

        # Cached object xy (locked at start of pre_grasp -- prevents
        # the descent setpoint from chasing object jitter).
        self._obj_xy: np.ndarray | None = None

        # Setpoint-ramp state: when a new target is published the gain
        # is held below 1.0 for the first few ticks so OSC can catch up
        # without overshoot.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6  # large -> no ramp at start

        self._last_action = np.zeros(7)
        self._last_action[6] = -1.0

    # ── Phase machinery ───────────────────────────────────────────────
    def _advance(self, next_phase: str):
        print(f"  [{self.step_in_phase:4d} ticks] {self.phase} -> {next_phase}")
        self.phase = next_phase
        self.step_in_phase = 0

    # ── Grasp catalogue lookup ───────────────────────────────────────
    def _select_grasp(self, env, obs):
        """Pick one entry from GRASP_CATALOG for the active YCB,
        compose the world-frame grasp quat with the live object yaw,
        and stash the descent z."""
        ycb = env._selected_ycb
        catalog = GRASP_CATALOG.get(ycb)
        if not catalog:
            # Fallback: top-down with no yaw offset, mid-height.
            catalog = [(0.0, 0.025)]

        idx = int(self.rng.integers(len(catalog)))
        yaw_offset, z_above_tbl = catalog[idx]

        obj_yaw = _object_world_yaw(env)
        gripper_yaw = obj_yaw + yaw_offset
        # Parallel-jaw grippers are symmetric under a 180-deg rotation
        # about their closing axis: yaw and yaw - pi produce the same
        # grasp. Wrap the commanded yaw to [-pi/2, pi/2) so we always
        # take the shorter wrist rotation -- avoids needless half-turns
        # and stays away from joint-limit corners.
        gripper_yaw = (gripper_yaw + np.pi / 2) % np.pi - np.pi / 2
        self._grasp_quat = _grasp_quat_for_yaw(gripper_yaw)
        self._grasp_z = self._table_z + z_above_tbl

        print(f"  [grasp] ycb={ycb}  catalog_idx={idx}/{len(catalog)}  "
              f"obj_yaw={np.degrees(obj_yaw):+.1f} deg  "
              f"yaw_offset={np.degrees(yaw_offset):+.1f} deg  "
              f"-> grip_yaw={np.degrees(gripper_yaw):+.1f} deg  "
              f"grasp_z={self._grasp_z:.3f}")

    # ── Setpoint ramp ─────────────────────────────────────────────────
    def _ramp_factor(self, target_pos: np.ndarray) -> float:
        """Smoothstep gain factor in [RAMP_START_FRAC, 1.0].

        Resets to RAMP_START_FRAC each time the linear setpoint shifts
        by more than TARGET_CHANGE_TOL, then ramps back up to 1.0 over
        RAMP_TICKS calls. Slows OSC's response to a freshly published
        waypoint so the EE can catch up without overshoot.
        """
        if (self._last_target_pos is None
                or float(np.linalg.norm(target_pos - self._last_target_pos))
                   > TARGET_CHANGE_TOL):
            self._ticks_since_target_change = 0
        self._last_target_pos = target_pos.copy()
        self._ticks_since_target_change += 1

        t = min(self._ticks_since_target_change / max(RAMP_TICKS, 1), 1.0)
        s = t * t * (3.0 - 2.0 * t)   # smoothstep
        return RAMP_START_FRAC + (1.0 - RAMP_START_FRAC) * s

    # ── Action helper ─────────────────────────────────────────────────
    def _move_to(self, eef_pos, eef_quat, target_pos, gripper,
                 gain_xy=GAIN_XY, gain_z=GAIN_Z, max_step=MAX_ACTION_STEP,
                 target_quat=None):
        ramp = self._ramp_factor(target_pos)
        action = np.zeros(7)
        delta = target_pos - eef_pos
        action[0] = float(np.clip(delta[0] * gain_xy * ramp, -max_step, max_step))
        action[1] = float(np.clip(delta[1] * gain_xy * ramp, -max_step, max_step))
        action[2] = float(np.clip(delta[2] * gain_z  * ramp, -max_step, max_step))
        tgt = TOP_DOWN_QUAT if target_quat is None else target_quat
        ori_err = _quat_err_axis_angle(eef_quat, tgt)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
        action[6] = gripper
        return action

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0]))
        obj_pos  = obs.get("object_pos", obs.get("cube_pos"))

        # Pull the (possibly moving) goal from the env each tick.
        if env is not None:
            live_goal = np.asarray(env._goal_pos, dtype=float).copy()
            self._table_z = float(getattr(env, "_table_height", 0.8))
            # Detect a goal teleport so we can reset the hover-stability counter.
            if (self._last_goal_xy is not None
                    and float(np.linalg.norm(live_goal - self._last_goal_xy))
                       > 0.05):
                self._tp_replans += 1
                self._hover_ticks_above_goal = 0
                print(f"  [goal moved] -> {np.round(live_goal, 3)}  "
                      f"(replan #{self._tp_replans})")
            self._last_goal_xy = live_goal
            self._goal_xy = live_goal

        # ── 1. Settle: let initial controller transients dissipate ────
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = -1.0
            if self.step_in_phase >= SETTLE_STEPS and env is not None:
                self._select_grasp(env, obs)
                # Lock the object xy now -- descent uses this rather
                # than the live obj_pos, which jitters during contact.
                self._obj_xy = obj_pos[:2].copy()
                self._advance("pre_grasp")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 2. Pre-grasp: hover above the object with grasp orientation ─
        if self.phase == "pre_grasp":
            target = np.array([
                self._obj_xy[0],
                self._obj_xy[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            xy_ok  = float(np.linalg.norm(target[:2] - eef_pos[:2])) < POS_TOL_XY
            z_ok   = abs(target[2] - eef_pos[2]) < POS_TOL_Z
            ori_ok = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, self._grasp_quat))) < 0.10
            if (xy_ok and z_ok and ori_ok) \
                    or self.step_in_phase >= PRE_GRASP_MAX_STEPS:
                self._advance("pre_grasp_settle")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 2b. Pre-grasp settle: hold the hover pose for a few ticks ─
        # OSC tends to overshoot the moment a new setpoint is reached;
        # without this hold the descent starts while the controller is
        # still oscillating around the hover, which biases the descent
        # path off-axis and lands the fingers next to the object
        # instead of around it.
        if self.phase == "pre_grasp_settle":
            target = np.array([
                self._obj_xy[0],
                self._obj_xy[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            if self.step_in_phase >= PRE_GRASP_STABILISE_STEPS:
                self._advance("descend")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 3. Descend straight down to the catalogued grasp z ────────
        # We don't gate on contact: the gripper finger collision geoms
        # are wider than the fingertips, so a passing brush against
        # the side of a tall/cylindrical object fires a false-positive
        # contact while the TCP is still well above the graspable
        # plane. Position + timeout is more reliable.
        if self.phase == "descend":
            target = np.array([self._obj_xy[0], self._obj_xy[1], self._grasp_z])
            # Higher gain_z than nominal so the descent still converges
            # within DESCEND_MAX_STEPS; max_step left at the global cap
            # so each tick stays small.
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_z=6.0, target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.005
            if reached or self.step_in_phase >= DESCEND_MAX_STEPS:
                self._advance("close_gripper")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 4. Close gripper, hold pose ───────────────────────────────
        if self.phase == "close_gripper":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase >= CLOSE_STEPS:
                self._advance("lift")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 5. Lift straight up to transport height ───────────────────
        if self.phase == "lift":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + TRANSPORT_HEIGHT_ABOVE_TBL])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.015
            if reached or self.step_in_phase >= LIFT_MAX_STEPS:
                self._advance("transport")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 6. Transport: continuously chase env._goal_pos at height ──
        # We DON'T gate on mg_remaining here -- we just always servo to
        # the live goal. When it teleports, the eef just keeps chasing.
        # Once we've been within HOVER_XY_TOL of the goal for
        # HOVER_SETTLE_TICKS *and* mg_remaining == 0, we drop into
        # ``lower``.
        if self.phase == "transport":
            target = np.array([
                float(self._goal_xy[0]),
                float(self._goal_xy[1]),
                self._table_z + TRANSPORT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)

            xy_err = float(np.linalg.norm(eef_pos[:2] - target[:2]))
            mg_remaining = int(getattr(env, "_mg_remaining", 0))

            if xy_err < HOVER_XY_TOL:
                self._hover_ticks_above_goal += 1
            else:
                self._hover_ticks_above_goal = 0

            ready = (self._hover_ticks_above_goal >= HOVER_SETTLE_TICKS
                     and mg_remaining == 0)
            if ready:
                self._advance("lower")
            elif self.step_in_phase >= TRANSPORT_MAX_STEPS:
                # Safety net.
                print(f"  [transport] timeout, mg_remaining={mg_remaining}, "
                      f"xy_err={xy_err:.3f}")
                self._advance("lower")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 7a. Place-align (xy-only): slide laterally at transport
        #         altitude until centred over the goal. Z is held at
        #         the transport height -- pure rectilinear xy motion. ─
        if self.phase == "lower":
            target = np.array([
                float(self._goal_xy[0]),
                float(self._goal_xy[1]),
                self._table_z + TRANSPORT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            if xy_err < PLACE_XY_ALIGN_TOL \
                    or self.step_in_phase >= LOWER_MAX_STEPS:
                self._advance("place_descend")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 7b. Place-descend (z-only): drop straight down to the
        #         place height. xy held at the goal centre. ──────────
        if self.phase == "place_descend":
            target = np.array([
                float(self._goal_xy[0]),
                float(self._goal_xy[1]),
                self._table_z + PLACE_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_z=6.0, target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.015
            if reached or self.step_in_phase >= LOWER_MAX_STEPS:
                self._advance("release")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 8. Release: open gripper, hold ────────────────────────────
        if self.phase == "release":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = -1.0
            if self.step_in_phase >= RELEASE_STEPS:
                self._advance("release_settle")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 8b. Release-settle: hold the goal-centre pose so OSC can
        #         finish converging and the object can rest cleanly
        #         before the arm starts moving away. ────────────────
        if self.phase == "release_settle":
            target = np.array([
                float(self._goal_xy[0]),
                float(self._goal_xy[1]),
                self._table_z + PLACE_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_z=6.0, target_quat=self._grasp_quat)
            if self.step_in_phase >= RELEASE_SETTLE_STEPS:
                self._advance("retract")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 9. Retract: lift off so the object can settle for at-rest
        #         success check ──────────────────────────────────────
        if self.phase == "retract":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + TRANSPORT_HEIGHT_ABOVE_TBL])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=TOP_DOWN_QUAT)
            reached = abs(target[2] - eef_pos[2]) < 0.02
            if reached or self.step_in_phase >= RETRACT_MAX_STEPS:
                self._advance("done")
            self.step_in_phase += 1
            self._last_action = action
            return action

        # ── 10. Done: hold open, let the object settle ───────────────
        action = np.zeros(7)
        action[6] = -1.0
        self.step_in_phase += 1
        self._last_action = action
        return action


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env    = ib.make("ib-p6-pick-place-ycb-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = PickPlaceYCBPolicy()

    for ep in range(5):
        obs, info = env.reset(seed=ep * 31 + 5)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===  "
              f"ycb={env._selected_ycb}  "
              f"goal={np.round(env._goal_pos, 3)}  "
              f"mg_total={env._mg_total}")
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
        print(f"Episode {ep}  score={info.get('task_score', 0.0):.3f}  "
              f"phase={policy.phase}  replans={policy._tp_replans}")
    env.close()
