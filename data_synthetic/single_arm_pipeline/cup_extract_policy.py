"""Scripted policy for ``ib-p4-cup-extract-v1``.

Strategy
--------
A red cuboid is trapped inside a handled, open-top cup standing
upright on the table. The cup is reset with yaw=pi so the flat,
horizontal handle (a thin red box at cup-local (0.064, 0, 0.05) with
half-size (0.025, 0.010, 0.020)) points toward the robot.

The cup body is too wide to pinch with a parallel-jaw gripper, but
the handle is only 2 cm wide (in cup-local y) and 5 cm long (in
cup-local x) -- a perfect top-down pinch target. Once the handle is
clamped, rolling the wrist 180 deg about its OWN local x axis
(= the handle's long axis) inverts the cup without unseating the
grip: the jaws stay aligned with the handle the entire time.

Extraction sequence:
  1. Top-down grasp on the handle.
  2. Lift cup straight up clear of the table.
  3. Translate at altitude so the CUP BODY (not the handle, which is
     offset by ~6 cm) sits over a TEMP DROP location 0.2 m away from
     the goal -- bare table, away from the raised goal marker, so the
     cube lands flat without bouncing off the goal pad's edge.
  4. Slowly slew the wrist 180 deg about gripper-local x. The cup
     inverts about a horizontal axis through the handle; the cuboid
     starts sliding past the wall lip around 60-90 deg of tilt and
     drops onto the table once the opening points down.
  5. Hold the inverted pose for the cuboid to settle on the table.
  6. Unflip: rotate the wrist -180 deg about gripper-local x to bring
     the cup back upright (still gripping the handle).
  7. Carry the upright cup to a PARK xy that is 0.2 m from the goal
     in a direction orthogonal to the temp-drop direction (so it
     clears both the goal and the dropped cube), lower it until the
     cup base is just above the table, and open the jaws to release.
  8. Lift the arm clear, traverse to the dropped cube's xy with a
     fresh top-down pose, descend, and close on the cube.
  9. Lift the cube, traverse to the goal xy, descend, and open the
     jaws so the cube settles on the green goal pad.

Phase machine:
  settle
  -> align_above_handle  (xy over handle, z = table + PRE_GRASP, top-down)
  -> descend_to_handle   (drop to handle z; jaws straddle the handle)
  -> close_gripper       (clamp; hold pose for jaws to converge)
  -> lift_cup            (rise straight up to LIFT altitude)
  -> translate_to_goal   (xy moves so cup body, not handle, is over temp drop)
  -> flip_cup            (smoothly slew wrist +180 deg about local x)
  -> hold_for_drop       (wait while the cuboid falls and settles)
  -> unflip_cup          (slew wrist -180 deg back to upright)
  -> cup_lift            (small rise so cup clears the dropped cube)
  -> cup_translate       (carry cup to park xy at altitude)
  -> cup_descend         (lower cup base toward the table)
  -> cup_release         (open jaws, cup settles on table)
  -> arm_lift            (rise off the released cup)
  -> approach_cube       (xy hover over the dropped cube)
  -> descend_cube        (descend to cube grasp z)
  -> close_cube          (close jaws on the cube)
  -> lift_cube           (rise to LIFT altitude)
  -> translate_cube      (carry cube to goal xy)
  -> descend_goal        (lower cube onto the goal pad)
  -> release_cube        (open jaws, cube settles on goal)
  -> done

Grasp orientation:
  TOP_DOWN_QUAT places gripper-local x along world +x. The handle
  long axis is along world -x (cup yaw = pi), but jaws are symmetric
  so we still grasp at grip_yaw = 0 -- the handle's narrow dimension
  (cup-local y, world +/-y after yaw=pi) ends up between the jaws.
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw -- gripper Z down

# Handle pose (cup-local frame, from deps/extras/open_cup.xml).
# The handle box spans cup-local x in [0.039, 0.089], center at 0.064.
# Grasping near the free tip (x=0.080) gives the Panda hand more
# clearance from the cup wall, but the cup slips during the flip --
# the moment arm from the grasp point to the cup CoM is too long, so
# the cup torques out of the jaws. Pull in toward the handle's
# geometric center so the lever arm is shorter and the grasp holds
# through the 180 deg roll.
HANDLE_LOCAL_POS  = np.array([0.060, 0.0, 0.05])
HANDLE_LOCAL_HALF = np.array([0.025, 0.010, 0.020])

# Vertical setpoints (above table top).
PRE_GRASP_HEIGHT_ABOVE_TBL = 0.18   # hover here before descending onto handle
# Handle z extent is [0.03, 0.07]. Aim slightly below center so the
# fingertips definitely overlap the handle body when they close.
HANDLE_GRASP_Z_ABOVE_TBL   = 0.045
LIFT_HEIGHT_ABOVE_TBL      = 0.4   # handle z while flipping; cup never hits table
# During the 180 deg roll the cup body sweeps through an arc of radius
# ~0.10 m (cup height) below the handle. At 0.25 m the gripper fingers
# scrape the table mid-flip; bumping to 0.32 m keeps the arc clear.

# Two-stage drop: flip the cup at a temp location away from the goal so
# the cube lands cleanly on bare table first (avoids bouncing off the
# raised goal marker), then a later phase moves the cube onto the real
# goal. Distance is from the goal center, in xy.
TEMP_DROP_DISTANCE_XY      = 0.20
# Table half-extent in xy (from single_arm_base default 0.5x0.5 table).
# Used to clamp the temp drop so the cup never overhangs the table edge.
TABLE_HALF_XY              = 0.22

# Cup park: after the cuboid is released onto the table, the arm
# unflips the cup, carries it to a CUP_PARK_DISTANCE_XY offset from the
# goal in a direction orthogonal to the temp-drop direction (so it
# clears both the goal and the dropped cube), then lowers the cup base
# onto the table and releases it.
CUP_PARK_DISTANCE_XY       = 0.20
# Cup base hovers ~LIFT above the table while carried; to set it down
# the handle (which is offset ~5 cm above the cup base) needs to be
# only ~5-6 cm above the table. Use 0.06 m for clearance.
CUP_RELEASE_HANDLE_Z_ABOVE_TBL = 0.060

# Cube pick-and-place: cube half-z = 0.03 (see cup_extract_v1.py).
# Top-down pinch on the cube center -> grasp z = table + 0.03.
# When placing, the cube center should be at table + 0.03, so the
# gripper finger tip (which is essentially at eef z when closed on a
# small object) targets table + 0.03 too.
CUBE_GRASP_Z_ABOVE_TBL     = 0.030
CUBE_PLACE_Z_ABOVE_TBL     = 0.030

# Wrist flip: full 180 deg about gripper-local x. The cuboid actually
# escapes between ~60 and 120 deg, but we go all the way so the cup
# opening points down for the settle phase.
FLIP_TARGET_ANGLE = np.pi


# ── Phase budgets (fractions of episode horizon, sampled at reset) ──
# Effective Cartesian eef speed under this OSC is ~10 cm/s, so an
# initial ~20 cm xy traverse + descent needs ~3-4 s. Budgets sized like
# MirrorPickPlace's (which only does grasp+lift) leave align/descend
# starved, with the eef stalling 5-7 cm above the handle at xy_err
# >4 cm, so these are larger.
SETTLE_FRAC          = 0.03
ALIGN_FRAC           = 0.09   # far-corner cup positions need ~22 cm xy traverse
DESCEND_FRAC         = 0.07   # ~14 cm z + xy fine-tune; needs slack for tolerance
CLOSE_FRAC           = 0.03
LIFT_FRAC            = 0.04
TRANSLATE_FRAC       = 0.05
FLIP_FRAC            = 0.08   # slow slew so the cup tilts gently
HOLD_DROP_FRAC       = 0.05   # wait for cuboid to fall + settle
# Unflip + cup park + cube pick + cube place chain.
UNFLIP_FRAC          = 0.08   # mirror of FLIP_FRAC
CUP_LIFT_FRAC        = 0.03   # short rise so cup clears the dropped cube before traverse
CUP_TRANSLATE_FRAC   = 0.06   # carry cup to park xy
CUP_DESCEND_FRAC     = 0.05   # lower cup base to table
CUP_RELEASE_FRAC     = 0.03   # open jaws, let cup settle
ARM_LIFT_FRAC        = 0.03   # rise off the released cup before traversing
APPROACH_CUBE_FRAC   = 0.06   # xy traverse to hover over the cube
DESCEND_CUBE_FRAC    = 0.05
CLOSE_CUBE_FRAC      = 0.03
LIFT_CUBE_FRAC       = 0.03
TRANSLATE_CUBE_FRAC  = 0.05   # carry cube to goal xy
DESCEND_GOAL_FRAC    = 0.04
RELEASE_CUBE_FRAC    = 0.02
# ~3% remainder is slack absorbed by the trailing "done" hold.

# Episode-length distribution (Cup Extract Single-Arm teleop): clip(N(mean, std), min, max).
# Each reset draws a horizon and slices it into per-phase budgets via the
# fractions above. Stages stretch or compress with the sampled horizon so the
# realised motion-time distribution mirrors the teleop one.
HORIZON_MEAN = 5069
HORIZON_STD  = 1113
HORIZON_MIN  = 2541
HORIZON_MAX  = 9797


# ── Tolerances ──────────────────────────────────────────────────────
POS_TOL_XY = 0.010
POS_TOL_Z  = 0.008
ORI_TOL    = 0.10


# ── Position-control gains ──────────────────────────────────────────
GAIN_XY  = 2.5
GAIN_Z   = 2.5
ROT_GAIN = 1.8
MAX_ACTION_STEP = 0.35

RAMP_TICKS        = 20
RAMP_START_FRAC   = 0.2
TARGET_CHANGE_TOL = 0.015


# ── Quaternion helpers (xyzw) ───────────────────────────────────────

def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    n = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm < 1e-9 or abs(angle) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    n = n / norm
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([n[0] * s, n[1] * s, n[2] * s, np.cos(half)])


def _grasp_quat_for_yaw(world_yaw: float) -> np.ndarray:
    half = 0.5 * world_yaw
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


def _quat_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z),   2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),       1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),       2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _yaw_from_quat(q_xyzw: np.ndarray) -> float:
    """Project the body-X axis onto world XY and take its angle."""
    x, y, z, w = q_xyzw
    bx_x = 1.0 - 2.0 * (y * y + z * z)
    bx_y = 2.0 * (x * y + z * w)
    return float(np.arctan2(bx_y, bx_x))


def _shortest_grasp_yaw(handle_yaw: float) -> float:
    """Wrap to [-pi/4, pi/4]. The parallel-jaw grip has 2-fold
    symmetry; rotating by pi/2 swaps which face each jaw touches but
    gives an equivalent grasp on a rectangular handle (the handle
    long/short axes match the gripper local x/y at yaw=0)."""
    y = (handle_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(y)


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

class CupExtractPolicy(PolicyInterface):
    """Scripted grasp-handle / lift / flip / drop solver."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8
        self._lift_z: float  = self._table_z + LIFT_HEIGHT_ABOVE_TBL

        # Sample horizon and slice into phase budgets.
        horizon = float(self.rng.normal(HORIZON_MEAN, HORIZON_STD))
        self.horizon = int(np.clip(round(horizon), HORIZON_MIN, HORIZON_MAX))
        self.budgets = {
            "settle":         max(1, int(round(SETTLE_FRAC         * self.horizon))),
            "align":          max(1, int(round(ALIGN_FRAC          * self.horizon))),
            "descend":        max(1, int(round(DESCEND_FRAC        * self.horizon))),
            "close":          max(1, int(round(CLOSE_FRAC          * self.horizon))),
            "lift":           max(1, int(round(LIFT_FRAC           * self.horizon))),
            "translate":      max(1, int(round(TRANSLATE_FRAC      * self.horizon))),
            "flip":           max(1, int(round(FLIP_FRAC           * self.horizon))),
            "hold_drop":      max(1, int(round(HOLD_DROP_FRAC      * self.horizon))),
            "unflip":         max(1, int(round(UNFLIP_FRAC         * self.horizon))),
            "cup_lift":       max(1, int(round(CUP_LIFT_FRAC       * self.horizon))),
            "cup_translate":  max(1, int(round(CUP_TRANSLATE_FRAC  * self.horizon))),
            "cup_descend":    max(1, int(round(CUP_DESCEND_FRAC    * self.horizon))),
            "cup_release":    max(1, int(round(CUP_RELEASE_FRAC    * self.horizon))),
            "arm_lift":       max(1, int(round(ARM_LIFT_FRAC       * self.horizon))),
            "approach_cube":  max(1, int(round(APPROACH_CUBE_FRAC  * self.horizon))),
            "descend_cube":   max(1, int(round(DESCEND_CUBE_FRAC   * self.horizon))),
            "close_cube":     max(1, int(round(CLOSE_CUBE_FRAC     * self.horizon))),
            "lift_cube":      max(1, int(round(LIFT_CUBE_FRAC      * self.horizon))),
            "translate_cube": max(1, int(round(TRANSLATE_CUBE_FRAC * self.horizon))),
            "descend_goal":   max(1, int(round(DESCEND_GOAL_FRAC   * self.horizon))),
            "release_cube":   max(1, int(round(RELEASE_CUBE_FRAC   * self.horizon))),
        }
        print(f"  [horizon] sampled={self.horizon}  budgets={self.budgets}",
              flush=True)

        # Latched at the end of settle: handle pose, grasp orientation,
        # cup body offset (handle-frame). Refreshed at the end of lift
        # so the "translate to goal" target uses the in-air offset, not
        # the on-table offset (which can be perturbed by close/lift).
        self._handle_xy: np.ndarray | None  = None
        self._grasp_quat: np.ndarray        = TOP_DOWN_QUAT.copy()
        # Snapshot of grasp_quat right before the flip; reused for the
        # unflip target (no rotation about local x) so we don't lose the
        # original yaw if the cup yawed during close/lift.
        self._pre_flip_quat: np.ndarray     = TOP_DOWN_QUAT.copy()
        # Direction (unit xy) used for the temp drop, so the cup-park
        # location can be picked orthogonal to it (and clear of both the
        # goal and the dropped cube).
        self._temp_drop_dir: np.ndarray     = np.array([1.0, 0.0])
        self._cup_to_handle_xy: np.ndarray  = np.array([0.064, 0.0])  # default for yaw=pi
        self._goal_xy: np.ndarray | None    = None
        self._temp_drop_xy: np.ndarray | None        = None
        self._translate_target_xy: np.ndarray | None = None
        self._cup_park_xy: np.ndarray | None         = None
        self._cup_park_eef_xy: np.ndarray | None     = None
        self._cube_pickup_xy: np.ndarray | None      = None

        # Setpoint-ramp state.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

    def _advance(self, next_phase: str):
        print(f"  [{self.step_in_phase:4d} ticks] {self.phase} -> {next_phase}",
              flush=True)
        self.phase = next_phase
        self.step_in_phase = 0

    def _ramp_factor(self, target_pos: np.ndarray) -> float:
        if (self._last_target_pos is None
                or float(np.linalg.norm(target_pos - self._last_target_pos))
                   > TARGET_CHANGE_TOL):
            self._ticks_since_target_change = 0
        self._last_target_pos = target_pos.copy()
        self._ticks_since_target_change += 1

        t = min(self._ticks_since_target_change / max(RAMP_TICKS, 1), 1.0)
        s = t * t * (3.0 - 2.0 * t)
        return RAMP_START_FRAC + (1.0 - RAMP_START_FRAC) * s

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

    # ── Helpers that read live cup pose ──────────────────────────────
    def _handle_world_pos(self, cup_pos: np.ndarray, cup_quat: np.ndarray) -> np.ndarray:
        return cup_pos + _quat_to_rotmat(cup_quat) @ HANDLE_LOCAL_POS

    def _read_goal_xy(self, env) -> np.ndarray:
        # Task stores goal_xy on the env instance (see CupExtractV1.reset).
        gp = getattr(env, "_goal_pos", None)
        if gp is not None:
            return np.asarray(gp, dtype=float)[:2]
        # Last resort: try to read static body position from the sim.
        try:
            sim = env._env.sim
            for name in ("goal_marker_main", "goal_marker"):
                try:
                    bid = sim.model.body_name2id(name)
                    return np.asarray(sim.data.body_xpos[bid], dtype=float)[:2]
                except Exception:
                    continue
        except Exception:
            pass
        return np.zeros(2)

    def _read_cube_pos(self, env) -> np.ndarray | None:
        # The trapped cuboid is body "small_object_main"; obs only exposes
        # the cup pose ("object_pos"), so we read the cube directly from
        # the sim like the task scorer does (cup_extract_v1.py).
        if env is None:
            return None
        try:
            sim = env._env.sim
            bid = sim.model.body_name2id("small_object_main")
            return np.asarray(sim.data.body_xpos[bid], dtype=float).copy()
        except Exception:
            return None

    def _read_cube_quat_xyzw(self, env) -> np.ndarray | None:
        # MuJoCo stores quats as wxyz; the rest of this file uses xyzw.
        if env is None:
            return None
        try:
            sim = env._env.sim
            bid = sim.model.body_name2id("small_object_main")
            w, x, y, z = np.asarray(sim.data.body_xquat[bid], dtype=float)
            return np.array([x, y, z, w])
        except Exception:
            return None

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0.0, 0.0, 0.0, 1.0])))
        cup_pos  = np.asarray(obs.get("object_pos",  obs.get("cube_pos")),  dtype=float)
        cup_quat = np.asarray(obs.get("object_quat", obs.get("cube_quat")), dtype=float)

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))
            self._lift_z  = self._table_z + LIFT_HEIGHT_ABOVE_TBL

        # ── 1. Settle: latch handle xy, grasp yaw, goal xy ───────────
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = -1.0
            if self.step_in_phase >= self.budgets["settle"]:
                handle_world = self._handle_world_pos(cup_pos, cup_quat)
                self._handle_xy = handle_world[:2].copy()
                handle_yaw = _yaw_from_quat(cup_quat)  # cup local-x in world
                grip_yaw   = _shortest_grasp_yaw(handle_yaw)
                self._grasp_quat = _grasp_quat_for_yaw(grip_yaw)
                self._cup_to_handle_xy = (handle_world[:2] - cup_pos[:2]).copy()
                if env is not None:
                    self._goal_xy = self._read_goal_xy(env)
                else:
                    self._goal_xy = np.zeros(2)
                print(f"  [plan] handle_xy={np.round(self._handle_xy, 3)}  "
                      f"handle_yaw={np.degrees(handle_yaw):+.1f}  "
                      f"grip_yaw={np.degrees(grip_yaw):+.1f}  "
                      f"goal_xy={np.round(self._goal_xy, 3)}", flush=True)
                self._advance("align_above_handle")
            self.step_in_phase += 1
            return action

        # ── 2. Align above handle (top-down quat, hover) ─────────────
        if self.phase == "align_above_handle":
            target = np.array([
                self._handle_xy[0],
                self._handle_xy[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_xy=4.0, target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            xy_ok  = xy_err < POS_TOL_XY
            z_ok   = abs(target[2] - eef_pos[2]) < POS_TOL_Z
            ori_ok = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, self._grasp_quat))) < ORI_TOL
            if (xy_ok and z_ok and ori_ok) \
                    or self.step_in_phase >= self.budgets["align"]:
                z_above_tbl = float(eef_pos[2]) - self._table_z
                print(f"  [align]   eef_xy_err={xy_err:.3f}  "
                      f"eef_z=table+{z_above_tbl:.3f}  "
                      f"reached_tol={xy_ok and z_ok and ori_ok}", flush=True)
                self._advance("descend_to_handle")
            self.step_in_phase += 1
            return action

        # ── 3. Descend onto handle ───────────────────────────────────
        if self.phase == "descend_to_handle":
            target = np.array([
                self._handle_xy[0],
                self._handle_xy[1],
                self._table_z + HANDLE_GRASP_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_z=6.0, target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.005
            if reached or self.step_in_phase >= self.budgets["descend"]:
                z_above_tbl = float(eef_pos[2]) - self._table_z
                xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
                print(f"  [descend] reached eef_z=table+{z_above_tbl:.3f}  "
                      f"target=table+{HANDLE_GRASP_Z_ABOVE_TBL:.3f}  "
                      f"xy_err={xy_err:.3f}  reached_tol={reached}", flush=True)
                self._advance("close_gripper")
            self.step_in_phase += 1
            return action

        # ── 4. Close gripper ─────────────────────────────────────────
        if self.phase == "close_gripper":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase >= self.budgets["close"]:
                self._advance("lift_cup")
            self.step_in_phase += 1
            return action

        # ── 5. Lift straight up ──────────────────────────────────────
        if self.phase == "lift_cup":
            target = np.array([eef_pos[0], eef_pos[1], self._lift_z])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.012
            if reached or self.step_in_phase >= self.budgets["lift"]:
                # Refresh cup-to-handle offset using the in-air pose so
                # the translate target reflects whatever the cup yawed
                # to during the close/lift squeeze.
                handle_world_now = self._handle_world_pos(cup_pos, cup_quat)
                self._cup_to_handle_xy = (handle_world_now[:2] - cup_pos[:2]).copy()

                # Pick a temp drop xy: TEMP_DROP_DISTANCE_XY away from
                # the goal, in the direction that keeps it furthest
                # from the table edge. We try +x, -x, +y, -y and pick
                # the candidate with the largest margin to the table
                # boundary (after clamping to TABLE_HALF_XY).
                best_xy = None
                best_dir = np.array([1.0, 0.0])
                best_margin = -np.inf
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    cand = np.array([
                        self._goal_xy[0] + dx * TEMP_DROP_DISTANCE_XY,
                        self._goal_xy[1] + dy * TEMP_DROP_DISTANCE_XY,
                    ])
                    cand_clamped = np.clip(cand, -TABLE_HALF_XY, TABLE_HALF_XY)
                    # Margin = how close we got to the requested offset
                    # after clamping (closer to 0 means we hit the edge).
                    margin = -float(np.linalg.norm(cand - cand_clamped))
                    if margin > best_margin:
                        best_margin = margin
                        best_xy = cand_clamped
                        best_dir = np.array([float(dx), float(dy)])
                self._temp_drop_xy = best_xy
                self._temp_drop_dir = best_dir

                # Eef.xy ≈ handle.xy. We want cup_body.xy = temp_drop_xy
                # for the flip, so:
                #   eef.xy = temp_drop_xy + (handle.xy - cup_body.xy)
                self._translate_target_xy = (
                    self._temp_drop_xy + self._cup_to_handle_xy
                )
                # Snapshot the pre-flip wrist orientation so the unflip
                # phase can drive back to the same yaw without any
                # accumulated drift from the flip slew.
                self._pre_flip_quat = self._grasp_quat.copy()
                print(f"  [plan] goal_xy={np.round(self._goal_xy, 3)}  "
                      f"temp_drop_xy={np.round(self._temp_drop_xy, 3)}",
                      flush=True)
                self._advance("translate_to_goal")
            self.step_in_phase += 1
            return action

        # ── 6. Translate at altitude so cup body sits over the goal ──
        if self.phase == "translate_to_goal":
            target = np.array([
                self._translate_target_xy[0],
                self._translate_target_xy[1],
                self._lift_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)
            xy_ok = float(np.linalg.norm(target[:2] - eef_pos[:2])) < POS_TOL_XY
            if xy_ok or self.step_in_phase >= self.budgets["translate"]:
                self._advance("flip_cup")
            self.step_in_phase += 1
            return action

        # ── 7. Flip: smoothly slew wrist 180 deg about local x ───────
        # Local x of the gripper is (by construction of grasp_quat) the
        # handle's long axis -- the only axis that lets the jaws stay
        # aligned with the handle as the cup rolls over. Post-multiply
        # so the rotation is in the gripper's own frame.
        if self.phase == "flip_cup":
            progress = min(self.step_in_phase / max(self.budgets["flip"], 1), 1.0)
            angle    = progress * FLIP_TARGET_ANGLE
            q_local  = _axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), angle)
            target_q = _quat_mul_xyzw(self._grasp_quat, q_local)
            target_p = np.array([
                self._translate_target_xy[0],
                self._translate_target_xy[1],
                self._lift_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   target_quat=target_q)
            if self.step_in_phase >= self.budgets["flip"]:
                # Lock the inverted orientation as the new "target_quat"
                # for the remaining phases.
                self._grasp_quat = _quat_mul_xyzw(
                    self._grasp_quat,
                    _axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), FLIP_TARGET_ANGLE),
                )
                self._advance("hold_for_drop")
            self.step_in_phase += 1
            return action

        # ── 8. Hold inverted pose; cuboid falls and settles ──────────
        if self.phase == "hold_for_drop":
            target_p = np.array([
                self._translate_target_xy[0],
                self._translate_target_xy[1],
                self._lift_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   target_quat=self._grasp_quat)
            if self.step_in_phase >= self.budgets["hold_drop"]:
                self._advance("unflip_cup")
            self.step_in_phase += 1
            return action

        # ── 9. Unflip: rotate wrist -180 deg back to upright ─────────
        # Mirror of flip_cup. We slew from the inverted pose (current
        # _grasp_quat = pre_flip_quat * Rx(pi)) back to pre_flip_quat
        # by post-multiplying a rotation that goes from pi -> 0 about
        # gripper-local x.
        if self.phase == "unflip_cup":
            progress = min(self.step_in_phase / max(self.budgets["unflip"], 1), 1.0)
            angle    = (1.0 - progress) * FLIP_TARGET_ANGLE
            q_local  = _axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), angle)
            target_q = _quat_mul_xyzw(self._pre_flip_quat, q_local)
            target_p = np.array([
                self._translate_target_xy[0],
                self._translate_target_xy[1],
                self._lift_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   target_quat=target_q)
            if self.step_in_phase >= self.budgets["unflip"]:
                # Cup is upright again; lock grasp_quat back to the
                # pre-flip wrist pose for the remaining cup-park phases.
                self._grasp_quat = self._pre_flip_quat.copy()

                # Plan the cup park location: orthogonal to the temp
                # drop direction (so the cup ends up clear of BOTH the
                # goal and the dropped cube), at CUP_PARK_DISTANCE_XY
                # from the goal. Try both orthogonal sides and pick the
                # one with more table-edge margin.
                ortho = np.array([
                    -self._temp_drop_dir[1],
                    self._temp_drop_dir[0],
                ])
                best_park = None
                best_margin = -np.inf
                for s in (1.0, -1.0):
                    cand = self._goal_xy + s * ortho * CUP_PARK_DISTANCE_XY
                    cand_clamped = np.clip(cand, -TABLE_HALF_XY, TABLE_HALF_XY)
                    margin = -float(np.linalg.norm(cand - cand_clamped))
                    if margin > best_margin:
                        best_margin = margin
                        best_park = cand_clamped
                self._cup_park_xy = best_park
                # Eef.xy when carrying the upright cup = handle.xy.
                # cup_body.xy = eef.xy - cup_to_handle_xy, so for the
                # cup body to land at park xy:
                self._cup_park_eef_xy = (
                    self._cup_park_xy + self._cup_to_handle_xy
                )
                print(f"  [plan] cup_park_xy={np.round(self._cup_park_xy, 3)}",
                      flush=True)
                self._advance("cup_lift")
            self.step_in_phase += 1
            return action

        # ── 10. Short rise so the cup clears the dropped cube ────────
        if self.phase == "cup_lift":
            target_p = np.array([
                eef_pos[0], eef_pos[1],
                self._table_z + LIFT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   target_quat=self._grasp_quat)
            if self.step_in_phase >= self.budgets["cup_lift"]:
                self._advance("cup_translate")
            self.step_in_phase += 1
            return action

        # ── 11. Carry cup to park xy at altitude ─────────────────────
        if self.phase == "cup_translate":
            target_p = np.array([
                self._cup_park_eef_xy[0],
                self._cup_park_eef_xy[1],
                self._lift_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   target_quat=self._grasp_quat)
            xy_ok = float(np.linalg.norm(target_p[:2] - eef_pos[:2])) < POS_TOL_XY
            if xy_ok or self.step_in_phase >= self.budgets["cup_translate"]:
                self._advance("cup_descend")
            self.step_in_phase += 1
            return action

        # ── 12. Lower cup base to the table ──────────────────────────
        # The handle is ~5 cm above the cup base; bringing the handle
        # to z=table+CUP_RELEASE_HANDLE_Z_ABOVE_TBL puts the base ~1 cm
        # above the table -- close enough that releasing the jaws drops
        # the cup gently rather than from the carry altitude.
        if self.phase == "cup_descend":
            target_p = np.array([
                self._cup_park_eef_xy[0],
                self._cup_park_eef_xy[1],
                self._table_z + CUP_RELEASE_HANDLE_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   gain_z=4.0, target_quat=self._grasp_quat)
            reached = abs(target_p[2] - eef_pos[2]) < 0.008
            if reached or self.step_in_phase >= self.budgets["cup_descend"]:
                self._advance("cup_release")
            self.step_in_phase += 1
            return action

        # ── 13. Open jaws to release the cup ─────────────────────────
        if self.phase == "cup_release":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
            action[6] = -1.0
            if self.step_in_phase >= self.budgets["cup_release"]:
                self._advance("arm_lift")
            self.step_in_phase += 1
            return action

        # ── 14. Rise off the released cup before traversing ──────────
        if self.phase == "arm_lift":
            target_p = np.array([
                eef_pos[0], eef_pos[1],
                self._table_z + LIFT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=-1.0,
                                   target_quat=TOP_DOWN_QUAT)
            if self.step_in_phase >= self.budgets["arm_lift"]:
                # Latch the cube's current xy as the pickup target. We
                # snapshot it once so a small post-release wobble of the
                # cube doesn't keep moving the setpoint.
                cube_pos = self._read_cube_pos(env)
                if cube_pos is not None:
                    self._cube_pickup_xy = cube_pos[:2].copy()
                else:
                    # Fallback: assume the cube ended up at temp_drop_xy.
                    self._cube_pickup_xy = self._temp_drop_xy.copy()
                # Align the grasp yaw to the cube's resting orientation.
                # The cube has 4-fold symmetry about its own z (square
                # base, half_xy = 0.02), so wrapping the world-yaw to
                # [-pi/4, pi/4] picks the closest equivalent grasp --
                # avoids a >45 deg wrist twist when the cube settled
                # rotated. Falls back to top-down if we can't read the
                # quat for any reason.
                cube_quat = self._read_cube_quat_xyzw(env)
                if cube_quat is not None:
                    cube_yaw = _yaw_from_quat(cube_quat)
                    grip_yaw = _shortest_grasp_yaw(cube_yaw)
                    self._grasp_quat = _grasp_quat_for_yaw(grip_yaw)
                    print(f"  [plan] cube_pickup_xy="
                          f"{np.round(self._cube_pickup_xy, 3)}  "
                          f"cube_yaw={np.degrees(cube_yaw):+.1f}  "
                          f"grip_yaw={np.degrees(grip_yaw):+.1f}",
                          flush=True)
                else:
                    self._grasp_quat = TOP_DOWN_QUAT.copy()
                    print(f"  [plan] cube_pickup_xy="
                          f"{np.round(self._cube_pickup_xy, 3)}  "
                          f"cube_yaw=unknown -> top-down", flush=True)
                self._advance("approach_cube")
            self.step_in_phase += 1
            return action

        # ── 15. Hover above the dropped cube ─────────────────────────
        # Yaw target = self._grasp_quat, set at end of arm_lift to the
        # cube's settled yaw wrapped into [-pi/4, pi/4] (4-fold cube
        # symmetry) so the wrist takes the shortest twist to align.
        if self.phase == "approach_cube":
            target_p = np.array([
                self._cube_pickup_xy[0],
                self._cube_pickup_xy[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=-1.0,
                                   gain_xy=4.0, target_quat=self._grasp_quat)
            xy_ok = float(np.linalg.norm(target_p[:2] - eef_pos[:2])) < POS_TOL_XY
            if xy_ok or self.step_in_phase >= self.budgets["approach_cube"]:
                self._advance("descend_cube")
            self.step_in_phase += 1
            return action

        # ── 16. Descend onto the cube ────────────────────────────────
        # Both z AND xy must converge before we close. If approach_cube
        # ran out of budget on a long traverse, the arm enters descend
        # still chasing xy at altitude; exiting on z-reached alone would
        # clamp on empty air.
        if self.phase == "descend_cube":
            target_p = np.array([
                self._cube_pickup_xy[0],
                self._cube_pickup_xy[1],
                self._table_z + CUBE_GRASP_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=-1.0,
                                   gain_z=6.0, target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target_p[:2] - eef_pos[:2]))
            z_reached = abs(target_p[2] - eef_pos[2]) < 0.005
            xy_ok = xy_err < POS_TOL_XY
            if (z_reached and xy_ok) or self.step_in_phase >= self.budgets["descend_cube"]:
                self._advance("close_cube")
            self.step_in_phase += 1
            return action

        # ── 17. Close on the cube ────────────────────────────────────
        if self.phase == "close_cube":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase >= self.budgets["close_cube"]:
                self._advance("lift_cube")
            self.step_in_phase += 1
            return action

        # ── 18. Lift the cube off the table ──────────────────────────
        if self.phase == "lift_cube":
            target_p = np.array([
                eef_pos[0], eef_pos[1],
                self._table_z + LIFT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   target_quat=self._grasp_quat)
            if self.step_in_phase >= self.budgets["lift_cube"]:
                self._advance("translate_cube")
            self.step_in_phase += 1
            return action

        # ── 19. Carry cube to goal xy at altitude ────────────────────
        if self.phase == "translate_cube":
            target_p = np.array([
                self._goal_xy[0],
                self._goal_xy[1],
                self._table_z + LIFT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   target_quat=self._grasp_quat)
            xy_ok = float(np.linalg.norm(target_p[:2] - eef_pos[:2])) < POS_TOL_XY
            if xy_ok or self.step_in_phase >= self.budgets["translate_cube"]:
                self._advance("descend_goal")
            self.step_in_phase += 1
            return action

        # ── 20. Lower cube onto the goal ─────────────────────────────
        if self.phase == "descend_goal":
            target_p = np.array([
                self._goal_xy[0],
                self._goal_xy[1],
                self._table_z + CUBE_PLACE_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target_p, gripper=1.0,
                                   gain_z=4.0, target_quat=self._grasp_quat)
            reached = abs(target_p[2] - eef_pos[2]) < 0.005
            if reached or self.step_in_phase >= self.budgets["descend_goal"]:
                self._advance("release_cube")
            self.step_in_phase += 1
            return action

        # ── 21. Open jaws so cube settles on the goal ────────────────
        if self.phase == "release_cube":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
            action[6] = -1.0
            if self.step_in_phase >= self.budgets["release_cube"]:
                self._advance("done")
            self.step_in_phase += 1
            return action

        # ── 22. Done: hold pose, gripper open ────────────────────────
        target_p = np.array([eef_pos[0], eef_pos[1],
                             self._table_z + LIFT_HEIGHT_ABOVE_TBL])
        action = self._move_to(eef_pos, eef_quat, target_p, gripper=-1.0,
                               target_quat=self._grasp_quat)
        self.step_in_phase += 1
        return action


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
    env    = ib.make("ib-p4-cup-extract-v1", privileged_obs=True, render_mode=render_mode)
    policy = CupExtractPolicy()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 17)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===  "
              f"cup_pos={np.round(obs.get('object_pos', obs.get('cube_pos')), 3)}",
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
