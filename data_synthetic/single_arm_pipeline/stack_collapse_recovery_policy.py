"""Scripted policy for ``ib-p6-stack-collapse-recovery-v1``.

Task semantics (from imbench/tasks/puzzle/stack_collapse_recovery_v1.py):
- Three cubes spawn stacked largest-on-bottom: cube_large (half=0.030),
  cube_medium (half=0.025), cube_small (half=0.020).
- A few control steps in, the tower is *always* knocked over. Two
  collapse modes (50/50 by default):
    * "top"  — only cube_small is kicked; medium stays seated on large.
    * "both" — medium + small both fly in different directions.
  cube_large is never kicked, so it's the natural anchor for the
  rebuild.
- Success requires: exactly one cube touches the table (i.e. the bottom
  of a 3-stack), no cube touches the gripper, held for 100 ticks.

Strategy
--------
1. Settle: cache body IDs, snapshot table_z, wait for the kicked cubes
   to come to rest after the collapse fires (collapse fires at step 5;
   give it ~SETTLE_STEPS ticks more for the cubes to stop sliding).
2. Snapshot the bottom cube's xy as the rebuild site (cube_large is
   never kicked, so it's a stable anchor).
3. Re-stack medium-on-large, then small-on-medium. Each pick-place is:
     pre_grasp → descend → close → lift → pre_place → lower → release
   followed by a transient `retract_after_place` lift so we don't drag
   the cube off when moving on.
4. After the small cube is released, retreat the gripper well clear
   of the tower (HOLD_RETRACT_H above the small cube top, gripper open)
   and idle until the episode ends — the scorer needs `stable_steps`
   consecutive ticks with no gripper contact.

Skip-already-stacked optimisation
---------------------------------
In "top" collapse mode the medium cube is still seated on the large
cube. We detect that at end of settle by checking whether medium's
xyz matches the expected stacked pose; if so, we skip directly to
the small-cube placement and save a redundant pick-place.

Phase machine
─────────────
  settle
  → m_pre_grasp → m_descend → m_close → m_lift
  → m_pre_place → m_lower → m_release → m_retract
  → s_pre_grasp → s_descend → s_close → s_lift
  → s_pre_place → s_lower → s_release → s_retract
  → hold (idle while scorer counts stable ticks)
"""

from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface


# ── Cube geometry (mirrors stack_collapse_recovery_v1._CUBES) ───────
LARGE_HALF  = 0.030
MEDIUM_HALF = 0.025
SMALL_HALF  = 0.020

# Body names in robosuite "<name>_main" convention
BODY_LARGE  = "cube_large_main"
BODY_MEDIUM = "cube_medium_main"
BODY_SMALL  = "cube_small_main"

# ── Gripper orientation ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw, EEF Z → world −Z

# ── Heights above table top (m) ──────────────────────────────────────
PRE_GRASP_H = 0.16   # hover above a cube before descending to grab it
LIFT_H      = 0.20   # carry altitude (well clear of the tower-in-progress)
PRE_PLACE_H = 0.16   # hover above the stack target before lowering
# When releasing, we want the cube's BOTTOM to just touch the support's
# top. Place z is computed dynamically from the support's top + cube_half
# + a small safety margin so the gripper doesn't ram the existing stack.
PLACE_SAFETY_GAP = 0.004
# After release, retract straight up before moving away — keeps the
# gripper fingers clear of the just-placed cube.
RETRACT_H_AFTER_PLACE = 0.18
# Final retreat before the scoring hold. Must be tall enough that a
# fully open gripper (Panda finger length ~5 cm) sits clear of the
# small cube top. small_top is at table + 2*large + 2*medium + 2*small =
# table + 0.150; HOLD_RETRACT_H is above that.
HOLD_RETRACT_H = 0.30

# ── Stack-detection tolerance ────────────────────────────────────────
# Used at end of settle to decide whether the medium cube is already in
# its correct stacked position (collapse mode == "top"). Tight: in
# "both" mode the medium can land anywhere within ~10 cm of the large
# cube and we MUST NOT skip the pickup. The "top" mode leaves medium
# essentially exactly on the large cube, so a 5 mm tolerance is plenty.
STACK_DETECT_XY_TOL = 0.005
STACK_DETECT_Z_TOL  = 0.005

# ── Phase step budgets (ticks @ 100 Hz) ─────────────────────────────
# Episode horizon is 2000 ticks and the scorer needs 100 *consecutive*
# stable ticks → the manipulation must finish in ~1900 ticks. With the
# slower per-tick action cap (MAX_ACTION_STEP=0.5), effective eef
# velocity is ~6 cm/s, so a 15-cm descent needs ~250 ticks. Budgets
# below give each motion ~2× the original headroom and still fit two
# pick-places + holds well within the horizon.
SETTLE_STEPS         = 60
PRE_GRASP_MAX        = 240
DESCEND_MAX          = 260
CLOSE_STEPS          = 30
LIFT_MAX             = 220
PRE_PLACE_MAX        = 240
LOWER_MAX            = 220
RELEASE_STEPS        = 20
RETRACT_AFTER_MAX    = 160
HOLD_RETRACT_MAX     = 200

# ── Tolerances ───────────────────────────────────────────────────────
POS_TOL_XY = 0.010
POS_TOL_Z  = 0.010
GRASP_Z_REACHED_TOL = 0.012   # looser than POS_TOL_Z; the cube wobbles
PLACE_Z_REACHED_TOL = 0.010

# ── Control gains ────────────────────────────────────────────────────
# OSC interprets the per-axis action as a Cartesian delta with kp=200.
# `MAX_ACTION_STEP` caps the per-tick action; 0.5 → roughly half the
# saturated velocity (≈6–8 cm/s in xy/z) for cleaner, slower demos.
# Gains stay high enough to reach saturation when far from target;
# the cap is what slows the steady-state motion.
GAIN_XY         = 8.0
GAIN_Z          = 10.0
ROT_GAIN        = 2.0
MAX_ACTION_STEP = 0.5

# ── Setpoint ramp (smoothstep) ───────────────────────────────────────
# Same scheme as cup_extract_policy: when the target setpoint changes,
# ramp the action magnitude from RAMP_START_FRAC up to 1.0 over
# RAMP_TICKS so we don't slam into a brand-new target with a full action.
RAMP_TICKS        = 16
RAMP_START_FRAC   = 0.3
TARGET_CHANGE_TOL = 0.015


# ── Quaternion helpers (xyzw) ────────────────────────────────────────

def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])


def _quat_err_axis_angle(cur: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Axis-angle error rotation that takes `cur` → `tgt`."""
    cx, cy, cz, cw = cur
    tx, ty, tz, tw = tgt
    ccx, ccy, ccz, ccw = -cx, -cy, -cz, cw
    ex = tw*ccx + tx*ccw + ty*ccz - tz*ccy
    ey = tw*ccy - tx*ccz + ty*ccw + tz*ccx
    ez = tw*ccz + tx*ccy - ty*ccx + tz*ccw
    ew = tw*ccw - tx*ccx - ty*ccy - tz*ccz
    if ew < 0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    sin_half = float(np.sqrt(ex*ex + ey*ey + ez*ez))
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, ew)
    return np.array([ex, ey, ez]) / sin_half * angle


def _top_face_yaw_from_xmat(xmat: np.ndarray) -> float:
    """Yaw (about world Z) of one of the cube's top-face edges.

    The cube can be resting on any face — so the body axis that's
    currently pointing up (in world Z) is whichever of ±body-X, ±body-Y,
    ±body-Z has the largest |z| component. The *other two* body axes
    span the horizontal top face; we take whichever has the smaller
    |z| (the most-horizontal one) and return its world XY bearing.

    The bearing is collapsed mod π/2 (cube top is square — 4-fold
    symmetric — so all four edges are interchangeable) and then mod π
    (gripper jaws are 2-fold symmetric, but π/2 already implies π).
    Final range: (−π/4, π/4]. That keeps the wrist rotation small.
    """
    cols = [xmat[:, i] for i in range(3)]                # body axes in world
    up = int(np.argmax([abs(c[2]) for c in cols]))       # body axis nearest world ±Z
    horiz = [i for i in range(3) if i != up]
    edge = cols[min(horiz, key=lambda i: abs(cols[i][2]))]   # most-horizontal of the other two
    yaw = float(np.arctan2(edge[1], edge[0]))
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4    # wrap to (−π/4, π/4]
    return yaw


def _grasp_quat_for_yaw(yaw: float) -> np.ndarray:
    """Top-down grasp with the wrist rotated by `yaw` about world Z.

    Note: the EEF body Z points along world −Z under TOP_DOWN_QUAT, so a
    rotation about the gripper's own body Z is opposite-sign to a world-Z
    rotation. We want the wrist to end up yawed by `+yaw` about world Z;
    that means right-multiplying by qz of −yaw in the body frame, OR
    (equivalent) left-multiplying by qz of +yaw in world but applying
    TOP_DOWN_QUAT first. Use right-multiplication so the world-frame
    yaw matches the cube's world-frame yaw directly.
    """
    half = 0.5 * (-yaw)   # body-Z rotation is opposite of world-Z under top-down
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(TOP_DOWN_QUAT, qz)


# ── Policy ───────────────────────────────────────────────────────────

class StackCollapseRecoveryPolicy(PolicyInterface):
    """Scripted re-stack policy for ib-p6-stack-collapse-recovery-v1.

    Reads cube poses from ``env.sim`` directly so it adapts to the
    randomised tower xy and to whichever collapse mode fired.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8

        # Body-ID cache (filled lazily once env is bound).
        self._bid_large:  int | None = None
        self._bid_medium: int | None = None
        self._bid_small:  int | None = None

        # Locked at end of settle.
        self._stack_xy:    np.ndarray | None = None  # rebuild site (large cube xy)
        self._large_top_z: float | None      = None  # support z for medium placement
        self._medium_already_stacked: bool   = False

        # Per-pickup grasp orientation. Locked when entering pre_grasp for
        # each cube so the descent/close use a consistent target quat even
        # if the cube rolls slightly while the gripper is closing.
        self._grasp_quat: np.ndarray = TOP_DOWN_QUAT.copy()
        # Latched once we've computed the grasp quat for each cube. Reset
        # at the top of each pickup so we capture the cube's *current*
        # yaw exactly once. Avoids re-reading on every tick (and avoids
        # the step_in_phase==0 trap — _advance sets it to 0 but the
        # caller's `step_in_phase += 1` runs immediately after, so the
        # very first tick of a phase already has step_in_phase==1).
        self._medium_yaw_locked: bool = False
        self._small_yaw_locked:  bool = False

        # Setpoint ramp.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

    # ── Phase transition helper ───────────────────────────────────────

    def _advance(self, next_phase: str, **kw):
        if self.verbose:
            extra = "  ".join(f"{k}={v}" for k, v in kw.items())
            print(
                f"  [SCR] [{self.step_in_phase:4d}t] {self.phase} → {next_phase}"
                + (f"  {extra}" if extra else ""),
                flush=True,
            )
        self.phase = next_phase
        self.step_in_phase = 0

    # ── Body ID cache ─────────────────────────────────────────────────

    def _cache_body_ids(self, env) -> bool:
        if self._bid_large is not None:
            return True
        try:
            sim = env.sim
            self._bid_large  = sim.model.body_name2id(BODY_LARGE)
            self._bid_medium = sim.model.body_name2id(BODY_MEDIUM)
            self._bid_small  = sim.model.body_name2id(BODY_SMALL)
            if self.verbose:
                print(
                    f"  [SCR] body IDs cached  large={self._bid_large}  "
                    f"medium={self._bid_medium}  small={self._bid_small}",
                    flush=True,
                )
            return True
        except (ValueError, KeyError) as e:
            if self.verbose:
                print(f"  [SCR] WARNING: body cache failed: {e}", flush=True)
            return False

    # ── Sim reads ─────────────────────────────────────────────────────

    def _body_xyz(self, env, bid: int) -> np.ndarray:
        return env.sim.data.body_xpos[bid].copy()

    def _body_yaw_pi(self, env, bid: int) -> float:
        """Cube top-face yaw, wrapped into (−π/4, π/4]. Cube can be
        resting on any face — picks the body axis nearest vertical as
        the up-axis, and uses the most-horizontal remaining body axis
        as the edge whose world-XY bearing defines yaw."""
        xmat = np.asarray(env.sim.data.body_xmat[bid], dtype=float).reshape(3, 3)
        return _top_face_yaw_from_xmat(xmat)

    # ── Setpoint ramp ─────────────────────────────────────────────────

    def _ramp(self, target: np.ndarray) -> float:
        if (self._last_target_pos is None
                or np.linalg.norm(target - self._last_target_pos) > TARGET_CHANGE_TOL):
            self._ticks_since_target_change = 0
        self._last_target_pos = target.copy()
        self._ticks_since_target_change += 1
        t = min(self._ticks_since_target_change / max(RAMP_TICKS, 1), 1.0)
        s = t * t * (3.0 - 2.0 * t)
        return RAMP_START_FRAC + (1.0 - RAMP_START_FRAC) * s

    # ── Move-to helper ────────────────────────────────────────────────

    def _move_to(self, eef_pos, eef_quat, target_pos, gripper,
                 gain_xy=GAIN_XY, gain_z=GAIN_Z, max_step=MAX_ACTION_STEP,
                 target_quat=None):
        ramp = self._ramp(target_pos)
        d = target_pos - eef_pos
        action = np.zeros(7)
        action[0] = float(np.clip(d[0] * gain_xy * ramp, -max_step, max_step))
        action[1] = float(np.clip(d[1] * gain_xy * ramp, -max_step, max_step))
        action[2] = float(np.clip(d[2] * gain_z  * ramp, -max_step, max_step))
        tgt_q = TOP_DOWN_QUAT if target_quat is None else target_quat
        ori_err = _quat_err_axis_angle(eef_quat, tgt_q)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.15, 0.15)
        action[6] = float(gripper)
        return action

    # ── Pickup-target helpers ─────────────────────────────────────────

    def _grasp_z(self, cube_z: float) -> float:
        """EEF z when closing on a cube.

        Panda fingers extend ~5 cm below the EEF frame; aiming the EEF
        slightly below the cube centre lets the fingertips wrap fully
        around the cube body rather than catching just the top edge.
        Even if descent times out a few mm short, the fingers still
        engage solid cube material.
        """
        return cube_z - 0.005

    def _place_z(self, support_top_z: float, cube_half: float) -> float:
        """EEF z when releasing a cube onto a support. Place it slightly
        above the support's top so the cube settles down rather than
        ramming the existing stack."""
        return support_top_z + cube_half + PLACE_SAFETY_GAP

    # ── Main act() ────────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0., 0., 0., 1.])))

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))

        # ────────────────────────────────────────────────────────────
        # 1. Settle: wait for collapse to finish, snapshot anchors
        # ────────────────────────────────────────────────────────────
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = -1.0   # open
            if env is not None:
                self._cache_body_ids(env)

            if (self.step_in_phase >= SETTLE_STEPS
                    and env is not None
                    and self._bid_large is not None):
                large_xyz = self._body_xyz(env, self._bid_large)
                self._stack_xy    = large_xyz[:2].copy()
                self._large_top_z = float(large_xyz[2]) + LARGE_HALF

                # Detect "top" collapse mode: is medium already stacked?
                medium_xyz = self._body_xyz(env, self._bid_medium)
                expected_med_z = self._large_top_z + MEDIUM_HALF
                xy_close = (np.linalg.norm(medium_xyz[:2] - self._stack_xy)
                            < STACK_DETECT_XY_TOL)
                z_close  = abs(medium_xyz[2] - expected_med_z) < STACK_DETECT_Z_TOL
                self._medium_already_stacked = bool(xy_close and z_close)

                if self.verbose:
                    print(
                        f"  [SCR] settle done  "
                        f"large_xyz={np.round(large_xyz, 3)}  "
                        f"medium_xyz={np.round(medium_xyz, 3)}  "
                        f"medium_already_stacked={self._medium_already_stacked}",
                        flush=True,
                    )

                next_phase = "s_pre_grasp" if self._medium_already_stacked else "m_pre_grasp"
                self._advance(next_phase)
            self.step_in_phase += 1
            return action

        # ───────────────── Medium pick-and-place ─────────────────────

        if self.phase == "m_pre_grasp":
            if not self._medium_yaw_locked:
                yaw = self._body_yaw_pi(env, self._bid_medium)
                self._grasp_quat = _grasp_quat_for_yaw(yaw)
                self._medium_yaw_locked = True
                if self.verbose:
                    print(f"  [SCR] medium grasp_yaw={np.degrees(yaw):+.1f}°",
                          flush=True)
            cube_xyz = self._body_xyz(env, self._bid_medium)
            target = np.array([cube_xyz[0], cube_xyz[1],
                               self._table_z + PRE_GRASP_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_err  = abs(target[2] - eef_pos[2])
            if (xy_err < POS_TOL_XY and z_err < POS_TOL_Z) \
                    or self.step_in_phase >= PRE_GRASP_MAX:
                self._advance("m_descend",
                              xy_err=round(xy_err, 4), z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= PRE_GRASP_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "m_descend":
            cube_xyz = self._body_xyz(env, self._bid_medium)
            grasp_z  = self._grasp_z(cube_xyz[2])
            target   = np.array([cube_xyz[0], cube_xyz[1], grasp_z])
            action   = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                     gain_z=5.0,
                                     target_quat=self._grasp_quat)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < GRASP_Z_REACHED_TOL or self.step_in_phase >= DESCEND_MAX:
                self._advance("m_close",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= DESCEND_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "m_close":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.25, 0.25)
            action[6] = 1.0   # close
            if self.step_in_phase >= CLOSE_STEPS:
                self._advance("m_lift")
            self.step_in_phase += 1
            return action

        if self.phase == "m_lift":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + LIFT_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < POS_TOL_Z or self.step_in_phase >= LIFT_MAX:
                self._advance("m_pre_place",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= LIFT_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "m_pre_place":
            target = np.array([self._stack_xy[0], self._stack_xy[1],
                               self._table_z + PRE_PLACE_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_err  = abs(target[2] - eef_pos[2])
            if (xy_err < POS_TOL_XY and z_err < POS_TOL_Z) \
                    or self.step_in_phase >= PRE_PLACE_MAX:
                self._advance("m_lower",
                              xy_err=round(xy_err, 4),
                              timeout=(self.step_in_phase >= PRE_PLACE_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "m_lower":
            # Refresh the support height each tick — large cube's z can
            # drift very slightly from solver jitter even though it's
            # never kicked.
            large_xyz = self._body_xyz(env, self._bid_large)
            support_top_z = float(large_xyz[2]) + LARGE_HALF
            place_z = self._place_z(support_top_z, MEDIUM_HALF)
            target  = np.array([self._stack_xy[0], self._stack_xy[1], place_z])
            action  = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                    gain_z=4.0)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < PLACE_Z_REACHED_TOL or self.step_in_phase >= LOWER_MAX:
                self._advance("m_release",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= LOWER_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "m_release":
            action = np.zeros(7)
            action[6] = -1.0   # open
            if self.step_in_phase >= RELEASE_STEPS:
                self._advance("m_retract")
            self.step_in_phase += 1
            return action

        if self.phase == "m_retract":
            # Lift straight up so the open gripper clears the just-placed
            # medium cube before we move toward the small cube.
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + RETRACT_H_AFTER_PLACE])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < POS_TOL_Z or self.step_in_phase >= RETRACT_AFTER_MAX:
                self._advance("s_pre_grasp",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= RETRACT_AFTER_MAX))
            self.step_in_phase += 1
            return action

        # ───────────────── Small pick-and-place ──────────────────────

        if self.phase == "s_pre_grasp":
            if not self._small_yaw_locked:
                yaw = self._body_yaw_pi(env, self._bid_small)
                self._grasp_quat = _grasp_quat_for_yaw(yaw)
                self._small_yaw_locked = True
                if self.verbose:
                    print(f"  [SCR] small grasp_yaw={np.degrees(yaw):+.1f}°",
                          flush=True)
            cube_xyz = self._body_xyz(env, self._bid_small)
            target = np.array([cube_xyz[0], cube_xyz[1],
                               self._table_z + PRE_GRASP_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_err  = abs(target[2] - eef_pos[2])
            if (xy_err < POS_TOL_XY and z_err < POS_TOL_Z) \
                    or self.step_in_phase >= PRE_GRASP_MAX:
                self._advance("s_descend",
                              xy_err=round(xy_err, 4), z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= PRE_GRASP_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "s_descend":
            cube_xyz = self._body_xyz(env, self._bid_small)
            grasp_z  = self._grasp_z(cube_xyz[2])
            target   = np.array([cube_xyz[0], cube_xyz[1], grasp_z])
            action   = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                     gain_z=5.0,
                                     target_quat=self._grasp_quat)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < GRASP_Z_REACHED_TOL or self.step_in_phase >= DESCEND_MAX:
                self._advance("s_close",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= DESCEND_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "s_close":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.25, 0.25)
            action[6] = 1.0
            if self.step_in_phase >= CLOSE_STEPS:
                self._advance("s_lift")
            self.step_in_phase += 1
            return action

        if self.phase == "s_lift":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + LIFT_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < POS_TOL_Z or self.step_in_phase >= LIFT_MAX:
                self._advance("s_pre_place",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= LIFT_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "s_pre_place":
            target = np.array([self._stack_xy[0], self._stack_xy[1],
                               self._table_z + PRE_PLACE_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_err  = abs(target[2] - eef_pos[2])
            if (xy_err < POS_TOL_XY and z_err < POS_TOL_Z) \
                    or self.step_in_phase >= PRE_PLACE_MAX:
                self._advance("s_lower",
                              xy_err=round(xy_err, 4),
                              timeout=(self.step_in_phase >= PRE_PLACE_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "s_lower":
            # Place small on top of medium. Read medium's live z each
            # tick — its top can drift slightly while we approach.
            medium_xyz = self._body_xyz(env, self._bid_medium)
            support_top_z = float(medium_xyz[2]) + MEDIUM_HALF
            place_z = self._place_z(support_top_z, SMALL_HALF)
            target  = np.array([self._stack_xy[0], self._stack_xy[1], place_z])
            action  = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                    gain_z=4.0)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < PLACE_Z_REACHED_TOL or self.step_in_phase >= LOWER_MAX:
                self._advance("s_release",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= LOWER_MAX))
            self.step_in_phase += 1
            return action

        if self.phase == "s_release":
            action = np.zeros(7)
            action[6] = -1.0
            if self.step_in_phase >= RELEASE_STEPS:
                self._advance("s_retract")
            self.step_in_phase += 1
            return action

        if self.phase == "s_retract":
            # Final retreat — gripper open, well clear of the small cube
            # so the scorer's "no gripper contact" check passes for the
            # full stable_steps window.
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + HOLD_RETRACT_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0)
            z_err = abs(target[2] - eef_pos[2])
            if z_err < POS_TOL_Z or self.step_in_phase >= HOLD_RETRACT_MAX:
                self._advance("hold",
                              z_err=round(z_err, 4),
                              timeout=(self.step_in_phase >= HOLD_RETRACT_MAX))
            self.step_in_phase += 1
            return action

        # ─────────────── Hold while scorer counts stable ticks ───────
        # Idle action with the gripper open and no motion. The arm
        # may drift slightly under gravity but the scorer only cares
        # about cube-table and cube-gripper contacts.
        action = np.zeros(7)
        action[6] = -1.0
        return action
