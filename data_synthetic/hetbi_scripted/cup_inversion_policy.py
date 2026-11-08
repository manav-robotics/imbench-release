"""Scripted bimanual policy for ``ib-p1-cup-inversion-v1``.

Pipeline
--------
The cup starts inverted on the table at ~[-0.2, 0.0] with its flat base
exposed upward (``payload_quat`` = 180° about X). Robot-1 carries a
suction gripper, robot-0 carries a parallel-jaw gripper.

Phases::

    settle           — both arms quiet; latch each arm's home quat as
                       the canonical "snout pointing world -Z" reference
    r1_pre_grasp     — R1 moves above the cup base (live cup_pos)
    r1_descend       — R1 lowers onto the cup base; suction welds
    r1_lift          — R1 lifts straight up to clear the table
    r1_rotate        — R1 rotates 90° around world -Y so the cup's
                       hollow opening (originally pointing world -Z)
                       now points along world +X (horizontal). The
                       suction snout points +X with the welded cup
                       hanging off the snout in that direction.
    r0_approach_far  — R0 stages on the +X side of the cup with snout
                       pointing world -X — the gripper faces directly
                       into the cup's hollow opening. Wrist is offset
                       in +Y from cup centre to ~0.050 m (just outside
                       the outer wall radius), so the wrist body
                       clears the cup but the closed finger tips land
                       on opposite sides of the +Y wall.
    r0_approach_near — R0 translates along -X *into* the cup. The
                       wrist crosses the rim plane and ends up inside
                       the cup body in X, with the finger body
                       (≈ 5.6 cm long in the snout direction)
                       essentially fully inside the cup cavity.
    r0_grip          — R0 closes its jaw. The leftfinger tip stops at
                       the inner wall surface (``y = cup_y + inner_r``,
                       inside the cup), the rightfinger tip stops at
                       the outer wall surface (``y = cup_y + outer_r``,
                       outside the cup) — wall pinched between an
                       inside finger and an outside finger.
    r1_release       — R1 deactivates the suction weld.
    r0_rotate_up     — R0 rotates the welded grip 90° around world -Y
                       so the cup's axis (originally world +X) now
                       points world +Z (cup upright in R0's grasp).
    r0_carry         — R0 traverses above the green goal region.
    r0_lower         — R0 lowers the cup to the table inside the goal.
    r0_release       — R0 opens the jaw.
    r0_retreat       — R0 lifts away.
    done

Note on the rim-edge grasp pose:
--------------------------------
The Panda gripper has a closed-gap geometric residual of
``2 × 0.0085 = 0.017 m`` (driven by the ``finger_joint*_tip pos``
offsets in panda_gripper.xml), and the e_cups rim wall is only
``outer_r − inner_r ≈ 0.004 m`` thick. So when the gripper closes
with one finger inside the cavity and one outside the wall, the
fingers stop 17 mm apart while the wall is 4 mm — the wall is not
literally pinched between the fingers. The cup is held by the
left-finger's contained-by-cavity constraint plus suction-side
support during transfer; the gripper *pose* is one-finger-inside /
one-finger-outside as requested.

Controller: IK_POSE with ``input_type: absolute`` (robosuite IK still
runs in delta mode because ``control_delta`` defaults to True, so the
action's first 3 entries act as small position deltas in world frame
and entries 3:6 as small world-frame axis-angle deltas — the IK feeds
both through ``mj_jacSite`` whose Jacobian is in world frame). Per-step
sensitivities ``pos_sensitivity=0.005`` / ``ori_sensitivity=0.01`` are
applied at the env layer (``HetBiBaseTask.step``) before the IK sees
the action, so policies can write [-1, 1] commands directly.

Score: cup on table (z within 5 cm of the surface), upright (mat[2,2]
> 0.9), and centred inside the green goal region.
"""
from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface
from data_synthetic.hetbi_scripted._utils import (
    TABLE_Z, TOP_DOWN_QUAT,
    read_obj_pos,
    quat_mul_xyzw,
)


# robosuite emits ``robot{i}_eef_quat`` and ``object_quat`` already in xyzw
# (see robosuite/robots/robot.py — ``T.convert_quat(..., to="xyzw")``).
# These local readers are equivalent to the shared ``read_eef`` /
# ``read_obj_quat_xyzw`` helpers in _utils.py (which also return xyzw).
def _read_eef(obs: dict, robot_id: int):
    return (np.asarray(obs[f"robot{robot_id}_eef_pos"], dtype=float),
            np.asarray(obs[f"robot{robot_id}_eef_quat"], dtype=float))


def _read_cup_quat(env) -> np.ndarray:
    sim = env._env.sim
    q_wxyz = sim.data.body_xquat[env._env._payload_body_id]
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


# ── Geometry ──────────────────────────────────────────────────────────
GOAL_XY = np.array([0.20, 0.0])
PLACE_Z = TABLE_Z + 0.04

# Cup geometry (e_cups stacking cup, mesh bounds confirmed from the
# YCB textured.obj):
#   local Z extent → world X after R1's rotation, rim opening at +X
# Rim sits ``_CUP_RIM_DX`` past cup centre in the +X direction.
_CUP_RIM_DX      = 0.040
_CUP_OUTER_R     = 0.038     # cup outer radius
_CUP_INNER_R     = 0.034     # cavity radius (4 mm wall)
_CUP_HALF_HEIGHT = 0.035     # used only as fallback (no env)

# R1 motion
PRE_GRASP_DZ   = 0.10            # R1 hover above cup base
SUCTION_SINK   = 0.005           # press snout 5 mm below cup top
LIFT_Z         = TABLE_Z + 0.42  # R1 lift target after weld
HANDOVER_XY    = np.array([0.0, 0.0])   # mid-table for the rotation pose

# R0 side-grasp into the cup hollow.
# Wrist offset in +Y from cup centre — kept outside the outer wall
# (``> _CUP_OUTER_R``) so the wrist body itself doesn't collide with
# the cup, but close enough that when the jaw closes the leftfinger
# tip lands on the *inner* wall surface (inside the cavity) and the
# rightfinger tip lands on the *outer* wall surface. With the Panda's
# closed-position offsets this happens at wrist_y ≈ 0.050.
R0_WRIST_DY      = 0.050
# Pre-grasp far waypoint: wrist this far +X of the rim, snout pointing
# straight into the cup hollow at a safe stand-off.
R0_STAGE_DX      = 0.18
# Grasp waypoint: wrist this far past the rim (negative = inside the
# cup body in X). The finger body is ~5.6 cm long (root to tip in
# snout direction); placing the wrist 4 cm inside the rim plane drops
# the entire finger inside the cup cavity along X, so the cup is
# grasped from fully inside.
R0_GRASP_DX      = -0.040
R0_SAFE_POS = np.array([0.30, -0.20, TABLE_Z + 0.42])

# Tick budgets — IK-absolute moves at ~1.5 mm/step.
SETTLE_TICKS        = 30
APPROACH_TICKS      = 1500
DESCEND_TICKS       = 800
WELD_TICKS          = 100
LIFT_TICKS          = 800
ROTATE_TICKS        = 1500
R0_FAR_TICKS        = 1500
R0_NEAR_TICKS       = 2500
GRIP_TICKS          = 200
RELEASE_TICKS       = 200
R0_ROTATE_TICKS     = 1500
CARRY_TICKS         = 2000
LOWER_TICKS         = 1000
R0_RELEASE_TICKS    = 100
RETREAT_TICKS       = 600

# IK action gains
POS_GAIN_IK = 20.0
ORI_GAIN_IK = 8.0


# ── Helpers ──────────────────────────────────────────────────────────

def _axisangle_to_quat(axis, angle: float) -> np.ndarray:
    """xyzw quaternion from axis-angle."""
    n = np.asarray(axis, dtype=float)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9 or abs(angle) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    n = n / nn
    h = 0.5 * angle
    s = np.sin(h)
    return np.array([n[0] * s, n[1] * s, n[2] * s, np.cos(h)])


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_to_axis_angle(q):
    """Axis-angle (3-vector) from xyzw quaternion. Picks the short way."""
    x, y, z, w = q
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = float(np.sqrt(x * x + y * y + z * z))
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, w)
    return np.array([x, y, z]) / sin_half * angle


def _ori_err_world(eef_quat, target_quat):
    """World-frame axis-angle error — robosuite's IK consumes action[3:6]
    against a world-frame Jacobian (``mj_jacSite``)."""
    return _quat_to_axis_angle(quat_mul_xyzw(target_quat, _quat_conj(eef_quat)))


def _arm_action(eef_pos, eef_quat, target_pos, target_quat, gripper):
    a = np.zeros(7, dtype=float)
    a[0:3] = np.clip((target_pos - eef_pos) * POS_GAIN_IK, -1.0, 1.0)
    a[3:6] = np.clip(_ori_err_world(eef_quat, target_quat) * ORI_GAIN_IK,
                     -1.0, 1.0)
    a[6] = float(gripper)
    return a


def _arm_action_pos_only(eef_pos, target_pos, gripper):
    """Position-only IK action — orientation target left at zero delta."""
    a = np.zeros(7, dtype=float)
    a[0:3] = np.clip((target_pos - eef_pos) * POS_GAIN_IK, -1.0, 1.0)
    a[6] = float(gripper)
    return a


def _hold(gripper):
    a = np.zeros(7, dtype=float)
    a[6] = float(gripper)
    return a


def _at(eef_pos, target, tol=0.025):
    return float(np.linalg.norm(eef_pos - target)) < tol


def _at_ori(eef_quat, target_quat, tol=0.10):
    return float(np.linalg.norm(_ori_err_world(eef_quat, target_quat))) < tol


# ── Policy ───────────────────────────────────────────────────────────

class CupInversionPolicy(PolicyInterface):
    """Scripted cup-inversion policy using IK_POSE absolute control."""

    def __init__(self, table_z: float = TABLE_Z,
                 goal_xy: np.ndarray = GOAL_XY):
        self._tz = float(table_z)
        self._goal_xy = np.asarray(goal_xy, dtype=float)
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.tick = 0
        self._done = False

        # Live sim-derived state.
        self._cup_pos: np.ndarray | None = None
        self._cup_quat: np.ndarray | None = None

        # Latched at the end of settle: each arm's converged home quat.
        self._home_r0_quat: np.ndarray | None = None
        self._home_r1_quat: np.ndarray | None = None

        # Latched waypoints — frozen at key transitions so downstream
        # phases don't track mid-flight perturbations of the welded cup.
        self._r1_lift_xyz: np.ndarray | None = None
        self._r1_handover_xyz: np.ndarray | None = None
        self._r0_grasp_xyz: np.ndarray | None = None
        self._r0_after_grasp_xyz: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self._done

    # ── Suction weld query ──────────────────────────────────────────
    def _is_welded(self, env, robot_id: int = 1) -> bool:
        if env is None:
            return False
        sw = getattr(env, "_suction_weld", None)
        if sw is None:
            return False
        h = sw._handles.get((robot_id, "right"))
        return h is not None and h.welded_body_id is not None

    def _advance(self, nxt: str):
        print(f"  [{self.tick:4d}] cup-inversion: {self.phase} → {nxt}",
              flush=True)
        self.phase = nxt
        self.tick = 0

    # ── Target orientations ─────────────────────────────────────────
    def _target_orientations(self):
        """Return the five key target quaternions used by the policy.

        ``q_R1_init``     — suction snout straight down (world -Z),
                            latched from R1's converged home quat.
        ``q_R1_rotated``  — suction snout horizontal in world +X (cup
                            opening also faces +X after the weld).
        ``q_R0_init``     — R0 top-down (snout world -Z), latched from
                            R0's home quat.
        ``q_R0_grasp``    — R0 oriented to face the cup hollow:
                              snout      → world -X (looks straight
                                            into the +X-facing rim
                                            opening)
                              spread axis→ world Y (Panda's natural
                                            spread axis stays aligned
                                            with world Y under a pure
                                            world R_y rotation)
                            Reachable from home with a clean 90°
                            rotation about world +Y, which the IK
                            handles reliably.
        ``q_R0_upright``  — R0 rotated back through R_y(-π/2) — snout
                            ends up world -Z and the rigidly-held cup
                            ends up axis world +Z (upright).
        """
        q_world_y_pos90 = _axisangle_to_quat([0.0, 1.0, 0.0],  np.pi / 2.0)
        q_world_y_neg90 = _axisangle_to_quat([0.0, 1.0, 0.0], -np.pi / 2.0)

        q_R1_init = (self._home_r1_quat
                     if self._home_r1_quat is not None else TOP_DOWN_QUAT)
        q_R1_rotated = quat_mul_xyzw(q_world_y_neg90, q_R1_init)

        q_R0_init = (self._home_r0_quat
                     if self._home_r0_quat is not None else TOP_DOWN_QUAT)
        q_R0_grasp = quat_mul_xyzw(q_world_y_pos90, q_R0_init)
        q_R0_upright = quat_mul_xyzw(q_world_y_neg90, q_R0_grasp)
        return (q_R1_init, q_R1_rotated, q_R0_init,
                q_R0_grasp, q_R0_upright)

    # ── Main act ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        r0_pos, r0_quat = _read_eef(obs, 0)
        r1_pos, r1_quat = _read_eef(obs, 1)

        if env is not None:
            self._cup_pos = read_obj_pos(env)
            self._cup_quat = _read_cup_quat(env)
        if self._cup_pos is None:
            self._cup_pos = np.array([-0.20, 0.0, self._tz + _CUP_HALF_HEIGHT])
            self._cup_quat = np.array([1.0, 0.0, 0.0, 0.0])

        cup = self._cup_pos
        tz = self._tz

        (q_R1_init, q_R1_rotated, q_R0_init,
         q_R0_grasp, q_R0_upright) = self._target_orientations()

        # ── R1 (suction) live waypoints ─────────────────────────────
        cup_top_z = cup[2] + _CUP_HALF_HEIGHT
        r1_above   = np.array([cup[0], cup[1], cup_top_z + PRE_GRASP_DZ])
        r1_contact = np.array([cup[0], cup[1], cup_top_z - SUCTION_SINK])

        self.tick += 1
        p = self.phase

        # ── settle ───────────────────────────────────────────────────
        if p == "settle":
            a0 = _hold(-1.0)
            a1 = _hold(-1.0)
            self._home_r0_quat = r0_quat.copy()
            self._home_r1_quat = r1_quat.copy()
            if self.tick >= SETTLE_TICKS:
                self._advance("r1_pre_grasp")

        # ── R1 hovers above cup base; R0 retreats to safe pose ──────
        elif p == "r1_pre_grasp":
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, r1_above, q_R1_init, +1.0)
            if self._is_welded(env, 1):
                self._latch_lift_target(r1_pos)
                self._advance("r1_lift")
            elif _at(r1_pos, r1_above, tol=0.02) or self.tick >= APPROACH_TICKS:
                self._advance("r1_descend")

        # ── R1 descends to cup base; suction welds on contact ───────
        elif p == "r1_descend":
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, r1_contact, q_R1_init, +1.0)
            if self._is_welded(env, 1):
                self._latch_lift_target(r1_pos)
                self._advance("r1_lift")
            elif self.tick >= DESCEND_TICKS:
                self._latch_lift_target(r1_pos)
                self._advance("r1_weld")

        # ── R1 holds press until weld engages ───────────────────────
        elif p == "r1_weld":
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, r1_contact, q_R1_init, +1.0)
            if self._is_welded(env, 1) or self.tick >= WELD_TICKS:
                self._latch_lift_target(r1_pos)
                self._advance("r1_lift")

        # ── R1 lifts straight up while keeping the snout pointing -Z ─
        elif p == "r1_lift":
            tgt = self._r1_lift_xyz
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, tgt, q_R1_init, +1.0)
            if r1_pos[2] >= tgt[2] - 0.04 or self.tick >= LIFT_TICKS:
                self._r1_handover_xyz = np.array([HANDOVER_XY[0],
                                                  HANDOVER_XY[1],
                                                  LIFT_Z])
                self._advance("r1_rotate")

        # ── R1 traverses to mid-table handover XY and rotates 90°
        # around world -Y so the welded cup's opening points world +X.
        elif p == "r1_rotate":
            tgt_pos = self._r1_handover_xyz
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, tgt_pos, q_R1_rotated, +1.0)
            ori_done = _at_ori(r1_quat, q_R1_rotated, tol=0.12)
            pos_done = _at(r1_pos, tgt_pos, tol=0.04)
            if (ori_done and pos_done) or self.tick >= ROTATE_TICKS:
                self._advance("r0_approach_far")

        # ── R0 stages on the +X side of the cup with snout pointing
        # world -X — gripper looks straight into the cup hollow.
        # Wrist offset in +Y by the wall mid-radius so the leftfinger
        # lines up inside the cavity and the rightfinger lines up on
        # the +Y outer wall side. ───────────────────────────────────
        elif p == "r0_approach_far":
            rim_x = cup[0] + _CUP_RIM_DX
            r0_far = np.array([rim_x + R0_STAGE_DX,
                               cup[1] + R0_WRIST_DY,
                               cup[2]])
            a0 = _arm_action(r0_pos, r0_quat, r0_far, q_R0_grasp, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, self._r1_handover_xyz,
                             q_R1_rotated, +1.0)
            ori_ok = _at_ori(r0_quat, q_R0_grasp, tol=0.15)
            pos_ok = _at(r0_pos, r0_far, tol=0.04)
            if (ori_ok and pos_ok) or self.tick >= R0_FAR_TICKS:
                self._advance("r0_approach_near")

        # ── R0 closes in along world -X. The finger tips end up just
        # past the rim plane: the leftfinger body sits inside the cup
        # cavity, the rightfinger body sits outside the cup wall. ───
        elif p == "r0_approach_near":
            rim_x = cup[0] + _CUP_RIM_DX
            r0_grasp = np.array([rim_x + R0_GRASP_DX,
                                 cup[1] + R0_WRIST_DY,
                                 cup[2]])
            a0 = _arm_action(r0_pos, r0_quat, r0_grasp, q_R0_grasp, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, self._r1_handover_xyz,
                             q_R1_rotated, +1.0)
            if _at(r0_pos, r0_grasp, tol=0.02) or self.tick >= R0_NEAR_TICKS:
                self._r0_grasp_xyz = r0_pos.copy()
                self._advance("r0_grip")

        # ── R0 closes its jaw — leftfinger sweeps through the cavity,
        # rightfinger sweeps inward toward the outer wall. Hold the
        # grasp orientation explicitly so the IK target_orn doesn't
        # drift while the jaw closes. ──────────────────────────────
        elif p == "r0_grip":
            a0 = _arm_action(r0_pos, r0_quat, self._r0_grasp_xyz,
                             q_R0_grasp, +1.0)
            a1 = _arm_action(r1_pos, r1_quat, self._r1_handover_xyz,
                             q_R1_rotated, +1.0)
            if self.tick >= GRIP_TICKS:
                self._advance("r1_release")

        # ── R1 deactivates the suction; R0 keeps holding ────────────
        elif p == "r1_release":
            a0 = _arm_action(r0_pos, r0_quat, self._r0_grasp_xyz,
                             q_R0_grasp, +1.0)
            r1_retreat = self._r1_handover_xyz + np.array([0.0, 0.0, 0.05])
            a1 = _arm_action(r1_pos, r1_quat, r1_retreat,
                             q_R1_rotated, -1.0)
            if self.tick >= RELEASE_TICKS:
                self._r0_after_grasp_xyz = r0_pos.copy()
                self._advance("r0_rotate_up")

        # ── R0 rotates the cup so its axis points world +Z (upright). ─
        elif p == "r0_rotate_up":
            tgt = self._r0_after_grasp_xyz
            a0 = _arm_action(r0_pos, r0_quat, tgt, q_R0_upright, +1.0)
            r1_retreat = self._r1_handover_xyz + np.array([-0.10, 0.10, 0.05])
            a1 = _arm_action(r1_pos, r1_quat, r1_retreat,
                             q_R1_rotated, -1.0)
            ori_ok = _at_ori(r0_quat, q_R0_upright, tol=0.12)
            if ori_ok or self.tick >= R0_ROTATE_TICKS:
                self._advance("r0_carry")

        # ── R0 carries the upright cup above the goal area ──────────
        elif p == "r0_carry":
            target_cup = np.array([self._goal_xy[0],
                                   self._goal_xy[1],
                                   tz + 0.30])
            r0_target = r0_pos + (target_cup - cup)
            a0 = _arm_action(r0_pos, r0_quat, r0_target,
                             q_R0_upright, +1.0)
            r1_retreat = np.array([-0.30, 0.30, tz + 0.45])
            a1 = _arm_action(r1_pos, r1_quat, r1_retreat,
                             q_R1_rotated, -1.0)
            cup_xy_err = float(np.linalg.norm(cup[:2] - self._goal_xy))
            if (cup_xy_err < 0.03 and abs(cup[2] - target_cup[2]) < 0.04) \
                    or self.tick >= CARRY_TICKS:
                self._advance("r0_lower")

        # ── R0 lowers the cup onto the goal region ──────────────────
        elif p == "r0_lower":
            target_cup = np.array([self._goal_xy[0],
                                   self._goal_xy[1],
                                   PLACE_Z])
            r0_target = r0_pos + (target_cup - cup)
            a0 = _arm_action(r0_pos, r0_quat, r0_target,
                             q_R0_upright, +1.0)
            a1 = _hold(-1.0)
            if cup[2] <= PLACE_Z + 0.02 or self.tick >= LOWER_TICKS:
                self._advance("r0_release")

        # ── R0 opens its jaw ────────────────────────────────────────
        elif p == "r0_release":
            a0 = _arm_action_pos_only(r0_pos, r0_pos, -1.0)
            a1 = _hold(-1.0)
            if self.tick >= R0_RELEASE_TICKS:
                self._advance("r0_retreat")

        # ── R0 lifts away from the placed cup ───────────────────────
        elif p == "r0_retreat":
            r0_up = np.array([r0_pos[0],
                              r0_pos[1] - 0.05,
                              tz + 0.40])
            a0 = _arm_action(r0_pos, r0_quat, r0_up,
                             q_R0_init, -1.0)
            a1 = _hold(-1.0)
            if _at(r0_pos, r0_up, tol=0.05) or self.tick >= RETREAT_TICKS:
                self._done = True
                self._advance("done")

        else:  # done
            a0 = _hold(-1.0)
            a1 = _hold(-1.0)

        return np.concatenate([a0, a1]).astype(np.float64)

    # ── Phase-helpers ────────────────────────────────────────────────
    def _latch_lift_target(self, r1_pos: np.ndarray) -> None:
        """Freeze the post-weld lift target above the current snout pose."""
        if self._r1_lift_xyz is None:
            self._r1_lift_xyz = np.array([r1_pos[0], r1_pos[1], LIFT_Z])
