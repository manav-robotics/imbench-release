"""Scripted bimanual policy for ``ib-p1-bowl-table-v1``.

Pipeline
--------
The bowl starts upside-down inside the +y compartment of the rack.
Robot-1 carries a suction gripper, robot-0 carries a parallel-jaw
gripper.  The controller is IK_POSE in absolute mode (delta inputs in
world frame, pre-scaled at the env layer).

Phases::

    settle           — both arms quiet; latch each arm's converged home
                       quat as the canonical "snout pointing world -Z"
                       reference for downstream rotations.
    r1_pre_grasp     — R1 hovers above the bowl base (live bowl_pos).
    r1_descend       — R1 lowers onto the upside-down bowl's flat base
                       (which now faces world +Z); suction welds.
    r1_lift          — R1 lifts straight up to clear the rack walls.
    r1_pitch         — R1 traverses to mid-table and rotates 90° about
                       world -Y so the welded bowl's cavity (originally
                       facing world -Z because the bowl is upside-down)
                       now faces world +X (cavity *horizontal*).
    r0_approach_far  — R0 stages on the +X side of the bowl with snout
                       pointing world -X.  Wrist offset in -Y from bowl
                       centre by ~ (inner_r + outer_r)/2 so that when
                       the jaw later closes one finger pad lands inside
                       the cavity (at the -Y wall) and the other pad
                       lands just past the outer wall.  Stand-off in +X.
    r0_approach_near — R0 translates *parallel* (snout direction held
                       fixed) along world -X so the wrist crosses the
                       rim plane: the finger bodies straddle the -Y
                       rim wall — one finger inside the cavity, the
                       other outside the bowl wall.
    r0_grip          — R0 closes the jaw.  Suction stays welded so the
                       bowl is fully supported while the jaw closes.
    r1_release       — R1 deactivates suction; the bowl is now held by
                       R0's jaw alone.  R0 holds its grasp pose.
    r1_retreat       — R1 pulls clear of the bowl by translating in
                       +Y (back toward its base) and up.
    r0_rotate_up     — R0 alone rotates the held bowl by R_y(-π/2):
                       cavity goes from world +X (horizontal) → world
                       +Z (upright).  R1 holds clear.
    r0_lower         — R0 lowers the bowl in place (no lateral carry)
                       until its base is on the table.
    r0_release       — R0 opens the jaw.
    r0_retreat       — R0 lifts away.
    done

Score: bowl on table (z within 5 cm of the surface) and upright
(``mat[2,2] > 0.9``) per ``BowlTableV1._compute_task_score``.
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


def _read_bowl_quat(env) -> np.ndarray:
    sim = env._env.sim
    q_wxyz = sim.data.body_xquat[env._env._payload_body_id]
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


# ── Bowl geometry (YCB ``bowl`` mesh, measured from textured.obj) ─────
#
#   x ∈ [-0.078, +0.082]   (≈ Ø 0.16 outer)
#   y ∈ [-0.081, +0.080]
#   z ∈ [-0.028, +0.027]   (≈ 0.055 m tall — half-height ≈ 0.028)
#
# Walls are thin: outer radius ≈ 0.082, inner cavity radius ≈ 0.072.
_BOWL_OUTER_R     = 0.082
_BOWL_INNER_R     = 0.072
_BOWL_HALF_HEIGHT = 0.028          # along the bowl's local Z (cavity axis)

# After the R1 pitch (R_y(-π/2) applied to the upside-down bowl), the
# bowl's cavity opening points world +X; the rim plane sits at
# bowl_x + _BOWL_HALF_HEIGHT in world coordinates.
_BOWL_RIM_DX      = _BOWL_HALF_HEIGHT
_BOWL_APPROACH    =  -0.060

# ── Place zone (on the table, well inside R0's reach) ────────────────
GOAL_XY = np.array([0.30, -0.20])
PLACE_Z = TABLE_Z + 0.04            # bowl rim/base just above table

# ── R1 motion ────────────────────────────────────────────────────────
PRE_GRASP_DZ      = 0.10            # hover above bowl base before contact
SUCTION_SINK      = 0.005           # press snout 5 mm past the bowl top
LIFT_Z            = TABLE_Z + 0.25  # carry height after weld
# Bring the bowl into R0's natural reach band so the parallel-jaw can
# actually achieve the "leftfinger inside cavity / rightfinger outside
# outer wall" pose.  With the bowl at y=0 (workspace centreline) the IK
# saturated ≈5 mm short on y and the rightfinger ended up *inside* the
# wall material instead of outside it.
HANDOVER_XY       = np.array([0.05, -0.10])

# ── R0 side-grasp into the bowl hollow ───────────────────────────────
# Wrist offset in -Y from bowl centre.  R0's base is mounted at y=-0.6
# facing +X; reaching the +Y side of the bowl (bowl_y ≈ 0) requires
# Panda to extend across the workspace centreline and saturates the
# IK — empirically the EEF lands ~0.15 m short on y when commanded to
# the +Y side.  The -Y side is the natural reach band for R0, and the
# resulting grasp pose still has one finger inside the cavity and one
# outside the outer wall (just mirrored across the bowl axis).
R0_WRIST_DY       = -0.077
# Stand-off (wrist x - rim_x) for the approach waypoint.
R0_STAGE_DX       = 0.20
# Final grasp wrist x relative to rim_x.  Panda finger pads sit
# ≈ 0.093 m past the wrist along the snout direction; we drive the
# wrist all the way to the rim plane so the pad sinks 90+ mm past
# the rim, deep enough that the bowl's inner-wall surfaces fully
# capture the leftfinger body — only *after* the wrist has fully
# entered the cavity does R0 close the jaw and the suction release.
# IK + bowl back-wall collision keep the pad from over-driving past
# the bowl floor.
R0_GRASP_DX       = -0.080
# Safe parking pose for R0 while R1 is performing the pick + pitch.
R0_SAFE_POS       = np.array([0.30, -0.30, TABLE_Z + 0.45])

# ── Tick budgets — IK-absolute moves at ~5 mm/step in position. ──────
SETTLE_TICKS        = 30
APPROACH_TICKS      = 1500
DESCEND_TICKS       = 800
WELD_TICKS          = 100
LIFT_TICKS          = 800
PITCH_TICKS         = 1500
R0_FAR_TICKS        = 800
R0_NEAR_TICKS       = 2500
GRIP_TICKS          = 200
RELEASE_TICKS       = 200
R1_RETREAT_TICKS    = 600
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


def _quat_rotate_vec(q, v):
    """Rotate 3-vector ``v`` by xyzw quaternion ``q``."""
    x, y, z, w = q
    vx, vy, vz = v
    # Standard rotation: q * (0, v) * q^{-1}
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ], dtype=float)


def _quat_to_axis_angle(q):
    x, y, z, w = q
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    sin_half = float(np.sqrt(x * x + y * y + z * z))
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, w)
    return np.array([x, y, z]) / sin_half * angle


def _ori_err_world(eef_quat, target_quat):
    return _quat_to_axis_angle(quat_mul_xyzw(target_quat, _quat_conj(eef_quat)))


def _arm_action(eef_pos, eef_quat, target_pos, target_quat, gripper):
    a = np.zeros(7, dtype=float)
    a[0:3] = np.clip((target_pos - eef_pos) * POS_GAIN_IK, -1.0, 1.0)
    a[3:6] = np.clip(_ori_err_world(eef_quat, target_quat) * ORI_GAIN_IK,
                     -1.0, 1.0)
    a[6] = float(gripper)
    return a


def _arm_action_pos_only(eef_pos, target_pos, gripper):
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

class BowlTablePolicy(PolicyInterface):
    """Scripted bowl-table policy using IK_POSE absolute control."""

    def __init__(self, table_z: float = TABLE_Z,
                 goal_xy: np.ndarray = GOAL_XY):
        self._tz = float(table_z)
        self._goal_xy = np.asarray(goal_xy, dtype=float)
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.tick = 0
        self._done = False

        self._bowl_pos: np.ndarray | None = None
        self._bowl_quat: np.ndarray | None = None

        self._home_r0_quat: np.ndarray | None = None
        self._home_r1_quat: np.ndarray | None = None

        self._r1_lift_xyz: np.ndarray | None = None
        self._r1_handover_xyz: np.ndarray | None = None
        self._r0_grasp_xyz: np.ndarray | None = None
        self._r0_after_grasp_xyz: np.ndarray | None = None

        # Pose latches for the R0-only rotation (r0_rotate_up runs after
        # R1 has released and retreated).
        self._rot_pivot: np.ndarray | None = None
        self._r0_pre_rot_pos: np.ndarray | None = None
        self._r0_pre_rot_quat: np.ndarray | None = None
        self._r1_pre_rot_pos: np.ndarray | None = None
        self._r1_pre_rot_quat: np.ndarray | None = None
        self._r0_post_rot_pos: np.ndarray | None = None
        self._r0_post_rot_quat: np.ndarray | None = None
        self._r1_post_rot_pos: np.ndarray | None = None
        self._r1_post_rot_quat: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self._done

    def _is_welded(self, env, robot_id: int = 1) -> bool:
        if env is None:
            return False
        sw = getattr(env, "_suction_weld", None)
        if sw is None:
            return False
        h = sw._handles.get((robot_id, "right"))
        return h is not None and h.welded_body_id is not None

    def _advance(self, nxt: str):
        b = self._bowl_pos
        bs = (f"bowl=({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})"
              if b is not None else "bowl=?")
        print(f"  [{self.tick:4d}] bowl-table: {self.phase} → {nxt}  {bs}",
              flush=True)
        self.phase = nxt
        self.tick = 0

    # ── Target orientations ─────────────────────────────────────────
    def _target_orientations(self):
        """Return the key target quaternions used by the policy.

        ``q_R1_init``     — suction snout straight down (world -Z),
                            latched from R1's converged home quat.
        ``q_R1_pitched``  — suction snout horizontal in world +X (the
                            bowl's cavity also points world +X after
                            the weld, exposing the rim opening to R0).
        ``q_R0_init``     — R0 top-down (world -Z snout).
        ``q_R0_grasp``    — R0 oriented to face the bowl hollow:
                              snout       → world -X (looks straight
                                            into the +X-facing rim
                                            opening).
                              spread axis → world Y (Panda's natural
                                            spread axis stays world Y
                                            under a pure world R_y
                                            rotation).
        ``q_R0_upright``  — R0 rotated back through R_y(-π/2) so the
                            held bowl's cavity ends up world +Z (bowl
                            upright in R0's grasp).
        """
        q_world_y_pos90 = _axisangle_to_quat([0.0, 1.0, 0.0],  np.pi / 2.0)
        q_world_y_neg90 = _axisangle_to_quat([0.0, 1.0, 0.0], -np.pi / 2.0)

        q_R1_init = (self._home_r1_quat
                     if self._home_r1_quat is not None else TOP_DOWN_QUAT)
        q_R1_pitched = quat_mul_xyzw(q_world_y_neg90, q_R1_init)

        q_R0_init = (self._home_r0_quat
                     if self._home_r0_quat is not None else TOP_DOWN_QUAT)
        q_R0_grasp = quat_mul_xyzw(q_world_y_pos90, q_R0_init)
        q_R0_upright = quat_mul_xyzw(q_world_y_neg90, q_R0_grasp)
        return (q_R1_init, q_R1_pitched, q_R0_init,
                q_R0_grasp, q_R0_upright)

    # ── Main act ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        r0_pos, r0_quat = _read_eef(obs, 0)
        r1_pos, r1_quat = _read_eef(obs, 1)

        if env is not None:
            self._bowl_pos = read_obj_pos(env)
            self._bowl_quat = _read_bowl_quat(env)
        if self._bowl_pos is None:
            self._bowl_pos = np.array([0.10, 0.25, self._tz + _BOWL_HALF_HEIGHT])
            self._bowl_quat = np.array([1.0, 0.0, 0.0, 0.0])

        bowl = self._bowl_pos
        tz = self._tz

        (q_R1_init, q_R1_pitched, q_R0_init,
         q_R0_grasp, q_R0_upright) = self._target_orientations()

        # ── R1 (suction) live waypoints ─────────────────────────────
        # Bowl is upside-down: its flat base is now the topmost surface
        # and sits at bowl_z + half_height.
        bowl_top_z = bowl[2] + _BOWL_HALF_HEIGHT
        r1_above   = np.array([bowl[0], bowl[1], bowl_top_z + PRE_GRASP_DZ])
        r1_contact = np.array([bowl[0], bowl[1], bowl_top_z - SUCTION_SINK])

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

        # ── R1 hovers above the bowl base; R0 parks clear ───────────
        elif p == "r1_pre_grasp":
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, r1_above, q_R1_init, +1.0)
            if self._is_welded(env, 1):
                self._latch_lift_target(r1_pos)
                self._advance("r1_lift")
            elif _at(r1_pos, r1_above, tol=0.02) or self.tick >= APPROACH_TICKS:
                self._advance("r1_descend")

        # ── R1 descends to the flat base; suction welds on contact ──
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

        # ── R1 lifts straight up keeping the snout pointing world -Z ─
        elif p == "r1_lift":
            tgt = self._r1_lift_xyz
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, tgt, q_R1_init, +1.0)
            if r1_pos[2] >= tgt[2] - 0.04 or self.tick >= LIFT_TICKS:
                self._r1_handover_xyz = np.array([HANDOVER_XY[0],
                                                  HANDOVER_XY[1],
                                                  LIFT_Z])
                self._advance("r1_pitch")

        # ── R1 traverses to mid-table and rotates 90° about world -Y ─
        # so the welded bowl's cavity ends up pointing world +X. ────
        elif p == "r1_pitch":
            tgt_pos = self._r1_handover_xyz
            a0 = _arm_action(r0_pos, r0_quat, R0_SAFE_POS,
                             q_R0_init, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, tgt_pos, q_R1_pitched, +1.0)
            ori_done = _at_ori(r1_quat, q_R1_pitched, tol=0.12)
            pos_done = _at(r1_pos, tgt_pos, tol=0.04)
            if (ori_done and pos_done) or self.tick >= PITCH_TICKS:
                self._advance("r0_approach_far")

        # ── R0 stages on the +X side of the bowl, snout pointing -X.
        # Wrist offset in -Y by R0_WRIST_DY so one closed finger ends
        # up just inside the -Y cavity wall and the other just outside
        # the outer -Y wall. ───────────────────────────────────────
        elif p == "r0_approach_far":
            rim_x = bowl[0] + _BOWL_RIM_DX
            r0_far = np.array([rim_x + R0_STAGE_DX,
                               bowl[1] + R0_WRIST_DY,
                               bowl[2]])
            a0 = _arm_action(r0_pos, r0_quat, r0_far, q_R0_grasp, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, self._r1_handover_xyz,
                             q_R1_pitched, +1.0)
            ori_ok = _at_ori(r0_quat, q_R0_grasp, tol=0.20)
            pos_ok = _at(r0_pos, r0_far, tol=0.06)
            if (ori_ok and pos_ok) or self.tick >= R0_FAR_TICKS:
                self._advance("r0_approach_near")

        # ── R0 closes in *parallel* (snout direction held fixed)
        # along world -X.  Closed-loop check: only transition to the
        # grip phase once the *predicted closed-pad positions* relative
        # to the bowl actually satisfy "leftfinger pad inside the
        # cavity (radial < inner_r) AND rightfinger pad outside the
        # outer wall (radial > outer_r)".  If the IK undershoots the
        # commanded pose we keep iterating instead of closing the jaw
        # on empty space. ────────────────────────────────────────
        elif p == "r0_approach_near":
            rim_x = bowl[0] + _BOWL_APPROACH
            r0_grasp = np.array([rim_x + R0_GRASP_DX,
                                 bowl[1] + R0_WRIST_DY,
                                 bowl[2]])
            a0 = _arm_action(r0_pos, r0_quat, r0_grasp, q_R0_grasp, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, self._r1_handover_xyz,
                             q_R1_pitched, +1.0)

            # Predicted closed-jaw pad positions in world Y (R0's
            # spread axis stays world Y under the q_R0_grasp rotation).
            # Approaching from -Y, the "inner" finger (closer to bowl
            # centre) is at eef_y + 0.0085 and the "outer" finger is
            # at eef_y - 0.0085 when the jaw is closed.
            inner_pad_y = r0_pos[1] + 0.0085
            outer_pad_y = r0_pos[1] - 0.0085
            inner_r_dist = abs(inner_pad_y - bowl[1])
            outer_r_dist = abs(outer_pad_y - bowl[1])
            grasp_pose_ok = (
                inner_r_dist < _BOWL_INNER_R - 0.001  # inside cavity
                and outer_r_dist > _BOWL_OUTER_R + 0.001  # outside outer
                and r0_pos[0] < rim_x + 0.080  # wrist past rim plane
                and abs(r0_pos[2] - bowl[2]) < 0.04
            )
            if grasp_pose_ok or self.tick >= R0_NEAR_TICKS:
                self._r0_grasp_xyz = r0_pos.copy()
                self._advance("r0_grip")

        # ── R0 closes the jaw — leftfinger sweeps inside the cavity,
        # rightfinger sweeps inward toward the outer wall.  Suction
        # stays welded so the bowl is fully supported while the
        # parallel-jaw closes around it. ───────────────────────────
        elif p == "r0_grip":
            a0 = _arm_action(r0_pos, r0_quat, self._r0_grasp_xyz,
                             q_R0_grasp, +1.0)
            a1 = _arm_action(r1_pos, r1_quat, self._r1_handover_xyz,
                             q_R1_pitched, +1.0)
            if self.tick >= GRIP_TICKS:
                self._r0_after_grasp_xyz = r0_pos.copy()
                # Latch R0's grasp-time pose so it can hold steady
                # while R1 deactivates and lifts away.
                self._r0_pre_rot_pos = r0_pos.copy()
                self._r0_pre_rot_quat = r0_quat.copy()
                # Pre-compute the upright orientation R0 will rotate
                # to once R1 is out of the way.  Bowl is held by R0's
                # deep jaw grip (leftfinger fully captured by the bowl
                # cavity) so a single-arm rotation about R_y(-π/2)
                # safely brings the cavity from horizontal +X to
                # upright +Z.
                dq = _axisangle_to_quat([0.0, 1.0, 0.0], -np.pi / 2.0)
                self._r0_upright_quat = quat_mul_xyzw(
                    dq, self._r0_pre_rot_quat
                )
                self._r1_pre_release_pos = r1_pos.copy()
                self._r1_pre_release_quat = r1_quat.copy()
                self._advance("r1_release")

        # ── R1 deactivates the suction immediately after the jaw
        # closes — the bowl is now held by R0 alone.  R0 keeps its
        # grasp pose unchanged so the bowl does not jostle. ────────
        elif p == "r1_release":
            a0 = _arm_action(r0_pos, r0_quat,
                             self._r0_pre_rot_pos,
                             self._r0_pre_rot_quat,
                             +1.0)
            a1 = _arm_action(r1_pos, r1_quat,
                             self._r1_pre_release_pos,
                             self._r1_pre_release_quat,
                             -1.0)
            if self.tick >= RELEASE_TICKS:
                self._advance("r1_retreat")

        # ── R1 retreats by pulling +Y back toward its base and up
        # +Z, well clear of R0 and the bowl. ─────────────────────
        elif p == "r1_retreat":
            a0 = _arm_action(r0_pos, r0_quat,
                             self._r0_pre_rot_pos,
                             self._r0_pre_rot_quat,
                             +1.0)
            r1_clear = np.array([self._r1_pre_release_pos[0],
                                 self._r1_pre_release_pos[1] + 0.30,
                                 self._r1_pre_release_pos[2] + 0.20])
            a1 = _arm_action(r1_pos, r1_quat, r1_clear,
                             self._r1_pre_release_quat, -1.0)
            if _at(r1_pos, r1_clear, tol=0.12) or self.tick >= R1_RETREAT_TICKS:
                self._advance("r0_rotate_up")

        # ── R0 alone rotates the held bowl by R_y(-π/2): cavity
        # goes from world +X (horizontal) → world +Z (upright).
        # The deep finger insertion captures the bowl rim
        # geometrically, so the bowl follows R0's wrist through
        # the rotation. ───────────────────────────────────────
        elif p == "r0_rotate_up":
            a0 = _arm_action(r0_pos, r0_quat,
                             self._r0_pre_rot_pos,
                             self._r0_upright_quat,
                             +1.0)
            a1 = _hold(-1.0)
            r0_ori_ok = _at_ori(r0_quat, self._r0_upright_quat, tol=0.15)
            if r0_ori_ok or self.tick >= R0_ROTATE_TICKS:
                self._r0_after_rot_pos = r0_pos.copy()
                self._advance("r0_lower")

        # ── R0 lowers the upright bowl onto the table.  We track
        # the bowl's live world pose (sim) and adjust R0 by the
        # same delta so the bowl ends at PLACE_Z.  R0's
        # orientation is held at q_R0_upright to keep the bowl
        # vertical during the descent. ────────────────────────
        elif p == "r0_lower":
            target_bowl = np.array([bowl[0], bowl[1], PLACE_Z])
            r0_target = r0_pos + (target_bowl - bowl)
            a0 = _arm_action(r0_pos, r0_quat, r0_target,
                             self._r0_upright_quat, +1.0)
            a1 = _hold(-1.0)
            if bowl[2] <= PLACE_Z + 0.02 or self.tick >= LOWER_TICKS:
                self._advance("r0_release")

        # ── R0 opens the jaw — bowl rests on the table. ─────────
        elif p == "r0_release":
            a0 = _arm_action_pos_only(r0_pos, r0_pos, -1.0)
            a1 = _hold(-1.0)
            if self.tick >= R0_RELEASE_TICKS:
                self._advance("r0_retreat")

        # ── R0 lifts away from the placed bowl ──────────────────
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

    # ── Phase-helpers ───────────────────────────────────────────────
    def _latch_lift_target(self, r1_pos: np.ndarray) -> None:
        """Freeze the post-weld lift target above the current snout pose."""
        if self._r1_lift_xyz is None:
            self._r1_lift_xyz = np.array([r1_pos[0], r1_pos[1], LIFT_Z])
