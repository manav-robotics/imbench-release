"""Scripted single-arm policy for ``ib-p1-cracker-box-suction-v1``.

Task
----
Suction-only variant of the cracker-box slot-insertion task. Only the
suction arm (R1) can move; the parallel-jaw arm (R0) is frozen.

Strategy
--------
The task config lays the cracker box flat on the table (R_y(-90°)) so
its largest face — 0.166 m × 0.216 m — points straight up. The suction
snout descends top-down onto the centre of that long face, welds to it,
lifts the box, traverses to a position above the slot, and drops the
box through the open top of the slot. Both the box and the slot rest
flat on the table top (z = 0.94 and 0.955 respectively) and their
footprints do not overlap. Box pose and slot pose are queried from the
sim every tick before grasp so the policy adapts to placement jitter.

The IK_POSE controller in the opposed bimanual configuration is
unstable when asked to push the EEF down to the slot floor at the
slot's lateral pose — the IK rotates the EEF as it changes joint
configurations, swinging the welded payload off-axis. Rather than
fight that with tighter orientation control (which the absolute-mode
IK_POSE handles awkwardly), the policy hovers the snout above the
slot opening with the box centre closed-loop driven onto the slot's
XY centre, then deactivates the suction. The box drops freely through
the open top of the slot and settles on the channel floor — score is
measured against the final settled centroid, which lands inside the
0.04 m success tolerance of the slot interior centre.

Phase machine::

    settle → align → pre_grasp → descend → weld
        → lift → carry → slot_align → lower → release
        → retract → done

The ``align`` phase reads the live box and slot world quaternions
from the sim, projects each long-axis onto the world XY plane, and
rotates the suction wrist about world +Z so the welded box's long
axis matches the slot's channel direction before the descent. After
the suction welds and the gripper traverses to a hover above the
slot (``carry``), ``slot_align`` re-reads the live orientations and
yaws R1 to correct any residual drift, then ``lower`` closed-loop
slides the BOX centre onto the slot interior centre as the snout
descends.

Controller: IK_POSE with input_type=absolute. Per-axis position deltas
in [-1, 1] are pre-multiplied by ``ik_scale.pos_sensitivity`` in the
HetBi base ``step()`` wrapper before reaching the IK controller. Only
the ``align`` and ``slot_align`` phases drive orientation deltas (a yaw
about world +Z); every other phase commands position only, so the IK
holds the latched orientation (suction snout pointing world -Z).
"""
from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface
from data_synthetic.hetbi_scripted._utils import (
    TABLE_Z, read_obj_pos, read_fixture_pos, read_eef,
    quat_mul_xyzw,
)


# ── Cracker-box geometry ─────────────────────────────────────────────
# Mesh half-extents (from STL bounds): X=0.0382, Y=0.083, Z=0.108.
# After R_y(-90°) the local +X axis (thinnest, 0.075 m) → world +Z, so the
# box top face sits at box_centre_z + 0.0382 m.
_BOX_TOP_DZ    = 0.0382
_SNOUT_SINK    = 0.012     # press snout 12 mm below the top face for solid contact

_PRE_GRASP_DZ  = 0.10
_LIFT_DZ       = 0.18      # lift this far above the welded box top
_DROP_HOVER_DZ = 0.14      # carry-hover above the slot interior centre
_DROP_RELEASE_DZ = 0.03    # snout height above slot_z when releasing: the box
                           # hangs inside the channel ~2.5 cm above its floor,
                           # so it lands flat instead of tipping onto a wall
_LOWER_XY_TOL  = 0.010     # box-centre xy error below which the box descends into the slot
_RELEASE_XY_TOL = 0.006    # box-centre xy error required before releasing

# Slot interior centre offset from fixture origin (matches hetbi_env bounds).
_SLOT_CENTRE_DY = -0.10

# Phase tick budgets (control_freq = 100 Hz). At ik_pos_sensitivity=0.005
# m × IK user_sensitivity 0.3, the controller's target accumulator advances
# at most 1.5 mm/step, so generous budgets are required for the EEF to
# actually catch the latched waypoint.
_SETTLE_TICKS    = 30
_ALIGN_TICKS     = 1200
_APPROACH_TICKS  = 1500
_DESCEND_TICKS   = 1000
_WELD_TICKS      = 100
_LIFT_TICKS      = 800
_CARRY_TICKS     = 1500
_LOWER_TICKS     = 1500
_RELEASE_TICKS   = 400      # ~4 s of free-fall + settle on the slot floor
_RETRACT_TICKS   = 400

ORI_GAIN_IK = 8.0


def _pos_action(eef_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-axis position delta in [-1, 1].

    Proportional within ±5 cm, saturated beyond that. The gentle
    saturation keeps the IK_POSE reference target close to the EEF
    even when the EEF physically can't reach the commanded waypoint.
    """
    return np.clip((target - eef_pos) * 20.0, -1.0, 1.0)


def _arm_action(eef_pos: np.ndarray, target_pos: np.ndarray,
                gripper: float) -> np.ndarray:
    """7-DOF IK_POSE arm action: [pos delta (3), ori delta (3) = 0, gripper]."""
    a = np.zeros(7, dtype=float)
    a[0:3] = _pos_action(eef_pos, target_pos)
    # Orientation deltas left at zero — the IK keeps the controller's
    # initial (home-pose) orientation target, which already points the
    # snout at world -Z.
    a[6] = gripper
    return a


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
    """World-frame axis-angle error consumed by IK action[3:6]."""
    return _quat_to_axis_angle(quat_mul_xyzw(target_quat, _quat_conj(eef_quat)))


def _arm_action_full(eef_pos, eef_quat, target_pos, target_quat,
                     gripper: float) -> np.ndarray:
    """7-DOF IK_POSE arm action with explicit orientation target."""
    a = np.zeros(7, dtype=float)
    a[0:3] = _pos_action(eef_pos, target_pos)
    if target_quat is not None and eef_quat is not None:
        a[3:6] = np.clip(_ori_err_world(eef_quat, target_quat) * ORI_GAIN_IK,
                         -1.0, 1.0)
    a[6] = gripper
    return a


def _quat_apply(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector ``v`` by xyzw quaternion ``q``."""
    qx, qy, qz, qw = q_xyzw
    vx, vy, vz = v
    tx = qw * vx + qy * vz - qz * vy
    ty = qw * vy - qx * vz + qz * vx
    tz = qw * vz + qx * vy - qy * vx
    tw = -(qx * vx + qy * vy + qz * vz)
    rx = tw * (-qx) + tx * qw + ty * (-qz) - tz * (-qy)
    ry = tw * (-qy) - tx * (-qz) + ty * qw + tz * (-qx)
    rz = tw * (-qz) + tx * (-qy) - ty * (-qx) + tz * qw
    return np.array([rx, ry, rz], dtype=float)


def _read_body_quat_xyzw(env, body_id: int) -> np.ndarray:
    """World orientation (xyzw) of an arbitrary MuJoCo body."""
    sim = env._env.sim
    q_wxyz = sim.data.body_xquat[body_id]
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def _hold(gripper: float) -> np.ndarray:
    a = np.zeros(7, dtype=float)
    a[6] = gripper
    return a


def _at(eef_pos: np.ndarray, target: np.ndarray, tol: float = 0.020) -> bool:
    return float(np.linalg.norm(eef_pos - target)) < tol


def _at_ori(eef_quat, target_quat, tol: float = 0.10) -> bool:
    return float(np.linalg.norm(_ori_err_world(eef_quat, target_quat))) < tol


class CrackerBoxSuctionPolicy(PolicyInterface):
    """Top-down suction pick-and-drop into the slot, IK_POSE controller."""

    def __init__(self, table_z: float = TABLE_Z):
        self._tz = table_z
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.tick = 0
        self._done = False
        # Live sim readings (refreshed each tick before grasp).
        self._box_pos: np.ndarray | None = None
        self._slot_pos: np.ndarray | None = None
        # Latched at the moment the suction weld engages — these targets
        # are then *not* re-derived from the welded payload position.
        self._lifted_xyz: np.ndarray | None = None
        self._carry_xyz: np.ndarray | None = None
        self._retract_xyz: np.ndarray | None = None
        # Snout-to-box 3D offset latched the moment the suction welds.
        self._snout_box_offset: np.ndarray | None = None
        # Latched at end of settle: R1's converged home quat.
        self._home_r1_quat: np.ndarray | None = None
        # Yaw-aligned target orientation for R1 — computed in `align`
        # by reading the box and slot world quaternions and projecting
        # their long axes onto the world XY plane. Held through every
        # subsequent phase so the box's long axis stays parallel to the
        # slot's channel direction throughout the pick-up and slot drop.
        self._r1_aligned_quat: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self._done

    # ── Helpers ──────────────────────────────────────────────────────

    def _is_welded(self, env) -> bool:
        if env is None:
            return False
        sw = getattr(env, "_suction_weld", None)
        if sw is None:
            return False
        h = sw._handles.get((1, "right"))
        return h is not None and h.welded_body_id is not None

    def _advance(self, nxt: str):
        print(f"  [{self.tick:4d}] cracker-box-suction: {self.phase} → {nxt}",
              flush=True)
        self.phase = nxt
        self.tick = 0

    def _slot_centre(self) -> np.ndarray:
        s = self._slot_pos
        return np.array([s[0], s[1] + _SLOT_CENTRE_DY, s[2]])

    def _compute_align_target(self, env, r1_quat: np.ndarray) -> None:
        """Read the live box and slot world quaternions and compute a
        yaw-rotated R1 target orientation so the box's long axis (mesh
        local +Z, 0.216 m) ends up parallel to the slot's channel
        direction (mesh local +X). The two long-axes are projected onto
        the world XY plane and the residual yaw between them is wrapped
        into a rotation about world +Z applied to R1's home quat."""
        if env is None:
            self._r1_aligned_quat = (self._home_r1_quat.copy()
                                     if self._home_r1_quat is not None
                                     else r1_quat.copy())
            return

        box_id = env._env._payload_body_id
        slot_id = env._env._fixture_body_id
        box_q = _read_body_quat_xyzw(env, box_id)
        slot_q = _read_body_quat_xyzw(env, slot_id)

        box_long_world = _quat_apply(box_q, np.array([0.0, 0.0, 1.0]))
        bx, by = float(box_long_world[0]), float(box_long_world[1])
        if abs(bx) < 1e-6 and abs(by) < 1e-6:
            self._r1_aligned_quat = r1_quat.copy()
            return

        slot_long_world = _quat_apply(slot_q, np.array([1.0, 0.0, 0.0]))
        sx, sy = float(slot_long_world[0]), float(slot_long_world[1])

        box_angle = np.arctan2(by, bx)
        slot_angle = np.arctan2(sy, sx)
        delta = slot_angle - box_angle
        # Long-axis is bidirectional — wrap into (-π/2, π/2].
        while delta > np.pi / 2.0:
            delta -= np.pi
        while delta <= -np.pi / 2.0:
            delta += np.pi

        yaw_q = _axisangle_to_quat([0.0, 0.0, 1.0], delta)
        base = (self._home_r1_quat
                if self._home_r1_quat is not None else r1_quat)
        self._r1_aligned_quat = quat_mul_xyzw(yaw_q, base)
        print(f"  [align] box_long_xy=({bx:.3f},{by:.3f})  "
              f"slot_long_xy=({sx:.3f},{sy:.3f})  "
              f"yaw_delta={np.degrees(delta):.2f}°", flush=True)

    def _latch_carry_targets(self, r1_pos: np.ndarray, env=None):
        slot_c = self._slot_centre()
        self._lifted_xyz = np.array(
            [r1_pos[0], r1_pos[1], r1_pos[2] + _LIFT_DZ]
        )
        # Latch the snout-to-box offset at weld so the slot_align /
        # lower phases can target the box position rather than the
        # snout position.
        offset = np.zeros(3)
        if env is not None:
            box = read_obj_pos(env)
            offset = r1_pos - box
        self._snout_box_offset = offset
        # Carry hover sits well above the slot opening for clearance.
        self._carry_xyz = np.array(
            [slot_c[0], slot_c[1], slot_c[2] + _DROP_HOVER_DZ]
        )
        self._retract_xyz = self._carry_xyz + np.array([0.0, 0.0, 0.10])

    # ── Main act ─────────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        _r0_pos, _r0_quat = read_eef(obs, 0)
        r1_pos, r1_quat = read_eef(obs, 1)

        # Refresh box / slot poses while the box is still loose. After
        # the weld engages, downstream waypoints are computed in absolute
        # world coords and never re-derived from the welded payload.
        if env is not None and self._lifted_xyz is None:
            self._box_pos  = read_obj_pos(env)
            self._slot_pos = read_fixture_pos(env)
        elif self._box_pos is None:
            self._box_pos  = np.array([0.30, 0.10, 0.94])
            self._slot_pos = np.array([0.10, 0.10, 0.955])

        box = self._box_pos

        top_face_z = box[2] + _BOX_TOP_DZ
        pre_grasp = np.array([box[0], box[1], top_face_z + _PRE_GRASP_DZ])
        contact   = np.array([box[0], box[1], top_face_z - _SNOUT_SINK])

        a0 = _hold(-1.0)        # R0 (frozen jaw) — env zeroes its slice.

        self.tick += 1
        p = self.phase

        if p == "settle":
            a1 = _hold(-1.0)
            self._home_r1_quat = r1_quat.copy()
            if self.tick >= _SETTLE_TICKS:
                self._advance("align")

        # ── Read box + slot orientations from the sim and yaw R1 so
        # the welded box will end up with its long axis parallel to
        # the slot's channel direction. Done BEFORE descending onto
        # the box so the snout welds in the aligned pose. Only this
        # phase and ``slot_align`` drive orientation; the other phases
        # command pos only (action[3:6] = 0) so the IK holds the
        # latched yaw. ──
        elif p == "align":
            if self._r1_aligned_quat is None:
                self._compute_align_target(env, r1_quat)
            target = self._r1_aligned_quat
            a1 = _arm_action_full(r1_pos, r1_quat, r1_pos, target, -1.0)
            if _at_ori(r1_quat, target, tol=0.10) \
                    or self.tick >= _ALIGN_TICKS:
                self._advance("pre_grasp")

        elif p == "pre_grasp":
            # Suction ON the whole way — only welds on actual contact.
            a1 = _arm_action(r1_pos, pre_grasp, +1.0)
            if self._is_welded(env):
                self._latch_carry_targets(r1_pos, env)
                self._advance("lift")
            elif (_at(r1_pos, pre_grasp, tol=0.008)
                    or self.tick >= _APPROACH_TICKS):
                self._advance("descend")

        elif p == "descend":
            a1 = _arm_action(r1_pos, contact, +1.0)
            if self._is_welded(env):
                self._latch_carry_targets(r1_pos, env)
                self._advance("lift")
            elif self.tick >= _DESCEND_TICKS:
                self._advance("weld")

        elif p == "weld":
            a1 = _arm_action(r1_pos, contact, +1.0)
            if self._is_welded(env):
                self._latch_carry_targets(r1_pos, env)
                self._advance("lift")
            elif self.tick >= _WELD_TICKS:
                self._latch_carry_targets(r1_pos, env)
                self._advance("lift")

        elif p == "lift":
            tgt = self._lifted_xyz
            a1 = _arm_action(r1_pos, tgt, +1.0)
            if r1_pos[2] >= tgt[2] - 0.03 or self.tick >= _LIFT_TICKS:
                self._advance("carry")

        # ── Carry the welded box across the table to a hover above
        # the slot interior centre. ``self._carry_xyz`` puts the
        # snout above the slot xy at a clearance height; the live
        # box position is closed-loop slid onto the slot xy in the
        # ``lower`` phase below. ──────────────────────────────────
        elif p == "carry":
            tgt = self._carry_xyz
            a1 = _arm_action(r1_pos, tgt, +1.0)
            if _at(r1_pos, tgt, tol=0.04) or self.tick >= _CARRY_TICKS:
                self._advance("slot_align")

        # ── Re-align: now that the box is hovering directly above the
        # slot centre, read the current box and slot world quats and
        # re-yaw R1 so the welded box's long axis is parallel to the
        # slot channel direction before the descent. The previous
        # ``align`` phase set the snout yaw before pickup; this phase
        # corrects any residual yaw drift introduced during lift /
        # carry while the box was hanging from the suction snout. ──
        elif p == "slot_align":
            self._compute_align_target(env, r1_quat)
            target = self._r1_aligned_quat
            a1 = _arm_action_full(r1_pos, r1_quat, r1_pos, target, +1.0)
            if _at_ori(r1_quat, target, tol=0.10) \
                    or self.tick >= _ALIGN_TICKS:
                self._advance("lower")

        elif p == "lower":
            # Drive the *box centre* (not the snout) onto the slot's
            # XY centre, while holding the snout at a fixed hover Z above
            # the slot floor. The IK_POSE controller rotates the EEF as
            # it changes joint configurations during the lift / carry,
            # which swings the welded box laterally — closing the loop on
            # the box itself cancels whatever offset the carry left.
            # Centre the box over the slot at hover height first, then
            # descend into the channel.
            slot_c  = self._slot_centre()
            box_now = read_obj_pos(env) if env is not None else r1_pos
            xy_err  = slot_c[:2] - box_now[:2]
            xy_dist = float(np.linalg.norm(xy_err))
            target_z = slot_c[2] + (
                _DROP_RELEASE_DZ if xy_dist < _LOWER_XY_TOL else _DROP_HOVER_DZ
            )
            snout_target = np.array([
                r1_pos[0] + xy_err[0],
                r1_pos[1] + xy_err[1],
                target_z,
            ])
            a1 = _arm_action(r1_pos, snout_target, +1.0)
            xy_ok = xy_dist < _RELEASE_XY_TOL
            z_ok  = abs(r1_pos[2] - (slot_c[2] + _DROP_RELEASE_DZ)) < 0.01
            if (xy_ok and z_ok) or self.tick >= _LOWER_TICKS:
                self._advance("release")

        elif p == "release":
            # Suction OFF — the box drops through the slot's open top and
            # settles on the channel floor. Hold pose so we don't bump it.
            a1 = _hold(-1.0)
            if self.tick >= _RELEASE_TICKS:
                self._advance("retract")

        elif p == "retract":
            tgt = self._retract_xyz
            a1 = _arm_action(r1_pos, tgt, -1.0)
            if _at(r1_pos, tgt, tol=0.05) or self.tick >= _RETRACT_TICKS:
                self._done = True
                self._advance("done")

        else:  # done
            a1 = _hold(-1.0)

        return np.concatenate([a0, a1]).astype(np.float64)
