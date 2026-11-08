"""Scripted bimanual policy for ``ib-p1-plate-shelf-hetbi-v1``.

Pipeline (IK_POSE absolute controller):

    1. R0 (suction) hovers above the plate, descends, suction welds.
    2. R0 lifts the welded plate to a safe carry height.
    3. R0 PITCHES the plate to vertical by rotating its wrist with the
       composite quaternion ``q_pitch = (0.5, -0.5, 0.5, 0.5)`` (a
       120° rotation about the (1,-1,1)/√3 world axis).  After this
       rotation R0's snout points world +Y and the plate's body axes
       map to world as:

           body X (0.218 m, longest) → world +Z   (plate stands tall)
           body Y (0.175 m)           → world -X   (plate width along X)
           body Z (0.030 m, thinnest) → world -Y   (plate thickness in Y)

       This is the orientation that fits the shelf's central slot:
       width 0.175 m matches the shelf's 0.175 m interior X span, and
       thickness 0.030 m fits the 0.130 m central-slot Y span.
    4. R0 carries the now-vertical plate to a handover spot between
       the two robots.
    5. R1 (parallel jaw) reaches in with a TOP-DOWN orientation
       (snout world -Z, jaw spread world Y) and converges right above
       the plate's top edge — a thin (0.030 m × 0.175 m) horizontal
       strip at world z = plate_z + 0.109.  Spread along Y straddles
       the 0.030 m edge thickness perfectly.
    6. R1 jaw closes on the top edge while R0's suction is still
       active (overlap window so the plate is fully supported).
    7. R0 deactivates the suction and retreats — R1 now solely holds
       the vertical plate by its top edge.
    8. R1 carries the vertical plate over the shelf's central slot.
    9. R1 lowers the plate until its bottom edge rests on the shelf
       base; the plate centre lands inside the success volume.
   10. R1 opens the jaw and retreats.

Controller: IK_POSE with input_type=absolute, scaled by
``ik_scale.pos_sensitivity`` (0.005 m / unit) and
``ik_scale.ori_sensitivity`` (0.01 rad / unit) at the HetBi step()
wrapper.  Saturated actions move the IK reference target ~5 mm /
0.01 rad per tick at 100 Hz.
"""
from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface
from data_synthetic.hetbi_scripted._utils import (
    TABLE_Z, read_obj_pos, read_fixture_pos, quat_mul_xyzw,
)


# ── Plate geometry (YCB plate body axes: x=0.218, y=0.175, z=0.030) ──
PLATE_HALF_X      = 0.109     # post-pitch: plate half-height (vertical)
PLATE_HALF_Y      = 0.0875    # post-pitch: half-width  (world -X)
PLATE_HALF_THICK  = 0.015     # post-pitch: half-thickness (world -Y)

# Panda parallel-jaw finger pads sit essentially AT the eef site —
# verified empirically by reading sim.data.geom_xpos of the
# ``gripper1_right_finger{1,2}_pad_collision`` geoms.
FINGER_OFFSET     = 0.005

# ── World rotation that pitches the plate vertical ──────────────────
# q_pitch = (qx, qy, qz, qw) = (0.5, -0.5, 0.5, 0.5), a 120° rotation
# about (1, -1, 1)/√3.  Rotation matrix:
#   R = [[0, -1,  0],
#        [0,  0, -1],
#        [1,  0,  0]]
# This sends:
#   body X → world +Z  (height)
#   body Y → world -X  (width)
#   body Z → world -Y  (thickness)
# After applying to R0's home pose (≈identity for the suction
# gripper), R0's snout points world +Y.  R0 base at y = -0.6 reaches
# naturally into +Y, so this orientation stays inside R0's reachable
# workspace.
PITCH_QUAT        = np.array([0.5, -0.5, 0.5, 0.5])

# ── Heights (world Z) ────────────────────────────────────────────────
PRE_PICK_DZ       = 0.10                  # R0 hover above plate
SUCTION_SINK      = 0.060                 # press snout 60 mm into plate
LIFT_Z            = TABLE_Z + 0.27        # 1.17 — pre-pitch safe carry
HANDOVER_PLATE_Z  = TABLE_Z + 0.20        # 1.10 — pitched plate centre
                                          # at handover.  With plate
                                          # half-height 0.109 m the
                                          # plate's top edge sits at
                                          # z=1.209 and bottom at 0.991.

# ── Handover plate-centre target ─────────────────────────────────────
# Plate spawns at (0.40, -0.30); shelf sits at (-0.30, 0.0) at the
# back of the table.  Handover at (0.15, -0.05) — R0 wrist after pitch
# is in -Y from plate (offset ~7 cm), so R0 wrist y ≈ -0.12, well
# inside R0's +Y reach band.  R1's top-down grip target (above pitched
# plate's top edge at z ≈ plate_z + 0.119) lands at (0.15, -0.05, 1.22),
# 0.74 m from R1 base — inside its reach radius.
HANDOVER_XY       = np.array([0.15, -0.05])

# ── Park positions ──────────────────────────────────────────────────
R0_PARK_XYZ       = np.array([0.30, -0.45, TABLE_Z + 0.30])
R1_PARK_XYZ       = np.array([0.30, +0.45, TABLE_Z + 0.30])

# ── R1 top-down grip offsets ────────────────────────────────────────
# Vertical plate's top edge centred at (plate_x, plate_y, plate_z +
# PLATE_HALF_X).  R1 with home orientation (snout -Z, spread world Y)
# above the edge: wrist sits FINGER_OFFSET above the edge so the
# fingertips drop right onto plate top.  Spread world Y straddles the
# 0.030 m edge thickness; closing pinches the edge between the pads.
R1_STAGE_DZ       = 0.08                  # standoff above plate top
R1_GRIP_DZ        = FINGER_OFFSET - 0.005 # ~0 mm above edge (pads pinch
                                          # the 0.030 m thick edge body)

# ── Tick budgets (control_freq = 100 Hz) ─────────────────────────────
SETTLE_TICKS         = 30
HOVER_TICKS          = 1500
DESCEND_TICKS        = 800
WELD_TICKS           = 100
LIFT_TICKS           = 800
PITCH_TICKS          = 2500           # 120° wrist rotation is large
TO_HANDOVER_TICKS    = 1500
R1_FAR_TICKS         = 2000
R1_NEAR_TICKS        = 1500
GRIP_TICKS           = 250
RELEASE_TICKS        = 150
R0_RETREAT_TICKS     = 800
CARRY_TICKS          = 2500
LOWER_TICKS          = 1500
R1_RELEASE_TICKS     = 200
R1_RETREAT_TICKS     = 600

POS_GAIN_IK = 20.0
ORI_GAIN_IK = 8.0


# ── Helpers ─────────────────────────────────────────────────────────

def _read_eef(obs: dict, robot_id: int):
    return (np.asarray(obs[f"robot{robot_id}_eef_pos"], dtype=float),
            np.asarray(obs[f"robot{robot_id}_eef_quat"], dtype=float))


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


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


def _quat_rotate_vec(q, v):
    """Rotate 3-vector ``v`` by xyzw quaternion ``q`` (active rotation)."""
    qx, qy, qz, qw = q
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return np.array([
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ], dtype=float)


def _arm_action(eef_pos, eef_quat, target_pos, target_quat, gripper):
    a = np.zeros(7, dtype=float)
    a[0:3] = np.clip((target_pos - eef_pos) * POS_GAIN_IK, -1.0, 1.0)
    a[3:6] = np.clip(_ori_err_world(eef_quat, target_quat) * ORI_GAIN_IK,
                     -1.0, 1.0)
    a[6] = float(gripper)
    return a


def _hold(gripper):
    a = np.zeros(7, dtype=float)
    a[6] = float(gripper)
    return a


def _at(eef_pos, target, tol=0.025):
    return float(np.linalg.norm(eef_pos - target)) < tol


def _at_ori(eef_quat, target_quat, tol=0.15):
    return float(np.linalg.norm(_ori_err_world(eef_quat, target_quat))) < tol


# ── Policy ───────────────────────────────────────────────────────────

class PlateShelfPolicy(PolicyInterface):
    """Pitch-and-handover policy for the plate-shelf hetbi task.

    R0 (suction) picks the plate flat, pitches it vertical via a
    120° wrist rotation, and carries it to the handover spot.  R1
    (parallel jaw) reaches in top-down, grips the now-vertical plate's
    top edge, and lowers it into the shelf's central slot.
    """

    def __init__(self, table_z: float = TABLE_Z):
        self._tz = float(table_z)
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.tick = 0
        self._done = False

        self._plate_pos: np.ndarray | None = None
        self._shelf_pos: np.ndarray | None = None

        self._home_r0_quat: np.ndarray | None = None
        self._home_r1_quat: np.ndarray | None = None

        # Pitched R0 orientation (set in settle once home quat is known).
        self._r0_pitched_quat: np.ndarray | None = None

        # Latched offsets:
        # _r0_plate_offset       — plate world - r0_pos at weld time
        #                          (R0 still at home orientation).
        # _r0_plate_offset_pitched — that same offset after applying the
        #                          pitch rotation; used post-pitch to
        #                          compute where R0 wrist should be to
        #                          place the plate at a desired world
        #                          position.
        # _r1_plate_offset       — plate world - r1_pos at the moment R1
        #                          closes the jaw.
        self._r0_plate_offset: np.ndarray | None = None
        self._r0_plate_offset_pitched: np.ndarray | None = None
        self._r1_plate_offset: np.ndarray | None = None

        # Target R1 wrist pose latched at the end of approach_near, so
        # the grip/carry phases keep commanding a stable reference even
        # when plate contact tugs the wrist.
        self._r1_grasp_eef_xyz: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self._done

    # ── Internals ───────────────────────────────────────────────────

    def _is_welded(self, env, robot_id: int = 0) -> bool:
        if env is None:
            return False
        sw = getattr(env, "_suction_weld", None)
        if sw is None:
            return False
        h = sw._handles.get((robot_id, "right"))
        return h is not None and h.welded_body_id is not None

    def _plate_in_shelf(self, env) -> bool:
        if env is None or self._plate_pos is None:
            return False
        # Use the same complete-geometry insertion gate as the task. A
        # centroid inside the shelf is not sufficient to release the plate.
        raw = env._get_sim_observations(force_update=True)
        return env._inserted_in_bounds_progress(raw) >= 1.0

    def _advance(self, nxt: str):
        p = self._plate_pos
        s = self._shelf_pos
        ps = (f"plate=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
              if p is not None else "plate=?")
        ss = (f"shelf=({s[0]:+.3f},{s[1]:+.3f},{s[2]:+.3f})"
              if s is not None else "shelf=?")
        print(f"  [{self.tick:4d}] plate-shelf: {self.phase} → {nxt}  {ps} {ss}",
              flush=True)
        self.phase = nxt
        self.tick = 0

    # ── Main act ────────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        r0_pos, r0_quat = _read_eef(obs, 0)
        r1_pos, r1_quat = _read_eef(obs, 1)

        if env is not None:
            self._plate_pos = read_obj_pos(env)
            self._shelf_pos = read_fixture_pos(env)
        if self._plate_pos is None:
            self._plate_pos = np.array([0.40, -0.30, self._tz + 0.016])
        if self._shelf_pos is None:
            self._shelf_pos = np.array([-0.20, -0.20, self._tz + 0.001])

        plate = self._plate_pos
        shelf = self._shelf_pos

        q_r0_home = (self._home_r0_quat if self._home_r0_quat is not None
                     else r0_quat)
        q_r1_home = (self._home_r1_quat if self._home_r1_quat is not None
                     else r1_quat)
        q_r0_pitched = (self._r0_pitched_quat
                        if self._r0_pitched_quat is not None
                        else quat_mul_xyzw(PITCH_QUAT, q_r0_home))

        handover_plate_xyz = np.array([HANDOVER_XY[0], HANDOVER_XY[1],
                                       HANDOVER_PLATE_Z])

        self.tick += 1
        p = self.phase

        # ── 0. settle ───────────────────────────────────────────────
        if p == "settle":
            a0 = _hold(-1.0)
            a1 = _hold(-1.0)
            self._home_r0_quat = r0_quat.copy()
            self._home_r1_quat = r1_quat.copy()
            self._r0_pitched_quat = quat_mul_xyzw(PITCH_QUAT, self._home_r0_quat)
            if self.tick >= SETTLE_TICKS:
                self._advance("r0_pre_grasp")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 1. R0 hovers above plate ────────────────────────────────
        if p == "r0_pre_grasp":
            target = np.array([plate[0], plate[1],
                               plate[2] + PLATE_HALF_THICK + PRE_PICK_DZ])
            a0 = _arm_action(r0_pos, r0_quat, target, q_r0_home, +1.0)
            a1 = _arm_action(r1_pos, r1_quat, R1_PARK_XYZ, q_r1_home, -1.0)
            if self._is_welded(env, 0):
                self._r0_plate_offset = self._plate_pos - r0_pos
                self._advance("r0_lift")
            elif _at(r0_pos, target, tol=0.025) or self.tick >= HOVER_TICKS:
                self._advance("r0_descend")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 2. R0 descends, suction welds on contact ────────────────
        if p == "r0_descend":
            target = np.array([plate[0], plate[1],
                               plate[2] + PLATE_HALF_THICK - SUCTION_SINK])
            a0 = _arm_action(r0_pos, r0_quat, target, q_r0_home, +1.0)
            a1 = _arm_action(r1_pos, r1_quat, R1_PARK_XYZ, q_r1_home, -1.0)
            if self._is_welded(env, 0):
                self._r0_plate_offset = self._plate_pos - r0_pos
                self._advance("r0_lift")
            elif self.tick >= DESCEND_TICKS:
                self._advance("r0_weld")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 2b. R0 holds press until weld engages ───────────────────
        if p == "r0_weld":
            target = np.array([plate[0], plate[1],
                               plate[2] + PLATE_HALF_THICK - SUCTION_SINK])
            a0 = _arm_action(r0_pos, r0_quat, target, q_r0_home, +1.0)
            a1 = _arm_action(r1_pos, r1_quat, R1_PARK_XYZ, q_r1_home, -1.0)
            if self._is_welded(env, 0) or self.tick >= WELD_TICKS:
                self._r0_plate_offset = self._plate_pos - r0_pos
                self._advance("r0_lift")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 3. R0 lifts the welded plate flat ───────────────────────
        if p == "r0_lift":
            offset = (self._r0_plate_offset
                      if self._r0_plate_offset is not None else np.zeros(3))
            target_plate = np.array([plate[0], plate[1], LIFT_Z])
            target_eef = target_plate - offset
            a0 = _arm_action(r0_pos, r0_quat, target_eef, q_r0_home, +1.0)
            a1 = _arm_action(r1_pos, r1_quat, R1_PARK_XYZ, q_r1_home, -1.0)
            if (abs(plate[2] - LIFT_Z) < 0.04
                    or self.tick >= LIFT_TICKS):
                # Precompute the post-pitch offset (R_pitch applied to
                # the world-frame latched offset).  Used for placing
                # the pitched plate.
                self._r0_plate_offset_pitched = _quat_rotate_vec(
                    PITCH_QUAT, self._r0_plate_offset)
                self._advance("r0_pitch")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 4. R0 pitches the plate to vertical while moving to the
        # handover spot.  Target plate position = handover; commanded
        # R0 wrist orientation = q_r0_pitched. ────────────────────
        if p == "r0_pitch":
            offset_p = (self._r0_plate_offset_pitched
                        if self._r0_plate_offset_pitched is not None
                        else np.zeros(3))
            target_eef = handover_plate_xyz - offset_p
            a0 = _arm_action(r0_pos, r0_quat, target_eef,
                             q_r0_pitched, +1.0)
            a1 = _arm_action(r1_pos, r1_quat, R1_PARK_XYZ, q_r1_home, -1.0)
            # 120° wrist rotation is the rate-limiter and the IK
            # leaves a residual ~10-15° tilt that doesn't matter for
            # the handover (the plate's top edge is still horizontal
            # enough for R1's top-down grip to land).  Exit as soon as
            # the plate has converged on the handover xy/z.
            plate_ok = _at(self._plate_pos, handover_plate_xyz, tol=0.05)
            if plate_ok or self.tick >= PITCH_TICKS:
                self._advance("r1_approach_far")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 5. R1 stages above the plate top edge with top-down
        # orientation (home pose).  Plate top edge at world z =
        # plate_z + PLATE_HALF_X. ─────────────────────────────────
        if p == "r1_approach_far":
            offset_p = (self._r0_plate_offset_pitched
                        if self._r0_plate_offset_pitched is not None
                        else np.zeros(3))
            target_r0_eef = handover_plate_xyz - offset_p
            a0 = _arm_action(r0_pos, r0_quat, target_r0_eef,
                             q_r0_pitched, +1.0)
            edge_z = self._plate_pos[2] + PLATE_HALF_X
            r1_stage = np.array([self._plate_pos[0],
                                 self._plate_pos[1],
                                 edge_z + R1_STAGE_DZ])
            a1 = _arm_action(r1_pos, r1_quat, r1_stage, q_r1_home, -1.0)
            if (_at(r1_pos, r1_stage, tol=0.08)
                    or self.tick >= R1_FAR_TICKS):
                self._advance("r1_approach_near")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 6. R1 lowers above the plate top edge, fingers open,
        # ready to close. ─────────────────────────────────────────
        if p == "r1_approach_near":
            offset_p = (self._r0_plate_offset_pitched
                        if self._r0_plate_offset_pitched is not None
                        else np.zeros(3))
            target_r0_eef = handover_plate_xyz - offset_p
            a0 = _arm_action(r0_pos, r0_quat, target_r0_eef,
                             q_r0_pitched, +1.0)
            edge_z = self._plate_pos[2] + PLATE_HALF_X
            r1_grip = np.array([self._plate_pos[0],
                                self._plate_pos[1],
                                edge_z + R1_GRIP_DZ])
            a1 = _arm_action(r1_pos, r1_quat, r1_grip, q_r1_home, -1.0)
            if (_at(r1_pos, r1_grip, tol=0.045)
                    or self.tick >= R1_NEAR_TICKS):
                self._r1_grasp_eef_xyz = r1_grip.copy()
                self._advance("r1_grip")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 7. R1 closes the jaw on the plate top edge.  R0 still
        # suctioned so the plate is fully supported. ─────────────
        if p == "r1_grip":
            offset_p = (self._r0_plate_offset_pitched
                        if self._r0_plate_offset_pitched is not None
                        else np.zeros(3))
            target_r0_eef = handover_plate_xyz - offset_p
            a0 = _arm_action(r0_pos, r0_quat, target_r0_eef,
                             q_r0_pitched, +1.0)
            a1 = _arm_action(r1_pos, r1_quat,
                             self._r1_grasp_eef_xyz, q_r1_home, +1.0)
            if self.tick >= GRIP_TICKS:
                # Latch R1↔plate offset for the carry/lower phases.
                self._r1_plate_offset = self._plate_pos - r1_pos
                self._advance("r0_release")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 8. R0 deactivates suction; R1 keeps the jaw closed ────
        if p == "r0_release":
            a0 = _arm_action(r0_pos, r0_quat, r0_pos, q_r0_pitched, -1.0)
            a1 = _arm_action(r1_pos, r1_quat,
                             self._r1_grasp_eef_xyz, q_r1_home, +1.0)
            if self.tick >= RELEASE_TICKS:
                self._advance("r0_retreat")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 9. R0 retreats away from R1 ─────────────────────────
        if p == "r0_retreat":
            r0_target = np.array([r0_pos[0],
                                  r0_pos[1] - 0.15,
                                  r0_pos[2] + 0.10])
            a0 = _arm_action(r0_pos, r0_quat, r0_target,
                             q_r0_pitched, -1.0)
            a1 = _arm_action(r1_pos, r1_quat,
                             self._r1_grasp_eef_xyz, q_r1_home, +1.0)
            if self.tick >= R0_RETREAT_TICKS:
                self._advance("r1_carry_to_shelf")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 10. R1 carries the vertical plate over the shelf ──────
        # Plate target = (shelf_x, shelf_y, hover_z) where hover_z
        # is high enough that the plate's bottom edge clears the
        # shelf walls.  Plate centre needs to end inside shelf bounds
        # (z ∈ (0.911, 1.171)). ──────────────────────────────────
        if p == "r1_carry_to_shelf":
            offset = (self._r1_plate_offset
                      if self._r1_plate_offset is not None else np.zeros(3))
            target_plate = np.array([shelf[0], shelf[1],
                                     TABLE_Z + 0.30])  # 1.20
            target_r1_eef = target_plate - offset
            a0 = _arm_action(r0_pos, r0_quat, R0_PARK_XYZ,
                             q_r0_pitched, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, target_r1_eef,
                             q_r1_home, +1.0)
            if (np.linalg.norm(r1_pos - target_r1_eef) < 0.05
                    or self.tick >= CARRY_TICKS):
                self._advance("r1_lower")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 11. R1 lowers the vertical plate into the central slot ─
        # Target plate centre at z=1.02 (bottom edge resting on
        # shelf base 0.911 + PLATE_HALF_X 0.109). ──────────────────
        if p == "r1_lower":
            offset = (self._r1_plate_offset
                      if self._r1_plate_offset is not None else np.zeros(3))
            target_plate = np.array([shelf[0], shelf[1],
                                     TABLE_Z + 0.12])  # 1.02
            target_r1_eef = target_plate - offset
            a0 = _arm_action(r0_pos, r0_quat, R0_PARK_XYZ,
                             q_r0_pitched, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, target_r1_eef,
                             q_r1_home, +1.0)
            if self._plate_in_shelf(env) or self.tick >= LOWER_TICKS:
                self._advance("r1_release")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 12. R1 opens the jaw; plate settles inside the slot ────
        if p == "r1_release":
            a0 = _arm_action(r0_pos, r0_quat, R0_PARK_XYZ,
                             q_r0_pitched, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, r1_pos, q_r1_home, -1.0)
            if self.tick >= R1_RELEASE_TICKS:
                self._advance("r1_retreat")
            return np.concatenate([a0, a1]).astype(np.float64)

        # ── 13. R1 retreats vertically out of the shelf footprint ──
        if p == "r1_retreat":
            r1_back = np.array([r1_pos[0],
                                r1_pos[1] + 0.10,
                                r1_pos[2] + 0.20])
            a0 = _arm_action(r0_pos, r0_quat, R0_PARK_XYZ,
                             q_r0_pitched, -1.0)
            a1 = _arm_action(r1_pos, r1_quat, r1_back, q_r1_home, -1.0)
            if self.tick >= R1_RETREAT_TICKS:
                self._done = True
                self._advance("done")
            return np.concatenate([a0, a1]).astype(np.float64)

        # done
        a0 = _hold(-1.0)
        a1 = _hold(-1.0)
        self._done = True
        return np.concatenate([a0, a1]).astype(np.float64)
