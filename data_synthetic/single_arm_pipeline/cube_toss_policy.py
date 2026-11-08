"""Scripted single-swing toss policy for ``ib-p2-cube-toss-v1``.

The robot grasps the cube and then performs **one continuous swing**
along a quadratic Bézier curve

    B(s) = (1−s)² · P_start + 2(1−s)s · P_control + s² · P_release

picked so that the curve's tangent at ``s = 1`` is the 45° launch
direction toward the goal:

    throw_dir = cos45° · dir_xy + sin45° · ẑ

For a quadratic Bézier the endpoint tangent is ``2·(P_release − P_control)``,
so ``P_control = P_release − k · throw_dir`` (with ``k > 0``) gives the
desired release direction. ``k`` controls the arc height.

The swing parameter ``s`` is advanced each tick at a rate proportional
to the projectile-required release speed, so that ``ds/dt · ‖B'(1)‖ ≈ v_target``.
The gripper opens once ``s ≥ 1`` and the measured EE speed along the
throw direction has reached the target.

All distances in metres, angles in radians.
"""

from __future__ import annotations

import numpy as np
from imbench.policy.base_policy import PolicyInterface


GRAVITY = 9.81

# Top-down grasp orientation: gripper Z points down (−world Z), gripper
# X axis aligned with world X. We multiply by an additional rotation
# about Z to match the cube's yaw (see ``_grasp_quat_for_cube_yaw``).
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# ── Grasp geometry ──────────────────────────────────────────────────
PRE_GRASP_HEIGHT_ABOVE_CUBE = 0.12
GRASP_OFFSET_ABOVE_CUBE     = 0.005
# Pre-swing lift: raise the EE to this height above the table after
# grasping, so the swing arc stays well clear of the table top.
LIFT_HEIGHT_ABOVE_TABLE     = 0.45
LIFT_MAX_STEPS              = 120

# ── Swing geometry (cubic Bézier) ───────────────────────────────────
# We use a cubic Bézier B(s) = (1-s)³·P0 + 3(1-s)²s·P1 + 3(1-s)s²·P2 + s³·P3
# so the start and end tangents can be set independently:
#   B'(0) = 3·(P1 - P0)   ← initial direction (we point straight up to
#                          lift the cube cleanly off the table)
#   B'(1) = 3·(P3 - P2)   ← release direction (the 45° throw vector)
#
# The magnitudes ‖P1-P0‖ and ‖P3-P2‖ act as "speed scale" at the two
# endpoints. Increasing END_HANDLE makes the EE move faster at release
# for the same swing duration.
# Length of the swing chord along throw_dir_3d. P3 sits SWING_CHORD
# metres from P0 along the 45° throw direction. The Panda's max reach
# is ~0.85 m from base, so we keep the chord modest to avoid hitting
# the joint-limit singularity at full extension.
SWING_CHORD               = 0.60
# Throw angle above the world horizontal (degrees). 45° maximises
# projectile range; lower angles trade range for a flatter trajectory
# that puts more of the chord into +xy and less into +z, keeping the
# arc clear of the Panda's vertical reach limit.
THROW_ANGLE_DEG           = 45.0
# Bézier handles. Both along throw_dir_3d → curve is a near-straight
# line so the EE only ever needs to move in the throw direction.
START_HANDLE_LEN          = 0.35
END_HANDLE_LEN            = 0.35
# Extra horizontal extension of the release point in the xy plane
# (without raising the apex). Useful when the throw angle is low —
# the release tangent stays at THROW_ANGLE_DEG.
RELEASE_FORWARD_BOOST_XY  = 0.05
# 0 = pure +ẑ, 1 = pure throw_dir_3d.
START_HANDLE_TILT         = 1.0
# Wall-clock duration of the forward swing (s=0 → s=1) in ticks @100Hz.
# More steps = more wall-clock time for the EE to actually traverse
# the curve. The geometric upper bound on EE speed is chord/(DUR·dt).
SWING_DURATION_STEPS      = 60
# Lookahead in s used to pick the OSC setpoint. Empirically, modest
# lookahead (~10-15% of the curve) gives the best EE tracking; larger
# lookahead just makes OSC fight directional error rather than
# accumulate forward velocity.
SWING_LOOKAHEAD_STEPS     = 4
# How much s before the apex (s=1) to open the gripper. With OSC
# tracking the EE's velocity peak occurs right around s≈1.0, so we
# open just before the apex to catch the EE while still accelerating.
RELEASE_LEAD_S            = 0.1 
# Exponential-moving-average smoothing factor for v_along (lower = more
# smoothing). Used only for logging — release is a simple s-threshold
# gate.
VEL_SMOOTH_ALPHA          = 0.25
# How far past s=1 the swing keeps driving (with gripper open) so the
# motion stays continuous and never decelerates near release.
SWING_OVERRUN_S           = 0.15
# Fraction of v_target the EE must reach before we open the gripper.
RELEASE_VELOCITY_FRAC     = 0.5
# After grasping, rotate the EE back to canonical TOP_DOWN_QUAT (gripper
# X aligned with world X) before launching the swing. Holding the cube
# tightly while the gripper rotates about Z just turns the cube — the
# pinch grip survives the rotation.
REORIENT_MAX_STEPS        = 60
REORIENT_DONE_RAD         = 0.06   # ~3.4° tolerance
# Brief pause after reorient: hold the EE at its current pose with both
# orientation and position locked so any residual velocity damps out
# before the swing begins. Without this, lateral velocity from the
# reorient correction leaks into the swing as |v| that is not aligned
# with throw_dir_3d.
PRE_SWING_SETTLE_STEPS    = 60
PRE_SWING_SETTLE_VEL      = 0.05   # |v| threshold to declare settled (m/s)
# Ease-in: ramp the swing's effective ds/step from 0 → full over this
# many ticks, so the cube doesn't slip out of a friction grip when the
# EE jerks at saturated gain. Ease curve: smoothstep on tick fraction.
# Keep below ~25% of SWING_DURATION_STEPS so most of the swing is at
# full speed.
SWING_EASE_IN_STEPS       = 6

# ── Position-control gains ──────────────────────────────────────────
GAIN_XY  = 5.0
GAIN_Z   = 5.0
ROT_GAIN = 3.0
SWING_GAIN_XY = 60.0
SWING_GAIN_Z  = 60.0

# ── Phase budgets (ticks @ 100 Hz, dt = 0.01 s) ────────────────────
SETTLE_STEPS         = 30
PRE_GRASP_MAX_STEPS  = 500
DESCEND_MAX_STEPS    = 500
CLOSE_STEPS          = 40
SWING_MAX_STEPS      = 350
SETTLE_AFTER_STEPS   = 250

POS_TOL = 0.008


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
    sin_half = np.sqrt(ex * ex + ey * ey + ez * ez)
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, ew)
    axis = np.array([ex, ey, ez]) / sin_half
    return axis * angle


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _yaw_from_quat_xyzw(q: np.ndarray) -> float:
    """Extract yaw (rotation about world Z) from an (x, y, z, w) quat."""
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _grasp_quat_for_cube_yaw(cube_yaw: float) -> np.ndarray:
    """Return the gripper quat (x, y, z, w) for a top-down grasp aligned
    with the cube's yaw.

    The cube has 4-fold symmetry about Z, so we fold the yaw into
    [−π/4, π/4] and pick the closest 90° step to minimise gripper rotation.

    Composition: q = qz(yaw) ⊗ TOP_DOWN_QUAT,
    where qz(α) is a rotation by α about world Z.
    """
    folded = (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    half = 0.5 * folded
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


def _release_velocity(d_xy: float, h: float, angle_deg: float) -> float:
    """Speed for a projectile launched at angle θ above horizontal to
    land at horizontal distance d_xy given release height h above the
    landing point.

    Range equation:
        d = (v cosθ) · (v sinθ + sqrt((v sinθ)² + 2 g h)) / g
    Solving for v² (standard form):
        v² = g · d² / (2 cos²θ · (d tanθ + h))
    """
    theta = np.radians(angle_deg)
    c = float(np.cos(theta))
    t = float(np.tan(theta))
    denom = max(2.0 * c * c * (d_xy * t + h), 1e-3)
    return float(np.sqrt(max(GRAVITY * d_xy * d_xy / denom, 0.0)))


class CubeTossPolicy(PolicyInterface):
    """Single-swing scripted toss."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._goal_xy: np.ndarray | None = None
        self._table_z: float = 0.8

        # Bézier swing plan (built once at the start of `swing`)
        self._P0: np.ndarray | None = None
        self._P1: np.ndarray | None = None
        self._P2: np.ndarray | None = None
        self._P3: np.ndarray | None = None
        self._throw_dir_xy: np.ndarray | None = None
        self._throw_dir_3d: np.ndarray | None = None
        self._release_speed: float = 0.0
        self._s: float = 0.0
        self._ds_per_step: float = 0.0

        # EE velocity (finite-diff) tracking
        self._prev_eef_pos: np.ndarray | None = None
        self._eef_vel: np.ndarray = np.zeros(3)

        # Grasp orientation (locked in once we begin pre_grasp from the
        # cube's measured yaw, so descent doesn't track jitter).
        self._grasp_quat: np.ndarray | None = None

        # EE position locked in at the end of reorient, used as the
        # holding setpoint during pre_swing_settle.
        self._swing_start_pos: np.ndarray | None = None

        # Velocity-peak detector state for the swing's release trigger.
        self._peak_v_along: float = 0.0
        self._smoothed_v_along: float = 0.0

    # ── Phase machinery ───────────────────────────────────────────────
    def _advance(self, next_phase: str):
        print(f"  [{self.step_in_phase:4d} steps] {self.phase} → {next_phase}")
        self.phase = next_phase
        self.step_in_phase = 0

    # ── Bézier helpers (cubic) ───────────────────────────────────────
    def _bezier(self, s: float) -> np.ndarray:
        """Position on the cubic Bézier.

        For ``s > 1`` we linearly extrapolate along B'(1) so the swing
        has somewhere to go past release without decelerating.
        """
        if s <= 1.0:
            s = max(s, 0.0)
            u = 1.0 - s
            return (u * u * u) * self._P0 \
                 + (3.0 * u * u * s) * self._P1 \
                 + (3.0 * u * s * s) * self._P2 \
                 + (s * s * s) * self._P3
        tangent = self._bezier_deriv(1.0)
        return self._P3 + (s - 1.0) * tangent

    def _bezier_deriv(self, s: float) -> np.ndarray:
        """B'(s) = 3(1-s)²(P1-P0) + 6(1-s)s(P2-P1) + 3s²(P3-P2)."""
        s = float(np.clip(s, 0.0, 1.0))
        u = 1.0 - s
        return 3.0 * u * u * (self._P1 - self._P0) \
             + 6.0 * u * s * (self._P2 - self._P1) \
             + 3.0 * s * s * (self._P3 - self._P2)

    # ── Plan the single swing once we have grasped the cube ──────────
    def _plan_swing(self, eef_pos: np.ndarray):
        goal_xy = self._goal_xy

        # Throw direction in XY: from the EEF (still above the cube) to the goal.
        delta_xy = goal_xy - eef_pos[:2]
        norm = np.linalg.norm(delta_xy)
        tdir_xy = delta_xy / norm if norm > 1e-6 else np.array([1.0, 0.0])
        self._throw_dir_xy = tdir_xy

        # Throw direction in 3D: cosθ along tdir_xy + sinθ along +ẑ,
        # where θ = THROW_ANGLE_DEG above horizontal.
        theta = np.radians(THROW_ANGLE_DEG)
        c, s = float(np.cos(theta)), float(np.sin(theta))
        self._throw_dir_3d = np.array([
            tdir_xy[0] * c,
            tdir_xy[1] * c,
            s,
        ])

        # P0: start of swing — current EE position (cube just gripped).
        self._P0 = eef_pos.copy()

        # P3: release point — placed SWING_CHORD metres from P0 *along
        # the throw direction*. This makes the chord parallel to the
        # release tangent → curve stays nearly straight → no lateral
        # motion is wasted, the EE accelerates monotonically along the
        # throw direction.
        self._P3 = self._P0 + self._throw_dir_3d * SWING_CHORD
        # Push the release point further along the throw direction in
        # the xy plane only — moves P3 toward the goal without raising
        # the apex.
        self._P3 = self._P3 + np.array([
            tdir_xy[0] * RELEASE_FORWARD_BOOST_XY,
            tdir_xy[1] * RELEASE_FORWARD_BOOST_XY,
            0.0,
        ])

        # P1: start tangent direction is a blend of +ẑ (vertical lift) and
        # throw_dir_3d (45° forward+up). Tilt=0 gives a pure-lift start
        # which often produces a backward-dipping curve when the throw
        # direction differs from +ẑ; tilt=0.5 keeps the curve monotonic
        # in the throw direction while still arcing upward.
        lift_dir = (1.0 - START_HANDLE_TILT) * np.array([0.0, 0.0, 1.0]) \
                 + START_HANDLE_TILT * self._throw_dir_3d
        lift_dir = lift_dir / max(np.linalg.norm(lift_dir), 1e-9)
        self._P1 = self._P0 + lift_dir * START_HANDLE_LEN

        # P2: makes B'(1) = 3·(P3-P2) be the 45° throw direction.
        # Length sets release tangent speed: ‖B'(1)‖ = 3·END_HANDLE_LEN.
        self._P2 = self._P3 - self._throw_dir_3d * END_HANDLE_LEN

        # Constant ds/step → constant parametric speed. Endpoint tangent
        # speed at release: ‖B'(1)‖ · (ds/step) / dt.
        self._ds_per_step = 1.0 / max(SWING_DURATION_STEPS, 1)
        self._s = 0.0

        # For diagnostics: the projectile speed required to reach goal.
        release_xy = self._P3[:2]
        release_z  = self._P3[2]
        d_xy = float(np.linalg.norm(goal_xy - release_xy))
        h    = release_z - self._table_z
        self._release_speed = _release_velocity(d_xy, h, THROW_ANGLE_DEG)

        # Realised tangent speed at endpoint (what the EE should achieve
        # if it tracks the curve perfectly).
        end_tangent = self._bezier_deriv(1.0)
        end_speed   = float(np.linalg.norm(end_tangent)) \
                    * self._ds_per_step / 0.01

        print(f"  [plan]  P0={np.round(self._P0, 3)}  "
              f"P1={np.round(self._P1, 3)}")
        print(f"          P2={np.round(self._P2, 3)}  "
              f"P3={np.round(self._P3, 3)}")
        print(f"  [plan]  d_xy={d_xy:.3f}  h={h:.3f}  "
              f"v_target={self._release_speed:.3f} m/s  "
              f"v_planned={end_speed:.3f} m/s  "
              f"ds/step={self._ds_per_step:.4f}")

    # ── Action helpers ────────────────────────────────────────────────
    def _move_to(self, eef_pos, eef_quat, target_pos, gripper,
                 gain_xy=GAIN_XY, gain_z=GAIN_Z, max_step=1.0,
                 keep_orient=True, target_quat=None):
        action = np.zeros(7)
        delta = target_pos - eef_pos
        action[0] = float(np.clip(delta[0] * gain_xy, -max_step, max_step))
        action[1] = float(np.clip(delta[1] * gain_xy, -max_step, max_step))
        action[2] = float(np.clip(delta[2] * gain_z,  -max_step, max_step))
        if keep_orient:
            tgt = TOP_DOWN_QUAT if target_quat is None else target_quat
            ori_err = _quat_err_axis_angle(eef_quat, tgt)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
        action[6] = gripper
        return action

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0]))
        cube_pos = obs["object_pos"]

        if self._goal_xy is None and env is not None:
            self._goal_xy = np.asarray(env._goal_pos, dtype=float).copy()
            self._table_z = float(getattr(env, "_table_height", 0.8))

        # EE velocity estimate.
        if self._prev_eef_pos is not None:
            self._eef_vel = (eef_pos - self._prev_eef_pos) / 0.01
        self._prev_eef_pos = eef_pos.copy()

        action = np.zeros(7)
        action[6] = -1.0   # default: gripper open

        # ── 1. Settle ─────────────────────────────────────────────────
        if self.phase == "settle":
            if self.step_in_phase >= SETTLE_STEPS:
                # Lock in the grasp orientation from the cube's current
                # yaw (cube has 4-fold symmetry; pick the closest 90°).
                cube_quat = obs.get("object_quat",
                                    np.array([0.0, 0.0, 0.0, 1.0]))
                cube_yaw  = _yaw_from_quat_xyzw(cube_quat)
                self._grasp_quat = _grasp_quat_for_cube_yaw(cube_yaw)
                print(f"  [grasp]  cube_yaw={np.degrees(cube_yaw):+.1f}°  "
                      f"folded={np.degrees((cube_yaw + np.pi/4) % (np.pi/2) - np.pi/4):+.1f}°  "
                      f"grasp_quat={np.round(self._grasp_quat, 3)}")
                self._advance("pre_grasp")
            self.step_in_phase += 1
            return action

        # ── 2. Above cube ─────────────────────────────────────────────
        if self.phase == "pre_grasp":
            target = np.array([
                cube_pos[0],
                cube_pos[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_CUBE,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            pos_reached = (np.linalg.norm(target - eef_pos) < POS_TOL)
            ori_reached = (np.linalg.norm(
                _quat_err_axis_angle(eef_quat, self._grasp_quat)) < 0.10)
            if (pos_reached and ori_reached) \
                    or self.step_in_phase >= PRE_GRASP_MAX_STEPS:
                self._advance("descend")
            self.step_in_phase += 1
            return action

        # ── 3. Descend ────────────────────────────────────────────────
        if self.phase == "descend":
            target = np.array([
                cube_pos[0],
                cube_pos[1],
                cube_pos[2] + GRASP_OFFSET_ABOVE_CUBE,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_z=1.0, max_step=0.6,
                                   target_quat=self._grasp_quat)
            reached = (np.linalg.norm(target - eef_pos) < 0.01)
            if reached or self.step_in_phase >= DESCEND_MAX_STEPS:
                self._advance("close_gripper")
            self.step_in_phase += 1
            return action

        # ── 4. Close gripper around cube, still at grasp orientation ─
        if self.phase == "close_gripper":
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase >= CLOSE_STEPS:
                self._advance("lift")
            self.step_in_phase += 1
            return action

        # ── 4b. Lift: raise the EE+cube straight up to a safe height
        #         BEFORE reorienting or swinging, so subsequent motion
        #         is well clear of the table. ──────────────────────────
        if self.phase == "lift":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + LIFT_HEIGHT_ABOVE_TABLE])
            # Keep the grasp orientation here — reorient comes next.
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.015
            if reached or self.step_in_phase >= LIFT_MAX_STEPS:
                print(f"  [lift done]  eef_z={eef_pos[2]:.3f}")
                self._advance("reorient")
            self.step_in_phase += 1
            return action

        # ── 4c. Reorient: rotate the gripper (with cube held, well
        #         above table) back to the canonical TOP_DOWN_QUAT. ────
        if self.phase == "reorient":
            ori_err = _quat_err_axis_angle(eef_quat, TOP_DOWN_QUAT)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.7, 0.7)
            action[6] = 1.0
            if (np.linalg.norm(ori_err) < REORIENT_DONE_RAD
                    or self.step_in_phase >= REORIENT_MAX_STEPS):
                # Lock in the swing-start position now so pre_swing_settle
                # can damp residual velocity around it.
                self._swing_start_pos = eef_pos.copy()
                self._advance("pre_swing_settle")
            self.step_in_phase += 1
            return action

        # ── 4d. Pre-swing settle: damp residual velocity from reorient
        #         so the swing starts from rest. ────────────────────────
        if self.phase == "pre_swing_settle":
            target = self._swing_start_pos
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=TOP_DOWN_QUAT)
            speed = float(np.linalg.norm(self._eef_vel))
            if (speed < PRE_SWING_SETTLE_VEL and self.step_in_phase >= 10) \
                    or self.step_in_phase >= PRE_SWING_SETTLE_STEPS:
                if self._goal_xy is not None:
                    self._plan_swing(eef_pos)
                print(f"  [pre_swing_settle done]  residual |v|={speed:.3f}")
                self._advance("swing")
            self.step_in_phase += 1
            return action

        # ── 5. Single continuous Bézier swing (grasp → release → overrun) ─
        if self.phase == "swing":
            # Allow s to run past 1.0 so motion stays continuous past
            # release. Ease ds/step in over SWING_EASE_IN_STEPS ticks so
            # the cube doesn't slip from a friction grip on the first
            # saturated jerk.
            ease_t = min(self.step_in_phase / max(SWING_EASE_IN_STEPS, 1), 1.0)
            ease   = ease_t * ease_t * (3.0 - 2.0 * ease_t)   # smoothstep
            ds = self._ds_per_step * ease
            s_max = 1.0 + SWING_OVERRUN_S
            self._s = min(self._s + ds, s_max)

            # Lookahead target along the curve / extrapolation, so the
            # OSC setpoint is always ahead of the EE → never settles.
            # Use the *eased* ds so the lookahead also ramps up softly
            # at the start of the swing.
            s_ahead = min(
                self._s + ds * SWING_LOOKAHEAD_STEPS, s_max
            )
            target = self._bezier(s_ahead)

            # Open the gripper RELEASE_LEAD_S before reaching apex.
            # Gate on actual EE speed in the throw direction. We compare
            # against the planned tangent speed (what the curve was set
            # up to deliver) rather than v_target (what the projectile
            # needs to reach the goal), because the OSC controller will
            # often underdeliver — better to throw weakly than not throw.
            v_along = float(np.dot(self._eef_vel, self._throw_dir_3d))
            speed   = float(np.linalg.norm(self._eef_vel))
            v_target = max(self._release_speed, 1.0)
            end_tangent_norm = float(np.linalg.norm(self._bezier_deriv(1.0)))
            v_planned = end_tangent_norm * self._ds_per_step / 0.01
            speed_ok = v_along >= RELEASE_VELOCITY_FRAC * v_planned

            # Smooth v_along (EMA) for cleaner logging. Not used as a
            # release trigger anymore — the EE's true velocity peak in
            # this OSC setup happens right at s≈1.0, so a peak detector
            # mostly fires on intermediate dips and drops the cube too
            # early. Simple s-threshold release is more reliable here.
            self._smoothed_v_along = (
                VEL_SMOOTH_ALPHA * v_along
                + (1.0 - VEL_SMOOTH_ALPHA) * self._smoothed_v_along
            )
            v_smooth = self._smoothed_v_along
            if v_smooth > self._peak_v_along:
                self._peak_v_along = v_smooth

            # Release at apex once the EE is actually moving forward.
            at_apex = self._s >= 1.0 - RELEASE_LEAD_S
            forward = v_smooth > 0.05

            # Never open during ease-in.
            opened = ease >= 0.99 and at_apex and forward
            gripper = -1.0 if opened else 1.0

            action = self._move_to(eef_pos, eef_quat, target, gripper=gripper,
                                   gain_xy=SWING_GAIN_XY, gain_z=SWING_GAIN_Z,
                                   max_step=1.0)

            if self.step_in_phase % 10 == 0:
                print(f"  [swing s={self._s:.2f} ease={ease:.2f}]  "
                      f"v_smooth={v_smooth:+.2f}  peak={self._peak_v_along:.2f}  "
                      f"|v|={speed:.2f}  "
                      f"eef=({eef_pos[0]:+.2f},{eef_pos[1]:+.2f},{eef_pos[2]:.2f})  "
                      f"{'OPEN' if opened else 'closed'}")

            cube_v = obs.get("cube_vel", None)
            if opened and cube_v is not None and self.step_in_phase % 3 == 0:
                print(f"    cube_v={np.round(cube_v, 2)}  "
                      f"|v|={np.linalg.norm(cube_v):.2f}")

            done_swing = self._s >= s_max - 1e-3
            timeout    = self.step_in_phase >= SWING_MAX_STEPS
            if done_swing or timeout:
                print(f"  [swing-done]  s={self._s:.3f}  "
                      f"v_along={v_along:+.2f}  v_target={v_target:.2f}  "
                      f"reason={'overrun-end' if done_swing else 'timeout'}")
                self._advance("settle_after")
            self.step_in_phase += 1
            return action

        # ── 7. Wait for the cube to land & rest ──────────────────────
        if self.phase == "settle_after":
            action[6] = -1.0
            if self.step_in_phase >= SETTLE_AFTER_STEPS:
                self._advance("done")
            self.step_in_phase += 1
            return action

        # ── 8. Done ──────────────────────────────────────────────────
        action[6] = -1.0
        self.step_in_phase += 1
        return action


# ── Stand-alone runner ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    import imbench as ib

    env = ib.make("ib-p2-cube-toss-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = CubeTossPolicy()
    for ep in range(3):
        obs, info = env.reset(seed=ep * 13 + 1)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===  goal={env._goal_pos}")
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
        print(f"Episode {ep}  score={info.get('task_score', 0.0):.3f}  "
              f"final cube pos={obs['object_pos']}  phase={policy.phase}")
    env.close()
