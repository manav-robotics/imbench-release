#!/usr/bin/env python3
"""Scripted bimanual policy for ib-p1-plate-shelf-hobi-v1.

Strategy (rim approach derived from a teleoperated demonstration):
  1. Arm 1 (+Y side) sweeps to the plate +Y rim with its gripper closed.
  2. Arm 0 (-Y side) approaches with its jaw open; arm 1 pushes the rim in.
  3. Arm 0 closes while arm 1 supports the plate, then lifts slowly.
  4. Starts near the IK workspace limit use one physical relocation and
     regrasp before turning the plate on edge.
  5. Arm 0 uses the live plate pose to align and insert it into the shelf,
     then releases and withdraws while the plate settles.

Controller: IK_POSE, input_type=absolute, action ∈ [-1, 1].
  pos action → scaled by ik_pos_sensitivity = 0.005 m/step
  ori action → scaled by ik_ori_sensitivity = 0.01 rad/step
  The task integrates these deltas into a persistent EEF target, then sends
  IK the correction (target − current_EEF). Position commands advance that
  target toward a waypoint, avoiding accumulated error at blocked contacts.

Plate geometry (from YCB STL bounding box):
  radius ≈ 0.131 m  |  thickness ≈ 0.030 m  |  spawn Z ≈ 0.9161

Shelf geometry (from task reset + XML, after +90° Z rotation):
  placed at [-0.25, 0, 0.901]  |  opening faces +X
  success bounds (world, centre slot):
    X ∈ [-0.335, -0.160]   Y ∈ [-0.060, +0.060]   Z ∈ [0.911, 1.171]

Robot home positions (from teleop initial state):
  Arm 0: (0.457, -0.660, 1.011) — on the -Y side of the workspace
  Arm 1: (0.457, +0.660, 1.011) — on the +Y side of the workspace

This is the scripted collector for this task.  It saves successful episodes
in LeRobot v2.1 format under ``<out-dir>/<task-id>/M<machine-id>/`` (see
``data_synthetic/collection.py``).

Usage::

    python -m data_synthetic.mp_hobi_plate_shelf --n-episodes 10
    python -m data_synthetic.mp_hobi_plate_shelf --render
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from imbench.policy.base_policy import PolicyInterface

from data_synthetic.collection import run_task_cli
from data_synthetic.hetbi_scripted._utils import quat_err_axis_angle

logger = logging.getLogger(__name__)

TASK_ID = "ib-p1-plate-shelf-hobi-v1"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

_PLATE_RADIUS     = 0.131    # horizontal rim radius (from STL)
_PLATE_SPAWN_Z    = 0.9161   # plate body-centre Z at default spawn


# Y offset from plate centre to EEF grip point (slightly inside plate rim).
# Teleop observed: arm1 at Y=+0.062 with plate centre at Y=-0.061 → offset ≈ 0.123.
_GRASP_Y          = 0.120

_LIFT_Z           = 1.22     # safe travel height for carry (above shelf walls at ≈1.181)
_ARM1_HOME        = np.array([0.457, 0.660, 1.011])  # arm1 resting position

# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

_ORI_SENS    = 0.01    # rad per unit action (from YAML ik_scale.ori_sensitivity)
_CONV_POS    = 0.030   # m  — position convergence tolerance (loosened for tilted EEF)
_CONV_ORI    = 0.05    # rad — orientation convergence tolerance (~3°)
_ARM0_TILT_RAD = np.deg2rad(45)  # arm0 tilt towards arm1 (+Y) during approach
_MAX_STEPS      = 200   # step timeout per FSM phase
_GRASP_SETTLE   = 80    # steps to dwell after gripper closes before advancing
_MAX_RETRIES    = 2     # max grasp reattempts before aborting episode

# ---------------------------------------------------------------------------
# Waypoint / arc-carry constants  (architecture ported from mass_sort_policy)
# ---------------------------------------------------------------------------

_OVERSHOOT_MAG   = 0.0    # disabled — overshoot nudge causes back-and-forth in IK_POSE
_OVERSHOOT_DECAY = 0.70
_WP_XY_TOL       = 0.030  # tighter lookahead: more samples engaged so the arc actually carves
_WP_Z_TOL        = 0.030
_WP_HOLD_STEPS   = 1      # advance on first touch, no dwell
_N_WP_APPROACH   = 6      # 6 intermediate samples → smooth approach arc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_converged(current: np.ndarray, target: np.ndarray) -> bool:
    return float(np.linalg.norm(current - target)) < _CONV_POS


def _quat_roll(q_xyzw: np.ndarray) -> float:
    """Extract rotation around world X-axis (roll) from an xyzw quaternion."""
    x, y, z, w = q_xyzw
    return float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))


def _ori_err_x(q_xyzw: np.ndarray, target_roll: float) -> float:
    """Signed X-axis angular error wrapped to [-π, π] (always takes shortest path)."""
    err = target_roll - _quat_roll(q_xyzw)
    return float((err + np.pi) % (2 * np.pi) - np.pi)


def _ori_action_x(q_xyzw: np.ndarray, target_roll: float, clip: float = 0.3) -> float:
    """Soft-clipped X-axis orientation action; tanh-shaped to avoid bang-bang."""
    err = _ori_err_x(q_xyzw, target_roll) / _ORI_SENS
    return float(np.tanh(err / 3.0) * clip)


def _make_arc_waypoints(
    start: np.ndarray, end: np.ndarray, n: int, z_lift: float = 0.0
) -> list[np.ndarray]:
    """Interpolate n waypoints from start to end with a sinusoidal Z bow."""
    wps = []
    for i in range(1, n + 1):
        t  = i / n
        pt = ((1.0 - t) * start + t * end).copy()
        pt[2] += z_lift * np.sin(np.pi * t)
        wps.append(pt)
    return wps


# ---------------------------------------------------------------------------
# OvershootOscillator  (ported from mass_sort_policy)
# ---------------------------------------------------------------------------

class OvershootOscillator:
    """Single nudge past the final waypoint that decays to zero within ~10 steps."""

    def __init__(self) -> None:
        self._offset = np.zeros(3)
        self._active = False

    def trigger(self, direction: np.ndarray) -> None:
        n = np.linalg.norm(direction)
        if n < 1e-6:
            return
        self._offset = direction / n * _OVERSHOOT_MAG
        self._active = True

    def step(self) -> np.ndarray:
        if not self._active:
            return np.zeros(3)
        out = self._offset.copy()
        self._offset *= _OVERSHOOT_DECAY
        if np.linalg.norm(self._offset) < 1e-4:
            self._active = False
            self._offset = np.zeros(3)
        return out

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False
        self._offset = np.zeros(3)


# ---------------------------------------------------------------------------
# WaypointTracker  (ported from mass_sort_policy, adapted for IK_POSE format)
# ---------------------------------------------------------------------------

class WaypointTracker:
    """Step through arc waypoints; trigger a small overshoot nudge at the end.

    Returns the effective target position for each step so callers can feed it
    into the persistent-target position controller.
    .finished is True once the last waypoint is held AND overshoot has decayed.
    """

    def __init__(self) -> None:
        self._waypoints: list[np.ndarray] = []
        self._wp_idx   = 0
        self._wp_hold  = 0
        self._dir      = np.zeros(3)
        self._overshoot = OvershootOscillator()
        self._loaded   = False

    def init(
        self, start: np.ndarray, end: np.ndarray, n: int, z_lift: float = 0.0
    ) -> None:
        self._waypoints = _make_arc_waypoints(start, end, n, z_lift=z_lift)
        self._wp_idx   = 0
        self._wp_hold  = 0
        self._overshoot.reset()
        diff = end - start
        norm = np.linalg.norm(diff)
        self._dir    = diff / norm if norm > 1e-6 else np.zeros(3)
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def finished(self) -> bool:
        return (
            self._loaded
            and self._wp_idx >= len(self._waypoints)
            and not self._overshoot.active
        )

    def tick(self, eef_pos: np.ndarray) -> np.ndarray:
        """Return the effective target (waypoint + overshoot offset)."""
        if not self._loaded or not self._waypoints:
            return eef_pos.copy()

        idx = min(self._wp_idx, len(self._waypoints) - 1)
        wp  = self._waypoints[idx]

        if self._wp_idx < len(self._waypoints):
            xy_err = np.linalg.norm(eef_pos[:2] - wp[:2])
            z_err  = abs(eef_pos[2] - wp[2])
            if xy_err < _WP_XY_TOL and z_err < _WP_Z_TOL:
                self._wp_hold += 1
                if self._wp_hold >= _WP_HOLD_STEPS:
                    self._wp_hold  = 0
                    self._wp_idx  += 1
                    if self._wp_idx >= len(self._waypoints) and not self._overshoot.active:
                        self._overshoot.trigger(self._dir)
            else:
                self._wp_hold = 0

        offset = self._overshoot.step()
        return wp + offset

    def reset(self) -> None:
        self._waypoints = []
        self._wp_idx    = 0
        self._wp_hold   = 0
        self._dir       = np.zeros(3)
        self._overshoot.reset()
        self._loaded    = False


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class _RimGraspPolicy(PolicyInterface):
    """Acquire the rim with the original two-arm approach sequence.

    The public policy keeps the pusher in place while the receiver closes,
    slows the lift, and takes over at R0_CARRY_LIFT for pose-based placement.
    """

    _PHASES = [
        "R0_PRETILT",
        "R1_APPROACH",
        "R1_DESCEND",
        "R1_GRASP",
        "R0_APPROACH",
        "R0_DESCEND",
        "R0_SLIDE",
        "R1_RELEASE",
        "R0_LIFT",
        "R0_CARRY_LIFT",
        "DONE",
    ]

    def __init__(
        self,
        plate_start_pos: np.ndarray,
        arm0_init_quat: np.ndarray,
        arm1_init_quat: np.ndarray,
    ) -> None:
        plate_pos = np.asarray(plate_start_pos, dtype=float)
        self._plate_xy = plate_pos[:2].copy()
        self._plate_z  = float(plate_pos[2])

        # Tilt target is relative to the arm's actual starting orientation so we
        # never cross the ±180° wraparound boundary.
        self._arm0_base_roll   = _quat_roll(np.asarray(arm0_init_quat, dtype=float))
        self._arm0_tilt_target = self._arm0_base_roll + _ARM0_TILT_RAD

        self._phase: str = "R0_PRETILT"
        self._phase_steps: int = 0
        self._hold_counter: int = 0
        self._grasp_settled: int = 0
        self._r0_retries: int = 0
        self._r1_retries: int = 0
        self._tracker = WaypointTracker()
        self.done: bool = False
        self._grasp_eef_roll: float = self._arm0_tilt_target

    @property
    def phase(self) -> str:
        return self._phase

    # ── PolicyInterface ──────────────────────────────────────────────────────

    def reset(self) -> None:
        self._phase = "R0_PRETILT"
        self._phase_steps = 0
        self._hold_counter = 0
        self._grasp_settled = 0
        self._r0_retries = 0
        self._r1_retries = 0
        self._tracker.reset()
        self.done = False
        self._grasp_eef_roll = self._arm0_tilt_target

    @staticmethod
    def _position(env, arm, target):
        return np.clip((target - env._target_pos[arm]) / env._ik_pos_sensitivity, -0.6, 0.6)

    def act(self, obs: dict, env) -> np.ndarray:
        action = np.zeros(14, dtype=np.float64)
        action[6] = action[13] = -1.0   # default: both grippers open

        if self.done:
            return action

        self._phase_steps += 1
        eef0 = obs["robot0_eef_pos"]
        eef1 = obs["robot1_eef_pos"]

        px, py = self._plate_xy
        pz     = self._plate_z

        # Fixed targets derived from initial plate position
        t1_above = np.array([px,  py + _PLATE_RADIUS + 0.03, pz + 0.10])  # Arm 1: 10 cm above, 2 cm before +Y rim
        t1_grasp = np.array([px,  py + _PLATE_RADIUS + 0.03, pz])       # Arm 1: 2 cm before +Y rim (pre-contact)
        t0_above = np.array([px,  py - _PLATE_RADIUS - 0.05, pz + 0.10]) # Arm 0: above, clear of plate
        t0_grasp    = np.array([px,  py - _PLATE_RADIUS + 0.02, pz + 0.03])    # Arm 0: holds here as plate slides in
        t1_slide    = np.array([px,  py + _GRASP_Y - 0.05,    pz])    # Arm 1: slide 5 cm in -Y, pushing plate into arm0
        t0_lift      = np.array([px,      py - _PLATE_RADIUS, _LIFT_Z])        # Arm 0: lifted above plate
        # ── Phase logic ──────────────────────────────────────────────────────

        if self._phase == "R0_PRETILT":
            # Roll arm0 EEF 45° towards arm1 (+Y) before any arm movement.
            action[3] = _ori_action_x(obs["robot0_eef_quat"], self._arm0_tilt_target)
            tilt_err  = abs(_ori_err_x(obs["robot0_eef_quat"], self._arm0_tilt_target))
            if tilt_err < _CONV_ORI or self._phase_steps > _MAX_STEPS:
                self._next_phase()

        elif self._phase == "R1_APPROACH":
            # Arm 1 follows arc waypoints to 10 cm above the plate +Y rim.
            if not self._tracker.loaded:
                self._tracker.init(eef1, t1_above, _N_WP_APPROACH)
            eff_target   = self._tracker.tick(eef1)
            action[7:10] = self._position(env, 1, eff_target)
            action[13]   = 1.0
            if self._tracker.finished or self._phase_steps > _MAX_STEPS:
                self._next_phase()

        elif self._phase == "R1_DESCEND":
            # Arm 1 descends to plate +Y rim; no waypoints needed for vertical drop.
            action[7:10] = self._position(env, 1, t1_grasp)
            action[13]   = 1.0
            if _pos_converged(eef1, t1_grasp) or self._phase_steps > _MAX_STEPS:
                self._next_phase()

        elif self._phase == "R1_GRASP":
            # Arm 1 arrives 2 cm before the +Y rim with gripper already closed.
            # No plate contact check — arm1 will push the plate into arm0 during R0_SLIDE.
            action[7:10] = self._position(env, 1, t1_grasp)
            action[13]   = 1.0
            if self._phase_steps == 1:
                self._grasp_eef_roll = _quat_roll(obs["robot0_eef_quat"])
            if _pos_converged(eef1, t1_grasp) or self._phase_steps > _MAX_STEPS:
                self._next_phase()

        elif self._phase == "R0_APPROACH":
            # Arm 0 follows arc waypoints to 10 cm above the plate -Y rim.
            if not self._tracker.loaded:
                self._tracker.init(eef0, t0_above, _N_WP_APPROACH)
            eff_target   = self._tracker.tick(eef0)
            action[7:10] = self._position(env, 1, t1_grasp)
            action[13]   = 1.0
            action[0:3]  = self._position(env, 0, eff_target)
            action[3]    = _ori_action_x(obs["robot0_eef_quat"], self._arm0_tilt_target)
            action[6]    = -1.0
            if self._tracker.finished or self._phase_steps > _MAX_STEPS:
                self._next_phase()

        elif self._phase == "R0_DESCEND":
            # Arm 0 descends to plate -Y rim; Arm 1 holds.
            action[7:10] = self._position(env, 1, t1_grasp)
            action[13]   = 1.0
            action[0:3]  = self._position(env, 0, t0_grasp)
            action[3]    = _ori_action_x(obs["robot0_eef_quat"], self._arm0_tilt_target)
            action[6]    = -1.0
            if _pos_converged(eef0, t0_grasp) or self._phase_steps > _MAX_STEPS:
                self._next_phase()

        elif self._phase == "R0_SLIDE":
            # Arm 0 holds open at t0_grasp as stationary receiver.
            # Arm 1 slides from pre-contact (-Y) into the plate, pushing its rim into arm0.
            # Arm 0 gripper stays open — it closes in R1_RELEASE once the plate is seated.
            action[0:3]  = self._position(env, 0, t0_grasp)
            action[3]    = _ori_action_x(obs["robot0_eef_quat"], self._arm0_tilt_target)
            action[6]    = -1.0
            action[7:10] = self._position(env, 1, t1_slide)
            action[13]   = 1.0
            if _pos_converged(eef1, t1_slide):
                self._next_phase()
            elif self._phase_steps > _GRASP_SETTLE:
                if self._r0_retries < _MAX_RETRIES:
                    self._r0_retries  += 1
                    self._phase        = "R0_APPROACH"
                    self._phase_steps  = 0
                    logger.debug("R0 slide failed — retry %d/%d", self._r0_retries, _MAX_RETRIES)
                else:
                    logger.warning("R0 slide failed after %d retries — aborting", _MAX_RETRIES)
                    self.done = True

        elif self._phase == "R1_RELEASE":
            # Arm 0 NOW closes its gripper to complete the grasp; arm 1 opens and retracts.
            # Holding for 40 steps lets physics settle before the lift begins.
            action[0:3]  = self._position(env, 0, t0_grasp)
            action[6]    = 1.0  # arm0 closes — grasp completes here
            action[7:10] = self._position(env, 1, t1_slide)
            action[13]   = -1.0
            if self._phase_steps >= 40:
                self._next_phase()

        elif self._phase == "R0_LIFT":
            # Arm 0 lifts plate slowly and rolls EEF back to 0° tilt; arm 1 retracts.
            action[0:3]  = self._position(env, 0, t0_lift)
            action[3]    = _ori_action_x(obs["robot0_eef_quat"], self._arm0_base_roll, clip=0.25)
            action[6]    = 1.0
            action[7:10] = self._position(env, 1, _ARM1_HOME)
            action[13]   = -1.0
            if _pos_converged(eef0, t0_lift):
                self._hold_counter += 1
            if self._hold_counter >= 100 or self._phase_steps > _MAX_STEPS:
                self._next_phase()

        return action

    # ── Internals ────────────────────────────────────────────────────────────

    def _next_phase(self) -> None:
        idx = self._PHASES.index(self._phase)
        if idx + 1 < len(self._PHASES):
            self._phase        = self._PHASES[idx + 1]
            self._phase_steps  = 0
            self._hold_counter = 0
            self._grasp_settled = 0
            self._tracker.reset()
            logger.debug("→ %s", self._phase)
        if self._phase == "DONE":
            self.done = True


class PlateShelfPolicy(PolicyInterface):
    """Sequential bimanual policy for ib-p1-plate-shelf-hobi-v1.

    Arm 1 (+Y side) pushes the rim into Arm 0's open fingers. Arm 0
    closes and lifts while Arm 1 retracts, then turns the plate vertical
    and carries it into the shelf. Transport requires a real grasp.

    Phase sequence
    --------------
    R0_PRETILT   : Arm 0 rolls EEF 45° towards arm1 (+Y) before any arm movement.
    R1_APPROACH  : Arm 1 follows arc waypoints to 10 cm above the plate +Y rim.
    R1_DESCEND   : Arm 1 descends from approach height to plate +Y rim.
    R1_GRASP     : Arm 1 holds its closed gripper just outside the rim.
    R0_APPROACH  : Arm 0 follows arc waypoints to 10 cm above the plate -Y rim.
    R0_DESCEND   : Arm 0 descends to plate -Y rim.
    R0_SLIDE     : Arm 0 holds open as a stationary receiver; arm 1 pushes
                   the plate rim into its fingers. Retries up to MAX_RETRIES.
    R1_RELEASE   : Arm 0 closes; arm 1 keeps supporting the rim for 40 steps.
    R0_LIFT      : Arm 0 lifts slowly without rolling; arm 1 retracts.
    R0_CARRY_LIFT: Hand over from the rim-grasp sequence to placement.
    regrasp_*   : For starts near the workspace limit, put the plate down
                   within reach, return home and acquire the rim once more.
    orient      : Turn the plate on edge above the shelf walls.
    carry       : Align the plate with the centre slot using its live pose.
    lower       : Lower the plate to insertion height in front of the shelf.
    insert      : Move into the slot; release only after reaching the opening.
    release     : Open the jaw while holding the wrist in place.
    withdraw    : Lift clear and wait for the task's settled-success check.
    failed      : Stop on a lost grasp, blocked motion or failed settling.

    Parameters
    ----------
    plate_start_pos : array-like (3,)
        Plate body-centre XYZ at episode start (obs["object_pos"]).
    arm0_init_quat : array-like (4,)  xyzw
        Used to compute the tilt target and return home before regrasping.
    arm1_init_quat : array-like (4,)  xyzw
        Unused; retained for API compatibility.
    """

    def __init__(self, plate_start_pos, arm0_init_quat, arm1_init_quat):
        self._initial = tuple(np.asarray(x).copy() for x in
                              (plate_start_pos, arm0_init_quat, arm1_init_quat))
        self.reset()

    def reset(self):
        self._grasp = _RimGraspPolicy(*self._initial)
        self._placement = None
        self._ticks = 0
        self._lost_steps = 0
        self._relocated = False
        self.done = self.failed = False
        self.failure_reason = None
        self._hold = self._home = self._prep_quat = None

    @property
    def phase(self):
        return self._placement or self._grasp.phase

    def _fail(self, reason):
        self.done = self.failed = True
        self.failure_reason = reason
        self._placement = "failed"
        return np.zeros(14, dtype=np.float64)

    def _advance(self, phase):
        self._placement, self._ticks = phase, 0

    @staticmethod
    def _position(env, arm, target, speed=0.45):
        return np.clip((target - env._target_pos[arm]) / env._ik_pos_sensitivity,
                       -speed, speed)

    def act(self, obs, env=None):
        if env is None:
            raise ValueError("PlateShelfPolicy requires the task for grasp and pose feedback")
        if self.done:
            return np.zeros(14, dtype=np.float64)
        pos = np.asarray(obs["robot0_eef_pos"])
        quat = np.asarray(obs["robot0_eef_quat"])
        plate = np.asarray(obs["object_pos"])
        grasped = env._gripper_contacts_plate(0)
        p = self._grasp
        # A low lift and regrasp avoids reorienting at the IK workspace edge.
        # This is bounded to one attempt; it is not a simulator reset.
        if (not self._relocated and p._plate_xy[0] < 0.15
                and p.phase == "R0_LIFT" and p._phase_steps > 70
                and grasped and self._placement is None):
            self._advance("regrasp_carry")
            self._prep_quat, self._home = quat.copy(), pos.copy()
            self._relocated = True
        if p.phase == "R0_CARRY_LIFT" and self._placement is None:
            if not grasped:
                return self._fail("receiver did not establish a grasp")
            self._advance("orient")
            self._home = np.r_[pos[:2], 1.28]

        if self._placement is None:
            action = p.act(obs, env=env)
            if p.phase == "R1_RELEASE":
                # Do not remove the support while the receiver is closing.
                action[7:10] = 0
                action[13] = 1
            if p.phase == "R0_LIFT":
                target = np.array([p._plate_xy[0], p._plate_xy[1] - 0.111, _LIFT_Z])
                action[:3] = self._position(env, 0, target, speed=0.25)
                action[3:6] = 0  # rotating during this lift rolled the plate out
                if p._phase_steps > 80:
                    self._lost_steps = 0 if grasped else self._lost_steps + 1
                    if self._lost_steps >= 30:
                        return self._fail("receiver lost the plate during the lift")
            if p.done:
                return self._fail("rim grasp failed")
            return action

        self._ticks += 1
        phase = self._placement
        sim = env._env.sim
        plate_id = sim.model.body_name2id(env._plate.root_body)
        normal = sim.data.body_xmat[plate_id].reshape(3, 3)[:, 2]
        target_normal = np.array([0., 1., 0.04])
        target_normal /= np.linalg.norm(target_normal)
        rotation, _ = Rotation.align_vectors([target_normal], [normal])
        target_quat = (rotation * Rotation.from_quat(quat)).as_quat()
        action = np.zeros(14, dtype=np.float64)
        action[6], action[13] = 1, -1
        target = self._home.copy()
        next_phase = None
        if phase in ("regrasp_carry", "regrasp_lower"):
            goal = np.array([0.25, 0, 1.02 if phase == "regrasp_carry" else 0.935])
            target = pos + goal - plate
            target_quat = self._prep_quat
            if np.linalg.norm(plate - goal) < 0.012:
                next_phase = "regrasp_lower" if phase == "regrasp_carry" else "regrasp_release"
                if next_phase == "regrasp_release":
                    self._hold = target.copy()
        elif phase == "regrasp_release":
            target, target_quat = self._hold, self._prep_quat
            action[6] = -1
            if self._ticks > 80:
                next_phase = "regrasp_retreat"
        elif phase == "regrasp_retreat":
            target, target_quat = self._hold + [0, 0, 0.2], self._prep_quat
            action[6] = -1
            if np.linalg.norm(pos - target) < 0.015:
                next_phase = "regrasp_home"
        elif phase == "regrasp_home":
            target, target_quat = np.array([0.457, -0.660, 1.011]), self._initial[1]
            action[6] = -1
            if (np.linalg.norm(pos - target) < 0.015
                    and np.linalg.norm(quat_err_axis_angle(quat, target_quat)) < 0.06):
                self._grasp = _RimGraspPolicy(plate, quat, obs["robot1_eef_quat"])
                self._grasp._arm0_base_roll = _quat_roll(self._initial[1])
                self._grasp._arm0_tilt_target = self._grasp._arm0_base_roll + _ARM0_TILT_RAD
                self._placement = None
                self._lost_steps = 0
        elif phase == "orient":
            if (abs(pos[2] - self._home[2]) < 0.01 and abs(normal[2] - 0.04) < 0.03
                    and abs(normal[0]) < 0.08):
                next_phase = "carry"
        elif phase in ("carry", "lower", "insert"):
            goal = np.array([-0.20 if phase == "insert" else 0.02,
                             -0.03, 1.19 if phase == "carry" else 1.048])
            target = pos + goal - plate
            # Release a vertical plate inside the slot instead of forcing the
            # wrist through the shelf to reach a fixed wrist-height target.
            seated = (phase == "insert" and env._plate_in_bounds(plate)
                      and plate[0] < -0.165 and plate[2] < 1.10
                      and abs(plate[1] + 0.03) < 0.02 and abs(normal[2]) < 0.25)
            if np.linalg.norm(plate - goal) < 0.01 or seated:
                next_phase = {"carry": "lower", "lower": "insert", "insert": "release"}[phase]
        elif phase == "release":
            target, action[6] = self._hold.copy(), -1
            if self._ticks > 60:
                next_phase = "withdraw"
        elif phase == "withdraw":
            target, action[6] = self._hold + [0, 0, 0.18], -1
            if self._ticks > 300:
                return self._fail("plate did not settle inside the shelf")

        holding = phase in ("orient", "carry", "lower", "insert", "regrasp_carry", "regrasp_lower")
        if holding:
            self._lost_steps = 0 if grasped else self._lost_steps + 1
            if self._lost_steps >= 20:
                return self._fail("plate slipped out of the receiver")
        if self._ticks > 800:
            return self._fail(f"{phase} target was not reached")
        action[:3] = self._position(env, 0, target)
        if phase not in ("release", "withdraw"):
            action[3:6] = np.clip(
                quat_err_axis_angle(env._target_orn_xyzw[0], target_quat) / env._ik_ori_sensitivity,
                -0.35, 0.35,
            )
        action[7:10] = self._position(env, 1, _ARM1_HOME, speed=0.5)
        if next_phase is not None:
            if next_phase == "release":
                self._hold = pos.copy()
            self._advance(next_phase)
        return action


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _make_policy(env, obs: dict, rng: np.random.Generator) -> PlateShelfPolicy:
    return PlateShelfPolicy(
        plate_start_pos=obs.get("object_pos", np.array([0.1, 0.0, _PLATE_SPAWN_Z])),
        arm0_init_quat=obs["robot0_eef_quat"],
        arm1_init_quat=obs["robot1_eef_quat"],
    )


if __name__ == "__main__":
    run_task_cli(
        TASK_ID, _make_policy,
        description=f"Scripted bimanual demos for {TASK_ID}",
        epilog=__doc__,
        stop_when_policy_done=True,
        log_prefix="mp_hobi_plate_shelf",
    )
