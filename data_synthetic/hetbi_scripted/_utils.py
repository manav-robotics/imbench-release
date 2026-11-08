"""Shared utilities for hetbi scripted motion policies.

Provides:
  - Sim-state readers (object/fixture position & orientation)
  - Proportional Cartesian delta controller
  - Quaternion helpers (xyzw convention throughout)
  - Phase-transition helpers

All positions are in the **world** frame and world-frame deltas are passed
directly as actions (the het-bi tasks run IK_POSE, pre-scaled per step by
HetBiBaseTask) — the same approach used by the single-arm scripted
policies in this project.

Action layout (14D for bimanual tasks):
    [0:3]   robot-0 position delta   (world frame, metres)
    [3:6]   robot-0 orientation delta (axis-angle, rad)
    [6]     robot-0 gripper  (+1 = close/suck, -1 = open/release)
    [7:10]  robot-1 position delta
    [10:13] robot-1 orientation delta
    [13]    robot-1 gripper
"""
from __future__ import annotations

import numpy as np

# ── Shared geometry constants ────────────────────────────────────────────────

TABLE_Z = 0.9                    # table surface height (matches all hetbi YAMLs)

# Top-down EEF orientation for the FINGER (parallel-jaw) gripper.
# The Panda's flange sits at ~identity xyzw at rest; flipping 180° about X
# brings the fingers pointing straight down.
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw — finger gripper Z down (180° about X)

# Top-down EEF orientation for the SUCTION gripper.
# The PandaSuctionGripper XML has the pump 'hand' body pre-rotated by -90°
# about Z relative to the Panda flange.  At the robot's natural home pose
# (obs eef_quat ≈ identity = [0,0,0,1] xyzw), this pre-rotation already
# points the snout straight down in world Z.  Commanding identity xyzw keeps
# the flange in that pose — do NOT apply the finger 180°-about-X flip here.
SUCTION_DOWN_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw — suction snout down (identity)

# ── Control gains & limits ───────────────────────────────────────────────────
POS_GAIN      = 6.0    # higher gain → faster convergence
ORI_GAIN      = 2.5
MAX_POS_STEP  = 0.50    # per-axis action magnitude cap
MAX_ORI_STEP  = 0.35

# Position tolerances for phase transitions
# Use coarser tolerances — OSC controllers never reach sub-cm precision at
# workspace extremes; always pair with tick-budget fallback in policies.
TOL_XY    = 0.03   # metres
TOL_Z     = 0.025
TOL_COARSE = 0.06   # looser tolerance for long-distance moves

# Setpoint-ramp parameters (smooth start when target changes)
RAMP_TICKS      = 12
RAMP_START_FRAC = 0.3
TARGET_CHANGE_TOL = 0.015  # m — below this, treat as the same setpoint


# ── Quaternion helpers (xyzw) ────────────────────────────────────────────────

def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=float)


def quat_err_axis_angle(cur_xyzw: np.ndarray, tgt_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle error from *cur* to *tgt* (both xyzw). Used as action[3:6]."""
    cx, cy, cz, cw = cur_xyzw
    tx, ty, tz, tw = tgt_xyzw
    # conjugate of cur
    ccx, ccy, ccz, ccw = -cx, -cy, -cz, cw
    # error quat = tgt * conj(cur)
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
    return (np.array([ex, ey, ez]) / sin_half) * angle


def yaw_quat(yaw_rad: float) -> np.ndarray:
    """xyzw quaternion for a pure Z-axis rotation by *yaw_rad* degrees."""
    half = 0.5 * yaw_rad
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=float)


def side_grasp_quat(approach_dir_xy: np.ndarray) -> np.ndarray:
    """EEF quaternion for a horizontal side-approach along *approach_dir_xy*.

    The resulting quaternion tilts the gripper 90° from top-down so its
    approach axis (local Z) points in the given XY direction.  This lets
    a parallel-jaw gripper pinch an object from the side rather than from
    above.

    Parameters
    ----------
    approach_dir_xy : (2,) unit vector in world XY; the direction the
        gripper snout should point toward the object.
    """
    d = np.asarray(approach_dir_xy, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    # Axis for the 90° tilt: perpendicular to approach and to world Z.
    # Rotating TOP_DOWN (Z_world down) by 90° around this axis gives
    # the gripper Z pointing in the approach_dir_xy direction.
    tilt_axis = np.array([-d[1], d[0], 0.0])
    norm = np.linalg.norm(tilt_axis)
    if norm < 1e-8:
        return TOP_DOWN_QUAT.copy()
    tilt_axis /= norm
    half = 0.5 * (np.pi / 2.0)  # 90° tilt
    tilt_q = np.array([
        tilt_axis[0] * np.sin(half),
        tilt_axis[1] * np.sin(half),
        tilt_axis[2] * np.sin(half),
        np.cos(half),
    ], dtype=float)
    return quat_mul_xyzw(tilt_q, TOP_DOWN_QUAT)


# ── Sim-state readers ────────────────────────────────────────────────────────

def read_obj_pos(env) -> np.ndarray:
    """World-frame position of the payload body."""
    sim = env._env.sim
    return np.array(sim.data.body_xpos[env._env._payload_body_id], dtype=float)


def read_obj_quat_xyzw(env) -> np.ndarray:
    """Payload orientation as xyzw quaternion (MuJoCo wxyz → xyzw)."""
    sim = env._env.sim
    q_wxyz = sim.data.body_xquat[env._env._payload_body_id]
    # MuJoCo body_xquat is wxyz; convert to xyzw.
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def read_fixture_pos(env) -> np.ndarray:
    """World-frame position of the fixture body (shelf/rack/slot)."""
    sim = env._env.sim
    return np.array(sim.data.body_xpos[env._env._fixture_body_id], dtype=float)


def read_goal_pos(env) -> np.ndarray:
    """Pre-computed goal target position stored on the inner env."""
    return np.array(env._env._target_goal_pos, dtype=float)


def read_eef(obs: dict, robot_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos [3], quat_xyzw [4]) for *robot_id* (0 or 1) from obs.

    Robosuite emits ``robot{i}_eef_quat`` already in xyzw
    (see robosuite/robots/robot.py — ``T.convert_quat(..., to="xyzw")``),
    so we read it directly without re-applying any wxyz→xyzw conversion.
    """
    pos  = np.array(obs[f"robot{robot_id}_eef_pos"],  dtype=float)
    quat = np.asarray(obs[f"robot{robot_id}_eef_quat"], dtype=float)
    return pos, quat


# ── Proportional controller ──────────────────────────────────────────────────

class _RampState:
    """Per-robot setpoint-ramp state (smooth out jerk on target changes)."""
    def __init__(self):
        self._last_target: np.ndarray | None = None
        self._ticks: int = 10_000

    def factor(self, target: np.ndarray) -> float:
        if (self._last_target is None
                or np.linalg.norm(target - self._last_target) > TARGET_CHANGE_TOL):
            self._ticks = 0
        self._last_target = target.copy()
        self._ticks += 1
        t = min(self._ticks / max(RAMP_TICKS, 1), 1.0)
        s = t * t * (3.0 - 2.0 * t)  # smoothstep
        return RAMP_START_FRAC + (1.0 - RAMP_START_FRAC) * s


def move_to(
    eef_pos:  np.ndarray,
    eef_quat: np.ndarray,
    target_pos:  np.ndarray,
    target_quat: np.ndarray,
    gripper:  float,
    ramp:     _RampState | None = None,
    gain_xy:  float = POS_GAIN,
    gain_z:   float = POS_GAIN,
    max_step: float = MAX_POS_STEP,
) -> np.ndarray:
    """Compute a 7D delta action to move toward *target_pos* / *target_quat*.

    Returns [dx, dy, dz, drx, dry, drz, gripper].
    """
    r = ramp.factor(target_pos) if ramp is not None else 1.0
    delta = target_pos - eef_pos
    action = np.zeros(7, dtype=float)
    action[0] = float(np.clip(delta[0] * gain_xy * r, -max_step, max_step))
    action[1] = float(np.clip(delta[1] * gain_xy * r, -max_step, max_step))
    action[2] = float(np.clip(delta[2] * gain_z  * r, -max_step, max_step))
    ori_err = quat_err_axis_angle(eef_quat, target_quat)
    action[3:6] = np.clip(ori_err * ORI_GAIN, -MAX_ORI_STEP, MAX_ORI_STEP)
    action[6] = gripper
    return action


def hold(eef_pos: np.ndarray, eef_quat: np.ndarray,
         gripper: float) -> np.ndarray:
    """Emit a zero-delta action that holds position and activates gripper."""
    return move_to(eef_pos, eef_quat, eef_pos, eef_quat, gripper)


def at_pos(pos: np.ndarray, target: np.ndarray,
           tol_xy: float = TOL_XY, tol_z: float = TOL_Z) -> bool:
    """True when *pos* is within tolerances of *target*."""
    d = np.abs(pos - target)
    return bool(d[0] < tol_xy and d[1] < tol_xy and d[2] < tol_z)


def dist_xy(pos: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pos[:2] - target[:2]))
