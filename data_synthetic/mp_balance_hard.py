#!/usr/bin/env python3
"""Scripted policy for ``ib-p7-balance-hard-v1``.

Strategy
--------
The rod spawns lying flat on the table with its long axis along world X
(quat_lying = 90° about Y).  The platform's ridge runs along world Y
(after a 90° spawn rotation about Z).  Unlike the medium variant, this
policy stands the rod *vertically* on the ridge — a "skyscraper" balance
where the rod's bottom end rests on the ridge top.

Standing the rod up needs a 90° re-orientation between grasp (rod
horizontal) and place (rod vertical).  Doing it in one swing about world
Y forces the wrist to point along world ±X at carry height, which is the
Panda workspace region where the elbow flies up.  We split the rotation
in two so the wrist never visits that region:

  - ROLL_TO_SIDE: +90° about world X.  The rod lies along the rotation
    axis so it does not move; the fingers slide 90° around the cylinder
    while the wrist sweeps -Z → +Y in the y-z plane.
  - ROTATE: -90° about world Y.  With gripper-z now at world +Y this is
    a pure wrist roll — the forward axis stays at +Y while the rod axis
    sweeps +X → +Z, standing it up with the bulk hanging below the grip.

Phase sequence::

    INIT          : zero-action settle, gripper open
    APPROACH      : arc to a hover point above the rod's +X end
    DESCEND       : drop straight down onto the rod near its end
    GRASP         : close the gripper, dwell to settle
    LIFT          : raise (still horizontal) to carry height
    ROLL_TO_SIDE  : roll wrist +90° about world X (gripper-z: -Z → +Y)
    ROTATE        : wrist roll -90° about world Y (rod swings to vertical)
    MOVE_OVER     : sweep horizontally to above the platform ridge centre
    LOWER         : descend until the rod's bottom end is just above the ridge
    SETTLE        : hold the placement pose so contacts stabilise
    RELEASE       : open the gripper while holding XY/Z fixed
    RETRACT       : pull the gripper straight up to clear the rod's top
    HOLD          : zero-action wait while balance counter accumulates
    DONE

Controller
----------
OSC_POSE, ``cartesian_delta``, action ∈ [-1, 1]^7.
  - action[0:3]  position delta in world frame, scaled by output_max=0.05 m
  - action[3:6]  axis-angle delta in world frame, scaled by output_max=0.5 rad
  - action[6]    gripper (-1 open, +1 closed)

Usage::

    python -m data_synthetic.mp_balance_hard --n-episodes 10
    python -m data_synthetic.mp_balance_hard --render
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_synthetic.collection import run_task_cli

logger = logging.getLogger(__name__)

TASK_ID = "ib-p7-balance-hard-v1"

# ── Geometry tuning ──────────────────────────────────────────────────────────
GRASP_INSET       = 0.045   # m — distance from rod end (toward centre) at which to grip
GRASP_Z_PRESS     = 0.015   # m — descend this far below rod centre so fingers straddle the rod
HOVER_ABOVE_ROD   = 0.10    # m — pre-grasp hover height above grasp point
ROTATE_CLEARANCE  = 0.03    # m — gap between rod's hanging tip and ridge top after rotation
PLACE_Z_MARGIN    = -0.0125 # m — eef target at placement; negative drops the eef 10 mm below the "rod-bottom kissing the ridge" reference for firm contact
PLACE_X_BIAS      = -0.004  # m — shift LOWER target_x by +5 mm to compensate for residual rod tilt (the rod's bottom lands ~5 mm in -X of the eef when ROTATE finishes a few deg short of full vertical)
LOWER_Z_TOL       = 0.0010  # m — strict z-convergence for the placement phase
RELEASE_LIFT      = 0.10    # m — final retract distance above rod's top end

XY_TOL            = 0.008
Z_TOL             = 0.006
ORI_TOL           = np.deg2rad(5.0)

# ── Control gains ────────────────────────────────────────────────────────────
GAIN_XY           = 4.0
GAIN_Z            = 4.0
GAIN_FAST         = 5.0
GAIN_ORI          = 3.0
GAIN_LOWER_Z      = 6.0

# ── Phase timing ─────────────────────────────────────────────────────────────
# Worst-case sum across all phases must stay under the episode horizon (YAML: 2500).
# Allocation: 8+150+150+60+80+100+100+200+350+25+35+60+200 = 1518.
INIT_HOLD         = 8
APPROACH_BUDGET   = 150
DESCEND_BUDGET    = 150
GRASP_HOLD        = 60
LIFT_BUDGET       = 80
ROLL_BUDGET       = 100
ROTATE_BUDGET     = 100
MOVE_BUDGET       = 200
LOWER_BUDGET      = 350
SETTLE_HOLD       = 25
RELEASE_HOLD      = 35
RETRACT_BUDGET    = 60
HOLD_STEPS        = 200      # > balance_hold_steps=100 with grace margin

# ── Grasp verification ──────────────────────────────────────────────────────
MAX_GRASP_ATTEMPTS    = 3      # incl. the first try
GRASP_LIFT_MIN_HEIGHT = 0.05   # m — rod_z must rise this far above table_z post-LIFT
GRASP_RETRY_INSET_BUMP = 0.012 # m — each retry grips this much further from the end (more rod between fingers)
GRASP_CLOSURE_BUDGET  = 120    # steps — hard cap on GRASP duration if the closure-stability check never fires
GRASP_STABLE_WINDOW   = 5      # steps — gripper aperture must be steady over this many consecutive samples
GRASP_STABLE_TOL      = 5e-4   # m — max aperture delta across the stability window

# ── Waypoint tracker ─────────────────────────────────────────────────────────
WAYPOINT_HOLD_STEPS = 4
WP_XY_TOL           = 0.025
WP_Z_TOL            = 0.025
ARC_Z_LIFT          = 0.06
N_WP_APPROACH       = 6
N_WP_MOVE_OVER      = 8

# ── Gripper levels ───────────────────────────────────────────────────────────
GRIP_OPEN  = -1.0
GRIP_CLOSE =  1.0

# ── Rotations: top-grasp → side-grasp → upright ──────────────────────────────
# Two world-frame deltas applied sequentially (see module docstring for why):
#
#   side_grasp_quat = R_x(±π/2) · init_quat        # ROLL_TO_SIDE
#   upright_quat    = R_y(-π/2) · side_grasp_quat  # ROTATE  (wrist roll)
#
# The roll sign (±π/2 about world X) decides which side of the eef the gripper
# body sits on.  Per-episode IK probing at reset picks the sign that yields a
# feasible carry-pose joint configuration over the platform.  Either way the
# wrist axis (gripper-z) stays at world ±Y after the roll, never along ±X.
_SIDE_GRASP_DELTA_AXIS   = np.array([1.0, 0.0, 0.0])  # roll axis = world X
_UPRIGHT_DELTA_ROTVEC    = np.array([0.0, -np.pi / 2.0, 0.0])

# ── IK probe ─────────────────────────────────────────────────────────────────
IK_PROBE_MAX_ITERS  = 80
IK_PROBE_DAMPING    = 1e-3
IK_PROBE_DT         = 0.05
IK_PROBE_POS_TOL    = 0.01     # m
IK_PROBE_ORI_TOL    = 0.10     # rad
IK_PROBE_LIMIT_PAD  = 0.05     # rad — keep this far from joint limits


# ── Helpers ──────────────────────────────────────────────────────────────────

def _clip_action(dx, dy, dz, drx, dry, drz, gripper):
    a = np.zeros(7)
    a[0] = np.clip(dx,  -1.0, 1.0)
    a[1] = np.clip(dy,  -1.0, 1.0)
    a[2] = np.clip(dz,  -1.0, 1.0)
    a[3] = np.clip(drx, -1.0, 1.0)
    a[4] = np.clip(dry, -1.0, 1.0)
    a[5] = np.clip(drz, -1.0, 1.0)
    a[6] = gripper
    return a


def _no_move(gripper=GRIP_OPEN):
    return _clip_action(0, 0, 0, 0, 0, 0, gripper)


def _make_arc_waypoints(start, end, n, z_lift=0.0):
    wps = []
    for i in range(1, n + 1):
        t  = i / n
        pt = ((1.0 - t) * start + t * end).copy()
        pt[2] += z_lift * np.sin(np.pi * t)
        wps.append(pt)
    return wps


def _orientation_error(current_quat_xyzw, target_quat_xyzw):
    """Axis-angle vector that rotates *current* to *target* (world frame)."""
    r_err = (
        Rotation.from_quat(target_quat_xyzw)
        * Rotation.from_quat(current_quat_xyzw).inv()
    )
    return r_err.as_rotvec()


class WaypointTracker:
    """Lightweight arc-waypoint tracker (no overshoot — overshoot is risky here)."""

    def __init__(self):
        self._wps = []
        self._idx = 0
        self._hold = 0
        self._loaded = False

    def init(self, start, end, n, z_lift=0.0):
        self._wps = _make_arc_waypoints(start, end, n, z_lift=z_lift)
        self._idx = 0
        self._hold = 0
        self._loaded = True

    @property
    def loaded(self):
        return self._loaded

    @property
    def finished(self):
        return self._loaded and self._idx >= len(self._wps)

    def tick(self, eef_pos):
        if not self._loaded or not self._wps:
            return eef_pos.copy()
        idx = min(self._idx, len(self._wps) - 1)
        wp = self._wps[idx]
        if self._idx < len(self._wps):
            xy_err = np.linalg.norm(eef_pos[:2] - wp[:2])
            z_err  = abs(eef_pos[2] - wp[2])
            if xy_err < WP_XY_TOL and z_err < WP_Z_TOL:
                self._hold += 1
                if self._hold >= WAYPOINT_HOLD_STEPS:
                    self._hold = 0
                    self._idx += 1
            else:
                self._hold = 0
        return wp

    def reset(self):
        self._wps = []
        self._idx = 0
        self._hold = 0
        self._loaded = False


# ── Policy ───────────────────────────────────────────────────────────────────

class BalanceHardPolicy:
    """FSM policy that picks the rod, stands it vertical, and lowers it onto
    the ridge centreline so the rod balances on its end (skyscraper)."""

    _PHASES = [
        "INIT",
        "APPROACH",
        "DESCEND",
        "GRASP",
        "LIFT",
        "ROLL_TO_SIDE",
        "ROTATE",
        "MOVE_OVER",
        "LOWER",
        "SETTLE",
        "RELEASE",
        "RETRACT",
        "HOLD",
        "DONE",
    ]

    def __init__(self, env, force_yaw: Optional[float] = None):
        self.env = env
        self._force_yaw = force_yaw  # if set, bypasses IK probe and uses this yaw
        self._table_z          = float(env._table_height)
        self._rod_half_len     = float(env._rod_half_length)
        self._rod_radius       = float(env._rod_radius)
        self._ridge_height     = float(env._ridge_height)
        base_half_z            = float(env._base_half_size[2])
        self._ridge_top_z      = self._table_z + 2.0 * base_half_z + self._ridge_height

        self._grasp_inset      = GRASP_INSET
        # Length of rod hanging below gripper after rotation (centre-to-far-end
        # distance through the grasp point along the rod axis).
        self._rod_long_grasped = 2.0 * self._rod_half_len - self._grasp_inset

        # Cached at episode start
        self._rod_pos_at_grasp: Optional[np.ndarray] = None
        self._platform_xy: Optional[np.ndarray] = None
        self._eef_init_quat: Optional[np.ndarray] = None
        self._eef_side_grasp_quat: Optional[np.ndarray] = None
        self._eef_upright_quat: Optional[np.ndarray] = None
        self._gripper_z_at_lift: float = 0.0
        self._rotate_xy: Optional[np.ndarray] = None
        self._grasp_attempts: int = 0
        self._grasp_aperture_hist: list[float] = []

        self.phase = "INIT"
        self.step_in_phase = 0
        self.done = False
        self._tracker = WaypointTracker()

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self, obs):
        self.phase = "INIT"
        self.step_in_phase = 0
        self.done = False
        self._tracker.reset()

        self._rod_pos_at_grasp = None
        self._platform_xy = np.array(obs["platform_pos"][:2], dtype=float)
        self._eef_init_quat = np.array(obs["robot0_eef_quat"], dtype=float)
        self._rotate_xy = None
        self._grasp_attempts = 0
        self._grasp_aperture_hist = []

        # Carry height: keep the rod's hanging tip clear of the ridge top by
        # ROTATE_CLEARANCE once the rod is rotated to vertical.
        self._gripper_z_at_lift = (
            self._ridge_top_z + self._rod_long_grasped + ROTATE_CLEARANCE
        )

        # Pick the carry yaw (rotation about world Z applied to the base
        # upright_quat).  Either an externally-forced value (e.g. from an
        # empirical retry loop) or chosen via IK probing along the carry path.
        if self._force_yaw is not None:
            carry_yaw = float(self._force_yaw)
            logger.info("  carry_yaw forced to %+.1f° (skipping IK probe)",
                        np.rad2deg(carry_yaw))
        else:
            carry_yaw = self._choose_carry_yaw(obs)
        side_grasp_delta = (np.pi / 2.0) * _SIDE_GRASP_DELTA_AXIS
        side_grasp_rot = (
            Rotation.from_rotvec(side_grasp_delta)
            * Rotation.from_quat(self._eef_init_quat)
        )
        self._eef_side_grasp_quat = side_grasp_rot.as_quat()
        base_upright = (
            Rotation.from_rotvec(_UPRIGHT_DELTA_ROTVEC)
            * Rotation.from_quat(self._eef_side_grasp_quat)
        )
        upright_rot = (
            Rotation.from_rotvec(np.array([0.0, 0.0, carry_yaw]))
            * base_upright
        )
        self._eef_upright_quat = upright_rot.as_quat()

        logger.info(
            "phase=INIT  rod_hl=%.3f  rod_long=%.3f  ridge_top_z=%.3f  carry_z=%.3f  carry_yaw=%+.1f°",
            self._rod_half_len, self._rod_long_grasped,
            self._ridge_top_z, self._gripper_z_at_lift, np.rad2deg(carry_yaw),
        )

    def act(self, obs):
        if self.done or self.phase == "DONE":
            self.done = True
            return _no_move(GRIP_OPEN)

        eef_pos  = np.array(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=float)
        rod_pos  = np.array(obs["object_pos"], dtype=float)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
        self._last_eef_pos = eef_pos
        self._last_rod_pos = rod_pos

        action = self._dispatch(eef_pos, eef_quat, rod_pos, gripper_qpos)
        self.step_in_phase += 1
        return action

    # ── IK probe (roll-direction selection) ──────────────────────────────────

    def _choose_carry_yaw(self, obs) -> float:
        """Return the world-Z yaw to apply to the base upright_quat so that
        the carry pose is feasible along the entire lift→platform path.

        The rod is cylindrical so yawing the wrist about world Z (= the rod's
        vertical axis post-ROTATE) keeps the rod vertical and just shifts
        which horizontal direction the gripper body points.  The two
        ±π/2-roll cases correspond to yaw=0 and yaw=π.

        Sample yaws inside the "safe" bands (away from gripper-z along ±X,
        which is the Panda elbow-blowup region), probe each along 5 carry
        waypoints, and pick the one with the most feasible waypoints
        (tiebreak on joint-limit margin).
        """
        rod_xy = np.array(obs["object_pos"][:2], dtype=float)
        lift_xy = np.array([
            rod_xy[0] + (self._rod_half_len - self._grasp_inset),
            rod_xy[1],
        ])
        carry_z = self._gripper_z_at_lift

        n_wp = 5
        waypoints = [
            np.array([
                (1.0 - t) * lift_xy[0] + t * self._platform_xy[0],
                (1.0 - t) * lift_xy[1] + t * self._platform_xy[1],
                carry_z,
            ])
            for t in np.linspace(0.0, 1.0, n_wp)
        ]

        # Safe yaw bands: avoid α near ±π/2 where gripper-z points along ±X.
        # 9 candidates in each band; 18 total.
        deg = np.deg2rad
        yaw_grid = np.concatenate([
            np.linspace(deg(-40), deg(+40), 9),         # near yaw=0    (sign +1 family)
            np.linspace(np.pi + deg(-40), np.pi + deg(+40), 9),  # near yaw=π  (sign -1 family)
        ])

        # Base upright_quat (sign=+1, no extra yaw).
        side_quat = (
            Rotation.from_rotvec((np.pi / 2.0) * _SIDE_GRASP_DELTA_AXIS)
            * Rotation.from_quat(self._eef_init_quat)
        ).as_quat()
        base_upright_quat = (
            Rotation.from_rotvec(_UPRIGHT_DELTA_ROTVEC)
            * Rotation.from_quat(side_quat)
        ).as_quat()

        results = []
        for yaw in yaw_grid:
            yawed = (
                Rotation.from_rotvec(np.array([0.0, 0.0, yaw]))
                * Rotation.from_quat(base_upright_quat)
            ).as_quat()
            wp_results = [self._ik_probe(wp, yawed) for wp in waypoints]
            n_feasible = sum(1 for r in wp_results if r["feasible"])
            min_margin = min(r["margin"] for r in wp_results)
            max_pos_err = max(r["pos_err"] for r in wp_results)
            results.append({
                "yaw":         float(yaw),
                "n_feasible":  n_feasible,
                "min_margin":  min_margin,
                "max_pos_err": max_pos_err,
            })

        # Pick: most feasible waypoints, tiebreak on margin (higher = better).
        best = max(results, key=lambda r: (r["n_feasible"], r["min_margin"]))
        logger.info(
            "  PICK yaw=%+.1f°  n_feasible=%d/%d  min_margin=%+.3f  max_pos_err=%.4f",
            np.rad2deg(best["yaw"]), best["n_feasible"], n_wp,
            best["min_margin"], best["max_pos_err"],
        )
        # Brief summary of runner-up bands.
        best_pos_band = max(
            (r for r in results if 0 <= r["yaw"] < np.pi / 2),
            key=lambda r: (r["n_feasible"], r["min_margin"]),
        )
        best_neg_band = max(
            (r for r in results if r["yaw"] >= np.pi / 2),
            key=lambda r: (r["n_feasible"], r["min_margin"]),
        )
        logger.info(
            "    band_0:   best yaw=%+.1f° n=%d/%d  band_π: best yaw=%+.1f° n=%d/%d",
            np.rad2deg(best_pos_band["yaw"]), best_pos_band["n_feasible"], n_wp,
            np.rad2deg(best_neg_band["yaw"]), best_neg_band["n_feasible"], n_wp,
        )
        return float(best["yaw"])

    def _ik_probe(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        seed_qpos: Optional[np.ndarray] = None,
    ) -> dict:
        """Damped least-squares IK on the 7 Panda arm joints.  Saves and
        restores qpos/qvel — non-destructive to the live sim.  If
        ``seed_qpos`` is given, the probe starts from that joint config (so
        we converge to the IK branch nearest to it — the same branch the OSC
        will track).  Otherwise starts from the current sim qpos.
        """
        sim = self.env._env.sim
        m = sim.model._model
        d = sim.data._data

        site_id = m.site("gripper0_right_grip_site").id
        dof_ids   = np.array([int(m.jnt_dofadr[i])  for i in range(7)])
        qpos_ids  = np.array([int(m.jnt_qposadr[i]) for i in range(7)])
        jnt_lo    = m.jnt_range[:7, 0].copy()
        jnt_hi    = m.jnt_range[:7, 1].copy()

        qpos_save = d.qpos.copy()
        qvel_save = d.qvel.copy()
        if seed_qpos is not None:
            d.qpos[qpos_ids] = np.clip(seed_qpos, jnt_lo, jnt_hi)

        # mujoco quat is wxyz; our policy uses xyzw
        target_quat_wxyz = np.array([
            target_quat_xyzw[3], target_quat_xyzw[0],
            target_quat_xyzw[1], target_quat_xyzw[2],
        ])

        site_quat       = np.zeros(4)
        site_quat_conj  = np.zeros(4)
        err_quat        = np.zeros(4)
        err_aa          = np.zeros(3)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))

        for _ in range(IK_PROBE_MAX_ITERS):
            mujoco.mj_forward(m, d)
            site_pos = d.site_xpos[site_id].copy()
            mujoco.mju_mat2Quat(site_quat, d.site_xmat[site_id].copy())

            pos_err = target_pos - site_pos
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(err_quat, target_quat_wxyz, site_quat_conj)
            mujoco.mju_quat2Vel(err_aa, err_quat, 1.0)

            if (np.linalg.norm(pos_err) < IK_PROBE_POS_TOL
                    and np.linalg.norm(err_aa) < IK_PROBE_ORI_TOL):
                break

            twist = np.concatenate([pos_err, err_aa]) / IK_PROBE_DT
            mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])  # (6, 7)
            JJt = jac @ jac.T + IK_PROBE_DAMPING * np.eye(6)
            dq = jac.T @ np.linalg.solve(JJt, twist)
            new_q = d.qpos[qpos_ids] + dq * IK_PROBE_DT
            d.qpos[qpos_ids] = np.clip(new_q, jnt_lo, jnt_hi)

        mujoco.mj_forward(m, d)
        site_pos = d.site_xpos[site_id].copy()
        mujoco.mju_mat2Quat(site_quat, d.site_xmat[site_id].copy())
        pos_err = float(np.linalg.norm(target_pos - site_pos))
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(err_quat, target_quat_wxyz, site_quat_conj)
        mujoco.mju_quat2Vel(err_aa, err_quat, 1.0)
        ori_err = float(np.linalg.norm(err_aa))

        final_q = d.qpos[qpos_ids].copy()
        margin = float(np.min(np.minimum(final_q - jnt_lo, jnt_hi - final_q)))

        feasible = (
            pos_err < IK_PROBE_POS_TOL
            and ori_err < IK_PROBE_ORI_TOL
            and margin > IK_PROBE_LIMIT_PAD
        )

        # Restore live sim state
        d.qpos[:] = qpos_save
        d.qvel[:] = qvel_save
        mujoco.mj_forward(m, d)

        return {
            "feasible": feasible,
            "margin":   margin,
            "pos_err":  pos_err,
            "ori_err":  ori_err,
            "qpos":     final_q,
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _advance(self, nxt):
        cur_eef = getattr(self, "_last_eef_pos", None)
        cur_rod = getattr(self, "_last_rod_pos", None)
        if cur_eef is not None and cur_rod is not None:
            print(
                f"  [{self.step_in_phase:4d}]  {self.phase:>9s} → {nxt:<9s}  "
                f"eef_z={cur_eef[2]:.4f}  rod_z={cur_rod[2]:.4f}  "
                f"ridge_top={self._ridge_top_z:.4f}"
            )
        else:
            print(f"  [{self.step_in_phase:4d}]  {self.phase} → {nxt}")
        self.phase = nxt
        self.step_in_phase = 0
        self._tracker.reset()
        if nxt == "DONE":
            self.done = True

    def _ori_action(self, eef_quat, target_quat):
        ori_err = _orientation_error(eef_quat, target_quat)
        return np.clip(ori_err / 0.5 * GAIN_ORI, -1.0, 1.0)

    def _dispatch(self, eef_pos, eef_quat, rod_pos, gripper_qpos):
        # Three orientation regimes:
        #   pre-roll  (top-grasp default, gripper-z = world -Z)
        #   roll      (side-grasp,        gripper-z = world +Y)
        #   post-roll (upright wrist roll about gripper-z, rod hangs vertical)
        if self.phase in ("INIT", "APPROACH", "DESCEND", "GRASP", "LIFT"):
            target_quat = self._eef_init_quat
        elif self.phase == "ROLL_TO_SIDE":
            target_quat = self._eef_side_grasp_quat
        else:
            target_quat = self._eef_upright_quat
        ori_action = self._ori_action(eef_quat, target_quat)

        # ── INIT ─────────────────────────────────────────────────────────────
        if self.phase == "INIT":
            if self.step_in_phase >= INIT_HOLD:
                self._advance("APPROACH")
            return _no_move(GRIP_OPEN)

        # Grasp point: inset from the rod's +X end toward its centre.  Each
        # retry shifts the inset further so the second/third attempt grips
        # more of the rod's body — gives the fingers something to land on if
        # the prior attempt closed past the end.
        retry_inset_bump = GRASP_RETRY_INSET_BUMP * self._grasp_attempts
        grasp_xy = np.array([
            rod_pos[0] + (self._rod_half_len - self._grasp_inset - retry_inset_bump),
            rod_pos[1],
        ])

        # ── APPROACH ─────────────────────────────────────────────────────────
        if self.phase == "APPROACH":
            hover_z = rod_pos[2] + HOVER_ABOVE_ROD
            target  = np.array([grasp_xy[0], grasp_xy[1], hover_z])
            if not self._tracker.loaded:
                self._tracker.init(eef_pos, target, N_WP_APPROACH)
            wp = self._tracker.tick(eef_pos)
            delta = wp - eef_pos
            dist = np.linalg.norm(delta)
            gain = GAIN_FAST if dist > 0.12 else GAIN_XY
            xy_err = np.linalg.norm(eef_pos[:2] - target[:2])
            z_err  = abs(eef_pos[2] - target[2])
            if (self._tracker.finished
                    and xy_err < XY_TOL and z_err < Z_TOL) \
                    or self.step_in_phase >= APPROACH_BUDGET:
                self._advance("DESCEND")
            return _clip_action(
                delta[0]*gain, delta[1]*gain, delta[2]*gain,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_OPEN,
            )

        # ── DESCEND ──────────────────────────────────────────────────────────
        if self.phase == "DESCEND":
            grasp_target = np.array([grasp_xy[0], grasp_xy[1], rod_pos[2] - GRASP_Z_PRESS])
            dx = (grasp_target[0] - eef_pos[0]) * GAIN_XY
            dy = (grasp_target[1] - eef_pos[1]) * GAIN_XY
            dz = (grasp_target[2] - eef_pos[2]) * GAIN_Z
            xy_err = np.linalg.norm(eef_pos[:2] - grasp_target[:2])
            z_err  = abs(eef_pos[2] - grasp_target[2])
            if (z_err < Z_TOL and xy_err < XY_TOL) or self.step_in_phase >= DESCEND_BUDGET:
                self._rod_pos_at_grasp = rod_pos.copy()
                self._advance("GRASP")
            return _clip_action(
                dx, dy, dz,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_OPEN,
            )

        # ── GRASP ────────────────────────────────────────────────────────────
        if self.phase == "GRASP":
            target_z = self._rod_pos_at_grasp[2] - GRASP_Z_PRESS
            dz = max((target_z - eef_pos[2]) * GAIN_Z, -0.2)
            # Gate the LIFT transition on the gripper actually closing — track
            # finger aperture (qpos[0] - qpos[1], positive when open) and only
            # advance once it has stabilised across a few consecutive steps.
            # Otherwise LIFT can begin while fingers are still mid-closure and
            # the rod gets dragged, not picked up.
            aperture = float(gripper_qpos[0] - gripper_qpos[1])
            self._grasp_aperture_hist.append(aperture)
            if len(self._grasp_aperture_hist) > GRASP_STABLE_WINDOW:
                self._grasp_aperture_hist = self._grasp_aperture_hist[-GRASP_STABLE_WINDOW:]
            window = self._grasp_aperture_hist
            closure_stable = (
                len(window) >= GRASP_STABLE_WINDOW
                and (max(window) - min(window)) < GRASP_STABLE_TOL
            )
            min_hold_done = self.step_in_phase >= GRASP_HOLD
            timed_out     = self.step_in_phase >= GRASP_CLOSURE_BUDGET
            if (min_hold_done and closure_stable) or timed_out:
                self._grasp_aperture_hist = []
                self._advance("LIFT")
            return _clip_action(
                0, 0, dz,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_CLOSE,
            )

        # ── LIFT ─────────────────────────────────────────────────────────────
        if self.phase == "LIFT":
            tz = self._gripper_z_at_lift
            dz = (tz - eef_pos[2]) * GAIN_Z
            if eef_pos[2] > tz - 0.015 or self.step_in_phase >= LIFT_BUDGET:
                # Verify the grasp held: the rod should have lifted with the
                # gripper.  If rod_z is still near table height the fingers
                # closed on air or slipped — drop and retry from APPROACH.
                rod_lifted = rod_pos[2] > self._table_z + GRASP_LIFT_MIN_HEIGHT
                if not rod_lifted and self._grasp_attempts + 1 < MAX_GRASP_ATTEMPTS:
                    self._grasp_attempts += 1
                    logger.warning(
                        "LIFT done but rod_z=%.3f still near table_z=%.3f — "
                        "reattempting grasp (%d/%d)",
                        rod_pos[2], self._table_z,
                        self._grasp_attempts + 1, MAX_GRASP_ATTEMPTS,
                    )
                    self._rod_pos_at_grasp = None
                    self._advance("APPROACH")
                    return _clip_action(
                        0, 0, 0,
                        ori_action[0], ori_action[1], ori_action[2],
                        GRIP_OPEN,  # release before retrying
                    )
                # Cache XY so ROLL_TO_SIDE and ROTATE can hold position while
                # the orientation controller does its work.
                self._rotate_xy = eef_pos[:2].copy()
                self._advance("ROLL_TO_SIDE")
            return _clip_action(
                0, 0, dz,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_CLOSE,
            )

        # ── ROLL_TO_SIDE ─────────────────────────────────────────────────────
        # Hold position; rotate +90° about world X so gripper-z swings from
        # world -Z to world +Y.  The rod lies along the rotation axis (world
        # X), so it does not move — only the fingers slide 90° around the
        # cylinder.  The wrist sweep stays in the y-z plane and never points
        # along ±X (the Panda elbow-blowup region at low/mid z).
        if self.phase == "ROLL_TO_SIDE":
            assert self._rotate_xy is not None
            tz = self._gripper_z_at_lift
            dx = (self._rotate_xy[0] - eef_pos[0]) * GAIN_XY
            dy = (self._rotate_xy[1] - eef_pos[1]) * GAIN_XY
            dz = (tz - eef_pos[2]) * GAIN_Z
            ori_err_norm = float(np.linalg.norm(
                _orientation_error(eef_quat, target_quat)
            ))
            if ori_err_norm < ORI_TOL or self.step_in_phase >= ROLL_BUDGET:
                self._advance("ROTATE")
            return _clip_action(
                dx, dy, dz,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_CLOSE,
            )

        # ── ROTATE ───────────────────────────────────────────────────────────
        # Pure wrist roll about gripper-z (= world +Y): the wrist forward-axis
        # stays put while the rod axis sweeps from world +X to world +Z.  Rod
        # ends up vertical with its bulk hanging below the gripper (we grip
        # near the +X end).  Carry height was chosen so the swept arc clears
        # the ridge once we move over.
        if self.phase == "ROTATE":
            assert self._rotate_xy is not None
            tz = self._gripper_z_at_lift
            dx = (self._rotate_xy[0] - eef_pos[0]) * GAIN_XY
            dy = (self._rotate_xy[1] - eef_pos[1]) * GAIN_XY
            dz = (tz - eef_pos[2]) * GAIN_Z
            ori_err_norm = float(np.linalg.norm(
                _orientation_error(eef_quat, target_quat)
            ))
            if ori_err_norm < ORI_TOL or self.step_in_phase >= ROTATE_BUDGET:
                self._advance("MOVE_OVER")
            return _clip_action(
                dx, dy, dz,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_CLOSE,
            )

        # ── MOVE_OVER ────────────────────────────────────────────────────────
        if self.phase == "MOVE_OVER":
            target = np.array([
                self._platform_xy[0],
                self._platform_xy[1],
                self._gripper_z_at_lift,
            ])
            if not self._tracker.loaded:
                self._tracker.init(eef_pos, target, N_WP_MOVE_OVER, z_lift=ARC_Z_LIFT)
            wp = self._tracker.tick(eef_pos)
            delta = wp - eef_pos
            dist  = np.linalg.norm(delta)
            gain  = GAIN_FAST if dist > 0.12 else GAIN_XY
            xy_err = np.linalg.norm(eef_pos[:2] - target[:2])
            if (self._tracker.finished and xy_err < XY_TOL) \
                    or self.step_in_phase >= MOVE_BUDGET:
                self._advance("LOWER")
            return _clip_action(
                delta[0]*gain, delta[1]*gain, delta[2]*gain,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_CLOSE,
            )

        # ── LOWER ────────────────────────────────────────────────────────────
        # With the rod hanging vertical below the gripper, drive eef_z down
        # so that rod_bottom = eef_z - rod_long_grasped lands just above the
        # ridge top.  XY: hold over the ridge centre.
        if self.phase == "LOWER":
            target_eef_z = self._ridge_top_z + self._rod_long_grasped + PLACE_Z_MARGIN
            target_xy = np.array([
                self._platform_xy[0] + PLACE_X_BIAS,
                self._platform_xy[1],
            ])
            dx = (target_xy[0] - eef_pos[0]) * GAIN_XY
            dy = (target_xy[1] - eef_pos[1]) * GAIN_XY
            dz = (target_eef_z - eef_pos[2]) * GAIN_LOWER_Z
            xy_err = float(np.linalg.norm(eef_pos[:2] - target_xy))
            z_err  = abs(eef_pos[2] - target_eef_z)
            if self.step_in_phase % 25 == 0:
                rod_bottom = eef_pos[2] - self._rod_long_grasped
                print(
                    f"      LOWER step {self.step_in_phase:3d}  "
                    f"eef_z={eef_pos[2]:.4f}  target={target_eef_z:.4f}  "
                    f"rod_bottom={rod_bottom:.4f}  ridge_top={self._ridge_top_z:.4f}  "
                    f"z_err={z_err:.4f}  xy_err={xy_err:.4f}  "
                    f"action_dz={np.clip(dz, -1, 1):+.3f}"
                )
            if (xy_err < XY_TOL and z_err < LOWER_Z_TOL) \
                    or self.step_in_phase >= LOWER_BUDGET:
                self._advance("SETTLE")
            return _clip_action(
                dx, dy, dz,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_CLOSE,
            )

        # ── SETTLE ───────────────────────────────────────────────────────────
        if self.phase == "SETTLE":
            if self.step_in_phase >= SETTLE_HOLD:
                self._advance("RELEASE")
            return _clip_action(
                0, 0, 0,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_CLOSE,
            )

        # ── RELEASE ──────────────────────────────────────────────────────────
        if self.phase == "RELEASE":
            if self.step_in_phase >= RELEASE_HOLD:
                self._advance("RETRACT")
            return _clip_action(
                0, 0, 0,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_OPEN,
            )

        # ── RETRACT ──────────────────────────────────────────────────────────
        # Lift straight up past the rod's top end so the open jaws clear it.
        if self.phase == "RETRACT":
            target_gz = (
                self._ridge_top_z + self._rod_long_grasped + RELEASE_LIFT
            )
            dz = (target_gz - eef_pos[2]) * GAIN_Z
            if eef_pos[2] > target_gz - 0.01 \
                    or self.step_in_phase >= RETRACT_BUDGET:
                self._advance("HOLD")
            return _clip_action(
                0, 0, dz,
                ori_action[0], ori_action[1], ori_action[2],
                GRIP_OPEN,
            )

        # ── HOLD ─────────────────────────────────────────────────────────────
        # Stay still while the env's balance counter accumulates to the threshold.
        if self.phase == "HOLD":
            if self.step_in_phase >= HOLD_STEPS:
                self._advance("DONE")
            return _no_move(GRIP_OPEN)

        return _no_move(GRIP_OPEN)


# ── Collection ───────────────────────────────────────────────────────────────

def _make_policy(env, obs: dict, rng: np.random.Generator) -> BalanceHardPolicy:
    policy = BalanceHardPolicy(env)
    policy.reset(obs)
    return policy


if __name__ == "__main__":
    run_task_cli(
        TASK_ID, _make_policy,
        description=f"Scripted demos for {TASK_ID}",
        epilog=__doc__,
        stop_when_policy_done=True,
        generation_method="scripted_motion",
        log_prefix="mp_balance_hard",
    )
