"""Scripted pick-and-lift policy for ``ib-m-mirror-pick-place-v1``.

A small cube spawns at a randomized xy AND yaw on the table; the task
rewards lifting the cube centre ``lift_height`` (default 0.08 m) above
its resting height, i.e. 0.10 m above the table top. Robosuite's
default ``UniformRandomSampler`` randomizes yaw in ``[0, 2pi)`` since
the task config doesn't override ``rotation``.

The task samples a horizontal, vertical or combined image flip once per
episode for every camera; there is no unmirrored view. Proprio (eef pose, joint state) and
``object_pos`` / ``object_quat`` are NOT flipped --
they remain in the true world frame -- so this scripted policy
operates in real-world coordinates and ignores the visual mirroring.

The cube has 4-fold rotational symmetry about world Z, so we wrap the
target gripper yaw to ``[-pi/4, pi/4]`` -- the shortest rotation that
aligns the jaws with one pair of cube faces.

Phase machine:
  settle -> pre_grasp -> descend -> close -> lift -> done
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw -- gripper Z down

PRE_GRASP_HEIGHT_ABOVE_TBL = 0.18   # hover here before descent
GRASP_Z_ABOVE_TBL          = 0.012  # cube half-size (0.02/2) + skim
LIFT_HEIGHT_ABOVE_TBL      = 0.18   # comfortable margin over 0.10 success threshold

# ── Phase budgets ────────────────────────────────────────────────────
# Nominal *fractions* of the episode horizon. They sum to 1.01 and get
# multiplied by the horizon sampled at reset() to produce per-phase
# tick budgets. ``CLOSE_FRAC`` covers a ~80-tick gripper-closure window
# at the mean horizon (800 ticks); it scales with the budget like
# the others, which is fine since slower budgets just give the jaws
# more time to settle.
SETTLE_FRAC     = 0.2   # ~160 ticks at horizon=800
PRE_GRASP_FRAC  = 0.25   # ~200
DESCEND_FRAC    = 0.21   # ~168
CLOSE_FRAC      = 0.1   # ~80
LIFT_FRAC       = 0.25   # ~200
# There is no slack: the budgets are upper bounds and phases normally
# advance early on convergence.

# Episode horizon distribution. Sampled once per ``reset()``.
HORIZON_MEAN = 800
HORIZON_STD  = 100
HORIZON_MIN  = 500       # floor: below this we can't reliably finish
HORIZON_MAX  = 1000      # matches task max_episode_steps; ~2.3% of
                         # N(800, 100) draws clip at the cap

# ── Tolerances ──────────────────────────────────────────────────────
POS_TOL_XY = 0.010
POS_TOL_Z  = 0.008
ORI_TOL    = 0.10

# ── Position-control gains ──────────────────────────────────────────
GAIN_XY  = 2.5
GAIN_Z   = 2.5
ROT_GAIN = 1.8
MAX_ACTION_STEP = 0.35

# Setpoint-change ramp (smoothstep gain on freshly published targets).
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


def _grasp_quat_for_yaw(world_yaw: float) -> np.ndarray:
    """Top-down grasp quat with the gripper rotated by ``world_yaw``
    about world Z (xyzw)."""
    half = 0.5 * world_yaw
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


def _cube_world_yaw(quat_xyzw: np.ndarray) -> float:
    """Extract yaw about world Z from an xyzw quaternion. Project the
    body-X axis onto world XY and take its angle."""
    x, y, z, w = quat_xyzw
    bx_x = 1.0 - 2.0 * (y * y + z * z)
    bx_y = 2.0 * (x * y + z * w)
    return float(np.arctan2(bx_y, bx_x))


def _shortest_grasp_yaw(cube_yaw: float) -> float:
    """Wrap to ``[-pi/4, pi/4]`` -- the cube has 4-fold symmetry, so
    grasping along yaw, yaw+pi/2, yaw+pi, yaw+3pi/2 are all equivalent.
    Pick the representative closest to zero."""
    y = (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
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

class MirrorPickPlacePolicy(PolicyInterface):
    """Scripted pick-and-lift for the mirror task. Visual flipping is a
    perception challenge for learned policies; this scripted policy
    works from un-flipped state, so it just executes a straight grasp."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8
        self._lift_target_z: float = self._table_z + LIFT_HEIGHT_ABOVE_TBL

        # Sample this episode's horizon and slice it into per-phase
        # budgets. Budgets are caps -- a phase advances early on
        # tolerance, so a generous horizon doesn't slow successful
        # episodes; it just leaves more headroom for hard cases.
        horizon = float(self.rng.normal(HORIZON_MEAN, HORIZON_STD))
        self.horizon = int(np.clip(round(horizon), HORIZON_MIN, HORIZON_MAX))
        self.budgets = {
            "settle":    max(1, int(round(SETTLE_FRAC    * self.horizon))),
            "pre_grasp": max(1, int(round(PRE_GRASP_FRAC * self.horizon))),
            "descend":   max(1, int(round(DESCEND_FRAC   * self.horizon))),
            "close":     max(1, int(round(CLOSE_FRAC     * self.horizon))),
            "lift":      max(1, int(round(LIFT_FRAC      * self.horizon))),
        }
        print(f"  [horizon] sampled={self.horizon}  budgets={self.budgets}",
              flush=True)

        # Cube xy + grasp quat locked at end of settle so descent doesn't
        # chase contact-induced jitter.
        self._cube_xy: np.ndarray | None = None
        self._grasp_quat: np.ndarray = TOP_DOWN_QUAT.copy()

        # Setpoint-ramp state.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

    def _advance(self, next_phase: str):
        print(f"  [{self.step_in_phase:4d} ticks] {self.phase} -> {next_phase}")
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

    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0]))
        cube_pos = obs.get("object_pos", obs.get("cube_pos"))
        cube_quat = obs.get("object_quat", obs.get("cube_quat"))

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))
            self._lift_target_z = self._table_z + LIFT_HEIGHT_ABOVE_TBL

        # ── 1. Settle: let initial transients dissipate ───────────────
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = -1.0
            if self.step_in_phase >= self.budgets["settle"]:
                self._cube_xy = cube_pos[:2].copy()
                if cube_quat is not None:
                    cube_yaw = _cube_world_yaw(np.asarray(cube_quat))
                    grip_yaw = _shortest_grasp_yaw(cube_yaw)
                    self._grasp_quat = _grasp_quat_for_yaw(grip_yaw)
                    print(f"  [grasp] cube_yaw={np.degrees(cube_yaw):+.1f} deg "
                          f"-> grip_yaw={np.degrees(grip_yaw):+.1f} deg")
                self._advance("pre_grasp")
            self.step_in_phase += 1
            return action

        # ── 2. Pre-grasp: hover above the cube with grasp orientation ─
        if self.phase == "pre_grasp":
            target = np.array([
                self._cube_xy[0],
                self._cube_xy[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            xy_ok  = float(np.linalg.norm(target[:2] - eef_pos[:2])) < POS_TOL_XY
            z_ok   = abs(target[2] - eef_pos[2]) < POS_TOL_Z
            ori_ok = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, self._grasp_quat))) < ORI_TOL
            if (xy_ok and z_ok and ori_ok) \
                    or self.step_in_phase >= self.budgets["pre_grasp"]:
                self._advance("descend")
            self.step_in_phase += 1
            return action

        # ── 3. Descend straight down to grasp height ──────────────────
        if self.phase == "descend":
            target = np.array([
                self._cube_xy[0],
                self._cube_xy[1],
                self._table_z + GRASP_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_z=6.0, target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.005
            if reached or self.step_in_phase >= self.budgets["descend"]:
                self._advance("close_gripper")
            self.step_in_phase += 1
            return action

        # ── 4. Close gripper, hold pose ───────────────────────────────
        if self.phase == "close_gripper":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase >= self.budgets["close"]:
                self._advance("lift")
            self.step_in_phase += 1
            return action

        # ── 5. Lift straight up past the success threshold ────────────
        if self.phase == "lift":
            target = np.array([eef_pos[0], eef_pos[1], self._lift_target_z])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.010
            if reached or self.step_in_phase >= self.budgets["lift"]:
                self._advance("done")
            self.step_in_phase += 1
            return action

        # ── 6. Done: keep gripper closed and hold the lift altitude ───
        target = np.array([eef_pos[0], eef_pos[1], self._lift_target_z])
        action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                               target_quat=self._grasp_quat)
        self.step_in_phase += 1
        return action


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5,
                        help="number of episodes to run (default: 5)")
    parser.add_argument("--seed", type=int, default=3,
                        help="base seed; episode i uses seed = base + i*17 (default: 3)")
    parser.add_argument("--no-render", action="store_true",
                        help="disable the on-screen MuJoCo viewer")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env    = ib.make("ib-m-mirror-pick-place-v1", privileged_obs=True, render_mode=render_mode)
    policy = MirrorPickPlacePolicy()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 17)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===  "
              f"cube_pos={np.round(obs.get('object_pos', obs.get('cube_pos')), 3)}",
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
