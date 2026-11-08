"""Scripted policy for ``ib-p3-gap-funnel-v1``.

Strategy
--------
A medium sphere sits on the robot's side of the table. A static barrier
strip runs across the table at ``barrier_x`` with a single narrow gap
centred at ``gap_center_y``. A green goal zone covers the far side of
the barrier.

The policy reads the gap centre and ball position from the env and
constructs a straight push line that goes ``ball -> gap_center``.
The eef is positioned BEHIND the ball along that line and driven
forward with the gripper held OPEN so its two fingers act as a fork
around the ball -- the ball is kept on the push line and can't squirt
laterally even with the env's 2x velocity amplification on every
gripper-ball contact tick. The fork's closing axis is rotated so the
fingers straddle the push line (perpendicular to push_dir).

The eef stops short of the barrier (``barrier_x - PUSH_END_BACKOFF``)
so it doesn't collide with the wall segments. Whatever momentum the
ball has at that point carries it the rest of the way.

Phase machine:
  settle
  -> plan                (reads ball xy, gap xy; builds push line)
  -> align_behind_ball   (eef hovers above ball - standoff*push_dir)
  -> descend             (drop to push_z = table + ball_radius)
  -> push                (drive forward to push_end_xy, broadside hit)
  -> follow_through      (continue past nominal end so ball clears gap)
  -> retract_up          (rise, evaluate)
  -> done | replan       (one replan attempt if ball not yet in zone)
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw -- gripper Z down

# Vertical setpoints (above table top).
HOVER_Z_ABOVE_TBL    = 0.18    # safe altitude before descending behind ball
PUSH_Z_ABOVE_TBL     = 0.045   # eef target during strike. Set to roughly
                               # ball_radius (3 cm) + small clearance so
                               # the closed-jaw flat hits the ball at its
                               # equator, broadside. Lower targets have
                               # the wrist nose-diving into the table
                               # under the SHOCK gain.
PUSH_Z_FLOOR_OFFSET  = 0.020   # hard Z floor above table — the push
                               # action will never command the EEF below
                               # table_z + this, regardless of OSC
                               # redundancy resolution under high gain.
RETRACT_Z_ABOVE_TBL  = 0.20    # final clearance height

# Push geometry.
PUSH_STANDOFF      = 0.05      # how far behind the ball to start the push
PUSH_END_BACKOFF   = 0.03      # how far short of the barrier to stop the eef
FOLLOW_THROUGH_M   = 0.04      # extra metres past push_end_xy on follow_through

# Tolerances.
POS_TOL_XY = 0.010
POS_TOL_Z  = 0.008
ORI_TOL    = 0.10

# Position-control gains.
GAIN_XY   = 4.0
GAIN_Z    = 4.0
ROT_GAIN  = 2.5
MAX_ACTION_STEP = 0.6
PUSH_GAIN_XY = 12.0  # SHOCK strike — high gain so the OSC commands
                     # peak forward velocity from tick 1 and the
                     # gripper smashes into the ball with momentum.
PUSH_MAX_STEP = 1.0  # full-scale action (vs. the default 0.6) so the
                     # OSC runs at strike speed.

# Setpoint-change ramp.
RAMP_TICKS        = 20
RAMP_START_FRAC   = 0.2
TARGET_CHANGE_TOL = 0.015

# Phase budgets (ticks, episode horizon 1000). The OSC's effective
# Cartesian speed is ~8 cm/s, so a 50-60 cm push along the push line
# (worst-case spawn at base+0.6 m) consumes ~700 ticks.
SETTLE_STEPS         = 10
APPROACH_MAX_STEPS   = 320     # combined xy+z to behind-ball pose with
                               # ori convergence (wrist may rotate ~75°)
PUSH_MAX_STEPS       = 500     # 50-60 cm along push line; keeps contact long
FOLLOW_MAX_STEPS     = 50
RETRACT_MAX_STEPS    = 180     # need ~18 cm vertical lift at OSC speed
REPLAN_LIMIT         = 0       # one shot — fits in 1000 ticks


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
    half = 0.5 * world_yaw
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


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

class GapFunnelPolicy(PolicyInterface):
    """Push a sphere through a single gap in a barrier into a goal zone."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8
        self._ball_radius: float = 0.03
        self._barrier_x: float = 0.15
        self._gap_center_y: float = 0.0

        self._goal_x_min: float = 0.0
        self._goal_x_max: float = 0.0
        self._goal_y_min: float = 0.0
        self._goal_y_max: float = 0.0

        # Plan-time state.
        self._push_dir: np.ndarray | None = None        # unit vec in xy
        self._start_xy: np.ndarray | None = None        # eef hover position
        self._push_end_xy: np.ndarray | None = None     # eef stop (short of barrier)
        self._follow_xy: np.ndarray | None = None       # follow-through target
        self._push_quat: np.ndarray = TOP_DOWN_QUAT.copy()

        self._replans: int = 0
        self._push_start_ball_xy: np.ndarray | None = None

        # Setpoint-ramp state.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

    # ── Phase helpers ────────────────────────────────────────────────
    def _advance(self, next_phase: str):
        print(f"  [{self.step_in_phase:4d} ticks] {self.phase} -> {next_phase}",
              flush=True)
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

    # ── Plan helpers ─────────────────────────────────────────────────
    def _read_env_geometry(self, env):
        self._table_z = float(getattr(env, "_table_height", self._table_z))
        self._ball_radius = float(getattr(env, "_ball_radius", self._ball_radius))
        self._barrier_x = float(getattr(env, "_barrier_x", self._barrier_x))
        self._gap_center_y = float(getattr(env, "_gap_center_y", self._gap_center_y))
        self._goal_x_min = float(getattr(env, "_goal_x_min", self._goal_x_min))
        self._goal_x_max = float(getattr(env, "_goal_x_max", self._goal_x_max))
        self._goal_y_min = float(getattr(env, "_goal_y_min", self._goal_y_min))
        self._goal_y_max = float(getattr(env, "_goal_y_max", self._goal_y_max))

    def _build_push_plan(self, ball_xy: np.ndarray):
        """Compute push_dir (ball -> gap_center), eef start/end positions,
        and a gripper yaw so the closed-jaw flat is normal to push_dir."""
        gap_xy = np.array([self._barrier_x, self._gap_center_y])
        push_vec = gap_xy - ball_xy
        norm = float(np.linalg.norm(push_vec))
        if norm < 1e-6:
            self._push_dir = np.array([1.0, 0.0])
        else:
            self._push_dir = push_vec / norm

        # Hover behind the ball along -push_dir, at safe Z.
        self._start_xy = ball_xy - self._push_dir * PUSH_STANDOFF

        # Stop the eef short of the barrier so it doesn't hit a wall
        # segment. Compute scalar t along push_dir from ball to gap_xy
        # such that eef.x = barrier_x - PUSH_END_BACKOFF.
        end_x = self._barrier_x - PUSH_END_BACKOFF
        if abs(self._push_dir[0]) > 1e-6:
            t = (end_x - ball_xy[0]) / self._push_dir[0]
        else:
            t = norm
        # Clamp t so we always go forward but don't overshoot the gap.
        t = max(0.05, min(t, norm + 0.10))
        self._push_end_xy = ball_xy + self._push_dir * t

        self._follow_xy = self._push_end_xy + self._push_dir * FOLLOW_THROUGH_M

        # Yaw the gripper so its closing axis (gripper-local Y, the
        # normal of the closed finger-pad flat face) aligns with
        # push_dir. The closed jaws form a flat plate parallel to
        # gripper-XZ; that flat face's outward normal is local +Y.
        # Pointing it along push_dir means the eef hits the ball
        # BROADSIDE with the flat -- not with the rounded edge of the
        # jaw, which deflects the ball laterally.
        push_dir_yaw = float(np.arctan2(self._push_dir[1], self._push_dir[0]))
        gripper_yaw = push_dir_yaw + np.pi / 2
        # Wrap to [-pi/2, pi/2] for shortest wrist twist (parallel-jaw is
        # symmetric under +pi rotation about its own Z, so equivalent).
        gripper_yaw = (gripper_yaw + np.pi / 2) % np.pi - np.pi / 2
        self._push_quat = _grasp_quat_for_yaw(gripper_yaw)

        print(f"  [plan] ball_xy={np.round(ball_xy, 3)}  "
              f"gap_xy={np.round(gap_xy, 3)}  "
              f"dir={np.round(self._push_dir, 3)}  "
              f"start_xy={np.round(self._start_xy, 3)}  "
              f"end_xy={np.round(self._push_end_xy, 3)}  "
              f"grip_yaw={np.degrees(gripper_yaw):+.1f}deg",
              flush=True)

    def _ball_in_zone(self, ball_xy: np.ndarray) -> bool:
        return (
            self._goal_x_min <= ball_xy[0] <= self._goal_x_max
            and self._goal_y_min <= ball_xy[1] <= self._goal_y_max
        )

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0.0, 0.0, 0.0, 1.0])))
        ball_pos = np.asarray(
            obs.get("object_pos", obs.get("cube_pos", np.zeros(3))),
            dtype=float,
        )

        if env is not None:
            self._read_env_geometry(env)

        # ── 1. Settle ─────────────────────────────────────────────────
        # Gripper held CLOSED throughout — closed jaws form a flat
        # plate broader than the gap-aligned face, hitting the ball
        # broadside. Open fingers (8 cm spread) are wider than the
        # 6 cm ball, so the ball would pass between them.
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = 1.0
            if self.step_in_phase >= SETTLE_STEPS:
                self._advance("plan")
            self.step_in_phase += 1
            return action

        # ── 2. Plan ───────────────────────────────────────────────────
        if self.phase == "plan":
            self._build_push_plan(ball_pos[:2].copy())
            self._advance("approach")
            action = np.zeros(7)
            action[6] = 1.0
            return action

        # ── 3. Approach: combined xy + z to (start_xy, push_z) ───────
        # Diagonal trajectory from current pose to behind-ball pose at
        # push altitude. Splitting into hover + descend wasted ticks
        # fighting the OSC's workspace floor on the pure-z drop. The
        # diagonal path glides into the valid pose naturally.
        if self.phase == "approach":
            target = np.array([
                self._start_xy[0],
                self._start_xy[1],
                self._table_z + PUSH_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=GAIN_XY, gain_z=GAIN_Z,
                                   target_quat=self._push_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_err = abs(target[2] - eef_pos[2])
            ori_err = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, self._push_quat)))
            if (xy_err < POS_TOL_XY and z_err < POS_TOL_Z and ori_err < ORI_TOL) \
                    or self.step_in_phase >= APPROACH_MAX_STEPS:
                print(f"  [approach] xy_err={xy_err:.3f}  z_err={z_err:.3f}  "
                      f"ori_err={ori_err:.3f}  "
                      f"eef_z=table+{eef_pos[2]-self._table_z:.3f}",
                      flush=True)
                self._advance("push")
            self.step_in_phase += 1
            return action

        # ── 5. Push: shock strike then GET OUT OF THE WAY. Drive the
        # gripper hard at a target ahead of the ball; as soon as the
        # ball has clearly moved (the strike connected), exit so the
        # gripper retracts and the ball rolls toward the gap unobstructed.
        if self.phase == "push":
            cur_ball_xy = ball_pos[:2].copy()
            # Latch the ball's xy at push start (first entry to push)
            # so we can tell once it has moved.
            if self._push_start_ball_xy is None:
                self._push_start_ball_xy = cur_ball_xy.copy()
            gap_xy = np.array([self._barrier_x, self._gap_center_y])
            push_vec = gap_xy - cur_ball_xy
            push_norm = float(np.linalg.norm(push_vec))
            if push_norm < 1e-6:
                cur_push_dir = self._push_dir
            else:
                cur_push_dir = push_vec / push_norm
            # Target eef pose: 8 cm AHEAD of the ball along the live
            # push direction. Big enough chase distance for the OSC to
            # actually drive forward into the ball with force, not sit
            # there matching ball position.
            push_target_xy = cur_ball_xy + cur_push_dir * 0.08
            # Hold strike altitude throughout the push — any lift
            # mid-push loses contact and the ball doesn't get the full
            # impulse. Takeoff happens slowly in follow_through, after
            # the ball has been driven forward.
            push_z = PUSH_Z_ABOVE_TBL
            target = np.array([
                push_target_xy[0],
                push_target_xy[1],
                self._table_z + push_z,
            ])
            # Wrist orientation stays at the initial push_quat — letting
            # it re-yaw every tick made the OSC orientation controller
            # chase a moving target and never converge.
            #
            # Skip the setpoint ramp during the push so the action
            # saturates from tick 1 — we want a SHOCK impulse, not a
            # gentle ease-in. Build action manually instead of going
            # through _move_to (which always applies the ramp).
            delta = target - eef_pos
            action = np.zeros(7)
            action[0] = float(np.clip(delta[0] * PUSH_GAIN_XY,
                                      -PUSH_MAX_STEP, PUSH_MAX_STEP))
            action[1] = float(np.clip(delta[1] * PUSH_GAIN_XY,
                                      -PUSH_MAX_STEP, PUSH_MAX_STEP))
            # Hard Z floor: never command the EEF below table+floor_offset
            # during the strike. Under PUSH_GAIN_XY=12 + PUSH_MAX_STEP=1.0,
            # OSC redundancy occasionally resolves into a downward dip,
            # pinning the ball into the table. If the EEF is at or below
            # the floor, zero any down-command and let any up-command
            # through normally.
            z_floor = self._table_z + PUSH_Z_FLOOR_OFFSET
            dz_cmd = float(np.clip(delta[2] * GAIN_Z,
                                   -MAX_ACTION_STEP, MAX_ACTION_STEP))
            if eef_pos[2] <= z_floor and dz_cmd < 0.0:
                dz_cmd = 0.0
            action[2] = dz_cmd
            ori_err = _quat_err_axis_angle(eef_quat, self._push_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
            action[6] = 1.0
            # Early exit: as soon as the ball has clearly moved from its
            # starting position the strike has connected and the ball
            # is now rolling on its own. Get out of the way so the
            # gripper isn't between the ball and the gap.
            ball_displacement = float(np.linalg.norm(
                cur_ball_xy - self._push_start_ball_xy))
            ball_moved = ball_displacement > 0.07   # 7 cm — well past noise
            ball_past_barrier = cur_ball_xy[0] >= self._barrier_x - 0.02
            if ball_past_barrier or ball_moved \
                    or self.step_in_phase >= PUSH_MAX_STEPS:
                print(f"  [push]    ball_xy={np.round(cur_ball_xy, 3)}  "
                      f"displacement={ball_displacement:.3f}  "
                      f"past_barrier={ball_past_barrier}",
                      flush=True)
                self._advance("follow_through")
            self.step_in_phase += 1
            return action

        # ── 6. Takeoff: lift the gripper STRAIGHT UP (xy locked) so it
        # gets out of the ball's path immediately after the strike.
        # No forward motion — the ball is now rolling, the gripper
        # only needs to clear vertical space above it.
        if self.phase == "follow_through":
            takeoff_progress = min(self.step_in_phase / float(FOLLOW_MAX_STEPS), 1.0)
            takeoff_z = PUSH_Z_ABOVE_TBL + takeoff_progress * 0.10
            target = np.array([
                eef_pos[0],     # xy locked at current pose — get UP
                eef_pos[1],
                self._table_z + takeoff_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=2.0, gain_z=4.0, max_step=0.4,
                                   target_quat=self._push_quat)
            if self.step_in_phase >= FOLLOW_MAX_STEPS:
                self._advance("retract_up")
            self.step_in_phase += 1
            return action

        # ── 7. Retract upward, decide done vs replan ─────────────────
        if self.phase == "retract_up":
            target = np.array([
                eef_pos[0],
                eef_pos[1],
                self._table_z + RETRACT_Z_ABOVE_TBL,
            ])
            # Smooth-ish lift — cap action below the default 0.6 so the
            # rise isn't a jerk, but not so slow that the gripper lingers
            # near the ball and drags it during retract.
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_z=4.0, max_step=0.5,
                                   target_quat=self._push_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.012
            if reached or self.step_in_phase >= RETRACT_MAX_STEPS:
                ball_xy = ball_pos[:2]
                in_zone = self._ball_in_zone(ball_xy)
                past_barrier = ball_xy[0] > self._goal_x_min
                print(f"  [retract] ball_xy={np.round(ball_xy, 3)}  "
                      f"in_zone={in_zone}  past_barrier={past_barrier}",
                      flush=True)
                if in_zone or self._replans >= REPLAN_LIMIT:
                    self._advance("done")
                else:
                    self._replans += 1
                    print(f"  [replan #{self._replans}] re-aligning behind ball",
                          flush=True)
                    self._advance("plan")
            self.step_in_phase += 1
            return action

        # ── 8. Done: hold pose, gripper closed ───────────────────────
        action = np.zeros(7)
        action[6] = 1.0
        ori_err = _quat_err_axis_angle(eef_quat, self._push_quat)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
        self.step_in_phase += 1
        return action


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = ib.make("ib-p3-gap-funnel-v1", privileged_obs=True, render_mode=render_mode)
    policy = GapFunnelPolicy()

    scores = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 17)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep}  seed={args.seed + ep * 17} ===  "
              f"ball_xy={np.round(obs.get('object_pos', obs.get('cube_pos'))[:2], 3)}  "
              f"gap_y={env._gap_center_y:.3f}",
              flush=True)
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            done = terminated or truncated
        score = float(info.get("task_score", 0.0))
        scores.append(score)
        print(f"Episode {ep}  score={score:.3f}  phase={policy.phase}  "
              f"replans={policy._replans}", flush=True)

    print(f"\nMean score over {args.episodes} eps: {np.mean(scores):.3f}",
          flush=True)
    env.close()
