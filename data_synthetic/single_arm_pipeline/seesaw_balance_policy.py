"""Scripted policy for ``ib-p3-seesaw-balance-v1``.

Strategy
--------
A red target cube sits on the +x side of a seesaw at a randomized arm
distance; a single blue counterweight cube sits on the table at -y.
Both masses are randomized per episode. The lever law gives a unique
placement that balances the plank:

    ideal_counter_arm = m_target * target_arm / m_counter

The env pre-computes this and stores it as ``env._ideal_counter_arm``
during reset, so the policy reads it directly rather than estimating
from observations.

Action plan: pick the blue cube on the table, fly over the seesaw at
high altitude (the -x plank end can sit ~10 cm above the table while
the +x side is tilted down), descend to a fixed safe altitude over
the placement xy, open the gripper, and retract STRAIGHT UP before
any horizontal motion -- otherwise the wrist would graze the rising
-x plank end as it levels out. Then move clear of the seesaw so the
gripper-seesaw contact budget (100 cumulative ticks per episode)
isn't burnt while we wait for the balance counter to fill.

Phase machine:
  settle
  -> align_above_weight    (xy over blue cube, z = table + PRE_GRASP, top-down)
  -> descend_to_weight     (z down to weight grasp height)
  -> close_gripper         (clamp the blue cube)
  -> lift_high             (rise to LIFT_HIGH_Z, clears tilted plank)
  -> translate_above_place (xy = (fulcrum_x - ideal_counter_arm, 0), z held high)
  -> descend_to_plank      (z down to PLACE_Z; safe above worst-case tilt)
  -> open_gripper          (release; cube drops onto plank)
  -> retract_up            (rise straight up; xy locked so wrist doesn't graze)
  -> retreat_aside         (move xy off the seesaw)
  -> wait_for_balance      (hold pose for the success counter to fill)
  -> done

Success criterion (from the task):
  |plank_tilt_deg| <= 5 for 200 consecutive ticks AND gripper not
  currently in contact with the counterweight (so we MUST release and
  back off, not hang on to the cube to "stabilise" the plank).
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw -- gripper Z down

# Vertical setpoints (above table top).
PRE_GRASP_HEIGHT_ABOVE_TBL = 0.18
# Cube full size 0.04 m -> top at table+0.04, center at table+0.02. Target
# the lower half of the cube so the fingertips definitely sit below the
# top face when they close. Mirrors what MirrorPickPlace does (0.012).
WEIGHT_GRASP_Z_ABOVE_TBL   = 0.014
LIFT_HIGH_Z_ABOVE_TBL      = 0.28    # clears -x end at ~10-12 cm tilt
# Placement target: cube CENTER z when the plank is exactly level.
#   plank_top_z (level) = table + fulcrum_height (0.04) + plank_thickness (0.015) = table + 0.055
#   cube center on plank top                              = table + 0.055 + 0.020 = table + 0.075
# We descend the gripped cube SLOWLY toward this z. The cube presses
# the plank down progressively (no impact, no bounce, no stiction lock
# at a tilted rest angle). The phase exits the moment the plank crosses
# level -- even if z hasn't quite reached target -- so we never push
# past level and tip the seesaw backwards.
PLACE_Z_ABOVE_TBL          = 0.075
# Press target is set this far BELOW the level-plank cube z so the
# OSC keeps commanding strong downward force throughout the press,
# instead of decaying to near-zero action (proportional to delta) as
# the eef approaches the level z. The plank-level exit (angle<1.5deg)
# stops the descent before the eef physically tries to drive the
# cube past level. Without this, the descent crawls and the gripper
# opens with the cube still hovering ~1.5 cm above the tilted plank.
PLACE_Z_PRESS_BELOW        = 0.05
RETRACT_HIGH_Z_ABOVE_TBL   = 0.30
RETREAT_XY                 = np.array([-0.20, -0.30])  # off seesaw, in -y


# ── Phase budgets (fractions of episode horizon) ────────────────────
# Same OSC speed limits as the cup task. Translate is the longest
# motion in this task: ~27 cm from the grasp xy (~-0.10, -0.25) to
# the lever-law placement (~0, 0) -- needs ~3 s at the OSC's ~10 cm/s
# effective speed. Descend-to-plank drops 19 cm too. Tighter budgets
# release the cube half-way to the placement xy.
SETTLE_FRAC          = 0.02
ALIGN_FRAC           = 0.18   # ~30 cm xy from rest to far-corner cube
DESCEND_WEIGHT_FRAC  = 0.13   # 16 cm z drop
STABILIZE_GRASP_FRAC = 0.04   # hold at grasp pose so xy converges <1 cm
CLOSE_FRAC           = 0.04
LIFT_HIGH_FRAC       = 0.04
STABILIZE_PLACE_FRAC = 0.04   # press-and-hold so plank settles at level
TRANSLATE_FRAC       = 0.14
DESCEND_PLANK_FRAC   = 0.13   # slow press: ~20 cm at gain_z=2.0
OPEN_FRAC            = 0.03
RETRACT_UP_FRAC      = 0.04
RETREAT_FRAC         = 0.04
WAIT_BALANCE_FRAC    = 0.13   # >= 200 ticks at horizon 1500+
# remainder is slack absorbed by the trailing "done" hold

HORIZON_MEAN = 1700
HORIZON_STD  = 80
HORIZON_MIN  = 1500
HORIZON_MAX  = 1800


# ── Tolerances ──────────────────────────────────────────────────────
POS_TOL_XY = 0.010
POS_TOL_Z  = 0.008
ORI_TOL    = 0.10


# ── Position-control gains ──────────────────────────────────────────
GAIN_XY  = 2.5
GAIN_Z   = 2.5
ROT_GAIN = 1.8
MAX_ACTION_STEP = 0.35

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


def _yaw_from_quat(q_xyzw: np.ndarray) -> float:
    x, y, z, w = q_xyzw
    bx_x = 1.0 - 2.0 * (y * y + z * z)
    bx_y = 2.0 * (x * y + z * w)
    return float(np.arctan2(bx_y, bx_x))


def _shortest_grasp_yaw(cube_yaw: float) -> float:
    """4-fold cube symmetry -> wrap to [-pi/4, pi/4]."""
    y = (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(y)


# ── Policy ──────────────────────────────────────────────────────────

class SeesawBalancePolicy(PolicyInterface):
    """Scripted lever-law placement solver for the seesaw balance task."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8

        # Sample horizon and slice into phase budgets.
        horizon = float(self.rng.normal(HORIZON_MEAN, HORIZON_STD))
        self.horizon = int(np.clip(round(horizon), HORIZON_MIN, HORIZON_MAX))
        self.budgets = {
            "settle":           max(1, int(round(SETTLE_FRAC          * self.horizon))),
            "align":            max(1, int(round(ALIGN_FRAC           * self.horizon))),
            "descend_weight":   max(1, int(round(DESCEND_WEIGHT_FRAC  * self.horizon))),
            "stabilize_grasp":  max(1, int(round(STABILIZE_GRASP_FRAC * self.horizon))),
            "close":            max(1, int(round(CLOSE_FRAC           * self.horizon))),
            "lift_high":        max(1, int(round(LIFT_HIGH_FRAC       * self.horizon))),
            "translate":        max(1, int(round(TRANSLATE_FRAC       * self.horizon))),
            "descend_plank":    max(1, int(round(DESCEND_PLANK_FRAC   * self.horizon))),
            "stabilize_place":  max(1, int(round(STABILIZE_PLACE_FRAC * self.horizon))),
            "open":             max(1, int(round(OPEN_FRAC            * self.horizon))),
            "retract_up":       max(1, int(round(RETRACT_UP_FRAC      * self.horizon))),
            "retreat":          max(1, int(round(RETREAT_FRAC         * self.horizon))),
            "wait_balance":     max(1, int(round(WAIT_BALANCE_FRAC    * self.horizon))),
        }
        print(f"  [horizon] sampled={self.horizon}  budgets={self.budgets}",
              flush=True)

        # Latched at end of settle from env / sim.
        self._weight_xy: np.ndarray | None  = None
        self._weight_yaw: float             = 0.0
        self._grasp_quat: np.ndarray        = TOP_DOWN_QUAT.copy()
        self._fulcrum_x: float              = 0.10
        self._ideal_counter_arm: float      = 0.0
        self._place_xy: np.ndarray | None   = None

        # Cube-to-TCP xy offset, measured AFTER lifting so the
        # placement target compensates for any in-grip slip from the
        # close. Without this, a 1-2 cm cube-vs-TCP offset becomes a
        # 10-20% lever-arm error and the plank never balances.
        self._cube_to_eef_xy: np.ndarray = np.zeros(2)
        self._eef_target_place_xy: np.ndarray | None = None

        # Latched at the moment descend_to_plank exits: the eef z
        # at which the plank reached level. stabilize_place holds
        # exactly here so we don't over-press past the balance point.
        self._stab_lock_z: float = 0.0
        self._press_was_level: bool = False

        # Setpoint-ramp state.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

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

    # ── Sim probes ───────────────────────────────────────────────────
    def _read_plank_angle_deg(self, env) -> float:
        """Plank tilt magnitude in degrees, from the seesaw_hinge joint
        qpos. Used to exit the descent the moment the plank levels:
        press until both cubes are at the same height, then stop. Returns inf if the joint isn't found
        (so we never accidentally exit on a missing reading)."""
        try:
            sim = env._env.sim
            jnt_id = sim.model.joint_name2id("seesaw_hinge")
            qpos_addr = sim.model.jnt_qposadr[jnt_id]
            return float(np.degrees(abs(sim.data.qpos[qpos_addr])))
        except Exception:
            return float("inf")

    def _read_weight_pose(self, env) -> tuple[np.ndarray, np.ndarray]:
        """Counterweight world (pos, quat-xyzw) from sim. The base
        env's ``object_pos`` points at the TARGET cube here, so we
        read the counterweight directly by name."""
        sim = env._env.sim
        bid = sim.model.body_name2id("weight_0_main")
        pos = np.asarray(sim.data.body_xpos[bid], dtype=float)
        quat_wxyz = np.asarray(sim.data.body_xquat[bid], dtype=float)  # mujoco wxyz
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos, quat_xyzw

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0.0, 0.0, 0.0, 1.0])))

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))

        # ── 1. Settle: latch positions and lever-law placement ───────
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = -1.0
            if self.step_in_phase >= self.budgets["settle"] and env is not None:
                w_pos, w_quat = self._read_weight_pose(env)
                self._weight_xy = w_pos[:2].copy()
                self._weight_yaw = _yaw_from_quat(w_quat)
                grip_yaw = _shortest_grasp_yaw(self._weight_yaw)
                self._grasp_quat = _grasp_quat_for_yaw(grip_yaw)

                self._fulcrum_x = float(getattr(env, "_fulcrum_x", 0.10))
                self._ideal_counter_arm = float(
                    getattr(env, "_ideal_counter_arm", 0.10))
                self._place_xy = np.array([
                    self._fulcrum_x - self._ideal_counter_arm,
                    0.0,
                ])

                tm = float(getattr(env, "_target_mass_actual", 0.0))
                cm = float(getattr(env, "_counter_mass_actual", 0.0))
                ta = float(getattr(env, "_target_arm", 0.0))
                print(f"  [plan] weight_xy={np.round(self._weight_xy, 3)}  "
                      f"grip_yaw={np.degrees(grip_yaw):+.1f}  "
                      f"fulcrum_x={self._fulcrum_x:+.3f}  "
                      f"ideal_arm={self._ideal_counter_arm:.3f}  "
                      f"place_xy={np.round(self._place_xy, 3)}  "
                      f"m_t={tm:.3f}  m_c={cm:.3f}  arm_t={ta:.3f}",
                      flush=True)
                self._advance("align_above_weight")
            self.step_in_phase += 1
            return action

        # ── 2. Align above blue cube ─────────────────────────────────
        if self.phase == "align_above_weight":
            target = np.array([
                self._weight_xy[0],
                self._weight_xy[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            ])
            # gain_xy=6 instead of 4 so the last few cm don't crawl --
            # at delta=0.03 m the action is 0.18 (vs. 0.12 with gain 4),
            # cutting the closing-distance time-to-tolerance roughly in
            # half. Far enough out the action saturates at max_step
            # either way, so this only affects the final approach.
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_xy=6.0, target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            xy_ok  = xy_err < POS_TOL_XY
            z_ok   = abs(target[2] - eef_pos[2]) < POS_TOL_Z
            ori_ok = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, self._grasp_quat))) < ORI_TOL
            if (xy_ok and z_ok and ori_ok) \
                    or self.step_in_phase >= self.budgets["align"]:
                z_above = float(eef_pos[2]) - self._table_z
                print(f"  [align]   xy_err={xy_err:.3f}  "
                      f"z=table+{z_above:.3f}  reached_tol={xy_ok and z_ok and ori_ok}",
                      flush=True)
                self._advance("descend_to_weight")
            self.step_in_phase += 1
            return action

        # ── 3. Descend onto blue cube ────────────────────────────────
        if self.phase == "descend_to_weight":
            target = np.array([
                self._weight_xy[0],
                self._weight_xy[1],
                self._table_z + WEIGHT_GRASP_Z_ABOVE_TBL,
            ])
            # Both gains high so xy keeps converging during the z drop;
            # otherwise default gain_xy=2.5 means a 3 cm residual error
            # from align takes ~30 ticks to clear, by which time the
            # gripper is already at grasp z and closing on the edge.
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_xy=6.0, gain_z=6.0,
                                   target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.005
            if reached or self.step_in_phase >= self.budgets["descend_weight"]:
                z_above = float(eef_pos[2]) - self._table_z
                xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
                print(f"  [descend] eef_z=table+{z_above:.3f}  "
                      f"target=table+{WEIGHT_GRASP_Z_ABOVE_TBL:.3f}  "
                      f"xy_err={xy_err:.3f}  reached_tol={reached}", flush=True)
                self._advance("stabilize_grasp")
            self.step_in_phase += 1
            return action

        # ── 3b. Stabilize at grasp pose so xy fully converges before
        #         the jaws close. The Panda's OSC sacrifices xy
        #         precision for z while the arm is in a reach-extended
        #         configuration; without this hold, the descent ends
        #         with xy_err ~3 cm at far-corner cubes, the jaws
        #         strike the cube edge, shove it ~5 cm and miss the
        #         grasp entirely. ─────────────────────────────────
        if self.phase == "stabilize_grasp":
            target = np.array([
                self._weight_xy[0],
                self._weight_xy[1],
                self._table_z + WEIGHT_GRASP_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_xy=8.0, gain_z=4.0,
                                   target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            tight = xy_err < 0.006
            if (tight and self.step_in_phase >= self.budgets["stabilize_grasp"] // 2) \
                    or self.step_in_phase >= self.budgets["stabilize_grasp"]:
                print(f"  [stab_grasp] xy_err={xy_err:.4f}  tight={tight}",
                      flush=True)
                self._advance("close_gripper")
            self.step_in_phase += 1
            return action

        # ── 4. Close gripper, hold pose ──────────────────────────────
        if self.phase == "close_gripper":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase >= self.budgets["close"]:
                self._advance("lift_high")
            self.step_in_phase += 1
            return action

        # ── 5. Lift to safe traverse altitude ────────────────────────
        if self.phase == "lift_high":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + LIFT_HIGH_Z_ABOVE_TBL])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.012
            if reached or self.step_in_phase >= self.budgets["lift_high"]:
                # Read the IN-AIR cube xy and measure how far it sits
                # from the TCP. The placement target is the cube xy,
                # so the eef target must subtract this offset.
                if env is not None:
                    cube_pos, _ = self._read_weight_pose(env)
                    self._cube_to_eef_xy = cube_pos[:2] - eef_pos[:2]
                self._eef_target_place_xy = (
                    self._place_xy - self._cube_to_eef_xy
                )
                print(f"  [lift]    cube_to_eef_xy="
                      f"{np.round(self._cube_to_eef_xy, 3)}  "
                      f"eef_place_xy={np.round(self._eef_target_place_xy, 3)}",
                      flush=True)
                self._advance("translate_above_place")
            self.step_in_phase += 1
            return action

        # ── 6. Fly over to the placement xy at high altitude ─────────
        if self.phase == "translate_above_place":
            place_xy = self._eef_target_place_xy
            target = np.array([
                place_xy[0], place_xy[1],
                self._table_z + LIFT_HIGH_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=8.0, target_quat=self._grasp_quat)
            eef_xy_err  = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            cube_xy_err = float("inf")
            if env is not None:
                cube_pos, _ = self._read_weight_pose(env)
                cube_xy_err = float(
                    np.linalg.norm(cube_pos[:2] - self._place_xy)
                )
            # Exit on CUBE alignment, not eef alignment -- that's what
            # the lever law cares about. POS_TOL_XY (1 cm) corresponds
            # to a few-percent arm-length error, well inside the 5 deg
            # angle threshold given the seesaw_frictionloss dead zone.
            cube_ok = cube_xy_err < POS_TOL_XY
            if cube_ok or self.step_in_phase >= self.budgets["translate"]:
                print(f"  [translate] eef_xy_err={eef_xy_err:.3f}  "
                      f"cube_xy_err={cube_xy_err:.3f}  reached_tol={cube_ok}",
                      flush=True)
                self._advance("descend_to_plank")
            self.step_in_phase += 1
            return action

        # ── 7. Slow press-to-level descent. Cube stays gripped; the
        #         gripper actively pushes the cube down on the plank
        #         while monitoring the plank angle. Exit the moment
        #         the plank crosses level -- never push past, or the
        #         seesaw flips backwards. If the plank refuses to
        #         level (lever law off beyond friction tolerance),
        #         allow the press to keep going for the configured
        #         budget. ────────────────────────────────────────
        if self.phase == "descend_to_plank":
            place_xy = self._eef_target_place_xy
            press_target_z = (
                self._table_z + PLACE_Z_ABOVE_TBL - PLACE_Z_PRESS_BELOW
            )
            target = np.array([place_xy[0], place_xy[1], press_target_z])
            # Slow z gain (2.0 vs 4.0) so the press happens
            # progressively over hundreds of ms -- gives the plank
            # time to rotate without bouncing.
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=8.0, gain_z=2.5,
                                   target_quat=self._grasp_quat)

            angle_deg = self._read_plank_angle_deg(env) if env is not None else float("inf")
            # Exit slightly EARLY (3 deg, not 1.5) so plank's residual
            # angular momentum doesn't carry it past level after we
            # stop pressing. Damping + friction settles the rest.
            level = angle_deg < 3.0
            z_above = float(eef_pos[2]) - self._table_z
            z_at_target = z_above < (PLACE_Z_ABOVE_TBL - PLACE_Z_PRESS_BELOW + 0.005)

            # Print every 15 ticks normally, every tick once the plank
            # has started moving (angle dropped below the initial 10 deg
            # rest tilt by >1 deg) so we see the level transition clearly.
            verbose_window = angle_deg < 9.0
            if (self.step_in_phase % (3 if verbose_window else 15)) == 0:
                print(f"  [press] tick={self.step_in_phase:4d}  "
                      f"eef_z=table+{z_above:.3f}  angle={angle_deg:.2f}deg",
                      flush=True)

            if level or z_at_target \
                    or self.step_in_phase >= self.budgets["descend_plank"]:
                # Latch the eef z at exit. Hold here in stabilize_place;
                # going lower would push the plank past level.
                self._stab_lock_z = float(eef_pos[2])
                self._press_was_level = level
                print(f"  [press_done] eef_z=table+{z_above:.3f}  "
                      f"angle={angle_deg:.2f}deg  level={level}  "
                      f"z_at_target={z_at_target}  "
                      f"lock_z=table+{self._stab_lock_z - self._table_z:.3f}",
                      flush=True)
                self._advance("stabilize_place")
            self.step_in_phase += 1
            return action

        # ── 7b. Hold at the latched press-exit z so the plank
        #         settles at level under static gripper force --
        #         WITHOUT pushing further down. Continuing to
        #         command the deep press_target here over-presses,
        #         tipping the plank past level and rebounding it
        #         past the 5 deg success angle on the other side.
        #         If the press timed out without reaching level, we
        #         allow a small additional press before holding. ──
        if self.phase == "stabilize_place":
            place_xy = self._eef_target_place_xy
            # If level was reached: hold exactly at lock_z.
            # If timeout: allow a tiny additional press (2 mm extra)
            # to nudge the plank without flipping it.
            target_z = self._stab_lock_z - (0.0 if self._press_was_level else 0.002)
            target = np.array([place_xy[0], place_xy[1], target_z])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=8.0, gain_z=3.0,
                                   target_quat=self._grasp_quat)
            angle_deg = self._read_plank_angle_deg(env) if env is not None else float("inf")
            z_above = float(eef_pos[2]) - self._table_z
            level = angle_deg < 1.5
            settled_long_enough = self.step_in_phase >= self.budgets["stabilize_place"]
            if (level and self.step_in_phase >= self.budgets["stabilize_place"] // 2) \
                    or settled_long_enough:
                print(f"  [stab_place] eef_z=table+{z_above:.3f}  "
                      f"angle={angle_deg:.2f}deg  level={level}", flush=True)
                self._advance("open_gripper")
            self.step_in_phase += 1
            return action

        # ── 8. Open gripper, hold pose ───────────────────────────────
        if self.phase == "open_gripper":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = -1.0
            if self.step_in_phase >= self.budgets["open"]:
                self._advance("retract_up")
            self.step_in_phase += 1
            return action

        # ── 9. Retract straight up; xy locked so wrist clears the
        #         rising plank end as it levels out. ─────────────────
        if self.phase == "retract_up":
            place_xy = self._eef_target_place_xy
            target = np.array([
                place_xy[0], place_xy[1],
                self._table_z + RETRACT_HIGH_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            reached = abs(target[2] - eef_pos[2]) < 0.012
            if reached or self.step_in_phase >= self.budgets["retract_up"]:
                self._advance("retreat_aside")
            self.step_in_phase += 1
            return action

        # ── 10. Retreat clear of the seesaw ──────────────────────────
        if self.phase == "retreat_aside":
            target = np.array([
                RETREAT_XY[0], RETREAT_XY[1],
                self._table_z + RETRACT_HIGH_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_xy=4.0, target_quat=self._grasp_quat)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            xy_ok = xy_err < 0.02
            if xy_ok or self.step_in_phase >= self.budgets["retreat"]:
                self._advance("wait_for_balance")
            self.step_in_phase += 1
            return action

        # ── 11. Hold pose; the env's balance counter fills here ──────
        if self.phase == "wait_for_balance":
            target = np.array([
                RETREAT_XY[0], RETREAT_XY[1],
                self._table_z + RETRACT_HIGH_Z_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=self._grasp_quat)
            if self.step_in_phase >= self.budgets["wait_balance"]:
                self._advance("done")
            self.step_in_phase += 1
            return action

        # ── 12. Done: keep holding the retreat pose ─────────────────
        target = np.array([
            RETREAT_XY[0], RETREAT_XY[1],
            self._table_z + RETRACT_HIGH_Z_ABOVE_TBL,
        ])
        action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                               target_quat=self._grasp_quat)
        self.step_in_phase += 1
        return action


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env    = ib.make("ib-p3-seesaw-balance-v1", privileged_obs=True, render_mode=render_mode)
    policy = SeesawBalancePolicy()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 17)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===", flush=True)
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            done = terminated or truncated
        print(f"Episode {ep}  score={info.get('task_score', 0.0):.3f}  "
              f"phase={policy.phase}", flush=True)
    env.close()
