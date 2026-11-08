"""Scripted intercept-and-grasp policy for ``ib-p2-linear-intercept-v1``.

The task spawns a sphere near the table edge with an initial linear velocity
roughly toward the opposite side. The arm has to grasp it on the fly and
hold the grasp for ``hold_steps`` ticks.

Strategy
--------
A reactive chase loses every time because the OSC controller has finite
bandwidth — by the time the EE has driven to "where the ball is now,"
the ball is somewhere else and the EE chases the lead point asymptotically
without ever catching up. The fix is a **pre-plan**: at the end of
``rise`` we project the ball's straight-line trajectory forward and pick
the *soonest* point on that trajectory the EE can reach, given an honest
estimate of EE travel time. The policy then commits to that single
intercept point during ``track`` — no chasing, just a fixed waypoint that
the ball will arrive at at a known time.

The pre-plan solves for the smallest T such that

    EE_travel_time(eef_xy → ball_xy + v·T)  ≤  T

Geometrically: among all points along the ball's path, pick the closest
one the EE can reach before the ball does. Travel time is approximated as
``‖P − eef_xy‖ / EE_NOMINAL_XY_SPEED``. Because we want the EE to arrive
*before* the ball (so the EE is settled and ready when the ball passes
under it), we pad the travel time with ``EE_SETTLE_MARGIN``.

After the pre-plan, ``track`` drives to the planned intercept XY at
hover height. ``descend`` and ``close`` *do* keep re-predicting (with
shorter lead times) so they're robust to small modelling errors near the
grasp, but the bulk of the EE's travel is committed to the pre-plan, so
it doesn't lag behind a moving target.

Phases
------
1. ``rise``     — lift the EE to a safe pre-intercept height above the
                  table while estimating the ball's velocity from finite
                  differences. Runs for a few ticks so the velocity
                  estimate has settled before we plan.
2. ``track``    — hover above the predicted intercept point with the
                  gripper open. The lead time shrinks as the EE closes
                  the XY gap, so the hover point converges to the ball's
                  actual position as the EE arrives.
3. ``descend``  — drop toward the ball while *still* re-predicting the
                  intercept XY each tick using the residual Z-time. High
                  vertical gain; XY tracks live.
4. ``close``    — close gripper while keeping XY locked onto the live
                  ball position (last reactive step). Releases tracking
                  only once the close hold has elapsed.
5. ``lift``     — lift straight up; task scores success once contact is
                  held for ``hold_steps``.

Gripper convention: action[6] in [-1, 1]; -1 fully open, +1 fully closed.
"""

from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface


# ── Gripper orientation ─────────────────────────────────────────────
# Top-down: gripper Z points down. xyzw convention (matches obs["robot0_eef_quat"]).
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# ── Heights ─────────────────────────────────────────────────────────
# All heights are absolute world-Z (table top sits at ``self._table_z``).
HOVER_HEIGHT_ABOVE_BALL = 0.10   # m, EE hover height above ball during track
GRASP_HEIGHT_ABOVE_BALL = -0.003  # m, EE z target relative to ball center.
                                  # Slightly below the equator so jaws
                                  # wrap the lower hemisphere — but only
                                  # by 3 mm, leaving table clearance.
EE_TABLE_CLEARANCE      = 0.005   # m, hard floor on EE-tip Z above the
                                  # table top. The descend / close phases
                                  # clamp ``grasp_z`` to this minimum so
                                  # the gripper never crashes the surface.
LIFT_HEIGHT_ABOVE_TABLE = 0.15    # m, post-grasp EE-tip height above the
                                  # table. Must clear the task's 0.10 m
                                  # ball-lift success threshold by enough
                                  # margin that EE-tip→ball-center offset
                                  # (~3-4 cm) doesn't drop the ball under it.

# ── Prediction / intercept tuning ───────────────────────────────────
# Effective steady-state XY speed of the EE under our OSC gain
# (GAIN_XY_TRACK ≈ 14, max_step 1.0). Measured during track-phase
# reaches, it saturates around 0.55 m/s in straight-line travel, with a
# brief acceleration/deceleration tail.
# Used by the pre-plan to estimate "how far ahead can the EE actually be
# in time T."
EE_NOMINAL_XY_SPEED = 0.55       # m/s
# Nominal EE vertical descent rate (m/s) when descending. Used to
# estimate the time the EE needs to drop from hover to grasp height,
# so descend can lead the ball by exactly that much.
EE_NOMINAL_Z_SPEED  = 0.55       # m/s
# Time the EE needs to settle at the planned intercept point after
# arriving — i.e. the EE should reach the waypoint a bit *before* the
# ball does, so its residual velocity has damped out by the time the
# ball arrives. Larger margin = more conservative (further-ahead pick).
# Descend / gripper-close are NOT folded into this constant — those are
# handled by the descend phase via XY lookahead, so the planner stays
# aggressive about picking the soonest reachable intercept.
EE_SETTLE_MARGIN    = 0.15       # s
# Floor and ceiling on the predicted lead time to keep early/late ticks
# from producing degenerate targets.
LEAD_TIME_MIN = 0.0              # s — allow chasing the actual ball
LEAD_TIME_MAX = 1.50              # s — plan up to 1.5 s of ball flight
# Velocity-estimator EMA factor (1.0 = no smoothing; lower = more smoothing).
# Higher alpha → more reactive to actual ball motion (e.g. after a
# collision with the gripper) at the cost of more noise.
VEL_EMA_ALPHA = 0.55
# Minimum ticks of velocity estimation before we trust it.
VEL_WARMUP_STEPS = 4
# Pre-plan numerics: scan T over [0, LEAD_TIME_MAX] in this many slices
# to find the soonest reachable intercept. 60 slices @ 1.5 s ⇒ 25 ms
# resolution, more than enough.
PRE_PLAN_T_SLICES   = 60

# ── Phase transitions ───────────────────────────────────────────────
RISE_DONE_Z_TOL  = 0.025  # m, EE within this of hover height → enter track
# Loose thresholds — descend is reactive, so we don't need a tight
# XY lock-in before dropping. Triggering descend earlier means we start
# closing vertical distance while the ball is still some way out.
TRACK_XY_TOL     = 0.025  # m, XY distance EE↔ball under which we descend
TRACK_TTA_TOL    = 0.50   # s, predicted time-to-arrival at which we descend
DESCEND_Z_TOL    = 0.012  # m, EE close enough to ball z → consider closing
DESCEND_XY_TOL   = 0.012  # m, XY alignment of *current* ball pos with EE
# Time the parallel-jaw gripper takes to clamp on a 4 cm sphere from
# fully-open at action[6]=+1. At 100 Hz this is roughly 0.08-0.12 s; we
# trigger close this many seconds *before* the ball arrives under the EE
# so the jaws are mid-close at contact rather than just starting to move.
GRIPPER_CLOSE_TIME = 0.10  # s
CLOSE_STEPS      = 25     # ticks holding gripper closed before lifting
LIFT_DONE_Z_TOL  = 0.02   # m, lift target reached → enter hold
HOLD_STEPS       = 80     # ticks of holding after lift (sized > task hold_steps)
# Grasp-failure detection (during lift). After this many lift ticks we
# inspect whether the ball came up with the EE; if not, the policy
# declares the grasp a failure and terminates the episode early.
GRASP_CHECK_DELAY    = 25     # ticks into lift before checking
GRASP_CHECK_LIFT_TOL = 0.02   # m, ball must be this far above its
                              # at-rest z (table_z + ball_radius)

# ── Phase budgets (safety timeouts; episode is 1000 steps @ 100 Hz) ─
RISE_MAX_STEPS    = 80
TRACK_MAX_STEPS   = 600
DESCEND_MAX_STEPS = 80
LIFT_MAX_STEPS    = 120

# ── OSC gains (proportional on world-frame Cartesian error) ─────────
# Saturated for max linear speed: any error > ~0.05 m drives the action
# to ±1.0 (env clip). Higher gain reduces the residual gap when settling
# at the planned waypoint without affecting top speed.
GAIN_XY_TRACK   = 25.0   # big — saturates immediately for any non-trivial gap
GAIN_Z_TRACK    = 12.0
GAIN_XY_DESCEND = 25.0
GAIN_Z_DESCEND  = 18.0   # drop fast
GAIN_XY_CLOSE   = 20.0
GAIN_Z_CLOSE    = 12.0
GAIN_XY_LIFT    = 20.0
GAIN_Z_LIFT     = 20.0
ROT_GAIN        = 3.0

GRIPPER_OPEN  = -1.0
GRIPPER_CLOSE =  1.0


def _quat_err_axis_angle(cur_xyzw: np.ndarray, tgt_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle error vector that rotates ``cur`` toward ``tgt``.

    Both inputs are (x, y, z, w). Output is suitable as the rotational
    part of an OSC_POSE delta action.
    """
    cx, cy, cz, cw = cur_xyzw
    tx, ty, tz, tw = tgt_xyzw
    # q_err = tgt ⊗ cur*
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
    angle = 2.0 * float(np.arctan2(sin_half, ew))
    axis = np.array([ex, ey, ez]) / sin_half
    return axis * angle


class LinearInterceptPolicy(PolicyInterface):
    """Predictive intercept: lead the ball, descend, grasp, hold."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose  # print phase transitions and intercept plans
        self.reset()

    def reset(self):
        self.phase = "rise"
        self.step_in_phase = 0

        self._table_z: float = 0.8
        self._ball_radius: float = 0.02

        # Set to True when the policy has detected its grasp attempt
        # failed — consumed by the runner to break the episode loop.
        self.failed: bool = False

        # Ball state estimate
        self._prev_ball_pos: np.ndarray | None = None
        self._ball_vel_est: np.ndarray = np.zeros(3)
        self._vel_warm: int = 0

        # Lift-phase XY anchor. Latched only at the end of ``close``,
        # off the EE's actual position — by which point the gripper
        # holds the ball, so the EE pose is the truth. Track / descend
        # / close all stay reactive on the live ball position and never
        # use this.
        self._descend_xy: np.ndarray | None = None

        # Hover height during track. Cached so all phases agree on
        # "above the ball" without re-reading config every tick.
        self._hover_z: float = 0.8 + HOVER_HEIGHT_ABOVE_BALL

        # Pre-plan output: the fixed XY intercept waypoint chosen at the
        # end of ``rise``, and the absolute tick at which the ball is
        # predicted to arrive there. The track phase steers to this
        # single waypoint instead of recomputing every tick.
        self._planned_intercept_xy: np.ndarray | None = None
        self._planned_intercept_T:  float = 0.0
        # Tick counter spanning the whole episode (used to log how close
        # we hit the planned arrival time).
        self._global_step: int = 0
        # Previous EE XY for finite-diff EE-speed logging.
        self._prev_eef_xy: np.ndarray | None = None

    # ── Phase machinery ───────────────────────────────────────────────
    def _advance(self, next_phase: str):
        if self.verbose:
            print(f"  [{self.step_in_phase:4d} steps] {self.phase} → {next_phase}")
        self.phase = next_phase
        self.step_in_phase = 0

    # ── Action helpers ────────────────────────────────────────────────
    @staticmethod
    def _move_to(eef_pos: np.ndarray,
                 eef_quat: np.ndarray,
                 target_pos: np.ndarray,
                 gripper: float,
                 gain_xy: float,
                 gain_z: float,
                 max_step: float = 1.0,
                 target_quat: np.ndarray = TOP_DOWN_QUAT) -> np.ndarray:
        """P-controlled OSC_POSE delta action toward ``target_pos``."""
        action = np.zeros(7)
        delta = target_pos - eef_pos
        action[0] = float(np.clip(delta[0] * gain_xy, -max_step, max_step))
        action[1] = float(np.clip(delta[1] * gain_xy, -max_step, max_step))
        action[2] = float(np.clip(delta[2] * gain_z,  -max_step, max_step))
        ori_err = _quat_err_axis_angle(eef_quat, target_quat)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
        action[6] = gripper
        return action

    # ── Prediction core ──────────────────────────────────────────────
    def _update_ball_velocity(self, ball_pos: np.ndarray, dt: float = 0.01) -> None:
        """Finite-difference + EMA smoothing of the ball's linear velocity."""
        if self._prev_ball_pos is None:
            self._prev_ball_pos = ball_pos.copy()
            return
        v_inst = (ball_pos - self._prev_ball_pos) / dt
        if self._vel_warm == 0:
            self._ball_vel_est = v_inst
        else:
            self._ball_vel_est = (
                VEL_EMA_ALPHA * v_inst
                + (1.0 - VEL_EMA_ALPHA) * self._ball_vel_est
            )
        self._prev_ball_pos = ball_pos.copy()
        self._vel_warm = min(self._vel_warm + 1, VEL_WARMUP_STEPS + 5)

    def _predict_intercept(self,
                           ball_pos: np.ndarray,
                           eef_pos: np.ndarray,
                           lead_time: float | None = None) -> tuple[np.ndarray, float]:
        """Return (predicted ball XY at intercept, lead time in seconds).

        If ``lead_time`` is given (e.g. the EE's time-to-bottom during
        descend), use it directly — that's the time we expect to need
        before the gripper meets the ball, so the prediction lands at
        the right moment.

        Otherwise (track phase) solve the rough kinematic balance:
            EE_speed · T = ‖(ball_xy + v_xy · T) − eef_xy‖
        by taking T = current_gap / EE_NOMINAL_XY_SPEED, refining once,
        and clamping. One refinement is enough — further passes don't
        beat noise in ``v_xy``.
        """
        v_xy = self._ball_vel_est[:2]
        if lead_time is not None:
            T = float(np.clip(lead_time, LEAD_TIME_MIN, LEAD_TIME_MAX))
            return ball_pos[:2] + v_xy * T, T
        gap = float(np.linalg.norm(ball_pos[:2] - eef_pos[:2]))
        T = gap / max(EE_NOMINAL_XY_SPEED, 1e-3)
        # One refinement: if we'll lead by v·T, the new gap is to that
        # lead point, not the current ball position.
        lead_xy = ball_pos[:2] + v_xy * T
        gap2 = float(np.linalg.norm(lead_xy - eef_pos[:2]))
        T = gap2 / max(EE_NOMINAL_XY_SPEED, 1e-3)
        T = float(np.clip(T, LEAD_TIME_MIN, LEAD_TIME_MAX))
        lead_xy = ball_pos[:2] + v_xy * T
        return lead_xy, T

    def _pre_plan_intercept(self,
                            ball_pos: np.ndarray,
                            eef_pos: np.ndarray) -> tuple[np.ndarray, float]:
        """Pick the soonest point on the ball's trajectory the EE can reach.

        Scans ``T`` over ``[0, LEAD_TIME_MAX]`` in ``PRE_PLAN_T_SLICES``
        steps. For each candidate T computes:

            P(T)        = ball_pos + v_ball · T          (where ball will be)
            t_travel(T) = ‖P(T) − eef_xy‖ / EE_NOMINAL_XY_SPEED

        Returns the smallest T such that
            t_travel(T) + EE_SETTLE_MARGIN ≤ T
        — i.e. the EE arrives at hover with ``EE_SETTLE_MARGIN`` seconds
        to spare before the ball reaches the same XY. Descend and gripper
        close are NOT included here; they're absorbed by the descend
        phase, which looks ahead by ``t_descend + GRIPPER_CLOSE_TIME`` so
        its XY target lands exactly under where the ball will be when
        the jaws actually shut.

        If no T satisfies the bound, fall back to the largest T — the EE
        will at least chase in the right direction.
        """
        v_xy = self._ball_vel_est[:2]
        ts = np.linspace(0.0, LEAD_TIME_MAX, PRE_PLAN_T_SLICES)
        best_T: float | None = None
        best_P: np.ndarray | None = None
        for T in ts:
            P_xy = ball_pos[:2] + v_xy * T
            travel = float(np.linalg.norm(P_xy - eef_pos[:2])) \
                     / max(EE_NOMINAL_XY_SPEED, 1e-3)
            if travel + EE_SETTLE_MARGIN <= T:
                best_T = float(T)
                best_P = P_xy
                break
        if best_P is None:
            T = float(LEAD_TIME_MAX)
            best_T = T
            best_P = ball_pos[:2] + v_xy * T
        return best_P, best_T

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0.0, 0.0, 0.0, 1.0])),
                              dtype=float)
        ball_pos = np.asarray(obs["object_pos"], dtype=float)

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))
            self._ball_radius = float(getattr(env, "_object_radius",
                                              self._ball_radius))
            self._hover_z = self._table_z + HOVER_HEIGHT_ABOVE_BALL

        self._update_ball_velocity(ball_pos)
        self._global_step += 1

        # ── 1. Rise: get the EE to hover height while velocity warms up ─
        if self.phase == "rise":
            target = np.array([eef_pos[0], eef_pos[1], self._hover_z])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_OPEN,
                                   gain_xy=GAIN_XY_TRACK,
                                   gain_z=GAIN_Z_TRACK)
            z_ok = abs(eef_pos[2] - self._hover_z) < RISE_DONE_Z_TOL
            warm = self._vel_warm >= VEL_WARMUP_STEPS
            if (z_ok and warm) or self.step_in_phase >= RISE_MAX_STEPS:
                # PRE-PLAN: lock in the intercept waypoint now that the
                # velocity estimate has settled. The EE then commits to
                # this fixed XY for the entire track phase.
                P_xy, T = self._pre_plan_intercept(ball_pos, eef_pos)
                self._planned_intercept_xy = P_xy
                self._planned_intercept_T = T
                travel = float(np.linalg.norm(P_xy - eef_pos[:2])) \
                         / max(EE_NOMINAL_XY_SPEED, 1e-3)
                if self.verbose:
                    print(f"  [rise done]  v_ball≈{np.round(self._ball_vel_est, 2)}  "
                          f"|v|={np.linalg.norm(self._ball_vel_est[:2]):.2f}")
                    print(f"  [pre-plan]   intercept=({P_xy[0]:+.3f},{P_xy[1]:+.3f})  "
                          f"T={T:.2f}s  travel≈{travel:.2f}s  "
                          f"slack={T - travel:.2f}s")
                self._advance("track")
            self.step_in_phase += 1
            return action

        # ── 2. Track: drive to the pre-planned intercept waypoint ────
        # We do NOT recompute the lead point every tick here. The whole
        # point of pre-planning is to give the EE one fixed waypoint to
        # accelerate toward and settle at, so it isn't endlessly chasing
        # a moving target it can't keep up with.
        if self.phase == "track":
            P_xy = self._planned_intercept_xy
            target = np.array([P_xy[0], P_xy[1], self._hover_z])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_OPEN,
                                   gain_xy=GAIN_XY_TRACK,
                                   gain_z=GAIN_Z_TRACK)
            xy_err = float(np.linalg.norm(P_xy - eef_pos[:2]))
            # Time-to-arrival of the BALL at the planned waypoint, using
            # live velocity. Drives the descend trigger.
            v_xy = self._ball_vel_est[:2]
            v_mag = float(np.linalg.norm(v_xy))
            ball_to_P = P_xy - ball_pos[:2]
            tta = (float(np.dot(ball_to_P, v_xy)) / (v_mag * v_mag)
                   if v_mag > 1e-3 else 999.0)
            # Distance from current ball pos to plan waypoint (along v_xy
            # forward direction). Used for diagnostics.
            ready_xy  = xy_err < TRACK_XY_TOL
            ready_tta = 0.0 < tta < TRACK_TTA_TOL
            ee_speed = (
                float(np.linalg.norm(eef_pos[:2] - self._prev_eef_xy)) / 0.01
                if self._prev_eef_xy is not None else 0.0
            )
            if self.step_in_phase % 20 == 0:
                if self.verbose:
                    print(f"  [track t={self.step_in_phase}]  "
                          f"plan=({P_xy[0]:+.2f},{P_xy[1]:+.2f})  "
                          f"xy_err={xy_err:.3f}  tta={tta:+.2f}s  "
                          f"ee_v={ee_speed:.2f}m/s")
            self._prev_eef_xy = eef_pos[:2].copy()
            if (ready_xy and ready_tta) \
                    or self.step_in_phase >= TRACK_MAX_STEPS:
                self._advance("descend")
            self.step_in_phase += 1
            return action

        # ── 3. Descend: drop while tracking the ball XY with a lookahead
        #         that includes BOTH the descend time AND the gripper-
        #         close time. So the EE's XY at the bottom of the drop
        #         is exactly where the ball will be when the jaws shut,
        #         not just when the EE arrives at grasp height. This is
        #         what minimises the gripper-vs-ball offset at contact.
        if self.phase == "descend":
            grasp_z = max(
                float(ball_pos[2]) + GRASP_HEIGHT_ABOVE_BALL,
                self._table_z + EE_TABLE_CLEARANCE,
            )
            z_to_go = max(eef_pos[2] - grasp_z, 0.0)
            t_to_bottom = z_to_go / max(EE_NOMINAL_Z_SPEED, 1e-3)
            # Total horizon: where the ball will be when the jaws have
            # finished clamping. Includes both descent and gripper close.
            t_horizon = t_to_bottom + GRIPPER_CLOSE_TIME
            lead_xy, T = self._predict_intercept(
                ball_pos, eef_pos, lead_time=t_horizon,
            )
            target = np.array([lead_xy[0], lead_xy[1], grasp_z])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_OPEN,
                                   gain_xy=GAIN_XY_DESCEND,
                                   gain_z=GAIN_Z_DESCEND)
            # Close-trigger: where will the ball be when the jaws shut
            # (``GRIPPER_CLOSE_TIME`` from now)? If that point is already
            # under the EE AND we're at grasp Z, fire the close NOW.
            close_xy = ball_pos[:2] + self._ball_vel_est[:2] * GRIPPER_CLOSE_TIME
            close_xy_err = float(np.linalg.norm(close_xy - eef_pos[:2]))
            z_ok = abs(eef_pos[2] - grasp_z) < DESCEND_Z_TOL
            timed_close_ready = z_ok and (close_xy_err < DESCEND_XY_TOL)
            if self.step_in_phase % 5 == 0:
                if self.verbose:
                    print(f"  [descend t={self.step_in_phase}]  "
                          f"z_to_go={z_to_go:.3f}  t_horizon={t_horizon:.2f}s  "
                          f"lead=({lead_xy[0]:+.2f},{lead_xy[1]:+.2f})  "
                          f"close_err={close_xy_err:.3f}  z_ok={z_ok}")
            if timed_close_ready or self.step_in_phase >= DESCEND_MAX_STEPS:
                if self.verbose:
                    print(f"  [descend → close]  close_xy_err={close_xy_err:.3f}  "
                          f"v_ball={np.round(self._ball_vel_est[:2], 2)}")
                self._advance("close")
            self.step_in_phase += 1
            return action

        # ── 4. Close gripper while still tracking ball XY live ───────
        # The jaws aren't shut yet; we entered ``close`` GRIPPER_CLOSE_TIME
        # seconds before the ball was due to be centered, so for the first
        # few ticks the ball is still flying in. Lead the ball XY by the
        # *remaining* close-time so the EE meets it at jaw-shut time. After
        # the jaws have closed (~CLOSE_TIME elapsed) we just track ball pos.
        if self.phase == "close":
            grasp_z = max(
                float(ball_pos[2]) + GRASP_HEIGHT_ABOVE_BALL,
                self._table_z + EE_TABLE_CLEARANCE,
            )
            t_remaining = max(
                GRIPPER_CLOSE_TIME - self.step_in_phase * 0.01, 0.0,
            )
            close_xy = ball_pos[:2] + self._ball_vel_est[:2] * t_remaining
            target = np.array([close_xy[0], close_xy[1], grasp_z])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_CLOSE,
                                   gain_xy=GAIN_XY_CLOSE,
                                   gain_z=GAIN_Z_CLOSE,
                                   max_step=0.6)
            if self.step_in_phase >= CLOSE_STEPS:
                # Latch the lift XY off whatever the EE actually ended up at,
                # not the ball — because by now the gripper holds the ball
                # and EE position is the truth.
                self._descend_xy = eef_pos[:2].copy()
                self._advance("lift")
            self.step_in_phase += 1
            return action

        # ── 5. Lift straight up to settle the grasp ───────────────────
        # As soon as the EE has lifted clear, check whether the ball
        # came up with us. A successful grasp pulls ball z well above
        # its at-rest value (table_z + ball_radius); a failed grasp
        # leaves the ball on the table while the EE rises empty — set
        # ``failed`` and let the runner terminate the episode.
        if self.phase == "lift":
            target = np.array([self._descend_xy[0],
                               self._descend_xy[1],
                               self._table_z + LIFT_HEIGHT_ABOVE_TABLE])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_CLOSE,
                                   gain_xy=GAIN_XY_LIFT,
                                   gain_z=GAIN_Z_LIFT)
            if self.step_in_phase >= GRASP_CHECK_DELAY:
                rest_z = self._table_z + self._ball_radius
                ball_lifted = ball_pos[2] > rest_z + GRASP_CHECK_LIFT_TOL
                ee_lifted   = eef_pos[2] > self._table_z \
                              + EE_TABLE_CLEARANCE + 0.04
                if ee_lifted and not ball_lifted:
                    if self.verbose:
                        print(f"  [grasp FAILED]  ball_z={ball_pos[2]:.3f}  "
                              f"rest_z={rest_z:.3f}  ee_z={eef_pos[2]:.3f}  "
                              f"— terminating")
                    self.failed = True
                    self._advance("done")
                    return action
            reached = abs(eef_pos[2] - target[2]) < LIFT_DONE_Z_TOL
            if reached or self.step_in_phase >= LIFT_MAX_STEPS:
                self._advance("hold")
            self.step_in_phase += 1
            return action

        # ── 6. Hold: keep the ball aloft until the task awards score ──
        if self.phase == "hold":
            target = np.array([self._descend_xy[0],
                               self._descend_xy[1],
                               self._table_z + LIFT_HEIGHT_ABOVE_TABLE])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_CLOSE,
                                   gain_xy=GAIN_XY_LIFT,
                                   gain_z=GAIN_Z_LIFT)
            if self.step_in_phase >= HOLD_STEPS:
                self._advance("done")
            self.step_in_phase += 1
            return action

        # ── 7. Done ──────────────────────────────────────────────────
        action = np.zeros(7)
        action[6] = GRIPPER_CLOSE
        self.step_in_phase += 1
        return action


# ── Stand-alone runner ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    import imbench as ib

    env = ib.make("ib-p2-linear-intercept-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = LinearInterceptPolicy()
    for ep in range(3):
        obs, info = env.reset(seed=ep * 17 + 1)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===")
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated or policy.failed
        status = "FAILED" if policy.failed else "ended"
        print(f"Episode {ep} {status}  score={info.get('task_score', 0.0):.3f}  "
              f"phase={policy.phase}  ball={obs['object_pos']}")
    env.close()
