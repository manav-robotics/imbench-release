"""Scripted intercept-and-grasp policy for ``ib-p2-bounce-intercept-v1``.

Mirrors ``LinearInterceptPolicy`` but extends the predictor to handle
wall bounces. The ball travels in straight segments, reflects off the
table-edge walls, and the policy has to pick an intercept point on the
*piecewise* trajectory.

Strategy
--------
At the end of ``rise``, with a settled velocity estimate in hand, simulate
the ball's path forward in time:

* Propagate ``p(t) = p_0 + v · t`` until the ball hits one of the three
  active walls (the opening wall is parked under the table and ignored).
* Reflect ``v`` across the wall normal (perfectly elastic for planning;
  the real ``wall_restitution`` is ~0.9 so this slightly under-shoots
  speed after bounces — acceptable for a single-shot pre-plan).
* Repeat for up to ``MAX_BOUNCES`` rebounds or until time runs out.

Then scan candidate intercept times along that path and pick the smallest
T such that the EE can travel to ``p(T)`` with ``EE_SETTLE_MARGIN``
seconds to spare. Track / descend / close are reactive — descend looks
ahead by ``t_to_bottom + GRIPPER_CLOSE_TIME`` so the EE's XY at jaw-shut
time matches the ball's predicted position.

Phases mirror the linear-intercept policy: rise → track → descend →
close → lift → hold.
"""

from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface


# ── Gripper orientation ─────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# ── Heights ─────────────────────────────────────────────────────────
HOVER_HEIGHT_ABOVE_BALL = 0.10   # m, EE hover height above ball during track
GRASP_HEIGHT_ABOVE_BALL = -0.012  # m, EE z target relative to ball center.
                                  # Pulls the EE-tip 12 mm below the ball
                                  # center so the jaw throat (2-3 cm above
                                  # the tip) sits at the ball center —
                                  # ball ends up cradled, not pinched at
                                  # the tip. EE_TABLE_CLEARANCE clamp
                                  # prevents crashing the surface.
LIFT_HEIGHT_ABOVE_TABLE = 0.20    # m, EE-tip target height after grasp.
                                  # Comfortably clears the task's 0.10 m
                                  # ball-lift success threshold (the ball
                                  # center sits ~3 cm below EE-tip, so the
                                  # ball ends up at ~0.17 m above table).

# ── Prediction / intercept tuning ───────────────────────────────────
# Empirical EE max XY speed under saturated OSC action (measured
# ee_v ≈ 0.30 m/s peak even when v_cmd is 0.65). This is what the OSC
# actually delivers, not what we'd hope for.
EE_MAX_XY_SPEED = 0.35       # m/s — ceiling for the velocity feed-forward
EE_MAX_Z_SPEED  = 0.45       # m/s — Z is faster than XY under default gains
# Conservative steady-state estimate used by the planner's reachability
# check. Smaller than peak so the planner doesn't promise intercepts the
# EE can't actually make.
EE_NOMINAL_XY_SPEED = 0.28   # m/s — honest "average travel speed"
EE_NOMINAL_Z_SPEED  = 0.40   # m/s
# Settling margin at the hover waypoint before the ball arrives.
EE_SETTLE_MARGIN    = 0.15       # s
# Velocity-estimator EMA smoothing.
VEL_EMA_ALPHA = 0.55
VEL_WARMUP_STEPS = 4
# Pre-plan numerics.
LEAD_TIME_MIN = 0.0
LEAD_TIME_MAX = 3.00              # s — bouncing ball can take a while
PRE_PLAN_T_SLICES = 200           # 15 ms resolution at 3 s horizon
# Maximum number of wall bounces to simulate when projecting the ball's
# trajectory. After this many we just truncate; the planner will pick
# whatever point on the simulated path is reachable.
MAX_BOUNCES = 6
# Default coefficient of restitution for the planner's bounce model.
# Overridden at runtime from the env's ``wall_restitution`` if exposed.
PLAN_RESTITUTION_DEFAULT = 0.9
# Minimum simulated speed before we abandon further bounces — once the
# ball is creeping the predictor isn't useful.
MIN_PLAN_SPEED = 0.05            # m/s
# Minimum XY clearance the planner enforces between the planned EE
# waypoint and the walls. Without this the EE could descend right onto
# a wall edge. Walls are ``wall_size[0]`` thick (half-extent ~1 cm); the
# Panda fingertips need ~3-4 cm clearance to clear the wall body.
EE_WALL_CLEARANCE   = 0.04       # m
# Hard floor on EE-tip Z relative to the table top. The descend phase
# clamps grasp_z to ``table_z + EE_TABLE_CLEARANCE`` so the gripper
# never crashes into the surface — useful for small balls or when the
# negative ``GRASP_HEIGHT_ABOVE_BALL`` would put the tip too low.
EE_TABLE_CLEARANCE  = 0.005      # m

# ── Phase transitions ───────────────────────────────────────────────
RISE_DONE_Z_TOL  = 0.025
TRACK_XY_TOL     = 0.04
TRACK_TTA_TOL    = 0.50
# Allow this much time past the bounce-aware plan's expected arrival
# before replanning. A target behind the ball's current velocity can
# still be ahead on its reflected path, so straight-line time-to-arrival
# must not expire the plan before its expected bounce.
TRACK_TTA_STALE  = 0.30
MAX_REPLANS      = 3
DESCEND_Z_TOL    = 0.012
# Looser than the linear policy: bouncing balls have noisier prediction
# (a wall hit during the lookahead horizon adds geometric uncertainty),
# so 4 cm of slack lets the descend trigger fire before we time out.
DESCEND_XY_TOL   = 0.04
GRIPPER_CLOSE_TIME = 0.15  # give the jaws time to firmly clamp
                            # before we transition to lift.
# After the jaws have shut, keep targeting this far ahead of the ball
# so the EE drags along at the ball's speed — prevents the ball rolling
# out of the gripper basin during the post-close hold.
CLOSE_TRACK_LEAD = 0.05
CLOSE_STEPS      = 40       # 0.40 s of clamping before lift
LIFT_DONE_Z_TOL  = 0.02
HOLD_STEPS       = 80
# Grasp-failure detection (during lift). After this many lift ticks we
# inspect whether the ball came up with the EE; if not, the policy
# declares the grasp a failure and terminates the episode early.
GRASP_CHECK_DELAY    = 25     # ticks into lift before checking
GRASP_CHECK_LIFT_TOL = 0.02   # m, ball must be this far above its
                              # at-rest z (table_z + ball_radius) to
                              # count as "lifted with the EE"

# ── Phase budgets ───────────────────────────────────────────────────
RISE_MAX_STEPS    = 80
TRACK_MAX_STEPS   = 1200          # episode budget is 10000, plenty of room
DESCEND_MAX_STEPS = 80
LIFT_MAX_STEPS    = 120

# ── OSC gains ───────────────────────────────────────────────────────
GAIN_XY_TRACK   = 25.0
GAIN_Z_TRACK    = 12.0
GAIN_XY_DESCEND = 25.0
GAIN_Z_DESCEND  = 18.0
GAIN_XY_CLOSE   = 20.0
GAIN_Z_CLOSE    = 12.0
GAIN_XY_LIFT    = 20.0
GAIN_Z_LIFT     = 20.0
ROT_GAIN        = 3.0

GRIPPER_OPEN  = -1.0
GRIPPER_CLOSE =  1.0


def _quat_err_axis_angle(cur_xyzw: np.ndarray, tgt_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle error vector that rotates ``cur`` toward ``tgt`` (xyzw)."""
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
    angle = 2.0 * float(np.arctan2(sin_half, ew))
    axis = np.array([ex, ey, ez]) / sin_half
    return axis * angle


class BounceInterceptPolicy(PolicyInterface):
    """Predictive intercept with wall-bounce trajectory rollout."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose  # print phase transitions and intercept plans
        self.reset()

    def reset(self):
        self.phase = "rise"
        self.step_in_phase = 0

        self._table_z: float = 0.8
        self._ball_radius: float = 0.02
        # XY half-extents of the ball-reachable region inside the walls.
        # Set in act() once we have the env handle.
        self._bounce_hx: float = 0.34
        self._bounce_hy: float = 0.29
        # Restitution used by the bounce predictor; overridden from env.
        self._restitution: float = PLAN_RESTITUTION_DEFAULT

        # Ball state estimate
        self._prev_ball_pos: np.ndarray | None = None
        self._ball_vel_est: np.ndarray = np.zeros(3)
        self._vel_warm: int = 0

        # Lift-phase XY anchor (latched at end of close).
        self._descend_xy: np.ndarray | None = None

        # Hover height during track.
        self._hover_z: float = 0.8 + HOVER_HEIGHT_ABOVE_BALL

        # Set to True when the policy has detected its grasp attempt
        # failed — consumed by the runner to break out of the episode
        # loop early instead of stepping through the remaining hold/done
        # phases on an empty gripper.
        self.failed: bool = False

        # Pre-plan output: fixed XY intercept waypoint chosen at end of rise.
        self._planned_intercept_xy: np.ndarray | None = None
        self._planned_intercept_T:  float = 0.0
        # Number of times the track phase has re-planned the intercept
        # (bounded by MAX_REPLANS). Reset each episode.
        self._replan_count: int = 0
        # Absolute step at which the pre-plan was committed; combined with
        # global_step gives us the elapsed time since plan creation, so
        # we can compute T_remaining for the lookahead controller.
        self._plan_commit_step: int = 0
        # Whole-episode tick counter (diagnostics).
        self._global_step: int = 0
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
        action = np.zeros(7)
        delta = target_pos - eef_pos
        action[0] = float(np.clip(delta[0] * gain_xy, -max_step, max_step))
        action[1] = float(np.clip(delta[1] * gain_xy, -max_step, max_step))
        action[2] = float(np.clip(delta[2] * gain_z,  -max_step, max_step))
        ori_err = _quat_err_axis_angle(eef_quat, target_quat)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
        action[6] = gripper
        return action

    @staticmethod
    def _move_with_lookahead(eef_pos: np.ndarray,
                             eef_quat: np.ndarray,
                             target_pos: np.ndarray,
                             gripper: float,
                             time_budget: float,
                             v_max_xy: float = EE_MAX_XY_SPEED,
                             v_max_z: float = EE_MAX_Z_SPEED,
                             target_quat: np.ndarray = TOP_DOWN_QUAT) -> np.ndarray:
        """Velocity-feedforward controller.

        Picks a target speed = remaining_distance / time_budget so the EE
        arrives just in time. Slack budget → slow approach, no slack →
        full saturation. Caps at ``v_max`` to honor the OSC's real limits.
        Maps the resulting velocity into the cartesian_delta action space
        ([-1, 1] per axis) by normalizing to the axis-wise max speed.

        time_budget < dt is treated as "go as fast as you can" — clamp
        floor of 0.05 s prevents division blow-ups when we're late.
        """
        action = np.zeros(7)
        delta = target_pos - eef_pos
        dist_xy = float(np.linalg.norm(delta[:2]))
        dist_z  = float(abs(delta[2]))
        # Required average speed to land on time. With no slack the EE
        # commands its peak; with slack it commands less.
        budget = max(time_budget, 0.05)
        v_req_xy = dist_xy / budget
        v_req_z  = dist_z  / budget
        # Cap at hardware limits.
        v_xy = min(v_req_xy, v_max_xy)
        v_z  = min(v_req_z,  v_max_z)
        # Direction unit vectors.
        if dist_xy > 1e-6:
            ux, uy = delta[0] / dist_xy, delta[1] / dist_xy
        else:
            ux = uy = 0.0
        uz = np.sign(delta[2]) if dist_z > 1e-6 else 0.0
        # Action = velocity normalized to [-1, 1] per axis. The OSC's
        # cartesian_delta maps action=±1 to ≈ v_max, so v_xy / v_max_xy
        # gives the right action magnitude.
        action[0] = float(np.clip(ux * v_xy / v_max_xy, -1.0, 1.0))
        action[1] = float(np.clip(uy * v_xy / v_max_xy, -1.0, 1.0))
        action[2] = float(np.clip(uz * v_z  / v_max_z,  -1.0, 1.0))
        ori_err = _quat_err_axis_angle(eef_quat, target_quat)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
        action[6] = gripper
        return action

    # ── Velocity estimator ───────────────────────────────────────────
    def _update_ball_velocity(self, ball_pos: np.ndarray, dt: float = 0.01) -> None:
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

    # ── Bounce-aware predictor ───────────────────────────────────────
    def _predict_path_at(self,
                        ball_pos_xy: np.ndarray,
                        ball_vel_xy: np.ndarray,
                        T: float) -> np.ndarray:
        """Return the ball's XY position T seconds in the future, with
        reflections off the walls at x = ±_bounce_hx and y = ±_bounce_hy.

        The opening wall is unknown to the planner, so we treat all four
        sides as bouncing surfaces. In practice the policy is always
        triggered while the ball is still inside the U-shape; if the ball
        does escape through the opening the planner over-bounces it but
        the EE is already committed to a near-term intercept by then.
        """
        p = ball_pos_xy.astype(float).copy()
        v = ball_vel_xy.astype(float).copy()
        t_left = float(T)
        EPS = 1e-6  # tiny step away from the wall after reflection so
                    # the next iteration's t_hit doesn't immediately fire
                    # on the same surface due to floating-point drift.
        for _ in range(MAX_BOUNCES + 1):
            speed = float(np.linalg.norm(v))
            if speed < MIN_PLAN_SPEED:
                return p + v * t_left
            # Time to hit each wall.
            tx = (np.sign(v[0]) * self._bounce_hx - p[0]) / v[0] if abs(v[0]) > 1e-9 else np.inf
            ty = (np.sign(v[1]) * self._bounce_hy - p[1]) / v[1] if abs(v[1]) > 1e-9 else np.inf
            tx = tx if tx > 0 else np.inf
            ty = ty if ty > 0 else np.inf
            t_hit = min(tx, ty)
            if t_hit >= t_left:
                return p + v * t_left
            # Advance to the wall.
            p = p + v * t_hit
            t_left -= t_hit
            # Reflect velocity across the wall normal and damp by restitution.
            if tx < ty:
                v[0] = -v[0] * self._restitution
                p[0] += np.sign(v[0]) * EPS
            else:
                v[1] = -v[1] * self._restitution
                p[1] += np.sign(v[1]) * EPS
        return p + v * t_left

    def _is_safe_for_ee(self, P_xy: np.ndarray) -> bool:
        """Reject planned waypoints too close to a wall — the EE would
        clip the wall during descent. Walls are short (4 cm above the
        table), so this only matters for the descend / close phases,
        but the planner stays conservative.
        """
        clear_x = self._bounce_hx - EE_WALL_CLEARANCE
        clear_y = self._bounce_hy - EE_WALL_CLEARANCE
        return abs(P_xy[0]) <= clear_x and abs(P_xy[1]) <= clear_y

    def _pre_plan_intercept(self,
                            ball_pos: np.ndarray,
                            eef_pos: np.ndarray) -> tuple[np.ndarray, float]:
        """Pick the soonest reachable, EE-safe point on the trajectory.

        Scan T over [0, LEAD_TIME_MAX]; for each candidate require
            travel + EE_SETTLE_MARGIN ≤ T   (reachability)
            ``P_xy`` is at least EE_WALL_CLEARANCE from any wall.

        Return the first satisfying (T, P). If none qualifies, fall back
        to the last forecast — the EE chases in the right direction.
        """
        ts = np.linspace(0.0, LEAD_TIME_MAX, PRE_PLAN_T_SLICES)
        best_T: float | None = None
        best_P: np.ndarray | None = None
        for T in ts:
            P_xy = self._predict_path_at(
                ball_pos[:2], self._ball_vel_est[:2], float(T),
            )
            if not self._is_safe_for_ee(P_xy):
                continue
            travel = float(np.linalg.norm(P_xy - eef_pos[:2])) \
                     / max(EE_NOMINAL_XY_SPEED, 1e-3)
            if travel + EE_SETTLE_MARGIN <= T:
                best_T = float(T)
                best_P = P_xy
                break
        if best_P is None:
            T = float(LEAD_TIME_MAX)
            best_T = T
            best_P = self._predict_path_at(
                ball_pos[:2], self._ball_vel_est[:2], T,
            )
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
            full_size = getattr(env, "_table_full_size", None)
            wall_size = getattr(env, "_wall_size", [0.01, 0.20, 0.04])
            self._restitution = float(getattr(env, "_wall_restitution",
                                              PLAN_RESTITUTION_DEFAULT))
            if full_size is not None:
                # Effective half-extent the ball can reach: table half
                # minus wall half-thickness minus ball radius.
                self._bounce_hx = (full_size[0] / 2.0
                                   - wall_size[0]
                                   - self._ball_radius)
                self._bounce_hy = (full_size[1] / 2.0
                                   - wall_size[0]
                                   - self._ball_radius)

        self._update_ball_velocity(ball_pos)
        self._global_step += 1

        # ── 1. Rise: get to hover height while velocity warms up ─────
        if self.phase == "rise":
            target = np.array([eef_pos[0], eef_pos[1], self._hover_z])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_OPEN,
                                   gain_xy=GAIN_XY_TRACK,
                                   gain_z=GAIN_Z_TRACK)
            z_ok = abs(eef_pos[2] - self._hover_z) < RISE_DONE_Z_TOL
            warm = self._vel_warm >= VEL_WARMUP_STEPS
            if (z_ok and warm) or self.step_in_phase >= RISE_MAX_STEPS:
                P_xy, T = self._pre_plan_intercept(ball_pos, eef_pos)
                self._planned_intercept_xy = P_xy
                self._planned_intercept_T = T
                self._plan_commit_step = self._global_step
                travel = float(np.linalg.norm(P_xy - eef_pos[:2])) \
                         / max(EE_NOMINAL_XY_SPEED, 1e-3)
                if self.verbose:
                    print(f"  [rise done]  v_ball≈{np.round(self._ball_vel_est, 2)}  "
                          f"|v|={np.linalg.norm(self._ball_vel_est[:2]):.2f}  "
                          f"COR={self._restitution:.2f}")
                    print(f"  [pre-plan]   intercept=({P_xy[0]:+.3f},{P_xy[1]:+.3f})  "
                          f"T={T:.2f}s  travel≈{travel:.2f}s  "
                          f"slack={T - travel:.2f}s  "
                          f"bounce_hx={self._bounce_hx:.2f} hy={self._bounce_hy:.2f}")
                self._advance("track")
            self.step_in_phase += 1
            return action

        # ── 2. Track: drive to the pre-planned intercept waypoint with
        #         saturated high-gain P-control. The OSC's effective max
        #         speed (~0.30 m/s) is the binding constraint; running a
        #         lookahead-velocity controller below that just gives up
        #         margin we don't have.
        if self.phase == "track":
            P_xy = self._planned_intercept_xy
            target = np.array([P_xy[0], P_xy[1], self._hover_z])
            action = self._move_to(
                eef_pos, eef_quat, target,
                gripper=GRIPPER_OPEN,
                gain_xy=GAIN_XY_TRACK,
                gain_z=GAIN_Z_TRACK,
            )
            xy_err = float(np.linalg.norm(P_xy - eef_pos[:2]))
            v_xy = self._ball_vel_est[:2]
            v_mag = float(np.linalg.norm(v_xy))
            ball_to_P = P_xy - ball_pos[:2]
            tta = (float(np.dot(ball_to_P, v_xy)) / (v_mag * v_mag)
                   if v_mag > 1e-3 else 999.0)
            ready_xy  = xy_err < TRACK_XY_TOL
            ready_tta = 0.0 < tta < TRACK_TTA_TOL
            has_bounced = env is None or bool(getattr(env, "_has_bounced", False))
            plan_remaining = (self._planned_intercept_T
                              - (self._global_step - self._plan_commit_step) * 0.01)
            ee_speed = (
                float(np.linalg.norm(eef_pos[:2] - self._prev_eef_xy)) / 0.01
                if self._prev_eef_xy is not None else 0.0
            )
            if self.step_in_phase % 20 == 0:
                if self.verbose:
                    print(f"  [track t={self.step_in_phase}]  "
                          f"plan=({P_xy[0]:+.2f},{P_xy[1]:+.2f})  "
                          f"xy_err={xy_err:.3f}  tta_ball={tta:+.2f}s  "
                          f"ee_v={ee_speed:.2f}m/s")
            self._prev_eef_xy = eef_pos[:2].copy()
            # Stale-plan detection uses the arrival time from the reflected path.
            # A negative straight-line tta before a bounce does not mean
            # the ball has passed this waypoint. Give the plan its time
            # budget, then re-plan from the measured ball state.
            #
            # The off-table check still terminates — if the ball has
            # left the playing surface, no replan can recover it.
            ball_off_table = ball_pos[2] < self._table_z - 0.05
            if ball_off_table:
                if self.verbose:
                    print(f"  [track FAILED]  ball_z={ball_pos[2]:.3f} "
                          f"(off table) — terminating")
                self.failed = True
                self._advance("done")
                return action
            stale = plan_remaining < -TRACK_TTA_STALE
            if stale and self._replan_count < MAX_REPLANS:
                P_xy_new, T_new = self._pre_plan_intercept(ball_pos, eef_pos)
                travel_new = float(np.linalg.norm(P_xy_new - eef_pos[:2])) \
                             / max(EE_NOMINAL_XY_SPEED, 1e-3)
                self._planned_intercept_xy = P_xy_new
                self._planned_intercept_T = T_new
                self._plan_commit_step = self._global_step
                self._replan_count += 1
                if self.verbose:
                    print(f"  [track REPLAN #{self._replan_count}]  "
                          f"plan_remaining={plan_remaining:+.2f}s  "
                          f"new_intercept=({P_xy_new[0]:+.3f},{P_xy_new[1]:+.3f})  "
                          f"T={T_new:.2f}s  travel≈{travel_new:.2f}s  "
                          f"slack={T_new - travel_new:.2f}s")
                self.step_in_phase += 1
                return action
            if stale:
                # Out of replans — give up.
                if self.verbose:
                    print(f"  [track FAILED]  plan_remaining={plan_remaining:+.2f}s  "
                          f"replans exhausted — terminating")
                self.failed = True
                self._advance("done")
                return action
            if has_bounced and ((ready_xy and ready_tta)
                               or self.step_in_phase >= TRACK_MAX_STEPS):
                self._advance("descend")
            self.step_in_phase += 1
            return action

        # ── 3. Descend with bounce-aware lookahead and adaptive speed ─
        # XY target is where the ball will be after t_to_bottom +
        # gripper-close time. Velocity is set by the time budget so the
        # EE drops *just* fast enough to meet the ball at jaw-shut.
        if self.phase == "descend":
            grasp_z = max(
                float(ball_pos[2]) + GRASP_HEIGHT_ABOVE_BALL,
                self._table_z + EE_TABLE_CLEARANCE,
            )
            z_to_go = max(eef_pos[2] - grasp_z, 0.0)
            t_to_bottom = z_to_go / max(EE_NOMINAL_Z_SPEED, 1e-3)
            t_horizon = t_to_bottom + GRIPPER_CLOSE_TIME
            lead_xy = self._predict_path_at(
                ball_pos[:2], self._ball_vel_est[:2], t_horizon,
            )
            target = np.array([lead_xy[0], lead_xy[1], grasp_z])
            action = self._move_to(
                eef_pos, eef_quat, target,
                gripper=GRIPPER_OPEN,
                gain_xy=GAIN_XY_DESCEND,
                gain_z=GAIN_Z_DESCEND,
            )
            close_xy = self._predict_path_at(
                ball_pos[:2], self._ball_vel_est[:2], GRIPPER_CLOSE_TIME,
            )
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

        # ── 4. Close gripper, leading ball with the bounce predictor.
        #         While the jaws shut, lead by ``t_remaining`` so the EE
        #         meets the ball at jaw-shut. *After* shut, keep a small
        #         lead so the OSC commands a sustained velocity matching
        #         the ball — the jaws drag it along instead of letting
        #         it roll out of the gripper basin.
        if self.phase == "close":
            grasp_z = max(
                float(ball_pos[2]) + GRASP_HEIGHT_ABOVE_BALL,
                self._table_z + EE_TABLE_CLEARANCE,
            )
            t_remaining = max(
                GRIPPER_CLOSE_TIME - self.step_in_phase * 0.01, 0.0,
            )
            # Always keep at least CLOSE_TRACK_LEAD seconds of lead, so
            # once the jaws have fully shut the EE keeps moving with the
            # ball at roughly the ball's speed.
            effective_lead = max(t_remaining, CLOSE_TRACK_LEAD)
            close_xy = self._predict_path_at(
                ball_pos[:2], self._ball_vel_est[:2], effective_lead,
            )
            target = np.array([close_xy[0], close_xy[1], grasp_z])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_CLOSE,
                                   gain_xy=GAIN_XY_CLOSE,
                                   gain_z=GAIN_Z_CLOSE,
                                   max_step=0.6)
            if self.step_in_phase >= CLOSE_STEPS:
                self._descend_xy = eef_pos[:2].copy()
                self._advance("lift")
            self.step_in_phase += 1
            return action

        # ── 5. Lift ──────────────────────────────────────────────────
        # As soon as the EE has lifted clear of the grasp height, check
        # whether the ball came up with us. A successful grasp pulls the
        # ball z well above its at-rest value (table_z + ball_radius);
        # a failed grasp leaves the ball rolling on the table while the
        # EE rises empty — we detect that and terminate the episode.
        if self.phase == "lift":
            target = np.array([self._descend_xy[0],
                               self._descend_xy[1],
                               self._table_z + LIFT_HEIGHT_ABOVE_TABLE])
            action = self._move_to(eef_pos, eef_quat, target,
                                   gripper=GRIPPER_CLOSE,
                                   gain_xy=GAIN_XY_LIFT,
                                   gain_z=GAIN_Z_LIFT)
            # Failure detection: EE has risen at least 4 cm above grasp
            # height but the ball is still resting on the table. Wait at
            # least GRASP_CHECK_DELAY ticks so the lift has produced
            # enough separation for the test to be meaningful.
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

        # ── 6. Hold ──────────────────────────────────────────────────
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

    env = ib.make("ib-p2-bounce-intercept-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = BounceInterceptPolicy()
    for ep in range(3):
        obs, info = env.reset(seed=ep * 23 + 1)
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
