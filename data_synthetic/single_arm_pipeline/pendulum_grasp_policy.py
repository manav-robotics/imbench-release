"""Reactive scripted policy for ``ib-p2-pendulum-grasp-v1``.

Strategy
--------
The pendulum hinge axis is world X; the bob swings in the YZ plane. At
the start of the episode the policy picks a side (``+Y`` or ``-Y``) at
random and grabs the bob from there.

Two interlocking constraints drive the design:

1. The Panda gripper's *wrist body* in the side-approach orientation
   reaches ~10 cm forward of the TCP site along the approach
   direction. With the gripper sitting at the chosen-side apex pose,
   the wrist's near edge sits *inside* the bob's swing arc, where it
   grazes the bob on every pass and damps the swing. Each damped
   swing lowers the bob's apex z below the success-tolerance window.

2. The gripper takes ~30 ticks to fully shut, but the bob only
   dwells within 1 cm of apex_z for ~50 ms at the swing extreme. So
   the close command must fire *before* the bob arrives.

The policy threads the needle by aiming to grab the bob on its
**first** return to the chosen-side apex — before any wrist contact
has had a chance to decay the swing:

1. ``orient_y`` — at the start X (well off the swing plane in -X),
   translate along Y to the chosen-side park position *outside* the
   swing arc, while rotating the wrist to the side-grasp quat. This
   keeps the EEF clear of the swinging bob's YZ plane during the
   large rotation. The robot starts at X≈0.05 while the swing plane
   sits at X=0.12, so any Y travel here happens outside the bob's
   reach.

2. ``orient_x`` — translate along +X into the swing plane, holding the
   chosen-side apex Y and Z. Approaches the bob laterally only after
   the EEF is already on the correct side.

3. ``wait`` — at the catch pose (``CATCH_OUTBOARD_Y`` outboard of the
   apex), gripper open. Each tick we forecast the steps until the
   bob's next visit to the chosen-side apex (ideal-pendulum
   quarter-period ``T/4 = π/2 · √(L/g)``), sign-flipped on the swing
   leg. When the forecast drops to ``GRIPPER_CLOSE_STEPS`` we latch
   the close command so the jaws are mid-shut as the bob arrives.

4. ``close`` — gripper latched to ``+1`` for the rest of the episode.
   EEF holds the catch pose; env terminates on success.
"""

from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface


# ── Side-grasp orientations (xyzw, matches robosuite obs convention) ─
# Built from rotation matrices whose columns are the desired TCP basis
# expressed in world coords. The Panda gripper's *local* frame has
# wrist-body extending in local -Z, jaws closing along local Y.
#
#   -Y-side approach: wrist body at -Y (away from bob), jaws facing
#   +Y (toward bob).
#   +Y-side approach: the mirror pose turned 180° about the approach
#   axis (same grasp with symmetric jaws); the plain mirror is beyond
#   the Panda wrist's joint range.
SIDE_QUAT_PLUS_Y  = np.array([ 0.5,  0.5, -0.5,  0.5])
SIDE_QUAT_MINUS_Y = np.array([ 0.5,  0.5,  0.5, -0.5])

# ── Geometry ─────────────────────────────────────────────────────────
GRAVITY = 9.81  # m/s^2

# ── Tolerances ──────────────────────────────────────────────────────
ORIENT_POS_TOL    = 0.010  # m, pads must end up within a few mm of
                           # apex_z, otherwise the closing jaws miss
                           # the bob centre in Y.
ORIENT_QUAT_TOL   = 0.15   # rad, axis-angle magnitude

# ── Forecast / timing ───────────────────────────────────────────────
# Empirical close time of the Panda jaws under saturated action:
# qpos goes from 0.04 (open) to first contact in ~12 ticks, full pinch
# in ~30. Firing the close command this many ticks before the bob's
# predicted apex arrival has the jaws partway shut on contact.
GRIPPER_CLOSE_STEPS = 14

# ── Gains ───────────────────────────────────────────────────────────
GAIN_XY = 12.0
GAIN_Z  = 12.0
ROT_GAIN = 3.5

# Velocity feed-forward gain used during predictive tracking. The OSC
# controller alone can't keep up with a fast-moving target on pure P
# control, so we add a term proportional to the bob's predicted Y/Z
# velocity at the lookahead horizon. Tuned empirically: too high →
# overshoot past Y=0; too low → TCP lags the bob.
TRACK_FF_GAIN = 0.2

GRIPPER_OPEN  = -1.0
GRIPPER_CLOSE =  1.0

# ── Geometry ─────────────────────────────────────────────────────────
# During orient_y the EE retracts this far in -X behind its starting
# position so it has more clearance from the swing plane while sweeping
# along Y and rotating to the side-grasp quat. The orient_x phase then
# advances the EE forward into the swing plane.
ORIENT_Y_RETRACT = 0.08  # m

# The catch pose sits this far outboard (away from Y=0) of the bob's apex.
# With the TCP at the apex the palm stops the bob ~3 cm of arc short of it,
# so the jaws close on it below the task's apex tolerance.
CATCH_OUTBOARD_Y = 0.02  # m

# ── Phase budgets ───────────────────────────────────────────────────
ORIENT_Y_MAX_STEPS = 250
ORIENT_X_MAX_STEPS = 120
WAIT_CLEAR_MAX_STEPS = 600   # cap so a degenerate swing can't stall us
CLEAR_CONFIRM_STEPS  = 2     # need this many consecutive ticks of
                             # "bob on opposite side, outbound" before
                             # committing to the +X approach

CTRL_DT = 0.01  # control_freq=100

# When the bob is on the opposite side during wait, nudge the TCP this
# far inboard (toward Y=0, i.e. toward the bob) so the OSC keeps moving
# instead of stalling at the apex hold. As soon as the bob crosses back
# to our side, predictive tracking takes over.
OPPOSITE_SIDE_INBOARD_NUDGE = 0.15  # m

# ── Grasp verification ──────────────────────────────────────────────
# Close fires when the analytic forecast says the bob will arrive at
# our apex in GRIPPER_CLOSE_STEPS ticks (defined above). The bob is
# still moving when the close fires, so we wait JAW_SETTLE_STEPS for
# the jaws to shut and the bob to arrive, then re-anchor the bob's
# position. From that anchor onward, the bob must stay within
# GRASP_DRIFT_TOL for GRASP_VERIFY_HORIZON ticks; any larger drift
# means the bob slipped → re-open.
JAW_SETTLE_STEPS = 35          # ticks before the drift check engages
GRASP_DRIFT_TOL = 0.05         # m
GRASP_VERIFY_HORIZON = 100     # ticks
# Cap how many re-attempts we make so a degenerate scene can't loop.
MAX_REGRASP_ATTEMPTS = 4


def _quat_err_axis_angle(cur_xyzw: np.ndarray, tgt_xyzw: np.ndarray) -> np.ndarray:
    """Axis-angle vector that rotates ``cur`` toward ``tgt`` (xyzw)."""
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


class PendulumGraspPolicy(PolicyInterface):
    """Reactive side-grasp policy for the swinging-sphere pendulum task."""

    def __init__(self, seed: int | None = None, rng=None):
        self.rng = np.random.RandomState(seed) if rng is None else rng
        self.reset()

    def reset(self):
        self.phase = "orient_y"
        self.step_in_phase = 0
        self._clear_seen_steps: int = 0

        self._side: int | None = None
        self._side_quat: np.ndarray | None = None

        self._apex_pos: np.ndarray | None = None
        self._L_eff: float | None = None
        self._start_x: float | None = None    # EEF X at episode start

        self._prev_bob_pos: np.ndarray | None = None
        self._bob_vel_est:  np.ndarray = np.zeros(3)

        self._close_latched: bool = False
        self._steps_since_close: int = 0
        self._regrasp_attempts: int = 0
        self._grasp_anchor: np.ndarray | None = None  # bob pos at close time

    # ── Geometry latching ──────────────────────────────────────────────
    def _latch_geometry(self, bob_pos: np.ndarray):
        """Initial bob obs is at rest at the release apex (qvel=0).

        L_eff is estimated under the conservative assumption that the
        release amplitude is ≤ 45° (config caps it there), so
        L_eff = |apex_y| / sin(45°). Slight overestimate biases the
        period forecast toward an early close trigger — the safer
        direction.
        """
        self._apex_pos = bob_pos.copy()
        y_off = abs(bob_pos[1])
        self._L_eff = float(max(y_off / np.sin(np.radians(45.0)), 0.30))

    # ── Side selection ────────────────────────────────────────────────
    def _pick_side(self):
        self._side = int(self.rng.choice([-1, +1]))
        self._side_quat = (
            SIDE_QUAT_PLUS_Y if self._side == +1 else SIDE_QUAT_MINUS_Y
        )

    def _chosen_apex_xyz(self) -> np.ndarray:
        """Static catch pose on the chosen side — used for orient_x
        and wait_for_clear, where the EE must drive to a fixed target
        rather than chase the moving bob.

        X tracks the live bob X.
        """
        ap = self._apex_pos
        apex_y = abs(ap[1])
        bob_x = float(self._prev_bob_pos[0]) if self._prev_bob_pos is not None else float(ap[0])
        return np.array([bob_x, self._side * (apex_y + CATCH_OUTBOARD_Y), float(ap[2])])

    def _lookahead_track(self, bob_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive catch target + feed-forward velocity.

        Ideal pendulum: ``θ(t) = amp·cos(ω·t)`` with ``t=0`` at the apex.
        We aim where the bob will be ``GRIPPER_CLOSE_STEPS`` ticks ahead,
        and also return the bob's predicted Y/Z velocity at that point
        so the OSC can be pushed feed-forward instead of relying on
        pure proportional tracking (which lags a fast target).

        Returns ``(target_xyz, ff_velocity_xyz)``. On outbound / opposite-
        side legs (more than a half-period from our next apex visit),
        the target is the static apex and FF velocity is zero.
        """
        ap = self._apex_pos
        apex_y = abs(ap[1])
        L = max(self._L_eff or 0.7, 1e-3)
        omega = float(np.sqrt(GRAVITY / L))
        amp = float(np.arcsin(min(apex_y / L, 1.0)))
        pivot_z = float(ap[2]) + L * float(np.cos(amp))

        steps_to_apex = self._steps_to_chosen_apex(bob_pos)
        t_half_steps = int(round((np.pi / omega) / CTRL_DT))

        # Only look ahead on the half-period leading up to our apex.
        # Earlier than that (bob is on the opposite side), nudge the
        # TCP 5 cm inboard toward the bob so the EE doesn't get stuck
        # holding the apex. Predictive tracking resumes once the bob
        # crosses back into our half-plane.
        if steps_to_apex > t_half_steps:
            apex = self._chosen_apex_xyz()
            apex_y = abs(self._apex_pos[1])
            nudge_y = self._side * max(apex_y - OPPOSITE_SIDE_INBOARD_NUDGE, 0.0)
            target = np.array([apex[0], nudge_y, apex[2]])
            return target, np.zeros(3)

        t_at_H = max(0.0, (steps_to_apex - GRIPPER_CLOSE_STEPS) * CTRL_DT)
        theta_H = amp * float(np.cos(omega * t_at_H))
        x_target = float(bob_pos[0])
        y_pred = self._side * (L * float(np.sin(abs(theta_H))) + CATCH_OUTBOARD_Y)
        z_pred = pivot_z - L * float(np.cos(theta_H))
        # Bob's analytic velocity at the lookahead point.
        # dθ/dt = -amp·ω·sin(ω·t). At t_at_H we're approaching apex,
        # so on our side dy/dt has sign +side (moving toward our apex).
        dtheta = -amp * omega * float(np.sin(omega * t_at_H))
        # |y| = L·sin(|θ|); d|y|/dt = L·cos(|θ|)·d|θ|/dt with sign
        # tracking d(|θ|)/dt = sign(θ)·dθ/dt.
        sign_theta = 1.0 if theta_H >= 0 else -1.0
        d_abs_y = L * float(np.cos(abs(theta_H))) * sign_theta * dtheta
        vy_ff = self._side * d_abs_y
        vz_ff = L * float(np.sin(theta_H)) * dtheta
        return (
            np.array([x_target, y_pred, z_pred]),
            np.array([0.0, vy_ff, vz_ff]),
        )

    # ── Action helpers ────────────────────────────────────────────────
    def _pose_action(
        self,
        eef_pos: np.ndarray,
        eef_quat: np.ndarray,
        target_pos: np.ndarray,
        gripper: float,
        ff_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        action = np.zeros(7)
        delta = target_pos - eef_pos
        ff = ff_vel if ff_vel is not None else np.zeros(3)
        action[0] = float(np.clip(delta[0] * GAIN_XY + ff[0] * TRACK_FF_GAIN, -1.0, 1.0))
        action[1] = float(np.clip(delta[1] * GAIN_XY + ff[1] * TRACK_FF_GAIN, -1.0, 1.0))
        action[2] = float(np.clip(delta[2] * GAIN_Z  + ff[2] * TRACK_FF_GAIN, -1.0, 1.0))
        ori_err = _quat_err_axis_angle(eef_quat, self._side_quat)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
        action[6] = gripper
        return action

    # ── Bob velocity estimator ────────────────────────────────────────
    def _update_bob_vel(self, bob_pos: np.ndarray):
        if self._prev_bob_pos is None:
            self._bob_vel_est = np.zeros(3)
        else:
            inst = (bob_pos - self._prev_bob_pos) / CTRL_DT
            self._bob_vel_est = 0.6 * self._bob_vel_est + 0.4 * inst
        self._prev_bob_pos = bob_pos.copy()

    # ── Apex arrival forecast ─────────────────────────────────────────
    def _steps_to_chosen_apex(self, bob_pos: np.ndarray) -> int:
        """Steps until the bob next reaches the chosen-side apex.

        Ideal pendulum: ``y(t) = ±|apex_y| · cos(ω·(t - t_apex))``.
        Sign of v_y distinguishes outbound/inbound legs of the cycle.
        We split into 4 quadrants of the swing (A=our apex, B=midpoint
        outbound, C=opposite apex, D=midpoint inbound) and add the
        per-quadrant time accordingly.
        """
        apex_y = abs(self._apex_pos[1])
        if apex_y < 1e-3:
            return 1
        v_y = float(self._bob_vel_est[1])
        bob_y = float(bob_pos[1])
        on_side = np.sign(bob_y) == self._side

        omega = float(np.sqrt(GRAVITY / max(self._L_eff, 1e-3)))
        t_quarter = 0.5 * np.pi / omega
        ratio = float(np.clip(abs(bob_y) / apex_y, 0.0, 1.0))
        t_to_extreme = float(np.arccos(ratio)) / omega
        t_to_mid     = t_quarter - t_to_extreme

        if on_side and np.sign(v_y) == self._side:
            # On our side, outbound (toward our apex).
            t_to_apex = t_to_extreme
        elif on_side:
            # On our side, inbound (away from our apex). Bob still
            # has to: (i) reach midpoint, (ii) cross to opposite apex,
            # (iii) come back to midpoint, (iv) reach our apex.
            t_to_apex = t_to_mid + 3.0 * t_quarter
        else:
            if np.sign(v_y) == -self._side:
                # Opposite side, outbound (toward opposite apex).
                # Then T/2 to come back to our apex.
                t_to_apex = t_to_extreme + 2.0 * t_quarter
            else:
                # Opposite side, inbound (toward midpoint, then ours).
                t_to_apex = t_to_mid + t_quarter

        return max(0, int(round(t_to_apex / CTRL_DT)))

    # ── Phase advance ─────────────────────────────────────────────────
    def _advance(self, next_phase: str):
        self.phase = next_phase
        self.step_in_phase = 0

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
        bob_pos  = np.asarray(obs["object_pos"], dtype=float)

        if self._apex_pos is None:
            self._latch_geometry(bob_pos)
            self._pick_side()
            self._start_x = float(eef_pos[0])
        self._update_bob_vel(bob_pos)
        self.step_in_phase += 1

        apex = self._chosen_apex_xyz()
        # Hold-X waypoint = apex Y/Z but pulled ORIENT_Y_RETRACT behind
        # the start X — extra clearance from the swing plane during
        # the Y sweep + wrist rotation. orient_x then advances forward.
        safe_x = self._start_x - ORIENT_Y_RETRACT
        y_waypoint = np.array([safe_x, apex[1], apex[2]])

        # ── 1. Orient_Y — sweep to chosen side at safe X ───────────────
        if self.phase == "orient_y":
            action = self._pose_action(
                eef_pos, eef_quat, y_waypoint,
                gripper=GRIPPER_OPEN,
            )
            pos_err = float(np.linalg.norm(y_waypoint - eef_pos))
            ori_err = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, self._side_quat)
            ))
            ready = pos_err < ORIENT_POS_TOL and ori_err < ORIENT_QUAT_TOL
            if ready or self.step_in_phase >= ORIENT_Y_MAX_STEPS:
                self._advance("wait_for_clear")
            return action

        # ── 1b. Wait_for_clear — hold off-plane until bob is on the
        # opposite side and moving away from us. Then we have ~half a
        # period to slide into the swing plane before the bob returns.
        if self.phase == "wait_for_clear":
            v_y = float(self._bob_vel_est[1])
            bob_y = float(bob_pos[1])
            on_opposite = np.sign(bob_y) == -self._side
            outbound    = np.sign(v_y) == -self._side
            if on_opposite and outbound and abs(v_y) > 1e-3:
                self._clear_seen_steps += 1
            else:
                self._clear_seen_steps = 0
            if (
                self._clear_seen_steps >= CLEAR_CONFIRM_STEPS
                or self.step_in_phase >= WAIT_CLEAR_MAX_STEPS
            ):
                self._advance("orient_x")
            return self._pose_action(
                eef_pos, eef_quat, y_waypoint, gripper=GRIPPER_OPEN,
            )

        # ── 2. Orient_X — translate into swing plane along +X ──────────
        if self.phase == "orient_x":
            action = self._pose_action(
                eef_pos, eef_quat, apex,
                gripper=GRIPPER_OPEN,
            )
            pos_err = float(np.linalg.norm(apex - eef_pos))
            if pos_err < ORIENT_POS_TOL or self.step_in_phase >= ORIENT_X_MAX_STEPS:
                self._advance("wait")
            return action

        # ── 3. Wait — predictive track; fire close when the bob's
        # forecast apex arrival is GRIPPER_CLOSE_STEPS ticks out;
        # verify by checking the bob stays put for GRASP_VERIFY_HORIZON.
        steps_to_apex = self._steps_to_chosen_apex(bob_pos)
        if not self._close_latched and steps_to_apex <= GRIPPER_CLOSE_STEPS:
            self._close_latched = True
            self._steps_since_close = 0
            self._grasp_anchor = None  # set after JAW_SETTLE_STEPS

        if self._close_latched:
            self._steps_since_close += 1
            # Anchor the bob position once the jaws have settled.
            if self._steps_since_close == JAW_SETTLE_STEPS:
                self._grasp_anchor = bob_pos.copy()
            # Past the settle window, drift > tol means the bob slipped.
            in_verify = (
                JAW_SETTLE_STEPS < self._steps_since_close
                <= JAW_SETTLE_STEPS + GRASP_VERIFY_HORIZON
            )
            if in_verify and self._grasp_anchor is not None:
                drift = float(np.linalg.norm(bob_pos - self._grasp_anchor))
                if (
                    drift > GRASP_DRIFT_TOL
                    and self._regrasp_attempts < MAX_REGRASP_ATTEMPTS
                ):
                    self._close_latched = False
                    self._steps_since_close = 0
                    self._grasp_anchor = None
                    self._regrasp_attempts += 1
                    self._clear_seen_steps = 0
                    self._advance("wait_for_clear")
                    return self._pose_action(
                        eef_pos, eef_quat, self._chosen_apex_xyz(),
                        gripper=GRIPPER_OPEN,
                    )

        # Once the jaws have settled around a stable bob, freeze the
        # lookahead and hold the static apex pose so the bob comes to
        # rest at apex_z and the env's success check can fire.
        if (
            self._close_latched
            and self._steps_since_close > JAW_SETTLE_STEPS
            and self._grasp_anchor is not None
        ):
            return self._pose_action(
                eef_pos, eef_quat, self._chosen_apex_xyz(),
                gripper=GRIPPER_CLOSE,
            )

        gripper = GRIPPER_CLOSE if self._close_latched else GRIPPER_OPEN
        target, ff_vel = self._lookahead_track(bob_pos)
        return self._pose_action(
            eef_pos, eef_quat, target, gripper=gripper, ff_vel=ff_vel,
        )
