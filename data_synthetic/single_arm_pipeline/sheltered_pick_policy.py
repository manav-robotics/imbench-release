"""
Sheltered Grasp Policy — ib-p1-sheltered-grasp-v1  (smooth trajectories)
=========================================================================

Strategy phases
---------------
 INIT → OPEN → RETREAT_Y → ROTATE → N_ROTATE →
 ALIGN_XZ → INSERT_Y → GRASP →
 EXTRACT_Y → LIFT → DEROTATE →
 TRANSPORT → LOWER → RELEASE → DONE
"""

import numpy as np
import imbench as ib

from data_synthetic.episode_length_stats import sample_target_length


# ─────────────────────────────────────────────────────────────────────────────
#  Teleop episode-length reference (Sheltered Grasp, success-only)
# ─────────────────────────────────────────────────────────────────────────────
# Step counts of the successful teleoperated episodes of this task.
# Used at reset() to sample a target rollout length from N(mean, std)
# truncated to [min, TARGET_HORIZON_MAX].
TELEOP_N_EPISODES = 329
TELEOP_MIN        = 1559
TELEOP_MEAN       = 3701.4
TELEOP_STD        = 1274.5
TELEOP_MAX        = 19097
# Episodes run about 2x their target (2.15x at worst), so longer targets
# overrun the task's 7000-step budget.
TARGET_HORIZON_MAX = 2500

# Empirical natural rollout (in env steps) for an unscaled run with the
# default GAIN_XY / GAIN_Z / GAIN_ROT / PATH_STEPS / *_HOLD constants.
# Measured from a single-seed scripted run before the time-scaling logic
# was added (~7 path phases at PATH_STEPS=70 each + 3 holds + tolerance
# settle). The actual scaling factor is target_horizon / this baseline,
# applied to PATH_STEPS, the *_HOLD counters, and inverse-applied to the
# motion gains so tolerance-driven phases stretch as well.
_NATURAL_BASELINE_STEPS = 720


# ─────────────────────────────────────────────────────────────────────────────
#  Tunable constants
# ─────────────────────────────────────────────────────────────────────────────

GAIN_XY   = 5.0
GAIN_Z    = 6.0
GAIN_ROT       = 4.0
GAIN_ROT_SLOW  = 0.6   # used only in ROTATE / N_ROTATE to reduce speed

XY_TOL  = 0.012   # m
Z_TOL   = 0.010   # m
ROT_TOL = 4.0     # degrees

Y_RETREAT_OFFSET      = 0.125  # m behind current Y before rotating
Y_APPROACH_CLEARANCE  = 0.20   # m outside rack face before insertion
Y_INSERT_OVERSHOOT    = 0.01   # m past object centre during insertion

TRANSPORT_Z_ABOVE_TABLE = 0.15
# EEF height above the table after release, clear of the cube: success
# requires the robot to stop touching it.
POST_RELEASE_Z_ABOVE_TABLE = 0.15

OPEN_HOLD    = 15
CLOSE_HOLD   = 40
RELEASE_HOLD = 30

DEFAULT_ROLL_DEG  = 180.0
DEFAULT_PITCH_DEG = -7.3
DEFAULT_YAW_DEG   = 0.0

# Overshoot fraction per phase  (0 = no overshoot, 0.15 = 15 % past target)
OVERSHOOT_RETREAT  = 0.18
OVERSHOOT_ALIGN    = 0.12
OVERSHOOT_INSERT   = 0.20   # intentionally overshoot into rack, then settle
OVERSHOOT_EXTRACT  = 0.14
OVERSHOOT_TRANSPORT= 0.10

# Arc curvature height for retreat and extract arcs (m, perpendicular bulge)
ARC_BULGE_RETREAT  = 0.08
ARC_BULGE_EXTRACT  = 0.06

# Interpolation resolution — steps to pre-compute per path
PATH_STEPS = 70

# Per-phase speedup multipliers. Applied on top of the global time-scale
# (target_horizon / _NATURAL_BASELINE_STEPS). >1.0 means the phase runs
# proportionally faster than the global pacing — fewer interpolation
# steps and higher gains. RETREAT_Y is a free-space arc with nothing
# delicate to align, so it can move quickly even on long-horizon runs.
# ROTATE / N_ROTATE are pure-wrist twists with no contact, similarly
# safe to accelerate. ALIGN_XZ is a free-space cartesian move from the
# retreat pose to the rack-face approach point — also no contact.
RETREAT_Y_SPEEDUP = 4.0
ROTATE_SPEEDUP    = 2.0
N_ROTATE_SPEEDUP  = 2.0
ALIGN_XZ_SPEEDUP  = 4.0
# INSERT_Y drives the gripper into the rack with a 20% overshoot —
# higher speedups make the contact harsher; tune down if jaws bounce.
INSERT_Y_SPEEDUP  = 4.0
# EXTRACT_Y / LIFT are free-space pulls with the cube already grasped —
# fine to accelerate. DEROTATE intentionally < 1.0 to reduce angular
# momentum on the held cube while the wrist swings back to defaults.
EXTRACT_Y_SPEEDUP = 4.0
LIFT_SPEEDUP      = 4.0
DEROTATE_SPEEDUP  = 0.5
# TRANSPORT is a free-space arc over the table to the tray — no contact,
# fine to accelerate.
TRANSPORT_SPEEDUP = 4.0


# ─────────────────────────────────────────────────────────────────────────────
#  Smooth interpolation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _raised_cosine(t: np.ndarray) -> np.ndarray:
    """Smooth S-curve from 0→1 with zero velocity at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


def _raised_cosine_deriv(t: np.ndarray) -> np.ndarray:
    """Derivative of raised cosine (velocity profile, un-normalised)."""
    return 0.5 * np.pi * np.sin(np.pi * np.clip(t, 0.0, 1.0))


class WaypointPath:
    """
    Multi-stop path with optional overshoot after the final waypoint.

    Parameters
    ----------
    waypoints : list of ndarray (3,)
        Ordered list of 3-D positions the path passes through.
        The first waypoint is the starting position.
    overshoot : float
        Fraction of the last segment to continue past the final waypoint
        before doubling back.  0 disables overshoot.
    arc_axis : ndarray (3,) or None
        If provided, a sinusoidal bulge of amplitude `arc_amplitude` is
        added perpendicularly along this axis for the *first* segment only,
        creating a curved arc departure.
    arc_amplitude : float
        Peak displacement (m) of the arc bulge.
    n_steps : int
        Number of discrete positions to pre-compute.
    """

    def __init__(self, waypoints, overshoot=0.0,
                 arc_axis=None, arc_amplitude=0.0,
                 n_steps=PATH_STEPS):

        waypoints = [np.array(w, dtype=float) for w in waypoints]

        # Inject an overshoot waypoint beyond the last segment
        if overshoot > 0.0 and len(waypoints) >= 2:
            delta = waypoints[-1] - waypoints[-2]
            beyond = waypoints[-1] + overshoot * delta
            # True goal is still the last original waypoint
            waypoints = waypoints[:-1] + [beyond, waypoints[-1]]

        self.waypoints = waypoints
        self.arc_axis  = (arc_axis / (np.linalg.norm(arc_axis) + 1e-9)
                          if arc_axis is not None else None)
        self.arc_amp   = arc_amplitude

        # Assign equal parameter share to each segment
        n_seg = len(waypoints) - 1
        seg_ts = np.linspace(0.0, 1.0, n_steps)           # global t ∈ [0,1]
        self._positions  = self._build(seg_ts, waypoints,
                                       n_seg, arc_axis, arc_amplitude)
        self._velocities = np.gradient(self._positions, axis=0)
        self._t_vals     = seg_ts
        self._cursor     = 0
        self.done        = False

    # ------------------------------------------------------------------
    def _build(self, t_global, waypoints, n_seg, arc_axis, arc_amp):
        """Evaluate the path for all t values."""
        positions = np.zeros((len(t_global), 3))
        seg_size  = 1.0 / n_seg

        for i, t in enumerate(t_global):
            seg_idx  = min(int(t / seg_size), n_seg - 1)
            t_local  = (t - seg_idx * seg_size) / seg_size   # 0→1 within seg
            s        = _raised_cosine(np.array([t_local]))[0]

            p0 = waypoints[seg_idx]
            p1 = waypoints[seg_idx + 1]
            pos = p0 + s * (p1 - p0)

            # Arc bulge on the first segment only
            if seg_idx == 0 and arc_axis is not None and arc_amp != 0.0:
                bulge = np.sin(np.pi * t_local) * arc_amp
                pos   = pos + bulge * arc_axis

            positions[i] = pos

        return positions

    # ------------------------------------------------------------------
    def advance_to(self, current_pos: np.ndarray, dt_gain: float = 1.0):
        """
        Return (desired_pos, feed_forward_vel) for the next step.

        The cursor is advanced whenever the EEF has come within XY_TOL of
        the current target, ensuring we never stall on a waypoint.
        """
        if self.done:
            return self._positions[-1], np.zeros(3)

        # Advance cursor if close enough to current sample
        while self._cursor < len(self._positions) - 1:
            d = np.linalg.norm(current_pos - self._positions[self._cursor])
            if d < XY_TOL * 1.5:
                self._cursor += 1
            else:
                break

        if self._cursor >= len(self._positions) - 1:
            self.done = True

        tgt = self._positions[self._cursor]
        vel = self._velocities[self._cursor] * dt_gain
        return tgt, vel


# ─────────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rpy_deg(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-9)
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return np.array([roll, pitch, yaw])


def angle_error_deg(current: float, target: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def clip_action(dx=0.0, dy=0.0, dz=0.0,
                droll=0.0, dpitch=0.0, dyaw=0.0,
                gripper=-1.0, max_t=1.0, max_r=1.0) -> np.ndarray:
    a = np.zeros(7)
    a[0] = np.clip(dx,     -max_t, max_t)
    a[1] = np.clip(dy,     -max_t, max_t)
    a[2] = np.clip(dz,     -max_t, max_t)
    a[3] = np.clip(droll,  -max_r, max_r)
    a[4] = np.clip(dpitch, -max_r, max_r)
    a[5] = np.clip(dyaw,   -max_r, max_r)
    a[6] = gripper
    return a


def no_move(gripper=-1.0) -> np.ndarray:
    return clip_action(gripper=gripper)


def _path_action(path: WaypointPath, eef_pos: np.ndarray,
                 gripper: float, ff_scale: float = 4.0,
                 gain_xy: float = GAIN_XY,
                 gain_z: float  = GAIN_Z) -> np.ndarray:
    """
    Given an active WaypointPath, compute proportional + feed-forward action.
    ff_scale: weight of feed-forward velocity term relative to proportional.
    gain_xy / gain_z default to the module constants but can be overridden
    by the policy's per-episode time-scaled values.
    """
    tgt, vel = path.advance_to(eef_pos)
    err   = tgt - eef_pos
    dx    = err[0] * gain_xy + vel[0] * ff_scale
    dy    = err[1] * gain_xy + vel[1] * ff_scale
    dz    = err[2] * gain_z  + vel[2] * ff_scale
    return clip_action(dx=dx, dy=dy, dz=dz, gripper=gripper)


# ─────────────────────────────────────────────────────────────────────────────
#  Policy
# ─────────────────────────────────────────────────────────────────────────────

class ShelteredGraspPolicy:
    """
    Side-entry policy with smooth waypoint trajectories, overshoot/correction,
    and curved arc motions for the retreat and extraction transitions.
    """

    PHASES = [
        "INIT", "OPEN", "RETREAT_Y", "ROTATE", "N_ROTATE",
        "ALIGN_XZ", "INSERT_Y", "GRASP",
        "EXTRACT_Y", "LIFT", "DEROTATE",
        "TRANSPORT", "LOWER", "RELEASE", "DONE",
    ]

    def __init__(self, env=None, rng=None):
        self.env = env
        self.rng = np.random.default_rng() if rng is None else rng
        self._reset_state()

    def reset(self):
        self._reset_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        if env is not None:
            self.env = env
        # Plan the existing side-entry path in a canonical left-side frame.
        # Reflection changes polar translations and axial rotations with
        # different signs. Nothing in the actual task observations/actions
        # is relabelled; this is only the oracle's internal planning frame.
        self._side_sign = 1.0 if np.cos(self.env._rack_yaw) >= 0 else -1.0
        action = self._act_inner(obs)
        if self._side_sign < 0:
            action[[1, 3, 5]] *= -1
        self.step_in_phase += 1
        return action

    # ── Internal state ────────────────────────────────────────────────────────

    def _reset_state(self):
        self.phase          = "INIT"
        self.step_in_phase  = 0

        self._table_z          = None
        self._tray_pos         = None
        self._obj_pos_snap     = None
        self._transport_z      = None
        self._y_insert_target  = None
        self._retreat_y_target = None

        # Active WaypointPath per phase (None when not in use)
        self._path: WaypointPath | None = None

        # Sample target rollout length from the teleop distribution and
        # derive a per-episode time scale. Path step counts and hold
        # counts get multiplied by this scale; motion gains get divided
        # by it so the tolerance-gated settle phases stretch in lockstep.
        self._target_horizon = sample_target_length(
            self.rng, TELEOP_MEAN, TELEOP_STD, TELEOP_MIN, TARGET_HORIZON_MAX,
        )
        self._time_scale = max(
            self._target_horizon / float(_NATURAL_BASELINE_STEPS),
            1e-3,
        )

        # Per-instance scaled durations and gains. Modules constants are
        # the un-scaled "fast scripted" baseline; phases read these
        # attributes instead of the module names.
        self.path_steps   = max(2, int(round(PATH_STEPS    * self._time_scale)))
        self.open_hold    = max(1, int(round(OPEN_HOLD     * self._time_scale)))
        self.close_hold   = max(1, int(round(CLOSE_HOLD    * self._time_scale)))
        self.release_hold = max(1, int(round(RELEASE_HOLD  * self._time_scale)))
        self.gain_xy       = GAIN_XY      / self._time_scale
        self.gain_z        = GAIN_Z       / self._time_scale
        self.gain_rot      = GAIN_ROT     / self._time_scale
        self.gain_rot_slow = GAIN_ROT_SLOW / self._time_scale
        print(f"  [sheltered] target_horizon={self._target_horizon} "
              f"scale={self._time_scale:.2f} "
              f"path_steps={self.path_steps} "
              f"holds=({self.open_hold},{self.close_hold},{self.release_hold}) "
              f"(teleop mean={TELEOP_MEAN:.0f}±{TELEOP_STD:.0f})")

    def _advance(self, next_phase: str):
        print(f"  → {self.phase} ({self.step_in_phase} steps) → {next_phase}")
        self.phase         = next_phase
        self.step_in_phase = 0
        self._path         = None   # clear old path; new phase will create its own

    # ── Observation helpers ───────────────────────────────────────────────────

    def _read(self, obs: dict):
        eef_pos  = np.array(obs["robot0_eef_pos"],  dtype=float)
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=float)
        obj_pos  = np.array(obs["object_pos"],       dtype=float)
        if self._side_sign < 0:
            eef_pos[1] *= -1
            obj_pos[1] *= -1
            # Reflect world Y and local X. The extra half-turn about the
            # jaw approach axis exchanges the symmetric fingers, choosing
            # the wrist branch that stays within Panda's joint limits.
            # R_canonical = diag(1,-1,1) R_world diag(-1,1,1).
            x, y, z, w = eef_quat
            eef_quat = np.array([y, x, w, z])
        eef_rpy  = quat_to_rpy_deg(eef_quat)
        return eef_pos, eef_quat, eef_rpy, obj_pos

    def _table_height(self) -> float:
        if self._table_z is None:
            try:
                self._table_z = float(self.env._table_height)
            except Exception:
                self._table_z = 0.80
        return self._table_z

    # ── Rotation helpers ──────────────────────────────────────────────────────

    def _rot_cmd(self, current: float, target: float) -> float:
        err = angle_error_deg(current, target)
        return float(np.clip(err * self.gain_rot / 90.0, -1.0, 1.0))

    def _rot_cmd_slow(self, current: float, target: float) -> float:
        """Same as _rot_cmd but uses gain_rot_slow — for ROTATE / N_ROTATE only."""
        err = angle_error_deg(current, target)
        return float(np.clip(err * self.gain_rot_slow / 90.0, -1.0, 1.0))

    def _rotation_done(self, rpy, roll, pitch, yaw=None) -> bool:
        re = abs(angle_error_deg(rpy[0], roll))
        pe = abs(angle_error_deg(rpy[1], pitch))
        ye = abs(angle_error_deg(rpy[2], yaw)) if yaw is not None else 0.0
        return re < ROT_TOL and pe < ROT_TOL and ye < ROT_TOL

    # ── State machine ─────────────────────────────────────────────────────────

    def _act_inner(self, obs: dict) -> np.ndarray:
        eef_pos, eef_quat, eef_rpy, obj_pos = self._read(obs)
        th = self._table_height()

        # ── INIT ──────────────────────────────────────────────────────────────
        if self.phase == "INIT":
            self._tray_pos = np.array(self.env._tray_pos, dtype=float)
            self._tray_pos[1] *= self._side_sign
            rack_yaw = float(self.env._rack_yaw)
            rack_pos = np.array(self.env._rack_pos, dtype=float)
            print(f"  [INIT] rack_pos={rack_pos}  rack_yaw={np.degrees(rack_yaw):.1f}°")
            print(f"  [INIT] obj_pos={np.round(obj_pos, 3)}")
            print(f"  [INIT] tray={self._tray_pos}  eef_rpy={np.round(eef_rpy, 2)}")
            if self.step_in_phase >= 5:
                self._advance("OPEN")
            return no_move(-1.0)

        # ── OPEN ──────────────────────────────────────────────────────────────
        elif self.phase == "OPEN":
            if self.step_in_phase >= self.open_hold:
                self._advance("RETREAT_Y")
            return no_move(-1.0)

        # ── RETREAT_Y ─────────────────────────────────────────────────────────
        # Curved arc: swing the arm back in Y while rising slightly in Z,
        # so the gripper lifts clear of the rack corner.
        elif self.phase == "RETREAT_Y":
            if self._path is None:
                tgt_y = eef_pos[1] - Y_RETREAT_OFFSET
                # Midpoint rises by ARC_BULGE_RETREAT to arc over the rack lip
                mid   = np.array([eef_pos[0],
                                   (eef_pos[1] + tgt_y) * 0.5,
                                   eef_pos[2] + ARC_BULGE_RETREAT])
                goal  = np.array([eef_pos[0], tgt_y, eef_pos[2]])
                # Speed up this phase: fewer interp samples + higher gain.
                fast_steps = max(2, int(round(self.path_steps / RETREAT_Y_SPEEDUP)))
                self._path = WaypointPath(
                    waypoints       = [eef_pos.copy(), mid, goal],
                    overshoot       = OVERSHOOT_RETREAT,
                    n_steps         = fast_steps,
                )
                self._retreat_y_target = tgt_y

            action = _path_action(self._path, eef_pos, gripper=-1.0,
                                  gain_xy=self.gain_xy * RETREAT_Y_SPEEDUP,
                                  gain_z =self.gain_z  * RETREAT_Y_SPEEDUP)

            if self._path.done:
                self._advance("ROTATE")

            return action

        # ── ROTATE ────────────────────────────────────────────────────────────
        elif self.phase == "ROTATE":
            target_roll, target_pitch, target_yaw = -90.0, 0.0, 0.0

            # Local speedup: scale the slow-rotation gain up so this
            # contact-free wrist twist resolves in fewer ticks.
            roll_err = angle_error_deg(eef_rpy[0], target_roll)
            yaw_err  = angle_error_deg(eef_rpy[2], target_yaw)
            gain     = self.gain_rot_slow * ROTATE_SPEEDUP
            droll = float(np.clip(roll_err * gain / 90.0, -1.0, 1.0))
            dyaw  = float(np.clip(yaw_err  * gain / 90.0, -1.0, 1.0))

            if abs(roll_err) < ROT_TOL and abs(yaw_err) < ROT_TOL:
                self._obj_pos_snap = obj_pos.copy()
                self._advance("N_ROTATE")

            return clip_action(droll=droll, dyaw=dyaw, gripper=-1.0)

        # ── N_ROTATE ──────────────────────────────────────────────────────────
        elif self.phase == "N_ROTATE":
            target_roll, target_pitch, target_yaw = -90.0, -90.0, 90.0

            # Local speedup: pitch-only twist, no contact.
            pitch_err = angle_error_deg(eef_rpy[1], target_pitch)
            gain      = self.gain_rot_slow * N_ROTATE_SPEEDUP
            dpitch = float(np.clip(pitch_err * gain / 90.0, -1.0, 1.0))

            if abs(pitch_err) < ROT_TOL:
                self._obj_pos_snap = obj_pos.copy()
                self._advance("ALIGN_XZ")

            return clip_action(dpitch=dpitch, gripper=-1.0)

        # ── ALIGN_XZ ──────────────────────────────────────────────────────────
        # Smooth path to (obj_x, safe_y, obj_z) with a small overshoot in XZ.
        elif self.phase == "ALIGN_XZ":
            obj    = self._obj_pos_snap if self._obj_pos_snap is not None else obj_pos
            rack_y = self.env._rack_pos[1] * self._side_sign
            safe_y = rack_y - Y_APPROACH_CLEARANCE
            goal   = np.array([obj[0], safe_y, obj[2]])

            if self._path is None:
                # Speed up this phase: fewer interp samples.
                fast_steps = max(2, int(round(self.path_steps / ALIGN_XZ_SPEEDUP)))
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), goal],
                    overshoot = OVERSHOOT_ALIGN,
                    n_steps   = fast_steps,
                )

            action = _path_action(self._path, eef_pos, gripper=-1.0,
                                  gain_xy=self.gain_xy * ALIGN_XZ_SPEEDUP,
                                  gain_z =self.gain_z  * ALIGN_XZ_SPEEDUP)

            # Advance only when fully settled (path done + within tolerance)
            xy_err = np.linalg.norm([eef_pos[0] - goal[0],
                                     eef_pos[1] - goal[1]])
            z_err  = abs(eef_pos[2] - goal[2])

            if self._path.done and xy_err < XY_TOL and z_err < Z_TOL:
                self._y_insert_target = obj[1] + Y_INSERT_OVERSHOOT
                self._advance("INSERT_Y")

            return action

        # ── INSERT_Y ──────────────────────────────────────────────────────────
        # Slide along +Y into the rack.  Overshoot a little past centre then
        # the path corrects back — natural "probe and settle" motion.
        elif self.phase == "INSERT_Y":
            obj    = self._obj_pos_snap if self._obj_pos_snap is not None else obj_pos
            tgt_y  = self._y_insert_target - 0.05
            goal   = np.array([obj[0], tgt_y, obj[2]])

            if self._path is None:
                # Speed up this phase: fewer interp samples.
                fast_steps = max(2, int(round(self.path_steps / INSERT_Y_SPEEDUP)))
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), goal],
                    overshoot = OVERSHOOT_INSERT,
                    n_steps   = fast_steps,
                )

            action = _path_action(self._path, eef_pos, gripper=-1.0,
                                  gain_xy=self.gain_xy * INSERT_Y_SPEEDUP,
                                  gain_z =self.gain_z  * INSERT_Y_SPEEDUP)

            y_err = abs(eef_pos[1] - tgt_y)
            x_err = abs(eef_pos[0] - obj[0])
            z_err = abs(eef_pos[2] - obj[2])

            if self._path.done and y_err < XY_TOL and x_err < XY_TOL and z_err < Z_TOL:
                self._advance("GRASP")

            return action

        # ── GRASP ─────────────────────────────────────────────────────────────
        elif self.phase == "GRASP":
            if self.step_in_phase >= self.close_hold:
                self._advance("EXTRACT_Y")
            return clip_action(gripper=1.0)

        # ── EXTRACT_Y ─────────────────────────────────────────────────────────
        # Pull back out with a gentle upward arc to lift the cube clear of the
        # rack's lower shelf as it exits.
        elif self.phase == "EXTRACT_Y":
            obj    = self._obj_pos_snap
            rack_y = self.env._rack_pos[1] * self._side_sign
            safe_y = rack_y - Y_APPROACH_CLEARANCE

            if self._path is None:
                mid  = np.array([obj[0],
                                  (eef_pos[1] + safe_y) * 0.5,
                                  eef_pos[2] + ARC_BULGE_EXTRACT])
                goal = np.array([obj[0], safe_y, eef_pos[2]])
                # Speed up this phase: fewer interp samples.
                fast_steps = max(2, int(round(self.path_steps / EXTRACT_Y_SPEEDUP)))
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), mid, goal],
                    overshoot = OVERSHOOT_EXTRACT,
                    n_steps   = fast_steps,
                )

            action = _path_action(self._path, eef_pos, gripper=1.0,
                                  gain_xy=self.gain_xy * EXTRACT_Y_SPEEDUP,
                                  gain_z =self.gain_z  * EXTRACT_Y_SPEEDUP)

            if self._path.done and abs(eef_pos[1] - safe_y) < XY_TOL:
                self._transport_z = th + TRANSPORT_Z_ABOVE_TABLE
                self._advance("LIFT")

            return action

        # ── LIFT ──────────────────────────────────────────────────────────────
        # Smooth vertical rise with a tiny overshoot (a clean "pop" upward).
        elif self.phase == "LIFT":
            tz = self._transport_z + 0.25

            if self._path is None:
                # Build a single-segment vertical path
                goal = np.array([eef_pos[0], eef_pos[1], tz])
                # Speed up this phase: fewer interp samples.
                fast_steps = max(2, int(round(self.path_steps / LIFT_SPEEDUP)))
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), goal],
                    overshoot = 0.06,   # slight upward overshoot
                    n_steps   = fast_steps,
                )

            action = _path_action(self._path, eef_pos, gripper=1.0,
                                  gain_xy=self.gain_xy * LIFT_SPEEDUP,
                                  gain_z =self.gain_z  * LIFT_SPEEDUP)

            if self._path.done and eef_pos[2] > tz - 0.012:
                self._advance("DEROTATE")

            return action

        # ── DEROTATE ──────────────────────────────────────────────────────────
        elif self.phase == "DEROTATE":
            # Local slowdown: reduce rotation gain so the cube experiences
            # less angular acceleration as the wrist swings back to the
            # transport defaults. SPEEDUP < 1.0 means slower than nominal.
            roll_err  = angle_error_deg(eef_rpy[0], DEFAULT_ROLL_DEG)
            pitch_err = angle_error_deg(eef_rpy[1], DEFAULT_PITCH_DEG)
            yaw_err   = angle_error_deg(eef_rpy[2], DEFAULT_YAW_DEG)
            gain      = self.gain_rot * DEROTATE_SPEEDUP
            droll  = float(np.clip(roll_err  * gain / 90.0, -1.0, 1.0))
            dpitch = float(np.clip(pitch_err * gain / 90.0, -1.0, 1.0))
            dyaw   = float(np.clip(yaw_err   * gain / 90.0, -1.0, 1.0))

            if self._rotation_done(eef_rpy, DEFAULT_ROLL_DEG, DEFAULT_PITCH_DEG, DEFAULT_YAW_DEG):
                self._advance("TRANSPORT")

            return clip_action(droll=droll, dpitch=dpitch, dyaw=dyaw, gripper=1.0)

        # ── TRANSPORT ─────────────────────────────────────────────────────────
        # Curved arc over the table: rise to transport height while sweeping XY,
        # with a gentle overshoot past the tray that settles back.
        elif self.phase == "TRANSPORT":
            txy = self._tray_pos
            tz  = self._transport_z

            if self._path is None:
                # Mid-arc waypoint: halfway in XY, already at transport Z
                mid  = np.array([(eef_pos[0] + txy[0]) * 0.5,
                                  (eef_pos[1] + txy[1]) * 0.5,
                                  tz + 0.06])  # small extra height mid-arc
                goal = np.array([txy[0], txy[1], tz])
                # Speed up this phase: fewer interp samples.
                fast_steps = max(2, int(round(self.path_steps / TRANSPORT_SPEEDUP)))
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), mid, goal],
                    overshoot = OVERSHOOT_TRANSPORT,
                    n_steps   = fast_steps,
                )

            action = _path_action(self._path, eef_pos, gripper=1.0,
                                  gain_xy=self.gain_xy * TRANSPORT_SPEEDUP,
                                  gain_z =self.gain_z  * TRANSPORT_SPEEDUP)

            if self._path.done and np.linalg.norm(eef_pos[:2] - txy) < XY_TOL:
                self._advance("LOWER")

            return action

        # ── LOWER ─────────────────────────────────────────────────────────────
        elif self.phase == "LOWER":
            txy     = self._tray_pos
            place_z = th + 0.04

            if self._path is None:
                goal = np.array([txy[0], txy[1], place_z])
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), goal],
                    overshoot = 0.0,   # no overshoot while lowering (safety)
                    n_steps   = self.path_steps,
                )

            action = _path_action(self._path, eef_pos, gripper=1.0,
                                  gain_xy=self.gain_xy, gain_z=self.gain_z)

            obj_near = obj_pos[2] < th + 0.06
            if obj_near or eef_pos[2] < place_z + 0.015:
                self._advance("RELEASE")

            return action

        # ── RELEASE ───────────────────────────────────────────────────────────
        elif self.phase == "RELEASE":
            if self.step_in_phase >= self.release_hold:
                self._advance("DONE")
            return no_move(-1.0)

        # ── DONE ──────────────────────────────────────────────────────────────
        # Rise straight up with the gripper open so no finger rests on the
        # placed cube. Tolerance-driven phases stretch via the inverse-scaled
        # motion gains, and tick-driven phases via the scaled hold/path
        # counters, so the total rollout tracks the sampled target length.
        elif self.phase == "DONE":
            dz = GAIN_Z * (th + POST_RELEASE_Z_ABOVE_TABLE - eef_pos[2])
            return clip_action(dz=max(dz, 0.0), gripper=-1.0)

        return np.zeros(7)


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env    = ib.make("ib-p1-sheltered-grasp-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = ShelteredGraspPolicy(env)

    n_episodes = 5
    for ep in range(n_episodes):
        obs, info = env.reset()
        policy.reset()
        done       = False
        prev_phase = policy.phase
        step       = 0

        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if policy.phase != prev_phase:
                obj_pos = obs.get("object_pos", np.zeros(3))
                eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
                eef_rpy = quat_to_rpy_deg(
                    np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1])))
                print(
                    f"  ep={ep}  step={step:4d}  "
                    f"phase={prev_phase:12s}  "
                    f"obj_z={obj_pos[2]:.3f}  eef_z={eef_pos[2]:.3f}  "
                    f"eef_rpy={np.round(eef_rpy, 1)}  "
                    f"score={info.get('task_score', 0):.3f}"
                )
                prev_phase = policy.phase

            done = terminated or truncated
            step += 1

        print(
            f"\nEpisode {ep}  steps={step}  score={info.get('task_score', 0):.3f}"
            f"  final_phase={policy.phase}\n"
        )

    env.close()
