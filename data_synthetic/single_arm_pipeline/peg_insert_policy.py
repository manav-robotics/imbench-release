"""
peg_insert_policy.py
────────────────────
Policy for ib-p1-recover-peg-insert-v1.

Confirmed geometry:
  - Peg body: cube_main (body idx 27), joint: cube_joint0
  - Slot body: slot_main (body idx 28)
  - hole_center_world ≈ [0.003, 0.145, 1.125]
  - hole_normal_world  = [0, +1, 0]  (insertion is world +Y)
  - Entrance is on the −Y side (robot side), exit is +Y
  - Peg spawns upright: long_axis = [0,0,1], quat=[1,0,0,0]
  - Tip needed: long_axis → [0,1,0], i.e. −90° pitch about world X

Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]
  gripper: +1 = close, −1 = open
  All translational deltas in world frame.

Phase plan:
  INIT → OPEN → HOVER_ABOVE → DESCEND → GRASP → LIFT
  → TIP_HORIZONTAL → APPROACH → ALIGN_FINE → PUSH → RECOVER → RELEASE → DONE
"""

import numpy as np
import imbench as ib


# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────────────

# Heights / offsets
HOVER_Z_ABOVE_PEG    = 0.10   # above peg centroid when hovering
LIFT_Z_ABOVE_TABLE   = 0.15   # absolute height during carry (table=0.775+pad)
APPROACH_Y_OFFSET    = 0.3   # how far in front of entrance to stage (−Y side)

# Gains
GAIN_XY   = 6.0
GAIN_Z    = 6.0
GAIN_FAST = 12.0
GAIN_ROT  = 3.0    # rotation gain for tip phase

# Tolerances
XY_TOL   = 0.012
Z_TOL    = 0.010
ROT_TOL  = 0.05    # radians, for tip phase convergence

# Hold durations (steps)
OPEN_HOLD    = 20
CLOSE_HOLD   = 50
LIFT_HOLD    = 30

# Grasp descend budget
DESCEND_BUDGET = 200

# Push speed
PUSH_GAIN = 3.5

# Wrist correction during push (keeps alignment while perturb fires)
ALIGN_CORRECTION_GAIN = 4.0

# Alignment thresholds (from task)
ALIGN_THRESHOLD  = 0.92
LATERAL_TOL      = 0.015
INSERT_DEPTH_THR = 0.072   # (2*insert_fraction - 1)*peg_half_length

# Grasp height: grip at mid-shaft of the standing peg.
# Peg centroid is at table + peg_half_length. We grip slightly below centroid.
GRASP_Z_BELOW_CENTROID = 0.01

# ── WaypointPath constants ───────────────────────────────────────────────────

# Overshoot fraction per phase  (0 = no overshoot, 0.15 = 15 % past target)
OVERSHOOT_HOVER     = 0.00   # purely vertical — no overshoot needed
OVERSHOOT_DESCEND   = 0.08   # slight downward probe
OVERSHOOT_LIFT      = 0.07   # soft upward pop
OVERSHOOT_APPROACH  = 0.12   # forward swoop past approach point
OVERSHOOT_ALIGN     = 0.10   # XZ settle
OVERSHOOT_PULLBACK  = 0.12   # regrasp pullback
OVERSHOOT_TRANSPORT = 0.10   # regrasp re-approach

# Arc curvature (sinusoidal bulge, m) for curved transitions
ARC_BULGE_HOVER    = 0.06   # lateral safety arc when moving to hover
ARC_BULGE_LIFT     = 0.04   # small lateral bulge during lift to clear table

# Interpolation resolution — steps to pre-compute per path
PATH_STEPS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Smooth interpolation helpers  (ported from ShelteredGraspPolicy)
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
            delta  = waypoints[-1] - waypoints[-2]
            beyond = waypoints[-1] + overshoot * delta
            # True goal is still the last original waypoint
            waypoints = waypoints[:-1] + [beyond, waypoints[-1]]

        self.waypoints = waypoints
        self.arc_axis  = (arc_axis / (np.linalg.norm(arc_axis) + 1e-9)
                          if arc_axis is not None else None)
        self.arc_amp   = arc_amplitude

        # Assign equal parameter share to each segment
        n_seg    = len(waypoints) - 1
        seg_ts   = np.linspace(0.0, 1.0, n_steps)
        self._positions  = self._build(seg_ts, waypoints, n_seg,
                                       arc_axis, arc_amplitude)
        self._velocities = np.gradient(self._positions, axis=0)
        self._t_vals     = seg_ts
        self._cursor     = 0
        self.done        = False

    # ------------------------------------------------------------------
    def _build(self, t_global, waypoints, n_seg, arc_axis, arc_amp):
        positions = np.zeros((len(t_global), 3))
        seg_size  = 1.0 / n_seg

        for i, t in enumerate(t_global):
            seg_idx = min(int(t / seg_size), n_seg - 1)
            t_local = (t - seg_idx * seg_size) / seg_size   # 0→1 within seg
            s       = _raised_cosine(np.array([t_local]))[0]

            p0  = waypoints[seg_idx]
            p1  = waypoints[seg_idx + 1]
            pos = p0 + s * (p1 - p0)

            # Arc bulge on the first segment only
            if seg_idx == 0 and arc_axis is not None and arc_amp != 0.0:
                bulge = np.sin(np.pi * t_local) * arc_amp
                pos   = pos + bulge * (arc_axis / (np.linalg.norm(arc_axis) + 1e-9))

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


def _path_action(path: WaypointPath, eef_pos: np.ndarray,
                 gripper: float, ff_scale: float = 4.0,
                 droll: float = 0.0, dpitch: float = 0.0,
                 dyaw: float = 0.0) -> np.ndarray:
    """
    Given an active WaypointPath, compute proportional + feed-forward action.
    Rotation deltas are passed through unchanged (caller handles rot errors).
    ff_scale: weight of feed-forward velocity term relative to proportional.
    """
    tgt, vel = path.advance_to(eef_pos)
    err = tgt - eef_pos
    dx  = err[0] * GAIN_XY + vel[0] * ff_scale
    dy  = err[1] * GAIN_XY + vel[1] * ff_scale
    dz  = err[2] * GAIN_Z  + vel[2] * ff_scale
    return _clip(dx, dy, dz, droll, dpitch, dyaw, gripper)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip(dx, dy, dz, droll, dpitch, dyaw, grip, max_t=1.0, max_r=1.0):
    a = np.zeros(7)
    a[0] = np.clip(dx,     -max_t, max_t)
    a[1] = np.clip(dy,     -max_t, max_t)
    a[2] = np.clip(dz,     -max_t, max_t)
    a[3] = np.clip(droll,  -max_r, max_r)
    a[4] = np.clip(dpitch, -max_r, max_r)
    a[5] = np.clip(dyaw,   -max_r, max_r)
    a[6] = grip
    return a


def _noop(grip=-1.0):
    return _clip(0, 0, 0, 0, 0, 0, grip)


def _peg_pos_world(sim) -> np.ndarray:
    """Read peg centroid directly from MuJoCo (obs['object_pos'] was zeros)."""
    bid = sim.model.body_name2id("cube_main")
    return sim.data.body_xpos[bid].copy()


def _peg_long_axis(sim) -> np.ndarray:
    """Peg long axis in world frame. Local +Z rotated by body xmat."""
    bid = sim.model.body_name2id("cube_main")
    xmat = sim.data.body_xmat[bid].reshape(3, 3)
    return xmat @ np.array([0., 0., 1.])


def _peg_quat_xyzw(sim) -> np.ndarray:
    """Peg quaternion in robosuite [x,y,z,w] convention."""
    bid = sim.model.body_name2id("cube_main")
    w, x, y, z = sim.data.body_xquat[bid]
    return np.array([x, y, z, w])


def _axis_angle_to_action_rot(current_axis: np.ndarray,
                               target_axis: np.ndarray) -> np.ndarray:
    """
    Compute a world-frame rotation delta (roll, pitch, yaw) to steer
    current_axis toward target_axis.

    Uses the cross product as the rotation axis and the sine of the angle
    as magnitude, scaled by ALIGN_CORRECTION_GAIN.
    Returns [droll, dpitch, dyaw].
    Negated so the roll goes in the correct direction for this robot.
    """
    c = np.cross(current_axis, target_axis)   # rotation axis × sin(angle)
    return np.clip(-c * ALIGN_CORRECTION_GAIN, -1.0, 1.0)


def _tactile_contact(obs) -> bool:
    """True if either tactile pad has meaningful contact."""
    tl = np.asarray(obs.get("robot0_tactile_left",  np.zeros((8,8))), dtype=float)
    tr = np.asarray(obs.get("robot0_tactile_right", np.zeros((8,8))), dtype=float)
    return float(tl.max()) > 0.1 or float(tr.max()) > 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class PegInsertPolicy:
    """
    Phase-based policy for peg insertion with perturbation recovery.

    Phases
    ──────
    INIT            → settle, read initial state
    OPEN            → open gripper
    HOVER_ABOVE     → move eef to [peg_x, peg_y, peg_z + hover]
    DESCEND         → lower onto peg mid-shaft
    GRASP           → close gripper, wait
    LIFT            → raise to carry height
    TIP_HORIZONTAL  → rotate wrist so peg long axis → [0,1,0] (hole_normal)
    APPROACH        → move to approach waypoint in front of hole entrance
    ALIGN_FINE      → fine XZ centering, alignment check
    PUSH            → drive +Y with XZ correction + live alignment correction
    RELEASE         → open gripper
    DONE
    """

    def __init__(self, env=None):
        self.env = env
        self._reset_state()

    def reset(self):
        self._reset_state()

    def _reset_state(self):
        self.phase         = "INIT"
        self.step_in_phase = 0
        self._hole_c       = None
        self._hole_n       = None
        self._table_z      = None
        self._peg_half_len = None
        self._insert_thr   = None
        self._lat_tol      = None
        self._align_thr    = None
        # One WaypointPath slot reused across phases that need it
        self._path: WaypointPath | None = None

    def _advance(self, next_phase):
        eef = self._last_eef_pos if hasattr(self, "_last_eef_pos") else np.zeros(3)
        print(f"  [Policy] {self.phase:20s} → {next_phase:20s}  "
              f"step_in_phase={self.step_in_phase}  eef_z={eef[2]:.3f}")
        self.phase = next_phase
        self.step_in_phase = 0
        self._path = None   # clear path; new phase will build its own

    def _init_env_constants(self):
        """Cache task constants once on first act() call."""
        if self._hole_c is not None:
            return
        env = self.env
        self._hole_c       = env._hole_center_world.copy()
        self._hole_n       = env._hole_normal_world.copy()   # [0,1,0]
        self._table_z      = float(env._table_height)
        self._peg_half_len = float(env._peg_half_length)
        self._insert_thr   = float(env._insert_depth_threshold)
        self._lat_tol      = float(env._lateral_tolerance)
        self._align_thr    = float(env._alignment_threshold)

    def _read(self, obs):
        self._init_env_constants()
        sim = self.env._env.sim

        eef_pos  = np.asarray(obs["robot0_eef_pos"],  dtype=float)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=float)  # xyzw
        peg_pos  = _peg_pos_world(sim)
        peg_axis = _peg_long_axis(sim)

        self._last_eef_pos = eef_pos
        return eef_pos, eef_quat, peg_pos, peg_axis

    def _hole_rel(self, peg_pos, peg_axis):
        """Return (axial, lateral, align) of peg relative to hole."""
        hole_c = self._hole_c
        hole_n = self._hole_n
        rel     = peg_pos - hole_c
        axial   = float(np.dot(rel, hole_n))
        lat_vec = rel - axial * hole_n
        lateral = float(np.linalg.norm(lat_vec))
        axis_u  = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
        align   = abs(float(np.dot(axis_u, hole_n)))
        return axial, lateral, align

    # ──────────────────────────────────────────────────────────────────────────

    def act(self, obs, env=None):
        if env is not None:
            self.env = env
        action = self._act_inner(obs)
        self.step_in_phase += 1
        return action

    def _act_inner(self, obs):
        eef_pos, eef_quat, peg_pos, peg_axis = self._read(obs)
        hole_c = self._hole_c
        hole_n = self._hole_n   # [0, 1, 0]
        th     = self._table_z

        axial, lateral, align = self._hole_rel(peg_pos, peg_axis)

        # ── INIT ──────────────────────────────────────────────────────────────
        if self.phase == "INIT":
            if self.step_in_phase >= 5:
                self._advance("OPEN")
            return _noop(-1.0)

        # ── OPEN ──────────────────────────────────────────────────────────────
        elif self.phase == "OPEN":
            if self.step_in_phase >= OPEN_HOLD:
                self._advance("HOVER_ABOVE")
            return _noop(-1.0)

        # ── HOVER_ABOVE ───────────────────────────────────────────────────────
        # Smooth arc: lateral safety bulge in X so the gripper swings clear
        # of any table clutter before descending onto the peg.
        elif self.phase == "HOVER_ABOVE":
            target = np.array([peg_pos[0], peg_pos[1],
                                peg_pos[2] + HOVER_Z_ABOVE_PEG])

            if self._path is None:
                self._path = WaypointPath(
                    waypoints     = [eef_pos.copy(), target],
                    overshoot     = OVERSHOOT_HOVER,
                    arc_axis      = np.array([1.0, 0.0, 0.0]),  # X bulge
                    arc_amplitude = ARC_BULGE_HOVER,
                    n_steps       = PATH_STEPS,
                )

            action = _path_action(self._path, eef_pos, gripper=-1.0)

            delta = target - eef_pos
            if (self._path.done
                    and np.linalg.norm(delta[:2]) < XY_TOL
                    and abs(delta[2]) < Z_TOL):
                self._advance("DESCEND")
            return action

        # ── DESCEND ───────────────────────────────────────────────────────────
        # Smooth descent with a slight overshoot so the gripper probes
        # contact naturally rather than stopping dead above the peg.
        elif self.phase == "DESCEND":
            grasp_z = peg_pos[2] - GRASP_Z_BELOW_CENTROID
            target  = np.array([peg_pos[0], peg_pos[1], grasp_z + 0.07])

            if self._path is None:
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), target],
                    overshoot = OVERSHOOT_DESCEND,
                    n_steps   = PATH_STEPS,
                )

            delta   = target - eef_pos
            xy_err  = np.linalg.norm(delta[:2])
            z_err   = abs(delta[2])
            contact = _tactile_contact(obs)

            action = _path_action(self._path, eef_pos, gripper=-1.0)

            if ((self._path.done and xy_err < XY_TOL and z_err < Z_TOL)
                    or contact
                    or self.step_in_phase >= DESCEND_BUDGET):
                self._advance("GRASP")
            return action

        # ── GRASP ─────────────────────────────────────────────────────────────
        elif self.phase == "GRASP":
            # Keep pressing down slightly while closing.
            dz = (peg_pos[2] - eef_pos[2]) * GAIN_Z * 0.3
            if self.step_in_phase >= CLOSE_HOLD:
                self._advance("LIFT")
            return _clip(0, 0, dz, 0, 0, 0, 1.0)

        # ── LIFT ──────────────────────────────────────────────────────────────
        # Raised-cosine lift with a gentle lateral arc to swing the peg
        # clear of the table surface as we rise.
        elif self.phase == "LIFT":
            target_z = th + LIFT_Z_ABOVE_TABLE

            if self._path is None:
                goal = np.array([eef_pos[0], eef_pos[1], target_z])
                self._path = WaypointPath(
                    waypoints     = [eef_pos.copy(), goal],
                    overshoot     = OVERSHOOT_LIFT,
                    arc_axis      = np.array([1.0, 0.0, 0.0]),  # X bulge
                    arc_amplitude = ARC_BULGE_LIFT,
                    n_steps       = PATH_STEPS,
                )

            action = _path_action(self._path, eef_pos, gripper=1.0)

            if (self._path.done
                    and eef_pos[2] >= target_z - 0.012
                    and self.step_in_phase >= LIFT_HOLD):
                self._advance("TIP_HORIZONTAL")
            return action

        # ── TIP_HORIZONTAL ────────────────────────────────────────────────────
        elif self.phase == "TIP_HORIZONTAL":
            # Peg long axis is currently world +Z = [0,0,1].
            # Need it → hole_n = [0,1,0].
            # Cross([0,0,1], [0,1,0]) = [-1,0,0] → negative roll.
            # Drive action[3] (roll) negative.
            #
            # Use _axis_angle_to_action_rot for a proper error signal.
            target_axis  = hole_n.copy()   # [0,1,0]
            current_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
            dot          = float(np.dot(current_axis, target_axis))

            drot = _axis_angle_to_action_rot(current_axis, target_axis)
            droll, dpitch, dyaw = drot

            # Hold position while rotating.
            hold_target = np.array([hole_c[0],
                                     hole_c[1] - APPROACH_Y_OFFSET,
                                     hole_c[2]])
            pos_delta = hold_target - eef_pos
            dx = pos_delta[0] * GAIN_XY * 0.5
            dy = pos_delta[1] * GAIN_XY * 0.5
            dz = pos_delta[2] * GAIN_Z  * 0.5

            tipped = dot > ALIGN_THRESHOLD

            if self.step_in_phase % 50 == 0:
                print(f"    [TIP] step={self.step_in_phase} dot={dot:.3f} "
                      f"axis={np.round(current_axis,3)} "
                      f"drot={np.round(drot,3)}")

            if tipped or self.step_in_phase > 300:
                self._advance("APPROACH")
            return _clip(dx, dy, dz, droll, dpitch, dyaw, 1.0)

        # ── APPROACH ─────────────────────────────────────────────────────────
        # Smooth waypoint path to the approach staging point, with overshoot
        # so the robot swoops past and settles — mirrors ShelteredGrasp INSERT_Y.
        elif self.phase == "APPROACH":
            target = np.array([hole_c[0],
                                hole_c[1] - APPROACH_Y_OFFSET,
                                hole_c[2]])

            if self._path is None:
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), target],
                    overshoot = OVERSHOOT_APPROACH,
                    n_steps   = PATH_STEPS,
                )

            current_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
            drot = _axis_angle_to_action_rot(current_axis, hole_n)
            droll, dpitch, dyaw = drot

            action = _path_action(self._path, eef_pos, gripper=1.0,
                                  droll=droll, dpitch=dpitch, dyaw=dyaw)

            delta  = target - eef_pos
            xy_err = np.linalg.norm(delta[:2])
            z_err  = abs(delta[2])

            if self._path.done and xy_err < XY_TOL * 1.5 and z_err < Z_TOL * 1.5:
                self._advance("ALIGN_FINE")
            return action

        # ── ALIGN_FINE ────────────────────────────────────────────────────────
        # Smooth XZ centering with overshoot — lets the peg oscillate briefly
        # around the hole axis before the final push.
        elif self.phase == "ALIGN_FINE":
            # Target: peg centroid aligned to hole_c on X and Z,
            # Y held at the approach offset.
            goal = np.array([hole_c[0],
                             eef_pos[1],   # Y unchanged — hold approach Y
                             hole_c[2]])

            if self._path is None:
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), goal],
                    overshoot = OVERSHOOT_ALIGN,
                    n_steps   = PATH_STEPS,
                )

            current_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
            drot = _axis_angle_to_action_rot(current_axis, hole_n)
            droll, dpitch, dyaw = drot

            action = _path_action(self._path, eef_pos, gripper=1.0,
                                  droll=droll, dpitch=dpitch, dyaw=dyaw)

            x_ok   = abs(peg_pos[0] - hole_c[0]) < 0.007
            z_ok   = abs(peg_pos[2] - hole_c[2]) < 0.007
            rot_ok = align > 0.95

            if self.step_in_phase % 30 == 0:
                print(f"    [ALIGN] step={self.step_in_phase} "
                      f"x_err={peg_pos[0]-hole_c[0]:.4f} "
                      f"z_err={peg_pos[2]-hole_c[2]:.4f} "
                      f"align={align:.3f} lateral={lateral:.4f}")

            if (x_ok and z_ok and rot_ok) or self.step_in_phase > 200:
                self._advance("PUSH")
            return action

        # ── PUSH (first stroke) ──────────────────────────────────────────────
        elif self.phase == "PUSH":
            # Drive along +Y (hole_normal) while correcting X/Z drift.
            # Also apply live alignment correction to counter perturbation.

            # XZ corrections (keep peg centred in slot).
            dx_corr = (hole_c[0] - peg_pos[0]) * GAIN_XY * 3.0
            dz_corr = (hole_c[2] - peg_pos[2]) * GAIN_Z  * 3.0

            # Push along +Y.
            push = hole_n * PUSH_GAIN
            dx   = push[0] + dx_corr
            dy   = push[1]
            dz   = push[2] + dz_corr

            # Alignment correction (handles perturbation yaw ramp).
            current_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
            drot = _axis_angle_to_action_rot(current_axis, hole_n)
            droll, dpitch, dyaw = drot

            if self.step_in_phase % 20 == 0:
                print(f"    [PUSH]  step={self.step_in_phase} "
                      f"axial={axial:+.4f}/{self._insert_thr:.3f} "
                      f"lat={lateral:.4f}/{self._lat_tol:.3f} "
                      f"align={align:.3f}/{self._align_thr:.2f} "
                      f"perturb_fired={self.env._perturb_fired}")

            # Full insertion done — skip regrasp.
            if (axial >= self._insert_thr
                    and lateral < self._lat_tol
                    and align > self._align_thr):
                print(f"  [PUSH] ✓ insertion complete at step {self.step_in_phase}")
                self._advance("RELEASE")

            # EEF has stalled at the slot entrance (peg is partially in but
            # gripper is now flush with the fixture face).  Regrasp further
            # along the exposed peg tail and push the rest in.
            # Trigger: EEF is no longer advancing in Y (stall) but peg
            # is not yet fully inserted.  We use a step budget as proxy.
            elif self.step_in_phase >= 120:
                print(f"  [PUSH] gripper at entrance, triggering regrasp "
                      f"(axial={axial:+.3f})")
                self._advance("REGRASP_OPEN")

            # Safety: badly misaligned before entering — back off.
            elif lateral > self._lat_tol * 3 and axial < 0:
                print(f"  [PUSH] ✗ lateral drift ({lateral:.3f}), re-aligning")
                self._advance("ALIGN_FINE")

            return _clip(dx, dy, dz, droll, dpitch, dyaw, 1.0)

        # ── REGRASP_OPEN ─────────────────────────────────────────────────────
        elif self.phase == "REGRASP_OPEN":
            # Open gripper in place.
            if self.step_in_phase >= OPEN_HOLD:
                self._advance("REGRASP_PULLBACK")
            return _noop(-1.0)

        # ── REGRASP_PULLBACK ─────────────────────────────────────────────────
        # Smooth pullback along −Y with overshoot so the gripper swings clear
        # of the slot entrance before repositioning.
        elif self.phase == "REGRASP_PULLBACK":
            tgt_y  = hole_c[1] - APPROACH_Y_OFFSET + 0.25
            target = np.array([hole_c[0], tgt_y, hole_c[2]])

            if self._path is None:
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), target],
                    overshoot = OVERSHOOT_PULLBACK,
                    n_steps   = PATH_STEPS,
                )

            current_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
            drot = _axis_angle_to_action_rot(current_axis, hole_n)
            droll, dpitch, dyaw = drot

            action = _path_action(self._path, eef_pos, gripper=-1.0,
                                  droll=droll, dpitch=dpitch, dyaw=dyaw)

            if self._path.done and np.linalg.norm(target - eef_pos) < XY_TOL * 2:
                self._advance("REGRASP_DESCEND")
            return action

        # ── REGRASP_DESCEND ──────────────────────────────────────────────────
        # Smooth re-approach onto the exposed peg tail with overshoot.
        elif self.phase == "REGRASP_DESCEND":
            target = np.array([peg_pos[0],
                                hole_c[1] - 0.04,   # close to entrance on −Y
                                peg_pos[2]])

            if self._path is None:
                self._path = WaypointPath(
                    waypoints = [eef_pos.copy(), target],
                    overshoot = OVERSHOOT_TRANSPORT,
                    n_steps   = PATH_STEPS,
                )

            current_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
            drot = _axis_angle_to_action_rot(current_axis, hole_n)
            droll, dpitch, dyaw = drot

            action = _path_action(self._path, eef_pos, gripper=-1.0,
                                  droll=droll, dpitch=dpitch, dyaw=dyaw)

            if self.step_in_phase % 30 == 0:
                delta = target - eef_pos
                print(f"    [REGRASP_DESCEND] step={self.step_in_phase} "
                      f"delta={np.round(delta,4)}")

            if (self._path.done and np.linalg.norm(target - eef_pos) < XY_TOL * 1.5) \
                    or self.step_in_phase > 150:
                self._advance("REGRASP_CLOSE")
            return action

        # ── REGRASP_CLOSE ────────────────────────────────────────────────────
        elif self.phase == "REGRASP_CLOSE":
            # Close gripper on the peg tail, then go straight to PUSH2.
            if self.step_in_phase >= CLOSE_HOLD:
                self._advance("PUSH2")
            return _clip(0, 0, 0, 0, 0, 0, 1.0)

        # ── PUSH2 (second stroke — finish insertion) ──────────────────────────
        elif self.phase == "PUSH2":
            # Same logic as PUSH but no regrasp fallback — just push to done.
            dx_corr = (hole_c[0] - peg_pos[0]) * GAIN_XY * 3.0
            dz_corr = (hole_c[2] - peg_pos[2]) * GAIN_Z  * 3.0

            push = hole_n * PUSH_GAIN
            dx   = push[0] + dx_corr
            dy   = push[1]
            dz   = push[2] + dz_corr

            current_axis = peg_axis / (np.linalg.norm(peg_axis) + 1e-9)
            drot = _axis_angle_to_action_rot(current_axis, hole_n)
            droll, dpitch, dyaw = drot

            if self.step_in_phase % 20 == 0:
                print(f"    [PUSH2] step={self.step_in_phase} "
                      f"axial={axial:+.4f}/{self._insert_thr:.3f} "
                      f"lat={lateral:.4f}  align={align:.3f}")

            if (axial >= self._insert_thr
                    and lateral < self._lat_tol
                    and align > self._align_thr):
                print(f"  [PUSH2] ✓ insertion complete at step {self.step_in_phase}")
                self._advance("RELEASE")

            return _clip(dx, dy, dz, droll, dpitch, dyaw, 1.0)

        # ── RELEASE ───────────────────────────────────────────────────────────
        elif self.phase == "RELEASE":
            if self.step_in_phase >= OPEN_HOLD:
                self._advance("DONE")
            return _noop(-1.0)

        # ── DONE ──────────────────────────────────────────────────────────────
        elif self.phase == "DONE":
            return _noop(-1.0)

        return np.zeros(7)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env    = ib.make("ib-p1-recover-peg-insert-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = PegInsertPolicy(env)

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
                sim      = env._env.sim
                peg_pos  = _peg_pos_world(sim)
                peg_axis = _peg_long_axis(sim)
                axial, lateral, align = policy._hole_rel(peg_pos, peg_axis)
                print(
                    f"  ep={ep} step={step:4d}  "
                    f"{prev_phase:20s}→{policy.phase:20s}  "
                    f"axial={axial:+.3f} lat={lateral:.3f} align={align:.3f}  "
                    f"score={reward:.3f}"
                )
                prev_phase = policy.phase

            done = terminated or truncated
            step += 1

        print(
            f"\nEpisode {ep}  steps={step}  "
            f"score={info.get('task_score', 0):.3f}  "
            f"phase={policy.phase}  "
            f"perturb_fired={info.get('perturb_fired', False)}\n"
        )

    env.close()