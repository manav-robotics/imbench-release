"""
ib-p2-ramp-sort-v1  —  adaptive policy with smooth waypoint trajectories
========================================================================
"""

import numpy as np
import imbench as ib


# ═══════════════════════ TUNE ════════════════════════════
DROP_Z  = 1.00   # release height; the ball drops ~5 cm onto the ramp
HOVER_Z = 1.00   # must clear ramp (~0.95 m)
DROP_ALONG = -0.03  # release point along the ramp (m from its centre, + = downhill);
                    # further downhill the forearm can touch the fence top under yaw jitter
# ═════════════════════════════════════════════════════════

GAIN_XY   = 9.0
GAIN_Z    = 9.0
GAIN_FAST = 16.0
GAIN_ROT       = 4.0    # rotation gain for CHECK_ORIENT / regular rot phases
GAIN_ROT_SLOW  = 2.5    # slower gain — used in CHECK_ORIENT for a smooth visual

XY_TOL    = 0.013
Z_TOL     = 0.012
ROT_TOL   = 2.0         # degrees — threshold for orientation done

PICK_HOVER_ABOVE_BALL = 0.14

OPEN_HOLD      = 25
GRASP_HOLD     = 45
RELEASE_HOLD   = 35
RETREAT_HOLD   = 40
DESCEND_BUDGET = 400
RELEASE_XY_TOL = 0.003      # held ball within 3 mm of its lane point before release
RELEASE_SETTLE_STEPS = 5
# Level the hand before release: at the default pitch the opening fingertips
# are slanted and push the ball sideways, so it drifts across its lane.
RELEASE_PITCH_DEG = 0.0
RELEASE_MAX_SPEED = 0.01    # m/s: open only once the held ball is nearly still
LIFT_CHECK_DZ  = 0.05       # held ball sits ~1 cm below the eef; 5 cm means it was left behind
MAX_GRASP_TRIES = 3

# Default "looking straight down" gripper orientation (degrees)
DEFAULT_ROLL_DEG  = 180.0
DEFAULT_PITCH_DEG = -8.0    # environment spawns at ~-7 to -8; -8 is safe
DEFAULT_YAW_DEG   = 0.0

# Overshoot fractions per phase
OVERSHOOT_HOVER_PICK  = 0.08
OVERSHOOT_HOVER_DROP  = 0.10
OVERSHOOT_LIFT        = 0.06   # small upward pop after grasp

# Z bulge for curved arcs
ARC_BULGE_HOVER  = 0.10   # rise during lateral transit to pick hover
ARC_BULGE_DROP   = 0.08   # rise mid-way to drop hover

PATH_STEPS = 120


# ─────────────────────────────────────────────────────────────────────────────
#  Smooth interpolation helpers  (same as sheltered_pick_policy)
# ─────────────────────────────────────────────────────────────────────────────

def _raised_cosine(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))


class WaypointPath:
    """
    Multi-stop raised-cosine path with optional overshoot.

    waypoints   : list of ndarray(3,) — start → ... → goal
    overshoot   : fraction of last segment to overshoot before correcting
    n_steps     : pre-computed samples
    """

    def __init__(self, waypoints, overshoot=0.0, n_steps=PATH_STEPS):
        waypoints = [np.array(w, dtype=float) for w in waypoints]

        if overshoot > 0.0 and len(waypoints) >= 2:
            delta  = waypoints[-1] - waypoints[-2]
            beyond = waypoints[-1] + overshoot * delta
            waypoints = waypoints[:-1] + [beyond, waypoints[-1]]

        self.waypoints = waypoints
        n_seg          = len(waypoints) - 1
        t_vals         = np.linspace(0.0, 1.0, n_steps)

        self._positions  = self._build(t_vals, waypoints, n_seg)
        self._velocities = np.gradient(self._positions, axis=0)
        self._cursor     = 0
        self.done        = False

    def _build(self, t_global, waypoints, n_seg):
        positions = np.zeros((len(t_global), 3))
        seg_size  = 1.0 / n_seg
        for i, t in enumerate(t_global):
            seg_idx = min(int(t / seg_size), n_seg - 1)
            t_local = (t - seg_idx * seg_size) / seg_size
            s       = _raised_cosine(np.array([t_local]))[0]
            p0      = waypoints[seg_idx]
            p1      = waypoints[seg_idx + 1]
            positions[i] = p0 + s * (p1 - p0)
        return positions

    def advance_to(self, current_pos: np.ndarray, dt_gain: float = 1.0):
        if self.done:
            return self._positions[-1], np.zeros(3)
        while self._cursor < len(self._positions) - 1:
            if np.linalg.norm(current_pos - self._positions[self._cursor]) < XY_TOL * 1.5:
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

def _clip(dx=0.0, dy=0.0, dz=0.0,
          droll=0.0, dpitch=0.0, dyaw=0.0,
          gripper=-1.0, lim=1.0) -> np.ndarray:
    a = np.zeros(7)
    a[0] = np.clip(dx,     -lim, lim)
    a[1] = np.clip(dy,     -lim, lim)
    a[2] = np.clip(dz,     -lim, lim)
    a[3] = np.clip(droll,  -lim, lim)
    a[4] = np.clip(dpitch, -lim, lim)
    a[5] = np.clip(dyaw,   -lim, lim)
    a[6] = gripper
    return a


def _still(g: float) -> np.ndarray:
    return _clip(gripper=g)


def _path_action(path: WaypointPath, eef_pos: np.ndarray,
                 gripper: float, ff_scale: float = 4.0) -> np.ndarray:
    tgt, vel = path.advance_to(eef_pos)
    err = tgt - eef_pos
    dx  = err[0] * GAIN_FAST + vel[0] * ff_scale
    dy  = err[1] * GAIN_FAST + vel[1] * ff_scale
    dz  = err[2] * GAIN_Z    + vel[2] * ff_scale
    return _clip(dx=dx, dy=dy, dz=dz, gripper=gripper)


def quat_to_rpy_deg(q_xyzw: np.ndarray) -> np.ndarray:
    q = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-9)
    x, y, z, w = q
    roll  = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    yaw   = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return np.array([roll, pitch, yaw])


def angle_error_deg(current: float, target: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _rot_cmd_slow(current: float, target: float) -> float:
    err = angle_error_deg(current, target)
    return float(np.clip(err * GAIN_ROT_SLOW / 90.0, -1.0, 1.0))


def _orient_done(rpy: np.ndarray, yaw_deg: float = DEFAULT_YAW_DEG) -> bool:
    return (abs(angle_error_deg(rpy[0], DEFAULT_ROLL_DEG))  < ROT_TOL and
            abs(angle_error_deg(rpy[1], DEFAULT_PITCH_DEG)) < ROT_TOL and
            abs(angle_error_deg(rpy[2], yaw_deg))           < ROT_TOL)


# ─────────────────────────────────────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _colour_from_name(name):
    n = name.lower()
    if n.startswith("sphere"): return "blue"
    if n.startswith("box"):    return "red"
    return None

def _bin_colour_from_name(bname):
    n = bname.lower()
    if "sphere" in n: return "blue"
    if "cube"   in n: return "red"
    return None

def _dominant_rgba(rgba):
    r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
    return "red" if r >= g and r >= b else ("green" if g >= b else "blue")

def _body_rgba(sim, bid):
    fallback = (None, None)
    for gid in range(sim.model.ngeom):
        if sim.model.geom_bodyid[gid] != bid:
            continue
        name = sim.model.geom_id2name(gid) or f"geom_{gid}"
        rgba = sim.model.geom_rgba[gid].copy()
        if name.endswith("_vis"):
            return rgba, name
        if fallback[0] is None:
            fallback = rgba, name
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
#  Policy
# ─────────────────────────────────────────────────────────────────────────────

class RampSortPolicy:

    def __init__(self, env, verbose: bool = False):
        self.env = env
        self.verbose = verbose  # print phase transitions and the sort queue
        self._reset_state()

    def reset(self):
        self._reset_state()

    # ── state ─────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.phase         = "INIT"
        self.step_in_phase = 0
        self._queue        = []
        self._cur          = None
        self._ready        = False
        self._path: WaypointPath | None = None

    def _goto(self, phase):
        self.phase         = phase
        self.step_in_phase = 0
        self._on_lane_steps = 0
        self._path         = None   # always clear path on transition
        name = self._cur["name"] if self._cur else "-"
        if self.verbose:
            print(f"  -> {phase:<16s}  obj={name}")

    def _sim(self):
        return self.env._env.sim

    def _eef(self, obs):
        return np.array(obs["robot0_eef_pos"], dtype=float)

    def _eef_rpy(self, obs) -> np.ndarray:
        return quat_to_rpy_deg(np.array(obs["robot0_eef_quat"], dtype=float))

    def _pick_path_action(self, eef, goal):
        action = _path_action(self._path, eef, gripper=-1.0)
        if self._path.done:
            # Earlier picks can set the remaining balls rolling. The path
            # ends at a snapshot, but arrival is checked against the live
            # ball: holding that stale endpoint can deadlock until timeout.
            delta = goal - eef
            action = _clip(dx=delta[0]*GAIN_FAST, dy=delta[1]*GAIN_FAST,
                           dz=delta[2]*GAIN_Z, gripper=-1.0)
        return action

    def _obj_pos(self):
        return self._sim().data.body_xpos[self._cur["body_id"]].copy()

    def _ramp_frame(self):
        """(R, origin): ramp-local xy -> world xy is R @ local + origin."""
        yaw = float(self.env._ramp_yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        origin = np.asarray(getattr(self.env, "_ramp_origin_xy", np.zeros(2)), dtype=float)
        return np.array([[c, -s], [s, c]]), origin

    def _grasp_yaw_deg(self):
        """Gripper yaw that keeps the jaws square to the spawn-box rows.

        The rows run along the ramp's local x.  The jaws close along world y
        at yaw 0 and are symmetric under 180 deg, so match the ramp yaw
        modulo 180, wrapped to [-90, 90).
        """
        return (np.degrees(float(self.env._ramp_yaw)) + 90.0) % 180.0 - 90.0

    # ── queue builder ──────────────────────────────────────────────────────

    def _build_queue(self):
        task = self.env
        sim  = self._sim()

        bin_xy = {}
        for bid in range(sim.model.nbody):
            bname = sim.model.body_id2name(bid) or ""
            if not bname.endswith("_base_main") or "bin" not in bname:
                continue
            colour = _bin_colour_from_name(bname)
            if colour is None:
                rgba, _ = _body_rgba(sim, bid)
                colour  = _dominant_rgba(rgba) if rgba is not None else None
            if colour is None:
                continue
            world_xy = sim.data.body_xpos[bid][:2].copy()
            if colour not in bin_xy:
                bin_xy[colour] = world_xy
                if self.verbose:
                    print(f"  bin '{bname}'  colour={colour}  "
                          f"xy=[{world_xy[0]:+.3f}, {world_xy[1]:+.3f}]")

        if not bin_xy:
            if self.verbose:
                print("  WARNING: bin scan failed, using task xy attributes")
            bin_xy = {
                "blue": np.array(task._sphere_bin_world_xy[:2]),
                "red":  np.array(task._cube_bin_world_xy[:2]),
            }

        R, origin = self._ramp_frame()

        # A ball rolling down the lane at |x| enters its bin if it clears the
        # divider (bin centre - half width) and the ramp wall (0.08 inside);
        # aim midway between the two limits on the ball centre.
        r, half_bin, ramp_half = task._object_size, task._bin_width / 2, 0.08

        def _drop_for_colour(colour):
            # Release in the bin's lane, DROP_ALONG along the ramp.  The ball
            # rolls straight down the lane.
            xy = bin_xy.get(colour, next(iter(bin_xy.values())))
            bin_x = (R.T @ (xy - origin))[0]
            lo, hi = abs(bin_x) - half_bin + r, ramp_half - r
            lane_x = np.sign(bin_x) * 0.5 * (lo + hi)
            release = R @ np.array([lane_x, DROP_ALONG]) + origin
            return np.array([release[0], release[1], DROP_Z])

        queue = []
        for i in range(task._num_spheres):
            name   = f"sphere_{i}"
            bid    = sim.model.body_name2id(f"{name}_main")
            colour = _colour_from_name(name)
            if colour is None:
                rgba, _ = _body_rgba(sim, bid)
                colour  = _dominant_rgba(rgba) if rgba is not None else "blue"
            drop_pos = _drop_for_colour(colour)
            queue.append({"name": name, "body_id": bid,
                          "colour": colour, "drop_pos": drop_pos})
            if self.verbose:
                print(f"  queued {name:12s}  colour={colour}  "
                      f"drop={np.round(drop_pos, 3)}")

        for i in range(task._num_cubes):
            name   = f"box_{i}"
            bid    = sim.model.body_name2id(f"{name}_main")
            colour = _colour_from_name(name)
            if colour is None:
                rgba, _ = _body_rgba(sim, bid)
                colour  = _dominant_rgba(rgba) if rgba is not None else "red"
            drop_pos = _drop_for_colour(colour)
            queue.append({"name": name, "body_id": bid,
                          "colour": colour, "drop_pos": drop_pos})
            if self.verbose:
                print(f"  queued {name:12s}  colour={colour}  "
                      f"drop={np.round(drop_pos, 3)}")

        if self.verbose:
            print(f"  total queued: {len(queue)} object(s)")
        return queue

    # ── load next ─────────────────────────────────────────────────────────────

    def _load_next(self):
        if not self._queue:
            if self.verbose:
                print("  All objects placed -> DONE")
            self._goto("DONE")
            return
        self._cur  = self._queue.pop(0)
        p = self._sim().data.body_xpos[self._cur["body_id"]].copy()
        if self.verbose:
            print(f"\n-- picking {self._cur['name']} ({self._cur['colour']})  "
                  f"pos={np.round(p, 3)}  drop={np.round(self._cur['drop_pos'], 3)}  "
                  f"({len(self._queue)} left) --")
        self._goto("OPEN")

    # ── act ───────────────────────────────────────────────────────────────────

    def act(self, obs):
        action = self._step(obs)
        self.step_in_phase += 1
        return action

    # ── state machine ─────────────────────────────────────────────────────────

    def _step(self, obs):
        eef = self._eef(obs)
        rpy = self._eef_rpy(obs)

        # ── INIT ──────────────────────────────────────────────────────────────
        if self.phase == "INIT":
            if not self._ready:
                self._queue = self._build_queue()
                self._ready = True
            if self.step_in_phase >= 5:
                self._load_next()
            return _still(-1.0)

        # ── DONE ──────────────────────────────────────────────────────────────
        if self.phase == "DONE":
            return _still(-1.0)

        # ── OPEN ──────────────────────────────────────────────────────────────
        if self.phase == "OPEN":
            if self.step_in_phase >= OPEN_HOLD:
                self._goto("HOVER_PICK")
            return _still(-1.0)

        # ── HOVER_PICK ────────────────────────────────────────────────────────
        # Curved arc: rise in Z while sweeping XY, with an upward bulge
        # so the arm lifts clear before descending onto the object.
        elif self.phase == "HOVER_PICK":
            obj = self._obj_pos()
            goal = np.array([obj[0], obj[1], obj[2] + PICK_HOVER_ABOVE_BALL])

            if self._path is None:
                mid = np.array([(eef[0] + goal[0]) * 0.5,
                                 (eef[1] + goal[1]) * 0.5,
                                 max(eef[2], goal[2]) + ARC_BULGE_HOVER])
                self._path = WaypointPath(
                    waypoints = [eef.copy(), mid, goal],
                    overshoot = OVERSHOOT_HOVER_PICK,
                    n_steps   = PATH_STEPS,
                )

            action = self._pick_path_action(eef, goal)
            # Turn the jaws square to the spawn-box rows on the way over.
            action[5] = _rot_cmd_slow(rpy[2], self._grasp_yaw_deg())

            if (self._path.done and
                    np.linalg.norm(eef[:2] - goal[:2]) < XY_TOL and
                    abs(eef[2] - goal[2]) < Z_TOL and
                    abs(angle_error_deg(rpy[2], self._grasp_yaw_deg())) < ROT_TOL):
                self._goto("DESCEND_PICK")

            return action

        # ── DESCEND_PICK ──────────────────────────────────────────────────────
        # Straight smooth descent onto the object — no overshoot (safety).
        elif self.phase == "DESCEND_PICK":
            obj = self._obj_pos()

            if self._path is None:
                self._path = WaypointPath(
                    waypoints = [eef.copy(), obj.copy()],
                    overshoot = 0.0,
                    n_steps   = PATH_STEPS,
                )

            action = self._pick_path_action(eef, obj)
            action[5] = _rot_cmd_slow(rpy[2], self._grasp_yaw_deg())

            reached = (abs(eef[2] - obj[2]) < Z_TOL and
                       np.linalg.norm(eef[:2] - obj[:2]) < XY_TOL)
            if reached or self.step_in_phase >= DESCEND_BUDGET:
                self._goto("GRASP")

            return action

        # ── GRASP ─────────────────────────────────────────────────────────────
        elif self.phase == "GRASP":
            if self.step_in_phase >= GRASP_HOLD:
                self._goto("CHECK_ORIENT")
            return _still(1.0)

        # ── CHECK_ORIENT ──────────────────────────────────────────────────────
        # After closing the gripper, verify the EEF is pointing straight down
        # (roll≈180, pitch≈-8, yaw≈0).  If it has drifted — e.g. because the
        # object's contact pushed the wrist — rotate back slowly before lifting.
        # The gripper stays closed (gripper=1.0) throughout.
        elif self.phase == "CHECK_ORIENT":
            yaw_tgt = self._grasp_yaw_deg()
            already_ok = _orient_done(rpy, yaw_tgt)

            if already_ok and self.step_in_phase == 0:
                # Fast-path: orientation is already good, skip straight to LIFT
                if self.verbose:
                    print(f"  [CHECK_ORIENT] orientation OK  rpy={np.round(rpy, 1)}  -> skipping")
                self._goto("LIFT")
                return _still(1.0)

            droll  = _rot_cmd_slow(rpy[0], DEFAULT_ROLL_DEG)
            dpitch = _rot_cmd_slow(rpy[1], DEFAULT_PITCH_DEG)
            dyaw   = _rot_cmd_slow(rpy[2], yaw_tgt)

            if _orient_done(rpy, yaw_tgt) and self.step_in_phase > 0:
                if self.verbose:
                    print(f"  [CHECK_ORIENT] corrected to  rpy={np.round(rpy, 1)}  -> LIFT")
                self._goto("LIFT")

            return _clip(droll=droll, dpitch=dpitch, dyaw=dyaw, gripper=1.0)

        # ── LIFT ──────────────────────────────────────────────────────────────
        # Smooth vertical rise with a small upward overshoot ("pop").
        elif self.phase == "LIFT":
            if self._path is None:
                goal = np.array([eef[0], eef[1], HOVER_Z])
                self._path = WaypointPath(
                    waypoints = [eef.copy(), goal],
                    overshoot = OVERSHOOT_LIFT,
                    n_steps   = PATH_STEPS,
                )

            action = _path_action(self._path, eef, gripper=1.0)

            if self._path.done and eef[2] >= HOVER_Z - 0.015:
                # A missed or slipped grasp leaves the ball behind: open and
                # pick it again (at most MAX_GRASP_TRIES in all).
                if (eef[2] - self._obj_pos()[2] > LIFT_CHECK_DZ and
                        self._cur.get("tries", 1) < MAX_GRASP_TRIES):
                    self._cur["tries"] = self._cur.get("tries", 1) + 1
                    if self.verbose:
                        print(f"  [LIFT] {self._cur['name']} not lifted -> retry")
                    self._goto("OPEN")
                else:
                    self._goto("HOVER_DROP")

            return action

        # ── HOVER_DROP ────────────────────────────────────────────────────────
        # Curved arc over to the drop XY, bulging upward mid-transit.
        elif self.phase == "HOVER_DROP":
            drop = self._cur["drop_pos"]
            goal = np.array([drop[0], drop[1], HOVER_Z])

            if self._path is None:
                mid = np.array([(eef[0] + goal[0]) * 0.5,
                                 (eef[1] + goal[1]) * 0.5,
                                 HOVER_Z + ARC_BULGE_DROP])
                self._path = WaypointPath(
                    waypoints = [eef.copy(), mid, goal],
                    overshoot = OVERSHOOT_HOVER_DROP,
                    n_steps   = PATH_STEPS,
                )

            action = _path_action(self._path, eef, gripper=1.0)

            if (self._path.done and
                    np.linalg.norm(eef[:2] - goal[:2]) < XY_TOL and
                    abs(eef[2] - goal[2]) < Z_TOL):
                self._goto("DESCEND_DROP")

            return action

        # ── DESCEND_DROP ──────────────────────────────────────────────────────
        # Fine-position the held ball (not the hand) over its lane: the lane
        # leaves ~15 mm either side, less than the hand's arrival tolerance
        # plus the ball's offset in the jaws, and a ball released against a
        # ramp wall bounces across into the other lane.
        elif self.phase == "DESCEND_DROP":
            drop = self._cur["drop_pos"]
            ball = self._obj_pos()
            tgt  = drop.copy()
            tgt[:2] -= ball[:2] - eef[:2]
            err  = tgt - eef
            yaw_tgt = self._grasp_yaw_deg()
            action = _clip(dx=err[0] * GAIN_FAST, dy=err[1] * GAIN_FAST,
                           dz=err[2] * GAIN_Z,
                           droll=_rot_cmd_slow(rpy[0], DEFAULT_ROLL_DEG),
                           dpitch=_rot_cmd_slow(rpy[1], RELEASE_PITCH_DEG),
                           dyaw=_rot_cmd_slow(rpy[2], yaw_tgt), gripper=1.0)

            level = (abs(angle_error_deg(rpy[0], DEFAULT_ROLL_DEG)) < ROT_TOL and
                     abs(angle_error_deg(rpy[1], RELEASE_PITCH_DEG)) < ROT_TOL)
            speed = np.linalg.norm(self._sim().data.get_body_xvelp(
                self._sim().model.body_id2name(self._cur["body_id"])))
            on_lane = (np.linalg.norm(ball[:2] - drop[:2]) < RELEASE_XY_TOL and
                       abs(err[2]) < Z_TOL and level and speed < RELEASE_MAX_SPEED)
            self._on_lane_steps = self._on_lane_steps + 1 if on_lane else 0
            if (self._on_lane_steps >= RELEASE_SETTLE_STEPS or
                    self.step_in_phase >= DESCEND_BUDGET):
                self._hold_pos = eef.copy()
                self._goto("RELEASE")

            return action

        # ── RELEASE ───────────────────────────────────────────────────────────
        # Hold the hand's pose while the jaws open, so it doesn't push the
        # ball sideways on the way out.
        elif self.phase == "RELEASE":
            if self.step_in_phase >= RELEASE_HOLD:
                self._goto("RETREAT")
            err = self._hold_pos - eef
            return _clip(dx=err[0] * GAIN_FAST, dy=err[1] * GAIN_FAST,
                         dz=err[2] * GAIN_Z,
                         droll=_rot_cmd_slow(rpy[0], DEFAULT_ROLL_DEG),
                         dpitch=_rot_cmd_slow(rpy[1], RELEASE_PITCH_DEG),
                         dyaw=_rot_cmd_slow(rpy[2], self._grasp_yaw_deg()),
                         gripper=-1.0)

        # ── RETREAT ───────────────────────────────────────────────────────────
        elif self.phase == "RETREAT":
            dz = (HOVER_Z - eef[2]) * GAIN_Z
            if self.step_in_phase >= RETREAT_HOLD:
                self._load_next()
            return _clip(dz=dz, gripper=-1.0)

        return _still(-1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env    = ib.make("ib-p2-ramp-sort-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = RampSortPolicy(env)

    for ep in range(10):
        obs, info = env.reset()
        policy.reset()
        done = False
        step = 0

        print(f"\n{'='*60}")
        print(f"Episode {ep}   "
              f"spheres={env._num_spheres}  cubes={env._num_cubes}  "
              f"ramp_yaw={np.degrees(env._ramp_yaw):.1f} deg")
        print(f"  sphere_bin_xy = {np.round(env._sphere_bin_world_xy, 3)}")
        print(f"  cube_bin_xy   = {np.round(env._cube_bin_world_xy,   3)}")
        print(f"{'='*60}")

        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done  = terminated or truncated
            step += 1

        print(f"\nEp {ep}  steps={step}  "
              f"score={info.get('task_score', 0):.3f}  "
              f"phase={policy.phase}  "
              f"terminated={terminated}  truncated={truncated}")

    env.close()
