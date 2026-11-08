"""Ramp Separator — sort blue and red balls via a ramp into box containers.

Task ID: ``ib-p2-ramp-sort-v1``

Balls start inside a separate top-open spawn box placed on the table near
the robot (not on the ramp's elevated platform).  The spawn box has low
walls so the gripper can reach in without colliding with the rim.  The
robot picks balls out and places them on the ramp; each rolls straight
down its lane, so the lateral release point decides which bin at the
ramp's low end it lands in.  Blue and red balls are the same size; the
blue one goes to the blue bin, the red ones to the red bin.  (Red balls
keep their historical ``box_*`` body names and the red bin ``bin_cube_*``.)

Randomisations each episode
----------------------------
* Ramp orientation: a yaw from ``randomization.ramp_orientation.choices``
  (default the 4 cardinal yaws; the shipped config uses 0 and pi, which put
  the bins to the robot's left or right).
* Ramp pose jitter: the whole layout (ramp, bins, fence, spawn box) is
  rotated by up to ``randomization.ramp_pose.yaw_deg`` and shifted by up to
  ``randomization.ramp_pose.shift_xy``.
* Number of blue and red balls drawn from per-colour ranges.
* Ball layout: the balls sit in one row in the spawn box at random
  positions along it (at least ``SPAWN_MIN_GAP`` apart) in a random colour
  order, each jittered across the row by up to
  ``randomization.ball_layout.jitter``.
* Blue-bin / red-bin left-right swap.

Scoring: a ball counts once it has stayed at rest inside its own bin,
below the rim (on the floor or on other balls), for ``REST_TIME`` seconds;
a ball below the rim of the other bin fails the episode.

Workspace restriction: tall fence panels surround the bin area so the robot
arm cannot place objects directly — they must travel via the ramp. The
front, side and top panels block the balls too; only the ramp-side panel
lets them through.
"""

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase


def _quat_mul(q1, q2):
    """Hamilton product of two [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class RampSortV1(SingleArmBase):
    TASK_ID = "ib-p2-ramp-sort-v1"
    DESCRIPTION = (
        "Puzzle: pick blue and red balls from a low-walled spawn box on the "
        "table near the robot and release each on the ramp so it rolls down "
        "into the bin of its colour at the ramp's low end."
    )

    # Spawn-box interior half-extents (local x across the ramp, local y
    # along it).  Balls sit in one row along local x: jaws closing along
    # local y then never meet a neighbour, and the hand, which reaches
    # ~10 cm along the jaw axis, stays clear of the ramp's high leg.
    SPAWN_HALF_X = 0.140
    SPAWN_HALF_Y = 0.055
    SPAWN_MIN_GAP = 0.010   # min gap between neighbouring balls in the row
    SPAWN_WALL_GAP = 0.009  # min gap between a ball and the short walls

    # ── Config parsing ──────────────────────────────────────────────────────

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})

        # Object counts.  _num_* are the counts *in play* this episode;
        # _max_* is what the MuJoCo model always holds.  Every reset uses
        # the first _num_* of each kind and parks the rest off-scene, so
        # the model never has to be rebuilt.
        self._num_spheres = task_cfg.get("num_spheres", 2)
        self._num_cubes   = task_cfg.get("num_cubes", 2)

        self._num_spheres_range = task_cfg.get("num_spheres_range", [1, 3])
        self._num_cubes_range   = task_cfg.get("num_cubes_range",   [1, 3])
        self._max_spheres = max(self._num_spheres, self._num_spheres_range[1])
        self._max_cubes   = max(self._num_cubes,   self._num_cubes_range[1])
        self._object_size    = task_cfg.get("object_size", 0.015)
        self._ramp_angle_deg = task_cfg.get("ramp_angle_deg", 15)
        self._bin_width      = task_cfg.get("bin_width", 0.08)

        rand_cfg = self.cfg.get("randomization", {})
        self._rand_ramp_orient = rand_cfg.get("ramp_orientation", {}).get("enabled", True)
        self._ramp_orient_choices = rand_cfg.get("ramp_orientation", {}).get(
            "choices", [0.0, np.pi / 2, np.pi, -np.pi / 2]
        )
        pose_cfg = rand_cfg.get("ramp_pose", {})
        self._ramp_yaw_jitter = np.radians(pose_cfg.get("yaw_deg", 0.0))
        self._ramp_shift_xy   = np.asarray(pose_cfg.get("shift_xy", [0.0, 0.0]), dtype=float)
        layout_cfg = rand_cfg.get("ball_layout", {})
        self._shuffle_balls = layout_cfg.get("shuffle", False)
        self._ball_jitter   = layout_cfg.get("jitter", 0.0)
        # Placeholders updated each reset
        self._sphere_bin_world_xy = np.zeros(2)
        self._cube_bin_world_xy   = np.zeros(2)
        # A ball is inside a bin below the bin's wall top (bin base 8 mm +
        # 70 mm walls); balls carried above a bin don't count.  It is sorted
        # only once it is at rest there (on the floor or on other balls), so
        # a ball still falling or bouncing in is not yet counted.
        self._bin_top_z   = self._table_height + 0.008 + 0.070

    # ── Object factories ────────────────────────────────────────────────────

    def _make_objects(self):
        objects = []
        # Spheres — low friction so they roll easily
        for i in range(self._max_spheres):
            objects.append({
                "type": "ball",
                "name": f"sphere_{i}",
                "size": [self._object_size],
                "rgba": [0.2, 0.5, 0.8, 1.0],
                "density": 500,
                "friction": [0.3, 0.001, 0.0001],
                "joints": "default",
            })
        # Red spheres — same size as blue, distinguishable by colour and
        # slightly higher density so they sort to a different bin.
        for i in range(self._max_cubes):
            objects.append({
                "type": "ball",
                "name": f"box_{i}",
                "size": [self._object_size],
                "rgba": [0.85, 0.18, 0.12, 1.0],
                "density": 700,
                "friction": [0.25, 0.001, 0.0001],
                "joints": "default",
            })
        return objects

    def _make_static_objects(self):
        bw = self._bin_width
        wall_t = 0.004   # bin wall half-thickness
        wall_h = 0.035   # bin wall half-height
        bin_inner = bw / 2

        # Spawn-box wall proportions (low walls so the gripper can reach in
        # without colliding with the rim).
        sb_wall_t = 0.004   # spawn-box wall half-thickness
        sb_wall_h = 0.006   # spawn-box wall half-height (~12 mm tall walls)
        sb_half_x = self.SPAWN_HALF_X
        sb_half_y = self.SPAWN_HALF_Y

        return [
            # ── Ramp surface ─────────────────────────────────────────────
            {"type": "box", "name": "ramp",
             "size": [0.08, 0.14, 0.015],
             "rgba": [0.6, 0.5, 0.3, 1.0],
             "density": 5000, "friction": [0.4, 0.005, 0.005],
             "joints": None},
            {"type": "box", "name": "ramp_wall_l",
             "size": [0.003, 0.14, 0.03],
             "rgba": [0.5, 0.4, 0.3, 1.0],
             "density": 5000, "friction": [0.4, 0.005, 0.005],
             "joints": None},
            {"type": "box", "name": "ramp_wall_r",
             "size": [0.003, 0.14, 0.03],
             "rgba": [0.5, 0.4, 0.3, 1.0],
             "density": 5000, "friction": [0.4, 0.005, 0.005],
             "joints": None},
            # Support legs
            {"type": "box", "name": "ramp_leg_hi",
             "size": [0.08, 0.02, 0.075],
             "rgba": [0.45, 0.35, 0.25, 1.0],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "ramp_leg_lo",
             "size": [0.08, 0.02, 0.015],
             "rgba": [0.45, 0.35, 0.25, 1.0],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},

            # ── Spawn box (top-open, 5 parts, on the table near robot) ───
            # Objects start here; robot picks each one out over the low rim
            # and places it on the ramp's high end.
            {"type": "box", "name": "spawn_base",
             "size": [sb_half_x + sb_wall_t, sb_half_y + sb_wall_t, sb_wall_t],
             "rgba": [0.55, 0.45, 0.30, 0.95],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "spawn_front",
             "size": [sb_half_x + sb_wall_t, sb_wall_t, sb_wall_h],
             "rgba": [0.55, 0.45, 0.30, 0.95],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "spawn_back",
             "size": [sb_half_x + sb_wall_t, sb_wall_t, sb_wall_h],
             "rgba": [0.55, 0.45, 0.30, 0.95],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "spawn_left",
             "size": [sb_wall_t, sb_half_y, sb_wall_h],
             "rgba": [0.55, 0.45, 0.30, 0.95],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "spawn_right",
             "size": [sb_wall_t, sb_half_y, sb_wall_h],
             "rgba": [0.55, 0.45, 0.30, 0.95],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},

            # ── Sphere bin (blue, 5-panel box) ───────────────────────────
            {"type": "box", "name": "bin_sphere_base",
             "size": [bin_inner, bin_inner, wall_t],
             "rgba": [0.2, 0.4, 0.85, 0.9],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_sphere_front",
             "size": [bin_inner, wall_t, wall_h],
             "rgba": [0.2, 0.4, 0.85, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_sphere_back",
             "size": [bin_inner, wall_t, wall_h],
             "rgba": [0.2, 0.4, 0.85, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_sphere_left",
             "size": [wall_t, bin_inner + wall_t, wall_h],
             "rgba": [0.2, 0.4, 0.85, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_sphere_right",
             "size": [wall_t, bin_inner + wall_t, wall_h],
             "rgba": [0.2, 0.4, 0.85, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},

            # ── Cube bin (red, 5-panel box) ──────────────────────────────
            {"type": "box", "name": "bin_cube_base",
             "size": [bin_inner, bin_inner, wall_t],
             "rgba": [0.85, 0.25, 0.2, 0.9],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_cube_front",
             "size": [bin_inner, wall_t, wall_h],
             "rgba": [0.85, 0.25, 0.2, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_cube_back",
             "size": [bin_inner, wall_t, wall_h],
             "rgba": [0.85, 0.25, 0.2, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_cube_left",
             "size": [wall_t, bin_inner + wall_t, wall_h],
             "rgba": [0.85, 0.25, 0.2, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
            {"type": "box", "name": "bin_cube_right",
             "size": [wall_t, bin_inner + wall_t, wall_h],
             "rgba": [0.85, 0.25, 0.2, 0.7],
             "density": 5000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},

            # ── Workspace barrier — tall fence enclosing bin area ─────────
            # Three vertical walls (front/left/right) + a top panel block the
            # robot arm and the balls from directly accessing the bins.  The
            # ramp-side (back) panel blocks only the arm, so objects rolling
            # off the ramp can enter freely.
            {"type": "box", "name": "barrier_front",
             "size": [0.14, 0.004, 0.18],
             "rgba": [0.3, 0.3, 0.3, 0.3],
             "density": 8000, "friction": [0.1, 0.001, 0.0001],
             "joints": None},
            {"type": "box", "name": "barrier_left",
             "size": [0.004, 0.10, 0.18],
             "rgba": [0.3, 0.3, 0.3, 0.3],
             "density": 8000, "friction": [0.1, 0.001, 0.0001],
             "joints": None},
            {"type": "box", "name": "barrier_right",
             "size": [0.004, 0.10, 0.18],
             "rgba": [0.3, 0.3, 0.3, 0.3],
             "density": 8000, "friction": [0.1, 0.001, 0.0001],
             "joints": None},
            # Horizontal top panel — closes the enclosure from above so the
            # arm cannot drop objects in over the top of the side walls.
            {"type": "box", "name": "barrier_top",
             "size": [0.13, 0.13, 0.004],
             "rgba": [0.3, 0.3, 0.3, 0.25],
             "density": 8000, "friction": [0.1, 0.001, 0.0001],
             "joints": None},
            # Ramp-side panel — blocks arm from entering through the ramp
            # opening; balls pass through it via collision group filtering.
            {"type": "box", "name": "barrier_back",
             "size": [0.14, 0.004, 0.18],
             "rgba": [0.3, 0.3, 0.3, 0.25],
             "density": 8000, "friction": [0.1, 0.001, 0.0001],
             "joints": None},
        ]

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _parked_objects(self):
        return getattr(self, "_parked", set())

    def _ball_body_ids(self):
        """Cached (sphere_ids, cube_ids) body-id arrays for the counts in play."""
        sim = self._env.sim
        # Keyed on the active counts as well as the model: the model is
        # fixed across episodes, so the counts are what changes.
        key = (id(sim.model), self._num_spheres, self._num_cubes)
        cache = getattr(self, "_body_id_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        sids = [sim.model.body_name2id(f"sphere_{i}_main")
                for i in range(self._num_spheres)]
        cids = [sim.model.body_name2id(f"box_{i}_main")
                for i in range(self._num_cubes)]
        self._body_id_cache = (key, sids, cids)
        return sids, cids

    def _in_bin(self, bid, bin_xy, top_z=None):
        """True iff body *bid* is inside the bin centred at *bin_xy*, below *top_z* (default its rim)."""
        pos = self._env.sim.data.body_xpos[bid]
        dx, dy = pos[:2] - bin_xy
        top_z = self._bin_top_z if top_z is None else top_z
        return (dx * dx + dy * dy < (self._bin_width / 2) ** 2
                and self._table_height < pos[2] < top_z)

    #: Linear speed (m/s) below which a ball inside a bin counts as at rest,
    #: and how long (s) it must stay so before it is sorted.  The hold keeps
    #: a ball from counting at the top of a bounce or while it balances
    #: for an instant on another ball before rolling off.
    REST_SPEED = 0.02
    REST_TIME = 0.2

    def _ball_speed(self, bid):
        """Linear speed of ball body *bid*, from the free joint of the body or its nearest jointed ancestor."""
        model, data = self._env.sim.model, self._env.sim.data
        b = bid
        while model.body_jntadr[b] < 0 and b > 0:
            b = model.body_parentid[b]
        if model.body_jntadr[b] < 0:
            return 0.0
        dof = model.jnt_dofadr[model.body_jntadr[b]]
        return float(np.linalg.norm(data.qvel[dof:dof + 3]))

    def _still_in_bin(self, bid, bin_xy):
        """True iff body *bid* is inside the bin centred at *bin_xy*, below its rim, and moving slower than REST_SPEED now."""
        return self._in_bin(bid, bin_xy) and self._ball_speed(bid) < self.REST_SPEED

    def _update_rest_streaks(self):
        """Count, per ball, the consecutive control steps it has been still inside its own bin."""
        sids, cids = self._ball_body_ids()
        streaks = getattr(self, "_rest_steps", {})
        for bid, xy in [(b, self._sphere_bin_world_xy) for b in sids] + [(b, self._cube_bin_world_xy) for b in cids]:
            streaks[bid] = streaks.get(bid, 0) + 1 if self._still_in_bin(bid, xy) else 0
        self._rest_steps = streaks

    def _at_rest_in_bin(self, bid, bin_xy):
        """True iff body *bid* is inside the bin centred at *bin_xy*, below its rim, and has been still there for REST_TIME."""
        need = max(1, int(round(self.REST_TIME * self.control_freq)))
        return self._in_bin(bid, bin_xy) and getattr(self, "_rest_steps", {}).get(bid, 0) >= need

    def step(self, action):
        action = self._validate_action(action)
        raw_obs, _rs_reward, _rs_done, info = self._env.step(action)
        # Before scoring, so the score sees this step's rest streaks.
        self._update_rest_streaks()
        return self._finish_step(raw_obs, info)

    def _any_ball_in_wrong_bin(self):
        """True if a sphere landed in the cube bin or a cube in the sphere bin."""
        sids, cids = self._ball_body_ids()
        return (any(self._in_bin(bid, self._cube_bin_world_xy) for bid in sids)
                or any(self._in_bin(bid, self._sphere_bin_world_xy) for bid in cids))

    def _compute_task_score(self, raw_obs):
        # Any ball in the wrong bin is an unrecoverable failure → reward 0.
        if self._any_ball_in_wrong_bin():
            return 0.0
        sids, cids = self._ball_body_ids()
        total = len(sids) + len(cids)
        correct = (sum(self._at_rest_in_bin(bid, self._sphere_bin_world_xy) for bid in sids)
                   + sum(self._at_rest_in_bin(bid, self._cube_bin_world_xy) for bid in cids))
        return correct / max(1, total)

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # The model is not rebuilt per episode, so the counts can be drawn
        # *after* the env reset and SingleArmBase.reset() keeps sole
        # ownership of RNG seeding.
        obs, info = super().reset(seed=seed, options=options)
        sim = self._env.sim
        self._rest_steps = {}

        # ── Sample the counts in play; extras get parked below ────────
        lo_s, hi_s = self._num_spheres_range
        lo_c, hi_c = self._num_cubes_range
        self._num_spheres = int(self._rng.randint(lo_s, hi_s + 1))
        self._num_cubes   = int(self._rng.randint(lo_c, hi_c + 1))
        self._parked = (
            {f"sphere_{i}"
             for i in range(self._num_spheres, self._max_spheres)}
            | {f"box_{i}"
               for i in range(self._num_cubes, self._max_cubes)}
        )

        # ── Ramp orientation ──────────────────────────────────────────
        if self._rand_ramp_orient:
            yaw_idx  = self._rng.randint(0, len(self._ramp_orient_choices))
            ramp_yaw = self._ramp_orient_choices[yaw_idx]
        else:
            ramp_yaw = 0.0
        # Pose jitter moves the whole layout rigidly: every placement below
        # goes through rot2d, which applies this yaw and shift.
        ramp_yaw += self._rng.uniform(-self._ramp_yaw_jitter, self._ramp_yaw_jitter)
        shift = self._rng.uniform(-self._ramp_shift_xy, self._ramp_shift_xy)
        self._ramp_yaw = ramp_yaw
        self._ramp_origin_xy = shift

        def rot2d(x, y):
            c, s = np.cos(ramp_yaw), np.sin(ramp_yaw)
            return c * x - s * y + shift[0], s * x + c * y + shift[1]

        angle_rad     = np.radians(self._ramp_angle_deg)
        ramp_half_len = 0.14
        ramp_local_cy = 0.03
        ramp_cz = self._table_height + 0.10

        rx, ry = rot2d(0.0, ramp_local_cy)

        # Combined quaternion: yaw around Z, then tilt down around X
        q_tilt    = np.array([np.cos(angle_rad / 2), -np.sin(angle_rad / 2), 0.0, 0.0])
        q_yaw     = np.array([np.cos(ramp_yaw / 2),  0.0, 0.0, np.sin(ramp_yaw / 2)])
        q_ramp    = _quat_mul(q_yaw, q_tilt)
        q_yaw_only = np.array([np.cos(ramp_yaw / 2), 0.0, 0.0, np.sin(ramp_yaw / 2)])

        self._place_static("ramp", [rx, ry, ramp_cz], quat=q_ramp)

        wall_x_off = 0.08 + 0.003
        wl_x, wl_y = rot2d(-wall_x_off, ramp_local_cy)
        wr_x, wr_y = rot2d( wall_x_off, ramp_local_cy)
        self._place_static("ramp_wall_l", [wl_x, wl_y, ramp_cz], quat=q_ramp)
        self._place_static("ramp_wall_r", [wr_x, wr_y, ramp_cz], quat=q_ramp)

        high_local_y = ramp_local_cy - ramp_half_len * np.cos(angle_rad)
        low_local_y  = ramp_local_cy + ramp_half_len * np.cos(angle_rad)
        high_x, high_y = rot2d(0.0, high_local_y)
        low_x,  low_y  = rot2d(0.0, low_local_y)
        high_z = ramp_cz + ramp_half_len * np.sin(angle_rad)

        # Legs
        self._place_static("ramp_leg_hi", [high_x, high_y, self._table_height + 0.075],
                           quat=q_yaw_only)
        self._place_static("ramp_leg_lo", [low_x,  low_y,  self._table_height + 0.015],
                           quat=q_yaw_only)

        # ── Spawn box (top-open, on the table behind the ramp's high end) ─
        # Sits on the table with short walls so the gripper can easily dip
        # in to grab balls without colliding with the rim.  Placed on the
        # opposite side of the ramp from the bins; rotates with ramp yaw so
        # the robot-side layout remains consistent for each orientation.
        sb_wall_t = 0.004
        sb_wall_h = 0.006
        sb_half_x = self.SPAWN_HALF_X
        sb_half_y = self.SPAWN_HALF_Y

        # Push the box well back from the ramp's high end so the gripper
        # has a clear approach to pick up balls without clipping the ramp
        # structure.  Clearance from ramp_leg_hi
        # (local y ∈ [-0.125, -0.085]) and the table edge.
        spawn_local_y = high_local_y - 0.180  # ≈ -0.285 for 15° ramp
        sb_cx, sb_cy = rot2d(0.0, spawn_local_y)
        bz_base = self._table_height + sb_wall_t
        bz_wall = self._table_height + sb_wall_t * 2 + sb_wall_h
        self._place_static("spawn_base",
                           [sb_cx, sb_cy, bz_base], quat=q_yaw_only)

        sb_fx, sb_fy = rot2d(0.0, spawn_local_y + sb_half_y + sb_wall_t)
        sb_bx, sb_by = rot2d(0.0, spawn_local_y - sb_half_y - sb_wall_t)
        sb_lx, sb_ly = rot2d(-sb_half_x - sb_wall_t, spawn_local_y)
        sb_rx, sb_ry = rot2d( sb_half_x + sb_wall_t, spawn_local_y)
        self._place_static("spawn_front", [sb_fx, sb_fy, bz_wall], quat=q_yaw_only)
        self._place_static("spawn_back",  [sb_bx, sb_by, bz_wall], quat=q_yaw_only)
        self._place_static("spawn_left",  [sb_lx, sb_ly, bz_wall], quat=q_yaw_only)
        self._place_static("spawn_right", [sb_rx, sb_ry, bz_wall], quat=q_yaw_only)

        # Record spawn-box geometry for object placement below
        spawn_box = dict(
            local_y=spawn_local_y,
            half_x=sb_half_x,
            half_y=sb_half_y,
            top_z=bz_base + sb_wall_t,   # interior floor z
        )

        # ── Bins — randomly swap sphere/cube left-right ───────────────
        swap = bool(self._rng.randint(0, 2))
        bw        = self._bin_width
        wall_t    = 0.004
        wall_h    = 0.035
        bin_inner = bw / 2

        bin_along   = 0.065
        side_offset = bin_inner + wall_t + 0.005

        sphere_lx = (side_offset if swap else -side_offset)
        cube_lx   = (-side_offset if swap else side_offset)

        def place_bin(prefix, local_x, along):
            bly = low_local_y + along
            cx, cy   = rot2d(local_x, bly)
            bz_base  = self._table_height + wall_t
            bz_wall  = self._table_height + wall_t * 2 + wall_h

            self._place_static(f"{prefix}_base",  [cx, cy, bz_base], quat=q_yaw_only)
            fx,  fy  = rot2d(local_x, bly + bin_inner + wall_t)
            bkx, bky = rot2d(local_x, bly - bin_inner - wall_t)
            lwx, lwy = rot2d(local_x - bin_inner - wall_t, bly)
            rwx, rwy = rot2d(local_x + bin_inner + wall_t, bly)
            self._place_static(f"{prefix}_front", [fx,  fy,  bz_wall], quat=q_yaw_only)
            self._place_static(f"{prefix}_back",  [bkx, bky, bz_wall], quat=q_yaw_only)
            self._place_static(f"{prefix}_left",  [lwx, lwy, bz_wall], quat=q_yaw_only)
            self._place_static(f"{prefix}_right", [rwx, rwy, bz_wall], quat=q_yaw_only)

        place_bin("bin_sphere", sphere_lx, bin_along)
        place_bin("bin_cube",   cube_lx,   bin_along)

        # Record world-frame bin centres for scoring
        sly = low_local_y + bin_along
        sbx, sby = rot2d(sphere_lx, sly)
        cbx, cby = rot2d(cube_lx,   sly)
        self._sphere_bin_world_xy = np.array([sbx, sby])
        self._cube_bin_world_xy   = np.array([cbx, cby])

        # ── Workspace barrier ─────────────────────────────────────────
        fence_cy_local = low_local_y + bin_along
        fence_depth    = bin_along + bin_inner + wall_t + 0.02
        fence_half_w   = side_offset + bin_inner + 0.02

        frt_x, frt_y = rot2d(0.0,           fence_cy_local + fence_depth)
        bfl_x, bfl_y = rot2d(-fence_half_w,  fence_cy_local)
        bfr_x, bfr_y = rot2d( fence_half_w,  fence_cy_local)

        barrier_z = self._table_height + 0.18
        self._place_static("barrier_front", [frt_x, frt_y, barrier_z], quat=q_yaw_only)
        self._place_static("barrier_left",  [bfl_x, bfl_y, barrier_z], quat=q_yaw_only)
        self._place_static("barrier_right", [bfr_x, bfr_y, barrier_z], quat=q_yaw_only)

        # Top panel: sits just above the vertical walls, covers full enclosure.
        # Y centre is halfway between the open ramp-exit edge and the front wall.
        fence_back_y = low_local_y + 0.01   # back edge of the enclosure (ramp-exit side)
        top_cy_local = (fence_back_y + fence_cy_local + fence_depth) / 2.0
        top_x, top_y = rot2d(0.0, top_cy_local)
        top_z = self._table_height + 0.18 * 2 + 0.004   # just above wall tops
        self._place_static("barrier_top", [top_x, top_y, top_z], quat=q_yaw_only)

        # Ramp-side barrier: seals the enclosure at the ramp exit so the arm
        # cannot reach through.  Collision groups are patched below so balls
        # pass through it freely.
        bk_x, bk_y = rot2d(0.0, low_local_y + 0.01)
        self._place_static("barrier_back", [bk_x, bk_y, self._table_height + 0.18],
                           quat=q_yaw_only)

        # ── Place objects inside the spawn box ────────────────────────
        # One row along local x.  With shuffling on, the balls take random
        # positions along the row (uniform subject to SPAWN_MIN_GAP between
        # neighbours and SPAWN_WALL_GAP to the short walls) in a random
        # colour order; otherwise they are spaced evenly, blue first.  Each
        # ball is also jittered by up to ball_layout.jitter along local y.
        all_names = [f"sphere_{i}" for i in range(self._num_spheres)] + \
                    [f"box_{i}"    for i in range(self._num_cubes)]
        n_total = len(all_names)
        pitch = 2 * self._object_size + self.SPAWN_MIN_GAP
        reach = self.SPAWN_HALF_X - self._object_size - self.SPAWN_WALL_GAP
        slack = 2 * reach - (n_total - 1) * pitch
        assert slack >= 0, "spawn row too short for the balls in play"
        if self._shuffle_balls:
            offsets = np.sort(self._rng.uniform(0.0, slack, n_total))
            order = self._rng.permutation(n_total)
        else:
            offsets = np.full(n_total, slack / 2)
            order = np.arange(n_total)
        xs = -reach + offsets + pitch * np.arange(n_total)
        ys = self._rng.uniform(-self._ball_jitter, self._ball_jitter, n_total)
        slots = np.stack([xs[order], ys], axis=1)

        obj_z = spawn_box["top_z"] + self._object_size + 0.002

        for name, (lx, ly) in zip(all_names, slots):
            ox, oy = rot2d(lx, spawn_box["local_y"] + ly)
            self._place_object(name, [ox, oy, obj_z])

        # Park the objects this episode does not use.  They are stripped
        # from the obs dict by _parked_objects(), so the agent sees a
        # spawn box holding exactly n_total balls.
        self._park_objects(self._parked)

        sim.forward()

        # ── Selective collision: ramp-side panel blocks arm, not balls ─
        self._configure_barrier_collisions(sim)

        # Zero all object velocities to prevent contact-resolution explosions.
        for name in all_names:
            self._set_object_velocity(name, [0, 0, 0], [0, 0, 0])

        obs = self._rebuild_obs()
        return obs, info

    def _configure_barrier_collisions(self, sim):
        """Patch contype/conaffinity so the ramp-side panel stops the arm but not balls.

        MuJoCo collides geoms A and B iff
        ``(A.contype & B.conaffinity) | (B.contype & A.conaffinity) != 0``.
        Collision matrix, applied only to geoms that already collide
        (visual geoms stay non-colliding):

          - barrier_back   contype=2 conaffinity=2  (arm only, so balls
            rolling off the ramp enter the enclosure)
          - other barrier panels  contype=3 conaffinity=3  (arm and balls,
            so balls cannot reach the bins without the ramp)
          - arm geoms      contype=3 conaffinity=3  (bits 0+1 → also hit table)
          - everything else: default contype=1 conaffinity=1
        """
        BARRIER_PANELS = ("barrier_front", "barrier_left", "barrier_right",
                          "barrier_top")
        ARM_BODY_PREFIXES = ("robot0_", "gripper0_")

        for gid in range(sim.model.ngeom):
            if sim.model.geom_contype[gid] == 0 and sim.model.geom_conaffinity[gid] == 0:
                continue
            body_name = sim.model.body_id2name(sim.model.geom_bodyid[gid]) or ""
            geom_name = sim.model.geom_id2name(gid) or ""

            if geom_name.startswith("barrier_back_"):
                sim.model.geom_contype[gid]    = 2
                sim.model.geom_conaffinity[gid] = 2
            elif (any(geom_name.startswith(f"{bn}_") for bn in BARRIER_PANELS)
                  or any(body_name.startswith(p) for p in ARM_BODY_PREFIXES)):
                sim.model.geom_contype[gid]    = 3
                sim.model.geom_conaffinity[gid] = 3

    def _check_termination(self, raw_obs):
        # Failure: a ball landed in the wrong bin — unrecoverable.
        if self._any_ball_in_wrong_bin():
            return True
        # Success: everything correctly sorted.
        if self._compute_task_score(raw_obs) >= self._success_score_threshold:
            return True
        # Terminate when any dynamic object falls off the table (well below
        # bin floor level) — catches both initial explosions and mid-episode
        # falls before they trigger NaN from high-speed floor impact.
        fall_z = self._table_height - 0.25
        sim = self._env.sim
        sids, cids = self._ball_body_ids()
        for bid in sids + cids:
            z = sim.data.body_xpos[bid][2]
            if np.isfinite(z) and z < fall_z:
                return True
        return False
