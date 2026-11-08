"""Slide Catch — catch cubes sliding off a continuously tilting incline.

Task ID: ``ib-p2-slide-catch-v1``

Setup
-----
A flat incline plate sits on the table, hinged at its lower edge. The plate
starts tilted at a small initial angle and tilts upward throughout the
episode. Small flat cubes rest on the incline, each with its own friction
coefficient. As the tilt grows past each cube's static-friction threshold,
that cube starts to slide (not roll, since cubes are flat-faced) toward the
lower edge.

Randomization (shipped config, re-drawn every reset)
----------------------------------------------------
* 2-3 cubes, each with its own friction, size and spot on the plate, so
  which cube slides first, when, and in which lane changes per seed.
* The tilt schedule: start angle, rate, and one mid-episode event that
  either pauses the tilt or speeds it up.
* The incline's pose (sideways shift and yaw about its hinge, the hinge
  line always over the table's robot-facing edge) and colour; cube and
  light colours; the arm's start pose.

Every drawn scene is checked against a 1-D sliding model of each cube
(:meth:`_predict_departures`) and redrawn unless a teleoperator moving the
bin at ``feasibility.teleop_speed`` per axis, with ``reaction_s`` to react,
can reach each cube's drop point before it leaves the plate.

The robot holds a small open-top bin: welded to the gripper from the first
frame (``bin_attached_to_gripper``, the shipped default), or else standing on
the table to be grasped by its handle. Its job is to position the bin just
past the lower edge of the incline and catch every cube before it falls to
the floor.

Success: every cube is caught: inside the bin's inner footprint, between
its floor and rim, and moving with the bin. Failure: any cube falls off
(z < failure floor threshold), which also zeroes the score.

The plate's tilt follows its schedule through a PD torque on the plate's
hinge joint, applied each step, so MuJoCo resolves the resulting contact
dynamics on the cubes naturally.
"""

import numpy as np

from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward, progress_from_reset


class SlideCatchV1(SingleArmBase):
    TASK_ID = "ib-p2-slide-catch-v1"
    DESCRIPTION = (
        "Single arm: an incline tilts up from a shallow initial angle. "
        "Two or three cubes with different friction sit on it and begin "
        "to slide at different times. Move the bin held in the gripper "
        "to the bottom of the incline to catch every cube before it "
        "falls."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._bin_position_at_reset = 0.0

        # ── Incline geometry (flat plate, hinged at its lower edge) ──
        self._incline_length = task_cfg.get("incline_length", 0.32)
        self._incline_width = task_cfg.get("incline_width", 0.18)
        self._incline_thickness = task_cfg.get("incline_thickness", 0.008)
        self._incline_friction = task_cfg.get("incline_friction", 0.35)

        # Tilt schedule (degrees / degrees-per-second).
        self._initial_tilt_deg = task_cfg.get("initial_tilt_deg", 10.0)
        self._tilt_rate_deg_per_s = task_cfg.get("tilt_rate_deg_per_s", 3.5)
        self._max_tilt_deg = task_cfg.get("max_tilt_deg", 55.0)

        # Hinge offset from the table centre, toward the robot. The
        # incline's down-slope direction points at the robot (world -X),
        # so the hinge sits on the table's -X edge.
        self._hinge_offset = task_cfg.get("hinge_offset", 0.25)

        # ── Cubes ──
        # Half-extents: (footprint_x, footprint_y, height_z). The CoM
        # height is the third value; the footprint half-widths control
        # the tipping angle (tan(θ_tip) = footprint / height). Default
        # is a flat tile so cubes slide rather than topple on the slope.
        self._cube_size = task_cfg.get("cube_size", 0.018)
        self._cube_footprint = task_cfg.get("cube_footprint", 0.018)
        self._cube_height = task_cfg.get("cube_height", 0.006)
        self._num_cubes = task_cfg.get("num_cubes", 3)
        # Per-cube static-friction range, sampled uniformly each reset.
        # Listed in down-slope order: index 0 = top of slope, index N-1
        # = nearest the hinge (slides off first). When provided, this
        # overrides the single ``cube_friction_range`` knob.
        per_cube_ranges = task_cfg.get("cube_friction_ranges")
        if per_cube_ranges is not None:
            self._cube_friction_ranges = [
                tuple(r) for r in per_cube_ranges
            ]
        else:
            self._cube_friction_ranges = None
        # Single-range fallback (used when cube_friction_ranges is not
        # set).
        self._cube_friction_range = tuple(
            task_cfg.get("cube_friction_range", [0.20, 0.55])
        )

        # ── Per-episode randomization (all optional; absent = fixed) ──
        # Cube count: ``num_cubes`` bodies are built; each episode uses
        # a count drawn from ``num_cubes_range`` and parks the rest.
        lo_hi = task_cfg.get("num_cubes_range")
        self._num_cubes_range = (tuple(int(v) for v in lo_hi) if lo_hi
                                 else (self._num_cubes, self._num_cubes))
        if not 1 <= self._num_cubes_range[0] <= self._num_cubes_range[1] <= self._num_cubes:
            raise ValueError(f"num_cubes_range {self._num_cubes_range} must lie in [1, num_cubes={self._num_cubes}]")
        # Independent per-cube friction (any cube may slide first). Takes
        # precedence over the ordered ``cube_friction_ranges``.
        indep = task_cfg.get("cube_friction_independent")
        self._cube_friction_independent = tuple(indep) if indep else None
        # Cube footprint half-extent range; height = footprint * ratio.
        fp = task_cfg.get("cube_footprint_range")
        self._cube_footprint_range = (tuple(fp) if fp
                                      else (self._cube_footprint, self._cube_footprint))
        self._cube_height_ratio = task_cfg.get(
            "cube_height_ratio", self._cube_height / self._cube_footprint)
        # Tilt schedule ranges and the one mid-episode event.
        r = task_cfg.get("initial_tilt_deg_range")
        self._initial_tilt_range = (tuple(r) if r
                                    else (self._initial_tilt_deg, self._initial_tilt_deg))
        r = task_cfg.get("tilt_rate_deg_per_s_range")
        self._tilt_rate_range = (tuple(r) if r
                                 else (self._tilt_rate_deg_per_s, self._tilt_rate_deg_per_s))
        self._tilt_event_cfg = task_cfg.get("tilt_event")  # None = steady tilt
        # Incline pose and look.
        self._incline_y_shift = float(task_cfg.get("incline_y_shift", 0.0))
        self._incline_yaw_deg = float(task_cfg.get("incline_yaw_deg", 0.0))
        self._incline_color = bool(task_cfg.get("incline_color", False))
        # Arm start: uniform offset of the gripper from its home pose,
        # per axis [x, y, z_low, z_high] -> see _jitter_arm_start.
        self._eef_start_jitter = task_cfg.get("eef_start_jitter")
        # Teleop feasibility check on every drawn scene.
        feas = task_cfg.get("feasibility") or {}
        self._feas_enabled = bool(feas.get("enabled", False))
        self._feas_speed = float(feas.get("teleop_speed", 0.08))
        self._feas_reaction = float(feas.get("reaction_s", 1.0))
        self._feas_latest = float(feas.get("latest_departure_s", 27.0))
        self._feas_slip_margin = float(feas.get("min_slip_margin_deg", 2.0))
        self._feas_tries = int(feas.get("max_tries", 400))
        self._feas_pause_clear = float(feas.get("pause_slip_clearance_deg", 1.5))
        # Cubes may leave up to this much earlier than the model says.
        self._feas_model_margin = float(feas.get("model_margin_s", 1.0))
        # Where the bin centre waits for a cube: past the drop edge
        # (horizontal, down-slope) and below the table top.
        self._catch_out = float(feas.get("catch_out", 0.07))
        self._catch_below = float(feas.get("catch_below", 0.08))
        # 1-D sliding-model calibration (see _predict_departures).
        self._model_mu_scale = float(feas.get("model_mu_scale", 1.0))
        self._model_kinetic_scale = float(feas.get("model_kinetic_scale", 1.0))

        # ── Bin (graspable tray) ──
        self._bin_inner_half_x = task_cfg.get("bin_inner_half_x", 0.06)
        self._bin_inner_half_y = task_cfg.get("bin_inner_half_y", 0.06)
        self._bin_wall = task_cfg.get("bin_wall", 0.004)
        self._bin_wall_height = task_cfg.get("bin_wall_height", 0.025)
        self._bin_handle_height = task_cfg.get("bin_handle_height", 0.05)
        self._bin_handle_half_thickness = task_cfg.get(
            "bin_handle_half_thickness", 0.006
        )
        self._bin_density = task_cfg.get("bin_density", 250)
        self._bin_friction = task_cfg.get("bin_friction", 0.7)

        # Initial bin position on the table (within the robot's reach).
        self._bin_init_x = task_cfg.get("bin_init_x", -0.12)
        self._bin_init_y = task_cfg.get("bin_init_y", -0.18)

        # When True the bin is rigidly attached to the gripper at every
        # step (kinematic weld), and the gripper command is forced to a
        # closed value, so the operator can focus on positioning.
        self._bin_attached = task_cfg.get("bin_attached_to_gripper", True)

        # Episode state.
        self._tilt_deg = 0.0
        self._fallen = False

        self._active_cubes = self._cube_names(self._num_cubes_range[1])
        self._parked = set()
        self._cube_half_heights = {}
        self._staged_reward = self._build_staged_reward(len(self._active_cubes))

    @staticmethod
    def _cube_names(n):
        return ["cube" if i == 0 else f"cube_{i}" for i in range(n)]

    def _build_staged_reward(self, n_cubes):
        """Bin positioning plus one catch stage per cube in play this episode."""
        # Staged reward: bin handling (grasp + position) plus one stage
        # per cube caught, later cubes weighted higher (they are harder to
        # catch). The literal 0.20/0.25/0.30 split assumes
        # the default 3 cubes; other cube counts fall back to an even
        # split of the remaining catch budget. An attached bin starts out
        # held, so the grasp stage's weight goes to positioning instead.
        catch_budget = 0.75
        if n_cubes == 3:
            catch_weights = [0.20, 0.25, 0.30]
        else:
            catch_weights = [catch_budget / n_cubes] * n_cubes

        if self._bin_attached:
            stages = [
                Stage("position_bin_at_incline_base", 0.25,
                      lambda t, o: t._bin_position_progress(o)),
            ]
        else:
            stages = [
                Stage("grasp_bin", 0.10, lambda t, o: t._grasp_bin_progress(o)),
                Stage("position_bin_at_incline_base", 0.15,
                      lambda t, o: t._bin_position_progress(o)),
            ]
        for i, w in enumerate(catch_weights):
            stages.append(Stage(
                f"cube{i + 1}_caught", w,
                lambda t, o, idx=i: 1.0 if t._cube_in_bin(
                    t._env.sim, t._active_cubes[idx]
                ) else 0.0,
            ))
        return StagedReward(
            stages=stages,
            success_fn=lambda t, o: t._all_cubes_caught(),
        )

    # ── Object factories ────────────────────────────────────────────

    def _make_objects(self):
        """The cubes plus the graspable bin."""
        objects = []

        # Cubes — friction values are placeholders; reset() randomises and
        # writes them onto the geom_friction array directly. Geometry
        # is a flat tile (wide footprint, low height) so the cubes slide
        # before they topple on the slope.
        for i in range(self._num_cubes):
            objects.append({
                "type": "box",
                "name": "cube" if i == 0 else f"cube_{i}",
                # Built at the largest footprint; reset() rescales the
                # geoms to this episode's drawn size.
                "size": [self._cube_footprint_range[1],
                         self._cube_footprint_range[1],
                         self._cube_footprint_range[1] * self._cube_height_ratio],
                "rgba": _CUBE_COLORS[i % len(_CUBE_COLORS)],
                "density": 600,
                "friction": [0.5, 0.005, 0.0001],
                "joints": "default",
            })

        # Bin: composite of floor + 4 walls + vertical handle.
        bin_floor_half = self._bin_inner_half_x + self._bin_wall
        bin_floor_half_y = self._bin_inner_half_y + self._bin_wall
        bin_color = [0.55, 0.40, 0.20, 1.0]
        handle_color = [0.30, 0.20, 0.10, 1.0]

        floor_thick = self._bin_wall
        wall_h = self._bin_wall_height
        wall_t = self._bin_wall

        # Sub-locations are body-local coordinates, with the composite
        # body's origin in the centre of the floor's underside.
        floor_z = floor_thick / 2
        wall_z = floor_thick + wall_h / 2
        handle_z = floor_thick + wall_h + self._bin_handle_height / 2

        objects.append({
            "type": "composite",
            "name": "bin",
            "rgba": bin_color,
            "density": self._bin_density,
            "friction": [self._bin_friction, 0.005, 0.0001],
            "joints": "default",
            "sub_objects": [
                {"type": "box", "name": "floor",
                 "size": [bin_floor_half, bin_floor_half_y, floor_thick / 2],
                 "rgba": bin_color, "density": self._bin_density,
                 "friction": [self._bin_friction, 0.005, 0.0001]},
                {"type": "box", "name": "wall_px",
                 "size": [wall_t / 2, bin_floor_half_y, wall_h / 2],
                 "rgba": bin_color, "density": self._bin_density,
                 "friction": [self._bin_friction, 0.005, 0.0001]},
                {"type": "box", "name": "wall_nx",
                 "size": [wall_t / 2, bin_floor_half_y, wall_h / 2],
                 "rgba": bin_color, "density": self._bin_density,
                 "friction": [self._bin_friction, 0.005, 0.0001]},
                {"type": "box", "name": "wall_py",
                 "size": [bin_floor_half, wall_t / 2, wall_h / 2],
                 "rgba": bin_color, "density": self._bin_density,
                 "friction": [self._bin_friction, 0.005, 0.0001]},
                {"type": "box", "name": "wall_ny",
                 "size": [bin_floor_half, wall_t / 2, wall_h / 2],
                 "rgba": bin_color, "density": self._bin_density,
                 "friction": [self._bin_friction, 0.005, 0.0001]},
                # Vertical handle plate (graspable by parallel-jaw).
                {"type": "box", "name": "handle",
                 "size": [self._bin_handle_half_thickness,
                          self._bin_inner_half_y * 0.6,
                          self._bin_handle_height / 2],
                 "rgba": handle_color, "density": self._bin_density,
                 "friction": [self._bin_friction, 0.005, 0.0001]},
            ],
            "sub_locations": [
                [0.0, 0.0, floor_z],
                [bin_floor_half - wall_t / 2, 0.0, wall_z],
                [-(bin_floor_half - wall_t / 2), 0.0, wall_z],
                [0.0, bin_floor_half_y - wall_t / 2, wall_z],
                [0.0, -(bin_floor_half_y - wall_t / 2), wall_z],
                [0.0, 0.0, handle_z],
            ],
            "total_size": [
                bin_floor_half,
                bin_floor_half_y,
                self._bin_wall + wall_h + self._bin_handle_height,
            ],
        })

        return objects

    def _make_static_objects(self):
        """Hinged incline driven kinematically via qpos/qvel.

        The anchor is welded at the hinge world location; the plate is
        a child body on a real MuJoCo hinge joint about local -Y. Each
        step the task writes the target hinge angle (qpos) and angular
        rate (qvel) — so MuJoCo's contact solver sees real surface
        velocity and cubes slide based on friction.
        """
        return [
            {"type": "tilting_incline", "name": "incline",
             "plate_length": self._incline_length,
             "plate_half_width": self._incline_width / 2,
             "plate_half_thickness": self._incline_thickness / 2,
             "plate_rgba": [0.45, 0.55, 0.65, 1.0],
             "plate_friction": [self._incline_friction, 0.005, 0.0001],
             "damping": 100.0,
             "frictionloss": 0.0,
             "joints": None},
        ]

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        sim = self._env.sim

        self._fallen = False
        self._t = 0.0
        # Reset the bin-velocity finite-difference history so the first
        # post-reset snap doesn't compute a huge spurious velocity.
        self._prev_bin_pos = None

        # ── Arm start pose (first: the feasibility check needs the
        #    bin's real start position) ──
        self._jitter_arm_start(sim)
        if self._bin_attached:
            self._snap_bin_to_gripper(sim)
            sim.forward()
            self._prev_bin_pos = None
        bin_start = np.array(sim.data.body_xpos[self._object_body_id("bin")], dtype=float)
        self._bin_start = bin_start

        # ── Incline pose: yaw about the hinge centre and a sideways
        #    shift. The hinge centre moves toward the robot by
        #    |sin(yaw)|·half-width so the hinge line's inner end stays on
        #    the table's robot-facing edge: every cube that slides off
        #    falls past the table, never onto it. ──
        yaw_max = np.deg2rad(self._incline_yaw_deg)
        self._incline_yaw = float(self._rng.uniform(-yaw_max, yaw_max)) if yaw_max else 0.0
        # The sideways shift keeps the incline's footprint within the
        # table's width: a yawed incline shifts away from the side its
        # far corner swings toward.
        cy_, sy_ = np.cos(self._incline_yaw), np.sin(self._incline_yaw)
        half_w = self._incline_width / 2
        corner_y = [sy_ * lx + cy_ * ly for lx in (0.0, self._incline_length) for ly in (-half_w, half_w)]
        table_half_y = self._table_full_size[1] / 2
        lo = max(-self._incline_y_shift, -table_half_y - min(corner_y))
        hi = min(self._incline_y_shift, table_half_y - max(corner_y))
        y_shift = float(self._rng.uniform(lo, hi)) if hi > lo else 0.5 * (lo + hi)
        edge_x = self._table_offset[0] - self._hinge_offset
        hinge_x = edge_x - abs(np.sin(self._incline_yaw)) * self._incline_width / 2
        self._anchor_world_pos = np.array([hinge_x, y_shift, self._table_height])
        self._anchor_quat = np.array([np.cos(self._incline_yaw / 2), 0.0, 0.0,
                                      np.sin(self._incline_yaw / 2)])
        try:
            anchor_bid = sim.model.body_name2id("incline_main")
            sim.model.body_pos[anchor_bid] = self._anchor_world_pos
            sim.model.body_quat[anchor_bid] = self._anchor_quat
        except Exception:
            pass
        plate_rgba = None
        if self._incline_color:
            plate_rgba = np.array([*self._rng.uniform(0.15, 0.85, size=3), 1.0])
            for gid in self._plate_geoms(sim):
                sim.model.geom_rgba[gid] = plate_rgba

        # ── Draw the cubes and tilt schedule; redraw until feasible ──
        self._scene = self._draw_scene(bin_start)
        sc = self._scene
        self._active_cubes = self._cube_names(sc["n"])
        self._cube_half_heights = {}
        self._parked = set(self._cube_names(self._num_cubes)) - set(self._active_cubes)
        self._cube_frictions = list(sc["mu"])
        self._departures = self._predict_departures(sc)
        self._staged_reward = self._build_staged_reward(sc["n"])
        self._staged_reward.reset()

        self._tilt_deg = self._tilt_at(0.0)
        self._set_hinge_state(sim, self._tilt_deg, 0.0)
        sim.forward()

        # ── Place each cube flat on the plate ──
        plate_quat = self._quat_mul(
            self._anchor_quat,
            np.array([np.cos(np.deg2rad(self._tilt_deg) / 2), 0.0,
                      -np.sin(np.deg2rad(self._tilt_deg) / 2), 0.0]))
        half_thk = self._incline_thickness / 2
        for i, cube_name in enumerate(self._active_cubes):
            self._set_cube_friction(sim, cube_name, sc["mu"][i])
            self._set_cube_size(sim, cube_name, sc["fp"][i])
            self._cube_half_heights[cube_name] = sc["fp"][i] * self._cube_height_ratio
            local_z = 2 * half_thk + sc["fp"][i] * self._cube_height_ratio + 0.001
            world_pos = self._plate_local_to_world(
                np.array([sc["x"][i], sc["y"][i], local_z]), self._tilt_deg)
            self._place_object(cube_name, world_pos.tolist(), quat=plate_quat.tolist())
            self._set_object_velocity(cube_name, [0, 0, 0], [0, 0, 0])
        self._park_objects(self._parked)

        # ── Cube colours: random, but never close to the incline's or
        #    the bin's, so every cube stays visible ──
        if self._rand_object_color:
            avoid = [np.array([0.55, 0.40, 0.20])]
            avoid.append(plate_rgba[:3] if plate_rgba is not None else np.array([0.45, 0.55, 0.65]))
            for cube_name in self._active_cubes:
                for _ in range(100):
                    rgba = self._random_object_color()
                    if all(np.linalg.norm(rgba[:3] - a) >= 0.35 for a in avoid):
                        break
                self._set_object_color(cube_name, rgba)

        # ── Place the bin ──
        # In attached mode the bin is welded kinematically to the gripper
        # (placed above, before the scene draw).
        if not self._bin_attached:
            # Free-standing on the table within the robot's reach. The
            # bin spawns yawed 90° about world Z so the handle's thin
            # axis aligns with world Y.
            bx = float(self._bin_init_x + self._rng.uniform(-0.04, 0.04))
            by = float(self._bin_init_y + self._rng.uniform(-0.04, 0.04))
            bz = self._table_height + 0.001
            bin_quat = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
            self._place_object("bin", [bx, by, bz], quat=bin_quat)

        sim.forward()
        self._bin_position_at_reset = self._bin_position_shaping()
        obs = self._rebuild_obs()
        info = dict(info)
        info["num_cubes"] = sc["n"]
        return obs, info

    def _parked_objects(self):
        return self._parked

    # ── Scene draw and teleop feasibility ───────────────────────────

    def _sample_scene(self, n):
        """One draw of everything the *n* cubes' timing depends on."""
        rng = self._rng
        fp = [float(rng.uniform(*self._cube_footprint_range)) for _ in range(n)]

        if self._cube_friction_independent is not None:
            mu = [float(rng.uniform(*self._cube_friction_independent)) for _ in range(n)]
            # Anywhere along the slope: timing comes from friction and
            # distance together, so position alone does not give the order.
            fp_max = self._cube_footprint_range[1]
            x = [float(rng.uniform(0.06, self._incline_length - fp_max - 0.04)) for _ in range(n)]
        else:
            # Ordered layout: friction falls with down-slope slot,
            # so the cube nearest the hinge slides first.
            if self._cube_friction_ranges is not None:
                ranges = list(self._cube_friction_ranges)
                while len(ranges) < n:
                    ranges.append(ranges[-1])
                mu = [float(rng.uniform(lo_, hi_)) for lo_, hi_ in ranges[:n]]
            else:
                f_lo, f_hi = self._cube_friction_range
                mu = [float(m) for m in np.linspace(f_hi, f_lo, n)]
            top_margin, bottom_margin = 0.04, 0.06
            slots = (np.linspace(self._incline_length - top_margin, bottom_margin, n)
                     if n > 1 else np.array([self._incline_length / 2]))
            x_min = self._cube_footprint_range[1] + 0.005
            x_max = self._incline_length - self._cube_footprint_range[1] - 0.005
            x = [float(np.clip(s + rng.uniform(-0.03, 0.03), x_min, x_max)) for s in slots]

        # Lanes across the width never overlap, so cubes cannot collide.
        fp_max = self._cube_footprint_range[1]
        y_lim = self._incline_width / 2 - fp_max - 0.005
        min_sep = 2 * fp_max + 0.02
        y = []
        for _ in range(n):
            for _attempt in range(50):
                cand = float(rng.uniform(-y_lim, y_lim))
                if all(abs(cand - prev) >= min_sep for prev in y):
                    break
            y.append(cand)

        tilt0 = float(rng.uniform(*self._initial_tilt_range))
        rate = float(rng.uniform(*self._tilt_rate_range))
        event = None
        ev = self._tilt_event_cfg
        if ev:
            kind = str(ev["kinds"][int(rng.randint(len(ev["kinds"])))])
            event = {"kind": kind, "t": float(rng.uniform(*ev["time_s"]))}
            if kind == "pause":
                event["dur"] = float(rng.uniform(*ev["pause_s"]))
            elif kind == "speedup":
                event["factor"] = float(rng.uniform(*ev["speedup"]))
            else:
                raise ValueError(f"unknown tilt_event kind {kind!r}")
        return {"n": n, "fp": fp, "mu": mu, "x": x, "y": y,
                "tilt0": tilt0, "rate": rate, "event": event}

    def _draw_scene(self, bin_start):
        """Draw scenes until one passes the teleop feasibility check. The
        cube count is drawn once, so the check does not bias it (3-cube
        scenes need up to ~760 draws; 2,000 seeds checked). Should
        ``max_tries`` ever run out, one cube is dropped (never below the
        range's minimum) and the draw goes on."""
        lo, hi = self._num_cubes_range
        n = int(self._rng.randint(lo, hi + 1))
        attempt = 0
        while True:
            for _ in range(self._feas_tries):
                attempt += 1
                sc = self._sample_scene(n)
                if not self._feas_enabled or self._scene_feasible(sc, bin_start)[0]:
                    sc["tries"] = attempt
                    return sc
            if n <= lo:
                raise RuntimeError(f"slide catch: no teleop-feasible scene in {attempt} draws")
            n -= 1

    _PROFILE_DT = 0.01   # s; the tilt profile's time step (= one control step at 100 Hz)
    _PROFILE_T = 45.0    # s; longer than any episode
    _EVENT_RAMP = 0.5    # s; the tilt rate eases in and out of an event

    def _rate_at(self, t, sc):
        """Tilt rate (deg/s) at time t: steady, then one event entered and
        left with linear ramps, so the plate never jolts the cubes."""
        rate, ev = sc["rate"], sc["event"]
        if ev is None or t <= ev["t"]:
            return rate
        u = t - ev["t"]
        ramp = self._EVENT_RAMP
        if ev["kind"] == "pause":
            if u < ramp:                      # ease to a stop
                return rate * (1.0 - u / ramp)
            if u < ramp + ev["dur"]:          # hold
                return 0.0
            return rate * min(1.0, (u - ramp - ev["dur"]) / ramp)  # ease back
        # speed-up: ease from rate to rate * factor
        return rate * (1.0 + (ev["factor"] - 1.0) * min(1.0, u / ramp))

    def _tilt_profile(self, sc):
        """Tilt (deg) at every _PROFILE_DT step, cached in the scene."""
        prof = sc.get("_profile")
        if prof is None:
            n = int(self._PROFILE_T / self._PROFILE_DT) + 1
            rates = np.array([self._rate_at(k * self._PROFILE_DT, sc) for k in range(n)])
            prof = sc["tilt0"] + np.concatenate([[0.0], np.cumsum(0.5 * (rates[1:] + rates[:-1]))]) * self._PROFILE_DT
            prof = np.minimum(prof, self._max_tilt_deg)
            sc["_profile"] = prof
        return prof

    def _tilt_at(self, t, sc=None):
        """Target tilt (deg) at time t (s) for scene *sc* (default: current)."""
        sc = sc if sc is not None else self._scene
        prof = self._tilt_profile(sc)
        k = t / self._PROFILE_DT
        i = int(min(max(k, 0.0), len(prof) - 2))
        f = min(max(k - i, 0.0), 1.0)
        return float(prof[i] * (1.0 - f) + prof[i + 1] * f)

    def _predict_departures(self, sc, dt=0.01, t_max=40.0):
        """Per cube, when it starts sliding and when its centre passes the
        drop edge, from a 1-D sliding model: at rest until
        ``tan(tilt) > mu_s``, then ``a = g (sin(tilt) - mu_k cos(tilt))``
        along the slope. ``model_mu_scale`` and ``model_kinetic_scale``
        fit mu_s and mu_k to MuJoCo's soft contacts. ``None`` where the
        cube never leaves within t_max."""
        out = []
        for mu, x0 in zip(sc["mu"], sc["x"]):
            m_s = mu * self._model_mu_scale
            m_k = mu * self._model_kinetic_scale
            s, v, t, start, dep = float(x0), 0.0, 0.0, None, None
            while t < t_max:
                th = np.deg2rad(self._tilt_at(t, sc))
                moving = v > 0.0 or np.tan(th) > m_s
                a = 9.81 * (np.sin(th) - m_k * np.cos(th))
                if moving and (v > 0.0 or a > 0.0):
                    if start is None:
                        start = t
                    v = max(0.0, v + a * dt)
                    s -= v * dt
                    if s <= 0.0:
                        dep = t
                        break
                t += dt
            out.append({"start": start, "depart": dep})
        return out

    def _edge_coords(self, p):
        """World point -> (along the drop edge, out past it toward the robot)."""
        c, s = np.cos(self._incline_yaw), np.sin(self._incline_yaw)
        d = np.asarray(p[:2], dtype=float) - self._anchor_world_pos[:2]
        return float(-s * d[0] + c * d[1]), float(-(c * d[0] + s * d[1]))

    @property
    def _approach_clearance(self):
        """How far out past the edge line the bin centre stays while its rim
        is above the table top: the yawed incline's hinge line overhangs
        the table edge, and the bin must not come down through it. The bin
        does not yaw with the incline, so its reach toward the edge is its
        half-width times (|cos yaw| + |sin yaw|)."""
        half = self._bin_inner_half_x + self._bin_wall
        yaw = getattr(self, "_incline_yaw", 0.0)
        return half * (abs(np.cos(yaw)) + abs(np.sin(yaw))) + 0.02

    def _catch_point(self, local_y):
        """Bin-body position that catches a cube leaving at plate lane *local_y*."""
        c, s = np.cos(self._incline_yaw), np.sin(self._incline_yaw)
        edge = self._anchor_world_pos + np.array([-s * local_y, c * local_y, 0.0])
        out_dir = np.array([-c, -s, 0.0])  # horizontal down-slope direction
        p = edge + self._catch_out * out_dir
        p[2] = self._table_height - self._catch_below
        return p

    def _scene_feasible(self, sc, bin_start):
        """Can a teleoperator catch every cube? Moving the bin at
        ``teleop_speed`` per axis (axes at once), allowing ``reaction_s``
        per catch, from the bin's start pose to each cube's catch point in
        departure order, arriving ``model_margin_s`` before the model says
        that cube leaves the plate."""
        for mu in sc["mu"]:
            if np.rad2deg(np.arctan(mu * self._model_mu_scale)) < sc["tilt0"] + self._feas_slip_margin:
                return False, "a cube slides from the start"
        ev = sc["event"]
        if ev is not None and ev["kind"] == "pause":
            # A pause that holds the tilt at a cube's slip angle leaves
            # that cube on a knife edge (creep vs. slide): not predictable.
            hold = self._tilt_at(ev["t"] + self._EVENT_RAMP + 0.5 * ev["dur"], sc)
            for mu in sc["mu"]:
                slip = np.rad2deg(np.arctan(mu * self._model_mu_scale))
                if abs(hold - slip) < self._feas_pause_clear:
                    return False, "a pause holds the tilt at a cube's slip angle"
        deps = self._predict_departures(sc)
        if any(d["depart"] is None or d["depart"] > self._feas_latest for d in deps):
            return False, "a cube leaves too late"
        pos, t_prev = np.asarray(bin_start, dtype=float), 0.0
        # A bin that starts within the approach clearance of the edge line
        # first moves out to it (and later back in): extra distance.
        detour = 2.0 * max(0.0, self._approach_clearance - self._edge_coords(pos)[1])
        for i in sorted(range(sc["n"]), key=lambda k: deps[k]["depart"]):
            target = self._catch_point(sc["y"][i])
            need = (float(np.max(np.abs(target - pos))) + detour) / self._feas_speed + self._feas_reaction
            detour = 0.0
            if deps[i]["depart"] - self._feas_model_margin - t_prev < need:
                return False, "not enough time to reach a cube"
            pos, t_prev = target, deps[i]["depart"]
        return True, ""

    # ── Arm start, cube size and incline helpers ────────────────────

    def _jitter_arm_start(self, sim):
        """Move the gripper by a random offset from its home pose (same
        orientation), solving joint angles by damped least squares."""
        if not self._eef_start_jitter:
            return
        import mujoco
        jx, jy, jz_lo, jz_hi = (float(v) for v in self._eef_start_jitter)
        robot = self._env.robots[0]
        qidx = np.asarray(robot._ref_joint_pos_indexes)
        vidx = np.asarray(robot._ref_joint_vel_indexes)
        m, d = sim.model._model, sim.data._data
        site = sim.model.site_name2id("gripper0_right_grip_site")
        home = d.site_xpos[site].copy()
        R0 = d.site_xmat[site].reshape(3, 3).copy()
        target = home + np.array([self._rng.uniform(-jx, jx), self._rng.uniform(-jy, jy),
                                  self._rng.uniform(jz_lo, jz_hi)])
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(200):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[site].reshape(3, 3)
            err = np.concatenate([target - d.site_xpos[site],
                                  0.5 * sum(np.cross(R[:, k], R0[:, k]) for k in range(3))])
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            J = np.vstack([jacp[:, vidx], jacr[:, vidx]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            d.qpos[qidx] += np.clip(dq, -0.1, 0.1)
        d.qvel[vidx] = 0.0
        sim.forward()

    def _set_cube_size(self, sim, cube_name, fp):
        """Rescale a cube's box geoms (and its mass) to footprint half-extent *fp*."""
        h = fp * self._cube_height_ratio
        size = np.array([fp, fp, h])
        bid = sim.model.body_name2id(f"{cube_name}_main")
        aabb = getattr(sim.model._model, "geom_aabb", None)
        for gid in range(sim.model.ngeom):
            if sim.model.geom_bodyid[gid] == bid and sim.model.geom_type[gid] == 6:  # mjGEOM_BOX
                sim.model.geom_size[gid] = size
                sim.model.geom_rbound[gid] = float(np.linalg.norm(size))
                if aabb is not None:
                    aabb[gid, 3:] = size
        mass = 600.0 * 8 * fp * fp * h
        sim.model.body_mass[bid] = mass
        sim.model.body_inertia[bid] = mass / 3.0 * np.array([fp**2 + h**2, fp**2 + h**2, 2 * fp**2])

    @staticmethod
    def _plate_geoms(sim):
        bid = sim.model.jnt_bodyid[sim.model.joint_name2id("incline_hinge")]
        return [g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid]

    # ── Step: drive the tilt forward each frame ─────────────────────

    def step(self, action: np.ndarray):
        self._check_open()
        # Advance the target tilt each step and apply a PD torque to
        # the hinge joint via qfrc_applied. The PD controller drives
        # the plate to track the target angle, while the integrator
        # treats it as an external force — so the surface velocity at
        # cube contacts is real and friction engages correctly.
        dt = 1.0 / float(self.control_freq)
        prev_tilt_deg = self._tilt_deg
        self._t += dt
        self._tilt_deg = self._tilt_at(self._t)

        target_vel = np.deg2rad(self._tilt_deg - prev_tilt_deg) / dt
        self._apply_hinge_pd_torque(self._env.sim,
                                    np.deg2rad(self._tilt_deg),
                                    target_vel)

        # Force the gripper component closed every step so the operator
        # cannot accidentally drop the bin. The action's last dim is the
        # gripper command for a single-arm Panda.
        if self._bin_attached:
            action = np.asarray(action, dtype=np.float64).copy()
            if action.size >= 1:
                action[-1] = 1.0

        out = super().step(action)

        # After physics, snap the bin back onto the gripper so it stays
        # rigidly attached regardless of contact forces with the cubes
        # or the incline.
        if self._bin_attached:
            sim = self._env.sim
            self._snap_bin_to_gripper(sim)
            sim.forward()
        return out

    # ── Attached-bin (kinematic weld) helpers ───────────────────────

    def _bin_grip_offset(self):
        """Pose of the bin body in the gripper (eef) frame.

        The bin's body origin is at the centre of the floor's underside.
        The handle midpoint sits at body-local (0, 0, h) where
        h = floor_thick + wall_h + handle_height/2. We want the gripper
        site at the handle midpoint, with the bin oriented so its handle
        thin axis (body local +x) aligns with the gripper's finger-close
        axis (eef local +y).

        Returns
        -------
        p_rel : (3,) ndarray
            Bin body origin in the eef frame.
        q_rel : (4,) ndarray
            Bin orientation (w, x, y, z) in the eef frame.
        """
        h = (
            self._bin_wall  # floor thickness
            + self._bin_wall_height
            + self._bin_handle_height / 2
        )
        # Bin's body axes in eef axes: bin_x → eef_y, bin_y → eef_x,
        # bin_z → eef_(-z). That's a 180° rotation about (1,1,0)/√2.
        a = 1.0 / np.sqrt(2.0)
        q_rel = np.array([0.0, a, a, 0.0])
        # Bin origin in eef coords: applying R_be to bin-local
        # (0,0,h) gives (0, 0, -h) in eef, so the bin floor sits at
        # eef_z = -h (i.e. below the gripper for a top-down grasp).
        # We want eef = bin_origin + R_wb * (0,0,h_in_bin) → solve for
        # the bin origin in eef coords: it's just (0, 0, +h) because
        # R_be * (0,0,-h) = (0,0,h).
        p_rel = np.array([0.0, 0.0, h])
        return p_rel, q_rel

    @staticmethod
    def _quat_mul(q1, q2):
        """Hamilton product of two (w, x, y, z) quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vector v by quaternion q (w, x, y, z)."""
        w, x, y, z = q
        # Standard formula: v' = v + 2 * cross(qvec, cross(qvec, v) + w*v)
        qv = np.array([x, y, z])
        t = 2.0 * np.cross(qv, v)
        return v + w * t + np.cross(qv, t)

    def _snap_bin_to_gripper(self, sim):
        """Rewrite the bin's free-joint qpos to track the gripper rigidly.

        Sets the bin's qvel to the finite-difference velocity from the
        previous snap so MuJoCo's contact solver sees a non-zero surface
        velocity at the bin floor — without that, fast-falling cubes
        can tunnel through the bin's thin geometry.
        """
        try:
            site_id = sim.model.site_name2id("gripper0_right_grip_site")
        except Exception:
            return
        eef_pos = np.array(sim.data.site_xpos[site_id], dtype=float)
        # site_xmat is a 9-vector row-major rotation matrix.
        R = np.array(sim.data.site_xmat[site_id], dtype=float).reshape(3, 3)
        # Convert to (w, x, y, z) quat.
        q_eef = _mat_to_quat(R)

        p_rel, q_rel = self._bin_grip_offset()
        bin_pos = eef_pos + self._quat_rotate(q_eef, p_rel)
        bin_quat = self._quat_mul(q_eef, q_rel)

        # Estimate the bin's linear velocity from the previous snap so
        # the contact solver sees a moving bin (not a teleporting one).
        dt = 1.0 / float(self.control_freq)
        if getattr(self, "_prev_bin_pos", None) is not None:
            lin_vel = (bin_pos - self._prev_bin_pos) / dt
        else:
            lin_vel = np.zeros(3)
        self._prev_bin_pos = bin_pos.copy()

        self._place_object("bin", bin_pos.tolist(), quat=bin_quat.tolist())
        # Write the linear velocity into the bin's free-joint qvel so
        # MuJoCo doesn't think the bin is stationary on contact.
        try:
            obj = self._env.get_object("bin")
            if obj is not None and obj.joints:
                jname = obj.joints[0]
                qvel_addr = sim.model.get_joint_qvel_addr(jname)
                if isinstance(qvel_addr, tuple):
                    lo, hi = qvel_addr
                    # Free joint qvel layout: [vx, vy, vz, wx, wy, wz].
                    sim.data.qvel[lo:lo + 3] = lin_vel
                    sim.data.qvel[lo + 3:hi] = 0.0
                else:
                    sim.data.qvel[qvel_addr:qvel_addr + 3] = lin_vel
                    sim.data.qvel[qvel_addr + 3:qvel_addr + 6] = 0.0
        except Exception:
            pass

    # ── Tilt helpers ────────────────────────────────────────────────

    # PD-controller gains for the hinge tracking (units: torque per rad
    # and torque per rad/s respectively). The plate has a substantial
    # rotational inertia (mass 50 kg, length 0.55 m, so I ≈ 5.0 kg·m²
    # about the hinge), and the cubes apply tiny torques, so kp can be
    # high without instability. kd ≈ 2·sqrt(I·kp) for critical damping.
    _HINGE_KP = 20000.0
    _HINGE_KD = 200.0

    def _hinge_addrs(self, sim):
        jid = sim.model.joint_name2id("incline_hinge")
        return (
            int(sim.model.jnt_qposadr[jid]),
            int(sim.model.jnt_dofadr[jid]),
        )

    def _set_hinge_state(self, sim, tilt_deg, rate_rad_per_s):
        """One-time direct write of qpos/qvel (used at reset)."""
        try:
            qp, qv = self._hinge_addrs(sim)
        except Exception:
            return
        sim.data.qpos[qp] = float(np.deg2rad(tilt_deg))
        sim.data.qvel[qv] = float(rate_rad_per_s)

    def _apply_hinge_pd_torque(self, sim, target_rad, target_vel):
        """Apply a PD torque on the hinge joint via qfrc_applied.

        This is an external generalized force seen by MuJoCo's
        integrator, so the resulting plate motion is a real dynamics
        rotation — contact velocities at cube contacts are correct
        and friction engages naturally.
        """
        try:
            qp, qv = self._hinge_addrs(sim)
        except Exception:
            return
        cur_rad = float(sim.data.qpos[qp])
        cur_vel = float(sim.data.qvel[qv])
        torque = (
            self._HINGE_KP * (target_rad - cur_rad)
            + self._HINGE_KD * (target_vel - cur_vel)
        )
        sim.data.qfrc_applied[qv] = float(torque)

    def _plate_local_to_world(self, local_xyz, tilt_deg):
        """Convert a point in the plate's local frame to world.

        Plate-local frame: x=0 = low edge (hinge), +x = up-slope,
        +z = surface outward normal. The plate rotates by α about
        local -Y (the hinge axis); positive α lifts the high end.
        The anchor then yaws the whole incline by ``_incline_yaw``
        about world Z.
        """
        alpha = np.deg2rad(tilt_deg)
        c, s = np.cos(alpha), np.sin(alpha)
        lx, ly, lz = float(local_xyz[0]), float(local_xyz[1]), float(local_xyz[2])
        # R(-Y, α): (lx, ly, lz) → (c·lx − s·lz, ly, s·lx + c·lz)
        px = c * lx - s * lz
        py = ly
        wz = s * lx + c * lz
        # then the anchor's yaw about world Z.
        yaw = getattr(self, "_incline_yaw", 0.0)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return self._anchor_world_pos + np.array([cy * px - sy * py, sy * px + cy * py, wz])

    @staticmethod
    def _set_cube_friction(sim, cube_name, mu):
        """Write the static-friction column of every geom in *cube_name*."""
        try:
            body_id = sim.model.body_name2id(f"{cube_name}_main")
        except Exception:
            return
        for i in range(sim.model.ngeom):
            if sim.model.geom_bodyid[i] == body_id:
                fr = sim.model.geom_friction[i].copy()
                fr[0] = mu
                sim.model.geom_friction[i] = fr

    def _set_object_color(self, name, rgba):
        """Recolor every geom belonging to the named dynamic object."""
        sim = self._env.sim
        obj = self._env.get_object(name)
        if obj is None:
            return
        rgba = np.asarray(rgba, dtype=float)
        geom_names = set(getattr(obj, "contact_geoms", []) or [])
        geom_names.update(getattr(obj, "visual_geoms", []) or [])
        for gname in geom_names:
            try:
                gid = sim.model.geom_name2id(gname)
            except (ValueError, KeyError):
                continue
            sim.model.geom_rgba[gid] = rgba

    # ── Scoring ─────────────────────────────────────────────────────

    def _cube_in_bin(self, sim, cube_name):
        """Whether the named cube is caught: inside the bin and moving with it."""
        try:
            cube_bid = sim.model.body_name2id(f"{cube_name}_main")
            bin_bid = sim.model.body_name2id("bin_root")
        except Exception:
            try:
                bin_bid = sim.model.body_name2id("bin_main")
            except Exception:
                return False
        cpos = sim.data.body_xpos[cube_bid]
        bpos = sim.data.body_xpos[bin_bid]
        # Use the inner footprint as the catch zone (a cube whose centre is
        # within the inner half-extents of the bin is captured).
        dx = abs(cpos[0] - bpos[0])
        dy = abs(cpos[1] - bpos[1])
        if dx > self._bin_inner_half_x or dy > self._bin_inner_half_y:
            return False
        # Cube must be above the bin's floor (dropped in, not below) and
        # not above its rim: its centre at most its own half-height over
        # the rim, so a cube resting on another cube or leaning on a wall
        # counts, one flying over the bin does not.
        half_h = self._cube_half_heights.get(cube_name, self._cube_height)
        if not bpos[2] - 0.01 <= cpos[2] <= bpos[2] + self._bin_wall + self._bin_wall_height + half_h:
            return False
        # ... and moving with the bin, i.e. caught rather than passing through.
        relative = self._body_velocity(cube_bid)[:3] - self._body_velocity(bin_bid)[:3]
        return float(np.linalg.norm(relative)) < self.REST_SPEED

    def _any_cube_fell(self, raw_obs):
        """Did any cube fall below the floor threshold?"""
        sim = self._env.sim
        for cube_name in self._active_cubes:
            try:
                bid = sim.model.body_name2id(f"{cube_name}_main")
            except Exception:
                continue
            if sim.data.body_xpos[bid][2] < self._failure_floor_threshold:
                return True
        return False

    def _grasp_bin_progress(self, raw_obs):
        """Free-standing mode only: an attached bin is always held."""
        return 1.0 if self._gripper_contacts_object("bin") else 0.0

    def _bin_position_progress(self, raw_obs):
        """Bin-to-incline-base shaping, rescaled so the reset pose reads 0."""
        return progress_from_reset(self._bin_position_shaping(), self._bin_position_at_reset)

    def _bin_position_shaping(self):
        sim = self._env.sim
        try:
            bin_bid = sim.model.body_name2id("bin_root")
        except Exception:
            try:
                bin_bid = sim.model.body_name2id("bin_main")
            except Exception:
                return 0.0
        bin_xy = sim.data.body_xpos[bin_bid][:2]
        base_xy = self._anchor_world_pos[:2]
        dist = float(np.linalg.norm(bin_xy - base_xy))
        return float(1.0 - np.tanh(4.0 * dist))

    def _n_caught(self, sim):
        return sum(1 for cube_name in self._active_cubes if self._cube_in_bin(sim, cube_name))

    def _all_cubes_caught(self):
        return self._n_caught(self._env.sim) >= len(self._active_cubes)

    def _compute_task_score(self, raw_obs):
        # Latched here, not only in _check_termination (which runs after
        # scoring), so the step a cube falls on already scores 0.
        self._fallen = self._fallen or self._any_cube_fell(raw_obs)
        if self._fallen:
            self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
            return 0.0
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._any_cube_fell(raw_obs):
            self._fallen = True
            return True
        # Full success: every cube in the bin.
        score = self._compute_task_score(raw_obs)
        if score >= self._success_score_threshold:
            return True
        return False


def _mat_to_quat(R):
    """Convert a 3×3 rotation matrix to a (w, x, y, z) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


# Per-cube colours — distinct so the operator can track which cube slid first.
_CUBE_COLORS = [
    [0.90, 0.25, 0.25, 1.0],  # red
    [0.25, 0.55, 0.90, 1.0],  # blue
    [0.95, 0.75, 0.15, 1.0],  # yellow
    [0.40, 0.80, 0.35, 1.0],  # green
    [0.70, 0.30, 0.85, 1.0],  # purple
]
