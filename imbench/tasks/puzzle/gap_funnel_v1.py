"""Gap Funnel — push a sphere through the only opening in a barrier strip.

Task ID: ``ib-p3-gap-funnel-v1``

Scene:
- A medium sphere sits on the robot's side of the table.
- A static **barrier strip** runs across the table (along world +Y) at a
  fixed X. The strip is built from two box segments separated by a single
  narrow gap — every other point along Y is blocked. The gap Y-coordinate
  is randomized per episode.
- A green disc (goal zone) lies on the far side of the barrier.

The robot must push/guide the sphere through the gap into the goal zone.
Success requires the sphere to be on the table (not airborne) inside the
goal zone for ``goal_hold_steps`` consecutive steps; it may still be rolling.
"""

import numpy as np
from imbench.tasks.prehensile.contacts import body_geom_ids, geoms_touch, object_geom_ids
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward, progress_from_reset


class GapFunnelV1(SingleArmBase):
    """Gap funnel; the ball's linear velocity is doubled after every step it touches the gripper."""

    TASK_ID = "ib-p3-gap-funnel-v1"
    DESCRIPTION = (
        "Puzzle: a medium sphere sits on the robot's side of the table, "
        "a barrier strip blocks the far side except for a single narrow "
        "gap, and a green goal zone waits beyond. Push the sphere "
        "through the gap into the goal zone."
    )

    # ── Config ──────────────────────────────────────────────────────

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})

        # Sphere geometry — "medium" ball (~6 cm diameter).
        self._ball_radius = float(task_cfg.get("ball_radius", 0.03))
        self._ball_density = float(task_cfg.get("ball_density", 400.0))

        # Barrier strip — two collinear box segments along world +Y with
        # a single gap between them. The strip sits at a fixed X between
        # the sphere's spawn region (robot side) and the goal zone (far).
        self._barrier_x = float(task_cfg.get("barrier_x", 0.10))
        self._barrier_half_thickness = float(
            task_cfg.get("barrier_half_thickness", 0.012)
        )
        self._barrier_half_height = float(
            task_cfg.get("barrier_half_height", 0.03)
        )
        # Full span of the strip in Y (tip-to-tip), centred at y=0.
        self._barrier_full_length = float(
            task_cfg.get("barrier_full_length", 0.50)
        )
        # Half-width of the opening. Needs to exceed the ball radius plus
        # a small margin so the sphere can actually roll through.
        self._gap_half_width = float(task_cfg.get("gap_half_width", 0.05))
        # Gap centre Y jitter — per-episode, sampled uniformly in this range.
        self._gap_y_range = task_cfg.get("gap_y_range", [-0.10, 0.10])

        # Goal zone — a flat green rectangle that covers everything on
        # the far side of the barrier. Defined by its X extent (from the
        # barrier's back face out to some far X) and its Y extent (full
        # strip width by default). Rendered as a thin green box.
        self._goal_x_far = float(task_cfg.get("goal_x_far", 0.35))
        # Y half-extent of the zone — defaults to the barrier's half-length
        # so the zone spans the same Y range the strip does.
        self._goal_half_y = float(task_cfg.get(
            "goal_half_y", self._barrier_full_length / 2.0,
        ))
        # Consecutive steps the ball must be on the table inside the zone.
        self._goal_hold_steps = int(task_cfg.get("goal_hold_steps", 10))
        # Max deviation (m) of the ball centre from table_height + radius
        # for the ball to count as on the table.
        self._rest_z_tol = 0.005

        # Sphere spawn region — a rectangle in the robot base's frame
        # (world X/Y minus the base's world X/Y). Defaults put the ball
        # 0.2–0.4 m in front of the arm and within ±0.2 m laterally,
        # which is well inside the Panda's reachable workspace.
        spawn = task_cfg.get("spawn_region", {})
        self._spawn_x_range = spawn.get("x_range", [0.20, 0.40])
        self._spawn_y_range = spawn.get("y_range", [-0.20, 0.20])

        # Populated at reset time.
        self._gap_center_y = 0.0
        self._goal_x_min = 0.0
        self._goal_x_max = 0.0
        self._goal_y_min = 0.0
        self._goal_y_max = 0.0
        self._spawn_xy = None
        self._gap_alignment_at_reset = 0.0
        self._goal_hold = 0
        self._reached = False
        self._failed_off_table = False

        # Dense reward as named, independently-sticky stages: progress
        # toward the barrier, alignment with the gap, crossing the barrier,
        # and proximity to the goal zone.
        # x_progress and gap_alignment are measured from the spawn pose.
        self._staged_reward = StagedReward(
            stages=[
                Stage("x_progress_to_barrier", 0.4, lambda t, o: t._x_progress_to_barrier(o)),
                Stage("gap_alignment", 0.2, lambda t, o: t._gap_alignment(o)),
                Stage("barrier_crossed", 0.2, lambda t, o: t._barrier_crossed(o)),
                Stage("zone_proximity", 0.2, lambda t, o: t._zone_proximity(o)),
            ],
            success_fn=lambda t, o: t._reached,
        )

    # ── Object factories ────────────────────────────────────────────

    def _make_objects(self):
        vol = (4.0 / 3.0) * np.pi * self._ball_radius ** 3
        return [
            {
                "type": "ball",
                "name": "cube",  # first dynamic obj aliased as "cube_*"
                "size": [self._ball_radius],
                "rgba": [0.85, 0.35, 0.20, 1.0],
                "density": self._ball_density,
                "friction": [0.5, 0.005, 0.0001],
                "joints": "default",
            },
        ]

    def _make_static_objects(self):
        # Two wall segments flanking the gap — their lengths are updated
        # per reset so the gap can move along Y. Pre-size them here with
        # the maximum half-length they'll ever need (the full half-length
        # of the barrier) so the sim has enough geom budget.
        max_half = self._barrier_full_length / 2.0
        wall_rgba = [0.55, 0.45, 0.40, 1.0]
        wall_density = 5000.0
        wall_friction = [0.4, 0.005, 0.0001]

        return [
            {
                "type": "box",
                "name": "wall_neg",
                "size": [self._barrier_half_thickness, max_half,
                         self._barrier_half_height],
                "rgba": wall_rgba,
                "density": wall_density,
                "friction": wall_friction,
                "joints": None,
            },
            {
                "type": "box",
                "name": "wall_pos",
                "size": [self._barrier_half_thickness, max_half,
                         self._barrier_half_height],
                "rgba": wall_rgba,
                "density": wall_density,
                "friction": wall_friction,
                "joints": None,
            },
            {
                # Thin flat box drawn on the table as the goal rectangle.
                # Sized per reset so its footprint matches the current
                # barrier position / goal X extent.
                "type": "box",
                "name": "goal_marker",
                "size": [0.1, 0.1, 0.001],  # placeholder — resized in reset()
                "rgba": [0.0, 0.8, 0.0, 0.4],
                "density": 1.0,
                "friction": [0.1, 0.005, 0.0001],
                "joints": None,
            },
        ]

    # ── Geom resize helper ──────────────────────────────────────────

    @staticmethod
    def _resize_wall_segment(sim, name, half_len_y, half_thickness,
                             half_height):
        """Resize every geom under ``<name>_main`` to the given half-extents."""
        body_id = sim.model.body_name2id(f"{name}_main")
        for g_idx in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g_idx] == body_id:
                sim.model.geom_size[g_idx] = [
                    half_thickness, half_len_y, half_height,
                ]

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._staged_reward.reset()
        sim = self._env.sim
        # Contact priority makes the ball's own friction (and condim) govern its
        # contacts; MuJoCo otherwise takes the max with the table and fingers,
        # and the pushed ball jams and sinks into the table instead of rolling.
        for gname in self._env.get_object("cube").contact_geoms:
            sim.model.geom_priority[sim.model.geom_name2id(gname)] = 1
        self._goal_hold = 0
        self._reached = False
        self._failed_off_table = False

        # ── Place the barrier strip with a single gap ────────────────
        # Strip spans y ∈ [-L/2, L/2] along the line x = barrier_x.
        # Gap centre y ∈ gap_y_range, clamped so both segments stay on
        # the strip (i.e. each segment has length ≥ 0). The sphere ought
        # to actually fit past the gap, so we also enforce a minimum gap
        # half-width of ball_radius + 5 mm.
        L = self._barrier_full_length
        g_half = max(self._gap_half_width, self._ball_radius + 0.005)
        max_gap_center = max(0.0, L / 2.0 - g_half - 0.005)
        gy_lo = max(self._gap_y_range[0], -max_gap_center)
        gy_hi = min(self._gap_y_range[1], max_gap_center)
        if gy_lo > gy_hi:
            self._gap_center_y = 0.0
        else:
            self._gap_center_y = float(self._rng.uniform(gy_lo, gy_hi))

        # Wall segments: neg runs from -L/2 to gap_lo, pos from gap_hi to +L/2.
        gap_lo = self._gap_center_y - g_half
        gap_hi = self._gap_center_y + g_half
        neg_half_len = max((gap_lo - (-L / 2.0)) / 2.0, 1e-4)
        neg_center_y = (-L / 2.0 + gap_lo) / 2.0
        pos_half_len = max(((L / 2.0) - gap_hi) / 2.0, 1e-4)
        pos_center_y = (gap_hi + L / 2.0) / 2.0

        self._resize_wall_segment(
            sim, "wall_neg", neg_half_len,
            self._barrier_half_thickness, self._barrier_half_height,
        )
        self._resize_wall_segment(
            sim, "wall_pos", pos_half_len,
            self._barrier_half_thickness, self._barrier_half_height,
        )

        wall_z = self._table_height + self._barrier_half_height
        self._place_static(
            "wall_neg", [self._barrier_x, neg_center_y, wall_z],
        )
        self._place_static(
            "wall_pos", [self._barrier_x, pos_center_y, wall_z],
        )

        # ── Place the rectangular goal zone on the far side of the barrier ──
        # The zone covers X ∈ [barrier_back_face, goal_x_far] and
        # Y ∈ [-goal_half_y, goal_half_y]. Resize the goal_marker box so
        # its footprint matches, then centre it accordingly.
        barrier_back = self._barrier_x + self._barrier_half_thickness
        self._goal_x_min = barrier_back
        self._goal_x_max = self._goal_x_far
        self._goal_y_min = -self._goal_half_y
        self._goal_y_max = self._goal_half_y

        goal_half_x = max((self._goal_x_max - self._goal_x_min) / 2.0, 1e-4)
        goal_center_x = (self._goal_x_min + self._goal_x_max) / 2.0
        goal_center_y = (self._goal_y_min + self._goal_y_max) / 2.0

        goal_body_id = sim.model.body_name2id("goal_marker_main")
        for g_idx in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g_idx] == goal_body_id:
                sim.model.geom_size[g_idx] = [
                    goal_half_x, self._goal_half_y, 0.001,
                ]
                # Visual-only: the goal is a marker, not a physical slab.
                sim.model.geom_contype[g_idx] = 0
                sim.model.geom_conaffinity[g_idx] = 0
        self._place_static(
            "goal_marker",
            [goal_center_x, goal_center_y, self._table_height + 0.001],
        )

        # ── Place the sphere on the robot side of the barrier ───────
        # Spawn rectangle is specified in the robot base's frame; add
        # the base's world XY so the sphere lands relative to the arm
        # regardless of table offset.
        try:
            base_bid = sim.model.body_name2id("robot0_base")
            base_xy = np.asarray(sim.data.body_xpos[base_bid][:2], dtype=float)
        except Exception:
            base_xy = np.array([-0.81, 0.0])

        dx = float(self._rng.uniform(
            self._spawn_x_range[0], self._spawn_x_range[1],
        ))
        dy = float(self._rng.uniform(
            self._spawn_y_range[0], self._spawn_y_range[1],
        ))
        sx = base_xy[0] + dx
        sy = base_xy[1] + dy

        sz = self._table_height + self._ball_radius + 0.003
        self._place_object("cube", [sx, sy, sz])
        self._set_object_velocity("cube", [0, 0, 0])
        self._spawn_xy = np.array([sx, sy])
        self._gap_alignment_at_reset = self._alignment(sy)

        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ── Step (surface failure flag, amplify gripper impacts) ───────

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Double the ball's linear velocity whenever it's touching the
        # gripper — a simple way to amplify whatever impulse the fingers
        # just delivered.
        cube_obj = self._env.get_object("cube")
        sim = self._env.sim
        # robosuite 1.5 keys grippers by arm; the subtree also covers tactile taxels.
        gripper = self._env.robots[0].gripper["right"]
        if geoms_touch(sim, body_geom_ids(sim, gripper.root_body), object_geom_ids(self._env, "cube")):
            jname = cube_obj.joints[0]
            jid = sim.model.joint_name2id(jname)
            dof_addr = int(sim.model.jnt_dofadr[jid])
            sim.data.qvel[dof_addr:dof_addr + 3] *= 2.0
            sim.forward()

        info["failed_off_table"] = self._failed_off_table
        info["reached"] = self._reached
        return obs, reward, terminated, truncated, info

    # ── Scoring ─────────────────────────────────────────────────────

    def _ball_on_table_in_goal_zone(self, raw_obs):
        """True iff the ball's centre is over the goal zone and the ball is on the table."""
        if "cube_pos" not in raw_obs:
            return False
        x, y, z = raw_obs["cube_pos"]
        in_zone = (self._goal_x_min <= x <= self._goal_x_max
                   and self._goal_y_min <= y <= self._goal_y_max)
        if not in_zone:
            return False
        if abs(z - (self._table_height + self._ball_radius)) > self._rest_z_tol:
            return False
        ball = self._env.get_object("cube")
        return bool(self._env.check_contact(ball.contact_geoms, ["table_collision"]))

    def _update_goal_hold(self, raw_obs):
        """Advance the in-zone hold counter; latch ``_reached`` once it is long enough."""
        if self._ball_on_table_in_goal_zone(raw_obs):
            self._goal_hold += 1
        else:
            self._goal_hold = 0
        if self._goal_hold >= self._goal_hold_steps:
            self._reached = True

    def _x_progress_to_barrier(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0
        ball_x = float(raw_obs["cube_pos"][0])
        spawn_x = float(self._spawn_xy[0]) if self._spawn_xy is not None else ball_x
        return float(np.clip(
            (ball_x - spawn_x) / max(self._goal_x_min - spawn_x, 1e-6),
            0.0, 1.0,
        ))

    def _alignment(self, ball_y):
        """1 with the ball in line with the gap centre, 0 at L/4 or more off it."""
        align_err = abs(ball_y - self._gap_center_y)
        return max(0.0, 1.0 - align_err / max(self._barrier_full_length / 4.0, 1e-6))

    def _gap_alignment(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0
        return progress_from_reset(self._alignment(float(raw_obs["cube_pos"][1])),
                                   self._gap_alignment_at_reset)

    def _barrier_crossed(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0
        return 1.0 if float(raw_obs["cube_pos"][0]) > self._goal_x_min else 0.0

    def _zone_proximity(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0
        ball_xy = raw_obs["cube_pos"][:2]
        if ball_xy[0] <= self._goal_x_min:
            return 0.0
        y_err = max(
            self._goal_y_min - ball_xy[1],
            ball_xy[1] - self._goal_y_max,
            0.0,
        )
        return max(0.0, 1.0 - y_err / 0.25)

    def _compute_task_score(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0

        # Failure: ball fell off the table — stop rewarding progress.
        if self._object_fell_off_table(raw_obs, pos_key="cube_pos"):
            self._failed_off_table = True
            return 0.0

        self._update_goal_hold(raw_obs)
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._reached:
            return True
        return self._object_fell_off_table(raw_obs, pos_key="cube_pos")
