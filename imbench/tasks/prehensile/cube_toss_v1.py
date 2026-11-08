"""Cube Toss — pick up a cube and toss it into a distant goal zone.

Task ID: ``ib-p2-cube-toss-v1``

Success: the cube rests on the table inside the goal circle for
``rest_steps`` control steps, and its last robot contact was at least
``min_release_dist`` outside the goal circle (a cube carried or pushed into
the goal does not count).
"""

import numpy as np
from imbench.tasks.prehensile.contacts import object_geom_ids, robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class CubeTossV1(SingleArmBase):
    TASK_ID = "ib-p2-cube-toss-v1"
    DESCRIPTION = (
        "Single arm: pick up a cube and toss it so it lands in a "
        "distant goal zone outside the workspace."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._goal_pos = np.array(task_cfg.get("goal_pos", [0.6, 0.0]))
        self._goal_radius = task_cfg.get("goal_radius", 0.10)
        self._rest_velocity_threshold = task_cfg.get("rest_velocity_threshold", 0.01)
        self._rest_steps = task_cfg.get("rest_steps", 20)
        self._cube_mass = task_cfg.get("cube_mass", 0.05)
        self._cube_size = task_cfg.get("cube_size", 0.02)
        # Minimum xy distance (m) outside the goal circle of the cube's
        # position at its last robot contact.
        self._min_release_dist = task_cfg.get("min_release_dist", 0.15)
        self._release_xy = np.zeros(2)
        # On the tick the gripper releases the cube, multiply the cube's
        # xy linear velocity by this factor. Models a "snap-throw" boost.
        # ~35% of the boost is eaten by lingering finger contacts in
        # the next few physics ticks (verified empirically), so the
        # effective multiplier is roughly 0.65 × this value.
        self._release_xy_boost = task_cfg.get("release_xy_boost", 3.0)
        # The gripper takes a few control ticks to physically open after
        # the open command. Writing to the cube's qvel during those
        # ticks is partially undone by the fingers' contact constraint.
        # We wait this many ticks after the open command before applying
        # the boost so the cube is actually free.
        self._release_boost_delay = int(
            task_cfg.get("release_boost_delay", 5))
        self._at_rest_count = 0
        # Release-edge detector state (reset every episode in reset()).
        self._prev_gripper_cmd = 0.0
        self._release_boost_applied = False
        # Countdown: -1 = no pending boost, ≥0 = ticks remaining until
        # the boost is applied (set when the open-gripper edge is seen).
        self._release_boost_countdown = -1

        rand_cfg = self.cfg.get("randomization", {})
        goal_rand = rand_cfg.get("goal_region", {})
        self._rand_goal_dist = goal_rand.get("distance_range", [0.55, 0.8])
        self._rand_goal_angle = goal_rand.get("angle_range", [-45, 45])
        self._rand_goal = goal_rand.get("enabled", False)
        self._rand_radius = goal_rand.get("radius_range", None)

        self._staged_reward = StagedReward(
            stages=[
                Stage("grasp_and_lift", 0.20, lambda t, o: t._grasp_and_lift_progress(o)),
                Stage("release_toss", 0.20, lambda t, o: 1.0 if t._release_boost_applied else 0.0),
                Stage("land_in_goal", 0.30, lambda t, o: t._land_in_goal_progress(o)),
                Stage("settle_at_rest", 0.30, lambda t, o: t._settle_progress(o)),
            ],
            success_fn=lambda t, o: t._toss_success(o),
        )

    def _make_objects(self):
        return [
            {"type": "box", "name": "cube",
             "size": [self._cube_size] * 3,
             "rgba": [0.2, 0.5, 0.8, 1.0],
             "density": self._cube_mass / (8 * self._cube_size ** 3),
             "friction": [1.0, 0.005, 0.0001],
             "joints": "default"},
        ]

    def _make_static_objects(self):
        # Visual-only circular goal zone marker (very thin cylinder)
        return [
            {"type": "cylinder", "name": "goal_marker",
             "size": [self._goal_radius, 0.001],
             "rgba": [0.0, 0.8, 0.0, 0.3],
             "density": 1, "friction": [0.1, 0.005, 0.0001],
             "joints": None},
        ]

    def _clamp_goal_to_table(self):
        """Shift goal so the full circle stays on the table and outside arm reach."""
        r = self._goal_radius
        tx, ty = self._table_offset[0], self._table_offset[1]
        table_hx = self._table_full_size[0] / 2.0
        table_hy = self._table_full_size[1] / 2.0

        robot_base = np.array(self._robot_offset[:2]) if self._robot_offset else np.zeros(2)
        arm_reach = 0.855
        min_center_dist = arm_reach + r + 0.02  # 2 cm margin

        # ── 1. Clamp Y to table first ─────────────────────────────────
        self._goal_pos[1] = np.clip(
            self._goal_pos[1], ty - table_hy + r, ty + table_hy - r)

        # ── 2. Enforce reach: solve for minimum X ─────────────────────
        # After Y is fixed, the center must satisfy:
        #   (x - robot_x)^2 + (y - robot_y)^2 >= min_center_dist^2
        # Solve for the minimum x that satisfies this.
        dy = self._goal_pos[1] - robot_base[1]
        x_sq = min_center_dist ** 2 - dy ** 2
        if x_sq > 0:
            min_x = robot_base[0] + np.sqrt(x_sq)
        else:
            # dy alone already exceeds min_center_dist, any x is fine
            min_x = -np.inf
        self._goal_pos[0] = max(self._goal_pos[0], min_x)

        # ── 3. Clamp X to table ────────────────────────────────────────
        self._goal_pos[0] = np.clip(
            self._goal_pos[0], tx - table_hx + r, tx + table_hx - r)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Randomize goal region
        if self._rand_goal:
            dist = self._rng.uniform(*self._rand_goal_dist)
            angle = np.radians(self._rng.uniform(*self._rand_goal_angle))
            self._goal_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
            if self._rand_radius is not None:
                self._goal_radius = self._rng.uniform(*self._rand_radius)

        # Ensure the circle stays on the table and outside arm reach
        self._clamp_goal_to_table()

        self._at_rest_count = 0
        self._prev_gripper_cmd = 0.0
        self._release_boost_applied = False
        self._release_boost_countdown = -1
        self._staged_reward.reset()

        # Resize the goal marker to the goal radius and make it visual-only:
        # a cube resting on the collidable disc has a single box-cylinder
        # contact, cannot resist spin about it and never comes to rest.
        sim = self._env.sim
        body_id = sim.model.body_name2id("goal_marker_main")
        geom_start = sim.model.body_geomadr[body_id]
        geom_count = sim.model.body_geomnum[body_id]
        for gi in range(geom_start, geom_start + geom_count):
            sim.model.geom_size[gi] = [self._goal_radius, 0.001, 0.0]
            sim.model.geom_contype[gi] = 0
            sim.model.geom_conaffinity[gi] = 0

        # Position goal marker on the table
        self._place_static("goal_marker", [
            self._goal_pos[0], self._goal_pos[1],
            self._table_height + 0.001,
        ])
        self._env.sim.forward()
        self._release_xy = sim.data.body_xpos[sim.model.body_name2id("cube_main")][:2].copy()
        obs = self._rebuild_obs()
        return obs, info

    def step(self, action):
        # Track previous gripper command so we can detect the release
        # edge (close → open transition). Robosuite's contact-based
        # _check_grasp is unreliable for the Panda + small cube combo,
        # so we use the policy's commanded gripper instead.
        prev_cmd = self._prev_gripper_cmd

        obs, reward, terminated, truncated, info = super().step(action)

        cur_cmd = float(action[-1]) if action is not None and len(action) > 0 else 0.0

        # Heuristic for "was holding": gripper command was closed (>0)
        # AND the cube has been lifted off the table (z significantly
        # above its starting height). The lift check rules out spurious
        # boosts during the initial close-on-air phase.
        sim = self._env.sim
        cube_z = float(sim.data.qpos[sim.model.jnt_qposadr[
            sim.model.body_jntadr[sim.model.body_name2id("cube_main")]] + 2])
        cube_lifted = cube_z > self._table_height + 0.05

        # Falling edge: gripper was commanded closed, now commanded open,
        # cube is currently lifted → schedule the boost a few ticks
        # later (after the fingers physically open and stop dragging
        # the cube via contact constraints).
        release_edge = (
            prev_cmd > 0.0 and cur_cmd <= 0.0
            and cube_lifted
            and not self._release_boost_applied
            and self._release_boost_countdown < 0
            and self._release_xy_boost != 1.0
        )

        if release_edge:
            self._release_boost_countdown = self._release_boost_delay

        # Pending boost: tick down until 0 then scale the cube's xy velocity.
        if self._release_boost_countdown == 0:
            dof = sim.model.jnt_dofadr[sim.model.body_jntadr[sim.model.body_name2id("cube_main")]]
            sim.data.qvel[dof:dof + 2] *= self._release_xy_boost
            self._release_boost_applied = True
            self._release_boost_countdown = -1
        elif self._release_boost_countdown > 0:
            self._release_boost_countdown -= 1

        self._prev_gripper_cmd = cur_cmd
        return obs, reward, terminated, truncated, info

    def _cube_speed(self):
        """Linear speed of the cube's free joint (m/s)."""
        sim = self._env.sim
        dof = sim.model.jnt_dofadr[sim.model.body_jntadr[sim.model.body_name2id("cube_main")]]
        return float(np.linalg.norm(sim.data.qvel[dof:dof + 3]))

    def _cube_on_table(self, raw_obs):
        # Cube must be resting on (or very near) the table surface.  Use a
        # 4 cm window above the nominal table height so a cube resting on any
        # small object on the table still counts, while one still airborne
        # (e.g. mid-toss, > 5 cm up) is correctly rejected.
        return raw_obs["cube_pos"][2] <= self._table_height + 0.04

    def _grasp_and_lift_progress(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0
        cube_lifted = raw_obs["cube_pos"][2] > self._table_height + 0.05
        return 1.0 if (cube_lifted and self._gripper_contacts_object("cube")) else 0.0

    def _released_outside_goal(self):
        """Whether the cube's last robot contact was ``min_release_dist`` or more outside the goal."""
        dist = float(np.linalg.norm(self._release_xy - self._goal_pos))
        return dist - self._goal_radius >= self._min_release_dist

    def _land_in_goal_progress(self, raw_obs):
        if ("cube_pos" not in raw_obs or not self._cube_on_table(raw_obs)
                or not self._released_outside_goal()):
            return 0.0
        dist = float(np.linalg.norm(raw_obs["cube_pos"][:2] - self._goal_pos))
        if dist < self._goal_radius:
            return 1.0
        return float(1.0 - np.tanh(5.0 * (dist - self._goal_radius)))

    def _settle_progress(self, raw_obs):
        if ("cube_pos" not in raw_obs or not self._cube_on_table(raw_obs)
                or not self._released_outside_goal()):
            return 0.0
        dist = float(np.linalg.norm(raw_obs["cube_pos"][:2] - self._goal_pos))
        if dist >= self._goal_radius:
            return 0.0
        return self._at_rest_count / max(1, self._rest_steps)

    def _toss_success(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return False
        dist = float(np.linalg.norm(raw_obs["cube_pos"][:2] - self._goal_pos))
        return (
            dist < self._goal_radius
            and self._cube_on_table(raw_obs)
            and self._at_rest_count >= self._rest_steps
            and self._released_outside_goal()
        )

    def _compute_task_score(self, raw_obs):
        if "cube_pos" not in raw_obs:
            return 0.0

        if self._cube_speed() < self._rest_velocity_threshold:
            self._at_rest_count += 1
        else:
            self._at_rest_count = 0
        if robot_contacts(self._env.sim, object_geom_ids(self._env, "cube")):
            self._release_xy = np.array(raw_obs["cube_pos"][:2], dtype=float)

        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._compute_task_score(raw_obs) >= self._success_score_threshold:
            return True
        return self._object_fell_off_table(raw_obs)
