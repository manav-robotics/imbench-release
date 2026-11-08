"""Sheltered Grasp — reach into a 2-story rack from the side to extract a hidden cube.

Task ID: ``ib-p1-sheltered-grasp-v1``
"""

import numpy as np
from imbench.tasks.prehensile.contacts import (
    body_geom_ids, geoms_touch, object_geom_ids, robot_contacts,
)
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class ShelteredGraspV1(SingleArmBase):
    TASK_ID = "ib-p1-sheltered-grasp-v1"
    DESCRIPTION = (
        "Single arm: a cube sits inside a large two-story covered rack whose "
        "open face is accessible from either side, sampled each episode. The robot must navigate its "
        "gripper into the correct rack compartment (top or bottom story, "
        "randomised), grasp the cube, carefully extract it, and place it in "
        "the tray."
    )

    # Rack dimensions (covered_rack_2story.xml)
    _RACK_WIDTH  = 0.32   # x: ±0.16
    _RACK_DEPTH  = 0.20   # y: ±0.10
    _RACK_HEIGHT = 0.30   # z: 0..0.30
    _RACK_BACK_INNER_Y = 0.09                          # local y of the back panel's inner face
    _STORY_FLOOR_Z = {"bottom": 0.01, "top": 0.16}     # local z of each compartment floor

    #: Max linear speed (m/s) of a cube that counts as resting in the tray.
    _REST_SPEED = 0.05

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._tray_pos = np.array(task_cfg.get("tray_pos", [-0.10, 0.0]))
        self._tray_default_pos = self._tray_pos.copy()
        self._tray_tolerance = task_cfg.get("placement_threshold", 0.05)
        self._cube_size = task_cfg.get("cube_size", 0.025)
        self._cube_placed = False

        rack_cfg = task_cfg.get("rack", {})
        self._rack_default_pos = np.array(rack_cfg.get("default_pos", [0.12, 0.22]))
        self._rack_default_yaw = rack_cfg.get("default_yaw", 0.0)

        rand_cfg = self.cfg.get("randomization", {})
        side_cfg = rand_cfg.get("rack_side", {})
        self._randomize_rack_side = side_cfg.get("enabled", False)
        self._rack_side_choices = tuple(side_cfg.get("choices", ["left", "right"]))
        if not self._rack_side_choices or set(self._rack_side_choices) - {"left", "right"}:
            raise ValueError("rack_side.choices must contain left and/or right")
        self._layout_reflected = False
        rack_pos_cfg = rand_cfg.get("rack_position", {})
        self._rack_randomize_pos = rack_pos_cfg.get("enabled", True)
        self._rack_pos_x_range = rack_pos_cfg.get("x_range", [-0.05, 0.20])
        self._rack_pos_y_range = rack_pos_cfg.get("y_range", [0.15, 0.30])

        rack_ori_cfg = rand_cfg.get("rack_orientation", {})
        self._rack_randomize_ori = rack_ori_cfg.get("enabled", True)
        # Continuous yaw range — must stay near 0 so the open face (-y local)
        # always roughly faces the robot (at world -y).  ±π/2 or π are
        # forbidden because they either put a solid wall or the rack back
        # toward the robot.
        self._rack_yaw_range = rack_ori_cfg.get("yaw_range", [-np.pi / 6, np.pi / 6])

        cube_pose_cfg = rand_cfg.get("cube_pose", {})
        self._cube_randomize_pose = cube_pose_cfg.get("enabled", True)
        self._cube_local_x_range = cube_pose_cfg.get("local_x_range", [-0.08, 0.08])
        self._cube_local_y_range = cube_pose_cfg.get("local_y_range", [0.01, 0.035])
        self._cube_default_local_xy = cube_pose_cfg.get("default_local_xy", [0.0, 0.04])

        story_choice_cfg = rand_cfg.get("cube_story", {})
        self._story_randomize = story_choice_cfg.get("enabled", True)
        self._story_choices = story_choice_cfg.get("choices", ["bottom", "top"])
        self._story_weights = story_choice_cfg.get("weights", None)
        self._story_default = story_choice_cfg.get("default", "bottom")

        self._staged_reward = StagedReward(
            stages=[
                Stage("approach_compartment", 0.20, lambda t, o: t._approach_progress(o)),
                Stage("grasp", 0.25, lambda t, o: 1.0 if t._gripper_contacts_object("cube") else 0.0),
                Stage("extract_from_rack", 0.25, lambda t, o: t._extraction_progress(o)),
                Stage("place_in_tray", 0.30, lambda t, o: t._place_progress(o)),
            ],
            success_fn=lambda t, o: t._is_placed_in_tray(o),
        )

    def _make_objects(self):
        return [
            {"type": "box", "name": "cube",
             "size": [self._cube_size, 2*self._cube_size, self._cube_size],
             "rgba": [0.8, 0.2, 0.2, 1.0],
             "density": 100, "friction": [0.8, 0.005, 0.0001],
             "joints": "default"},
        ]

    def _make_static_objects(self):
        return [
            # Large two-story covered rack
            {"type": "extras", "name": "rack",
             "extras_name": "covered_rack_2story",
             "joints": None},
            # Goal tray
            {"type": "box", "name": "tray",
             "size": [0.07, 0.07, 0.001],
             "rgba": [0.2, 0.6, 0.2, 0.5],
             "density": 1000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._cube_placed = False
        self._staged_reward.reset()

        # ── Rack position ─────────────────────────────────────────────
        if self._rack_randomize_pos:
            rack_x = self._rng.uniform(*self._rack_pos_x_range)
            rack_y = self._rng.uniform(*self._rack_pos_y_range)
        else:
            rack_x, rack_y = self._rack_default_pos
        rack_z = self._table_height

        # ── Rack orientation ──────────────────────────────────────────
        if self._rack_randomize_ori:
            yaw = self._rng.uniform(*self._rack_yaw_range)
        else:
            yaw = self._rack_default_yaw

        # Sample once per reset. Reflect the entire canonical layout, not
        # just the visible opening: a right-side rack faces +Y, its cube
        # stays inside the same compartment, and the tray reflects to the
        # opposite side too. The rack asset is symmetric about local X,
        # so yaw pi-yaw realizes the reflected wall/open-face geometry.
        side = str(self._rng.choice(self._rack_side_choices)) if self._randomize_rack_side else None
        self._layout_reflected = side == "right"
        canonical_rack_xy = np.array([rack_x, rack_y])
        canonical_yaw = yaw
        self._tray_pos = self._tray_default_pos.copy()
        if self._layout_reflected:
            rack_y = -rack_y
            yaw = np.pi - yaw
            self._tray_pos[1] *= -1
        self._rack_opening_side = "left" if np.cos(yaw) >= 0 else "right"
        rack_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

        self._place_static("rack", [rack_x, rack_y, rack_z], quat=rack_quat)
        self._rack_pos = np.array([rack_x, rack_y])
        self._rack_yaw = yaw

        # ── Choose story (top or bottom) and place cube ───────────────
        if self._story_randomize:
            story = self._rng.choice(self._story_choices, p=self._story_weights)
        else:
            story = self._story_default
        self._cube_story = story

        # Local offset inside the rack compartment
        if self._cube_randomize_pose:
            local_x = self._rng.uniform(*self._cube_local_x_range)
            local_y = self._rng.uniform(*self._cube_local_y_range)
        else:
            local_x, local_y = self._cube_default_local_xy
        # Keep the (world-axis-aligned) cube clear of the back panel, and
        # rest it on the compartment floor.
        half_x, half_y = self._cube_size, 2 * self._cube_size
        depth_extent = abs(np.cos(yaw)) * half_y + abs(np.sin(yaw)) * half_x
        local_y = min(local_y, self._RACK_BACK_INNER_Y - depth_extent - 0.002)
        local_z = self._STORY_FLOOR_Z[story] + self._cube_size + 0.001

        cos_y, sin_y = np.cos(canonical_yaw), np.sin(canonical_yaw)
        cube_x = canonical_rack_xy[0] + cos_y * local_x - sin_y * local_y
        cube_y = canonical_rack_xy[1] + sin_y * local_x + cos_y * local_y
        if self._layout_reflected:
            cube_y = -cube_y
        cube_z = rack_z + local_z
        self._place_object("cube", [cube_x, cube_y, cube_z])

        # Baselines for the extract/place staged-progress metrics —
        # both measure shrinking distance from these as-spawned values.
        self._extract_start_margin = self._rack_containment_margin([cube_x, cube_y])
        self._tray_dist_start = float(np.linalg.norm(
            np.array([cube_x, cube_y]) - self._tray_pos
        ))

        # ── Goal tray ─────────────────────────────────────────────────
        self._place_static("tray", [
            self._tray_pos[0], self._tray_pos[1],
            self._table_height + 0.001,
        ])

        self._env.sim.forward()
        obs = self._rebuild_obs()
        self._mirror_front_cam(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._mirror_front_cam(obs)
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _mirror_front_cam(obs):
        for key in ("rgb_exo_front_cam", "depth_exo_front_cam"):
            if key in obs:
                obs[key] = np.ascontiguousarray(obs[key][:, ::-1])

    # ── Staged-scoring helpers ───────────────────────────────────────

    def _rack_local_xy(self, cube_xy):
        """World cube XY -> rack-local XY, using the rack's placed pose."""
        rel = np.asarray(cube_xy, dtype=float) - self._rack_pos
        cos_y, sin_y = np.cos(self._rack_yaw), np.sin(self._rack_yaw)
        local_x = cos_y * rel[0] + sin_y * rel[1]
        local_y = -sin_y * rel[0] + cos_y * rel[1]
        return local_x, local_y

    def _rack_containment_margin(self, cube_xy):
        """Smallest clearance (m) from the cube's XY to a rack-footprint
        wall, in the rack's local frame. Positive while inside the
        rack's footprint; crosses zero once the cube clears any wall.
        """
        local_x, local_y = self._rack_local_xy(cube_xy)
        half_w = self._RACK_WIDTH / 2.0
        half_d = self._RACK_DEPTH / 2.0
        return min(half_w - abs(local_x), half_d - abs(local_y))

    def _approach_progress(self, raw_obs):
        if "cube_pos" not in raw_obs or "robot0_eef_pos" not in raw_obs:
            return 0.0
        dist = float(np.linalg.norm(raw_obs["robot0_eef_pos"] - raw_obs["cube_pos"]))
        return 1.0 - float(np.tanh(6.0 * dist))

    def _extraction_progress(self, raw_obs):
        """Shrinking-margin progress toward clearing the rack footprint."""
        if "cube_pos" not in raw_obs:
            return 0.0
        margin = self._rack_containment_margin(raw_obs["cube_pos"][:2])
        start = getattr(self, "_extract_start_margin", None)
        if not start or start <= 0:
            return 1.0 if margin <= 0 else 0.0
        return float(np.clip((start - margin) / start, 0.0, 1.0))

    def _place_progress(self, raw_obs):
        """Shrinking-distance progress toward the tray, from the cube's
        as-spawned position (inside the rack) to the tray."""
        if "cube_pos" not in raw_obs:
            return 0.0
        dist = float(np.linalg.norm(raw_obs["cube_pos"][:2] - self._tray_pos))
        start = getattr(self, "_tray_dist_start", None)
        if not start or start <= 0:
            return 0.0
        return float(np.clip((start - dist) / start, 0.0, 1.0))

    def _cube_speed(self):
        """Linear speed (m/s) of the cube's free joint."""
        sim = self._env.sim
        dof = sim.model.jnt_dofadr[sim.model.body_jntadr[sim.model.body_name2id("cube_main")]]
        return float(np.linalg.norm(sim.data.qvel[dof:dof + 3]))

    def _is_placed_in_tray(self, raw_obs):
        """Cube released and at rest on the tray (or the table inside it)."""
        if "cube_pos" not in raw_obs:
            return False
        cube_xy = raw_obs["cube_pos"][:2]
        cube_z = raw_obs["cube_pos"][2]
        dist_to_tray = np.linalg.norm(cube_xy - self._tray_pos)
        placed = (dist_to_tray < self._tray_tolerance
                  and cube_z < self._table_height + 0.06
                  and self._cube_speed() < self._REST_SPEED)
        if placed:
            sim = self._env.sim
            cube_geoms = object_geom_ids(self._env, "cube")
            support = body_geom_ids(sim, "tray_main") | {sim.model.geom_name2id("table_collision")}
            placed = (not robot_contacts(sim, cube_geoms)
                      and geoms_touch(sim, cube_geoms, support))
        if placed:
            self._cube_placed = True
        return bool(placed)

    def _compute_task_score(self, raw_obs):
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._cube_placed:
            return True
        return self._object_fell_off_table(raw_obs)
