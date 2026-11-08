"""Occluder Push — push plates aside to find the red cube and place it on the green patch.

Task ID: ``ib-p5-occluder-push-v2``

Same scene as v1, but every cube is hidden under a stack of two or three
plates, drawn per episode. The model always holds three layers per stack;
the unused layer is parked off-scene.
"""

import numpy as np
from imbench.tasks.prehensile.contacts import (
    body_geom_ids, geoms_touch, object_geom_ids, robot_contacts,
)
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class OccluderPushV2(SingleArmBase):
    TASK_ID = "ib-p5-occluder-push-v2"
    DESCRIPTION = (
        "Single arm: several lightweight plates each hide a coloured "
        "cube. Push or knock the plates aside to find the red cube, "
        "then pick it up and place it on the green patch. The other "
        "cubes are distractors."
    )

    #: Gap between neighbouring lids (m) and the y-jitter of each cube in
    #: its slot; the jitter is small enough that lids never touch.
    _LID_GAP = 0.03
    _SLOT_Y_JITTER = 0.005
    #: Max linear speed (m/s) of a cube that counts as resting on the patch.
    _REST_SPEED = 0.05

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._patch_pos = np.array(task_cfg.get("patch_pos", [-0.10, 0.0]))
        self._patch_tolerance = task_cfg.get("placement_threshold", 0.04)
        self._cube_size = task_cfg.get("cube_size", 0.02)
        self._lid_thickness = task_cfg.get("lid_thickness", 0.005)
        self._lid_half_size = task_cfg.get("lid_half_size", 0.06)
        self._num_distractors = task_cfg.get("num_distractors", 2)
        self._max_lid_layers = 3  # layers are randomised to 2 or 3 at each reset
        self._cube_placed = False
        self._cube_spawn_xy = None
        self._num_lid_layers = self._max_lid_layers
        self._parked = set()

        self._staged_reward = StagedReward(
            stages=[
                Stage("reveal_target", 0.25, lambda t, o: t._reveal_target_progress(o)),
                Stage("grasp_target_cube", 0.25,
                      lambda t, o: 1.0 if t._gripper_contacts_object("cube") else 0.0),
                Stage("transport_toward_patch", 0.20, lambda t, o: t._transport_progress(o)),
                Stage("place_on_patch", 0.30, lambda t, o: t._place_on_patch_progress(o)),
            ],
            success_fn=lambda t, o: t._place_on_patch_progress(o) >= 1.0,
        )

    def _make_objects(self):
        """Red target cube + distractor cubes + one lid per layer per cube."""
        objects = [
            # Target (red)
            {"type": "box", "name": "cube",
             "size": [self._cube_size] * 3,
             "rgba": [0.8, 0.2, 0.2, 1.0],
             "density": 100, "friction": [0.8, 0.005, 0.0001],
             "joints": "default"},
        ]
        # Lids for the target cube (allocate max layers; extras are parked off-scene)
        for k in range(self._max_lid_layers):
            objects.append({
                "type": "box",
                "name": f"lid_layer_{k}",
                "size": [self._lid_half_size, self._lid_half_size,
                         self._lid_thickness],
                "rgba": [0.3, 0.3, 0.3, 1.0],
                "density": 200, "friction": [0.4, 0.005, 0.0001],
                "joints": "default",
            })
        # Distractor cubes + their lids
        distractor_colors = [
            [0.2, 0.5, 0.9, 1.0],   # blue
            [0.9, 0.7, 0.1, 1.0],   # yellow
            [0.6, 0.2, 0.8, 1.0],   # purple
            [1.0, 0.5, 0.1, 1.0],   # orange
        ]
        for i in range(self._num_distractors):
            objects.append({
                "type": "box",
                "name": f"distractor_{i}",
                "size": [self._cube_size] * 3,
                "rgba": distractor_colors[i % len(distractor_colors)],
                "density": 100, "friction": [0.8, 0.005, 0.0001],
                "joints": "default",
            })
            for k in range(self._max_lid_layers):
                objects.append({
                    "type": "box",
                    "name": f"lid_{i}_layer_{k}",
                    "size": [self._lid_half_size, self._lid_half_size,
                             self._lid_thickness],
                    "rgba": [0.3, 0.3, 0.3, 1.0],
                    "density": 200, "friction": [0.4, 0.005, 0.0001],
                    "joints": "default",
                })
        return objects

    def _make_static_objects(self):
        return [
            # Green patch — placement target (flat visual marker)
            {"type": "box", "name": "goal_patch",
             "size": [0.05, 0.05, 0.001],
             "rgba": [0.2, 0.7, 0.2, 0.5],
             "density": 1000, "friction": [0.5, 0.005, 0.0001],
             "joints": None},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._cube_placed = False
        self._staged_reward.reset()

        # Randomly choose 2 or 3 layers for this episode
        self._num_lid_layers = int(self._rng.choice([2, 3]))

        total_cubes = 1 + self._num_distractors  # target + distractors

        # Generate shuffled positions for all hidden cubes
        positions = self._generate_cube_positions(total_cubes)
        # Randomly assign slots; index 0 = red target cube
        slot_order = list(range(total_cubes))
        self._rng.shuffle(slot_order)
        target_slot = slot_order[0]

        # Place target cube + stacked lids
        cx, cy = positions[target_slot]
        cube_z = self._table_height + self._cube_size + 0.005
        self._place_object("cube", [cx, cy, cube_z])
        self._cube_spawn_xy = np.array([cx, cy])
        for k in range(self._num_lid_layers):
            lid_z = cube_z + self._cube_size + (2 * k + 1) * self._lid_thickness + 0.001
            self._place_object(f"lid_layer_{k}", [cx, cy, lid_z])

        # Place distractor cubes + their stacked lids
        for i in range(self._num_distractors):
            slot = slot_order[1 + i]
            dx, dy = positions[slot]
            self._place_object(f"distractor_{i}", [dx, dy, cube_z])
            for k in range(self._num_lid_layers):
                lid_z_d = cube_z + self._cube_size + (2 * k + 1) * self._lid_thickness + 0.001
                self._place_object(f"lid_{i}_layer_{k}", [dx, dy, lid_z_d])

        # Park the lid layers this episode does not use.
        unused = range(self._num_lid_layers, self._max_lid_layers)
        self._parked = {f"lid_layer_{k}" for k in unused} | {
            f"lid_{i}_layer_{k}" for i in range(self._num_distractors) for k in unused
        }
        self._park_objects(self._parked)

        # Place the green patch
        self._place_static("goal_patch", [
            self._patch_pos[0], self._patch_pos[1],
            self._table_height + 0.001,
        ])

        self._env.sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    def _parked_objects(self):
        return self._parked

    def _generate_cube_positions(self, n):
        """Return *n* (x, y) cube positions whose lids cannot overlap.

        Cubes sit in *n* y-slots one lid width plus ``_LID_GAP`` apart, each
        jittered slightly in y and freely in x; the caller shuffles which
        cube takes which slot.
        """
        pitch = 2 * self._lid_half_size + self._LID_GAP
        half_span = (n - 1) / 2 * pitch + self._SLOT_Y_JITTER + self._lid_half_size
        if half_span > self._table_full_size[1] / 2:
            raise ValueError(
                f"{n} lids of half-size {self._lid_half_size} do not fit side by "
                f"side on a table {self._table_full_size[1]} m wide"
            )
        positions = []
        for k in range(n):
            x = 0.12 + self._rng.uniform(-0.07, 0.07)
            y = (k - (n - 1) / 2) * pitch + self._rng.uniform(
                -self._SLOT_Y_JITTER, self._SLOT_Y_JITTER)
            positions.append((x, y))
        return positions

    def _reveal_target_progress(self, raw_obs):
        """Fraction of stacked lid layers currently displaced off the target
        cube; complete only once *every* active layer has been pushed clear."""
        if "cube_pos" not in raw_obs:
            return 0.0
        sim = self._env.sim
        cube_xy = raw_obs["cube_pos"][:2]
        fractions = []
        for k in range(self._num_lid_layers):
            try:
                lid_id = sim.model.body_name2id(f"lid_layer_{k}_main")
                lid_pos = sim.data.body_xpos[lid_id]
            except Exception:
                continue
            dist = float(np.linalg.norm(lid_pos[:2] - cube_xy))
            fractions.append(float(np.clip(dist / self._lid_half_size, 0.0, 1.0)))
        if not fractions:
            return 0.0
        return min(fractions)

    def _transport_progress(self, raw_obs):
        if "cube_pos" not in raw_obs or self._cube_spawn_xy is None:
            return 0.0
        cube_xy = raw_obs["cube_pos"][:2]
        dist = float(np.linalg.norm(cube_xy - self._patch_pos))
        init_dist = float(np.linalg.norm(self._cube_spawn_xy - self._patch_pos))
        if init_dist <= 1e-6:
            return 1.0
        return float(np.clip((init_dist - dist) / init_dist, 0.0, 1.0))

    def _cube_speed(self):
        """Linear speed (m/s) of the red cube's free joint."""
        sim = self._env.sim
        dof = sim.model.jnt_dofadr[sim.model.body_jntadr[sim.model.body_name2id("cube_main")]]
        return float(np.linalg.norm(sim.data.qvel[dof:dof + 3]))

    def _place_on_patch_progress(self, raw_obs):
        """1.0 iff the red cube is at rest on the patch (or the table inside the
        placement band) and untouched by the robot, else 0.0."""
        if "cube_pos" not in raw_obs:
            return 0.0
        cube_xy = raw_obs["cube_pos"][:2]
        cube_z = raw_obs["cube_pos"][2]
        dist_to_patch = np.linalg.norm(cube_xy - self._patch_pos)
        if not (dist_to_patch < self._patch_tolerance
                and cube_z < self._table_height + 0.05
                and self._cube_speed() < self._REST_SPEED):
            return 0.0
        sim = self._env.sim
        cube_geoms = object_geom_ids(self._env, "cube")
        support = (body_geom_ids(sim, "goal_patch_main")
                   | {sim.model.geom_name2id("table_collision")})
        if robot_contacts(sim, cube_geoms) or not geoms_touch(sim, cube_geoms, support):
            return 0.0
        return 1.0

    def _compute_task_score(self, raw_obs):
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        if self._stage_debug.get("_success"):
            self._cube_placed = True
        return score

    def _check_termination(self, raw_obs):
        if self._cube_placed:
            return True
        return self._object_fell_off_table(raw_obs)