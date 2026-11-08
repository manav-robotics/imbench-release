"""Cup Extraction — extract a cuboid from a handled cup and place it on a goal region.

Task ID: ``ib-p4-cup-extract-v1``
"""

import numpy as np

from imbench.tasks.prehensile.contacts import (
    geoms_touch,
    object_geom_ids,
    robot_contacts,
)
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class CupExtractV1(SingleArmBase):
    TASK_ID = "ib-p4-cup-extract-v1"
    DESCRIPTION = (
        "Single arm: a cuboid is trapped inside a handled cup on the table. "
        "Use the handle to manipulate the cup, extract the cuboid, and "
        "place it on a green goal region on the table."
    )

    def _parse_task_config(self):
        self._task_done = False
        self._staged_reward = StagedReward(
            stages=[
                Stage("extract_from_cup", 0.35, lambda t, o: t._extract_from_cup_progress(o)),
                Stage("transport_to_goal_region", 0.35, lambda t, o: t._transport_to_goal_progress(o)),
                Stage("place_resting", 0.30, lambda t, o: t._place_resting_progress(o)),
            ],
            success_fn=lambda t, o: (
                t._extract_from_cup_progress(o) >= 1.0
                and t._transport_to_goal_progress(o) >= 1.0
                and t._place_resting_progress(o) >= 1.0
            ),
        )

    # Open cup mesh dimensions (must match deps/extras/open_cup.stl)
    _CUP_RADIUS = 0.035     # inner radius
    _CUP_HEIGHT = 0.10       # wall height
    _CUP_BOTTOM = 0.004      # bottom thickness
    _CUBE_HALF_XY = 0.02     # cuboid half-size in x and y
    _CUBE_HALF_Z = 0.03      # cuboid half-size in z (taller)
    _GOAL_HALF = 0.08        # goal region half-size (16 cm x 16 cm)
    _REST_SPEED = 0.05       # max linear speed (m/s) of a resting cuboid
    _REST_ANG_SPEED = 0.5    # max angular speed (rad/s) of a resting cuboid

    def _make_objects(self):
        return [
            # Cup — open-top cylinder mesh from deps/extras/
            {"type": "open_cylinder", "name": "cube",
             "joints": "default"},
            # Small object trapped inside
            {"type": "box", "name": "small_object",
             "size": [self._CUBE_HALF_XY, self._CUBE_HALF_XY, self._CUBE_HALF_Z],
             "rgba": [0.9, 0.1, 0.1, 1.0],
             "density": 500,
             "friction": [0.5, 0.005, 0.0001],
             "joints": "default"},
        ]

    def _make_static_objects(self):
        return [
            # Goal region — thin green square on the table
            {"type": "box", "name": "goal_marker",
             "size": [self._GOAL_HALF, self._GOAL_HALF, 0.001],
             "rgba": [0.0, 0.8, 0.0, 0.3],
             "density": 1,
             "friction": [0.1, 0.005, 0.0001],
             "joints": None},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._task_done = False
        self._staged_reward.reset()

        sim = self._env.sim

        # Place open cylinder upright with randomized position (±10 cm).
        cup_x = self._rng.uniform(-0.10, 0.10)
        cup_y = self._rng.uniform(-0.10, 0.10)
        cup_z = self._table_height
        yaw = np.pi  # 180 degrees — handle faces toward the robot
        cup_quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]  # [w, x, y, z]
        self._place_object("cube", [cup_x, cup_y, cup_z], quat=cup_quat)

        # Place cube inside the cup, just above the interior floor.
        obj_z = cup_z + self._CUP_BOTTOM + self._CUBE_HALF_Z + 0.005
        self._place_object("small_object", [cup_x, cup_y, obj_z])

        # Place goal region on the table, away from the cup
        goal_x = self._rng.uniform(-0.15, 0.15)
        goal_y = self._rng.uniform(-0.15, 0.15)
        # Push goal away if it overlaps the cup
        while np.hypot(goal_x - cup_x, goal_y - cup_y) < self._GOAL_HALF + self._CUP_RADIUS + 0.02:
            goal_x = self._rng.uniform(-0.15, 0.15)
            goal_y = self._rng.uniform(-0.15, 0.15)
        self._goal_pos = np.array([goal_x, goal_y])
        self._place_static("goal_marker", [goal_x, goal_y, self._table_height + 0.001])

        # Randomize cube color
        if self._rand_object_color:
            cube_rgba = self._random_cube_color()
            obj_body_id = sim.model.body_name2id("small_object_main")
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == obj_body_id:
                    sim.model.geom_rgba[i] = cube_rgba

        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    def _random_cube_color(self):
        """Sample a random saturated color for the cube."""
        rgb = self._rng.uniform(0.15, 0.95, size=3)
        # boost saturation — pull away from gray
        gray = rgb.mean()
        rgb = np.clip(rgb + (rgb - gray) * 0.4, 0.05, 1.0)
        return np.array([*rgb, 1.0])

    def _small_object_pos(self):
        sim = self._env.sim
        return sim.data.body_xpos[sim.model.body_name2id("small_object_main")]

    def _small_object_at_rest(self):
        """Whether the cuboid's linear and angular speeds are below the rest thresholds."""
        sim = self._env.sim
        dof = sim.model.jnt_dofadr[sim.model.body_jntadr[sim.model.body_name2id("small_object_main")]]
        return (np.linalg.norm(sim.data.qvel[dof:dof + 3]) < self._REST_SPEED
                and np.linalg.norm(sim.data.qvel[dof + 3:dof + 6]) < self._REST_ANG_SPEED)

    def _small_object_in_cup(self):
        """Whether any part of the cuboid is still inside the cup.

        Evaluated in the cup's body frame (origin at the base, +z along the
        cup axis), so it holds however the cup is tilted or tipped over:
        inside means the cuboid's centre is above the base and within the
        inner radius of the axis, and its end nearest the base is below the rim.
        """
        sim = self._env.sim
        cup_id = sim.model.body_name2id("cube_main")
        obj_id = sim.model.body_name2id("small_object_main")
        cup_rot = sim.data.body_xmat[cup_id].reshape(3, 3)
        local = cup_rot.T @ (sim.data.body_xpos[obj_id] - sim.data.body_xpos[cup_id])
        cup_axis_in_obj = sim.data.body_xmat[obj_id].reshape(3, 3).T @ cup_rot[:, 2]
        half_extent = np.abs(cup_axis_in_obj) @ [
            self._CUBE_HALF_XY, self._CUBE_HALF_XY, self._CUBE_HALF_Z]
        return (0.0 <= local[2] and local[2] - half_extent <= self._CUP_HEIGHT
                and np.hypot(local[0], local[1]) < self._CUP_RADIUS)

    def _extract_from_cup_progress(self, raw_obs):
        return 0.0 if self._small_object_in_cup() else 1.0

    def _transport_to_goal_progress(self, raw_obs):
        if self._small_object_in_cup():
            return 0.0
        small_pos = self._small_object_pos()
        in_goal = (abs(small_pos[0] - self._goal_pos[0]) < self._GOAL_HALF and
                   abs(small_pos[1] - self._goal_pos[1]) < self._GOAL_HALF)
        return 1.0 if in_goal else 0.0

    def _place_resting_progress(self, raw_obs):
        """1.0 iff the extracted cuboid is at rest on the goal untouched by the robot, else 0.0."""
        if self._transport_to_goal_progress(raw_obs) < 1.0:
            return 0.0
        resting_z = self._table_height + self._CUBE_HALF_Z
        if (abs(self._small_object_pos()[2] - resting_z) >= 0.01
                or not self._small_object_at_rest()):
            return 0.0
        sim = self._env.sim
        cuboid = object_geom_ids(self._env, "small_object")
        support = (object_geom_ids(self._env, "goal_marker")
                   | {sim.model.geom_name2id("table_collision")})
        if robot_contacts(sim, cuboid) or not geoms_touch(sim, cuboid, support):
            return 0.0
        return 1.0

    def _compute_task_score(self, raw_obs):
        # Success (1.0): cube is outside the cup, inside the goal region,
        # and at rest on the table (z ≈ table + cube_size + small tolerance).
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        if self._stage_debug.get("_success"):
            self._task_done = True
        return score

    def _check_termination(self, raw_obs):
        if self._task_done:
            return True
        return (self._object_fell_off_table(raw_obs)
                or self._object_fell_off_table(raw_obs, pos_key="small_object_pos"))
