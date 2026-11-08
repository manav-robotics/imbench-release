"""Tool-Use Retrieval — use a hook tool to drag a distant cube into reach, then grasp it.

Task ID: ``ib-p4-tool-retrieve-v1``
"""

import numpy as np

from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class ToolRetrieveV1(SingleArmBase):
    TASK_ID = "ib-p4-tool-retrieve-v1"
    DESCRIPTION = (
        "Single arm: grasp a hook tool, use it to drag an out-of-reach "
        "cube into the workspace, then grasp and lift the cube."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._lift_height = task_cfg.get("lift_height", 0.05)
        self._cube_size = task_cfg.get("cube_size", 0.02)
        self._cube_mass = task_cfg.get("cube_mass", 0.05)
        self._hook_length = task_cfg.get("hook_length", 0.4)
        self._hook_width = task_cfg.get("hook_width", 0.03)
        self._hook_tip_length = task_cfg.get("hook_tip_length", 0.08)
        # Approximate Panda useful-reach boundary in the table frame (see
        # the reset() comment on cube spawn range) — the cube must be
        # dragged past this x threshold to count as "in reach".
        self._reach_threshold_x = task_cfg.get("reach_threshold_x", 0.27)
        self._cube_lifted = False
        self._cube_spawn_x = None

        self._staged_reward = StagedReward(
            stages=[
                Stage("grasp_tool", 0.20,
                      lambda t, o: 1.0 if t._gripper_contacts_object("hook_tool") else 0.0),
                Stage("drag_cube_into_reach", 0.30, lambda t, o: t._drag_progress(o)),
                Stage("grasp_cube", 0.20,
                      lambda t, o: 1.0 if t._gripper_contacts_object("cube") else 0.0),
                Stage("lift_cube", 0.30, lambda t, o: t._lift_progress(o)),
            ],
            success_fn=lambda t, o: t._lift_progress(o) >= 1.0,
        )

    def _make_objects(self):
        hl = self._hook_length   # shaft length
        hw = self._hook_width    # half cross-section
        tip = self._hook_tip_length  # tip length
        color = [0.4, 0.4, 0.4, 1.0]

        return [
            # Cube (target) — placed out of direct reach in reset
            {"type": "box", "name": "cube",
             "size": [self._cube_size] * 3,
             "rgba": [0.8, 0.2, 0.2, 1.0],
             "density": self._cube_mass / (8 * self._cube_size ** 3),
             "friction": [0.8, 0.005, 0.0001],
             "joints": "default"},
            # L-shaped hook tool (composite: shaft + perpendicular tip)
            {"type": "composite", "name": "hook_tool",
             "rgba": color,
             "density": 500,
             "friction": [0.6, 0.005, 0.0001],
             "joints": "default",
             "sub_objects": [
                 # Shaft (long part)
                 {"type": "box", "name": "shaft",
                  "size": [hl / 2, hw, hw],
                  "rgba": color, "density": 500,
                  "friction": [0.6, 0.005, 0.0001]},
                 # Tip (short perpendicular piece at one end)
                 {"type": "box", "name": "tip",
                  "size": [hw, tip / 2, hw],
                  "rgba": color, "density": 500,
                  "friction": [0.6, 0.005, 0.0001]},
             ],
             "sub_locations": [
                 [0.0, 0.0, 0.0],           # shaft centered at origin
                 [hl / 2, tip / 2, 0.0],     # tip at +x end, offset in y
             ],
             "total_size": [hl / 2 + hw, tip / 2 + hw, hw],
             },
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._cube_lifted = False

        # Cube: beyond the Panda's reach (base at x=-0.41, useful reach ~x=0.27)
        # but fully on the table (top spans x∈[-0.3, 0.7], y∈[-0.25, 0.25]).
        cube_x = self._rng.uniform(0.50, 0.62)
        cube_y = self._rng.uniform(-0.12, 0.12)
        cube_z = self._table_height + self._cube_size + 0.005
        self._place_object("cube", [cube_x, cube_y, cube_z])
        self._cube_spawn_x = cube_x
        self._staged_reward.reset()

        # Hook: well within reach. The shaft's half-length is
        # hook_length / 2 (0.15 m) in local x, so keep center near the robot
        # side of the table to stay on-table and away from the cube.
        hook_x = self._rng.uniform(-0.07, 0.05)
        hook_y = self._rng.uniform(-0.12, 0.12)
        hook_z = self._table_height + self._hook_width + 0.005
        self._place_object("hook_tool", [hook_x, hook_y, hook_z])

        if self._rand_object_color:
            self._set_object_color("cube", self._random_object_color())
            self._set_object_color("hook_tool", self._random_object_color())

        self._env.sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    def _set_object_color(self, name, rgba):
        # Works for simple (BoxObject) and composite objects: recolor every
        # geom the object declares, so sub-bodies of composites are covered.
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

    def _drag_progress(self, raw_obs):
        if "cube_pos" not in raw_obs or self._cube_spawn_x is None:
            return 0.0
        cube_x = raw_obs["cube_pos"][0]
        total = self._cube_spawn_x - self._reach_threshold_x
        if total <= 0:
            return 1.0 if cube_x <= self._reach_threshold_x else 0.0
        return float(np.clip((self._cube_spawn_x - cube_x) / total, 0.0, 1.0))

    def _lift_progress(self, raw_obs):
        # Gated on a grasp and measured from the cube's resting height, so the
        # sticky stage never keeps the spawn drop or a flicked cube.
        if "cube_pos" not in raw_obs or not self._gripper_contacts_object("cube"):
            return 0.0
        lift = raw_obs["cube_pos"][2] - (self._table_height + self._cube_size)
        return float(np.clip(lift / self._lift_height, 0.0, 1.0))

    def _compute_task_score(self, raw_obs):
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        if self._stage_debug.get("_success"):
            self._cube_lifted = True
        return score

    def _check_termination(self, raw_obs):
        if self._cube_lifted:
            return True
        if self._object_fell_off_table(raw_obs):
            return True
        return self._object_fell_off_table(raw_obs, pos_key="hook_tool_pos")
