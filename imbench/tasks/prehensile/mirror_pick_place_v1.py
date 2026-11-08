"""Mirror Pick Place — pick up a cube off the table to a target lift height.

Task ID: ``ib-m-mirror-pick-place-v1``
"""

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class MirrorPickPlaceV1(SingleArmBase):
    TASK_ID = "ib-m-mirror-pick-place-v1"
    DESCRIPTION = (
        "Single arm: pick up a cube lying on the table and lift it to a "
        "target height. Cube position and color are randomized each reset."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._cube_size = task_cfg.get("cube_size", 0.02)
        self._cube_mass = task_cfg.get("cube_mass", 0.05)
        # Lift of the cube centre above its resting height.
        self._lift_height = task_cfg.get("lift_height", 0.08)
        self._mirror_directions = tuple(task_cfg.get(
            "mirror_directions", ["horizontal", "vertical", "both"]))
        if not self._mirror_directions or set(self._mirror_directions) - {"horizontal", "vertical", "both"}:
            raise ValueError("mirror_directions must contain horizontal, vertical, or both")
        self._mirror_direction = "both"
        self._staged_reward = StagedReward(
            stages=[
                Stage("reach", 0.25, lambda t, o: t._reach_progress(o)),
                Stage("grasp", 0.30, lambda t, o: t._grasp_progress(o)),
                Stage("lift", 0.45, lambda t, o: t._lift_progress(o)),
            ],
            success_fn=lambda t, o: t._lift_progress(o) >= 1.0,
        )

    def _reach_progress(self, raw_obs):
        if "cube_pos" not in raw_obs or "robot0_eef_pos" not in raw_obs:
            return 0.0
        dist = float(np.linalg.norm(raw_obs["robot0_eef_pos"] - raw_obs["cube_pos"]))
        return 1.0 - np.tanh(10.0 * dist)

    def _grasp_progress(self, raw_obs):
        return 1.0 if self._gripper_contacts_object("cube") else 0.0

    def _lift_progress(self, raw_obs):
        # Gated on a grasp so the sticky stage never keeps the spawn drop or a flicked cube.
        if "cube_pos" not in raw_obs or not self._gripper_contacts_object("cube"):
            return 0.0
        lift = raw_obs["cube_pos"][2] - (self._table_height + self._cube_size)
        return max(lift, 0.0) / self._lift_height

    def _make_objects(self):
        return [
            {"type": "box", "name": "cube",
             "size": [self._cube_size] * 3,
             "rgba": [0.2, 0.5, 0.8, 1.0],
             "density": self._cube_mass / (8 * self._cube_size ** 3),
             "friction": [1.0, 0.005, 0.0001],
             "joints": "default"},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._staged_reward.reset()

        # Randomize cube color (independent of the visual-randomization
        # flag so this task always has a varying cube color).
        sim = self._env.sim
        cube_body_id = sim.model.body_name2id("cube_main")
        geom_start = sim.model.body_geomadr[cube_body_id]
        geom_count = sim.model.body_geomnum[cube_body_id]
        rgba = self._random_object_color()
        for gi in range(geom_start, geom_start + geom_count):
            sim.model.geom_rgba[gi] = rgba

        # Sample once, after the explicit reset seed has reached the task
        # RNG. Observing, rendering and stepping never resample this value.
        self._mirror_direction = str(self._rng.choice(self._mirror_directions))
        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    def _get_obs(self, raw_obs):
        # All observation routes, including _rebuild_obs(), use this hook.
        # BaseTask caches raw camera frames and returns copies, so repeated
        # observation calls cannot flip a cached frame back and forth.
        obs = super()._get_obs(raw_obs)
        self._flip_cameras(obs)
        return obs

    def _mirror_image(self, image):
        if self._mirror_direction == "horizontal":
            return np.ascontiguousarray(image[:, ::-1])
        if self._mirror_direction == "vertical":
            return np.ascontiguousarray(image[::-1])
        return np.ascontiguousarray(image[::-1, ::-1])

    def _flip_cameras(self, obs):
        # Every camera is mirrored; the task has no unmirrored view.
        for label in self.camera_cfgs:
            for prefix in ("rgb_", "depth_"):
                key = prefix + label
                if key in obs:
                    obs[key] = self._mirror_image(obs[key])

    def render(self):
        frame = super().render()
        if frame is None:
            return None
        return self._mirror_image(frame)

    def _compute_task_score(self, raw_obs):
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._compute_task_score(raw_obs) >= self._success_score_threshold:
            return True
        return self._object_fell_off_table(raw_obs)
