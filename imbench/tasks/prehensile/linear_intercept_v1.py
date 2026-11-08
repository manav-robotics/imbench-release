"""Linear Intercept — intercept a moving object crossing the table.

Task ID: ``ib-p2-linear-intercept-v1``
"""

import numpy as np
from imbench.tasks.prehensile.contacts import body_geom_ids, geoms_touch, object_geom_ids
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class LinearInterceptV1(SingleArmBase):
    TASK_ID = "ib-p2-linear-intercept-v1"
    DESCRIPTION = (
        "Single arm: intercept and grasp a ball moving in a straight "
        "line across the table before it exits the workspace."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._object_speed_range = task_cfg.get("speed_range", [0.20, 0.50])
        self._object_radius = task_cfg.get("object_radius", 0.02)
        self._hold_steps_required = task_cfg.get("hold_steps", 30)
        self._hold_count = 0
        self._last_score = 0.0
        self._initial_velocity = None
        self._lift_threshold = 0.10
        self._ball_geoms = set()
        self._mount_geoms = set()

        self._staged_reward = StagedReward(
            stages=[
                Stage("approach_ball", 0.20, lambda t, o: t._approach_ball_progress(o)),
                Stage("initial_liftoff", 0.30, lambda t, o: 1.0 if t._hold_count >= 1 else 0.0),
                Stage("sustained_hold_complete", 0.50,
                      lambda t, o: t._hold_count / max(1, t._hold_steps_required)),
            ],
            success_fn=lambda t, o: t._hold_count >= t._hold_steps_required,
        )

    def _make_objects(self):
        return [
            {"type": "ball", "name": "cube",
             "size": [self._object_radius],
             "rgba": [0.9, 0.5, 0.1, 1.0],
             "density": 500,
             "friction": [0.5, 0.005, 0.0001],
             "joints": "default"},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._hold_count = 0
        self._last_score = 0.0
        self._staged_reward.reset()

        # Start the ball 15 cm from the table centre so it crosses the workspace
        angle = self._rng.uniform(0, 2 * np.pi)
        edge_dist = 0.15
        start_x = edge_dist * np.cos(angle)
        start_y = edge_dist * np.sin(angle)
        ball_z = self._table_height + self._object_radius + 0.005
        self._place_object("cube", [start_x, start_y, ball_z])

        # Set velocity roughly toward the opposite side
        speed = self._rng.uniform(*self._object_speed_range)
        toward_center = np.array([-start_x, -start_y, 0.0])
        toward_center[:2] /= np.linalg.norm(toward_center[:2]) + 1e-8
        # Add a random angular offset so it's not perfectly centered
        jitter = self._rng.uniform(-0.4, 0.4)
        vx = toward_center[0] * np.cos(jitter) - toward_center[1] * np.sin(jitter)
        vy = toward_center[0] * np.sin(jitter) + toward_center[1] * np.cos(jitter)
        self._initial_velocity = np.array([speed * vx, speed * vy, 0.0])
        self._set_object_velocity("cube", self._initial_velocity)

        sim = self._env.sim
        self._ball_geoms = object_geom_ids(self._env, "cube")
        self._mount_geoms = body_geom_ids(sim, self._env.robots[0].robot_model.base.root_body)
        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    def _approach_ball_progress(self, raw_obs):
        eef_pos = raw_obs.get("robot0_eef_pos")
        if eef_pos is None or "cube_pos" not in raw_obs:
            return 0.0
        dist = float(np.linalg.norm(eef_pos - raw_obs["cube_pos"]))
        return float(1.0 - np.tanh(10.0 * dist))

    def _compute_task_score(self, raw_obs):
        # Success = ball stays at least ``_lift_threshold`` above the table
        # for ``hold_steps`` consecutive ticks. Contact is not
        # required; the lift threshold itself rules out passive cases (a
        # ball at rest on the table can never satisfy it without being
        # grasped). Idempotent within a step: ``_check_termination`` reads
        # ``_last_score`` rather than re-calling this and double-counting.
        ball_z = raw_obs["cube_pos"][2] if "cube_pos" in raw_obs else 0.0
        lifted = ball_z >= self._table_height + self._lift_threshold
        if lifted:
            self._hold_count += 1
        else:
            self._hold_count = 0

        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        self._last_score = score
        return score

    def _check_termination(self, raw_obs):
        if self._last_score >= self._success_score_threshold:
            return True
        # Terminate if ball left the table area
        if "cube_pos" in raw_obs:
            pos = raw_obs["cube_pos"]
            if abs(pos[0]) > 0.8 or abs(pos[1]) > 0.8:
                return True
        # The robot's pedestal stands flush with the table's robot-side edge:
        # a ball reaching it has left the workspace there instead of
        # rebounding onto the table as a static pick.
        if geoms_touch(self._env.sim, self._ball_geoms, self._mount_geoms):
            return True
        return self._object_fell_off_table(raw_obs)
