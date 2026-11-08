"""Post-Collision Intercept — intercept a ball after wall bounces.

Task ID: ``ib-p2-bounce-intercept-v1``
"""

import numpy as np
from imbench.tasks.prehensile.contacts import object_geom_ids, robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


class BounceInterceptV1(SingleArmBase):
    TASK_ID = "ib-p2-bounce-intercept-v1"
    DESCRIPTION = (
        "Single arm: a ball bounces off table-top walls; predict the "
        "post-collision trajectory and intercept it."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._object_speed_range = task_cfg.get("speed_range", [0.25, 0.50])
        self._object_radius = task_cfg.get("object_radius", 0.02)
        self._wall_restitution = task_cfg.get("wall_restitution", 0.8)
        self._wall_size = task_cfg.get("wall_size", [0.01, 0.20, 0.04])
        self._hold_steps_required = task_cfg.get("hold_steps", 30)
        self._hold_count = 0
        self._last_score = 0.0
        self._failed = False
        self._lift_threshold = 0.10
        self._has_bounced = False

        self._staged_reward = StagedReward(
            stages=[
                Stage("track_prebounce", 0.15, lambda t, o: t._track_prebounce_progress(o)),
                Stage("post_bounce_intercept_begin", 0.35,
                      lambda t, o: 1.0 if (t._has_bounced and t._hold_count >= 1) else 0.0),
                Stage("sustained_hold_complete", 0.50,
                      lambda t, o: t._hold_count / max(1, t._hold_steps_required)),
            ],
            success_fn=lambda t, o: t._hold_count >= t._hold_steps_required,
        )

    def _make_objects(self):
        return [
            {"type": "ball", "name": "cube",
             "size": [self._object_radius],
             "rgba": [0.9, 0.3, 0.1, 1.0],
             "density": 500,
             "friction": [0.3, 0.005, 0.0001],
             "joints": "default"},
        ]

    # All four edges of the table. Each entry: (name, axis, sign).
    _EDGES = [
        ("x_plus",  "x", +1),
        ("x_minus", "x", -1),
        ("y_plus",  "y", +1),
        ("y_minus", "y", -1),
    ]

    def _make_static_objects(self):
        table_hx = self._table_full_size[0] / 2
        table_hy = self._table_full_size[1] / 2
        thickness, _, height = self._wall_size
        walls = []
        for i, (_name, axis, _sign) in enumerate(self._EDGES):
            edge_half_len = table_hy if axis == "x" else table_hx
            size = [thickness, edge_half_len, height]
            walls.append({
                "type": "box", "name": f"wall_{i}",
                "size": size,
                "rgba": [0.5, 0.5, 0.5, 1.0],
                "density": 5000, "friction": [0.1, 0.005, 0.0001],
                "joints": None,
            })
        return walls

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._hold_count = 0
        self._last_score = 0.0
        self._failed = False
        self._has_bounced = False
        self._staged_reward.reset()

        sim = self._env.sim

        table_hx = self._table_full_size[0] / 2
        table_hy = self._table_full_size[1] / 2
        wall_z = self._table_height + self._wall_size[2]

        # wall_size entries are half-extents (robosuite BoxObject convention).
        half_thickness = self._wall_size[0]
        quat_y_edge = [0.7071, 0.0, 0.0, 0.7071]  # 90° about z: long axis → x

        # Randomize which edge is the opening; the other three form the U.
        opening_idx = int(self._rng.randint(len(self._EDGES)))
        _, open_axis, open_sign = self._EDGES[opening_idx]

        # Park the opening wall well below the table so it's out of play.
        hidden_z = self._table_height - 1.0
        for i, (_name, axis, sign) in enumerate(self._EDGES):
            if i == opening_idx:
                self._place_static(f"wall_{i}", [0.0, 0.0, hidden_z])
                continue
            if axis == "x":
                pos = [sign * (table_hx - half_thickness), 0.0, wall_z]
                quat = None
            else:
                pos = [0.0, sign * (table_hy - half_thickness), wall_z]
                quat = quat_y_edge
            self._place_static(f"wall_{i}", pos, quat=quat)

        # solref[1] is the walls' contact damping ratio: lower → bouncier.
        # MuJoCo averages it with the ball's default 1.0 (equal solmix), so
        # this only loosely sets the rebound: a ball hitting a wall at 1 m/s
        # comes back at about 0.37 m/s for wall_restitution 0.9 and about
        # 0.22 m/s for 0.0.
        solref_damping = float(np.clip(1.0 - self._wall_restitution, 0.0, 1.0))
        for i in range(len(self._EDGES)):
            try:
                gid = sim.model.geom_name2id(f"wall_{i}_g0")
                sim.model.geom_solref[gid] = [0.02, solref_damping]
                sim.model.geom_solimp[gid] = [0.998, 0.998, 0.001, 0.5, 2]
            except Exception:
                pass

        # Ball spawns near the wall OPPOSITE the opening so it travels across
        # the enclosure and ricochets off the side walls on its way back.
        spawn_axis, spawn_sign = open_axis, -open_sign
        clearance = 2 * half_thickness + self._object_radius + 0.05
        if spawn_axis == "x":
            ball_x = spawn_sign * (table_hx - clearance)
            ball_y = self._rng.uniform(-0.10, 0.10)
        else:
            ball_x = self._rng.uniform(-0.10, 0.10)
            ball_y = spawn_sign * (table_hy - clearance)
        ball_z = self._table_height + self._object_radius + 0.005
        self._place_object("cube", [ball_x, ball_y, ball_z])

        # Launch aimed mostly sideways (toward a side wall) with only a small
        # component toward the opening, so the ball is guaranteed to hit a
        # side wall before reaching the opening. `tilt` is measured from the
        # side-wall axis — 0° would be purely sideways (infinite bounces),
        # 45° is a diagonal, >45° would head straight for the opening.
        speed = self._rng.uniform(*self._object_speed_range)
        side = self._rng.choice([-1.0, 1.0])
        tilt = self._rng.uniform(np.deg2rad(10), np.deg2rad(35))
        across = side * speed * np.cos(tilt)
        along = speed * np.sin(tilt)  # toward opening
        if spawn_axis == "x":
            vx = -spawn_sign * along
            vy = across
        else:
            vx = across
            vy = -spawn_sign * along
        self._set_object_velocity("cube", [vx, vy, 0.0])

        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    def _detect_wall_bounce(self):
        """Set ``self._has_bounced`` once the ball contacts any wall geom."""
        if self._has_bounced:
            return
        sim = self._env.sim
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1 = sim.model.geom_id2name(c.geom1) or ""
            g2 = sim.model.geom_id2name(c.geom2) or ""
            ball_hit = "cube" in g1 or "cube" in g2
            wall_hit = g1.startswith("wall_") or g2.startswith("wall_")
            if ball_hit and wall_hit:
                self._has_bounced = True
                return

    def _track_prebounce_progress(self, raw_obs):
        # Only credited before the bounce happens; sticky max preserves
        # whatever tracking peak was reached once the ball ricochets.
        if self._has_bounced:
            return 0.0
        eef_pos = raw_obs.get("robot0_eef_pos")
        if eef_pos is None or "cube_pos" not in raw_obs:
            return 0.0
        dist = float(np.linalg.norm(eef_pos - raw_obs["cube_pos"]))
        return float(1.0 - np.tanh(6.0 * dist))

    def _compute_task_score(self, raw_obs):
        if not self._failed:
            self._detect_wall_bounce()
            # Intercepting the ball before it has hit a wall fails the episode.
            if not self._has_bounced and robot_contacts(
                    self._env.sim, object_geom_ids(self._env, "cube")):
                self._failed = True
        if self._failed:
            self._last_score = 0.0
            self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
            return 0.0

        # Success = ball stays at least ``_lift_threshold`` above the table
        # for ``hold_steps`` consecutive ticks after its first wall bounce.
        # Contact is not required — passive cases (a ball at rest on the
        # table) can't satisfy the lift threshold without being grasped.
        # Idempotent within a step: ``_check_termination`` reads
        # ``_last_score`` rather than re-calling this and double-counting.
        ball_z = raw_obs["cube_pos"][2] if "cube_pos" in raw_obs else 0.0
        lifted = ball_z >= self._table_height + self._lift_threshold
        if lifted and self._has_bounced:
            self._hold_count += 1
        else:
            self._hold_count = 0

        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        self._last_score = score
        return score

    def _check_termination(self, raw_obs):
        if self._failed:
            return True
        if self._last_score >= self._success_score_threshold:
            return True
        if self._object_fell_off_table(raw_obs):
            self._failed = True
            return True
        if "cube_pos" in raw_obs:
            pos = raw_obs["cube_pos"]
            if abs(pos[0]) > 1.0 or abs(pos[1]) > 1.0:
                self._failed = True
                return True
        return False
