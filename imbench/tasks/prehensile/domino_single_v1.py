"""Domino Chain Trigger — topple the first domino to cascade and reach a target.

Task ID: ``ib-p3-domino-single-v1``
"""

import numpy as np
from imbench.tasks.prehensile.contacts import body_geom_ids, robot_contacts
from imbench.tasks.prehensile.domino_cascade import DominoCascadeMixin
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.prehensile.domino_patterns import (
    ALL_PATTERNS,
    generate_safe_pattern,
    heading_to_quat,
    layout_has_collisions,
    rotate_layout,
    translate_layout,
)
from imbench.tasks.staged_reward import Stage, StagedReward

# 10-color palette for dominoes. Red is reserved for the target block.
DOMINO_COLORS = [
    [0.20, 0.40, 0.90, 1.0],  # blue
    [0.20, 0.70, 0.30, 1.0],  # green
    [0.95, 0.80, 0.15, 1.0],  # yellow
    [0.60, 0.30, 0.80, 1.0],  # purple
    [1.00, 0.55, 0.10, 1.0],  # orange
    [0.15, 0.75, 0.85, 1.0],  # cyan
    [0.95, 0.45, 0.70, 1.0],  # pink
    [0.55, 0.30, 0.15, 1.0],  # brown
    [0.70, 0.70, 0.70, 1.0],  # gray
    [0.10, 0.10, 0.10, 1.0],  # black
]


class DominoSingleV1(DominoCascadeMixin, SingleArmBase):
    TASK_ID = "ib-p3-domino-single-v1"
    DESCRIPTION = (
        "Single arm: topple the first domino in a chain to trigger a "
        "cascade that knocks over a target at the end."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._num_dominoes = task_cfg.get("num_dominoes", 6)
        self._domino_spacing = task_cfg.get("domino_spacing", 0.045)
        self._domino_size = task_cfg.get("domino_size", [0.06, 0.03, 0.01])
        self._target_displacement = task_cfg.get("target_displacement_threshold", 0.05)
        self._initial_target_pos = None

        rand_cfg = self.cfg.get("randomization", {})
        self._rand_spacing = rand_cfg.get("spacing_factor", {}).get("range")
        self._rand_num = rand_cfg.get("num_dominoes", {}).get("range")

        # Pattern randomization
        pat_cfg = rand_cfg.get("pattern", {})
        self._pattern_enabled = pat_cfg.get("enabled", False)
        self._patterns = pat_cfg.get("types", ALL_PATTERNS)

        # Placement offset randomization
        off_cfg = rand_cfg.get("placement_offset", {})
        self._rand_offset = off_cfg.get("enabled", False)
        self._offset_x_range = off_cfg.get("x_range", 0.05)
        self._offset_y_range = off_cfg.get("y_range", 0.05)

        # Max objects we'll ever need (for _make_objects)
        self._max_dominoes = self._num_dominoes
        if self._rand_num:
            self._max_dominoes = max(self._max_dominoes, self._rand_num[1])

        # Tracks the actual number of dominoes placed each episode (set in
        # reset()); defaulted here so stage functions are safe pre-reset.
        self._active_num_dominoes = self._num_dominoes
        self._topple_tilt_threshold_deg = task_cfg.get("topple_tilt_threshold_deg", 45.0)
        self._robot_touched_chain = False
        self._first_domino_toppled = False
        self._cascade_geoms = []
        self._link_touch_step = np.zeros(0, dtype=int)
        self._no_touch_geoms = set()
        self._parked = set()

        self._staged_reward = StagedReward(
            stages=[
                Stage("trigger_first_domino", 0.30,
                      lambda t, o: 1.0 if t._first_domino_toppled else 0.0),
                Stage("cascade_reaches_midpoint", 0.30,
                      lambda t, o: 1.0 if t._cascade_reached_midpoint() else 0.0),
                Stage("target_displaced", 0.40,
                      lambda t, o: float(t._cascade_displaced_target())),
            ],
            success_fn=lambda t, o: t._cascade_displaced_target(),
        )

    def _make_objects(self):
        objects = []
        # Create dominoes standing upright — size is [half_x, half_y, half_z]
        # Standing upright means the tallest dimension is along z
        # domino_size = [height, width, thickness] when standing
        ds = self._domino_size
        for i in range(self._max_dominoes):
            objects.append({
                "type": "box",
                "name": f"domino_{i}",
                "size": [ds[2], ds[1], ds[0]],  # thin x, medium y, tall z (standing)
                "rgba": [0.8, 0.8, 0.2, 1.0],
                "density": 800,
                "friction": [0.8, 0.005, 0.0001],
                "joints": "default",
            })
        # Target object at the end — thin slab that topples easily
        objects.append({
            "type": "box", "name": "red_block",
            "size": [0.005, 0.03, 0.04],  # thin slab standing upright
            "rgba": [0.9, 0.1, 0.1, 1.0],
            "density": 400,
            "friction": [0.5, 0.005, 0.0001],
            "joints": "default",
        })
        return objects

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._staged_reward.reset()

        sim = self._env.sim
        spacing = self._domino_spacing
        ds = self._domino_size  # [height, width, thickness] standing

        # Randomize spacing factor
        if self._rand_spacing:
            factor = self._rng.uniform(*self._rand_spacing)
            spacing *= factor

        # Randomize number of dominoes
        num_active = self._num_dominoes
        if self._rand_num:
            num_active = self._rng.randint(self._rand_num[0], self._rand_num[1] + 1)
        num_active = min(num_active, self._max_dominoes)
        self._active_num_dominoes = num_active

        # Choose pattern
        if self._pattern_enabled:
            pattern_name = self._rng.choice(self._patterns)
        else:
            pattern_name = "line"

        footprint = (ds[2], ds[1])  # (half_x, half_y) when standing
        layout = generate_safe_pattern(
            pattern_name, num_active, spacing, self._rng, footprint
        )

        # Optional random rotation (±30 °)
        if self._pattern_enabled:
            rot = self._rng.uniform(-np.pi / 6, np.pi / 6)
            candidate = rotate_layout(layout, rot)
            if not layout_has_collisions(candidate, footprint):
                layout = candidate

        # Placement offset
        if self._rand_offset:
            ox = self._rng.uniform(-self._offset_x_range, self._offset_x_range)
            oy = self._rng.uniform(-self._offset_y_range, self._offset_y_range)
        else:
            ox, oy = 0.0, 0.0

        # Centre layout at (ox, oy), then fit it on the table while
        # keeping clear of the robot arm (which rests near x ≈ 0).
        xs = [p[0] for p in layout]
        ys = [p[1] for p in layout]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        layout = translate_layout(layout, -cx + ox, -cy + oy)

        margin = 0.01
        half_tx = self._table_full_size[0] / 2 - margin
        half_ty = self._table_full_size[1] / 2 - margin
        min_safe_x = 0.08  # keep dominos clear of robot EE at x ≈ 0

        # If the chain is too wide for the safe zone, uniformly scale it
        # to fit within [min_safe_x, half_tx] × [-half_ty, half_ty].
        xs = [p[0] for p in layout]
        ys = [p[1] for p in layout]
        chain_w = max(xs) - min(xs)
        chain_h = max(ys) - min(ys)
        avail_x = half_tx - min_safe_x
        avail_y = 2 * half_ty
        scale = 1.0
        if chain_w > avail_x:
            scale = min(scale, avail_x / chain_w)
        if chain_h > avail_y:
            scale = min(scale, avail_y / chain_h)
        if scale < 1.0:
            cx2 = (min(xs) + max(xs)) / 2
            cy2 = (min(ys) + max(ys)) / 2
            layout = [
                ((x - cx2) * scale + cx2, (y - cy2) * scale + cy2, h)
                for x, y, h in layout
            ]

        # Shift right to keep min(x) >= min_safe_x, but stay on table.
        xs = [p[0] for p in layout]
        shift_r = max(0.0, min_safe_x - min(xs))
        if max(xs) + shift_r > half_tx:
            shift_r = max(0.0, half_tx - max(xs))
        if shift_r:
            layout = translate_layout(layout, shift_r, 0)

        # Shift y to keep all dominos on the table.
        ys = [p[1] for p in layout]
        if max(ys) > half_ty:
            layout = translate_layout(layout, 0, half_ty - max(ys))
        ys = [p[1] for p in layout]
        if min(ys) < -half_ty:
            layout = translate_layout(layout, 0, -half_ty - min(ys))

        for i in range(num_active):
            x, y, heading = layout[i]
            z = self._table_height + ds[0] + 0.002
            quat = heading_to_quat(heading)
            self._place_object(f"domino_{i}", [x, y, z], quat=quat)
            self._set_object_velocity(f"domino_{i}", [0, 0, 0])

        self._parked = {f"domino_{i}" for i in range(num_active, self._max_dominoes)}
        self._park_objects(self._parked)

        # Place target slab just past the end of the chain (clamped to table)
        last_x, last_y, last_h = layout[-1]
        target_x = np.clip(
            last_x + spacing * np.cos(last_h), -half_tx, half_tx)
        target_y = np.clip(
            last_y + spacing * np.sin(last_h), -half_ty, half_ty)
        target_z = self._table_height + 0.04 + 0.002
        target_quat = heading_to_quat(last_h)
        self._place_object("red_block", [target_x, target_y, target_z], quat=target_quat)

        sim.forward()

        # Zero object velocities to prevent drift from contact resolution.
        # Do NOT zero robot qvel — that disrupts the OSC controller state.
        for i in range(num_active):
            self._set_object_velocity(f"domino_{i}", [0, 0, 0], [0, 0, 0])
        self._set_object_velocity("red_block", [0, 0, 0], [0, 0, 0])

        # Randomize domino colors from the non-red palette. The target
        # block stays red so it remains visually distinct.
        if self._rand_object_color:
            for i in range(self._max_dominoes):
                color = DOMINO_COLORS[self._rng.randint(len(DOMINO_COLORS))]
                self._set_object_color(f"domino_{i}", color)
            sim.forward()

        # Record initial target position after forward pass
        try:
            target_id = sim.model.body_name2id("red_block_main")
            self._initial_target_pos = sim.data.body_xpos[target_id].copy()
        except Exception:
            self._initial_target_pos = np.zeros(3)

        self._robot_touched_chain = False
        self._first_domino_toppled = False
        # Cascade order: each chain domino, then the target.
        self._cascade_geoms = [
            body_geom_ids(sim, f"{name}_main")
            for name in [f"domino_{i}" for i in range(num_active)] + ["red_block"]
        ]
        self._link_touch_step = np.full(num_active, -1)
        self._update_cascade_links()
        # Only the first domino may be touched; the robot contacting any
        # later domino or the target disqualifies the episode.
        self._no_touch_geoms = set().union(*self._cascade_geoms[1:])

        obs = self._rebuild_obs()
        return obs, info

    def _parked_objects(self):
        return self._parked

    def _set_object_color(self, name, rgba):
        sim = self._env.sim
        body_id = sim.model.body_name2id(f"{name}_main")
        rgba = np.asarray(rgba, dtype=float)
        for g in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g] == body_id:
                sim.model.geom_rgba[g] = rgba

    def _robot_touches_chain(self):
        """Whether the robot touches the target or any domino after the first."""
        return robot_contacts(self._env.sim, self._no_touch_geoms)

    def _domino_toppled(self, index):
        """True once ``domino_{index}``'s tilt from vertical passes threshold.

        Mirrors the rod-tilt convention used in balance tasks: angle between
        the domino's local Z axis (its standing "tall" axis) and world Z.
        """
        sim = self._env.sim
        try:
            body_id = sim.model.body_name2id(f"domino_{index}_main")
        except Exception:
            return False
        xmat = sim.data.body_xmat[body_id].reshape(3, 3)
        body_z = xmat[:, 2]
        cos_angle = np.clip(abs(body_z[2]), 0.0, 1.0)
        tilt_deg = float(np.degrees(np.arccos(cos_angle)))
        return tilt_deg >= self._topple_tilt_threshold_deg

    def _cascade_reached_midpoint(self):
        """True once the domino at the chain's midpoint has toppled."""
        n = getattr(self, "_active_num_dominoes", self._num_dominoes)
        if n <= 1:
            return self._domino_toppled(0)
        return self._domino_toppled(n // 2)

    def _target_displaced(self):
        sim = self._env.sim
        try:
            target_id = sim.model.body_name2id("red_block_main")
            target_pos = sim.data.body_xpos[target_id].copy()
            displacement = np.linalg.norm(target_pos - self._initial_target_pos)
        except Exception:
            displacement = 0.0
        return displacement > self._target_displacement

    def _compute_task_score(self, raw_obs):
        # Hard zeroing guard: the robot toppling the target or the chain by
        # hand instead of via the cascade is disqualifying, regardless of
        # any stage progress already accrued this episode.
        if not self._robot_touched_chain and self._robot_touches_chain():
            self._robot_touched_chain = True
        if self._robot_touched_chain:
            self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
            return 0.0
        if self._domino_toppled(0):
            self._first_domino_toppled = True
        self._update_cascade_links()

        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        score = self._compute_task_score(raw_obs)
        if self._robot_touched_chain:
            return True
        if score >= self._success_score_threshold:
            return True
        return False
