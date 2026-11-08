"""Indirect Push Chain — scatter blocks, arrange them into a chain, then push to score.

Task ID: ``ib-p3-indirect-push-v1``
"""

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward, progress_from_reset

# Palette for chain blocks (target block is always red).
CHAIN_COLORS = [
    [0.30, 0.50, 0.85, 1.0],  # blue
    [0.20, 0.70, 0.35, 1.0],  # green
    [0.95, 0.80, 0.15, 1.0],  # yellow
    [0.60, 0.30, 0.80, 1.0],  # purple
    [1.00, 0.55, 0.10, 1.0],  # orange
    [0.15, 0.75, 0.85, 1.0],  # cyan
    [0.95, 0.45, 0.70, 1.0],  # pink
    [0.55, 0.30, 0.15, 1.0],  # brown
]


class IndirectPushV1(SingleArmBase):
    TASK_ID = "ib-p3-indirect-push-v1"
    DESCRIPTION = (
        "Puzzle: blocks and a red target are scattered randomly on the "
        "table. The robot must arrange the blocks into a contiguous "
        "chain leading from one side to the target, then push the "
        "chain so that force propagates and drives the target into "
        "the goal zone."
    )

    #: Consecutive steps the target must rest in the goal disc. A pushed block
    #: moves in stick-slip, so one slow sample does not mean the push is over.
    GOAL_REST_STEPS = 10

    # ── Config ──────────────────────────────────────────────────────

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._block_size = task_cfg.get("block_size", 0.025)
        self._goal_pos = np.array(task_cfg.get("goal_pos", [0.35, 0.0]))
        self._goal_tolerance = task_cfg.get("goal_tolerance", 0.03)
        self._num_chain = task_cfg.get("num_chain", 4)

        # Per-reset y-randomisation for the goal marker.
        goal_rand = self.cfg.get("randomization", {}).get("goal", {})
        self._rand_goal_y = goal_rand.get("y_range")  # e.g. [-0.15, 0.15]

        # Scatter region — the reachable area on the table
        scatter_cfg = task_cfg.get("scatter_region", {})
        self._scatter_x = scatter_cfg.get("x_range", [-0.05, 0.25])
        self._scatter_y = scatter_cfg.get("y_range", [-0.15, 0.15])

        rand_cfg = self.cfg.get("randomization", {})
        chain_rand = rand_cfg.get("chain", {})
        self._rand_num = chain_rand.get("num_range")        # e.g. [2, 6]
        self._rand_size = chain_rand.get("size_range")       # e.g. [0.018, 0.032]

        # Every block the task could ever need is built into the model
        # (see _make_objects); each reset uses the first _active_chain of
        # them and parks the rest off-scene.
        self._max_chain = self._num_chain
        if self._rand_num:
            self._max_chain = max(self._max_chain, self._rand_num[1])
        self._active_chain = self._num_chain

        self._active_chain = 0
        self._active_half_sizes = np.full(self._num_chain, self._block_size)
        self._target_start_dist = 1e-6
        self._chain_formed_at_reset = 0.0
        self._goal_rest_steps = 0

        # "chain_formed" scores how close consecutive blocks (sorted along
        # the push axis, chain blocks + target) are to touching, rescaled
        # so the scattered layout at reset scores 0 and a chain where every
        # link is edge-to-edge scores 1.
        self._staged_reward = StagedReward(
            stages=[
                Stage("chain_formed", 0.4, lambda t, o: t._chain_formed_progress(o)),
                Stage("target_progress", 0.3, lambda t, o: t._target_progress(o)),
                Stage("target_in_goal", 0.3, lambda t, o: 1.0 if t._target_in_goal() else 0.0),
            ],
            success_fn=lambda t, o: t._target_in_goal(),
        )

    # ── Object factories ────────────────────────────────────────────

    def _make_objects(self):
        objects = []
        for i in range(self._max_chain):
            objects.append({
                "type": "box",
                "name": f"chain_{i}",
                "size": [self._block_size] * 3,
                "rgba": CHAIN_COLORS[i % len(CHAIN_COLORS)],
                "density": 400,
                "friction": [0.5, 0.005, 0.0001],
                "joints": "default",
            })
        # Target block (always red)
        objects.append({
            "type": "box", "name": "target",
            "size": [self._block_size] * 3,
            "rgba": [0.85, 0.20, 0.15, 1.0],
            "density": 400,
            "friction": [0.5, 0.005, 0.0001],
            "joints": "default",
        })
        return objects

    def _make_static_objects(self):
        return [
            {"type": "cylinder", "name": "goal_marker",
             "size": [self._goal_tolerance, 0.001],
             "rgba": [0.0, 0.8, 0.0, 0.3],
             "density": 1, "friction": [0.1, 0.005, 0.0001],
             "joints": None},
        ]

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._staged_reward.reset()
        sim = self._env.sim

        # Randomise the goal's y-coordinate (x stays fixed so the
        # goal remains the "unreachable" side of the table).
        if self._rand_goal_y:
            self._goal_pos[1] = self._rng.uniform(*self._rand_goal_y)

        # Randomise active chain length
        num_active = self._num_chain
        if self._rand_num:
            num_active = self._rng.randint(self._rand_num[0],
                                           self._rand_num[1] + 1)
        num_active = min(num_active, self._max_chain)
        self._active_chain = num_active
        self._parked = {f"chain_{i}"
                        for i in range(num_active, self._max_chain)}

        # Per-block half-sizes (randomised per reset)
        half_sizes = np.full(num_active, self._block_size)
        if self._rand_size:
            half_sizes = self._rng.uniform(
                self._rand_size[0], self._rand_size[1], size=num_active)
        target_hs = self._block_size
        self._active_half_sizes = half_sizes

        bz_base = self._table_height + 0.005

        # ── Scatter chain blocks randomly on the table ──────────
        # min sep = 4× block half-size (= 2× full side length) gives a
        # gap of one full block between spawn centres, enough that
        # solver penetration never launches them on the first step.
        min_sep = self._block_size * 4
        positions = []
        for i in range(num_active):
            hs = float(half_sizes[i])
            self._resize_box(sim, f"chain_{i}", hs)
            self._set_object_collisions(sim, f"chain_{i}", True)
            pos = self._sample_non_overlapping(
                positions, min_sep,
                self._scatter_x, self._scatter_y,
            )
            positions.append(pos)
            bz = bz_base + hs
            self._place_object(f"chain_{i}", [pos[0], pos[1], bz])
            self._set_object_velocity(f"chain_{i}", [0, 0, 0])

        # Target block — placed directly next to the goal on the -x
        # side so a straight +x push propagates into the goal zone.
        target_offset = target_hs + self._goal_tolerance + 0.02
        target_x = self._goal_pos[0] - target_offset
        target_y = self._goal_pos[1]
        self._place_object("target",
                           [target_x, target_y, bz_base + target_hs])
        self._set_object_velocity("target", [0, 0, 0])
        self._target_start_dist = max(
            float(np.linalg.norm(np.array([target_x, target_y]) - self._goal_pos)),
            1e-6,
        )

        # Park the blocks this episode does not use.  They are stripped
        # from the obs dict by _parked_objects(), so the agent sees a
        # scene with exactly num_active blocks in it.
        self._park_objects(self._parked)

        # Goal marker, visual only: as a collidable 2 mm curb it snags the
        # target, which jams the chain and is then launched over it.
        self._place_static("goal_marker", [
            self._goal_pos[0], self._goal_pos[1],
            self._table_height + 0.001,
        ])
        self._set_object_collisions(sim, "goal_marker", False)
        self._goal_rest_steps = 0

        sim.forward()
        self._chain_formed_at_reset = self._chain_closeness()
        obs = self._rebuild_obs()
        return obs, info

    # ── Helpers ─────────────────────────────────────────────────────

    def _parked_objects(self):
        return getattr(self, "_parked", set())

    def _sample_non_overlapping(self, existing, min_sep, x_range, y_range,
                                max_attempts=50):
        """Sample a 2-D position that doesn't overlap with *existing* positions.

        The scatter region is small enough that 6 blocks at ``min_sep`` often
        cannot be packed by rejection sampling.  Rather than returning the
        last (arbitrarily bad) draw, keep the candidate with the largest
        clearance so blocks never spawn interpenetrating.
        """
        best, best_clearance = None, -np.inf
        for _ in range(max_attempts):
            x = self._rng.uniform(x_range[0], x_range[1])
            y = self._rng.uniform(y_range[0], y_range[1])
            clearance = min((np.hypot(x - ex, y - ey) for ex, ey in existing),
                            default=np.inf)
            if clearance >= min_sep:
                return (x, y)
            if clearance > best_clearance:
                best, best_clearance = (x, y), clearance
        return best

    @staticmethod
    def _set_object_collisions(sim, obj_name, enabled):
        """Enable/disable contacts for every geom belonging to *obj_name*."""
        body_id = sim.model.body_name2id(f"{obj_name}_main")
        flag = 1 if enabled else 0
        for g_idx in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g_idx] == body_id:
                sim.model.geom_contype[g_idx] = flag
                sim.model.geom_conaffinity[g_idx] = flag

    @staticmethod
    def _resize_box(sim, obj_name, half_size):
        """Resize all geoms belonging to *obj_name* to a cube of *half_size*."""
        body_id = sim.model.body_name2id(f"{obj_name}_main")
        for g_idx in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g_idx] == body_id:
                sim.model.geom_size[g_idx] = [half_size, half_size, half_size]

    # ── Scoring ─────────────────────────────────────────────────────

    def _target_pos(self):
        sim = self._env.sim
        try:
            target_id = sim.model.body_name2id("target_main")
            return np.array(sim.data.body_xpos[target_id][:2], dtype=float)
        except Exception:
            return None

    def _target_resting_in_goal(self):
        """Target within the goal disc and slower than ``REST_SPEED`` this step."""
        pos = self._target_pos()
        if pos is None:
            return False
        return (float(np.linalg.norm(pos - self._goal_pos)) < self._goal_tolerance
                and self._object_speed("target") < self.REST_SPEED)

    def _target_in_goal(self):
        """Target has rested in the goal disc for ``GOAL_REST_STEPS`` consecutive steps."""
        return self._goal_rest_steps >= self.GOAL_REST_STEPS

    def _target_progress(self, raw_obs):
        pos = self._target_pos()
        if pos is None:
            return 0.0
        dist = float(np.linalg.norm(pos - self._goal_pos))
        return float(np.clip(1.0 - dist / self._target_start_dist, 0.0, 1.0))

    def _chain_link_entries(self):
        """Return ``[(xy, half_size), ...]`` for every active chain block
        plus the target block — the objects that must relay a push."""
        sim = self._env.sim
        entries = []
        for i in range(self._active_chain):
            try:
                bid = sim.model.body_name2id(f"chain_{i}_main")
            except Exception:
                continue
            hs = float(self._active_half_sizes[i]) if i < len(self._active_half_sizes) else self._block_size
            entries.append((np.array(sim.data.body_xpos[bid][:2], dtype=float), hs))
        target_pos = self._target_pos()
        if target_pos is not None:
            entries.append((target_pos, self._block_size))
        return entries

    def _chain_formed_progress(self, raw_obs):
        """Chain closeness rescaled so the reset layout scores 0."""
        return progress_from_reset(self._chain_closeness(), self._chain_formed_at_reset)

    def _chain_closeness(self):
        """Sort chain blocks + target along the push axis and score how
        close each consecutive pair's edge-to-edge gap is to touching:
        1 per link when edge-to-edge, 0 at a gap of 4 block half-sizes.
        """
        entries = self._chain_link_entries()
        if len(entries) < 2:
            return 0.0
        entries.sort(key=lambda e: e[0][0])
        threshold = self._block_size * 4.0
        scores = []
        for (p0, h0), (p1, h1) in zip(entries[:-1], entries[1:]):
            gap = max(0.0, float(np.linalg.norm(p1 - p0)) - h0 - h1)
            scores.append(max(0.0, 1.0 - gap / max(threshold, 1e-6)))
        return float(np.mean(scores))

    def _compute_task_score(self, raw_obs):
        self._goal_rest_steps = self._goal_rest_steps + 1 if self._target_resting_in_goal() else 0
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _any_block_fell(self):
        """True if the target or any active chain block dropped below the floor threshold."""
        sim = self._env.sim
        names = [f"chain_{i}" for i in range(self._active_chain)] + ["target"]
        return any(
            sim.data.body_xpos[sim.model.body_name2id(f"{n}_main")][2]
            < self._failure_floor_threshold
            for n in names
        )

    def _check_termination(self, raw_obs):
        score = self._compute_task_score(raw_obs)
        if score >= self._success_score_threshold:
            return True
        return self._any_block_fell()
