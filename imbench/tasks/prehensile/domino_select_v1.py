"""Select the chain whose contact cascade knocks over the red target.

``task.layout`` picks the scene. ``patterns`` (v1) draws one of the
randomized layouts in :mod:`domino_layouts`: a true route, often a zig-zag,
curve or loop, among decoys that look connected to red or to the route but
stop at a physical break, meet another chain side-on, or end beside red or
short of it. ``merge`` (v2) is the fixed cross-chain Y-merge and ``lanes``
the independent lanes (also selected by task.interconnected=false).
Decoy topples still fail immediately; success requires every route contact
in order and the target actually displaced.
"""

import numpy as np
from imbench.tasks.prehensile.contacts import body_geom_ids, robot_contacts, geoms_touch
from imbench.tasks.prehensile.domino_cascade import DominoCascadeMixin
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.prehensile.domino_patterns import heading_to_quat
from imbench.tasks.physics import box_inertia, commit_model_edits, set_body_mass
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


class DominoSelectV1(DominoCascadeMixin, SingleArmBase):
    TASK_ID = "ib-p3-domino-select-v1"
    DESCRIPTION = (
        "Single arm: domino chains wind across the table toward a red "
        "target. Only one route topples it; look-alike chains stop at "
        "hidden breaks, meet other chains side-on, or miss the target."
    )

    _LAYOUT_YAW = 0.0
    # Compile at maximum dimensions so shrinking geoms at reset preserves
    # conservative MuJoCo collision bounds.
    _SIZE_MIN = np.array([0.85, 0.80, 0.85])
    _SIZE_MAX = np.array([1.10, 1.10, 1.10])

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._num_chains = task_cfg.get("num_chains", 5)
        self._dominoes_per_chain = task_cfg.get("dominoes_per_chain", 6)
        layout = task_cfg.get("layout")
        if task_cfg.get("interconnected") is False:
            layout = "lanes"
        elif layout is None:
            layout = "merge" if task_cfg.get("interconnected") else "lanes"
        if layout not in ("patterns", "merge", "lanes"):
            raise ValueError(f"Unknown domino-select layout: {layout}")
        self._layout_mode = layout
        # Patterns and the merge both route the cascade across chain pools.
        self._interconnected = layout != "lanes"
        if layout == "merge" and (self._num_chains < 5 or self._dominoes_per_chain < 6):
            raise ValueError("Interconnected domino selection needs at least five chains of six blocks")
        self._domino_size = task_cfg.get("domino_size", [0.06, 0.03, 0.01])
        self._correct_spacing = task_cfg.get("correct_spacing", 0.045)
        self._chain_y_spacing = task_cfg.get("chain_y_spacing", 0.145)
        self._target_displacement = task_cfg.get("target_displacement_threshold", 0.05)
        self._correct_chain = 0  # randomized in reset
        self._initial_target_pos = None

        # Pattern randomization
        rand_cfg = self.cfg.get("randomization", {})
        pat_cfg = rand_cfg.get("pattern", {})
        self._pattern_enabled = pat_cfg.get("enabled", False)
        self._patterns = pat_cfg.get("types", ["line", "arc", "bend"])
        unsupported = set(self._patterns) - {"line", "arc", "bend"}
        if unsupported:
            raise ValueError(f"Unsupported domino-select patterns: {sorted(unsupported)}")
        variation_cfg = rand_cfg.get("layout_variations", {})
        self._shape_probability = variation_cfg.get("shape_probability", 0.75)
        self._size_probability = variation_cfg.get("size_probability", 0.65)
        self._interruption_probability = variation_cfg.get("interruption_probability", 0.65)
        self._network_defects = rand_cfg.get("interconnections", {}).get("defects", ["gap", "missing", "short"])
        if not self._network_defects or set(self._network_defects) - {"gap", "missing", "short"}:
            raise ValueError("Interconnection defects must be gap, missing, or short")
        from imbench.tasks.prehensile.domino_layouts import PATTERNS
        self._layout_patterns = list(rand_cfg.get("layout_patterns") or PATTERNS)
        unknown = set(self._layout_patterns) - set(PATTERNS)
        if unknown:
            raise ValueError(f"Unknown domino-select layout patterns: {sorted(unknown)}")
        self._layout_pattern = None
        self._decoy_starts = []
        self._starter = (0, 0)

        self._topple_tilt_threshold_deg = task_cfg.get("topple_tilt_threshold_deg", 45.0)
        self._robot_touched_chain = False
        self._decoy_toppled = False
        self._decoy_dominoes = []
        self._first_domino_toppled = False
        self._cascade_geoms = []
        self._link_touch_step = np.zeros(0, dtype=int)
        self._no_touch_geoms = set()
        self._parked = set()

        self._staged_reward = StagedReward(
            stages=[
                Stage("trigger_correct_chain", 0.30,
                      lambda t, o: 1.0 if t._first_domino_toppled else 0.0),
                Stage("cascade_progress_correct_chain", 0.30,
                      lambda t, o: 1.0 if t._cascade_reached_midpoint(t._correct_chain) else 0.0),
                Stage("target_displaced", 0.40,
                      lambda t, o: float(t._cascade_displaced_target())),
            ],
            success_fn=lambda t, o: t._cascade_displaced_target(),
        )

    def _make_objects(self):
        objects = []
        ds = np.asarray(self._domino_size) * self._SIZE_MAX
        for chain_idx in range(self._num_chains):
            for i in range(self._dominoes_per_chain):
                objects.append({
                    "type": "box",
                    "name": f"chain{chain_idx}_domino_{i}",
                    "size": [ds[2], ds[1], ds[0]],
                    "rgba": [0.8, 0.8, 0.2, 1.0],
                    "density": 800,
                    "friction": [0.8, 0.005, 0.0001],
                    "joints": "default",
                })
        # Shared target — thin slab that topples easily
        objects.append({
            "type": "box", "name": "red_block",
            "size": [0.005, 0.03, 0.04],  # thin slab standing upright
            "rgba": [0.9, 0.1, 0.1, 1.0],
            "density": 400,
            "friction": [0.5, 0.005, 0.0001],
            "joints": "default",
        })
        return objects

    def _resize_dominoes(self):
        sim = self._env.sim
        for k in range(self._num_chains):
            for i in range(self._dominoes_per_chain):
                sizes = self._chain_sizes[k, i]
                half_sizes = sizes[[2, 1, 0]]
                body = sim.model.body_name2id(f"chain{k}_domino_{i}_main")
                for geom in body_geom_ids(sim, f"chain{k}_domino_{i}_main"):
                    sim.model.geom_size[geom] = half_sizes
                mass = 800 * 8 * np.prod(half_sizes)
                set_body_mass(sim, body, mass, box_inertia(mass, half_sizes))
        commit_model_edits(sim)

    def _sample_chains(self):
        rng = self._rng
        if self._layout_mode == "patterns":
            n = self._dominoes_per_chain
            self._episode_variations = {"size": bool(rng.rand() < self._size_probability)}
            self._chain_sizes = np.tile(self._domino_size, (self._num_chains, n, 1)).astype(float)
            if self._episode_variations["size"]:
                self._chain_sizes *= rng.uniform(self._SIZE_MIN, self._SIZE_MAX, self._chain_sizes.shape)
            from imbench.tasks.prehensile.domino_layouts import apply_layout
            route = apply_layout(self)
            self._episode_variations["pattern"] = self._layout_pattern
            return route
        self._episode_variations = {
            "shape": bool(self._pattern_enabled and rng.rand() < self._shape_probability),
            "size": bool(rng.rand() < self._size_probability),
            "interruption": bool(rng.rand() < self._interruption_probability),
        }
        n = self._dominoes_per_chain
        self._chain_sizes = np.tile(self._domino_size, (self._num_chains, n, 1)).astype(float)
        if self._episode_variations["size"]:
            self._chain_sizes *= rng.uniform(self._SIZE_MIN, self._SIZE_MAX, self._chain_sizes.shape)
        if self._layout_mode == "merge":
            from imbench.tasks.prehensile.domino_network import merge_layout
            return merge_layout(self)
        # Adapt lane spacing for configurations with more chains; no per-lane
        # squashing of an already oriented curve or circle-distance fallback.
        lane_spacing = min(self._chain_y_spacing,
                           (min(self._table_full_size[:2]) - 0.10) / max(1, self._num_chains - 1))
        max_width = self._domino_size[1] * self._SIZE_MAX[1]
        max_height = self._domino_size[0] * self._SIZE_MAX[0]
        turn = min(np.deg2rad(20), max(0.0, lane_spacing - 2 * max_width - 0.012)
                   / (2 * max_height + 2 * self._correct_spacing))
        decoys = [k for k in rng.permutation(self._num_chains) if k != self._correct_chain]
        defects = ["gap", "missing", "short"] if self._episode_variations["interruption"] else ["gap", "missing"]
        specs = []
        yaw = self._LAYOUT_YAW
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        for k in range(self._num_chains):
            pattern = str(rng.choice(self._patterns)) if self._episode_variations["shape"] else "line"
            defect = "none" if k == self._correct_chain else defects[decoys.index(k) % len(defects)]
            # A straight interrupted decoy is always present when there are
            # multiple choices. Straight chains can also be the correct one.
            if decoys and k == decoys[0]:
                pattern = "line"
            split = int(rng.randint(2, max(3, n - 1))) if self._episode_variations["interruption"] else n // 2
            split = min(max(1, split), n - 2)
            links = np.full(n - 1, self._correct_spacing)
            hidden = -1
            if defect == "gap":
                links[split - 1] = 2 * max_height + 0.028
            elif defect == "short":
                # Two low blocks stop propagation without changing the total
                # path length. One low block alone is bridged by its tall
                # predecessor, so it is not a reliable physical decoy.
                split = min(split, n - 3)
                self._chain_sizes[k, split:split + 2, 0] = self._domino_size[0] * 0.23
                pattern = "line"
            elif defect == "missing":
                links[split - 1:split + 1] = max_height + 0.018
                hidden = split
            if pattern == "arc":
                headings = np.linspace(-turn, turn, n - 1)
            elif pattern == "bend":
                headings = np.where(np.arange(n - 1) < (n - 1) / 2, -turn, turn)
            else:
                headings = np.zeros(n - 1)
            headings *= rng.choice([-1, 1])
            points = np.vstack([np.zeros(2), np.cumsum(links[:, None] * np.column_stack([np.cos(headings), np.sin(headings)]), axis=0)])
            points[:, 1] -= (points[:, 1].min() + points[:, 1].max()) / 2
            points[:, 1] += (k - (self._num_chains - 1) / 2) * lane_spacing
            # Keep the last block (and the correct chain's target) on-table.
            limit = min(self._table_full_size[:2]) / 2 - 0.035
            reserve = self._correct_spacing if defect == "none" else 0
            start = min(rng.uniform(0.025, 0.055), limit - points[-1, 0] - reserve)
            points[:, 0] += start
            points = points @ rot.T
            headings = np.r_[headings, headings[-1]] + yaw
            layout = [(float(x), float(y), float(h)) for (x, y), h in zip(points, headings)]
            specs.append({"pattern": pattern, "defect": defect, "break_index": split,
                          "hidden_index": hidden, "layout": layout})
        self._chain_specs = specs
        self._resize_dominoes()
        for k, spec in enumerate(specs):
            for i, (x, y, heading) in enumerate(spec["layout"]):
                name = f"chain{k}_domino_{i}"
                if i == spec["hidden_index"]:
                    self._parked.add(name)
                else:
                    self._place_object(name, [x, y, self._table_height + self._chain_sizes[k, i, 0] + 0.002],
                                       quat=heading_to_quat(heading))
        self._park_objects(self._parked)
        return specs[self._correct_chain]["layout"]

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._staged_reward.reset()
        self._correct_chain = self._rng.randint(0, self._num_chains)
        self._parked = set()

        correct_layout = self._sample_chains()
        last_x, last_y, last_h = correct_layout[-1]
        self._place_object(
            "red_block", [last_x + self._correct_spacing * np.cos(last_h),
                          last_y + self._correct_spacing * np.sin(last_h),
                          self._table_height + 0.04 + 0.002],
            quat=heading_to_quat(last_h),
        )

        self._env.sim.forward()

        # Zero all object velocities to prevent drift from contact resolution
        for chain_idx in range(self._num_chains):
            for i in range(self._dominoes_per_chain):
                self._set_object_velocity(
                    f"chain{chain_idx}_domino_{i}", [0, 0, 0], [0, 0, 0]
                )
        self._set_object_velocity("red_block", [0, 0, 0], [0, 0, 0])

        # Randomize domino colors from the non-red palette. The target
        # block stays red so it remains visually distinct.
        if self._rand_object_color:
            for chain_idx in range(self._num_chains):
                for i in range(self._dominoes_per_chain):
                    color = DOMINO_COLORS[
                        self._rng.randint(len(DOMINO_COLORS))
                    ]
                    self._set_object_color(
                        f"chain{chain_idx}_domino_{i}", color
                    )
            self._env.sim.forward()

        try:
            target_id = self._env.sim.model.body_name2id("red_block_main")
            self._initial_target_pos = self._env.sim.data.body_xpos[
                target_id
            ].copy()
        except Exception:
            self._initial_target_pos = np.zeros(3)

        self._robot_touched_chain = False
        self._first_domino_toppled = False
        # Cascade order: each domino of the correct chain, then the target.
        c = self._correct_chain
        route = self._solution_route if self._interconnected else [
            f"chain{c}_domino_{i}" for i in range(self._dominoes_per_chain)]
        self._cascade_geoms = [
            body_geom_ids(self._env.sim, f"{name}_main")
            for name in route + ["red_block"]
        ]
        self._starter = tuple(int(v) for v in route[0].replace("chain", "").split("_domino_"))
        if self._layout_mode == "patterns":
            # Pools are assigned at random, so only the starter's pool is known.
            self._correct_chain = self._starter[0]
        else:
            self._decoy_starts = [f"chain{k}_domino_0" for k in range(self._num_chains)
                                  if f"chain{k}_domino_0" not in route + list(self._parked)]
        self._link_touch_step = np.full(len(route), -1)
        self._update_cascade_links()
        # Only the correct chain's first domino may be touched; the robot
        # contacting any later domino of that chain or the target
        # disqualifies the episode.
        self._no_touch_geoms = set().union(*self._cascade_geoms[1:])
        # Toppling any decoy domino standing at reset fails the episode, so
        # pushing over every chain does not pass for choosing the right one.
        self._decoy_toppled = False
        self._decoy_dominoes = [
            (k, j) for k in range(self._num_chains)
            for j in range(self._dominoes_per_chain)
            if f"chain{k}_domino_{j}" not in route
            if f"chain{k}_domino_{j}" not in self._parked
            and not self._domino_toppled(k, j)
        ]

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
        """Whether the robot touches the target or the correct chain past its first domino."""
        return robot_contacts(self._env.sim, self._no_touch_geoms)

    def _domino_toppled(self, chain_idx, index):
        """True once ``chain{chain_idx}_domino_{index}``'s tilt from vertical
        passes threshold."""
        sim = self._env.sim
        try:
            body_id = sim.model.body_name2id(f"chain{chain_idx}_domino_{index}_main")
        except Exception:
            return False
        xmat = sim.data.body_xmat[body_id].reshape(3, 3)
        body_z = xmat[:, 2]
        cos_angle = np.clip(abs(body_z[2]), 0.0, 1.0)
        tilt_deg = float(np.degrees(np.arccos(cos_angle)))
        return tilt_deg >= self._topple_tilt_threshold_deg

    def _cascade_reached_midpoint(self, chain_idx):
        """True once the midpoint domino of *chain_idx* has toppled."""
        if self._interconnected:
            name = self._solution_route[len(self._solution_route)//2]
            k, j = (int(v) for v in name.replace("chain", "").split("_domino_"))
            return self._domino_toppled(k, j)
        n = self._dominoes_per_chain
        if n <= 1:
            return self._domino_toppled(chain_idx, 0)
        return self._domino_toppled(chain_idx, n // 2)

    def _update_cascade_links(self):
        if not self._interconnected:
            return super()._update_cascade_links()
        # A contact is credited only after the preceding route link is
        # active. Incidental early junction touches cannot pre-credit a
        # downstream route. Same-step forward propagation remains valid.
        for k in range(len(self._link_touch_step)):
            if self._link_touch_step[k] >= 0:
                continue
            if k and self._link_touch_step[k-1] < 0:
                break
            if geoms_touch(self._env.sim, self._cascade_geoms[k], self._cascade_geoms[k+1]):
                self._link_touch_step[k] = self._step_count

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
        # hand instead of via the cascade, or toppling a decoy chain, is
        # disqualifying, regardless of any stage progress already accrued
        # this episode.
        if not self._robot_touched_chain and self._robot_touches_chain():
            self._robot_touched_chain = True
        if not self._decoy_toppled and any(
                self._domino_toppled(k, j) for k, j in self._decoy_dominoes):
            self._decoy_toppled = True
        if self._robot_touched_chain or self._decoy_toppled:
            self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
            return 0.0
        if self._domino_toppled(*self._starter):
            self._first_domino_toppled = True
        self._update_cascade_links()

        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        score = self._compute_task_score(raw_obs)
        if self._robot_touched_chain or self._decoy_toppled:
            return True
        if score >= self._success_score_threshold:
            return True
        return False
