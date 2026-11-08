"""Shape Stack — stack differently-shaped objects until only one sits on the table.

Task ID: ``ib-p7-shape-stack-v1``

Each reset picks a random subset of archetypes from the catalogue below and
places them on the table. Success: exactly one object touches the table,
nothing is held by the gripper, and the arrangement is stable for
``stable_steps`` consecutive ticks.
"""

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward
from imbench.tasks.puzzle._stack_scoring import (
    object_body_id,
    count_table_and_gripper_contacts,
    stacking_progress,
    hold_duration_ramp,
)


# ---------------------------------------------------------------------------
# Catalogue of all stackable object archetypes
# ---------------------------------------------------------------------------
# Each entry:
#   name          — unique internal name
#   type          — robosuite object type ("box", "cylinder", "composite")
#   size          — primitive size (ignored for composites)
#   half_h        — half-height, used to spawn the object resting on the table
#   rgba          — default colour
#   extra_spec    — optional dict merged into the object spec at build time,
#                   used for composites (sub_objects, sub_locations, ...)

def _hex_prism_spec(size, rgba):
    """3 interlocking rectangular boxes at 60° approximating a hexagonal prism."""
    face_half = size[0]     # outer hex radius
    height_h  = size[1]     # half-height
    slab_size = [face_half, face_half * 0.577, height_h]  # tan(30°)=0.577
    def quat_z(angle):
        return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]
    return {
        "sub_objects": [
            {"type": "box", "name": "slab0", "size": slab_size, "rgba": rgba},
            {"type": "box", "name": "slab1", "size": slab_size, "rgba": rgba},
            {"type": "box", "name": "slab2", "size": slab_size, "rgba": rgba},
        ],
        "sub_locations": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "sub_quats": [
            quat_z(0.0),
            quat_z(np.pi / 3),
            quat_z(2 * np.pi / 3),
        ],
        "total_size": [face_half, face_half, height_h],
    }


def _cone_frustum_spec(size, rgba):
    """Stepped truncated cone: 3 coaxial cylinders with decreasing radii."""
    r_base, h_total = size
    h_layer = h_total / 3.0
    return {
        "sub_objects": [
            {"type": "cylinder", "name": "c0", "size": [r_base,          h_layer], "rgba": rgba},
            {"type": "cylinder", "name": "c1", "size": [r_base * 0.75,   h_layer], "rgba": rgba},
            {"type": "cylinder", "name": "c2", "size": [r_base * 0.55,   h_layer], "rgba": rgba},
        ],
        "sub_locations": [
            [0, 0, -h_layer],
            [0, 0, 0.0],
            [0, 0, h_layer],
        ],
        "sub_quats": None,
        "total_size": [r_base, r_base, h_total],
    }


_ALL_ARCHETYPES = [
    {"name": "cylinder",   "type": "cylinder", "size": [0.025, 0.020],           "half_h": 0.020, "rgba": [0.2, 0.6, 0.8, 1.0]},
    {"name": "cube",       "type": "box",      "size": [0.025, 0.025, 0.025],    "half_h": 0.025, "rgba": [0.8, 0.3, 0.2, 1.0]},
    {"name": "small_cube", "type": "box",      "size": [0.018, 0.018, 0.018],    "half_h": 0.018, "rgba": [0.9, 0.5, 0.2, 1.0]},
    {"name": "hex_prism",    "type": "composite", "size": [0.028, 0.022],
     "half_h": 0.022, "rgba": [0.6, 0.3, 0.8, 1.0],
     "extra_spec": _hex_prism_spec([0.028, 0.022], [0.6, 0.3, 0.8, 1.0])},
    # Three overlapping cylinders of half-height 0.02 centred at -0.02/0/+0.02.
    {"name": "cone_frustum", "type": "composite", "size": [0.030, 0.060],
     "half_h": 0.040, "rgba": [0.9, 0.7, 0.2, 1.0],
     "extra_spec": _cone_frustum_spec([0.030, 0.060], [0.9, 0.7, 0.2, 1.0])},
]

# Mutual-exclusion groups: at most ONE archetype from each group may be
# present in a single episode. These are awkwardly-shaped pieces that make
# stacking unsolvable if combined.
_EXCLUSIVE_GROUPS: list[set[str]] = [
    {"hex_prism", "cone_frustum"},
]


class ShapeStackV1(SingleArmBase):
    TASK_ID = "ib-p7-shape-stack-v1"
    DESCRIPTION = (
        "Puzzle: stack a randomly-chosen set of differently-shaped objects "
        "so that only one remains on the table, held stable."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        rand_cfg = self.cfg.get("randomization", {})
        self._stable_steps_required = task_cfg.get("stable_steps", 100)
        num_obj_cfg = rand_cfg.get("num_objects", {})
        self._num_objects_range = num_obj_cfg.get("range", [3, 5])
        self._stable_count = 0
        self._num_touching_table = 3
        self._gripper_contact_now = False
        # Archetypes in play this episode; redrawn in reset().
        self._selected_archetypes = _ALL_ARCHETYPES[:3]

        # Same base + ramp pattern as the hold-style tasks
        # (medium_base_balance_*, ib-p6-stack-collapse-recovery-v1): dense
        # stacking progress plus a gripper-clear check are both
        # instantaneous (non-sticky), and the hold-duration ramp is what
        # actually requires sustaining the stacked arrangement for
        # `_stable_steps_required` ticks.
        self._staged_reward = StagedReward(
            stages=[
                Stage("stacking_progress", 0.5,
                      lambda t, o: t._stacking_progress_value(), sticky=False),
                # Paid only for a released stack, never for an untouched scene.
                Stage("gripper_clear", 0.1,
                      lambda t, o: 1.0 if t._is_stacked() else 0.0,
                      sticky=False),
                Stage("hold_duration_ramp", 0.4,
                      lambda t, o: hold_duration_ramp(t._stable_count, t._stable_steps_required)
                      if t._is_stacked() else 0.0,
                      sticky=False),
            ],
            success_fn=lambda t, o: t._stable_count >= t._stable_steps_required,
        )

    def _sample_archetypes(self, k: int) -> list[dict]:
        """Pick k distinct archetypes such that no mutual-exclusion group
        contributes more than one member. k is clamped if the constraint
        would make it infeasible."""
        names_all = [a["name"] for a in _ALL_ARCHETYPES]
        by_name = {a["name"]: a for a in _ALL_ARCHETYPES}
        in_any_group = set().union(*_EXCLUSIVE_GROUPS) if _EXCLUSIVE_GROUPS else set()
        non_group_count = sum(1 for n in names_all if n not in in_any_group)
        feasible_max = non_group_count + sum(
            1 for g in _EXCLUSIVE_GROUPS if any(n in names_all for n in g)
        )
        k = min(k, feasible_max)
        for _attempt in range(50):
            idx = self._rng.choice(len(names_all), size=k, replace=False)
            chosen = [names_all[i] for i in idx]
            if all(sum(1 for n in chosen if n in g) <= 1 for g in _EXCLUSIVE_GROUPS):
                return [by_name[n] for n in chosen]
        # Greedy fallback — picks one per exclusive group, then fills.
        chosen: list[str] = []
        pool = list(names_all)
        self._rng.shuffle(pool)
        for n in pool:
            if len(chosen) >= k:
                break
            if any(n in g and any(c in g for c in chosen) for g in _EXCLUSIVE_GROUPS):
                continue
            chosen.append(n)
        return [by_name[n] for n in chosen]

    def _make_objects(self):
        objects = []
        for arch in _ALL_ARCHETYPES:
            spec = {
                "type": arch["type"],
                "name": arch["name"],
                "size": arch["size"],
                "rgba": arch["rgba"],
                "density": 500,
                "friction": [0.85, 0.005, 0.0001],
                "joints": "default",
            }
            if arch.get("extra_spec"):
                spec.update(arch["extra_spec"])
            objects.append(spec)
        return objects

    def reset(self, *, seed=None, options=None):
        # The whole catalogue is always in the model, so the selection can
        # be drawn *after* the env reset and SingleArmBase.reset() keeps
        # sole ownership of RNG seeding. Unselected archetypes are parked.
        obs, info = super().reset(seed=seed, options=options)
        lo, hi = self._num_objects_range
        num_requested = int(self._rng.randint(lo, hi + 1))
        self._selected_archetypes = self._sample_archetypes(num_requested)

        self._stable_count = 0
        self._failed = False
        self._num_touching_table = len(self._selected_archetypes)
        self._gripper_contact_now = False
        self._staged_reward.reset()

        # Tighten the MuJoCo solver so stacked contacts don't buzz. Robosuite
        # defaults are fine for grasping but too loose for multi-body stacks.
        mj_opt = self._env.sim.model.opt
        mj_opt.timestep = 0.002
        mj_opt.iterations = 150
        mj_opt.noslip_iterations = 3
        mj_opt.noslip_tolerance = 1e-6

        placements = self._sample_placements(self._selected_archetypes)
        for i, arch in enumerate(self._selected_archetypes):
            x, y = placements[i]
            z = self._table_height + arch["half_h"]
            self._place_object(arch["name"], [x, y, z])
        self._park_unselected()

        self._env.sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ── Placement helpers ───────────────────────────────────────────

    @staticmethod
    def _archetype_footprint_radius(arch: dict) -> float:
        """Conservative xy-plane footprint half-width for spawn spacing."""
        t = arch["type"]
        size = arch.get("size", [])
        if t == "box" and len(size) >= 2:
            return float(max(size[0], size[1]))
        if t in ("cylinder", "ball") and len(size) >= 1:
            return float(size[0])
        if t == "composite":
            extra = arch.get("extra_spec", {}) or {}
            total = extra.get("total_size")
            if total and len(total) >= 2:
                # Bump by 50% — composite ``total_size`` reports one slab's
                # half-extent, but rotated slabs poke further outward.
                return float(max(total[0], total[1])) * 1.5
        return float(arch.get("half_h", 0.025))

    def _sample_placements(self, archetypes: list[dict]) -> list[tuple[float, float]]:
        """Rejection-sample xy positions inside the placement box, keeping
        every pair at least ``r_i + r_j + margin`` apart."""
        placement_cfg = self.cfg.get("placement", {}) or {}
        xr = placement_cfg.get("x_range", 0.12)
        yr = placement_cfg.get("y_range", 0.12)
        if isinstance(xr, (int, float)):
            xr = [-xr, xr]
        if isinstance(yr, (int, float)):
            yr = [-yr, yr]
        margin = float(placement_cfg.get("spawn_margin", 0.01))
        max_attempts = int(placement_cfg.get("spawn_max_attempts", 200))

        radii = [self._archetype_footprint_radius(a) for a in archetypes]
        # Place largest-first so tight boxes don't dead-end on a small one.
        order = sorted(range(len(archetypes)), key=lambda i: -radii[i])

        chosen: dict[int, tuple[float, float]] = {}
        for i in order:
            placed = False
            for _ in range(max_attempts):
                x = float(self._rng.uniform(xr[0], xr[1]))
                y = float(self._rng.uniform(yr[0], yr[1]))
                if all(
                    (x - xj) ** 2 + (y - yj) ** 2 >= (radii[i] + radii[j] + margin) ** 2
                    for j, (xj, yj) in chosen.items()
                ):
                    chosen[i] = (x, y)
                    placed = True
                    break
            if not placed:
                # Rejection sampling failed — fall back to an even ring sized
                # so neighbours clear ``2 * r_max + margin``.
                n = len(archetypes)
                r_max = max(radii)
                required = 2 * r_max + margin
                ring_r = required / (2 * np.sin(np.pi / n)) if n >= 2 else 0.0
                angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                return [(ring_r * np.cos(a), ring_r * np.sin(a)) for a in angles]
        return [chosen[i] for i in range(len(archetypes))]

    # ── Parked (unselected) archetypes ─────────────────────────────

    def _park_unselected(self):
        """Park every catalogue archetype not in play this episode.

        Exposed separately from ``reset()`` so callers that override
        ``_selected_archetypes`` directly can re-park to match.
        """
        selected = {a["name"] for a in self._selected_archetypes}
        self._parked = {a["name"] for a in _ALL_ARCHETYPES
                        if a["name"] not in selected}
        self._park_objects(self._parked)

    def _parked_objects(self):
        return getattr(self, "_parked", set())

    # ── Model introspection ────────────────────────────────────────

    def _any_object_fell(self) -> bool:
        sim = self._env.sim
        for a in self._selected_archetypes:
            bid = object_body_id(sim, a["name"])
            if bid is None:
                continue
            if sim.data.body_xpos[bid][2] < self._failure_floor_threshold:
                return True
        return False

    # ── Scoring ────────────────────────────────────────────────────

    def _update_stack_contact_state(self):
        """Refresh ``_num_touching_table`` / ``_gripper_contact_now`` for
        this tick and advance ``_stable_count`` accordingly."""
        names = [a["name"] for a in self._selected_archetypes]
        n_touching, gripper_contact = count_table_and_gripper_contacts(
            self._env, names,
        )
        self._num_touching_table = n_touching
        self._gripper_contact_now = gripper_contact

        if self._is_stacked():
            self._stable_count += 1
        else:
            self._stable_count = 0

    def _is_stacked(self) -> bool:
        """True iff exactly one object touches the table and the gripper
        is clear — the instantaneous (pre-hold) success precondition."""
        return self._num_touching_table == 1 and not self._gripper_contact_now

    def _stacking_progress_value(self) -> float:
        return stacking_progress(self._num_touching_table, len(self._selected_archetypes))

    def _compute_task_score(self, raw_obs):
        """Graded reward via the shared StagedReward engine: dense stacking
        progress + a gripper-clear check, saturating to 1.0 once the fully
        stacked arrangement has held for ``_stable_steps_required``
        consecutive ticks (see class docstring)."""
        if getattr(self, "_failed", False):
            return 0.0

        self._update_stack_contact_state()
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if getattr(self, "_failed", False):
            return True
        if self._any_object_fell():
            self._failed = True
            return True
        score = self._compute_task_score(raw_obs)
        return score >= self._success_score_threshold
