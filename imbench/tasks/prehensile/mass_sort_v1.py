"""Adversarial Mass Sorting — sort identical-looking cubes by hidden mass.

Task ID: ``ib-p5-mass-sort-v1``
"""

import logging
import numpy as np
from imbench.tasks.physics import commit_model_edits, set_body_mass
from imbench.tasks.prehensile.contacts import object_geom_ids, robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase

logger = logging.getLogger(__name__)


class MassSortV1(SingleArmBase):
    TASK_ID = "ib-p5-mass-sort-v1"
    DESCRIPTION = (
        "Single arm: several identical-looking cubes have different hidden "
        "masses. Discover masses via lifting force and sort into zones."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._num_cubes = task_cfg.get("num_cubes", 3)
        self._cube_size = task_cfg.get("cube_size", 0.025)
        self._masses = task_cfg.get("masses", [0.05, 0.10, 0.20])
        self._zone_width = task_cfg.get("zone_width", 0.08)
        self._zone_spacing = task_cfg.get("zone_spacing", 0.12)
        self._mass_order = None  # randomized each episode

    # (name, rgba) — slot i gets entry i % len(CUBE_COLORS).
    CUBE_COLORS = [
        ("red",    [0.9, 0.2, 0.2, 1.0]),
        ("blue",   [0.2, 0.5, 0.9, 1.0]),
        ("yellow", [0.9, 0.7, 0.1, 1.0]),
        ("purple", [0.6, 0.2, 0.8, 1.0]),
        ("green",  [0.2, 0.8, 0.3, 1.0]),
        ("orange", [1.0, 0.5, 0.1, 1.0]),
    ]

    def _make_objects(self):
        objects = []
        vol = 8 * self._cube_size ** 3
        for i in range(self._num_cubes):
            mass = self._masses[i] if i < len(self._masses) else 0.10
            objects.append({
                "type": "box",
                "name": "cube" if i == 0 else f"cube_{i}",
                "size": [self._cube_size] * 3,
                "rgba": self.CUBE_COLORS[i % len(self.CUBE_COLORS)][1],
                "density": mass / vol,
                "friction": [0.8, 0.005, 0.0001],
                "joints": "default",
            })
        return objects

    def _make_static_objects(self):
        # Visual-only zone markers. Left → right is light → heavy;
        # green intensity scales *inversely* with mass so the lightest
        # cube's target zone is the brightest, and the heaviest is the
        # darkest. Matches operator intuition (lighter mass = lighter
        # shade).
        zones = []
        for i in range(self._num_cubes):
            intensity = 0.3 + 0.5 * (self._num_cubes - 1 - i) / max(1, self._num_cubes - 1)
            zones.append({
                "type": "box",
                "name": f"zone_{i}",
                "size": [self._zone_width / 2, self._zone_width / 2, 0.001],
                "rgba": [0.0, intensity, 0.0, 0.4],
                "density": 1, "friction": [0.1, 0.005, 0.0001],
                "joints": None,
            })
        return zones

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._last_outcome = None
        self._last_reason = None
        # Randomise mass assignment each episode
        self._mass_order = list(range(self._num_cubes))
        self._rng.shuffle(self._mass_order)

        sim = self._env.sim

        # Position zone markers in a row on the table (light→heavy, left→right)
        zone_xy = []
        for i in range(self._num_cubes):
            zone_x = (i - (self._num_cubes - 1) / 2) * self._zone_spacing
            zone_xy.append((zone_x, 0.15))
            self._place_static(f"zone_{i}", [
                zone_x, 0.15, self._table_height + 0.001,
            ])
        self._move_cubes_clear_of_zones(np.array(zone_xy))

        # Apply shuffled masses to the sim bodies, scaling inertia to match.
        for i in range(self._num_cubes):
            body_name = "cube_main" if i == 0 else f"cube_{i}_main"
            try:
                body_id = sim.model.body_name2id(body_name)
                set_body_mass(sim, body_id, self._masses[self._mass_order[i]])
            except Exception:
                pass
        commit_model_edits(sim)

        # Re-render obs after scene mutation (zone placement, mass
        # shuffle) so the first saved demo frame reflects the actual
        # episode layout, not the pre-mutation state from super().reset().
        obs = self._rebuild_obs()
        return obs, info

    def _cube_names(self):
        return ["cube" if i == 0 else f"cube_{i}" for i in range(self._num_cubes)]

    def _move_cubes_clear_of_zones(self, zone_xy, max_attempts=200):
        """Resample, with ``self._rng``, any cube spawned overlapping a zone.

        A cube is clear when no part of it can lie within a zone's scoring
        radius and its footprint cannot overlap another cube's. Cubes already
        clear keep the placement sampler's pose.
        """
        sim = self._env.sim
        half_diag = np.sqrt(2.0) * self._cube_size
        zone_clearance = self._zone_width / 2 + half_diag
        cube_clearance = 2 * half_diag
        placement = self.cfg.get("placement", {})
        bounds = []
        for key in ("x_range", "y_range"):
            r = placement.get(key, 0.0)
            bounds.append([-r, r] if isinstance(r, (int, float)) else list(r))
        lo, hi = np.array(bounds).T

        names = self._cube_names()
        joint_addrs = [
            sim.model.jnt_qposadr[sim.model.body_jntadr[sim.model.body_name2id(f"{n}_main")]]
            for n in names
        ]
        xy = np.array([sim.data.qpos[a:a + 2] for a in joint_addrs])

        def clear(i, p):
            if np.min(np.linalg.norm(zone_xy - p, axis=1)) <= zone_clearance:
                return False
            others = np.delete(xy, i, axis=0)
            return len(others) == 0 or np.min(np.linalg.norm(others - p, axis=1)) > cube_clearance

        for i, name in enumerate(names):
            if clear(i, xy[i]):
                continue
            for _ in range(max_attempts):
                p = self._rng.uniform(lo, hi)
                if clear(i, p):
                    break
            else:
                raise RuntimeError(
                    f"could not place {name} clear of the sorting zones in "
                    f"{max_attempts} attempts; widen the placement ranges"
                )
            xy[i] = p
            sim.data.qpos[joint_addrs[i]:joint_addrs[i] + 2] = p
            self._set_object_velocity(name, [0, 0, 0], [0, 0, 0])

    def _cube_zone_state(self):
        """Classify each cube as ``"correct"``, ``"wrong"``, or ``"unplaced"``.

        A cube is considered *placed* only when it rests on the table
        (centre at most 1 cm above its resting height, slower than
        ``REST_SPEED``, robot not touching it) and its xy position is
        inside some zone. Otherwise it's
        *unplaced* — still in play, being held or pushed by the robot,
        mid-transit, or simply not yet in any zone. Cubes that have fallen
        off the table are flagged separately via ``any_off_table``.

        Returns ``(per_cube_status, any_off_table)`` where
        ``per_cube_status[i] ∈ {"correct", "wrong", "unplaced"}``.
        """
        sim = self._env.sim
        sorted_masses = sorted(self._masses[self._mass_order[j]]
                               for j in range(self._num_cubes))

        per_cube_status = ["unplaced"] * self._num_cubes
        any_off_table = False

        for i, name in enumerate(self._cube_names()):
            try:
                body_id = sim.model.body_name2id(f"{name}_main")
                pos = sim.data.body_xpos[body_id]
            except Exception:
                continue

            if pos[2] < self._failure_floor_threshold:
                any_off_table = True

            # Only cubes resting on the table can be "placed". Held, falling or
            # sliding cubes stay "unplaced" even if their xy overlaps a zone.
            on_table = (
                self._table_height - self._cube_size
                <= pos[2]
                <= self._table_height + self._cube_size + 0.01
            )
            if (not on_table or self._object_speed(name) >= self.REST_SPEED
                    or robot_contacts(sim, object_geom_ids(self._env, name))):
                continue

            target_zone = sorted_masses.index(self._masses[self._mass_order[i]])

            for z in range(self._num_cubes):
                try:
                    zone_id = sim.model.body_name2id(f"zone_{z}_main")
                except Exception:
                    continue
                zone_pos = sim.data.body_xpos[zone_id]
                if np.linalg.norm(pos[:2] - zone_pos[:2]) < self._zone_width / 2:
                    per_cube_status[i] = "correct" if z == target_zone else "wrong"
                    break  # each cube can overlap at most one zone

        return per_cube_status, any_off_table

    def _compute_task_score(self, raw_obs):
        """Score = fraction of cubes currently in their correct zone."""
        statuses, _ = self._cube_zone_state()
        return sum(s == "correct" for s in statuses) / max(1, self._num_cubes)

    def _check_termination(self, raw_obs):
        """Terminate only on full success or any wrong placement / off-table.

        - All cubes ``"correct"``  → success, reward 1.
        - Any cube  ``"wrong"``    → failure, reward 0.
        - Any cube off the table   → failure, reward 0.
        - Mix of ``"correct"`` and ``"unplaced"`` → keep going.
        """
        statuses, any_off_table = self._cube_zone_state()
        if all(s == "correct" for s in statuses):
            self._last_outcome = "success"
            self._last_reason = "all cubes placed in their correct zones"
            return True
        if any_off_table:
            self._last_outcome = "failure"
            self._last_reason = (
                f"a cube fell off the table (z < {self._failure_floor_threshold})"
            )
            return True
        if any(s == "wrong" for s in statuses):
            wrong_idxs = [i for i, s in enumerate(statuses) if s == "wrong"]
            colors = [self.CUBE_COLORS[i % len(self.CUBE_COLORS)][0] for i in wrong_idxs]
            self._last_outcome = "failure"
            self._last_reason = f"wrong-zone placement: {', '.join(colors)}"
            return True
        self._last_outcome = None
        self._last_reason = None
        return False

    def step(self, action):
        """Step and add the terminal ``outcome``/``reason`` to ``info``.

        ``task_score`` is the fraction of cubes in their correct zone; the
        reward follows ``reward_type`` like every other task.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        info["outcome"] = self._last_outcome
        info["reason"] = self._last_reason
        if terminated:
            # Dump the per-cube ground-truth mapping so wrong-zone
            # failures are self-explanatory.
            statuses, _ = self._cube_zone_state()
            sim = self._env.sim
            sorted_masses = sorted(self._masses[self._mass_order[j]]
                                   for j in range(self._num_cubes))
            details = []
            for i in range(self._num_cubes):
                color = self.CUBE_COLORS[i % len(self.CUBE_COLORS)][0]
                mass_g = int(round(self._masses[self._mass_order[i]] * 1000))
                target = sorted_masses.index(self._masses[self._mass_order[i]])
                # Find which zone (if any) this cube is inside, and its
                # distance to the *target* zone so near-misses are visible.
                bname = "cube_main" if i == 0 else f"cube_{i}_main"
                in_zone = "-"
                d_to_target = float("nan")
                z_rel_to_table = float("nan")
                try:
                    bid = sim.model.body_name2id(bname)
                    cpos = sim.data.body_xpos[bid]
                    z_rel_to_table = cpos[2] - self._table_height
                    for z in range(self._num_cubes):
                        zpos = sim.data.body_xpos[sim.model.body_name2id(f"zone_{z}_main")]
                        if np.linalg.norm(cpos[:2] - zpos[:2]) < self._zone_width / 2:
                            in_zone = str(z)
                            break
                    tgt_pos = sim.data.body_xpos[sim.model.body_name2id(f"zone_{target}_main")]
                    d_to_target = float(np.linalg.norm(cpos[:2] - tgt_pos[:2]))
                except Exception:
                    in_zone = "?"
                details.append(
                    f"{color}({mass_g}g)→target=zone_{target} in=zone_{in_zone} "
                    f"d={d_to_target*100:.1f}cm z={z_rel_to_table*100:.1f}cm "
                    f"[{statuses[i]}]"
                )
            logger.info(
                "[MassSortV1] step=%d  reward=%.1f  outcome=%s  reason: %s\n    cubes: %s",
                self._step_count, reward, self._last_outcome,
                self._last_reason, " | ".join(details),
            )
        elif truncated:
            logger.info(
                "[MassSortV1] step=%d  reward=%.1f  outcome=truncated  "
                "reason: hit max_episode_steps (%d), task_score=%.3f",
                self._step_count, reward, self.MAX_EPISODE_STEPS,
                info.get("task_score", 0.0),
            )
        return obs, reward, terminated, truncated, info
