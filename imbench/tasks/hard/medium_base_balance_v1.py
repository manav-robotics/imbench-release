"""Medium-Base Balance — balance an elongated rod vertically on a narrow ridge.

Task ID: ``ib-p7-balance-medium-v1``
"""

import numpy as np
from imbench.objects import RIDGE_SOLREF
from imbench.tasks.physics import commit_model_edits, set_body_mass
from imbench.tasks.prehensile.contacts import robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


#: A rod rotating faster than this (rad/s) is falling, not balanced.
_BALANCED_MAX_ANG_SPEED = 0.05


class MediumBaseBalanceV1(SingleArmBase):
    TASK_ID = "ib-p7-balance-medium-v1"
    DESCRIPTION = (
        "Single arm: balance an elongated rod (pen/rod) vertically on a "
        "narrow perpendicular base. The rod must remain balanced for a "
        "sustained duration. Tactile feedback is critical for detecting "
        "tilt and making micro-corrections."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._rod_radius = task_cfg.get("rod_radius", 0.008)
        self._rod_half_length = task_cfg.get("rod_half_length", 0.12)
        self._rod_mass = task_cfg.get("rod_mass", 0.05)
        self._rod_friction = tuple(task_cfg.get("rod_friction", [1.0, 0.1, 0.01]))

        # Platform geometry
        self._ridge_half_length = task_cfg.get("ridge_half_length", 0.06)
        self._ridge_half_width = task_cfg.get("ridge_half_width", 0.003)
        self._ridge_height = task_cfg.get("ridge_height", 0.02)
        self._base_half_size = tuple(task_cfg.get("base_half_size", [0.04, 0.04, 0.005]))
        self._platform_friction = tuple(task_cfg.get("platform_friction", [1.0, 0.005, 0.0001]))

        # Scoring thresholds
        self._tilt_threshold_deg = task_cfg.get("tilt_threshold_deg", 15.0)
        self._balance_hold_steps = task_cfg.get("balance_hold_steps", 50)

        # Randomization
        rand_cfg = self.cfg.get("randomization", {})
        rod_rand = rand_cfg.get("rod", {})
        self._rand_rod_mass = rod_rand.get("mass_range", None)
        # Absolute XY sampling bounds on the table (meters, table-center frame).
        self._rand_rod_x_range = rod_rand.get("x_range", None)
        self._rand_rod_y_range = rod_rand.get("y_range", None)
        plat_rand = rand_cfg.get("platform", {})
        self._rand_plat_x_range = plat_rand.get("x_range", None)
        self._rand_plat_y_range = plat_rand.get("y_range", None)

        self._balance_counter = 0
        self._balance_grace = 0
        self._gripper_has_released = False

        # Graded balance reward through the shared StagedReward engine:
        # both stages are non-sticky (they reflect the *current* instant,
        # not a running max) and are gated by `_is_balanced()`, so the
        # two-branch behaviour (`0.8 + 0.2*progress` while balanced, `0.0`
        # otherwise) falls out of the weighted sum: 0.8*is_balanced +
        # 0.2*(progress if is_balanced else 0). `success_fn` saturates the
        # score to 1.0 on a completed hold window.
        self._staged_reward = StagedReward(
            stages=[
                Stage("is_balanced_instant", 0.8,
                      lambda t, o: 1.0 if t._is_balanced() else 0.0,
                      sticky=False),
                Stage("hold_duration_progress", 0.2,
                      lambda t, o: (t._balance_counter / max(1, t._balance_hold_steps))
                      if t._is_balanced() else 0.0,
                      sticky=False),
            ],
            success_fn=lambda t, o: t._balance_counter >= t._balance_hold_steps,
        )

    def _make_objects(self):
        # Square cross-section rod: full side = 2 * rod_radius.
        # A square profile gives the rod a stable flat face on the ridge,
        # so small tilts do not let it roll to the opposite side.
        vol = (2 * self._rod_radius) ** 2 * (2 * self._rod_half_length)
        density = self._rod_mass / vol if vol > 0 else 500.0
        return [
            {
                "type": "box", "name": "rod",
                "size": [self._rod_radius, self._rod_radius, self._rod_half_length],
                "rgba": [0.65, 0.45, 0.2, 1.0],
                "density": density,
                "friction": list(self._rod_friction),
                "joints": "default",
            },
        ]

    def _make_static_objects(self):
        return [
            {
                "type": "balance_platform", "name": "platform",
                "ridge_half_length": self._ridge_half_length,
                "ridge_half_width": self._ridge_half_width,
                "ridge_height": self._ridge_height,
                "base_half_size": list(self._base_half_size),
                "friction": list(self._platform_friction),
                "joints": None,
            },
        ]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        sim = self._env.sim
        self._balance_counter = 0
        self._balance_grace = 0
        self._gripper_has_released = False
        self._staged_reward.reset()

        # ── Randomize rod mass ──────────────────────────────────
        rod_mass = self._rod_mass
        if self._rand_rod_mass:
            rod_mass = self._rng.uniform(*self._rand_rod_mass)

        set_body_mass(sim, sim.model.body_name2id("rod_main"), rod_mass)

        # ── Sample non-overlapping positions for platform and rod ─────
        # Platform: ridge runs along world Y (rotated 90° about Z), so its
        # long footprint is ridge_half_length along Y and base_half_size
        # along X. Rod spawns lying flat along world X (rotate 90° about Y),
        # so its footprint is rod_half_length along X and rod_radius along Y.
        plat_half_x = self._base_half_size[0]
        plat_half_y = max(self._ridge_half_length, self._base_half_size[1])
        rod_half_x = self._rod_half_length
        rod_half_y = self._rod_radius
        # Minimum center-to-center separation along each axis to guarantee
        # the two AABBs don't overlap, plus a small clearance.
        clearance = 0.02
        min_sep_x = plat_half_x + rod_half_x + clearance
        min_sep_y = plat_half_y + rod_half_y + clearance

        plat_x_range = self._rand_plat_x_range or (0.10, 0.10)
        plat_y_range = self._rand_plat_y_range or (0.0, 0.0)
        rod_x_range = self._rand_rod_x_range or (0.0, 0.0)
        rod_y_range = self._rand_rod_y_range or (-0.12, -0.12)

        platform_xy = (0.10, 0.0)
        rod_xy = (0.0, -0.12)
        for _ in range(100):
            platform_xy = (
                self._rng.uniform(*plat_x_range),
                self._rng.uniform(*plat_y_range),
            )
            rod_xy = (
                self._rng.uniform(*rod_x_range),
                self._rng.uniform(*rod_y_range),
            )
            if (
                abs(platform_xy[0] - rod_xy[0]) >= min_sep_x
                or abs(platform_xy[1] - rod_xy[1]) >= min_sep_y
            ):
                break

        platform_quat = [0.7071068, 0.0, 0.0, 0.7071068]  # 90° about Z
        self._place_static("platform", [
            platform_xy[0], platform_xy[1], self._table_height,
        ], quat=platform_quat)

        rod_z = self._table_height + self._rod_radius + 0.005
        quat_lying = [0.7071068, 0.0, 0.7071068, 0.0]  # 90° about Y
        self._place_object("rod", [rod_xy[0], rod_xy[1], rod_z], quat=quat_lying)
        self._set_object_velocity("rod", [0, 0, 0], [0, 0, 0])

        # Enable torsional + rolling friction on the rod only.
        # MuJoCo ignores torsional/rolling friction unless condim ≥ 4.
        # Setting condim=4 on the table multiplies contact count with the
        # 128 tactile taxels and can overflow MuJoCo's contact buffer.
        for i in range(sim.model.ngeom):
            name = sim.model.geom_id2name(i)
            if name and "rod" in name and "vis" not in name:
                sim.model.geom_condim[i] = 4
                sim.model.geom_friction[i] = list(self._rod_friction)
                sim.model.geom_solref[i] = RIDGE_SOLREF

        self._cache_contact_geom_ids()
        commit_model_edits(sim)
        obs = self._rebuild_obs()
        return obs, info

    def _cache_contact_geom_ids(self):
        """Resolve the geom ids the balance predicate inspects (sim is rebuilt every reset)."""
        sim = self._env.sim
        self._rod_geom_ids = {
            sim.model.geom_name2id(g) for g in self._env.get_object("rod").contact_geoms
        }
        self._ridge_geom_id = sim.model.geom_name2id("platform_ridge")

    # ── Observations ────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)
        sim = self._env.sim

        # Rod pose (aliased as cube_pos/cube_quat by env, but expose
        # explicitly under canonical names too)
        if "rod_pos" in raw_obs:
            obs["object_pos"] = raw_obs["rod_pos"]
        elif "cube_pos" in raw_obs:
            obs["object_pos"] = raw_obs["cube_pos"]
        if "rod_quat" in raw_obs:
            obs["object_quat"] = raw_obs["rod_quat"]
        elif "cube_quat" in raw_obs:
            obs["object_quat"] = raw_obs["cube_quat"]

        # Platform (static) position — read directly from sim
        plat_id = sim.model.body_name2id("platform_main")
        plat_pos = np.array(sim.data.body_xpos[plat_id])
        obs["platform_pos"] = plat_pos

        # Rod tilt from vertical (degrees) — key signal for balancing
        tilt_deg = self._get_rod_tilt_deg()
        obs["rod_tilt_deg"] = np.array([tilt_deg], dtype=np.float64)

        # Rod body-Z axis projected into world frame (unit vector
        # indicating current orientation of the rod's long axis)
        rod_id = sim.model.body_name2id("rod_main")
        xmat = sim.data.body_xmat[rod_id].reshape(3, 3)
        obs["rod_axis"] = xmat[:, 2].copy()

        # Rod-to-platform horizontal distance
        if "object_pos" in obs:
            obs["rod_to_platform"] = np.array([
                np.linalg.norm(obs["object_pos"][:2] - plat_pos[:2])
            ], dtype=np.float64)

        return obs

    # ── Scoring ─────────────────────────────────────────────────────

    def _get_rod_tilt_deg(self):
        """Return the angle (degrees) between the rod's long axis and vertical.

        The rod's long axis is its local Z. ``arccos(|body_z[2]|)`` gives 0°
        for an upright rod and 90° for one lying flat; ``abs()`` makes the
        two ends equivalent and rotation about the long axis is ignored.
        """
        sim = self._env.sim
        body_id = sim.model.body_name2id("rod_main")
        xmat = sim.data.body_xmat[body_id].reshape(3, 3)
        body_z = xmat[:, 2]  # local Z (long axis) in world frame
        cos_angle = np.clip(abs(body_z[2]), 0.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _rod_supported_by_ridge_only(self):
        """True iff the rod touches the ridge and nothing else.

        Contact with the base plate, the table or any robot geom (hand,
        finger, link) means the rod is not freely balanced on the ridge.
        """
        sim = self._env.sim
        on_ridge = False
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            if c.geom1 in self._rod_geom_ids:
                other = c.geom2
            elif c.geom2 in self._rod_geom_ids:
                other = c.geom1
            else:
                continue
            if other != self._ridge_geom_id:
                return False
            on_ridge = True
        return on_ridge

    def _is_balanced(self):
        """Rod upright, standing on the ridge alone, and not falling."""
        return (
            self._get_rod_tilt_deg() <= self._tilt_threshold_deg
            and self._rod_supported_by_ridge_only()
            and np.linalg.norm(self._body_velocity(self._object_body_id("rod"))[3:]) < _BALANCED_MAX_ANG_SPEED
        )

    def _update_balance_counter(self):
        # The stability window only opens once the gripper has let go of the
        # rod at least once. Re-grabbing the rod after release is treated as
        # a hard reset — the agent must release again and restart the count.
        gripper_on_rod = self._gripper_contacts_object("rod")
        if not self._gripper_has_released and not gripper_on_rod:
            self._gripper_has_released = True

        if self._gripper_has_released and gripper_on_rod:
            self._balance_counter = 0
            self._balance_grace = 0
            self._gripper_has_released = False
            return

        if not self._gripper_has_released:
            return

        # Small grace window absorbs momentary tilt/contact flicker so a
        # single noisy step doesn't wipe out a held balance. A step on which
        # the robot touches the rod is support, not flicker: no grace.
        if self._is_balanced():
            self._balance_counter += 1
            self._balance_grace = 0
        elif robot_contacts(self._env.sim, self._rod_geom_ids):
            self._balance_counter = 0
            self._balance_grace = 0
        else:
            self._balance_grace += 1
            if self._balance_grace > 5:
                self._balance_counter = 0

    def _compute_task_score(self, raw_obs):
        """Graded reward for balancing the rod, via the shared StagedReward
        engine (see `_parse_task_config` for how the two stages combine):

        - 1.0 once the rod has been balanced for the full hold window.
        - 0.8..1.0 (linear in counter) while currently balanced but the hold
          window isn't yet complete — gives the agent credit for holding the
          target pose, so a single upright frame already scores >= 0.8.
        - 0.0 otherwise.

        Updates the counter here (before termination check) so the success
        step itself returns 1.0 rather than 0.0.
        """
        self._update_balance_counter()
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        """Terminate when rod has been balanced for the hold window, or fell off table."""
        if self._balance_counter >= self._balance_hold_steps:
            return True
        return self._object_fell_off_table(raw_obs, pos_key="rod_pos")
