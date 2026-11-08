"""Random-Angle Balance — OOD variant of ib-p7-balance-medium-v1.

Each episode the platform ridge and the rod are oriented at an angle
sampled uniformly over the full circle about the world Z-axis.  This
prevents any fixed-orientation policy from succeeding and forces
generalisation over arbitrary in-plane rotations.  Both share the one
angle, so the rod spawns parallel to the ridge (v1 spawns it
perpendicular).

Task ID: ``ib-p7-balance-medium-v2``
"""

import numpy as np
from imbench.objects import RIDGE_SOLREF
from imbench.tasks.physics import commit_model_edits, set_body_mass
from imbench.tasks.prehensile.contacts import robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def _quat_from_axis_angle(axis, angle_rad):
    """Build a [w, x, y, z] unit quaternion from an axis + angle."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2.0)
    return np.array([np.cos(angle_rad / 2.0), *(axis * s)])


def _quat_mul(q1, q2):
    """Hamilton product of two [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


# Fixed component: rod lies along world X before the in-plane rotation.
_ROD_LIE_FLAT = _quat_from_axis_angle([0, 1, 0], np.pi / 2)   # 90° about Y


def _make_platform_quat(angle_rad):
    """Platform quaternion: rotate the ridge by `angle_rad` about world Z."""
    return _quat_from_axis_angle([0, 0, 1], angle_rad)


def _make_rod_quat(angle_rad):
    """Rod quaternion: lie flat along world X, then rotate by `angle_rad` about Z."""
    rot_z = _quat_from_axis_angle([0, 0, 1], angle_rad)
    return _quat_mul(rot_z, _ROD_LIE_FLAT)


#: A rod rotating faster than this (rad/s) is falling, not balanced.
_BALANCED_MAX_ANG_SPEED = 0.05


class MediumBaseBalanceV2(SingleArmBase):
    TASK_ID = "ib-p7-balance-medium-v2"
    DESCRIPTION = (
        "OOD variant of ib-p7-balance-medium-v1: each episode the platform "
        "ridge and rod are oriented at a freshly sampled random angle about "
        "the world Z-axis, requiring policies to generalise across arbitrary "
        "in-plane orientations."
    )

    # ── Config ──────────────────────────────────────────────────────

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._rod_radius      = task_cfg.get("rod_radius",      0.008)
        self._rod_half_length = task_cfg.get("rod_half_length", 0.12)
        self._rod_mass        = task_cfg.get("rod_mass",        0.05)
        self._rod_friction    = tuple(task_cfg.get("rod_friction", [1.0, 0.1, 0.01]))

        # Platform geometry
        self._ridge_half_length = task_cfg.get("ridge_half_length", 0.06)
        self._ridge_half_width  = task_cfg.get("ridge_half_width",  0.003)
        self._ridge_height      = task_cfg.get("ridge_height",      0.02)
        self._base_half_size    = tuple(task_cfg.get("base_half_size", [0.04, 0.04, 0.005]))
        self._platform_friction = tuple(task_cfg.get("platform_friction", [1.0, 0.005, 0.0001]))

        # Scoring thresholds
        self._tilt_threshold_deg = task_cfg.get("tilt_threshold_deg", 15.0)
        self._balance_hold_steps = task_cfg.get("balance_hold_steps", 50)

        # Spawn position randomization
        rand_cfg  = self.cfg.get("randomization", {})
        rod_rand  = rand_cfg.get("rod", {})
        self._rand_rod_mass    = rod_rand.get("mass_range",        None)
        self._rand_rod_x_range = rod_rand.get("x_range",          None)
        self._rand_rod_y_range = rod_rand.get("y_range",          None)
        plat_rand = rand_cfg.get("platform", {})
        self._rand_plat_x_range = plat_rand.get("x_range", None)
        self._rand_plat_y_range = plat_rand.get("y_range", None)

        self._balance_counter      = 0
        self._balance_grace        = 0
        self._gripper_has_released = False
        self._episode_angle_deg    = 0.0   # sampled platform angle, exposed in obs

        # See MediumBaseBalanceV1._parse_task_config for why both stages
        # are non-sticky and gated by `_is_balanced()`, giving the
        # `0.8 + 0.2*progress` / `0.0` two-branch reward.
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

    # ── Object factories ─────────────────────────────────────────────

    def _make_objects(self):
        vol     = (2 * self._rod_radius) ** 2 * (2 * self._rod_half_length)
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
                "ridge_half_width":  self._ridge_half_width,
                "ridge_height":      self._ridge_height,
                "base_half_size":    list(self._base_half_size),
                "friction":          list(self._platform_friction),
                "joints":            None,
            },
        ]

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        sim = self._env.sim
        self._balance_counter      = 0
        self._balance_grace        = 0
        self._gripper_has_released = False
        self._staged_reward.reset()

        # ── Sample episode angle uniformly over full circle ──────
        angle_rad = self._rng.uniform(0.0, 2 * np.pi)
        self._episode_angle_deg = np.degrees(angle_rad)

        plat_quat = _make_platform_quat(angle_rad)   # shape (4,) [w,x,y,z]
        rod_quat  = _make_rod_quat(angle_rad)

        # ── Randomize rod mass ────────────────────────────────────
        rod_mass = self._rod_mass
        if self._rand_rod_mass:
            rod_mass = self._rng.uniform(*self._rand_rod_mass)

        set_body_mass(sim, sim.model.body_name2id("rod_main"), rod_mass)

        # ── Non-overlapping spawn positions ──────────────────────
        # At an arbitrary angle θ the AABB half-extent of a rectangle with
        # half-lengths (a, b) is at most hypot(a, b) <= max(a, b)*sqrt(2).
        # These angle-independent bounds keep the clearance check simple.
        plat_aabb_half = max(
            self._base_half_size[0],
            self._base_half_size[1],
            self._ridge_half_length,
        ) * np.sqrt(2.0)
        rod_aabb_half = np.hypot(self._rod_half_length, self._rod_radius)

        clearance = 0.02
        min_sep   = plat_aabb_half + rod_aabb_half + clearance

        plat_x_range = self._rand_plat_x_range or (0.10,  0.10)
        plat_y_range = self._rand_plat_y_range or (0.0,   0.0)
        rod_x_range  = self._rand_rod_x_range  or (0.0,   0.0)
        rod_y_range  = self._rand_rod_y_range  or (-0.15, -0.15)

        platform_xy = (0.10,  0.0)
        rod_xy      = (0.0,  -0.15)
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
                abs(platform_xy[0] - rod_xy[0]) >= min_sep
                or abs(platform_xy[1] - rod_xy[1]) >= min_sep
            ):
                break

        # ── Place objects ─────────────────────────────────────────
        self._place_static(
            "platform",
            [platform_xy[0], platform_xy[1], self._table_height],
            quat=list(plat_quat),
        )

        rod_z = self._table_height + self._rod_radius + 0.005
        self._place_object(
            "rod",
            [rod_xy[0], rod_xy[1], rod_z],
            quat=list(rod_quat),
        )
        self._set_object_velocity("rod", [0, 0, 0], [0, 0, 0])

        # Enable torsional + rolling friction on rod geoms.
        for i in range(sim.model.ngeom):
            name = sim.model.geom_id2name(i)
            if name and "rod" in name and "vis" not in name:
                sim.model.geom_condim[i]   = 4
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

    # ── Observations ─────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)
        sim = self._env.sim

        if "rod_pos" in raw_obs:
            obs["object_pos"] = raw_obs["rod_pos"]
        elif "cube_pos" in raw_obs:
            obs["object_pos"] = raw_obs["cube_pos"]
        if "rod_quat" in raw_obs:
            obs["object_quat"] = raw_obs["rod_quat"]
        elif "cube_quat" in raw_obs:
            obs["object_quat"] = raw_obs["cube_quat"]

        plat_id  = sim.model.body_name2id("platform_main")
        plat_pos = np.array(sim.data.body_xpos[plat_id])
        obs["platform_pos"] = plat_pos

        tilt_deg = self._get_rod_tilt_deg()
        obs["rod_tilt_deg"] = np.array([tilt_deg], dtype=np.float64)

        rod_id = sim.model.body_name2id("rod_main")
        xmat   = sim.data.body_xmat[rod_id].reshape(3, 3)
        obs["rod_axis"] = xmat[:, 2].copy()

        # Expose the current episode angle so the policy can condition on it
        # if desired (e.g. during fine-tuning or curriculum).
        obs["episode_angle_deg"] = np.array(
            [self._episode_angle_deg], dtype=np.float64
        )

        if "object_pos" in obs:
            obs["rod_to_platform"] = np.array([
                np.linalg.norm(obs["object_pos"][:2] - plat_pos[:2])
            ], dtype=np.float64)

        return obs

    # ── Scoring ──────────────────────────────────────────────────────

    def _get_rod_tilt_deg(self):
        """Angle (degrees) between the rod's long axis and vertical; 0° = upright."""
        sim     = self._env.sim
        body_id = sim.model.body_name2id("rod_main")
        xmat    = sim.data.body_xmat[body_id].reshape(3, 3)
        body_z  = xmat[:, 2]
        cos_angle = np.clip(abs(body_z[2]), 0.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _rod_supported_by_ridge_only(self):
        """True iff the rod touches the ridge and nothing else (base plate, table, robot)."""
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
        gripper_on_rod = self._gripper_contacts_object("rod")
        if not self._gripper_has_released and not gripper_on_rod:
            self._gripper_has_released = True

        if self._gripper_has_released and gripper_on_rod:
            self._balance_counter      = 0
            self._balance_grace        = 0
            self._gripper_has_released = False
            return

        if not self._gripper_has_released:
            return

        # Grace covers tilt/contact flicker; robot support resets the hold.
        if self._is_balanced():
            self._balance_counter += 1
            self._balance_grace    = 0
        elif robot_contacts(self._env.sim, self._rod_geom_ids):
            self._balance_counter = 0
            self._balance_grace   = 0
        else:
            self._balance_grace += 1
            if self._balance_grace > 5:
                self._balance_counter = 0

    def _compute_task_score(self, raw_obs):
        self._update_balance_counter()
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._balance_counter >= self._balance_hold_steps:
            return True
        return self._object_fell_off_table(raw_obs, pos_key="rod_pos")