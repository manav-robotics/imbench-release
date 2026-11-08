"""Base class for single-arm tasks using SingleArmTableEnv.

Provides shared config parsing, obs building, visual randomization,
and termination helpers that are common across most single-arm tasks.
"""

import numpy as np

import imbench.envs  # noqa: F401 — register envs
from imbench.tasks.base_task import BaseTask


class SingleArmBase(BaseTask):
    """Shared base for single-arm prehensile and puzzle tasks.

    Subclasses must set ``TASK_ID`` and may override:
    - ``_parse_task_config()`` — to read task-specific YAML keys
    - ``_make_objects()`` — return list of object spec dicts
    - ``_make_static_objects()`` — return list of static object spec dicts
    - ``_compute_task_score()`` / ``_check_termination()``
    - ``_parked_objects()`` — name objects parked off-scene this episode

    Tasks whose object count varies per episode build every object they
    could ever need into the model once, then each reset place the ones
    they use and park the rest at :attr:`PARKED_POS` via
    :meth:`_park_objects`.  Naming them in ``_parked_objects()`` zeroes
    their slots in ``object_state`` so the agent never sees them while the
    observation shape stays fixed.
    """

    TASK_ID: str = ""
    DESCRIPTION: str = ""

    #: Where unused dynamic objects are parked — far enough from the
    #: scene that they can never interact with it or show up in a camera
    #: (well beyond the renderer's far plane).  They free-fall from here
    #: and come to rest on the ground plane near x=y=100, metres away from
    #: anything the robot can touch.
    PARKED_POS: tuple[float, float, float] = (100.0, 100.0, 100.0)
    #: x-spacing between parked objects so they never collide with each other.
    PARKED_SPACING: float = 1.0
    #: Linear speed (m/s) below which a free object counts as at rest.
    REST_SPEED: float = 0.05

    def __init__(self, **kwargs):
        self._rng = np.random.RandomState(kwargs.get("seed", 0))
        super().__init__(**kwargs)

    # ── Public accessors ────────────────────────────────────────────

    @property
    def sim(self):
        return self._env.sim

    @property
    def env(self):
        return self._env

    # ── Config parsing ──────────────────────────────────────────────

    def _parse_common_config(self):
        """Parse scene, table, randomization keys shared by all single-arm tasks."""
        scene = self.cfg.get("scene", {})
        table_cfg = scene.get("table", {})
        rand_cfg = self.cfg.get("randomization", {})

        self._table_height = table_cfg.get("height", 0.8)
        self._table_full_size = tuple(table_cfg.get("full_size", [0.5, 0.5, 0.05]))
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)
        self._robot_offset = scene.get("robot_offset")

        # Visual randomization
        self._rand_object_color = rand_cfg.get("object_color", {}).get("enabled", False)
        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position = light_cfg.get("position", False)
        self._rand_light_color = light_cfg.get("color", False)

        # Task conditions (optional)
        cond_cfg = self.cfg.get("task_conditions", {})
        success_cfg = cond_cfg.get("success", {})
        failure_cfg = cond_cfg.get("failure", {})
        self._success_score_threshold = success_cfg.get("score_threshold", 1.0)
        self._failure_floor_threshold = failure_cfg.get("floor_z_threshold", 0.05)

    def _parse_task_config(self):
        """Override in subclass to parse task-specific config keys."""
        pass

    # ── Object factory — subclasses override these ──────────────────

    def _make_objects(self):
        """Return list of object spec dicts for dynamic objects."""
        return []

    def _make_static_objects(self):
        """Return list of object spec dicts for static objects."""
        return []

    # ── Environment construction ────────────────────────────────────

    def _build_env(self):
        self._parse_common_config()
        self._parse_task_config()

        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        objects = self._make_objects()
        static_objects = self._make_static_objects()

        placement_cfg = self.cfg.get("placement", {})
        x_range = placement_cfg.get("x_range", 0.0)
        y_range = placement_cfg.get("y_range", 0.0)
        if isinstance(x_range, (int, float)):
            x_range = [-x_range, x_range]
        if isinstance(y_range, (int, float)):
            y_range = [-y_range, y_range]

        env_kwargs = dict(
            objects=objects,
            static_objects=static_objects,
            robots="Panda",
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            placement_config={
                "x_range": x_range,
                "y_range": y_range,
                "z_offset": placement_cfg.get("z_offset", 0.01),
                "reference_pos": [0.0, 0.0, self._table_height],
                "ensure_valid_placement": placement_cfg.get("ensure_valid_placement", True),
            },
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            has_renderer=(self.render_mode == "human"),
            render_camera=None,
            has_offscreen_renderer=use_cameras,
            use_camera_obs=False,  # BaseTask renders camera frames itself
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            horizon=self.MAX_EPISODE_STEPS,
        )
        if self._robot_offset is not None:
            env_kwargs["robot_offset"] = self._robot_offset

        return self._make_robosuite_env("SingleArmTableEnv", **env_kwargs)

    # ── Parked (unused) objects ─────────────────────────────────────

    def _parked_objects(self):
        """Names of dynamic objects parked off-scene for this episode.

        Default: none.  Tasks with a per-episode object count override
        this so the parked objects are stripped from the obs dict.
        """
        return set()

    def _park_objects(self, names):
        """Teleport *names* near :attr:`PARKED_POS` with zero velocity.

        Objects are spaced :attr:`PARKED_SPACING` apart along x. Call this
        from ``reset()`` for every object the episode does not use, and
        return the same names from :meth:`_parked_objects`.
        """
        x0, y0, z0 = self.PARKED_POS
        # Sorted: set iteration order varies across processes (hash seeding).
        for i, name in enumerate(sorted(names)):
            self._place_object(name, [x0 + i * self.PARKED_SPACING, y0, z0])
            self._set_object_velocity(name, [0, 0, 0], [0, 0, 0])

    def _strip_parked_obs(self, raw_obs):
        """Return a copy of *raw_obs* with parked objects hidden.

        Drops each parked object's ``<name>_pos`` / ``<name>_quat`` /
        ``gripper_to_<name>_pos`` entries and rebuilds the flat
        ``object-state`` vector in robosuite's concatenation order (see
        ``MujocoEnv._get_observations``) with those entries zeroed, so its
        length is fixed across episodes and matches ``observation_space``.

        ``SingleArmTableEnv`` aliases the *first* dynamic object's
        observables as ``cube_pos`` / ``cube_quat``.  When that object is
        the parked one the alias is re-pointed at the first object still
        in the scene, so ``object_pos`` / ``object_quat`` always describe
        something the robot can actually reach.

        The original dict is left untouched — ``_compute_task_score()``
        and ``_check_termination()`` receive the unfiltered obs and can
        still reason about every object in the model.
        """
        parked = self._parked_objects()
        if not parked:
            return raw_obs

        observables = getattr(self._env, "_observables", {})

        drop = set()
        for name in parked:
            drop.update((f"{name}_pos", f"{name}_quat",
                         f"gripper_to_{name}_pos"))
        # Aliases share the Observable *object* with the key they alias,
        # so identity is what identifies them, not the name.
        parked_ids = {id(observables[k]) for k in drop if k in observables}
        drop |= {k for k, o in observables.items() if id(o) in parked_ids}

        obs = {k: v for k, v in raw_obs.items() if k not in drop}

        if "object-state" in raw_obs:
            parts = [
                np.zeros_like(np.atleast_1d(raw_obs[k])) if k in drop
                else np.atleast_1d(raw_obs[k])
                for k, o in observables.items()
                if o.modality == "object" and o.is_enabled() and o.is_active()
                and k in raw_obs
            ]
            obs["object-state"] = (np.concatenate(parts) if parts
                                   else np.zeros(0, dtype=np.float64))

        # Re-alias cube_pos/cube_quat when the aliased object is parked.
        if "cube_pos" in drop:
            for k, o in observables.items():
                if (o.modality == "object" and k.endswith("_pos")
                        and not k.startswith("gripper_to_") and k in obs):
                    obs["cube_pos"] = raw_obs[k]
                    quat_key = k[:-len("_pos")] + "_quat"
                    if quat_key in obs:
                        obs["cube_quat"] = raw_obs[quat_key]
                    break

        return obs

    # ── Obs ─────────────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        raw_obs = self._strip_parked_obs(raw_obs)
        obs = self._build_obs_from_config(raw_obs)
        # Add object positions if available
        if "cube_pos" in raw_obs:
            obs["object_pos"] = raw_obs["cube_pos"]
        if "cube_quat" in raw_obs:
            obs["object_quat"] = raw_obs["cube_quat"]
        return obs

    # ── Reset with visual randomization ─────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Only re-seed _rng when an explicit seed is provided;
        # otherwise keep the existing RNG so draws differ across episodes.
        if seed is not None:
            self._rng = np.random.RandomState(self.seed)

        sim = self._env.sim
        # Constrained light randomization
        if self._rand_light_color:
            for i in range(sim.model.nlight):
                sim.model.light_diffuse[i] = self._rng.uniform(0.6, 1.0, size=3)
                sim.model.light_specular[i] = sim.model.light_diffuse[i] * 0.5
        if self._rand_light_position:
            for i in range(sim.model.nlight):
                sim.model.light_pos[i][:2] += self._rng.uniform(-0.3, 0.3, size=2)
                sim.model.light_pos[i][2] = self._rng.uniform(1.0, 2.0)
        if self._rand_light_color or self._rand_light_position:
            obs = self._rebuild_obs()

        return obs, info

    # ── Visual randomization helpers ────────────────────────────────

    def _random_object_color(self):
        rgb = self._rng.uniform(0.1, 0.9, size=3)
        return np.array([*rgb, 1.0])

    # ── Termination helper ──────────────────────────────────────────

    def _object_fell_off_table(self, raw_obs, pos_key="cube_pos"):
        """Check if object z is below floor threshold."""
        if pos_key in raw_obs:
            return raw_obs[pos_key][2] < self._failure_floor_threshold
        return False

    # ── Object positioning helpers ──────────────────────────────────

    def _place_object(self, name, pos, quat=None):
        """Reposition a dynamic (free-joint) object after sampler placement.

        Parameters
        ----------
        name : str
            Object name (e.g. ``"cube"``, ``"domino_0"``).
        pos : array-like
            ``[x, y, z]`` world position.
        quat : array-like, optional
            ``[w, x, y, z]`` MuJoCo quaternion.  Defaults to ``[1, 0, 0, 0]``.
        """
        if quat is None:
            quat = [1, 0, 0, 0]
        obj = self._env.get_object(name)
        if obj is not None and obj.joints:
            self._env.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([np.array(pos, dtype=float),
                                np.array(quat, dtype=float)]),
            )

    def _place_static(self, name, pos, quat=None):
        """Reposition a static (welded) object.

        Parameters
        ----------
        name : str
            Object name (e.g. ``"goal_marker"``, ``"occluder"``).
        pos : array-like
            ``[x, y, z]`` world position.
        quat : array-like, optional
            ``[w, x, y, z]`` MuJoCo quaternion.
        """
        sim = self._env.sim
        body_id = sim.model.body_name2id(f"{name}_main")
        sim.model.body_pos[body_id] = np.array(pos, dtype=float)
        if quat is not None:
            sim.model.body_quat[body_id] = np.array(quat, dtype=float)

    def _set_object_velocity(self, name, linvel, angvel=None):
        """Set velocity for a dynamic object.

        Parameters
        ----------
        name : str
            Object name.
        linvel : array-like
            ``[vx, vy, vz]`` linear velocity.
        angvel : array-like, optional
            ``[wx, wy, wz]`` angular velocity. Defaults to zero.
        """
        sim = self._env.sim
        jnt_addr = sim.model.body_jntadr[self._object_body_id(name)]
        if jnt_addr >= 0:
            dof_addr = sim.model.jnt_dofadr[jnt_addr]
            sim.data.qvel[dof_addr:dof_addr + 3] = np.array(linvel, dtype=float)
            if angvel is not None:
                sim.data.qvel[dof_addr + 3:dof_addr + 6] = np.array(angvel, dtype=float)

    def _object_body_id(self, name) -> int:
        """Body id of object *name*.

        Composite objects (e.g. hex_prism, cone_frustum) register their root
        body as ``<name>_root``; primitives use ``<name>_main``.
        """
        model = self._env.sim.model
        for suffix in ("_main", "_root", ""):
            try:
                return model.body_name2id(name + suffix)
            except Exception:
                continue
        raise ValueError(f"No body found for object '{name}'")

    def _body_velocity(self, body_id) -> np.ndarray:
        """Free-joint velocity ``[vx, vy, vz, wx, wy, wz]`` of *body_id*; zeros without a joint."""
        sim = self._env.sim
        jnt_addr = sim.model.body_jntadr[body_id]
        if jnt_addr < 0:
            return np.zeros(6)
        dof_addr = sim.model.jnt_dofadr[jnt_addr]
        return np.array(sim.data.qvel[dof_addr:dof_addr + 6])

    def _object_speed(self, name) -> float:
        """Linear speed (m/s) of free object *name*; 0 for an object without a free joint."""
        return float(np.linalg.norm(self._body_velocity(self._object_body_id(name))[:3]))

    def _gripper_contacts_object(self, obj_name="cube"):
        """Check if gripper is in contact with the named object."""
        obj = self._env.get_object(obj_name)
        geoms = obj.contact_geoms if obj is not None else obj_name
        return self._check_grasp(
            gripper=self._env.robots[0].gripper,
            object_geoms=geoms,
        )
