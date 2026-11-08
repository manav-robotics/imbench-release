"""Edge Slide and Grasp v2 — same task as v1 but with a flat cylinder object.

Subclasses EdgeSlideV1 and overrides only the environment construction and
per-episode reset logic.  All reward, termination, observation, and helper
methods are inherited unchanged.
"""

import numpy as np
from robosuite.utils.placement_samplers import UniformRandomSampler
from imbench.tasks.physics import commit_model_edits, cylinder_inertia, set_body_mass
from imbench.tasks.prehensile.edge_slide_v1 import EdgeSlideV1
import imbench.envs.edge_slide_env_v2  # noqa: F401  (registers EdgeSlideEnvV2)


class EdgeSlideV2(EdgeSlideV1):
    TASK_ID = "ib-p1-edge-slide-v2"
    DESCRIPTION = (
        "Single arm: slide a flat cylinder to the table edge to create a "
        "graspable overhang, then grasp and lift it."
    )

    # ── config parsing: cylinder dims instead of box dims ───────────────

    def _parse_task_config(self):
        # Let v1 parse everything it knows about (table, placement,
        # visuals, robot, lighting).
        super()._parse_task_config()

        # Now override the object-specific fields with cylinder params.
        obj_cfg = self.cfg.get("scene", {}).get("object", {})
        rand_cfg = self.cfg.get("randomization", {})

        self._cylinder_radius      = obj_cfg.get("radius",      0.075)
        self._cylinder_half_height = obj_cfg.get("half_height", 0.0015)
        self._cylinder_mass        = obj_cfg.get("mass",        self._book_mass)
        self._cylinder_friction    = obj_cfg.get("friction",    self._book_friction)
        self._cylinder_color       = obj_cfg.get("color",       [0.0, 0.3, 0.7, 1.0])

        rad_rand  = rand_cfg.get("object_radius",   {}).get("range")
        fric_rand = rand_cfg.get("object_friction", {}).get("range")
        self._cylinder_radius_range   = tuple(rad_rand)  if rad_rand  else None
        self._cylinder_friction_range = tuple(fric_rand) if fric_rand else None

        # v1 sizes the overhang-progress footprint off the book's
        # half_length/half_width; this is a flat cylinder instead.
        self._object_footprint_radius = self._cylinder_radius

    # ── env construction: build EdgeSlideEnvV2 instead of EdgeSlideEnv ──

    def _build_env(self):
        self._parse_task_config()

        if self._rand_object_color:
            self._cylinder_color = list(self._random_object_color())

        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=None,
            x_range=[0.0, 0.0],
            y_range=[0.0, 0.0],
            rotation=0.0,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            reference_pos=np.array([0.0, 0.0, self._table_height]),
            z_offset=0.002,
        )

        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        max_radius   = (self._cylinder_radius_range[1]
                        if self._cylinder_radius_range   else self._cylinder_radius)
        max_friction = (self._cylinder_friction_range[1]
                        if self._cylinder_friction_range else self._cylinder_friction)
        min_friction = (self._cylinder_friction_range[0]
                        if self._cylinder_friction_range else self._cylinder_friction)

        env_kwargs = dict(
            cylinder_radius=max_radius,
            cylinder_half_height=self._cylinder_half_height,
            cylinder_color=self._cylinder_color,
            cylinder_friction=(max_friction, 0.005, 0.0001),
            table_friction=self._table_friction(min_friction),
            robot_offset=self._robot_offset,
            robot_ori=self._robot_ori,
            robots="Panda",
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            placement_initializer=placement_initializer,
            has_renderer=(self.render_mode == "human"),
            render_camera=self.viewer_camera,
            renderer_config={"cam_config": self._viewer_cam_config} if self.render_mode == "human" else None,
            has_offscreen_renderer=use_cameras,
            use_camera_obs=False,  # BaseTask renders camera frames itself
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            horizon=self.MAX_EPISODE_STEPS,
        )
        env = self._make_robosuite_env("EdgeSlideEnvV2", **env_kwargs)
        return env

    # ── reset: cylinder-aware geom patching ─────────────────────────────

    def reset(self, *, seed=None, options=None):
        # Call BaseTask.reset (grandparent) directly — same as v1 does —
        # to get the env reset without v1's book-specific geom patching.
        from imbench.tasks.base_task import BaseTask
        BaseTask.reset(self, seed=seed, options=options)

        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._book_fell_off = False   # v1 termination checks this name
        self._staged_reward.reset()

        sim = self._env.sim

        radius   = (self._rng.uniform(*self._cylinder_radius_range)
                    if self._cylinder_radius_range   else self._cylinder_radius)
        half_ht  = self._cylinder_half_height
        friction = (self._rng.uniform(*self._cylinder_friction_range)
                    if self._cylinder_friction_range else self._cylinder_friction)

        cube_body_id = self._find_body_id(sim, "cube")
        if cube_body_id is not None:
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == cube_body_id:
                    # MuJoCo cylinder: size = [radius, half_height, 0]
                    sim.model.geom_size[i]     = [radius, half_ht, 0.0]
                    sim.model.geom_friction[i] = [friction, 0.005, 0.0001]
            set_body_mass(sim, cube_body_id, self._cylinder_mass,
                          cylinder_inertia(self._cylinder_mass, radius, half_ht))

        x   = self._place_center[0] + self._rng.uniform(-self._place_x_range, self._place_x_range)
        y   = self._place_center[1] + self._rng.uniform(-self._place_y_range, self._place_y_range)
        yaw = self._rng.uniform(0, 2 * np.pi)
        z   = self._table_height + half_ht + 0.001

        self._teleport_cube(sim, x, y, z, yaw)
        self._randomize_visuals(sim)

        commit_model_edits(sim)

        self._camera_cache.clear()
        raw_obs = self._get_sim_observations(force_update=True)

        cube_body_id = self._find_body_id(sim, "cube")
        if cube_body_id is not None:
            raw_obs["cube_pos"] = np.array(sim.data.body_xpos[cube_body_id])

        self._rest_obj_z = (
            float(raw_obs["cube_pos"][2]) if "cube_pos" in raw_obs else self._table_height
        )

        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs  = self._observe(raw_obs)
        info = {"task_score": 0.0, "is_success": False, "step": 0}
        return obs, info