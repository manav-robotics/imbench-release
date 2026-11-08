"""Shared base for heterogeneous-bimanual task classes.

Each het-bi task variant has its own Python file under
``imbench/tasks/hetbi/`` and its own YAML config file under
``configs/tasks/hetbi/<task_id>.yaml``.

Subclasses must set ``self._staged_reward`` (a :class:`StagedReward`).
"""

from collections import deque

import numpy as np

import imbench.grippers  # noqa: F401  — register suction gripper
import imbench.envs  # noqa: F401  — register HetBiEnv with robosuite
from imbench.grippers.suction_weld import SuctionWeld
from imbench.objects import YCBObject, ExtrasObject
from imbench.tasks.base_task import BaseTask, load_task_config
from imbench.tasks.physics import commit_model_edits, set_body_mass
from imbench.tasks.placement import jitter_free_joint, table_overhang
from imbench.tasks.prehensile.contacts import body_geom_ids, geoms_touch


class HetBiBaseTask(BaseTask):
    """Shared base for heterogeneous bimanual tasks.

    Handles env construction, observations, domain randomization,
    suction weld, scoring, termination and reset/step cycles.  Subclasses
    define the task through ``self._staged_reward``, whose ``success_fn``
    is the success condition; the episode terminates when it holds or the
    payload falls off the table.

    Subclasses may override :attr:`ROBOSUITE_ENV_NAME` to swap in a
    specialised env (e.g. one that adds a goal marker), and
    :meth:`_extra_env_kwargs` to pass additional constructor arguments
    to that env.

    Parameters
    ----------
    task_id : str, optional
        YAML config name.  Defaults to the subclass's class-level ``TASK_ID``.
    **kwargs
        Forwarded to ``BaseTask`` (including ``obs_mode``, ``render_mode``, etc.).
    """

    ROBOSUITE_ENV_NAME = "HetBiEnv"
    # Action dims of an arm that must not move (zeroed after action noise), or None.
    _frozen_action_slice: slice | None = None

    def _extra_env_kwargs(self) -> dict:
        """Hook for subclasses to inject extra env constructor kwargs."""
        return {}

    def __init__(self, task_id: str | None = None, **kwargs):
        if task_id:
            self.TASK_ID = task_id
        self.CONFIG_FILE = f"hetbi/{self.TASK_ID}"
        self._suction_weld = SuctionWeld()
        self._payload_prefix = None
        self._rng = np.random.RandomState(kwargs.get("seed", 0))
        self._payload_fell_off = False
        self._payload_rest_z = None
        self._payload_handled = False
        self._pose_history: deque = deque()  # sized to the at-rest window at reset
        self._pose_history_step = None
        self._payload_was_stable = False
        self._stage_debug = {"_success": False}

        # Pre-load the per-task config to extract variant-specific fields
        # *before* super().__init__ (which triggers _build_env).
        _cfg = load_task_config(self.TASK_ID, config_file=self.CONFIG_FILE,
                               overrides=kwargs.get("config_overrides"))

        self.DESCRIPTION = _cfg.get("description", getattr(self, "DESCRIPTION", ""))
        self.YCB_PAYLOAD = _cfg.get("payload")
        self.FIXTURE = _cfg.get("fixture")
        self.ROBOTS = _cfg.get("robots", ["Panda", "Panda"])
        self.GRIPPER_TYPES = _cfg.get("grippers")
        self._variant_payload_pos = _cfg.get("payload_pos")
        self._variant_payload_quat = _cfg.get("payload_quat")
        self._variant_fixture_pos = _cfg.get("fixture_pos")
        self._variant_fixture_quat = _cfg.get("fixture_quat")
        if "max_episode_steps" in _cfg:
            self.MAX_EPISODE_STEPS = _cfg["max_episode_steps"]

        # Scene & randomization config
        scene_cfg = _cfg.get("scene", {})
        rand_cfg = _cfg.get("randomization", {})

        # Table / arm layout
        table_cfg = scene_cfg.get("table", {})
        self._table_height = table_cfg.get("height", 0.9)
        self._table_full_size = tuple(table_cfg.get("full_size", [1.0, 0.75, 0.05]))
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)
        self._robot_offset = scene_cfg.get("robot_offset")  # list of [x,y,z,...] per robot or None
        self._robot_rotation_z = scene_cfg.get("robot_rotation_z", 0.0)  # degrees; scalar or per-robot list

        # Scene base visuals
        self._object_color = scene_cfg.get("object_color")  # None = use mesh texture
        self._table_color = scene_cfg.get("table_color", [1.0, 1.0, 1.0, 1.0])

        # Physics randomization
        mass_cfg = rand_cfg.get("payload_mass", {})
        self._rand_payload_mass = mass_cfg.get("enabled", False)
        self._payload_mass_range = tuple(mass_cfg["range"]) if mass_cfg.get("range") else None

        pfric_cfg = rand_cfg.get("payload_friction", {})
        self._rand_payload_friction = pfric_cfg.get("enabled", False)
        self._payload_friction_range = tuple(pfric_cfg["range"]) if pfric_cfg.get("range") else None

        tfric_cfg = rand_cfg.get("table_friction", {})
        self._rand_table_friction = tfric_cfg.get("enabled", False)
        self._table_friction_range = tuple(tfric_cfg["range"]) if tfric_cfg.get("range") else None

        # Placement randomization
        place_cfg = rand_cfg.get("payload_placement", {})
        self._rand_placement = place_cfg.get("enabled", False)
        self._place_x_std = place_cfg.get("x_std", 0.02)
        self._place_y_std = place_cfg.get("y_std", 0.02)

        # Visual randomization
        self._rand_object_color = rand_cfg.get("object_color", {}).get("enabled", False)
        self._rand_table_color = rand_cfg.get("table_color", {}).get("enabled", False)
        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position = light_cfg.get("position", False)
        self._rand_light_color = light_cfg.get("color", False)

        # Dynamics randomization
        act_noise_cfg = rand_cfg.get("action_noise", {})
        self._rand_action_noise = act_noise_cfg.get("enabled", False)
        self._action_noise_std = act_noise_cfg.get("std", 0.01)

        obs_noise_cfg = rand_cfg.get("observation_noise", {})
        self._rand_obs_noise = obs_noise_cfg.get("enabled", False)
        self._obs_noise_std = obs_noise_cfg.get("std", 0.005)

        # Task conditions (failure — common to all hetbi tasks)
        conditions_cfg = _cfg.get("task_conditions", {})
        failure_cfg = conditions_cfg.get("failure", {})
        self._failure_floor_threshold = failure_cfg.get("floor_drop_below", 0.05)

        # IK input scaling (used only by tasks running an IK_POSE controller).
        # Per-arm step delta = action * sensitivity.  Defaults match the
        # hobi plate-shelf task.
        ik_scale_cfg = _cfg.get("ik_scale", {})
        self._ik_pos_sensitivity = ik_scale_cfg.get("pos_sensitivity", 0.005)
        self._ik_ori_sensitivity = ik_scale_cfg.get("ori_sensitivity", 0.01)
        # Detect IK_POSE controller from the YAML so step() can pre-scale.
        controller_list = _cfg.get("controller", [])
        self._uses_ik_pose = bool(
            controller_list and any(
                isinstance(c, dict) and str(c.get("type", "")).upper() == "IK_POSE"
                for c in controller_list
            )
        )

        super().__init__(**kwargs)

    # ── env construction ────────────────────────────────────────────

    def _build_env(self):
        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        common_kwargs = dict(
            robots=self.ROBOTS,
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            has_renderer=(self.render_mode == "human"),
            has_offscreen_renderer=use_cameras,
            render_camera=self.viewer_camera,
            renderer_config={"cam_config": self._viewer_cam_config} if self.render_mode == "human" else None,
            use_camera_obs=False,  # BaseTask renders camera frames itself
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            horizon=self.MAX_EPISODE_STEPS,
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
        )

        if self.GRIPPER_TYPES is not None:
            common_kwargs["gripper_types"] = self.GRIPPER_TYPES

        payload = YCBObject(
            name=self.YCB_PAYLOAD, ycb_name=self.YCB_PAYLOAD,
        )
        fixture = None
        if self.FIXTURE is not None:
            fixture = ExtrasObject(
                name=f"{self.FIXTURE}_target",
                extras_name=self.FIXTURE, joints=None,
            )
        self._payload_prefix = payload.naming_prefix
        env_kwargs = dict(
            payload_object=payload,
            fixture_object=fixture,
            **common_kwargs,
        )
        if self._robot_offset is not None:
            env_kwargs["robot_offset"] = self._robot_offset
        if self._robot_rotation_z:
            env_kwargs["robot_rotation_z"] = self._robot_rotation_z
        if self._variant_payload_pos is not None:
            env_kwargs["payload_pos"] = self._variant_payload_pos
        if self._variant_payload_quat is not None:
            env_kwargs["payload_quat"] = self._variant_payload_quat
        if self._variant_fixture_pos is not None:
            env_kwargs["fixture_pos"] = self._variant_fixture_pos
        if self._variant_fixture_quat is not None:
            env_kwargs["fixture_quat"] = self._variant_fixture_quat
        env_kwargs.update(self._extra_env_kwargs())
        env = self._make_robosuite_env(self.ROBOSUITE_ENV_NAME, **env_kwargs)
        return env

    # ── obs ─────────────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)

        # Payload / object state — only include when enabled in YAML config
        if self.object_state_enabled:
            pf = self._payload_prefix or "payload_"
            pos_key = f"{pf}pos"
            quat_key = f"{pf}quat"
            if pos_key in raw_obs:
                obs["object_pos"] = raw_obs[pos_key]
            if quat_key in raw_obs:
                obs["object_quat"] = raw_obs[quat_key]

        return obs

    # ── scoring & termination (shared by all hetbi tasks) ──────────

    def _payload_below_table(self, raw_obs) -> bool:
        """Latch and return whether the payload has fallen off the table."""
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if raw_obs[pos_key][2] < self._table_height - self._failure_floor_threshold:
            self._payload_fell_off = True
        return self._payload_fell_off

    def _compute_task_score(self, raw_obs):
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if pos_key not in raw_obs or self._payload_below_table(raw_obs):
            self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
            return 0.0
        # Record the payload pose every step, even when the success
        # predicate short-circuits before reaching its stability check.
        self._payload_is_stable_multistep()
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if pos_key not in raw_obs:
            return False
        if self._payload_below_table(raw_obs):
            return True
        self._compute_task_score(raw_obs)
        return bool(self._stage_debug["_success"])

    def _payload_lift_progress(self, raw_obs, height: float) -> float:
        """Payload rise above its height at reset, as a fraction of *height*."""
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if pos_key not in raw_obs or self._payload_rest_z is None:
            return 0.0
        lift = raw_obs[pos_key][2] - self._payload_rest_z
        return float(np.clip(lift / height, 0.0, 1.0))

    def _payload_released(self) -> bool:
        """True when no suction weld holds the payload and no gripper touches it."""
        payload_id = getattr(self._env, "_payload_body_id", None)
        if payload_id in self._suction_weld.welded_body_ids():
            return False
        return not self._gripper_contacts_payload()

    #: Duration (s) the payload must stay released and still to count as at rest.
    _AT_REST_WINDOW_S = 0.25
    #: Tolerance multiplier once at rest, so contact chatter does not flicker the flag.
    _AT_REST_HYSTERESIS = 2.0

    def _at_rest_window_steps(self) -> int:
        """Control steps in the at-rest window."""
        return max(1, round(self._AT_REST_WINDOW_S * self.control_freq))

    def _payload_is_stable_multistep(self, vel_thresh=0.02, ang_vel_thresh=0.2):
        """Whether the payload has been released and still for :attr:`_AT_REST_WINDOW_S`.

        Every step in the window must have the payload released, and every pose
        must lie within ``vel_thresh * T`` metres and ``ang_vel_thresh * T``
        radians of the current pose, ``T`` being the window's duration. Judging
        net motion instead of instantaneous velocity keeps the contact chatter
        of a resting object from reading as motion; once stable, twice the
        tolerance is allowed before it reads as moving again.

        Memoized on ``self._step_count``: one scoring pass evaluates it from
        ``_compute_task_score``, ``success_fn`` and stage functions, and the
        pose history must still grow only once per env step.
        """
        if self._pose_history_step == self._step_count:
            return self._payload_was_stable
        sim = self._env.sim
        body = self._env._payload_body_id
        self._pose_history.append(
            (np.array(sim.data.body_xpos[body]), np.array(sim.data.body_xquat[body]), self._payload_released())
        )
        self._pose_history_step = self._step_count
        n_steps = self._at_rest_window_steps()
        window = list(self._pose_history)[-(n_steps + 1):]
        stable = len(window) == n_steps + 1 and all(released for *_, released in window)
        if stable:
            slack = self._AT_REST_HYSTERESIS if self._payload_was_stable else 1.0
            duration = n_steps / self.control_freq
            pos_now, quat_now, _ = window[-1]
            for pos, quat, _ in window[:-1]:
                angle = 2.0 * np.arccos(np.clip(abs(np.dot(quat, quat_now)), 0.0, 1.0))
                if (np.linalg.norm(pos - pos_now) > slack * vel_thresh * duration
                        or angle > slack * ang_vel_thresh * duration):
                    stable = False
                    break
        self._payload_was_stable = stable
        return stable

    def _arm_gripper(self, robot_idx):
        """Gripper model of robot *robot_idx*."""
        g = self._env.robots[robot_idx].gripper
        return next(iter(g.values())) if isinstance(g, dict) else g

    def _gripper_contacts_payload_idx(self, robot_idx) -> bool:
        """Whether any geom on robot *robot_idx*'s gripper touches the payload.

        The gripper is its body subtree, so tactile taxels count as part of it
        (as in :meth:`_check_grasp`).
        """
        payload_body_id = getattr(self._env, "_payload_body_id", None)
        if payload_body_id is None:
            return False
        sim = self._env.sim
        gripper_geoms = body_geom_ids(sim, self._arm_gripper(robot_idx).root_body)
        payload_geoms = set(np.flatnonzero(sim.model.geom_bodyid == payload_body_id).tolist())
        return geoms_touch(sim, gripper_geoms, payload_geoms)

    def _payload_was_handled(self) -> bool:
        """Whether either gripper has touched the payload this episode (latched)."""
        if not self._payload_handled:
            self._payload_handled = self._gripper_contacts_payload()
        return self._payload_handled

    def _gripper_contacts_payload(self) -> bool:
        """Whether any robot's gripper touches the payload."""
        return any(self._gripper_contacts_payload_idx(i) for i in range(len(self._env.robots)))

    # ── step / reset with suction weld ──────────────────────────────

    def reset(self, seed=None, options=None):
        self._suction_weld.reset()
        self._payload_fell_off = False
        self._pose_history = deque(maxlen=self._at_rest_window_steps() + 1)
        self._pose_history_step = None
        self._payload_was_stable = False
        self._staged_reward.reset()
        self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
        super().reset(seed=seed, options=options)

        if seed is not None:
            self._rng = np.random.RandomState(self.seed)
        sim = self._env.sim

        # The suction snout should ONLY ever weld to the payload —
        # never to scene geometry (table/fixture), markers, or the robot
        # mount/links. A grazing contact during transit would otherwise
        # lock the EEF to whatever it brushed against.
        payload_id = getattr(self._env, "_payload_body_id", None)
        excluded: set[int] = set()
        if payload_id is not None:
            for bid in range(sim.model.nbody):
                if bid != int(payload_id):
                    excluded.add(int(bid))
        self._suction_weld.excluded_body_ids = excluded

        # ── Physics randomization ───────────────────────────────────
        self._randomize_physics(sim)

        # ── Placement randomization (Gaussian jitter on payload pos) ─
        if self._rand_placement:
            self._randomize_placement(sim)

        # ── Visual randomization ────────────────────────────────────
        self._randomize_visuals(sim)

        sim.forward()
        self._payload_rest_z = float(sim.data.body_xpos[self._env._payload_body_id][2])
        self._payload_handled = False

        # Re-observe after modifications.  force_update=True is required
        # because robosuite caches observables per timestep — without it
        # the cameras return stale images from before visual randomization.
        obs = self._rebuild_obs()
        info = {"task_score": 0.0, "is_success": False, "step": 0}
        return obs, info

    def step(self, action: np.ndarray):
        action = self._validate_action(action)

        # ── Action noise ────────────────────────────────────────────
        if self._rand_action_noise:
            noise = self._rng.normal(0, self._action_noise_std, size=action.shape)
            noise[6::7] = 0.0  # gripper channels are discrete commands (suction toggles on sign)
            action = action + noise
        if self._frozen_action_slice is not None:
            action[self._frozen_action_slice] = 0.0  # after the noise, so a frozen arm stays still

        # ── IK_POSE pre-scale ───────────────────────────────────────
        # Robosuite's IK_POSE controller multiplies action by user_sensitivity
        # internally (default 0.3 m / unit), giving large per-step deltas.
        # Pre-scale at the task layer: the per-step target delta is then
        # action * pos_sensitivity * 0.3 (the controller's user_sensitivity),
        # i.e. 1.5 mm/step at the default, unlike hobi's action * pos_sensitivity.
        # Gripper dim is left untouched.
        if self._uses_ik_pose:
            action = np.asarray(action, dtype=np.float64).copy()
            for base in (0, 7):
                if base + 6 < action.size:
                    action[base:base + 3] *= self._ik_pos_sensitivity
                    action[base + 3:base + 6] *= self._ik_ori_sensitivity

        obs, reward, terminated, truncated, info = super().step(action)
        self._suction_weld.update(self._env, action)

        # ── Observation noise (proprio keys only) ───────────────────
        if self._rand_obs_noise:
            for key in obs:
                if isinstance(obs[key], np.ndarray) and key.startswith("robot"):
                    obs[key] = obs[key] + self._rng.normal(
                        0, self._obs_noise_std, size=obs[key].shape
                    )

        return obs, reward, terminated, truncated, info

    # ── physics randomization ───────────────────────────────────────

    def _randomize_physics(self, sim):
        """Apply physics domain randomization (mass, friction) to the sim."""
        payload_body_id = getattr(self._env, "_payload_body_id", None)

        # ── Payload mass ────────────────────────────────────────────
        if self._rand_payload_mass and self._payload_mass_range and payload_body_id is not None:
            set_body_mass(sim, payload_body_id, self._rng.uniform(*self._payload_mass_range))
            commit_model_edits(sim)

        # ── Payload friction ────────────────────────────────────────
        if self._rand_payload_friction and self._payload_friction_range and payload_body_id is not None:
            fric = self._rng.uniform(*self._payload_friction_range)
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == payload_body_id:
                    sim.model.geom_friction[i][0] = fric

        # ── Table friction ──────────────────────────────────────────
        if self._rand_table_friction and self._table_friction_range:
            fric = self._rng.uniform(*self._table_friction_range)
            for i in range(sim.model.ngeom):
                name = sim.model.geom_id2name(i)
                if name and "table" in name and "visual" not in name:
                    sim.model.geom_friction[i][0] = fric

    # ── placement randomization ─────────────────────────────────────

    def _randomize_placement(self, sim):
        """Jitter the payload's x/y (see :func:`jitter_free_joint`).

        A start is invalid if it overhangs the table further than the nominal
        position, penetrates the fixture, or is already in the goal
        (:meth:`_placement_in_goal`).
        """
        joint_name = self._env._payload.joints[0]
        radius = float(self._env._payload.horizontal_radius)
        nominal_xy = np.array(sim.data.get_joint_qpos(joint_name)[:2])
        max_overhang = table_overhang(self._env, radius, nominal_xy)
        fixture_id = getattr(self._env, "_fixture_body_id", None)

        def valid(pos) -> bool:
            if table_overhang(self._env, radius, pos[:2]) > max_overhang + 1e-9:
                return False
            if fixture_id is not None and self._payload_penetrates(sim, fixture_id):
                return False
            return not self._placement_in_goal(pos)

        jitter_free_joint(sim, joint_name, self._rng, (self._place_x_std, self._place_y_std), valid)

    def _payload_penetrates(self, sim, body_id: int) -> bool:
        """True if any payload geom penetrates a geom of body *body_id*."""
        payload_id = self._env._payload_body_id
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            bodies = {int(sim.model.geom_bodyid[c.geom1]), int(sim.model.geom_bodyid[c.geom2])}
            if bodies == {payload_id, body_id} and c.dist < 0.0:
                return True
        return False

    def _placement_in_goal(self, pos) -> bool:
        """Whether a payload starting at *pos* would already satisfy the goal."""
        return False

    # ── visual randomization ────────────────────────────────────────

    def _randomize_visuals(self, sim):
        """Apply visual domain randomization (colors + lighting) to the sim."""
        payload_body_id = getattr(self._env, "_payload_body_id", None)

        # ── Object color ────────────────────────────────────────────
        if self._rand_object_color and payload_body_id is not None:
            obj_rgba = self._random_object_color()
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == payload_body_id:
                    sim.model.geom_matid[i] = -1  # strip texture
                    sim.model.geom_rgba[i] = obj_rgba
        elif self._object_color is not None and payload_body_id is not None:
            obj_rgba = np.array(self._object_color)
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == payload_body_id:
                    sim.model.geom_matid[i] = -1
                    sim.model.geom_rgba[i] = obj_rgba

        # ── Table color ─────────────────────────────────────────────
        table_rgba = (
            self._random_table_color() if self._rand_table_color
            else np.array(self._table_color)
        )
        for i in range(sim.model.ngeom):
            name = sim.model.geom_id2name(i)
            if name and name == "table_visual":
                sim.model.geom_rgba[i] = table_rgba
                mat_id = sim.model.geom_matid[i]
                if mat_id >= 0:
                    sim.model.mat_rgba[mat_id] = table_rgba

        # ── Lighting randomization ──────────────────────────────────
        for i in range(sim.model.nlight):
            if self._rand_light_position:
                sim.model.light_pos[i] = [
                    self._rng.uniform(0.0, 2.0),
                    self._rng.uniform(-1.0, 2.0),
                    self._rng.uniform(1.0, 2.5),
                ]
            if self._rand_light_color:
                # Near-white warm-to-cool lighting; avoid strongly colored
                # lights (e.g. red, green, blue).
                base = self._rng.uniform(0.6, 1.0)
                jitter = self._rng.uniform(-0.08, 0.08, size=3)
                diffuse = np.clip(base + jitter, 0.4, 1.0)
                sim.model.light_diffuse[i] = diffuse
                amb_base = self._rng.uniform(0.05, 0.25)
                amb_jitter = self._rng.uniform(-0.04, 0.04, size=3)
                sim.model.light_ambient[i] = np.clip(
                    amb_base + amb_jitter, 0.0, 0.4
                )

    # ── color helpers (realistic palette) ───────────────────────────

    def _random_object_color(self):
        """Sample a plausible object color — muted, avoids neon extremes."""
        rgb = self._rng.uniform(0.15, 0.85, size=3)
        gray = rgb.mean()
        rgb = 0.5 * rgb + 0.5 * gray
        rgb = np.clip(rgb, 0.1, 0.9)
        return np.array([*rgb, 1.0])

    def _random_table_color(self):
        """Sample a realistic table-top color (neutrals: grays, off-whites, light wood)."""
        base = self._rng.uniform(0.45, 0.95)
        jitter = self._rng.uniform(-0.06, 0.06, size=3)
        jitter[0] += self._rng.uniform(0.0, 0.04)
        rgb = np.clip(base + jitter, 0.3, 1.0)
        return np.array([*rgb, 1.0])
