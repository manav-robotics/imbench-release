"""Freeform task — a configurable table-top scene built from constructor arguments.

Usage
-----
    import imbench as ib

    task = ib.make("ib-m-freeform-v1",
        robots=[{"name": "Panda", "gripper": "default"}],
        objects=[
            {"type": "box",      "name": "red_cube",   "size": [0.02, 0.02, 0.02],
             "pos": [0.0, 0.0, 0.82], "rgba": [1, 0, 0, 1], "mass": 0.05},
            {"type": "cylinder", "name": "blue_cyl",   "size": [0.015, 0.04],
             "pos": [0.1, 0.1, 0.84], "rgba": [0, 0, 1, 1], "mass": 0.03},
            {"type": "sphere",   "name": "green_ball", "size": [0.02],
             "pos": [-0.1, 0.0, 0.82], "rgba": [0, 1, 0, 1], "mass": 0.02},
        ],
        goals=[
            {"object": "red_cube", "target_pos": [0.2, 0.0, 0.82], "threshold": 0.05, "weight": 0.5},
            {"object": "blue_cyl", "target_pos": [0.0, 0.2, 0.84], "threshold": 0.05, "weight": 0.5},
        ],
        controller="OSC_POSE",
        table_size=[0.8, 1.2, 0.05],
        camera_names=["frontview", "agentview"],
        camera_size=84,
        obs_mode="combo",
    )

Options
-------
``robots``
    List of ``{"name", "gripper", "pos"}``. ``gripper`` is a robosuite
    gripper class name, ``"default"`` (the robot's own gripper, the default)
    or ``None`` (no gripper). ``pos`` is the ``[x, y, z]`` base position
    (world frame, ``z = 0`` is the floor); the first robot defaults to the
    table edge, every further robot needs an explicit ``pos``.
``objects``
    List of free-floating primitives ``{"type": "box" | "cylinder" |
    "sphere", "name", "size", "pos", "quat", "rgba", "mass" | "density",
    "friction"}``; ``size`` follows robosuite (box half-extents, cylinder
    ``[radius, half_height]``, sphere ``[radius]``). Objects without
    ``pos`` are placed at random, non-overlapping spots within 0.15 m of the
    table centre with a random yaw; ``quat`` (``[w, x, y, z]``) requires
    ``pos``. The table top is at ``z = 0.8``.
``goals``
    List of ``{"object", "target_pos", "threshold", "weight"}``. Each goal
    contributes dense shaping on the object's current distance to its
    target (0 at the reset pose), weighted by its normalised weight; the
    task succeeds once every object is within its own ``threshold`` of its
    target.
``controller``
    Arm controller type for every robot (e.g. ``"OSC_POSE"``, ``"IK_POSE"``).
``table_size``, ``camera_names``, ``camera_size``
    Table full size, robosuite cameras to render and their square resolution.

Mesh objects and custom arenas are not supported.
"""

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

from imbench.envs.single_arm_table_env import SingleArmTableEnv
from imbench.tasks.base_task import BaseTask, _deep_merge
from imbench.tasks.staged_reward import Stage, StagedReward, progress_from_reset


_PRIMITIVE_TYPES = {"box": "box", "cylinder": "cylinder", "sphere": "ball"}
_OBJECT_KEYS = {"type", "name", "size", "pos", "quat", "rgba", "mass", "density", "friction"}
_ROBOT_KEYS = {"name", "gripper", "pos"}
_GOAL_KEYS = {"object", "target_pos", "threshold", "weight"}


def _check_keys(kind: str, cfg: dict, allowed: set) -> None:
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unsupported {kind} keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def _primitive_volume(obj_type: str, size) -> float:
    if obj_type == "box":
        return 8.0 * size[0] * size[1] * size[2]
    if obj_type == "cylinder":
        return np.pi * size[0] ** 2 * 2.0 * size[1]
    return 4.0 / 3.0 * np.pi * size[0] ** 3


def _object_spec(cfg: dict) -> dict:
    """Translate a freeform object config into a ``SingleArmTableEnv`` object spec."""
    obj_type = cfg["type"]
    if obj_type == "mesh":
        raise NotImplementedError(
            "Freeform 'mesh' objects are not supported; use box/cylinder/sphere primitives."
        )
    if obj_type not in _PRIMITIVE_TYPES:
        raise ValueError(f"Unknown object type '{obj_type}'. "
                         f"Supported: {list(_PRIMITIVE_TYPES)}")
    _check_keys(f"{obj_type} object", cfg, _OBJECT_KEYS)
    missing = {"name", "size"} - set(cfg)
    if missing:
        raise ValueError(f"{obj_type} object is missing {sorted(missing)}")
    expected = {"box": 3, "cylinder": 2, "sphere": 1}[obj_type]
    size = [float(v) for v in cfg["size"]]
    if len(size) != expected:
        raise ValueError(f"{obj_type} '{cfg['name']}' needs {expected} size values, got {size}")
    if "mass" in cfg and "density" in cfg:
        raise ValueError(f"Object '{cfg['name']}': give either 'mass' or 'density', not both")
    if "quat" in cfg and "pos" not in cfg:
        raise ValueError(f"Object '{cfg['name']}': 'quat' requires 'pos' "
                         "(objects without 'pos' are placed with a random yaw)")
    if "mass" in cfg:
        density = float(cfg["mass"]) / _primitive_volume(obj_type, size)
    else:
        density = float(cfg.get("density", 100.0))
    spec = {
        "type": _PRIMITIVE_TYPES[obj_type],
        "name": cfg["name"],
        "size": size,
        "rgba": list(cfg.get("rgba", [1, 0, 0, 1])),
        "density": density,
        "joints": "default",
    }
    if "friction" in cfg:
        spec["friction"] = list(cfg["friction"])
    return spec


class FreeformTableEnv(SingleArmTableEnv):
    """``SingleArmTableEnv`` with base positions for every robot and fixed object poses.

    Objects named in ``fixed_object_poses`` (``name -> (pos, wxyz quat)``)
    are put at that pose on every reset; the placement sampler spreads the
    remaining objects around them.
    """

    def __init__(self, extra_robot_positions=(), fixed_object_poses=None, **kwargs):
        self._extra_robot_positions = [np.asarray(p, dtype=float) for p in extra_robot_positions]
        self._fixed_object_poses = {
            name: (np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))
            for name, (pos, quat) in (fixed_object_poses or {}).items()
        }
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        for robot, pos in zip(self.robots[1:], self._extra_robot_positions):
            robot.robot_model.set_base_xpos(pos)
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects([
                o for o in self._dynamic_objects if o.name not in self._fixed_object_poses
            ])

    def _reset_internal(self):
        # Skip SingleArmTableEnv's sampling: it is unaware of the fixed poses.
        ManipulationEnv._reset_internal(self)
        if self.deterministic_reset:
            return
        placements = {
            obj.name: (*self._fixed_object_poses[obj.name], obj)
            for obj in self._dynamic_objects if obj.name in self._fixed_object_poses
        }
        if self.placement_initializer is not None and self.placement_initializer.mujoco_objects:
            placements = self.placement_initializer.sample(fixtures=placements)
        for pos, quat, obj in placements.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))


# ── Task ─────────────────────────────────────────────────────────────

class FreeformV1(BaseTask):
    TASK_ID = "ib-m-freeform-v1"
    DESCRIPTION = "Configurable table-top sandbox: robots, primitive objects and weighted goals."
    MAX_EPISODE_STEPS = 500

    def __init__(self, *, robots=None, objects=None, goals=None,
                 controller="OSC_POSE", table_size=None, arena_xml=None,
                 camera_names=None, camera_size=84, **kwargs):
        if arena_xml is not None:
            raise NotImplementedError("Freeform custom arenas (arena_xml) are not supported.")
        self._robot_cfgs = [dict(r) for r in (robots or [{"name": "Panda"}])]
        for i, r in enumerate(self._robot_cfgs):
            _check_keys("robot", r, _ROBOT_KEYS)
            if "name" not in r:
                raise ValueError(f"Robot {i} is missing 'name'")
            if i > 0 and "pos" not in r:
                raise ValueError(f"Robot {i} ({r['name']}) needs an explicit 'pos'")
        self._object_cfgs = list(objects or [])
        self._object_specs = [_object_spec(o) for o in self._object_cfgs]
        names = [o["name"] for o in self._object_cfgs]
        if len(set(names)) != len(names):
            raise ValueError(f"Object names must be unique, got {names}")
        self._goal_cfgs = list(goals or [])
        for g in self._goal_cfgs:
            _check_keys("goal", g, _GOAL_KEYS)
            if g["object"] not in names:
                raise ValueError(f"Goal object '{g['object']}' is not one of the objects {names}")
        self._table_size = list(table_size or [0.8, 1.2, 0.05])

        # Controller and cameras are expressed as config overrides so
        # BaseTask builds and renders them like any YAML-defined task.
        camera_names = camera_names or ["frontview", "agentview"]
        overrides = {
            "controller": [{"robot": r["name"], "type": controller} for r in self._robot_cfgs],
            "obs": {"cameras": {
                cam: {"enabled": True, "robosuite_name": cam,
                      "width": camera_size, "height": camera_size}
                for cam in camera_names
            }},
        }
        if kwargs.get("config_overrides"):
            _deep_merge(overrides, kwargs["config_overrides"])
        kwargs["config_overrides"] = overrides

        self._staged_reward = self._build_staged_reward()
        self._goal_progress_at_reset = [0.0] * len(self._goal_cfgs)
        super().__init__(**kwargs)

    # ── abstract implementations ─────────────────────────────────────

    def _build_env(self):
        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras
        robot0_pos = self._robot_cfgs[0].get("pos")
        return self._make_robosuite_env(
            "FreeformTableEnv",
            objects=self._object_specs,
            robots=[r["name"] for r in self._robot_cfgs],
            gripper_types=[r.get("gripper", "default") for r in self._robot_cfgs],
            robot_offset=list(robot0_pos) if robot0_pos is not None else None,
            extra_robot_positions=[r["pos"] for r in self._robot_cfgs[1:]],
            fixed_object_poses={
                o["name"]: (o["pos"], o.get("quat", [1, 0, 0, 0]))
                for o in self._object_cfgs if "pos" in o
            },
            table_full_size=tuple(self._table_size),
            placement_config={
                "x_range": [-0.15, 0.15],
                "y_range": [-0.15, 0.15],
                "z_offset": 0.005,
                "ensure_valid_placement": True,
            },
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
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

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)

        for i in range(len(self._robot_cfgs)):
            for suffix in ("_eef_pos", "_eef_quat", "_gripper_qpos",
                           "_joint_pos", "_joint_vel"):
                key = f"robot{i}{suffix}"
                if key in raw_obs:
                    obs[key] = np.atleast_1d(raw_obs[key])

        if self.object_state_enabled:
            for o in self._object_cfgs:
                name = o["name"]
                if f"{name}_pos" in raw_obs:
                    obs[f"obj_{name}_pos"] = raw_obs[f"{name}_pos"]
                if f"{name}_quat" in raw_obs:
                    obs[f"obj_{name}_quat"] = raw_obs[f"{name}_quat"]

        return obs

    def _build_staged_reward(self):
        """One Stage per configured goal, weighted by the goal's normalised weight.

        Stages track the current distance (non-sticky), rescaled so the
        reset pose scores 0. ``success_fn`` requires every object to be
        within its own goal ``threshold`` of its target, not just a high
        weighted average.
        """
        if not self._goal_cfgs:
            return None
        weights = [float(g.get("weight", 1.0)) for g in self._goal_cfgs]
        if any(w < 0 for w in weights):
            raise ValueError(f"Goal weights must be non-negative, got {weights}")
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("At least one goal needs a positive 'weight'")
        stages = [
            Stage(
                f"goal_{i}_{goal['object']}",
                w / total_weight,
                lambda t, o, i=i, g=goal: progress_from_reset(
                    t._goal_progress(g, o), t._goal_progress_at_reset[i]),
                sticky=False,
            )
            for i, (goal, w) in enumerate(zip(self._goal_cfgs, weights))
        ]
        return StagedReward(
            stages=stages,
            success_fn=lambda t, o: all(t._goal_reached(g, o) for g in t._goal_cfgs),
        )

    def _goal_progress(self, goal, raw_obs):
        """Smooth per-goal score: 1.0 when at target, tapers to 0 by 0.5m (or 10x threshold)."""
        obj_name = goal["object"]
        pos_key = f"{obj_name}_pos"
        if pos_key not in raw_obs:
            return 0.0
        target = np.array(goal["target_pos"])
        threshold = goal.get("threshold", 0.05)
        dist = np.linalg.norm(raw_obs[pos_key] - target)
        return float(np.clip(1.0 - dist / max(threshold * 10, 0.5), 0.0, 1.0))

    def _goal_reached(self, goal, raw_obs):
        """True once the object is within its own configured threshold of target."""
        obj_name = goal["object"]
        pos_key = f"{obj_name}_pos"
        if pos_key not in raw_obs:
            return False
        target = np.array(goal["target_pos"])
        threshold = goal.get("threshold", 0.05)
        dist = np.linalg.norm(raw_obs[pos_key] - target)
        return dist <= threshold

    def _compute_task_score(self, raw_obs):
        if self._staged_reward is None:
            return 0.0
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._staged_reward is None:
            return False
        self._compute_task_score(raw_obs)
        return bool(self._stage_debug["_success"])

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._staged_reward is not None:
            self._staged_reward.reset()
            raw_obs = self._get_sim_observations(force_update=True)
            self._goal_progress_at_reset = [self._goal_progress(g, raw_obs) for g in self._goal_cfgs]
        return obs, info
