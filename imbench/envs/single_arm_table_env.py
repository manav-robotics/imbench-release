"""Generic single-arm table environment for IMBench tasks.

Extends robosuite's ``ManipulationEnv`` directly to give full control
over scene construction, object placement, observables, and reset logic
without the workarounds required when subclassing ``Lift``.
"""

from __future__ import annotations

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    BoxObject, BallObject, CylinderObject, CompositeBodyObject,
)
from robosuite.models.tasks import ManipulationTask

from imbench.objects import YCBObject, PendulumObject, SeesawObject, TiltingInclineObject, HingedFlapObject, SlidingLidBoxObject, TriangularPrismObject, TorusObject, ExtrasObject, BalancePlatformObject
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class SingleArmTableEnv(ManipulationEnv):
    """Single-arm table environment with configurable objects.

    Parameters
    ----------
    objects : list[dict], optional
        Each dict describes one dynamic object::

            {"type": "box"|"ball"|"cylinder",
             "name": str, "size": list, "rgba": list,
             "density": float, "friction": list,
             "joints": "default"|None}

        ``joints="default"`` creates a free joint (dynamic body).
        ``joints=None`` makes the object static (welded to world).
    static_objects : list[dict], optional
        Static scene objects (e.g. goal markers, walls, bins).
        Same schema as *objects* but always ``joints=None``.
    table_full_size : tuple, optional
        ``(x, y, z)`` table dimensions.
    table_friction : tuple, optional
        MuJoCo friction triple for the table.
    table_offset : tuple, optional
        ``(x, y, z)`` world-frame position of the table centre.
    robot_offset : list, optional
        ``[x, y, z]`` base position for the robot.
    placement_config : dict, optional
        ``{x_range, y_range, z_offset, reference_pos}`` for the
        placement sampler that randomises dynamic object positions.
    reward_shaping : bool, optional
        Whether to use dense robosuite reward (unused by IMBench
        tasks, kept for API compatibility).
    **kwargs
        Forwarded to :class:`ManipulationEnv`.
    """

    def __init__(
        self,
        objects: list[dict] | None = None,
        static_objects: list[dict] | None = None,
        table_full_size: tuple[float, float, float] = (0.8, 0.8, 0.05),
        table_friction: tuple[float, float, float] = (1.0, 5e-3, 1e-4),
        table_offset: tuple[float, float, float] = (0, 0, 0.8),
        robot_offset: list[float] | None = None,
        placement_config: dict | None = None,
        reward_shaping: bool = False,
        **kwargs,
    ) -> None:
        self._obj_specs = objects or []
        self._static_specs = static_objects or []
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.asarray(table_offset)
        self._robot_offset = robot_offset
        self._placement_config = placement_config or {}
        self.reward_shaping = reward_shaping

        # Populated in _load_model
        self._dynamic_objects = []
        self._static_objs = []
        self.placement_initializer = None

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  _load_model
    # ------------------------------------------------------------------

    def _load_model(self):
        """Build a single-arm table scene with the configured objects."""
        super()._load_model()

        # ── Position robot ──────────────────────────────────────────
        # robot_offset may be either [x, y, z] or [x, y, z, rx, ry, rz]
        # (Euler radians); the latter rotates the base around its own origin.
        if self._robot_offset is not None:
            offset = list(self._robot_offset)
            self.robots[0].robot_model.set_base_xpos(offset[:3])
            if len(offset) >= 6:
                self.robots[0].robot_model.set_base_ori(np.array(offset[3:6]))
        else:
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)

        # ── Arena ───────────────────────────────────────────────────
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ── Create dynamic objects ──────────────────────────────────
        self._dynamic_objects = []
        all_mj_objects = []

        for spec in self._obj_specs:
            obj = self._make_object(spec)
            self._dynamic_objects.append(obj)
            all_mj_objects.append(obj)

        # ── Create static objects ───────────────────────────────────
        self._static_objs = []
        for spec in self._static_specs:
            spec = dict(spec)
            spec["joints"] = None          # force static
            obj = self._make_object(spec)
            self._static_objs.append(obj)
            all_mj_objects.append(obj)

        # ── Placement initializer (dynamic objects only) ────────────
        pc = self._placement_config
        ref_pos = pc.get("reference_pos", [0.0, 0.0, float(self.table_offset[2])])
        sampler_objects = self._dynamic_objects if self._dynamic_objects else []

        if sampler_objects:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=sampler_objects,
                x_range=pc.get("x_range", [0.0, 0.0]),
                y_range=pc.get("y_range", [0.0, 0.0]),
                rotation=pc.get("rotation", None),
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=pc.get("ensure_valid_placement", True),
                reference_pos=np.array(ref_pos),
                z_offset=pc.get("z_offset", 0.01),
            )

        # ── Build ManipulationTask ──────────────────────────────────
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_mj_objects,
        )

    # ------------------------------------------------------------------
    #  _setup_references
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()
        # Store body IDs for all dynamic objects for fast obs lookup
        self._obj_body_ids = {}
        for obj in self._dynamic_objects:
            self._obj_body_ids[obj.name] = self.sim.model.body_name2id(
                obj.root_body
            )

    # ------------------------------------------------------------------
    #  _setup_observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()

        modality = "object"

        # Expose pos/quat for every dynamic object
        for obj in self._dynamic_objects:
            body_id = self._obj_body_ids[obj.name]
            obj_name = obj.name

            def _make_pos_sensor(bid=body_id):
                @sensor(modality=modality)
                def _pos(obs_cache):
                    return np.array(self.sim.data.body_xpos[bid])
                return _pos

            def _make_quat_sensor(bid=body_id):
                @sensor(modality=modality)
                def _quat(obs_cache):
                    return convert_quat(
                        np.array(self.sim.data.body_xquat[bid]), to="xyzw"
                    )
                return _quat

            pos_sensor = _make_pos_sensor()
            pos_sensor.__name__ = f"{obj_name}_pos"
            quat_sensor = _make_quat_sensor()
            quat_sensor.__name__ = f"{obj_name}_quat"

            observables[pos_sensor.__name__] = Observable(
                name=pos_sensor.__name__,
                sensor=pos_sensor,
                sampling_rate=self.control_freq,
            )
            observables[quat_sensor.__name__] = Observable(
                name=quat_sensor.__name__,
                sensor=quat_sensor,
                sampling_rate=self.control_freq,
            )

        # Gripper-to-first-object distance (useful for reaching rewards)
        if self._dynamic_objects:
            first_obj_name = self._dynamic_objects[0].name
            arm_prefixes = self._get_arm_prefixes(
                self.robots[0], include_robot_name=False
            )
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes):
                grip_sens = self._get_obj_eef_sensor(
                    full_pf,
                    f"{first_obj_name}_pos",
                    f"{arm_pf}gripper_to_{first_obj_name}_pos",
                    modality,
                )
                observables[grip_sens.__name__] = Observable(
                    name=grip_sens.__name__,
                    sensor=grip_sens,
                    sampling_rate=self.control_freq,
                )

        # Alias first object's observables as "cube_*" so task code can use
        # the generic "cube_pos" / "cube_quat" keys regardless of object name.
        if self._dynamic_objects:
            first = self._dynamic_objects[0].name
            if first != "cube":
                for suffix in ("_pos", "_quat"):
                    src_key = f"{first}{suffix}"
                    alias_key = f"cube{suffix}"
                    if src_key in observables:
                        observables[alias_key] = observables[src_key]

        return observables

    # ------------------------------------------------------------------
    #  _reset_internal
    # ------------------------------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()

        if not self.deterministic_reset and self.placement_initializer is not None:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.joints:
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )

    # ------------------------------------------------------------------
    #  reward / _check_success  (required by MujocoEnv)
    # ------------------------------------------------------------------

    def reward(self, action: np.ndarray | None = None) -> float:
        """Unused by IMBench (task-level reward in BaseTask.step)."""
        return 0.0

    def _check_success(self) -> bool:
        """Unused by IMBench (task-level termination in BaseTask.step)."""
        return False

    # ------------------------------------------------------------------
    #  Object access helpers
    # ------------------------------------------------------------------

    def get_object(self, name: str) -> object | None:
        """Return an object (dynamic or static) by name, or None."""
        for obj in self._dynamic_objects:
            if obj.name == name:
                return obj
        for obj in self._static_objs:
            if obj.name == name:
                return obj
        return None

    @property
    def cube(self):
        """Return the first dynamic object (the generic primary object for single-arm tasks)."""
        if self._dynamic_objects:
            return self._dynamic_objects[0]
        return None

    # ------------------------------------------------------------------
    #  Object factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_object(spec: dict) -> object:
        """Create a robosuite MuJoCo object from a spec dict."""
        obj_type = spec.get("type", "box")
        name = spec["name"]
        size = spec.get("size", [0.02, 0.02, 0.02])
        rgba = spec.get("rgba", [1, 0, 0, 1])
        density = spec.get("density", 100)
        friction = spec.get("friction", [1.0, 0.005, 0.0001])
        joints = spec.get("joints", "default")

        # Convert joints param
        joint_arg = joints if joints != "default" else [{"type": "free"}]

        if obj_type == "composite":
            # Build a CompositeBodyObject from sub-object specs.
            # spec must contain "sub_objects", "sub_locations", and
            # optionally "total_size".
            sub_objs = []
            for sub in spec["sub_objects"]:
                sub_type = sub.get("type", "box")
                if sub_type == "box":
                    sub_objs.append(BoxObject(
                        name=sub["name"], size=sub["size"],
                        rgba=sub.get("rgba", rgba),
                        density=sub.get("density", density),
                        friction=sub.get("friction", friction),
                        joints=None,
                    ))
                elif sub_type == "cylinder":
                    sub_objs.append(CylinderObject(
                        name=sub["name"], size=sub["size"],
                        rgba=sub.get("rgba", rgba),
                        density=sub.get("density", density),
                        friction=sub.get("friction", friction),
                        joints=None,
                    ))
                elif sub_type == "ball":
                    sub_objs.append(BallObject(
                        name=sub["name"], size=sub["size"],
                        rgba=sub.get("rgba", rgba),
                        density=sub.get("density", density),
                        friction=sub.get("friction", friction),
                        joints=None,
                    ))
            return CompositeBodyObject(
                name=name,
                objects=sub_objs,
                object_locations=spec["sub_locations"],
                object_quats=spec.get("sub_quats"),
                joints=joint_arg,
                total_size=spec.get("total_size"),
            )

        if obj_type == "box":
            return BoxObject(
                name=name, size=size, rgba=rgba,
                density=density, friction=friction,
                joints=joint_arg,
            )
        elif obj_type == "ball":
            ball_size = size if isinstance(size, (list, tuple)) else [size]
            return BallObject(
                name=name, size=ball_size,
                rgba=rgba, density=density, friction=friction,
                joints=joint_arg,
            )
        elif obj_type == "ycb":
            return YCBObject(
                name=name,
                ycb_name=spec["ycb_name"],
                density=density,
                joints=joint_arg,
            )
        elif obj_type == "cylinder":
            return CylinderObject(
                name=name, size=size if len(size) == 2 else size[:2],
                rgba=rgba, density=density, friction=friction,
                joints=joint_arg,
            )
        elif obj_type == "pendulum":
            return PendulumObject(
                name=name,
                stand_height=spec.get("stand_height", 0.25),
                arm_length=spec.get("arm_length", 0.20),
                bob_size=spec.get("bob_size", 0.03),
                bob_rgba=spec.get("bob_rgba", rgba),
                damping=spec.get("damping", 0.0),
                bob_shape=spec.get("bob_shape", "sphere"),
            )
        elif obj_type == "prism":
            return TriangularPrismObject(
                name=name, size=size,
                rgba=rgba, density=density, friction=friction,
                joints=joint_arg,
            )
        elif obj_type == "tilting_incline":
            return TiltingInclineObject(
                name=name,
                plate_length=spec.get("plate_length", 0.55),
                plate_half_width=spec.get("plate_half_width", 0.225),
                plate_half_thickness=spec.get("plate_half_thickness", 0.004),
                plate_rgba=spec.get("plate_rgba", rgba),
                plate_friction=spec.get("plate_friction", (1.0, 0.005, 0.0001)),
                damping=spec.get("damping", 100.0),
                frictionloss=spec.get("frictionloss", 0.0),
            )
        elif obj_type == "hinged_flap":
            keys = ("half_x", "half_y", "thickness", "rail_height", "rail_width", "lip",
                    "flap_rgba", "rail_rgba", "flap_mass", "open_limit_deg", "damping", "end_walls")
            return HingedFlapObject(name=name, **{k: spec[k] for k in keys if k in spec})
        elif obj_type == "sliding_lid_box":
            keys = ("inner_half", "wall", "height", "lid_thickness", "travel", "knob_radius",
                    "knob_half_height", "box_rgba", "lid_rgba", "damping", "frictionloss")
            return SlidingLidBoxObject(name=name, **{k: spec[k] for k in keys if k in spec})
        elif obj_type == "seesaw":
            return SeesawObject(
                name=name,
                plank_half_length=spec.get("plank_half_length", 0.225),
                plank_half_width=spec.get("plank_half_width", 0.03),
                plank_half_thickness=spec.get("plank_half_thickness", 0.004),
                fulcrum_height=spec.get("fulcrum_height", 0.04),
                stopper_height=spec.get("stopper_height", 0.015),
                plank_rgba=spec.get("plank_rgba", rgba),
                plank_friction=spec.get("plank_friction", (0.15, 0.005, 0.0001)),
                damping=spec.get("damping", 0.01),
                frictionloss=spec.get("frictionloss", 0.001),
            )
        elif obj_type == "torus":
            return TorusObject(
                name=name,
                major_radius=spec.get("major_radius", 0.035),
                minor_radius=spec.get("minor_radius", 0.005),
                n_segments=spec.get("n_segments", 16),
                rgba=rgba, density=density, friction=friction,
                joints=joint_arg,
            )
        elif obj_type == "open_cylinder":
            return ExtrasObject(
                name=name,
                extras_name=spec.get("extras_name", "open_cup"),
                joints=joint_arg,
                duplicate_collision_geoms=False,
            )
        elif obj_type == "extras":
            return ExtrasObject(
                name=name,
                extras_name=spec.get("extras_name", ""),
                joints=joint_arg,
                duplicate_collision_geoms=spec.get(
                    "duplicate_collision_geoms", False),
                xml_path=spec.get("xml_path"),
            )
        elif obj_type == "balance_platform":
            return BalancePlatformObject(
                name=name,
                ridge_half_length=spec.get("ridge_half_length", 0.06),
                ridge_half_width=spec.get("ridge_half_width", 0.003),
                ridge_height=spec.get("ridge_height", 0.02),
                base_half_size=tuple(spec.get("base_half_size", [0.04, 0.04, 0.005])),
                friction=tuple(spec.get("friction", [1.0, 0.005, 0.0001])),
            )
        else:
            raise ValueError(f"Unknown object type: {obj_type}")
