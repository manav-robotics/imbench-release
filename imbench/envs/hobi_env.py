"""Two-arm homogeneous bimanual environment with multiple objects.

Inherits from robosuite's ``TwoArmEnv`` and builds a scene with a single
``TableArena``, multiple **dynamic** objects (free‐joint), and multiple
**static** fixtures — all placed on the same table.  Both robots use the
same gripper type (typically ``PandaGripper``).

This environment is the base for all ``ib-p1-plate-shelf-hobi-*`` and ``ib-m-*`` bimanual tasks.
"""

from __future__ import annotations

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor


class HoBiEnv(TwoArmEnv):
    """Two-arm environment with multiple objects on a shared table.

    Both robots share identical grippers.  The scene is populated with
    ``dynamic_objects`` (free‐joint, manipulable) and ``static_objects``
    (no joint, fixed in world).

    Parameters
    ----------
    dynamic_objects : list[MujocoObject]
        Objects with free joints that the robots can manipulate.
    static_objects : list[MujocoObject]
        Fixtures with no joints (welded to world).
    dynamic_positions : list[array-like] | None
        Per-object ``[x, y, z]`` initial positions.  ``None`` → auto.
    dynamic_quats : list[array-like] | None
        Per-object ``[w, x, y, z]`` initial orientations.  ``None`` → identity.
    static_positions : list[array-like] | None
        Per-fixture ``[x, y, z]`` positions.  ``None`` → auto.
    static_quats : list[array-like] | None
        Per-fixture ``[w, x, y, z]`` orientations.  ``None`` → identity.
    table_full_size : tuple
        ``(length_x, width_y, thickness_z)`` of the table.
    table_offset : tuple
        ``(x, y, z)`` position of the table centre.
    table_friction : tuple
        Sliding / torsional / rolling friction for the table surface.
    use_object_obs : bool
        If ``True``, expose per-object observables (positions, quats).
    **kwargs
        Forwarded to ``TwoArmEnv``.
    """

    def __init__(
        self,
        dynamic_objects: list | None = None,
        static_objects: list | None = None,
        dynamic_positions: list[list[float]] | None = None,
        dynamic_quats: list[list[float]] | None = None,
        static_positions: list[list[float]] | None = None,
        static_quats: list[list[float]] | None = None,
        table_full_size: tuple[float, float, float] = (1.0, 0.75, 0.05),
        table_offset: tuple[float, float, float] = (0, 0, 0.9),
        table_friction: tuple[float, float, float] = (1.0, 5e-3, 1e-4),
        robot_offset: list | None = None,
        use_object_obs: bool = True,
        **kwargs,
    ) -> None:
        self._dynamic_objects = dynamic_objects or []
        self._static_objects = static_objects or []

        self._dynamic_positions = dynamic_positions
        self._dynamic_quats = dynamic_quats
        self._static_positions = static_positions
        self._static_quats = static_quats

        self.table_full_size = np.array(table_full_size, dtype=np.float64)
        self.table_offset = np.array(table_offset, dtype=np.float64)
        self.table_friction = table_friction
        self.robot_offset = robot_offset
        self.use_object_obs = use_object_obs

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  Model building
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        # ── Position robots ─────────────────────────────────────────
        if self.robot_offset is not None:
            # Explicit per-robot positions from config
            for robot, pos in zip(self.robots, self.robot_offset):
                robot.robot_model.set_base_xpos(pos[:3])
                if len(pos) > 3:
                    robot.robot_model.set_base_ori(pos[3:])
        elif self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
        elif self.env_configuration == "opposed":
            for robot, rotation in zip(
                self.robots, (np.pi / 2, -np.pi / 2)
            ):
                xpos = robot.robot_model.base_xpos_offset["table"](
                    self.table_full_size[0]
                )
                rot = np.array((0, 0, rotation))
                xpos = T.euler2mat(rot) @ np.array(xpos)
                robot.robot_model.set_base_xpos(xpos)
                robot.robot_model.set_base_ori(rot)
        else:  # parallel
            for robot, offset in zip(self.robots, (-0.6, 0.6)):
                xpos = robot.robot_model.base_xpos_offset["table"](
                    self.table_full_size[0]
                )
                xpos = np.array(xpos) + np.array((0, offset, 0))
                robot.robot_model.set_base_xpos(xpos)

        # ── Arena (single table) ────────────────────────────────────
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            has_legs=True,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Cameras
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.8894354364730311, -3.5e-08, 1.7383813133506494],
            quat=[
                0.6530981063842773,
                0.2710406184196472,
                0.27104079723358154,
                0.6530979871749878,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="shouldercamera0",
            pos=[0.4430096057365183, -1.0697399743660143, 1.3639950119362048],
            quat=[
                0.804057240486145,
                0.5531665086746216,
                0.11286306381225586,
                0.18644218146800995,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="shouldercamera1",
            pos=[-0.40900714, 0.96137226, 1.3084073],
            quat=[
                0.15484197437763214,
                0.12077208608388901,
                -0.5476858019828796,
                -0.8133130073547363,
            ],
        )

        # ── Scene objects ───────────────────────────────────────────
        all_objects = list(self._dynamic_objects) + list(self._static_objects)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    # ------------------------------------------------------------------
    #  References
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()
        self._dynamic_body_ids = []
        for obj in self._dynamic_objects:
            self._dynamic_body_ids.append(
                self.sim.model.body_name2id(obj.root_body)
            )
        self._static_body_ids = []
        for obj in self._static_objects:
            self._static_body_ids.append(
                self.sim.model.body_name2id(obj.root_body)
            )

    # ------------------------------------------------------------------
    #  Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            for idx, obj in enumerate(self._dynamic_objects):
                pf = obj.naming_prefix
                body_id = self._dynamic_body_ids[idx]

                # Capture body_id in closure
                def _make_pos_sensor(_bid):
                    @sensor(modality="object")
                    def obj_pos(obs_cache):
                        return np.array(self.sim.data.body_xpos[_bid])
                    return obj_pos

                def _make_quat_sensor(_bid):
                    @sensor(modality="object")
                    def obj_quat(obs_cache):
                        return T.convert_quat(
                            self.sim.data.body_xquat[_bid], to="xyzw"
                        )
                    return obj_quat

                pos_s = _make_pos_sensor(body_id)
                quat_s = _make_quat_sensor(body_id)

                pos_name = f"{pf}pos"
                quat_name = f"{pf}quat"

                observables[pos_name] = Observable(
                    name=pos_name,
                    sensor=pos_s,
                    sampling_rate=self.control_freq,
                )
                observables[quat_name] = Observable(
                    name=quat_name,
                    sensor=quat_s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()
        table_z = self.table_offset[2]

        # ── Place dynamic objects ───────────────────────────────────
        for idx, obj in enumerate(self._dynamic_objects):
            if self._dynamic_positions and idx < len(self._dynamic_positions):
                pos = np.array(self._dynamic_positions[idx], dtype=np.float64)
                if len(pos) == 2:
                    pos = np.array([
                        pos[0], pos[1],
                        table_z + abs(obj.bottom_offset[2]) + 0.002,
                    ])
            else:
                # Default: spread objects across the table
                x_offset = -0.15 + 0.1 * idx
                pos = np.array([
                    x_offset, 0.0,
                    table_z + abs(obj.bottom_offset[2]) + 0.002,
                ])

            if self._dynamic_quats and idx < len(self._dynamic_quats):
                quat = np.array(self._dynamic_quats[idx], dtype=np.float64)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])

            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([pos, quat]),
            )

        # ── Place static objects ────────────────────────────────────
        for idx, obj in enumerate(self._static_objects):
            bid = self._static_body_ids[idx]
            if self._static_positions and idx < len(self._static_positions):
                pos = np.array(self._static_positions[idx], dtype=np.float64)
            else:
                pos = np.array([
                    -0.25 + 0.15 * idx, 0.0,
                    table_z + abs(obj.bottom_offset[2]) + 0.001,
                ])
            if self._static_quats and idx < len(self._static_quats):
                quat = np.array(self._static_quats[idx], dtype=np.float64)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])

            self.sim.model.body_pos[bid] = pos
            self.sim.model.body_quat[bid] = quat

        self.sim.forward()

    # ------------------------------------------------------------------
    #  Reward / success (task layer handles scoring)
    # ------------------------------------------------------------------

    def reward(self, action: np.ndarray | None = None) -> float:
        return 0.0

    def _check_success(self) -> bool:
        return False
