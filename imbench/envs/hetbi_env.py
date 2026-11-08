"""Two-arm environment with custom YCB objects on a single shared table.

Inherits from robosuite's ``TwoArmEnv`` for dual-arm robot helpers, and
builds the scene from scratch: a single ``TableArena``, a manipulated
**payload** object, and an optional static **fixture** (shelf, rack, box,
tray) — all placed on the same table surface.

No bins, lids, or trash objects are created — the scene contains only the
payload and fixture, giving full control over object placement and naming.
"""

from __future__ import annotations

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor


def _rest_height(obj, quat_wxyz) -> float:
    """Height of *obj*'s origin above the lowest corner of its bounding box
    when rotated by *quat_wxyz*, i.e. the origin height that rests it on a surface."""
    half = obj.get_bounding_box_half_size()
    z_range = (obj.bottom_offset[2], obj.top_offset[2])
    corners = np.array([
        [sx * half[0], sy * half[1], z]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for z in z_range
    ])
    rot = T.quat2mat(T.convert_quat(np.asarray(quat_wxyz, dtype=np.float64), to="xyzw"))
    return float(-(corners @ rot.T)[:, 2].min())


class HetBiEnv(TwoArmEnv):
    """Two-arm environment with a swappable payload and optional fixture.

    Both robots and all objects share a **single table**.  The payload
    (e.g. a YCB mug) is placed on the right side (x ≈ +0.1) and the
    optional fixture (shelf, rack, …) on the left side (x ≈ −0.25).

    Parameters
    ----------
    payload_object : MujocoObject
        The object to manipulate.
    fixture_object : MujocoObject | None
        Optional static fixture placed on the table.
    table_full_size : tuple
        ``(length_x, width_y, thickness_z)`` of the table.
    table_offset : tuple
        ``(x, y, z)`` position of the table centre (z is the surface height).
    table_friction : tuple
        Sliding / torsional / rolling friction for the table surface.
    use_object_obs : bool
        If ``True``, expose object-state observations (positions, quats).
    payload_pos : array-like | None
        Custom ``[x, y, z]`` for the payload, or ``[x, y]`` to rest it on
        the table in its ``payload_quat`` pose.  ``None`` → ``[0.1, 0]``.
    payload_quat : array-like | None
        Custom ``[w, x, y, z]`` for the payload.  ``None`` → upright.
    fixture_pos : array-like | None
        Custom ``[x, y, z]`` for the fixture.  ``None`` → ``x = -0.25``,
        ``y = 0`` and resting on the table in its ``fixture_quat`` pose.
    fixture_quat : array-like | None
        Custom ``[w, x, y, z]`` for the fixture.  ``None`` → +90° about Z.
    **kwargs
        Forwarded to ``TwoArmEnv`` (robots, grippers, controller_configs, …).
    """

    def __init__(
        self,
        payload_object: object,
        fixture_object: object | None = None,
        table_full_size: tuple[float, float, float] = (1.0, 0.75, 0.05),
        table_offset: tuple[float, float, float] = (0, 0, 0.9),
        table_friction: tuple[float, float, float] = (1.0, 5e-3, 1e-4),
        use_object_obs: bool = True,
        payload_pos: list[float] | None = None,
        payload_quat: list[float] | None = None,
        fixture_pos: list[float] | None = None,
        fixture_quat: list[float] | None = None,
        robot_offset: list | None = None,
        robot_rotation_z: float | list[float] = 0.0,
        **kwargs,
    ) -> None:
        # Store BEFORE super().__init__ — it triggers _load_model
        self._payload = payload_object
        self._fixture = fixture_object

        self.table_full_size = np.array(table_full_size, dtype=np.float64)
        self.table_offset = np.array(table_offset, dtype=np.float64)
        self.table_friction = table_friction

        # Per-object placement overrides (computed at reset if None)
        self._payload_pos = np.array(payload_pos) if payload_pos is not None else None
        self._payload_quat = np.array(payload_quat) if payload_quat is not None else None
        self._fixture_pos = np.array(fixture_pos) if fixture_pos is not None else None
        self._fixture_quat = (
            np.array(fixture_quat) if fixture_quat is not None
            # +90° around Z so the open front faces +x (toward camera)
            else np.array([0.7071068, 0.0, 0.0, 0.7071068])
        )

        self.use_object_obs = use_object_obs

        # Optional per-robot base position overrides: list of [x, y, z] or None
        self._robot_offset = robot_offset
        # Extra in-place Z rotation (degrees) per robot; scalar or per-robot list
        if isinstance(robot_rotation_z, (list, tuple)):
            self._robot_rotation_z = [np.radians(float(r)) for r in robot_rotation_z]
        else:
            self._robot_rotation_z = [np.radians(float(robot_rotation_z))] * 2

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  Model building
    # ------------------------------------------------------------------

    def _load_model(self):
        """Build a single-table scene with payload and optional fixture."""
        super()._load_model()

        # ── Position robots ─────────────────────────────────────────
        if self._robot_offset is not None:
            # Explicit per-robot positions from YAML config
            for robot, pos in zip(self.robots, self._robot_offset):
                robot.robot_model.set_base_xpos(pos[:3])
                if len(pos) > 3:
                    robot.robot_model.set_base_ori(pos[3:6])
        elif self.env_configuration == "single-robot":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
        elif self.env_configuration == "opposed":
            # ±90° Z rotation only — no extra Y offsets.  The rotation
            # of the base_xpos_offset vector sites the robots on
            # opposite sides of the table along Y.
            for robot, rotation, extra_z in zip(
                self.robots, (np.pi / 2, -np.pi / 2),
                self._robot_rotation_z,
            ):
                xpos = robot.robot_model.base_xpos_offset["table"](
                    self.table_full_size[0]
                )
                rot = np.array((0, 0, rotation))
                xpos = T.euler2mat(rot) @ np.array(xpos)
                robot.robot_model.set_base_xpos(xpos)
                # Orientation: base rotation + optional in-place extra Z
                ori = np.array((0, 0, rotation + extra_z))
                robot.robot_model.set_base_ori(ori)
        else:  # parallel
            for robot, offset, extra_z in zip(
                self.robots, (-0.6, 0.6),
                self._robot_rotation_z,
            ):
                xpos = robot.robot_model.base_xpos_offset["table"](
                    self.table_full_size[0]
                )
                xpos = np.array(xpos) + np.array((0, offset, 0))
                robot.robot_model.set_base_xpos(xpos)
                if extra_z:
                    robot.robot_model.set_base_ori(np.array([0, 0, extra_z]))

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
        all_objects = [self._payload]
        if self._fixture is not None:
            all_objects.append(self._fixture)

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    # ------------------------------------------------------------------
    #  References
    # ------------------------------------------------------------------

    @property
    def payload_body_id(self) -> int:
        """MuJoCo body ID of the payload object (set after ``reset()``)."""
        return self._payload_body_id

    @property
    def fixture_body_id(self) -> int | None:
        """MuJoCo body ID of the fixture object, or ``None`` if no fixture."""
        return getattr(self, "_fixture_body_id", None)

    def _setup_references(self):
        super()._setup_references()
        self._payload_body_id = self.sim.model.body_name2id(
            self._payload.root_body
        )
        if self._fixture is not None:
            self._fixture_body_id = self.sim.model.body_name2id(
                self._fixture.root_body
            )

    # ------------------------------------------------------------------
    #  Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            pf = self._payload.naming_prefix  # e.g. "mug_"

            @sensor(modality="object")
            def payload_pos(obs_cache):
                return np.array(
                    self.sim.data.body_xpos[self._payload_body_id]
                )

            @sensor(modality="object")
            def payload_quat(obs_cache):
                return T.convert_quat(
                    self.sim.data.body_xquat[self._payload_body_id],
                    to="xyzw",
                )

            sensors = [payload_pos, payload_quat]
            names = [f"{pf}pos", f"{pf}quat"]

            if self._fixture is not None:
                fpf = self._fixture.naming_prefix

                @sensor(modality="object")
                def fixture_pos(obs_cache):
                    return np.array(
                        self.sim.data.body_xpos[self._fixture_body_id]
                    )

                sensors.append(fixture_pos)
                names.append(f"{fpf}pos")

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        super()._reset_internal()

        table_z = self.table_offset[2]  # surface height (0.9)

        # ── Place payload (right side of table, x ≈ +0.1) ──────────
        pay_quat = (
            self._payload_quat.copy()
            if self._payload_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0])
        )
        pay_pos = self._payload_pos.copy() if self._payload_pos is not None else np.array([0.1, 0.0])
        if len(pay_pos) == 2:
            pay_pos = np.array([
                pay_pos[0], pay_pos[1],
                table_z + _rest_height(self._payload, pay_quat) + 0.002,
            ])
        self.sim.data.set_joint_qpos(
            self._payload.joints[0],
            np.concatenate([pay_pos, pay_quat]),
        )

        # ── Place fixture (left side of table, x ≈ −0.25) ──────────
        if self._fixture is not None:
            if self._fixture_pos is not None:
                fix_pos = self._fixture_pos.copy()
            else:
                fix_pos = np.array([
                    -0.25, 0.0,
                    table_z + _rest_height(self._fixture, self._fixture_quat) + 0.001,
                ])
            # Clamp the fixture's XY so its full footprint stays inside the
            # table, since an unclamped ``fixture_pos`` can leave the
            # rack/shelf hanging off the table edge or outside both
            # robots' reachable workspace.  We bound the centre using the
            # fixture's own footprint relative to its centre (taken from
            # ``_compute_fixture_bounds``) and the table half-extents.
            fix_pos = self._clamp_fixture_pos(fix_pos)
            bid = self._fixture_body_id
            self.sim.model.body_pos[bid] = fix_pos
            self.sim.model.body_quat[bid] = self._fixture_quat

        # ── Store goal position and fixture bounds for task-level scoring ──
        # _target_goal_pos: fixture centre at mid-height — used for dense
        # reward shaping (distance guidance toward the fixture).
        # _fixture_bounds: inner axis-aligned bounding box of the fixture so
        # that a valid placement inside counts as success.
        if self._fixture is not None:
            self._target_goal_pos = np.array([
                fix_pos[0], fix_pos[1], fix_pos[2] + 0.12,
            ])
            self._target_goal_quat = self._fixture_quat.copy()
            self._fixture_bounds = self._compute_fixture_bounds(fix_pos)
        else:
            self._target_goal_pos = np.array([
                -0.25, 0.0, table_z + 0.12,
            ])
            self._target_goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self._fixture_bounds = None

        self.sim.forward()

    def _clamp_fixture_pos(self, fix_pos):
        """Clamp ``fix_pos`` so the fixture footprint stays inside the table.

        The bowl-table / plate-shelf YAMLs occasionally placed the rack or
        shelf with its outer wall hanging off the table (e.g. fixture-y at
        +0.36 with a ±0.185 half-width on a 0.75 m table).  We clamp the
        centre using the same footprint that ``_compute_fixture_bounds``
        reports for that fixture, plus a small interior margin so the
        fixture sits fully on the table top and within the natural
        reachable workspace of the two arms.
        """
        fix_pos = np.asarray(fix_pos, dtype=float).copy()

        # Use the fixture's own bounds (relative to its centre) as the
        # footprint half-extents that must fit inside the table.
        sample = self._compute_fixture_bounds(fix_pos)
        fp_x_lo = fix_pos[0] - sample["x"][0]
        fp_x_hi = sample["x"][1] - fix_pos[0]
        fp_y_lo = fix_pos[1] - sample["y"][0]
        fp_y_hi = sample["y"][1] - fix_pos[1]

        margin = 0.02  # keep the fixture at least 2 cm inside the edge
        tx = float(self.table_full_size[0]) * 0.5 - margin
        ty = float(self.table_full_size[1]) * 0.5 - margin
        cx = float(self.table_offset[0])
        cy = float(self.table_offset[1])

        x_lo = cx - tx + fp_x_lo
        x_hi = cx + tx - fp_x_hi
        y_lo = cy - ty + fp_y_lo
        y_hi = cy + ty - fp_y_hi

        if x_lo > x_hi:
            x_lo = x_hi = 0.5 * (x_lo + x_hi)
        if y_lo > y_hi:
            y_lo = y_hi = 0.5 * (y_lo + y_hi)

        fix_pos[0] = float(np.clip(fix_pos[0], x_lo, x_hi))
        fix_pos[1] = float(np.clip(fix_pos[1], y_lo, y_hi))
        return fix_pos

    def _compute_fixture_bounds(self, fix_pos):
        """Compute axis-aligned inner bounds for the current fixture.

        Returns a dict ``{"x": (lo, hi), "y": (lo, hi), "z": (lo, hi)}``
        representing the interior volume of the fixture in world coordinates,
        accounting for fixture-specific geometry and orientation.
        """
        fixture_name = self._fixture.name.replace("_target", "")

        if fixture_name == "shelf":
            # Shelf rotated +90° Z: local x→world y, local y→world -x
            # dims from deps/extras/shelf.xml (W 0.40, D 0.20, H 0.28);
            # dividers at world y = ±0.065, so y spans the centre slot only
            # (less the plate half-thickness).
            return {
                "x": (fix_pos[0] - 0.085, fix_pos[0] + 0.090),
                "y": (fix_pos[1] - 0.06,  fix_pos[1] + 0.06),
                "z": (fix_pos[2] + 0.01,  fix_pos[2] + 0.27),
            }
        elif fixture_name == "rack":
            # Rack rotated +90° Z: same footprint as shelf, shorter height
            # dims from deps/extras/rack.xml (W 0.40, D 0.20, H 0.15)
            return {
                "x": (fix_pos[0] - 0.085, fix_pos[0] + 0.090),
                "y": (fix_pos[1] - 0.185, fix_pos[1] + 0.185),
                "z": (fix_pos[2] + 0.01,  fix_pos[2] + 0.15),
            }
        elif fixture_name == "slot":
            # Slot rotated +90° about world X (fixture_quat [w,x,y,z] =
            # [0.7071, 0.7071, 0, 0]): local +X → world +X,
            # local +Y → world +Z, local +Z → world -Y.
            # Interior bounds verified empirically from geom world poses;
            # dims from deps/extras/slot.xml (channel ~0.23×0.18×0.10
            # in local x/z/y after accounting for 0.005 wall thickness).
            return {
                "x": (fix_pos[0] - 0.115, fix_pos[0] + 0.115),
                "y": (fix_pos[1] - 0.19,  fix_pos[1] - 0.01),
                "z": (fix_pos[2] - 0.085, fix_pos[2] + 0.085),
            }
        else:
            # Fallback: use shelf-like bounds
            return {
                "x": (fix_pos[0] - 0.085, fix_pos[0] + 0.090),
                "y": (fix_pos[1] - 0.185, fix_pos[1] + 0.185),
                "z": (fix_pos[2] + 0.01,  fix_pos[2] + 0.27),
            }

    # ------------------------------------------------------------------
    #  Reward / success (task layer handles scoring)
    # ------------------------------------------------------------------

    def reward(self, action: np.ndarray | None = None) -> float:
        """Unused by IMBench (task-level reward in BaseTask.step)."""
        return 0.0

    def _check_success(self) -> bool:
        return False
