"""Edge-slide environment v2 — single-arm variant with a flat cylinder object.

Identical to EdgeSlideEnv except the book (BoxObject) is replaced with a
flat CylinderObject whose radius and half-height are configurable.

The object is still named ``"cube"`` internally so that all Lift-level
observables (``cube_pos``, ``cube_quat``, gripper-to-cube sensors) continue
to work without any modification.
"""

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class EdgeSlideEnvV2(Lift):
    """Single-arm table environment with a flat cylinder object.

    Parameters
    ----------
    cylinder_radius : float, optional
        Radius of the cylinder.  Default ``0.075``.
    cylinder_half_height : float, optional
        Half-height (half-thickness) of the cylinder.  Default ``0.0015``
        (3 mm thick, a tenth of the edge-slide v1 book's thickness).
    cylinder_density : float, optional
        Density (kg/m³).  Default ``500``.
    cylinder_friction : tuple of float, optional
        ``(sliding, torsional, rolling)`` friction.  Default ``(0.6, 0.005, 0.0001)``.
    cylinder_color : list of float, optional
        RGBA color.  Default ``[0.0, 0.3, 0.7, 1.0]`` (blue).
    **kwargs
        Forwarded to :class:`Lift`.
    """

    def __init__(
        self,
        cylinder_radius=0.075,
        cylinder_half_height=0.0015,
        cylinder_density=500,
        cylinder_friction=(0.6, 0.005, 0.0001),
        cylinder_color=None,
        robot_offset=None,
        robot_ori=None,
        table_offset=None,
        **kwargs,
    ):
        self.cylinder_radius = cylinder_radius
        self.cylinder_half_height = cylinder_half_height
        self.cylinder_density = cylinder_density
        self.cylinder_friction = tuple(cylinder_friction)
        self.cylinder_color = list(cylinder_color) if cylinder_color is not None else [0.0, 0.3, 0.7, 1.0]
        self._robot_offset = robot_offset
        self._robot_ori = robot_ori
        self._custom_table_offset = np.array(table_offset) if table_offset is not None else None
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  Model building
    # ------------------------------------------------------------------

    def _load_model(self):
        """Build scene identical to Lift but with a flat cylinder instead of a cube."""
        if self._custom_table_offset is not None:
            self.table_offset = self._custom_table_offset

        # Call grandparent (_load_model on ManipulationEnv) to set up robots.
        super(Lift, self)._load_model()

        # ── Position robot ──────────────────────────────────────────
        if self._robot_offset is not None:
            self.robots[0].robot_model.set_base_xpos(self._robot_offset)
        else:
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
        if self._robot_ori is not None:
            self.robots[0].robot_model.set_base_ori(np.array(self._robot_ori))

        # ── Arena ───────────────────────────────────────────────────
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # ── Cylinder object (replaces the book) ────────────────────
        # CylinderObject size = [radius, half_height]
        self.cube = CylinderObject(
            name="cube",
            size=[self.cylinder_radius, self.cylinder_half_height],
            rgba=self.cylinder_color,
            density=self.cylinder_density,
            friction=list(self.cylinder_friction),
        )

        # ── Placement initializer ──────────────────────────────────
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.002,
            )

        # ── Assemble task ──────────────────────────────────────────
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )