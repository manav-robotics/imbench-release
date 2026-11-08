"""Edge-slide environment — single-arm Lift variant with a flat book object.

Replaces the default small cube from robosuite's ``Lift`` with a flat
``BoxObject`` whose dimensions match the book used in the edge-slide task.
This ensures MuJoCo compiles the model with the correct collision geometry
(geom_rbound, inertia, etc.) from the start, avoiding the broadphase
mismatch that occurs when resizing geoms at runtime.

The object is still named ``"cube"`` internally so that all Lift-level
observables (``cube_pos``, ``cube_quat``, gripper-to-cube sensors) and
references continue to work without modification.
"""

import numpy as np

from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class EdgeSlideEnv(Lift):
    """Single-arm table environment with a flat book object.

    Parameters
    ----------
    book_size : tuple of float, optional
        ``(half_x, half_y, half_z)`` dimensions of the book.  Defaults to
        ``(0.10, 0.075, 0.015)`` — the upper bound of the randomisation
        range so that the compiled bounding radius is always large enough.
    book_density : float, optional
        Density of the book (kg/m³).  Default ``500``.
    book_friction : tuple of float, optional
        ``(sliding, torsional, rolling)`` friction.  Default ``(0.6, 0.005, 0.0001)``.
    book_color : list of float, optional
        RGBA color for the book object.  Default ``[1, 0, 0, 1]`` (red).
    **kwargs
        Forwarded to :class:`Lift`.
    """

    def __init__(
        self,
        book_size=(0.10, 0.075, 0.015),
        book_density=500,
        book_friction=(0.6, 0.005, 0.0001),
        book_color=None,
        robot_offset=None,
        robot_ori=None,
        table_offset=None,
        **kwargs,
    ):
        self.book_size = tuple(book_size)
        self.book_density = book_density
        self.book_friction = tuple(book_friction)
        self.book_color = list(book_color) if book_color is not None else [1, 0, 0, 1]
        self._robot_offset = robot_offset
        self._robot_ori = robot_ori
        # Lift hardcodes table_offset as an instance attribute (not a ctor kwarg).
        # Store it here and apply at the top of _load_model() before TableArena is built.
        self._custom_table_offset = np.array(table_offset) if table_offset is not None else None
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    #  Model building — override only the object creation
    # ------------------------------------------------------------------

    def _load_model(self):
        """Build scene identical to Lift but with a flat book instead of a cube."""
        # Apply custom table_offset before anything else reads self.table_offset.
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

        # ── Book object (replaces the small cube) ──────────────────
        self.cube = BoxObject(
            name="cube",
            size=list(self.book_size),
            rgba=self.book_color,
            density=self.book_density,
            friction=list(self.book_friction),
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
