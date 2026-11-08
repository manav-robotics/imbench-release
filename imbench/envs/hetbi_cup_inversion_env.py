"""HetBi environment variant for the cup-inversion task.

Extends :class:`HetBiEnv` with a visual-only green goal marker on the
table surface.  The marker is a thin, welded, non-colliding ``BoxObject``
(no free joint) whose footprint defines the region the cup must end up
inside for task success.  Pose is fully YAML-configurable via ``goal_area``
in the task config.
"""

from __future__ import annotations

import numpy as np

from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from imbench.envs.hetbi_env import HetBiEnv


class HetBiCupInversionEnv(HetBiEnv):
    """HetBi env with a welded visual goal marker on the table.

    Parameters
    ----------
    goal_pos : array-like | None
        ``[x, y]`` or ``[x, y, z]`` for the goal marker centre.  When z
        is omitted the marker is placed flush with the table surface.
        ``None`` → no marker is added.
    goal_half_size : tuple of float
        ``(half_x, half_y)`` footprint of the square goal region.
        Default ``(0.05, 0.05)`` → 10×10 cm region.
    goal_rgba : array-like
        RGBA color of the marker.  Default green with alpha 0.5.
    """

    def __init__(
        self,
        *args,
        goal_pos: list[float] | None = None,
        goal_half_size: tuple[float, float] = (0.05, 0.05),
        goal_rgba: tuple[float, float, float, float] = (0.0, 0.8, 0.0, 0.5),
        **kwargs,
    ) -> None:
        self._goal_pos_cfg = (
            np.array(goal_pos, dtype=np.float64) if goal_pos is not None else None
        )
        self._goal_half_size = tuple(goal_half_size)
        self._goal_rgba = tuple(goal_rgba)
        self._goal_marker = None
        self._goal_body_id = None
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        if self._goal_pos_cfg is None:
            return

        self._goal_marker = BoxObject(
            name="goal_marker",
            size=[self._goal_half_size[0], self._goal_half_size[1], 0.001],
            rgba=list(self._goal_rgba),
            density=1,
            joints=None,
            obj_type="visual",
        )

        all_objects = [self._payload]
        if self._fixture is not None:
            all_objects.append(self._fixture)
        all_objects.append(self._goal_marker)

        self.model = ManipulationTask(
            mujoco_arena=self.model.mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    def _setup_references(self):
        super()._setup_references()
        if self._goal_marker is not None:
            self._goal_body_id = self.sim.model.body_name2id(
                self._goal_marker.root_body
            )

    def _reset_internal(self):
        super()._reset_internal()
        if self._goal_marker is None or self._goal_body_id is None:
            return

        table_z = self.table_offset[2]
        gp = self._goal_pos_cfg
        if len(gp) == 2:
            goal_world = np.array([gp[0], gp[1], table_z + 0.001])
        else:
            goal_world = gp.copy()

        self.sim.model.body_pos[self._goal_body_id] = goal_world
        self.sim.model.body_quat[self._goal_body_id] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )
        self.sim.forward()

    def in_goal_region(self, xy: np.ndarray) -> bool:
        """Return True if world-frame xy lies inside the goal footprint.

        With no goal configured placement is unconstrained, so every xy counts.
        """
        if self._goal_pos_cfg is None:
            return True
        gp = self._goal_pos_cfg
        hx, hy = self._goal_half_size
        return bool(abs(xy[0] - gp[0]) <= hx and abs(xy[1] - gp[1]) <= hy)
