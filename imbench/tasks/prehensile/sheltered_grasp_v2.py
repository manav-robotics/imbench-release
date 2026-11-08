"""Sheltered Grasp — reach into a 2-story rack from the side to extract a hidden cube.

Task ID: ``ib-p1-sheltered-grasp-v2``

Same task as :class:`ShelteredGraspV1` with the rack mirrored to the
robot's right: it sits at a fixed world pose (0.12, -0.22) with yaw = π,
so its open face (local -y) still faces the robot. Everything else comes
from the task YAML.
"""

import numpy as np

from imbench.tasks.prehensile.sheltered_grasp_v1 import ShelteredGraspV1


class ShelteredGraspV2(ShelteredGraspV1):
    TASK_ID = "ib-p1-sheltered-grasp-v2"

    _RACK_POS = (0.12, -0.22)
    _RACK_YAW = np.pi

    def _parse_task_config(self):
        super()._parse_task_config()
        self._rack_default_pos = np.array(self._RACK_POS)
        self._rack_default_yaw = self._RACK_YAW
        self._rack_randomize_pos = False
        self._rack_randomize_ori = False
