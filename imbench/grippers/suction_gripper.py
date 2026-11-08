"""Panda Suction Gripper — a vacuum/suction end-effector for the Franka Panda.

Registers itself with robosuite on import so it can be passed as
``gripper_types=["PandaSuctionGripper"]`` to ``suite.make()``.

The gripper uses a MuJoCo weld equality constraint that is toggled at
runtime by :class:`imbench.grippers.suction_weld.SuctionWeld`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class PandaSuctionGripper(GripperModel):
    """Suction gripper compatible with the Franka Panda robot."""

    def __init__(self, idn: int = 0) -> None:
        xml_path = _ASSETS_DIR / "panda_suction.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Suction gripper XML not found: {xml_path}")
        super().__init__(str(xml_path), idn=idn)

    def format_action(self, action: np.ndarray) -> np.ndarray:
        """Map raw action to gripper command.

        Convention: continuous [-1, 1]. Positive = suck/close,
        negative = release/open. Suction is binary in practice:
        any value > 0 activates the weld, any value < 0 releases it.
        """
        return action

    @property
    def init_qpos(self):
        return np.array([0.0])

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    @property
    def contact_geoms(self):
        return [self.correct_naming("geo_snout")]

    @property
    def _important_sites(self):
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
            "ee": "grip_site",
            "ee_x": "ee_x",
            "ee_y": "ee_y",
            "ee_z": "ee_z",
        }

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
            "left_finger_pad": [],
            "right_finger_pad": [],
            "root": ["geo_snout"],
        }


# Auto-register with robosuite so ``gripper_types=["PandaSuctionGripper"]`` works.
register_gripper(PandaSuctionGripper)
