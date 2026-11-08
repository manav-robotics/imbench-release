"""Domino selection with the shared sampled layout rotated 90 degrees."""

import numpy as np

from imbench.tasks.prehensile.domino_select_v1 import DominoSelectV1


class DominoSelectV2(DominoSelectV1):
    TASK_ID = "ib-p3-domino-select-v2"
    _LAYOUT_YAW = np.pi / 2
