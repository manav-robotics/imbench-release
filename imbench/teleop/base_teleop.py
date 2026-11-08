from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class TeleopInterface(ABC):
    """Any device that can drive the robot end-effector."""

    ACTION_DIM = 7  # dx dy dz droll dpitch dyaw gripper (all continuous [-1, 1])

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> dict:
        """Return a dict with the current device state.

        Returns
        -------
        dict with keys:
            ``"action"`` : np.ndarray, shape (7,) or (7*num_arms,)
                Cartesian-delta action clipped to [-1, 1].
            ``"paused"`` : bool
                True when the operator is idle (no inputs active).
        """

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def should_end_episode(self) -> bool:
        """Return True when the operator wants to end the current episode."""
        return False
