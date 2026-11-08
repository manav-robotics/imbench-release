"""Abstract base class for IMBench evaluation policies."""

from abc import ABC, abstractmethod
import numpy as np


class PolicyInterface(ABC):
    """Minimal interface that every policy must implement.

    Concrete policies are passed to :func:`~imbench.eval.harness.run_benchmark`.
    Implement :meth:`act` (required) and optionally :meth:`reset`.

    The harness passes two optional keyword arguments, each only when the
    method's signature has a parameter of that name:

    * ``reset(seed=...)``: the episode's reset seed, so a stochastic policy
      can reseed its RNG and evaluation runs are reproducible.
    * ``act(obs, env=...)``: the live task. Scripted oracles use it to read
      privileged simulator state; a learned policy should act on ``obs`` alone.
    """

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Select an action given the current observation.

        Parameters
        ----------
        obs:
            Observation dict as returned by
            :meth:`~imbench.tasks.base_task.BaseTask.step`.

        Returns
        -------
        np.ndarray
            Action vector in the task's action space.
        """

    def reset(self) -> None:
        """Reset any internal policy state at the start of a new episode."""
        pass
