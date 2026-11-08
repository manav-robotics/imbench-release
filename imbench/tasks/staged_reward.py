"""Shared staged/dense reward abstraction.

Generalises the partial-credit patterns used across tasks (weighted
sub-scores, hold-counter ramps, fraction-of-N, multi-goal shaping, discrete
stage values) into one reusable shape:

    score = sum(weight_i * best_progress_i)   for each named stage

Terminal success saturates the score to ``1.0`` (via ``success_fn``) and
sets ``"_success"`` in the returned stage values; ``BaseTask`` reports that
flag as ``info["is_success"]``, which the eval harness uses for success
rates. Sticky partial credit can approach ``1.0`` without success, so never
threshold the score as a success proxy.

Usage from a task's ``_compute_task_score``::

    def _parse_task_config(self):
        ...
        self._staged_reward = StagedReward(
            stages=[
                Stage("reach", 0.25, lambda t, o: t._reach_progress(o)),
                Stage("grasp", 0.30, lambda t, o: 1.0 if t._is_grasping(o) else 0.0),
                Stage("lift", 0.45, lambda t, o: t._lift_progress(o)),
            ],
            success_fn=lambda t, o: t._lift_progress(o) >= 1.0,
        )

    def _compute_task_score(self, raw_obs):
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._staged_reward.reset()
        return obs, info
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def progress_from_reset(value: float, at_reset: float) -> float:
    """Shaping *value* rescaled to [0, 1] so that its value at reset reads 0.

    *at_reset* is capped just below 1 so a saturated baseline stays defined.
    """
    at_reset = min(at_reset, 0.99)
    return float(np.clip((value - at_reset) / (1.0 - at_reset), 0.0, 1.0))


@dataclass
class Stage:
    """One named milestone in a task's progress toward completion.

    Parameters
    ----------
    name:
        Stage identifier, used as the key in ``StagedReward.stage_values``
        and in per-task debugging/logging.
    weight:
        Contribution to the final score when this stage is fully
        achieved (``fn`` returns ``1.0``). All stage weights on a
        ``StagedReward`` must sum to ``1.0``.
    fn:
        ``(task, raw_obs) -> float in [0, 1]``. Progress toward this
        stage. Values outside ``[0, 1]`` are clipped.
    sticky:
        When ``True`` (default), the recorded value for this stage is
        ``max(previous, current)`` — once a milestone is reached,
        transient regressions (e.g. briefly losing a grasp after
        already having lifted the object) don't reduce the score. This
        matches robosuite's own staged-reward convention (``Stack``,
        ``PickPlace``, ``NutAssembly`` all use ``max()`` over stages).
        Set ``False`` for stages that should track current state exactly
        (e.g. "is currently balanced").
    """

    name: str
    weight: float
    fn: Callable[[Any, dict], float]
    sticky: bool = True


class StagedReward:
    """Weighted-sum-of-milestones score, saturating to 1.0 on success."""

    def __init__(
        self,
        stages: list[Stage],
        success_fn: Callable[[Any, dict], bool] | None = None,
        weight_tol: float = 1e-6,
    ) -> None:
        if not stages:
            raise ValueError("StagedReward requires at least one stage")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"Stage names must be unique, got {names}")
        negative = [s.name for s in stages if s.weight < 0]
        if negative:
            raise ValueError(f"Stage weights must be non-negative: {negative}")
        total_weight = sum(s.weight for s in stages)
        if not np.isclose(total_weight, 1.0, atol=weight_tol):
            raise ValueError(
                f"Stage weights must sum to 1.0, got {total_weight} "
                f"({[(s.name, s.weight) for s in stages]})"
            )
        self._stages = list(stages)
        self._success_fn = success_fn
        self._best: dict[str, float] = {s.name: 0.0 for s in self._stages}

    def reset(self) -> None:
        """Clear stage progress. Call from the task's ``reset()``."""
        self._best = {s.name: 0.0 for s in self._stages}

    def compute(self, task: Any, raw_obs: dict) -> tuple[float, dict]:
        """Return ``(score, stage_values)`` for the current state.

        ``stage_values`` maps each stage name to its recorded progress
        (post-sticky-max), plus ``"_success"`` indicating whether
        ``success_fn`` fired this call. ``success_fn`` sees the state before
        this call's stage updates; stage values are updated on every call,
        including the one on which success fires.
        """
        success = self._success_fn is not None and bool(self._success_fn(task, raw_obs))
        for stage in self._stages:
            # NaN progress (e.g. a degenerate geometry query) counts as none.
            value = float(np.clip(np.nan_to_num(stage.fn(task, raw_obs), nan=0.0), 0.0, 1.0))
            if stage.sticky:
                self._best[stage.name] = max(self._best[stage.name], value)
            else:
                self._best[stage.name] = value

        score = 1.0 if success else sum(s.weight * self._best[s.name] for s in self._stages)
        debug = dict(self._best)
        debug["_success"] = success
        return score, debug

    @property
    def stage_values(self) -> dict:
        return dict(self._best)
