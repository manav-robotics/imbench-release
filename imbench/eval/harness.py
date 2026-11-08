"""Evaluate a policy on IMBench tasks."""

from __future__ import annotations

import inspect
import logging
import os
import sys

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface
from imbench.registry import TASK_REGISTRY

logger = logging.getLogger(__name__)


def run_benchmark(
    policy: PolicyInterface,
    task_ids: list[str],
    n_episodes: int = 50,
    seed_offset: int = 0,
    obs_mode: str = "combo",
    render: bool = False,
    privileged_obs: bool = False,
    env_access: bool = False,
) -> dict[str, dict[str, float | int]]:
    """Run a policy on one or more IMBench tasks and aggregate results.

    Parameters
    ----------
    policy:
        A :class:`~imbench.policy.base_policy.PolicyInterface` instance. Its
        actions must match each task's ``action_space`` (7-D single-arm,
        14-D bimanual). ``reset`` receives the episode seed as ``seed=`` when
        its signature has a parameter of that name; ``act`` receives the task
        as ``env=`` only with ``env_access=True``.
    task_ids:
        Task IDs to evaluate (e.g. ``["ib-p2-cube-toss-v1"]``). Validated
        before any episode runs.
    n_episodes:
        Episodes per task; must be >= 1.
    seed_offset:
        Episode ``i`` is reset with ``seed_offset + i``.
    obs_mode:
        Observation modality passed to :func:`imbench.make`.
    render:
        When ``True``, opens an on-screen MuJoCo viewer. On Linux this needs
        a display (``DISPLAY`` or ``WAYLAND_DISPLAY``); without one a :class:`RuntimeError` is
        raised before any episode runs.
    privileged_obs:
        Passed to :func:`imbench.make`. ``False`` (the default for evaluation)
        drops simulator ground truth such as ``object_state`` and object poses
        from ``obs``, leaving proprioception, cameras, force/torque and tactile.
        Set ``True`` only for policies that are meant to read sim state, such as
        the scripted oracles.
    env_access:
        When ``True``, ``policy.act`` is called as ``act(obs, env=task)`` if it
        has an ``env`` parameter, giving it the live simulator. ``False`` (the
        default for evaluation) never passes the task; a policy whose ``act``
        requires ``env`` then raises :class:`ValueError` before any episode
        runs. Like ``privileged_obs``, only for scripted oracles.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of *task_id* → metrics: ``mean_score`` / ``std_score`` (final
        ``task_score``), ``success_rate`` (fraction of episodes in which
        ``info["is_success"]`` was ever true), ``mean_length`` and
        ``n_episodes``.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    unknown = [t for t in task_ids if t not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown task ids: {unknown}. See imbench.list_tasks().")
    # MuJoCo's viewer blocks on "Press Enter to exit" when it cannot open a window.
    if render and sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "render=True opens the MuJoCo viewer, which needs a display, but neither DISPLAY "
            "nor WAYLAND_DISPLAY is set; pass render=False or run under xvfb-run."
        )
    reset_takes_seed = "seed" in inspect.signature(policy.reset).parameters
    act_env = inspect.signature(policy.act).parameters.get("env")
    if act_env is not None and act_env.default is inspect.Parameter.empty and not env_access:
        raise ValueError(
            "policy.act requires env=, which exposes the live simulator; pass env_access=True "
            "only for scripted oracles"
        )
    act_takes_env = env_access and act_env is not None

    results = {}
    for task_id in task_ids:
        env = ib.make(task_id, obs_mode=obs_mode, privileged_obs=privileged_obs,
                      render_mode="human" if render else None)
        scores, lengths, successes = [], [], []
        try:
            for ep in range(n_episodes):
                seed = seed_offset + ep
                obs, _ = env.reset(seed=seed)
                if reset_takes_seed:
                    policy.reset(seed=seed)
                else:
                    policy.reset()
                success = False
                done = False
                while not done:
                    action = policy.act(obs, env=env) if act_takes_env else policy.act(obs)
                    obs, _, terminated, truncated, info = env.step(action)
                    success = success or info["is_success"]
                    done = terminated or truncated
                scores.append(info["task_score"])
                lengths.append(info["step"])
                successes.append(success)
        finally:
            env.close()
        results[task_id] = {
            "mean_score":  float(np.mean(scores)),
            "std_score":   float(np.std(scores)),
            "success_rate": float(np.mean(successes)),
            "mean_length": float(np.mean(lengths)),
            "n_episodes":  n_episodes,
        }
        logger.info(
            "%s  score=%.3f ± %.3f  sr=%.2f%%",
            task_id,
            results[task_id]["mean_score"],
            results[task_id]["std_score"],
            results[task_id]["success_rate"] * 100,
        )
    return results
