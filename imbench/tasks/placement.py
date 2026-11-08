"""Truncated-Gaussian spawn jitter shared by the bimanual task bases."""

from __future__ import annotations

from typing import Callable

import numpy as np

#: Jitter is truncated at this many standard deviations.
CLIP_STD = 2.0
#: Draws tried before falling back to the nominal position.
MAX_DRAWS = 20


def table_overhang(env, radius: float, xy) -> float:
    """How far a footprint of *radius* centred at *xy* extends past the table edge."""
    table_xy = np.asarray(env.table_offset[:2], dtype=np.float64)
    half = 0.5 * np.asarray(env.table_full_size[:2], dtype=np.float64)
    return float(np.max(np.maximum(np.abs(np.asarray(xy) - table_xy) + radius - half, 0.0)))


def jitter_free_joint(
    sim,
    joint_name: str,
    rng: np.random.RandomState,
    std_xy,
    is_valid: Callable[[np.ndarray], bool],
) -> None:
    """Jitter a free joint's x/y with a truncated Gaussian and zero its velocity.

    ``is_valid(pos)`` is called with each draw already applied to *sim*;
    invalid draws are redrawn, and if none of :data:`MAX_DRAWS` is valid the
    nominal position is kept.
    """
    nominal = np.array(sim.data.get_joint_qpos(joint_name), dtype=np.float64)
    std = np.asarray(std_xy, dtype=np.float64)
    limit = CLIP_STD * std
    for _ in range(MAX_DRAWS):
        qpos = nominal.copy()
        qpos[:2] += np.clip(rng.normal(0.0, std), -limit, limit)
        sim.data.set_joint_qpos(joint_name, qpos)
        sim.forward()
        if is_valid(qpos[:3]):
            break
    else:
        sim.data.set_joint_qpos(joint_name, nominal)
    sim.data.set_joint_qvel(joint_name, np.zeros(6))
    sim.forward()
