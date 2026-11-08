"""Runtime edits of compiled MuJoCo model constants.

MuJoCo builds the mass matrix of free bodies whose inertial frame is their
joint frame (``body_simple``) from constants it computes at compile time
(``dof_M0``), while gravity uses the live ``body_mass``. Editing a body's mass
or inertia after compilation therefore changes its weight but not its inertial
mass unless the model constants are recomputed with :func:`commit_model_edits`.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Integration state restored after mj_setConst, which resets qpos to qpos0.
_STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat")


def box_inertia(mass: float, half_sizes) -> np.ndarray:
    """Principal inertia of a solid box with half-extents ``(a, b, c)``."""
    a, b, c = half_sizes
    return mass / 3.0 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])


def cylinder_inertia(mass: float, radius: float, half_height: float) -> np.ndarray:
    """Principal inertia of a solid cylinder whose axis is the local z axis."""
    side = mass * (3.0 * radius**2 + 4.0 * half_height**2) / 12.0
    return np.array([side, side, mass * radius**2 / 2.0])


def set_body_mass(sim, body_id: int, mass: float, inertia=None) -> None:
    """Set a body's mass and inertia; inertia scales with the mass unless given.

    Call :func:`commit_model_edits` once all edits are done.
    """
    model = sim.model
    if inertia is None:
        inertia = model.body_inertia[body_id] * (mass / model.body_mass[body_id])
    model.body_inertia[body_id] = inertia
    model.body_mass[body_id] = mass


def commit_model_edits(sim) -> None:
    """Recompute mass-dependent model constants, keeping the simulation state."""
    m, d = sim.model._model, sim.data._data
    saved = {name: getattr(d, name).copy() for name in _STATE_FIELDS}
    time = d.time
    mujoco.mj_setConst(m, d)
    for name, value in saved.items():
        getattr(d, name)[:] = value
    d.time = time
    mujoco.mj_forward(m, d)
