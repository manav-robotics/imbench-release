"""Shared helpers for extracting robot state from MuJoCo sim data.

These utilities read joint positions and EE poses directly from the MuJoCo
sim data without triggering ``_get_observations()`` (which re-renders
cameras).  Used by both teleoperation collection and scripted pipelines.
"""
from __future__ import annotations

import numpy as np


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to ``[x, y, z, w]`` quaternion."""
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def extract_robot_state(env) -> tuple[np.ndarray, np.ndarray]:
    """Extract absolute joint positions and EE poses from the environment.

    Reads joint/EE state directly from the MuJoCo ``sim.data`` and the
    robot models, avoiding ``_get_observations()`` which would trigger a
    full camera re-render.

    Parameters
    ----------
    env : IMBench task wrapper
        Must expose ``env._env.robots``.

    Returns
    -------
    joint_positions : np.ndarray
        Concatenated ``[joint_pos, gripper_qpos]`` for every robot/arm.
    ee_poses : np.ndarray
        Concatenated ``[eef_pos(3), eef_quat_xyzw(4), gripper_qpos]``
        for every robot/arm.
    """
    joint_parts: list[np.ndarray] = []
    ee_parts: list[np.ndarray] = []
    for robot in env._env.robots:
        sim = robot.sim
        jpos = np.array([sim.data.qpos[x] for x in robot._ref_joint_pos_indexes])
        for arm in robot.arms:
            gqpos = np.array(
                [sim.data.qpos[x] for x in robot._ref_gripper_joint_pos_indexes[arm]]
            )
            epos = np.array(sim.data.site_xpos[robot.eef_site_id[arm]])
            emat = np.array(sim.data.site_xmat[robot.eef_site_id[arm]]).reshape(3, 3)
            equat = mat_to_quat(emat)
            joint_parts.extend([jpos, gqpos])
            ee_parts.extend([epos, equat, gqpos])
    return (
        np.concatenate(joint_parts).astype(np.float64) if joint_parts else np.zeros(0),
        np.concatenate(ee_parts).astype(np.float64) if ee_parts else np.zeros(0),
    )
