"""Tests for imbench.utils.robot_state."""
from __future__ import annotations

import numpy as np
import pytest

from imbench.utils.robot_state import extract_robot_state, mat_to_quat


# ── Rotation matrix → quaternion ────────────────────────────────────


def _quat_xyzw_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def _rotmat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _quats_equal_up_to_sign(q1: np.ndarray, q2: np.ndarray, atol: float = 1e-6) -> bool:
    return np.allclose(q1, q2, atol=atol) or np.allclose(q1, -q2, atol=atol)


class TestMatToQuat:

    def test_identity(self):
        q = mat_to_quat(np.eye(3))
        assert _quats_equal_up_to_sign(q, np.array([0.0, 0.0, 0.0, 1.0]))

    def test_trace_positive_branch(self):
        # Small rotation around Z keeps trace > 0.
        m = _rotmat_from_axis_angle([0, 0, 1], 0.3)
        expected = _quat_xyzw_from_axis_angle([0, 0, 1], 0.3)
        assert _quats_equal_up_to_sign(mat_to_quat(m), expected)

    def test_x_dominant_branch(self):
        # 180-degree rotation about X: trace = -1, m[0,0] = 1 is the largest
        # diagonal element.
        m = _rotmat_from_axis_angle([1, 0, 0], np.pi)
        expected = _quat_xyzw_from_axis_angle([1, 0, 0], np.pi)
        assert _quats_equal_up_to_sign(mat_to_quat(m), expected, atol=1e-5)

    def test_y_dominant_branch(self):
        m = _rotmat_from_axis_angle([0, 1, 0], np.pi)
        expected = _quat_xyzw_from_axis_angle([0, 1, 0], np.pi)
        assert _quats_equal_up_to_sign(mat_to_quat(m), expected, atol=1e-5)

    def test_z_dominant_branch(self):
        m = _rotmat_from_axis_angle([0, 0, 1], np.pi)
        expected = _quat_xyzw_from_axis_angle([0, 0, 1], np.pi)
        assert _quats_equal_up_to_sign(mat_to_quat(m), expected, atol=1e-5)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
    def test_roundtrip_random_rotations(self, seed):
        rng = np.random.default_rng(seed)
        axis = rng.standard_normal(3)
        angle = rng.uniform(-np.pi, np.pi)
        m = _rotmat_from_axis_angle(axis, angle)
        q = mat_to_quat(m)
        # Quaternion must be unit-length up to numerical tolerance.
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-5)

    def test_returns_xyzw_order(self):
        # Rotation about +X should put the imaginary mass on x, not w-last.
        m = _rotmat_from_axis_angle([1, 0, 0], np.pi / 2)
        q = mat_to_quat(m)
        # [x, y, z, w] — x-component should dominate y,z.
        assert abs(q[0]) > abs(q[1]) + 1e-6
        assert abs(q[0]) > abs(q[2]) + 1e-6

    def test_returns_float64(self):
        q = mat_to_quat(np.eye(3))
        assert q.dtype == np.float64


# ── extract_robot_state ─────────────────────────────────────────────


class TestExtractRobotState:
    """Exercise extract_robot_state against a real task env."""

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        from imbench.registry import make
        t = make("ib-p2-cube-toss-v1", obs_mode="proprio", render_mode=None,
                  seed=0, max_episode_steps=3)
        t.reset()
        yield t
        t.close()

    def test_returns_two_arrays(self, task):
        jp, ee = extract_robot_state(task)
        assert isinstance(jp, np.ndarray)
        assert isinstance(ee, np.ndarray)

    def test_arrays_are_float64(self, task):
        jp, ee = extract_robot_state(task)
        assert jp.dtype == np.float64
        assert ee.dtype == np.float64

    def test_single_arm_shape(self, task):
        """Single Panda: joint_pos (7) + gripper_qpos (2) = 9 joint, ee has
        pos(3) + quat(4) + gripper_qpos(2) = 9."""
        jp, ee = extract_robot_state(task)
        assert jp.shape == (9,)
        assert ee.shape == (9,)

    def test_quat_component_is_unit(self, task):
        _, ee = extract_robot_state(task)
        quat = ee[3:7]
        assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-4)
