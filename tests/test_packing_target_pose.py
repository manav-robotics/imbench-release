"""Per-piece target pose, exact shape symmetries, and all-pieces success."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import imbench as ib
from imbench.tasks.prehensile.packing_v1 import (
    _piece_symmetries, _BOX_BASE_THICK, _PIECE_THICKNESS,
    _PIECE_SETTLED_CONTACT_STEPS,
)


def piece(boxes):
    boxes = np.asarray(boxes, dtype=float)
    areas = boxes[:, 2] * boxes[:, 3]
    return {"subboxes": boxes, "centroid": np.average(boxes[:, :2], axis=0, weights=areas)}


@pytest.mark.parametrize("shape,count,accepted,rejected", [
    (piece([(0, 0, .08, .08)]), 8, [0, 90, 180, 270], [45]),
    (piece([(0, 0, .12, .06)]), 4, [0, 180], [90]),
    (piece([(-.02, 0, .04, .12), (.03, -.04, .06, .04)]), 1, [0], [90, 180]),
    # Equal-leg L: no nontrivial planar rotation, but turning it over
    # about its diagonal reflection axis reproduces the same solid.
    (piece([(-.02, 0, .04, .12), (.04, -.04, .08, .04)]), 2, [0], [90, 180]),
])
def test_symmetries_follow_the_actual_footprint(shape, count, accepted, rejected):
    symmetries = _piece_symmetries(shape)
    assert len(symmetries) == count
    for degrees in accepted:
        rotation = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
        assert any(np.allclose(rotation, s) for s in symmetries)
    for degrees in rejected:
        rotation = Rotation.from_euler("z", degrees, degrees=True).as_matrix()
        assert not any(np.allclose(rotation, s) for s in symmetries)
    np.testing.assert_allclose(np.linalg.det(symmetries), 1)


@pytest.fixture(scope="module")
def task():
    t = ib.make("ib-p7-packing-v1", obs_mode="proprio", privileged_obs=True)
    yield t
    t.close()


def set_pose(task, name, offset=(0, 0, 0), rotation=None, negate_quaternion=False):
    rotation = np.eye(3) if rotation is None else rotation
    sim = task._env.sim
    box = sim.model.body_name2id(task._box.root_body)
    goal_rotation = sim.data.body_xmat[box].reshape(3, 3)
    cx, cy = task._piece_by_name[name]["centroid"]
    centroid = sim.data.body_xpos[box] + goal_rotation @ np.array([
        cx, cy, _BOX_BASE_THICK + _PIECE_THICKNESS/2]) + offset
    world_rotation = goal_rotation @ rotation
    body_pos = centroid - world_rotation @ np.array([0, 0, _PIECE_THICKNESS/2])
    q = Rotation.from_matrix(world_rotation).as_quat()[[3, 0, 1, 2]]
    if negate_quaternion:
        q = -q
    block = task._block_by_name[name]
    sim.data.set_joint_qpos(block.joints[0], np.r_[body_pos, q])
    dof = sim.model.jnt_dofadr[sim.model.joint_name2id(block.joints[0])]
    sim.data.qvel[dof:dof+6] = 0
    sim.forward()


def goal_setup(task):
    task.reset(seed=0)
    for name in task._block_names:
        set_pose(task, name)
        # Constructed reward probe: precondition the contact timer only.
        # The separate existing success tests accumulate real contacts.
        task._bin_contact_streak[name] = _PIECE_SETTLED_CONTACT_STEPS


@pytest.mark.parametrize("kind", ["position", "yaw", "roll", "height"])
def test_one_bad_piece_prevents_success_among_other_correct_pieces(task, kind):
    goal_setup(task)
    assert task._compute_task_score({}) == 1
    name = task._block_names[-1]
    kwargs = {"position": {"offset": (.006, 0, 0)},
              "height": {"offset": (0, 0, .006)},
              "yaw": {"rotation": Rotation.from_euler("z", 180, degrees=True).as_matrix()},
              "roll": {"rotation": Rotation.from_euler("x", 180, degrees=True).as_matrix()}}[kind]
    set_pose(task, name, **kwargs)
    assert not task._piece_at_target(name)
    assert task._compute_task_score({}) < 1
    assert not task._is_success(task.task_score())
    assert all(task._piece_fully_packed(n) for n in task._block_names[:-1])


@pytest.mark.parametrize("distance,angle,expected", [
    (.0049, 0, True), (.0051, 0, False),
    (0, 4.9, True), (0, 5.1, False), (0, 359, True),
])
def test_centroid_and_angle_boundaries(task, distance, angle, expected):
    goal_setup(task)
    name = task._block_names[0]
    set_pose(task, name, offset=(distance, 0, 0),
             rotation=Rotation.from_euler("z", angle, degrees=True).as_matrix())
    assert bool(task._piece_at_target(name)) is expected


def test_quaternion_sign_and_real_settling_and_live_removal(task):
    task.reset(seed=0)
    for name in task._block_names:
        set_pose(task, name, negate_quaternion=True)
    for _ in range(_PIECE_SETTLED_CONTACT_STEPS + 10):
        _, reward, _, _, info = task.step(np.array([0, 0, 0, 0, 0, 0, -1.]))
    assert info["is_success"] and reward == 1
    name = task._block_names[0]
    set_pose(task, name, offset=(.008, 0, 0))
    task._compute_task_score({})
    assert not task._all_packed


def test_equivalent_rectangle_and_square_turnovers_use_true_centroid(task):
    # Analytic geometry fixtures exercise the scorer's symmetry quotient;
    # they do not replace the actual MuJoCo L meshes or claim robot runs.
    goal_setup(task)
    name = task._block_names[0]
    for shape, angle, expected in [
            (piece([(0, 0, .12, .06)]), 180, True),
            (piece([(0, 0, .12, .06)]), 90, False),
            (piece([(0, 0, .08, .08)]), 90, True),
            (piece([(0, 0, .08, .08)]), 45, False)]:
        task._piece_symmetry_rotations[name] = _piece_symmetries(shape)
        set_pose(task, name, rotation=Rotation.from_euler("z", angle, degrees=True).as_matrix())
        assert bool(task._piece_at_target(name)) is expected
    task._piece_symmetry_rotations[name] = _piece_symmetries(piece([(0, 0, .12, .06)]))
    set_pose(task, name, rotation=Rotation.from_euler("x", 180, degrees=True).as_matrix())
    assert task._piece_at_target(name)
    assert task._piece_in_box_flat(name)
    assert task._piece_pose_errors(name)[0] < 1e-9
