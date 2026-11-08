"""Whole-plate insertion into the tightened HetBi shelf, using real meshes."""

import numpy as np
import pytest
from robosuite.utils import transform_utils as T

from data_synthetic.hetbi_scripted.plate_shelf_policy import PlateShelfPolicy
from tests.conftest import cached_task


ON_EDGE = np.array([2**-.5, 0, 0, 2**-.5])  # xyzw; face normal along world Y


@pytest.fixture
def task():
    return cached_task('ib-p1-plate-shelf-hetbi-v1', obs_mode='proprio',
                       privileged_obs=True, max_episode_steps=400)


def raw_pose(task, mouth_clearance=None, quat=ON_EDGE):
    prefix = task._payload_prefix
    raw = {prefix + 'pos': task._env._target_goal_pos.copy(), prefix + 'quat': np.array(quat)}
    if mouth_clearance is not None:
        current = task._insertion_clearances(raw)[1]
        body = task._env._fixture_body_id
        into_shelf = task._env.sim.data.body_xmat[body].reshape(3, 3)[:, 1]
        raw[prefix + 'pos'] += (mouth_clearance - current) * into_shelf
    return raw


def place(task, raw):
    prefix = task._payload_prefix
    sim = task._env.sim
    quat = raw[prefix + 'quat'][[3, 0, 1, 2]]
    joint = task._env._payload.joints[0]
    sim.data.set_joint_qpos(joint, np.r_[raw[prefix + 'pos'], quat])
    sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()


def step(task):
    action = np.zeros(task.action_space.shape)
    action[6::7] = -1
    return task.step(action)


@pytest.mark.parametrize('clearance, inserted', [(0.002, True), (-0.002, False),
                                               (-0.05, False), (0.00075, True), (-0.00075, False)])
def test_front_boundary_uses_the_whole_plate_not_the_centre(task, clearance, inserted, monkeypatch):
    task.reset(seed=0)
    raw = raw_pose(task, clearance)
    centre = raw[task._payload_prefix + 'pos']
    assert all(lo <= centre[k] <= hi for k, (lo, hi) in enumerate(task._env._fixture_bounds.values()))
    assert task._on_edge_progress(raw) == 1.0
    assert task._inserted_in_bounds_progress(raw) == float(inserted)
    # Isolate containment after an already-earned handover. Actual robot
    # handovers and bypass attempts are exercised separately below.
    task._handoff_complete = True
    # Failure cannot be explained away as motion or a gripper holding it.
    monkeypatch.setattr(task, '_payload_released', lambda: True)
    monkeypatch.setattr(task, '_payload_is_stable_multistep', lambda: True)
    assert task._settled_in_shelf(raw) == inserted


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('clearance, expected', [(None, True), (0.002, True), (-0.002, False), (-0.05, False)])
def test_released_plate_settles_and_scores_only_when_fully_inserted(task, seed, clearance, expected):
    task.reset(seed=seed)
    task._handoff_complete = True  # constructed precondition for placement-only physics
    raw = raw_pose(task, clearance)
    place(task, raw)
    ever = False
    for _ in range(200):
        _, reward, terminated, _, info = step(task)
        ever |= info['is_success']
        assert reward == float(info['is_success'])
        if terminated:
            break
    assert ever == expected
    assert task._payload_released() and task._payload_is_stable_multistep()
    assert task._stage_debug['inserted_in_bounds'] == float(expected)
    if not expected:
        assert not terminated and info['task_score'] < 1.0
        current = task._get_sim_observations(force_update=True)
        assert task._insertion_clearances(current)[1] < -task._insertion_tolerance


def test_jaw_or_suction_still_holding_full_insertion_cannot_succeed(task, monkeypatch):
    task.reset(seed=0)
    task._handoff_complete = True
    raw = raw_pose(task)
    assert task._inserted_in_bounds_progress(raw) == 1.0
    monkeypatch.setattr(task, '_payload_released', lambda: False)
    monkeypatch.setattr(task, '_payload_is_stable_multistep', lambda: True)
    assert not task._settled_in_shelf(raw)


def test_removing_plate_loses_insertion_credit_and_policy_does_not_release_early(task):
    task.reset(seed=0)
    policy = PlateShelfPolicy()
    for clearance, expected in [(0.002, 1.0), (-0.05, 0.0)]:
        raw = raw_pose(task, clearance)
        place(task, raw)
        policy._plate_pos = raw[task._payload_prefix + 'pos']
        current = task._get_sim_observations(force_update=True)
        task._compute_task_score(current)
        assert task._stage_debug['inserted_in_bounds'] == expected
        assert policy._plate_in_shelf(task) == bool(expected)


def test_on_edge_but_crosswise_plate_does_not_fit(task):
    task.reset(seed=0)
    crosswise = np.array([0, 2**-.5, 0, 2**-.5])
    raw = raw_pose(task, quat=crosswise)
    assert task._on_edge_progress(raw) == 1.0
    assert task._inserted_in_bounds_progress(raw) == 0.0


def test_rotated_shelf_uses_local_faces_not_world_axis_bounds(task):
    task.reset(seed=0)
    sim = task._env.sim
    fixture = task._env._fixture_body_id
    angle = np.deg2rad(27)
    sim.model.body_quat[fixture] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
    sim.forward()
    task._cache_insertion_geometry()
    rotation = sim.data.body_xmat[fixture].reshape(3, 3)
    local_on_edge = T.quat2mat(np.array([0, 2**-.5, 0, 2**-.5]))
    quat = T.mat2quat(rotation @ local_on_edge)
    assert task._inserted_in_bounds_progress(raw_pose(task, 0.002, quat)) == 1.0
    assert task._inserted_in_bounds_progress(raw_pose(task, -0.01, quat)) == 0.0


def test_collision_geometry_fits_with_clearance_and_slot_is_tighter(task):
    task.reset(seed=0)
    size = task._slot_max - task._slot_min
    assert size[0] == pytest.approx(0.120)
    assert size[1] == pytest.approx(0.28)
    assert np.min(task._insertion_clearances(raw_pose(task))) > 0.005


def test_full_placement_without_handoff_is_not_success(task):
    task.reset(seed=0)
    place(task, raw_pose(task))
    for _ in range(150):
        _, reward, terminated, _, info = step(task)
        assert reward == 0 and not terminated and not info['is_success']
    assert task._payload_released() and task._payload_is_stable_multistep()
    assert task._stage_debug['inserted_in_bounds'] == 1.0
    assert not task._handoff_complete


def test_reset_and_renewed_suction_clear_the_handoff(task, monkeypatch):
    task.reset(seed=0)
    task._handoff_complete = True
    monkeypatch.setattr(task._suction_weld, 'welded_body_ids',
                        lambda: {task._env._payload_body_id})
    task._update_handoff()
    assert not task._handoff_complete
    task._handoff_complete = task._suction_lifted = True
    task.reset(seed=0)
    assert not task._handoff_complete and not task._suction_lifted
