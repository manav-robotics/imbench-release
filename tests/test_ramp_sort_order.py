"""Sorting is independent of placement order; the oracle tracks rolling balls."""
import contextlib
import io
import itertools

import numpy as np
import pytest

import imbench as ib
from data_synthetic.single_arm_pipeline.ramp_sort_policy import RampSortPolicy

NAMES = ("sphere_0", "box_0", "box_1")


@pytest.fixture(scope="module")
def task():
    t = ib.make("ib-p2-ramp-sort-v1", obs_mode="proprio", privileged_obs=True,
                config_overrides={"task": {"num_spheres_range": [1, 1], "num_cubes_range": [2, 2]}})
    yield t
    t.close()


@pytest.mark.parametrize("order", list(itertools.permutations(NAMES)))
def test_actual_policy_can_sort_every_three_ball_order(task, order):
    obs, _ = task.reset(seed=0)
    policy = RampSortPolicy(task)
    original = policy._build_queue
    def queue():
        items = {r["name"]: r for r in original()}
        return [items[n] for n in order]
    policy._build_queue = queue
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range(task.MAX_EPISODE_STEPS):
            obs, reward, term, trunc, info = task.step(policy.act(obs))
            if term or trunc:
                break
    assert info["is_success"] and reward == 1, (order, step+1, policy.phase, info)
    assert not task._any_ball_in_wrong_bin()


@pytest.mark.parametrize("order", list(itertools.permutations(NAMES)))
def test_reward_depends_on_named_active_balls_not_placement_sequence(task, order, monkeypatch):
    task.reset(seed=0)
    yaw = task._ramp_yaw
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    positions = {"sphere_0": task._sphere_bin_world_xy,
                 "box_0": task._cube_bin_world_xy + rotation @ [0, -.020],
                 "box_1": task._cube_bin_world_xy + rotation @ [0, .020]}
    for name in order:
        task._place_object(name, [*positions[name], task._table_height+.0282])
        task._set_object_velocity(name, [0, 0, 0])
        task._env.sim.forward()
    # A ball counts once it has stayed at rest in its bin for REST_TIME.
    for _ in range(int(round(task.REST_TIME * task.control_freq)) + 300):
        _, _, term, _, _ = task.step(np.zeros(task.action_space.shape[0]))
        if term:
            break
    assert task._compute_task_score({}) == 1
    spheres, reds = task._ball_body_ids()
    monkeypatch.setattr(task, "_ball_body_ids", lambda: (list(reversed(spheres)), list(reversed(reds))))
    assert task._compute_task_score({}) == 1


def test_count_changes_do_not_score_parked_balls(task):
    task.reset(seed=0)
    assert len(task._ball_body_ids()[1]) == 2
    task._num_cubes_range = [1, 1]
    try:
        task.reset(seed=0)
        assert "box_1" in task._parked
        assert len(task._ball_body_ids()[1]) == 1
    finally:
        task._num_cubes_range = [2, 2]
