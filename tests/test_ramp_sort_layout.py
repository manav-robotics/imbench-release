"""Ramp sort layouts vary per seed, stay on the table, and a ball scores only at rest inside its own bin."""
import contextlib
import io

import numpy as np
import pytest

import imbench as ib
from data_synthetic.single_arm_pipeline.ramp_sort_policy import RampSortPolicy

SEEDS = range(40)


@pytest.fixture(scope="module")
def task():
    t = ib.make("ib-p2-ramp-sort-v1", obs_mode="proprio", privileged_obs=True)
    yield t
    t.close()


def _balls(task):
    sids, cids = task._ball_body_ids()
    return sids + cids


def _layout(task):
    sim = task._env.sim
    return np.array([sim.data.body_xpos[b].copy() for b in _balls(task)])


def _teleport(task, name, pos):
    task._place_object(name, pos)
    task._set_object_velocity(name, [0, 0, 0], [0, 0, 0])
    task._env.sim.forward()


def test_layout_counts_and_pose_vary_across_seeds(task):
    layouts, counts, yaws = set(), set(), []
    for seed in SEEDS:
        task.reset(seed=seed)
        layouts.add(tuple(np.round(_layout(task)[:, :2], 2).ravel()))
        counts.add((task._num_spheres, task._num_cubes))
        yaws.append(np.degrees(task._ramp_yaw) % 180.0)
    assert len(layouts) == len(SEEDS)
    assert {n for pair in counts for n in [sum(pair)]} == {2, 3, 4, 5}
    # Yaw jitter is +/-15 deg around 0 and 180 deg.
    off = np.minimum(np.array(yaws), 180.0 - np.array(yaws))
    assert off.max() > 10.0 and off.max() <= 15.0 + 1e-6


def test_reset_is_reproducible_per_seed(task):
    task.reset(seed=7)
    first = _layout(task)
    task.reset(seed=3)
    task.reset(seed=7)
    np.testing.assert_allclose(_layout(task), first)


def test_balls_start_apart_settled_and_layout_on_table(task):
    sim, model = task._env.sim, task._env.sim.model
    half_y = task._table_full_size[1] / 2
    for seed in SEEDS:
        task.reset(seed=seed)
        start = _layout(task)
        gaps = [np.linalg.norm(start[i, :2] - start[j, :2]) - 2 * task._object_size
                for i in range(len(start)) for j in range(i)]
        assert min(gaps) >= task.SPAWN_MIN_GAP - 1e-6, seed
        for g in range(model.ngeom):
            name = model.geom_id2name(g) or ""
            if not name.startswith(("spawn_", "bin_", "ramp")):
                continue
            rot, half = sim.data.geom_xmat[g].reshape(3, 3), model.geom_size[g]
            for corner in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                assert abs((sim.data.geom_xpos[g] + rot @ (corner * half))[1]) < half_y, (seed, name)
        for _ in range(50):
            task.step(np.zeros(task.action_space.shape[0]))
        assert np.abs(_layout(task) - start).max() < 0.005, seed


def _teleport_moving(task, name, pos, vel):
    _teleport(task, name, pos)
    task._set_object_velocity(name, vel, [0, 0, 0])


def _place_sorted(task):
    """Put every ball on its own bin's floor, spread over the floor's corners."""
    floor = task._table_height + 0.008 + task._object_size + 0.001
    corners = [(-0.019, -0.019), (0.019, 0.019), (-0.019, 0.019)]
    for i in range(task._num_spheres):
        _teleport(task, f"sphere_{i}", [*(task._sphere_bin_world_xy + corners[i]), floor])
    for i in range(task._num_cubes):
        _teleport(task, f"box_{i}", [*(task._cube_bin_world_xy + corners[i]), floor])


def _idle(task, steps):
    for _ in range(steps):
        _, _, term, _, info = task.step(np.zeros(task.action_space.shape[0]))
        if term:
            break
    return term, info


def test_ball_counts_once_it_has_stayed_at_rest_inside_its_bin(task):
    task.reset(seed=0)
    _place_sorted(task)
    # Not yet: no ball has been still for REST_TIME.
    assert task._compute_task_score({}) < 1.0
    need = int(round(task.REST_TIME * task.control_freq))
    term, info = _idle(task, need + 400)
    assert term and info["is_success"]


def test_ball_still_falling_in_is_not_yet_sorted(task):
    task.reset(seed=0)
    _place_sorted(task)
    sid = task._env.sim.model.body_name2id("sphere_0_main")
    # Below the rim but dropping fast: inside the bin, not at rest.
    _teleport_moving(task, "sphere_0", [*task._sphere_bin_world_xy, task._table_height + 0.07], [0, 0, -0.6])
    task.step(np.zeros(task.action_space.shape[0]))
    assert task._in_bin(sid, task._sphere_bin_world_xy)
    assert not task._at_rest_in_bin(sid, task._sphere_bin_world_xy)
    assert task._compute_task_score({}) < 1.0
    assert not task._check_termination({})


def test_ball_resting_above_the_floor_counts(task):
    """A ball held still on top of another ball, below the rim, is sorted; one above the rim is not."""
    task.reset(seed=0)
    _place_sorted(task)
    sid = task._env.sim.model.body_name2id("sphere_0_main")
    xy = task._sphere_bin_world_xy
    for z, counts in [(task._table_height + 0.06, True), (task._bin_top_z + 0.01, False)]:
        task._rest_steps = {}
        for _ in range(int(round(task.REST_TIME * task.control_freq)) + 2):
            _teleport(task, "sphere_0", [*xy, z])   # hold it there, as if resting on other balls
            task._update_rest_streaks()
        assert bool(task._at_rest_in_bin(sid, xy)) is counts, z


def test_ball_below_wrong_rim_fails_before_reaching_the_floor(task):
    task.reset(seed=0)
    _teleport(task, "sphere_0", [*task._cube_bin_world_xy, task._table_height + 0.06])
    assert task._any_ball_in_wrong_bin()
    assert task._compute_task_score({}) == 0.0
    assert task._check_termination({})


@pytest.mark.parametrize("seed", [6, 9])
def test_scripted_policy_sorts_five_balls_on_a_jittered_ramp(task, seed):
    obs, _ = task.reset(seed=seed)
    assert task._num_spheres + task._num_cubes == 5
    policy = RampSortPolicy(task)
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range(task.MAX_EPISODE_STEPS):
            obs, reward, term, trunc, info = task.step(policy.act(obs))
            if term or trunc:
                break
    assert info["is_success"] and reward == 1, (seed, step + 1, policy.phase, info)
    assert all(task._at_rest_in_bin(b, task._sphere_bin_world_xy) for b in task._ball_body_ids()[0])
    assert all(task._at_rest_in_bin(b, task._cube_bin_world_xy) for b in task._ball_body_ids()[1])
