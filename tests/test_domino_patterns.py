"""Randomized Domino select layouts: catalog, geometry, physics and the scripted detector."""
import contextlib
import copy
import io

import numpy as np
import pytest

import imbench as ib
from data_synthetic.single_arm_pipeline.domino_select_policy import DominoSelectPolicy, detect_starter
from imbench.tasks.prehensile import domino_layouts as L
from tests.test_prehensile_regressions import _action, _finger_push

G = L.Geometry(0.045, np.array([0.06, 0.03, 0.01]) * 1.1, [0.005, 0.03, 0.04])


@pytest.fixture(scope="module")
def task():
    t = ib.make("ib-p3-domino-select-v1", obs_mode="proprio", max_episode_steps=5000)
    yield t
    t.close()


def test_catalog_has_twenty_patterns_and_every_one_is_drawn():
    assert len(L.PATTERNS) >= 20
    drawn = {L.sample_layout(np.random.RandomState(seed), G, list(L.PATTERNS), 48).pattern for seed in range(300)}
    assert drawn == set(L.PATTERNS)


@pytest.mark.parametrize("name", list(L.PATTERNS))
def test_layouts_keep_routes_gentle_and_aimed_at_red(name):
    for seed in range(20):
        layout = L.sample_layout(np.random.RandomState(seed), G, [name], 48)
        assert L.problem(layout, G, 48) is None
        route = layout.route.poses
        turns = [L._line_diff(a[2], b[2]) for a, b in zip(route, route[1:])]
        assert max(turns) <= L.MAX_TURN + 1e-6, (seed, turns)
        x, y, h = route[-1]
        rx, ry, rh = layout.red
        assert np.hypot(rx - x, ry - y) == pytest.approx(G.S)
        assert L._line_diff(h, np.arctan2(ry - y, rx - x)) < 1e-6 and L._line_diff(h, rh) < 1e-6
        assert not layout.route.low and layout.route.stop is None


def test_every_piece_is_within_the_robots_reach(task):
    """Route and decoy pieces alike stay within REACH of the robot base, so every starter can be pushed."""
    base = task.sim.data.body_xpos[task.sim.model.body_name2id("robot0_base")][:2]
    assert np.allclose(base, L.ROBOT_BASE_XY, atol=1e-3)
    for seed in range(60):
        task.reset(seed=seed)
        xy = np.array([p[:2] for p in task._network_layout.values()])
        assert np.hypot(*(xy - base).T).max() <= L.REACH + 1e-6, (seed, task._layout_pattern)


def test_layout_is_seeded_and_fixed_within_the_episode(task):
    task.reset(seed=11)
    before = copy.deepcopy((task._network_layout, task._solution_route, task._chain_specs))
    sizes = task.sim.model.geom_size.copy()
    task.reset(seed=12)
    task.reset(seed=11)
    assert before == (task._network_layout, task._solution_route, task._chain_specs)
    np.testing.assert_array_equal(sizes, task.sim.model.geom_size)
    for _ in range(5):
        task.step(_action(task))
    assert before == (task._network_layout, task._solution_route, task._chain_specs)


@pytest.mark.parametrize("name", list(L.PATTERNS))
def test_scripted_detector_finds_the_route_start(task, name):
    task._layout_patterns = [name]
    for seed in range(4):
        task.reset(seed=seed)
        found = detect_starter(task.sim, task._num_chains, task._dominoes_per_chain, task._table_height)
        assert found == task._solution_route[0], (name, seed, task._chain_specs)
    task._layout_patterns = list(L.PATTERNS)


@pytest.mark.parametrize("index,name", list(enumerate(L.PATTERNS)))
def test_route_topples_red_and_no_decoy_reaches_it(task, index, name):
    task._layout_patterns = [name]
    seed = 100 + index
    try:
        task.reset(seed=seed)
        push, ever = _finger_push(task, [task._solution_route[0]]), False
        for step in range(450):
            push(step)
            _, _, terminated, _, info = task.step(_action(task))
            ever |= info["is_success"]
            if terminated:
                break
        task.sim.data.xfrc_applied[:] = 0
        assert ever and not task._decoy_toppled, (seed, task._link_touch_step)
        assert (np.diff(task._link_touch_step) >= 0).all()
        for start in list(task._decoy_starts):
            task.reset(seed=seed)
            push, ever = _finger_push(task, [start]), False
            # Run past the scorer's failure: the break itself must stop the wave.
            for step in range(300):
                push(step)
                ever |= task.step(_action(task))[4]["is_success"]
            task.sim.data.xfrc_applied[:] = 0
            assert not ever and not task._target_displaced(), (seed, start)
    finally:
        task.sim.data.xfrc_applied[:] = 0
        task._layout_patterns = list(L.PATTERNS)


@pytest.mark.parametrize("seed", [6, 10, 25, 48])
def test_scripted_policy_topples_red_without_touching_the_route(seed):
    """Starters next to the robot base (6, 10), a strike whose hand would reach over
    the route (25) and one at the edge of reach (48)."""
    t = ib.make("ib-p3-domino-select-v1", obs_mode="proprio", privileged_obs=True)
    try:
        obs, _ = t.reset(seed=seed)
        policy, info = DominoSelectPolicy(rng=np.random.default_rng(seed)), {}
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(t.MAX_EPISODE_STEPS):
                obs, _, term, trunc, info = t.step(policy.act(obs, env=t))
                if term or trunc:
                    break
        assert info["is_success"], (seed, policy.phase, t._robot_touched_chain, t._decoy_toppled)
    finally:
        t.close()
