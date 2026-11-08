"""Slide Catch v1 per-seed randomization: what varies, that it is reproducible, that every
drawn scene passes the teleop feasibility check, and that the check holds in physics."""

import contextlib
import io

import numpy as np
import pytest

from imbench import make
from data_synthetic.single_arm_pipeline.slide_catch_policy import SlideCatchPolicy

TASK = "ib-p2-slide-catch-v1"


@pytest.fixture(scope="module")
def task():
    t = make(TASK, obs_mode="proprio", render_mode=None, seed=0, privileged_obs=True)
    yield t
    t.close()


def _scenes(task, seeds):
    out = []
    for seed in seeds:
        task.reset(seed=seed)
        out.append({**{k: v for k, v in task._scene.items() if k != "_profile"},
                    "yaw": task._incline_yaw, "anchor": task._anchor_world_pos.copy(),
                    "eef": task._env.sim.data.site_xpos[
                        task._env.sim.model.site_name2id("gripper0_right_grip_site")].copy()})
    return out


def test_scene_draws_vary_across_seeds(task):
    sc = _scenes(task, range(40))
    assert {s["n"] for s in sc} == {2, 3}
    assert {s["event"]["kind"] for s in sc} == {"pause", "speedup"}
    for key in ("tilt0", "rate", "yaw"):
        assert np.std([s[key] for s in sc]) > 0.0, key
    assert np.std([s["anchor"][1] for s in sc]) > 0.02          # sideways shift
    assert np.std([s["eef"][2] for s in sc]) > 0.01             # arm start height
    fps = [f for s in sc for f in s["fp"]]
    assert min(fps) < 0.016 and max(fps) > 0.020
    # Which cube leaves first is not tied to which one sits nearest the edge.
    firsts = []
    for s_ in sc:
        if s_["n"] > 1:
            deps = [d["depart"] for d in task._predict_departures(s_)]
            firsts.append(int(np.argmin(s_["x"])) == int(np.argmin(deps)))
    assert not all(firsts)


def test_reset_is_reproducible_per_seed(task):
    a, b = _scenes(task, [7]), _scenes(task, [7])
    for k in ("n", "fp", "mu", "x", "y", "tilt0", "rate", "event", "yaw"):
        assert a[0][k] == b[0][k], k
    np.testing.assert_allclose(a[0]["eef"], b[0]["eef"])


def test_every_scene_passes_the_teleop_check(task):
    for seed in range(30):
        task.reset(seed=seed)
        ok, why = task._scene_feasible(task._scene, task._bin_start)
        assert ok, (seed, why)
        assert all(d["depart"] is not None for d in task._departures)


def test_hinge_line_inner_end_is_on_the_table_edge(task):
    edge_x = task._table_offset[0] - task._hinge_offset
    for seed in range(10):
        task.reset(seed=seed)
        ends = [task._plate_local_to_world(np.array([0.0, y, 0.0]), 0.0)
                for y in (-task._incline_width / 2, task._incline_width / 2)]
        assert max(e[0] for e in ends) == pytest.approx(edge_x, abs=1e-6)


def test_unused_cubes_are_parked_and_ignored(task):
    for seed in range(20):
        obs, _ = task.reset(seed=seed)
        if task._scene["n"] < 3:
            break
    parked = sorted(task._parked)
    assert parked and set(parked).isdisjoint(task._active_cubes)
    for name in parked:
        assert f"{name}_pos" not in obs
    # A parked cube far off-scene never counts as fallen.
    for _ in range(5):
        _, _, terminated, _, _ = task.step(np.zeros(task.action_space.shape))
        assert not terminated


def test_cube_colours_stay_clear_of_the_incline(task):
    sim = task._env.sim
    for seed in range(10):
        task.reset(seed=seed)
        plate = sim.model.geom_rgba[task._plate_geoms(sim)[0]][:3]
        for name in task._active_cubes:
            bid = sim.model.body_name2id(f"{name}_main")
            gid = next(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] == bid)
            assert np.linalg.norm(sim.model.geom_rgba[gid][:3] - plate) >= 0.35


def test_arm_start_holds_still(task):
    for seed in (1, 4):
        obs, _ = task.reset(seed=seed)
        p0 = obs["robot0_eef_pos"].copy()
        for _ in range(50):
            obs, *_ = task.step(np.zeros(task.action_space.shape))
        assert np.linalg.norm(obs["robot0_eef_pos"] - p0) < 0.01


@pytest.mark.parametrize("seed", [2, 6, 10])
def test_cubes_leave_the_plate_when_the_model_says(seed):
    """Physics within the checker's model margin (early) / 1 s (late) of the 1-D model."""
    t = make(TASK, obs_mode="proprio", render_mode=None, seed=0, privileged_obs=True,
             config_overrides={"task_conditions": {"failure": {"floor_z_threshold": -100.0}}})
    try:
        t.reset(seed=seed)
        sim = t._env.sim
        c, s = np.cos(t._incline_yaw), np.sin(t._incline_yaw)
        left = {}
        lift = np.zeros(t.action_space.shape)
        lift[0], lift[2] = -0.3, 0.3          # bin up and back, out of the cubes' way
        for k in range(3000):
            t.step(lift if k < 150 else np.zeros(t.action_space.shape))
            for name in t._active_cubes:
                p = sim.data.body_xpos[sim.model.body_name2id(f"{name}_main")] - t._anchor_world_pos
                if name not in left and c * p[0] + s * p[1] <= 0.0:
                    left[name] = (k + 1) / 100
            if len(left) == len(t._active_cubes):
                break
        for name, dep in zip(t._active_cubes, t._departures):
            assert -t._feas_model_margin <= left[name] - dep["depart"] <= 1.0, (name, left[name], dep)
    finally:
        t.close()


@pytest.mark.parametrize("seed", [0, 7, 10])
def test_teleop_limited_policy_catches_every_cube(seed):
    """The scripted policy at the gamepad's action cap and filter, reacting to cube
    positions 1 s old, still catches every cube."""
    t = make(TASK, obs_mode="proprio", render_mode=None, seed=0, privileged_obs=True)
    try:
        obs, _ = t.reset(seed=seed)
        with contextlib.redirect_stdout(io.StringIO()):
            policy = SlideCatchPolicy(rng=np.random.default_rng(seed), max_action=0.3, obs_delay_steps=100)
            filt = np.zeros(6)
            while True:
                a = policy.act(obs, env=t)
                filt = filt + np.clip(0.8 * filt + 0.2 * a[:6] - filt, -0.4, 0.4)  # gamepad action_filter
                a = a.copy()
                a[:6] = filt
                obs, _, terminated, truncated, info = t.step(a)
                if terminated or truncated:
                    break
        assert info["is_success"], (seed, t._scene["n"])
    finally:
        t.close()


def test_incline_footprint_stays_within_the_table_width(task):
    half_y = task._table_full_size[1] / 2
    for seed in range(30):
        task.reset(seed=seed)
        for lx in (0.0, task._incline_length):
            for ly in (-task._incline_width / 2, task._incline_width / 2):
                assert abs(task._plate_local_to_world(np.array([lx, ly, 0.0]), 0.0)[1]) <= half_y + 1e-6
