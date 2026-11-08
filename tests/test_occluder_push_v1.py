"""Occluder Push v1: per-seed cover layouts, cover physics, scoring and the scripted policy."""

import contextlib
import io
from collections import Counter

import numpy as np
import pytest

import imbench as ib
from imbench.tasks.prehensile.occluder_push_v1 import (
    BOX_TRAVEL, COVER_KINDS, DISTRACTOR_COLORS, _MAX_PER_SEED, _POOL_OF, _POOL_SIZE, _boxes_overlap, _may_overlap,
)

TASK = "ib-p5-occluder-push-v1"


def _make(**kw):
    kw.setdefault("obs_mode", "proprio")
    kw.setdefault("max_episode_steps", 3000)
    with contextlib.redirect_stdout(io.StringIO()):
        return ib.make(TASK, seed=0, **kw)


@pytest.fixture(scope="module")
def task():
    t = _make(privileged_obs=True)
    yield t
    t.close()


def _steps(task, n, grip=-1.0):
    a = np.zeros(7)
    a[6] = grip
    out = None
    for _ in range(n):
        out = task.step(a)
    return out


def _signature(task):
    return [(c["kind"], c["name"], tuple(np.round(c["pos"], 6)), round(c["yaw"], 6), c["cube"])
            for c in task._covers]


# ── Layout ──────────────────────────────────────────────────────────

class TestLayout:

    def test_draws_follow_the_rules(self, task):
        counts, kinds, empties, red = Counter(), Counter(), Counter(), Counter()
        original = 0
        for seed in range(60):
            task.reset(seed=seed)
            covers = task._covers
            n = len(covers)
            ks = [c["kind"] for c in covers]
            assert sum(c["cube"] == "cube" for c in covers) == 1
            assert len({c["name"] for c in covers}) == n
            if task._classic:
                # The original scene: three plates in a row, every one on a cube.
                original += 1
                assert ks == ["plate"] * 3 and all(c["cube"] for c in covers)
                ys = sorted(c["pos"][1] for c in covers)
                assert ys[1] - ys[0] > 0.14 and ys[2] - ys[1] > 0.14
                continue
            counts[n] += 1
            kinds.update(ks)
            red[task._red_cover["kind"]] += 1
            assert 3 <= n <= 4
            assert len(set(ks)) >= 2
            per_pool = Counter(_POOL_OF[k] for k in ks)
            assert all(per_pool[p] <= _MAX_PER_SEED[p] for p in per_pool)
            # Plates always rest on a cube; 0-2 covers are empty.
            assert all(c["cube"] is not None for c in covers if c["kind"] == "plate")
            n_empty = sum(c["cube"] is None for c in covers)
            empties[n_empty] += 1
            assert 0 <= n_empty <= 2
        assert 6 <= original <= 26          # about a quarter
        assert set(counts) == {3, 4}
        assert set(kinds) == set(COVER_KINDS)
        assert set(empties) == {0, 1, 2}
        assert len(red) >= 4

    def test_distractor_colours_are_never_red_like(self, task):
        allowed = np.array(list(DISTRACTOR_COLORS.values()))
        for seed in range(10):
            task.reset(seed=seed)
            sim = task._env.sim
            for name in task._active_cubes[1:]:
                bid = sim.model.body_name2id(f"{name}_main")
                rgba = next(tuple(sim.model.geom_rgba[g]) for g in range(sim.model.ngeom)
                            if sim.model.geom_bodyid[g] == bid)
                assert np.isclose(allowed, rgba, atol=1e-5).all(axis=1).any()
                r, g, b, _ = rgba
                assert not (r > 0.7 and g < 0.6 and b < 0.6), "distractor looks red/orange/pink"

    def test_layout_is_reproducible_per_seed(self, task):
        task.reset(seed=7)
        first = _signature(task)
        task.reset(seed=8)
        assert _signature(task) != first
        task.reset(seed=7)
        assert _signature(task) == first

    def test_every_cover_can_still_be_removed(self, task):
        """Re-check the layout against the validator's own rules."""
        for seed in range(20):
            task.reset(seed=seed)
            rects = []
            for c in task._covers:
                rs, pts, high = task._cover_rects(c["kind"], c["pos"], c["yaw"], c["push_dir"])
                assert all(task._on_table(b) for b, _, _ in rs)
                assert all(task._reachable(p) for p in pts)
                assert all(task._reachable(p, high=True) for p in high)
                rects.append(rs)
                if c["kind"] == "cup":
                    assert task._reachable(c["setdown"])
            keep_out = (task._patch_pos, np.eye(2), np.full(2, task._patch_keepout))
            for i, rs in enumerate(rects):
                for b, cls, kind in rs:
                    if not _may_overlap((cls, kind), ("rest", "patch")):
                        assert not _boxes_overlap(b, keep_out, 0.0)
                    for j in range(i):
                        for ob, ocls, okind in rects[j]:
                            if not _may_overlap((cls, kind), (ocls, okind)):
                                assert not _boxes_overlap(b, ob, 0.0), (seed, i, j, cls, ocls)

    def test_unused_bodies_are_parked(self, task):
        task.reset(seed=3)
        used = {c["name"] for c in task._covers} | set(task._active_cubes)
        for pool in ("plate", "cup", "flap", "box"):
            for i in range(_POOL_SIZE[pool]):
                name = f"{pool}_{i}"
                p = task._body_pos(name)
                assert (np.linalg.norm(p[:2]) < 1.0) == (name in used)
        for i in range(4):
            name = f"distractor_{i}"
            assert (np.linalg.norm(task._body_pos(name)[:2]) < 1.0) == (name in used)


# ── Start state ─────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_scene_rests_without_credit(task, seed):
    obs, _ = task.reset(seed=seed)
    np.testing.assert_allclose(obs["object_pos"], task._body_pos("cube"))
    names = task._active_cubes + [c["name"] for c in task._covers if c["kind"] in ("plate", "cup")]
    before = {n: task._body_pos(n) for n in names}
    _, _, terminated, _, info = _steps(task, 60)
    assert info["task_score"] == 0.0
    assert not terminated and not info["is_success"]
    for n in names:
        assert np.linalg.norm(task._body_pos(n) - before[n]) < 0.006, n
    for c in task._covers:
        assert task.cover_opening(c) == 0.0


@pytest.mark.parametrize("seed", [0, 3])
def test_hidden_cubes_are_not_visible(seed):
    t = _make(obs_mode="combo")
    try:
        obs, _ = t.reset(seed=seed)
        for _ in range(10):
            obs, *_ = t.step(np.zeros(7))
        for cam in ("rgb_ego_cam", "rgb_exo_bird_cam"):
            im = obs[cam].astype(int)
            r, g, b = im[..., 0], im[..., 1], im[..., 2]
            assert int(((r > 150) & (g < 90) & (b < 90)).sum()) < 5, cam
    finally:
        t.close()


# ── Cover physics ───────────────────────────────────────────────────

def _flap_of(task, kind):
    for seed in range(60):
        task.reset(seed=seed)
        for c in task._covers:
            if c["kind"] == kind:
                return c
    raise AssertionError(f"no {kind} in 60 seeds")


@pytest.mark.parametrize("kind,start_deg,opens", [
    ("flap", 95, True), ("flap", 70, False),
    ("spring_flap", 95, False), ("spring_flap", 105, True),
])
def test_flap_hinges(task, kind, start_deg, opens):
    """A plain flap stays open once past upright; a spring flap only past its latch."""
    c = _flap_of(task, kind)
    qa, da = task._joint_adr(f"{c['name']}_flap_hinge")
    sim = task._env.sim
    sim.data.qpos[qa] = np.deg2rad(start_deg)
    sim.data.qvel[da] = 0.0
    sim.forward()
    _steps(task, 250)
    angle = np.rad2deg(sim.data.qpos[qa])
    if opens:
        assert angle > 100.0
    else:
        assert angle < 3.0


def test_box_lid_only_slides_along_its_axis(task):
    for seed in range(60):
        task.reset(seed=seed)
        box = next((c for c in task._covers if c["kind"] == "box"), None)
        if box:
            break
    sim = task._env.sim
    qa, da = task._joint_adr(f"{box['name']}_lid_slide")
    lid = sim.model.body_name2id(f"{box['name']}_lid")
    side = np.array([-np.sin(box["yaw"]), np.cos(box["yaw"]), 0.0])
    start = np.array(sim.data.xpos[lid])
    for _ in range(100):
        sim.data.xfrc_applied[lid, :3] = 5.0 * side
        task.step(np.zeros(7))
    sim.data.xfrc_applied[lid, :3] = 0.0
    assert abs(sim.data.qpos[qa]) < 0.003
    assert np.linalg.norm(sim.data.xpos[lid][:2] - start[:2]) < 0.003
    sim.data.qpos[qa] = BOX_TRAVEL
    sim.forward()
    _steps(task, 50)
    assert sim.data.qpos[qa] > BOX_TRAVEL - 0.01
    if box["cube"]:
        assert task.cover_opening(box) == 1.0


def test_opening_each_cover_reveals_it(task):
    seen = set()
    for seed in range(40):
        task.reset(seed=seed)
        sim = task._env.sim
        for c in task._covers:
            if c["kind"] in seen or c["cube"] is None:
                continue
            seen.add(c["kind"])
            if c["kind"] == "plate":
                p = task._body_pos(c["name"])
                task._place_object(c["name"], [p[0] + 0.0, p[1] + 0.0, p[2] + 0.3])
                sim.forward()
                p2 = task._body_pos(c["name"])
                task._place_object(c["name"], [p2[0] + 0.2, p2[1], p2[2]])
            elif c["kind"] == "cup":
                p = task._body_pos(c["name"])
                task._place_object(c["name"], [p[0], p[1], p[2] + 0.12])
            elif c["kind"] in ("flap", "spring_flap"):
                qa, _ = task._joint_adr(f"{c['name']}_flap_hinge")
                sim.data.qpos[qa] = np.deg2rad(105)
            else:
                qa, _ = task._joint_adr(f"{c['name']}_lid_slide")
                sim.data.qpos[qa] = BOX_TRAVEL
            sim.forward()
            assert task.cover_opening(c) == 1.0, c["kind"]
        if seen == set(COVER_KINDS):
            break
    assert seen == set(COVER_KINDS)


# ── Scoring and failure ─────────────────────────────────────────────

def _rest_on_patch(task, name, dy=0.0):
    p = task._patch_pos
    z = task._table_height + 0.002 + task._cube_size + 0.001
    task._place_object(name, [p[0], p[1] + dy, z])
    task._set_object_velocity(name, [0, 0, 0], [0, 0, 0])
    task._env.sim.forward()


def test_red_cube_on_patch_succeeds(task):
    task.reset(seed=4)
    _rest_on_patch(task, "cube")
    _, _, terminated, _, info = _steps(task, 15)
    assert info["is_success"] and terminated and info["task_score"] == 1.0


def test_distractor_on_patch_blocks_success(task):
    for seed in range(20):
        task.reset(seed=seed)
        if len(task._active_cubes) > 1:
            break
    _rest_on_patch(task, task._active_cubes[1], dy=0.03)
    _rest_on_patch(task, "cube", dy=-0.015)
    _, _, terminated, _, info = _steps(task, 15)
    assert task._distractor_on_patch()
    assert not info["is_success"] and not terminated


@pytest.mark.parametrize("what", ["red", "distractor", "plate", "cup"])
def test_anything_off_the_table_fails(task, what):
    for seed in range(60):
        task.reset(seed=seed)
        kinds = [c["kind"] for c in task._covers]
        if (what == "distractor" and len(task._active_cubes) > 1) or what == "red" or what in kinds:
            break
    name = {"red": "cube", "distractor": task._active_cubes[-1]}.get(what) or next(
        c["name"] for c in task._covers if c["kind"] == what)
    task._place_object(name, [0.0, 1.0, 0.03])
    task._env.sim.forward()
    _, _, terminated, _, info = _steps(task, 3)
    assert terminated and not info["is_success"]


# ── Scripted policy ─────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [1, 6])
def test_scripted_policy_solves(seed):
    from data_synthetic.single_arm_pipeline.occluder_push_policy import OccluderPushPolicy
    t = _make(privileged_obs=True, max_episode_steps=20000)
    try:
        obs, _ = t.reset(seed=seed)
        pol = OccluderPushPolicy(verbose=False, rng=np.random.default_rng(seed))
        info = {}
        while True:
            obs, _, term, trunc, info = t.step(pol.act(obs, env=t))
            if term or trunc or pol.phase == "done":
                break
        assert info["is_success"], [c["kind"] for c in t._covers]
    finally:
        t.close()
