"""Regression tests for the public task API: seeding, spaces, validation, harness."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
import yaml

import imbench as ib
from imbench.eval.harness import run_benchmark
from imbench.policy.base_policy import PolicyInterface
from imbench.tasks.staged_reward import Stage, StagedReward

TASK = "ib-p2-cube-toss-v1"


def _qpos(task) -> np.ndarray:
    return task._env.sim.data.qpos.copy()


@pytest.fixture
def task():
    t = ib.make(TASK, obs_mode="proprio", seed=5)
    yield t
    t.close()


# ── Seeding ──────────────────────────────────────────────────────────


def test_constructor_seed_reproduces_first_reset():
    a = ib.make(TASK, obs_mode="proprio", seed=5)
    b = ib.make(TASK, obs_mode="proprio", seed=5)
    c = ib.make(TASK, obs_mode="proprio", seed=6)
    try:
        a.reset(), b.reset(), c.reset()
        assert np.allclose(_qpos(a), _qpos(b))
        assert not np.allclose(_qpos(a), _qpos(c))
    finally:
        a.close(), b.close(), c.close()


def test_explicit_seed_reproduces_and_unseeded_resets_vary(task):
    task.reset(seed=11)
    first = _qpos(task)
    task.reset()
    second = _qpos(task)
    task.reset(seed=11)
    assert np.allclose(first, _qpos(task))
    assert not np.allclose(first, second)


@pytest.mark.parametrize("task_id", [TASK, "ib-p7-packing-v1", "ib-m-keyboard-typing-v2"])
def test_make_and_reset_leave_global_rng_untouched(task_id):
    """Env construction, including the per-reset rebuilds of packing and
    keyboard-typing-v2, never advances the caller's global ``np.random``."""
    np.random.seed(123)
    expected = np.random.RandomState(123).random_sample(6)
    t = ib.make(task_id, obs_mode="proprio", seed=5)
    try:
        assert np.allclose(np.random.random_sample(3), expected[:3])
        t.reset(seed=9)
        t.reset()
        assert np.allclose(np.random.random_sample(3), expected[3:])
    finally:
        t.close()


def test_packing_seeded_layout_continues_task_stream():
    """Layout draws continue the stream that drew the piece count and split
    seed instead of replaying it."""
    overrides = {"randomization": {"block_placement": {"enabled": True}}}
    t = ib.make("ib-p7-packing-v1", obs_mode="proprio", seed=0, config_overrides=overrides)
    try:
        t.reset(seed=7)
        replay = np.random.RandomState(7).uniform(-t._place_x_range, t._place_x_range)
        assert t._piece_spawn_xy["lpiece_0"][0] != pytest.approx(replay)
    finally:
        t.close()


# ── Spaces ───────────────────────────────────────────────────────────


def test_observation_space_access_does_not_reset_episode(task):
    task.reset(seed=3)
    for _ in range(3):
        obs, *_ = task.step(np.zeros(task.action_space.shape))
    qpos, step = _qpos(task), task._step_count
    space = task.observation_space
    assert task._step_count == step
    assert np.allclose(_qpos(task), qpos)
    assert space.contains(obs)


def test_observation_space_probe_preserves_first_episode():
    a = ib.make(TASK, obs_mode="proprio", seed=4)
    b = ib.make(TASK, obs_mode="proprio", seed=4)
    try:
        _ = a.observation_space
        a.reset(), b.reset()
        assert np.allclose(_qpos(a), _qpos(b))
    finally:
        a.close(), b.close()


def test_action_space_is_float32_without_precision_warning(recwarn):
    t = ib.make(TASK, obs_mode="proprio")
    try:
        space = t.action_space
        assert space.dtype == np.float32
        assert not [w for w in recwarn if "precision lowered" in str(w.message)]
    finally:
        t.close()


def test_combo_frames_do_not_alias_across_steps():
    t = ib.make(TASK, obs_mode="combo", seed=0)
    try:
        obs0, _ = t.reset()
        obs1, *_ = t.step(np.zeros(t.action_space.shape))
        key = next(k for k in obs0 if k.startswith("rgb_"))
        assert obs0[key] is not obs1[key]
        assert not np.shares_memory(obs0[key], obs1[key])
        assert obs0[key].flags["C_CONTIGUOUS"]
        assert t.observation_space.contains(obs1)
    finally:
        t.close()


def test_rgb_array_render_without_cameras():
    t = ib.make(TASK, obs_mode="proprio", render_mode="rgb_array")
    try:
        t.reset(seed=0)
        frame = t.render()
        assert frame.ndim == 3 and frame.dtype == np.uint8
    finally:
        t.close()


def _fresh_frames(t) -> dict[str, np.ndarray]:
    sim = t._env.sim
    sim._render_context_offscreen.gl_ctx.make_current()
    return {
        f"rgb_{label}": sim.render(width=c["width"], height=c["height"], camera_name=c["robosuite_name"])[::-1]
        for label, c in t.camera_cfgs.items()
    }


@pytest.mark.parametrize("first_recorded_step", [0, 1])
def test_camera_frames_render_on_recorded_steps(first_recorded_step):
    """Every period-th obs from the first recorded step holds a frame of that step."""
    t = ib.make(TASK, obs_mode="combo")
    try:
        t.align_camera_frames(first_recorded_step)
        t.reset(seed=0)
        rng = np.random.default_rng(0)
        for step in range(1, 14):
            obs, *_ = t.step(rng.uniform(-1, 1, t.action_space.shape))
            if (step - first_recorded_step) % t._camera_render_skip == 0:
                for key, frame in _fresh_frames(t).items():
                    assert np.array_equal(frame, obs[key]), (step, key)
    finally:
        t.close()


def test_camera_frames_survive_gc_after_reset():
    """Collecting the context a hard reset replaced must not corrupt the live one."""
    t = ib.make(TASK, obs_mode="combo")
    try:
        obs, _ = t.reset(seed=0)
        gc.collect()
        for key, frame in _fresh_frames(t).items():
            assert np.array_equal(frame, obs[key]), key
    finally:
        t.close()


@pytest.mark.parametrize("close", [True, False], ids=["closed", "dropped"])
def test_retiring_one_env_keeps_another_rendering(close):
    a = ib.make(TASK, obs_mode="combo")
    b = ib.make(TASK, obs_mode="combo")
    try:
        a.reset(seed=0)
        obs_b, _ = b.reset(seed=1)
        a.step(np.zeros(a.action_space.shape))
        # Compare fresh renders of b taken before and after a is retired. The
        # obs frames from reset() are not a bit-exact baseline: GPU (EGL)
        # rasterisation can differ by 1 LSB on a few edge pixels between the
        # obs pipeline and a direct sim.render(), independent of a's lifetime.
        before = _fresh_frames(b)
        assert before.keys() == {k for k in obs_b if k.startswith("rgb_")}
        if close:
            a.close()
        del a
        gc.collect()
        after = _fresh_frames(b)
        for key, frame in after.items():
            assert frame.any(), f"{key} rendered blank after retiring another env"
            assert np.array_equal(frame, before[key]), key
    finally:
        b.close()


# ── Step contract ────────────────────────────────────────────────────


def test_step_info_reports_success_flag(task):
    _, info = task.reset(seed=0)
    assert info["is_success"] is False
    *_, info = task.step(np.zeros(task.action_space.shape))
    assert isinstance(info["is_success"], bool)
    assert isinstance(info["task_score"], float)


def test_default_reward_is_sparse_success(task, monkeypatch):
    task.reset(seed=0)
    *_, reward, _, _, info = task.step(np.zeros(task.action_space.shape))
    assert reward == 0.0 and info["is_success"] is False
    monkeypatch.setattr(task, "_is_success", lambda score: True)
    *_, reward, _, _, info = task.step(np.zeros(task.action_space.shape))
    assert reward == 1.0 and info["is_success"] is True


def test_dense_reward_is_task_score():
    t = ib.make(TASK, obs_mode="proprio", seed=5, config_overrides={"reward_type": "dense"})
    try:
        t.reset(seed=0)
        *_, reward, _, _, info = t.step(np.zeros(t.action_space.shape))
        assert reward == info["task_score"]
    finally:
        t.close()


def test_every_task_config_sets_sparse_reward():
    configs = sorted((Path(ib.__file__).parents[1] / "configs" / "tasks").rglob("*.yaml"))
    assert configs
    for path in configs:
        assert yaml.safe_load(path.read_text()).get("reward_type") == "sparse", path.name


def test_unknown_reward_type_rejected():
    with pytest.raises(ValueError, match="reward_type"):
        ib.make(TASK, config_overrides={"reward_type": "shaped"})


def test_task_score_is_side_effect_free(task):
    task.reset(seed=0)
    *_, info = task.step(np.zeros(task.action_space.shape))
    for _ in range(20):
        assert task.task_score() == info["task_score"]
    assert isinstance(task.task_score(recompute=True), float)


@pytest.mark.parametrize(
    "action",
    [np.full(7, np.nan), np.full(7, np.inf), np.zeros(6), np.zeros((1, 7))],
)
def test_invalid_actions_rejected(task, action):
    task.reset(seed=0)
    with pytest.raises(ValueError):
        task.step(action)


# ── Construction validation ──────────────────────────────────────────


def test_unknown_kwarg_rejected():
    with pytest.raises(TypeError, match="control_freq"):
        ib.make(TASK, control_freq=10)


@pytest.mark.parametrize("kwargs", [{"obs_mode": "bogus"}, {"render_mode": "bogus"}])
def test_invalid_modes_rejected(kwargs):
    with pytest.raises(ValueError):
        ib.make(TASK, **kwargs)


def test_unknown_controller_type_rejected():
    with pytest.raises(ValueError, match="controller type"):
        ib.make(TASK, config_overrides={"controller": {"type": "IK"}})


def test_unknown_task_suggests_close_match():
    with pytest.raises(ValueError, match="ib-p2-cube-toss-v1"):
        ib.make("ib-p2-cube-tos-v1")


def test_list_tasks_sorted():
    tasks = ib.list_tasks()
    assert tasks == sorted(tasks)
    assert len(tasks) == 50


# ── StagedReward ─────────────────────────────────────────────────────


def _stage(name, weight, value=1.0):
    return Stage(name, weight, lambda t, o: value)


def test_staged_reward_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        StagedReward([_stage("a", 0.5), _stage("a", 0.5)])


def test_staged_reward_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        StagedReward([_stage("a", 1.5), _stage("b", -0.5)])


def test_staged_reward_treats_nan_as_no_progress():
    reward = StagedReward([_stage("a", 0.5, np.nan), _stage("b", 0.5)])
    score, debug = reward.compute(None, {})
    assert score == pytest.approx(0.5)
    assert debug["a"] == 0.0


# ── Eval harness ─────────────────────────────────────────────────────


class _ZeroPolicy(PolicyInterface):
    def act(self, obs):
        return np.zeros(7)


class _FailingPolicy(PolicyInterface):
    def act(self, obs):
        raise RuntimeError("policy failure")


def test_harness_validates_inputs():
    with pytest.raises(ValueError, match="n_episodes"):
        run_benchmark(_ZeroPolicy(), [TASK], n_episodes=0)
    with pytest.raises(ValueError, match="Unknown task ids"):
        run_benchmark(_ZeroPolicy(), [TASK, "ib-nope-v1"], n_episodes=1)


def test_harness_closes_env_when_policy_raises(monkeypatch):
    closed = []
    real_make = ib.make

    def tracking_make(*args, **kwargs):
        env = real_make(*args, **kwargs)
        real_close = env.close
        env.close = lambda: (closed.append(True), real_close())
        return env

    monkeypatch.setattr(ib, "make", tracking_make)
    with pytest.raises(RuntimeError, match="policy failure"):
        run_benchmark(_FailingPolicy(), [TASK], n_episodes=1, obs_mode="proprio")
    assert closed == [True]


# ── Grasp detection ──────────────────────────────────────────────────


def test_grasp_detected_through_tactile_taxels():
    t = ib.make("ib-m-mirror-pick-place-v1", obs_mode="proprio", seed=0)
    try:
        t.reset(seed=0)
        assert t.tactile_enabled
        assert not t._gripper_contacts_object("cube")
        sim = t._env.sim
        gripper = t._env.robots[0].gripper["right"]
        site = sim.data.site_xpos[sim.model.site_name2id(gripper.important_sites["grip_site"])].copy()
        close = np.zeros(7)
        close[-1] = 1.0
        for _ in range(40):
            t._place_object("cube", site)
            t._set_object_velocity("cube", [0, 0, 0], [0, 0, 0])
            t.step(close)
        cube = t._env.get_object("cube")
        assert not t._env._check_grasp(gripper=gripper, object_geoms=cube.contact_geoms)
        assert t._gripper_contacts_object("cube")
    finally:
        t.close()
