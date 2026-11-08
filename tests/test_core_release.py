"""Regression tests for core task API fixes: action bounds, seeding, lifecycle,
input validation, gymnasium registration and object factories."""

from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import gymnasium
import numpy as np
import pytest

import imbench as ib
from imbench.envs.single_arm_table_env import SingleArmTableEnv
from imbench.registry import TASK_REGISTRY
from imbench.tasks import base_task

TASK = "ib-p2-cube-toss-v1"
HOBI = "ib-p1-plate-shelf-hobi-v1"


# ── Action bounds ────────────────────────────────────────────────────


@pytest.mark.parametrize("task_id", [HOBI, "ib-p1-bowl-table-v1", TASK], ids=["hobi_ik", "hetbi_ik", "osc_pose"])
def test_out_of_range_actions_act_as_the_bound(task_id):
    """+z actions of 1 and 10 move every arm identically (docs: clipped to [-1, 1])."""
    t = ib.make(task_id, obs_mode="proprio", seed=0)
    try:
        eef = []
        for magnitude in (1.0, 10.0):
            t.reset(seed=0)
            action = np.zeros(t.action_space.shape)
            action[2::7] = magnitude
            for _ in range(5):
                obs, *_ = t.step(action)
            eef.append(np.concatenate([obs[k] for k in sorted(obs) if k.endswith("eef_pos")]))
        np.testing.assert_allclose(eef[0], eef[1], atol=1e-6)
    finally:
        t.close()


def test_hobi_rejected_action_leaves_env_usable():
    t = ib.make(HOBI, obs_mode="proprio", seed=0)
    try:
        t.reset(seed=0)
        n = t.action_space.shape[0]
        with pytest.raises(ValueError, match="non-finite"):
            t.step(np.full(n, np.nan))
        with pytest.raises(ValueError, match="does not match action_space"):
            t.step(np.zeros(7))
        t.step(np.zeros(n))
        assert all(np.isfinite(target).all() for target in t._target_pos)
    finally:
        t.close()


# ── Seeding ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "task_id", ["ib-p1-edge-slide-v1", "ib-p6-pick-place-ycb-v1", "ib-m-keyboard-typing-v2"]
)
def test_constructor_seed_sets_first_episode(task_id):
    """``make(seed=s).reset()`` is the episode ``reset(seed=s)`` gives, even for
    tasks that draw from their stream while being constructed."""
    a = ib.make(task_id, obs_mode="proprio", seed=3)
    b = ib.make(task_id, obs_mode="proprio", seed=0)
    try:
        obs_a, info_a = a.reset()
        obs_b, info_b = b.reset(seed=3)
        assert info_a == info_b
        assert obs_a.keys() == obs_b.keys()
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key], err_msg=key)
    finally:
        a.close(), b.close()


# ── Keyboard-typing-v2 word display assets ───────────────────────────


def test_keyboard_v2_reuses_word_textures_across_resets():
    """MuJoCo caches textures by file path, so a word's glyphs keep their path."""
    t = ib.make("ib-m-keyboard-typing-v2", obs_mode="proprio", seed=0)
    try:
        t.reset(seed=0)
        first = t._word_display_xml_path
        t.reset(seed=0)
        assert t._word_display_xml_path == first
        assets_dir = t._word_display_assets_dir
    finally:
        t.close()
    assert not assets_dir.exists()


def test_keyboard_v1_vocabulary_reuses_word_textures_across_resets():
    """A word drawn again from the vocabulary reuses its glyph files' path."""
    t = ib.make("ib-m-keyboard-typing-v1", obs_mode="proprio",
                config_overrides={"word": None, "words": ["ROBOT", "BENCH"]})
    try:
        paths = {}
        for seed in range(6):
            t.reset(seed=seed)
            paths.setdefault(t._word, set()).add(t._word_display_xml_path)
        assert set(paths) == {"ROBOT", "BENCH"}
        assert all(len(p) == 1 for p in paths.values())
        assets_dir = t._word_display_assets_dir
    finally:
        t.close()
    assert not assets_dir.exists()


# ── Lifecycle and input validation ───────────────────────────────────


@pytest.mark.parametrize("task_id", [
    "ib-p2-slide-catch-v1", "ib-p1-recover-peg-insert-v1", "ib-p1-recover-peg-insert-v2",
    "ib-p3-ball-cage-escape-v1", "ib-p6-stack-collapse-recovery-v1", "ib-p7-packing-v1",
])
def test_closed_task_with_custom_step_or_reset_raises(task_id):
    t = ib.make(task_id, obs_mode="proprio")
    t.reset(seed=0)
    t.close()
    for call in (lambda: t.step(np.zeros(t.action_space.shape)), lambda: t.reset(seed=0)):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_closed_task_raises_on_use():
    t = ib.make(TASK, obs_mode="proprio", render_mode="rgb_array")
    t.reset(seed=0)
    t.close()
    t.close()
    for call in (lambda: t.step(np.zeros(7)), lambda: t.reset(seed=0), t.render):
        with pytest.raises(RuntimeError, match="closed"):
            call()


def test_human_render_without_display_raises(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match="neither DISPLAY nor WAYLAND_DISPLAY"):
        ib.make(TASK, obs_mode="proprio", render_mode="human")


@pytest.mark.parametrize("steps", [0, -5, 2.5])
def test_invalid_max_episode_steps_rejected(steps):
    with pytest.raises(ValueError, match="max_episode_steps"):
        ib.make(TASK, obs_mode="proprio", max_episode_steps=steps)


@pytest.mark.parametrize("task_id", [TASK, HOBI])
def test_non_dict_config_overrides_rejected(task_id):
    with pytest.raises(TypeError, match="config_overrides"):
        ib.make(task_id, obs_mode="proprio", config_overrides="x")


@pytest.mark.parametrize("task_id", [None, 5])
def test_non_string_task_id_rejected(task_id):
    with pytest.raises(TypeError, match="task_id must be a str"):
        ib.make(task_id)


def test_missing_configs_dir_asks_for_editable_install(monkeypatch, tmp_path):
    monkeypatch.setattr(base_task, "CONFIGS_DIR", tmp_path / "configs" / "tasks")
    with pytest.raises(FileNotFoundError, match="pip install -e"):
        base_task.load_task_config(TASK)


# ── Rendered arrays ──────────────────────────────────────────────────


def test_rgb_array_render_is_contiguous():
    t = ib.make(TASK, obs_mode="proprio", render_mode="rgb_array")
    try:
        t.reset(seed=0)
        assert t.render().flags["C_CONTIGUOUS"]
    finally:
        t.close()


def test_sheltered_mirrored_front_cam_is_contiguous():
    t = ib.make("ib-p1-sheltered-grasp-v1", obs_mode="combo", seed=0)
    try:
        obs, _ = t.reset(seed=0)
        assert obs["rgb_exo_front_cam"].flags["C_CONTIGUOUS"]
        obs, *_ = t.step(np.zeros(t.action_space.shape))
        assert obs["rgb_exo_front_cam"].flags["C_CONTIGUOUS"]
    finally:
        t.close()


# ── Gymnasium registration ───────────────────────────────────────────


def test_gymnasium_does_not_offer_ood_variants_as_upgrades():
    """``-v2`` OOD variants live in the ``imbench-ood`` namespace, so gymnasium
    never reads them as newer versions of the ``-v1`` tasks."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for task_id, entry_point in TASK_REGISTRY.items():
            gym_id = f"imbench-ood/{task_id}" if task_id.endswith("-v2") else task_id
            assert gymnasium.spec(gym_id).entry_point is entry_point
    assert not [w for w in caught if "out of date" in str(w.message)]
    assert "ib-p1-edge-slide-v2" not in gymnasium.envs.registry


def test_gymnasium_makes_ood_variant():
    env = gymnasium.make("imbench-ood/ib-p1-edge-slide-v2", obs_mode="proprio")
    try:
        assert env.unwrapped.TASK_ID == "ib-p1-edge-slide-v2"
    finally:
        env.close()


@pytest.mark.slow
def test_gymnasium_compat_script_passes_without_optional_sb3():
    script = Path(__file__).resolve().parents[1] / "scripts" / "test_gymnasium_compat.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout[-3000:]
    assert "out of date" not in result.stderr


# ── Object factories ─────────────────────────────────────────────────


@pytest.mark.parametrize("obj_type", ["prism", "torus"])
def test_prism_and_torus_static_or_free(obj_type):
    static = SingleArmTableEnv._make_object({"type": obj_type, "name": "a", "joints": None})
    free = SingleArmTableEnv._make_object({"type": obj_type, "name": "b"})
    assert static.joints == []
    assert len(free.joints) == 1
