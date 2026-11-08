"""Tests for base_task.py methods and registry completeness.

Covers:
- imbench.tasks.base_task (filter_obs_for_saving, render, task_score, save_extras, step_count)
- Registry completeness (all registered tasks have YAML configs)
"""

from pathlib import Path

import numpy as np
import pytest

from imbench.registry import TASK_REGISTRY, make
from imbench.tasks.base_task import BaseTask


# ── Helpers ─────────────────────────────────────────────────────────

def _make_task(task_id, **kw):
    defaults = dict(obs_mode="combo", render_mode=None, seed=42, max_episode_steps=10)
    defaults.update(kw)
    return make(task_id, **defaults)


# =====================================================================
#  base_task.py — filter_obs_for_saving, render, task_score
# =====================================================================

class TestBaseTaskMethods:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p2-cube-toss-v1")
        t.reset()
        yield t
        t.close()

    def test_filter_obs_for_saving(self, task):
        obs, _ = task.reset()
        filtered = task.filter_obs_for_saving(obs)
        assert isinstance(filtered, dict)
        if task._save_proprio:
            for k in obs:
                if not k.startswith("rgb_") and not k.startswith("depth_"):
                    assert k in filtered

    def test_filter_obs_camera_keys(self, task):
        obs, _ = task.reset()
        rgb_keys = [k for k in obs if k.startswith("rgb_")]
        assert len(rgb_keys) > 0, "combo mode should include camera obs"

    def test_task_score_property(self, task):
        task.reset()
        score = task.task_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_render_noop_offscreen(self, task):
        """render() with non-human mode should not crash."""
        task.render()

    def test_save_extras_property(self, task):
        extras = task.save_extras
        assert isinstance(extras, dict)
        for key in ("action", "reward", "task_score"):
            assert key in extras, f"save_extras missing key: {key!r}"

    def test_step_count_resets(self, task):
        task.reset()
        assert task._step_count == 0
        task.step(np.zeros(7))
        assert task._step_count == 1
        task.reset()
        assert task._step_count == 0


# =====================================================================
#  Registry completeness
# =====================================================================

class TestRegistryCompleteness:

    def test_all_registered_tasks_have_configs(self):
        """All registered tasks should have matching YAML configs
        (except custom/test tasks)."""
        config_dir = Path(__file__).resolve().parent.parent / "configs" / "tasks"
        for tid in TASK_REGISTRY:
            if tid.startswith("ib-test-"):
                continue
            yaml_files = list(config_dir.rglob(f"{tid}.yaml"))
            assert len(yaml_files) > 0, f"No YAML config for {tid}"

    def test_make_returns_base_task(self):
        for tid in ["ib-p2-cube-toss-v1", "ib-p2-pendulum-grasp-v1"]:
            task = _make_task(tid)
            try:
                assert isinstance(task, BaseTask)
            finally:
                task.close()
