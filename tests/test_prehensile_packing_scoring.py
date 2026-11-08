"""Tests for packing task scoring and termination.

Covers:
- imbench.tasks.prehensile.packing_v1 (_compute_task_score, _check_termination)
- Packing lifecycle and scoring
"""

import numpy as np
import pytest

from imbench.registry import make


# =====================================================================
#  Packing task lifecycle
# =====================================================================

class TestPackingLifecycle:

    @pytest.mark.parametrize("task_id", [
        "ib-p7-packing-v1",
    ])
    def test_lifecycle(self, task_id):
        task = make(task_id, obs_mode="proprio", render_mode=None, seed=42,
                     max_episode_steps=10)
        try:
            obs, info = task.reset()
            assert "task_score" in info
            lo, _ = task.action_space.low, task.action_space.high
            obs2, r, t, tr, info2 = task.step(np.zeros(lo.shape[0]))
            assert 0.0 <= info2["task_score"] <= 1.0
        finally:
            task.close()


# =====================================================================
#  Packing scoring / termination paths
# =====================================================================

class TestPackingScoring:

    @pytest.mark.parametrize("task_id", [
        "ib-p7-packing-v1",
    ])
    def test_score_and_termination(self, task_id):
        task = make(task_id, obs_mode="proprio", render_mode=None, seed=42,
                     max_episode_steps=20)
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
            term = task._check_termination(raw)
            assert isinstance(term, (bool, np.bool_))
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", [
        "ib-p7-packing-v1",
    ])
    def test_step_5_times(self, task_id):
        task = make(task_id, obs_mode="proprio", render_mode=None, seed=42,
                     max_episode_steps=20)
        try:
            task.reset()
            lo, _ = task.action_space.low, task.action_space.high
            for _ in range(5):
                obs, r, t, tr, info = task.step(np.zeros(lo.shape[0]))
            assert 0.0 <= info["task_score"] <= 1.0
        finally:
            task.close()
