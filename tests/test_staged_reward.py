"""Tests for the shared StagedReward utility and its pilot integrations.

Covers:
- imbench.tasks.staged_reward.StagedReward / Stage (unit tests)
- BaseTask.step() reward-routing fix (reward == task_score by default)
- Pilot retrofits: ib-m-mirror-pick-place-v1, ib-p6-pick-place-ycb-v1,
  ib-p7-balance-medium-v1/_v2
"""

import numpy as np
import pytest

from imbench.registry import make, list_tasks
from imbench.tasks.staged_reward import Stage, StagedReward


# ── Helpers ─────────────────────────────────────────────────────────

def _make_task(task_id, **kw):
    defaults = dict(obs_mode="proprio", render_mode=None, seed=42, max_episode_steps=10)
    defaults.update(kw)
    return make(task_id, **defaults)


DENSE = {"reward_type": "dense"}


def _zero_action(task):
    lo, _ = task.action_space.low, task.action_space.high
    return np.zeros(lo.shape[0])


# =====================================================================
#  StagedReward / Stage — unit tests
# =====================================================================

class TestStagedRewardUnit:

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            StagedReward(stages=[
                Stage("a", 0.5, lambda t, o: 0.0),
                Stage("b", 0.4, lambda t, o: 0.0),
            ])

    def test_requires_at_least_one_stage(self):
        with pytest.raises(ValueError):
            StagedReward(stages=[])

    def test_weighted_sum_no_success_fn(self):
        sr = StagedReward(stages=[
            Stage("a", 0.5, lambda t, o: o["a"]),
            Stage("b", 0.5, lambda t, o: o["b"]),
        ])
        score, values = sr.compute(None, {"a": 0.4, "b": 0.6})
        assert score == pytest.approx(0.5)
        assert values["a"] == pytest.approx(0.4)
        assert values["b"] == pytest.approx(0.6)
        assert values["_success"] is False

    def test_success_fn_saturates_to_one(self):
        sr = StagedReward(
            stages=[Stage("a", 1.0, lambda t, o: 0.1)],
            success_fn=lambda t, o: o["done"],
        )
        score, values = sr.compute(None, {"done": True})
        assert score == 1.0
        assert values["_success"] is True

    def test_sticky_stage_keeps_best_value(self):
        sr = StagedReward(stages=[Stage("a", 1.0, lambda t, o: o["a"], sticky=True)])
        score1, _ = sr.compute(None, {"a": 0.8})
        score2, _ = sr.compute(None, {"a": 0.1})
        assert score1 == pytest.approx(0.8)
        assert score2 == pytest.approx(0.8)  # regression ignored, sticky max kept

    def test_non_sticky_stage_tracks_current_value(self):
        sr = StagedReward(stages=[Stage("a", 1.0, lambda t, o: o["a"], sticky=False)])
        sr.compute(None, {"a": 0.8})
        score2, _ = sr.compute(None, {"a": 0.1})
        assert score2 == pytest.approx(0.1)

    def test_reset_clears_stage_progress(self):
        sr = StagedReward(stages=[Stage("a", 1.0, lambda t, o: o["a"], sticky=True)])
        sr.compute(None, {"a": 0.9})
        sr.reset()
        assert sr.stage_values["a"] == 0.0

    def test_values_outside_unit_range_are_clipped(self):
        sr = StagedReward(stages=[Stage("a", 1.0, lambda t, o: o["a"], sticky=False)])
        score, _ = sr.compute(None, {"a": 5.0})
        assert score == pytest.approx(1.0)
        score, _ = sr.compute(None, {"a": -5.0})
        assert score == pytest.approx(0.0)


# =====================================================================
#  Pilot task integration — bounded score + reward routing
# =====================================================================

class TestPilotTaskStaging:

    @pytest.mark.parametrize("task_id", [
        "ib-m-mirror-pick-place-v1",
        "ib-p6-pick-place-ycb-v1",
        "ib-p7-balance-medium-v1",
        "ib-p7-balance-medium-v2",
    ])
    def test_initial_score_bounded(self, task_id):
        task = _make_task(task_id)
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", [
        "ib-m-mirror-pick-place-v1",
        "ib-p6-pick-place-ycb-v1",
        "ib-p7-balance-medium-v1",
        "ib-p7-balance-medium-v2",
    ])
    def test_reward_tracks_task_score(self, task_id):
        """With ``reward_type: dense``, `reward` should equal
        `info['task_score']` (no custom reward_fn configured for these
        pilots)."""
        task = _make_task(task_id, config_overrides=DENSE)
        try:
            task.reset()
            action = _zero_action(task)
            for _ in range(3):
                _obs, reward, terminated, truncated, info = task.step(action)
                assert reward == pytest.approx(info["task_score"])
                if terminated or truncated:
                    break
        finally:
            task.close()

    def test_mirror_pick_place_stage_values_bounded(self):
        task = _make_task("ib-m-mirror-pick-place-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            task._compute_task_score(raw)
            for name, val in task._stage_debug.items():
                if name == "_success":
                    continue
                assert 0.0 <= val <= 1.0
        finally:
            task.close()


class TestRewardRoutingAllTasks:
    """Several task files implement `step()` directly against the raw
    robosuite env (bypassing both `BaseTask.step()` and
    `SingleArmBase.step()`'s reward routing; e.g. `ib-m-dining-table-v1`,
    `ib-m-keyboard-typing-v1` and `ib-m-keyboard-typing-v2`). Such a task
    can leave `reward` stuck at the raw env's hardcoded ``0.0`` while
    `info['task_score']` is meaningful.

    This sweeps every registered task, forces a known nonzero
    `_compute_task_score` (``reward_type: dense``) or success flag (the
    default ``reward_type: sparse``), and asserts `reward` reflects it (for
    tasks with no custom `reward_fn` configured).
    """

    @pytest.mark.parametrize("task_id", sorted(list_tasks()))
    def test_sparse_reward_routes_success_for_every_task(self, task_id):
        task = _make_task(task_id)
        try:
            task.reset()
            if task._reward_fn is not None:
                pytest.skip(f"{task_id} configures a custom reward_fn")
            assert task.reward_type == "sparse"
            action = _zero_action(task)
            task._is_success = lambda task_score: False
            _obs, reward, _term, _trunc, info = task.step(action)
            assert reward == 0.0 and info["is_success"] is False
            task._is_success = lambda task_score: True
            _obs, reward, _term, _trunc, info = task.step(action)
            assert reward == 1.0 and info["is_success"] is True, (
                f"{task_id}: sparse reward ({reward}) doesn't track the "
                f"forced success flag -- its step() likely bypasses reward routing."
            )
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", sorted(list_tasks()))
    def test_reward_routes_task_score_for_every_task(self, task_id):
        task = _make_task(task_id, config_overrides=DENSE)
        try:
            task.reset()
            if task._reward_fn is not None:
                pytest.skip(f"{task_id} configures a custom reward_fn")
            task._compute_task_score = lambda raw_obs: 0.77
            action = _zero_action(task)
            _obs, reward, _term, _trunc, _info = task.step(action)
            assert reward == pytest.approx(0.77), (
                f"{task_id}: reward ({reward}) doesn't track the forced "
                f"task_score (0.77) -- its step() likely bypasses reward "
                f"routing (raw env reward instead of task_score)."
            )
        finally:
            task.close()


class TestBalanceMediumGraded:

    def test_balance_medium_v1_score_zero_when_not_balanced(self):
        """Sanity check that the ported StagedReward reproduces the
        original two-branch formula: score is exactly 0.0 whenever the
        rod isn't currently balanced (not a nonzero hold-progress leak)."""
        task = _make_task("ib-p7-balance-medium-v1")
        try:
            task.reset()
            raw = task._get_sim_observations()
            if not task._is_balanced():
                assert task._compute_task_score(raw) == pytest.approx(0.0)
        finally:
            task.close()
