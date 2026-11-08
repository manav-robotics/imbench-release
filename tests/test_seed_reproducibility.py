"""Regression coverage for identical-reset-per-seed reproducibility.

A task that seeds an internal RNG from process entropy (``os.urandom``) or
the *global* ``random`` module, independent of ``self.seed``, breaks the eval
harness's "K rollout repeats per seed, identical reset" protocol
(consecutive ``reset(seed=7)`` calls must reproduce the same sim state).
"""

import numpy as np
import pytest

from imbench.registry import list_tasks, make


TASKS_WITH_PER_EPISODE_RANDOMIZATION = [
    "ib-p6-slip-recovery-v1",
    "ib-p6-slip-recovery-v2",
    "ib-p6-pick-place-ycb-v1",
    "ib-m-keyboard-typing-v1",
    "ib-m-keyboard-typing-v2",
]


def _make_task(task_id):
    return make(task_id, obs_mode="proprio", render_mode=None, seed=0,
                max_episode_steps=10)


def _qpos_fingerprint(task):
    raw = task._get_sim_observations(force_update=True)
    return np.round(task._env.sim.data.qpos, 6).tobytes()


class TestSeedReproducibility:

    @pytest.mark.parametrize("task_id", TASKS_WITH_PER_EPISODE_RANDOMIZATION)
    def test_same_seed_reproduces_identical_reset(self, task_id):
        task = _make_task(task_id)
        try:
            task.reset(seed=7)
            fp1 = _qpos_fingerprint(task)
            task.reset(seed=7)
            fp2 = _qpos_fingerprint(task)
            assert fp1 == fp2, (
                f"{task_id}: reset(seed=7) called twice produced different "
                f"sim state — an internal RNG is drifting independently of "
                f"the passed seed."
            )
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", TASKS_WITH_PER_EPISODE_RANDOMIZATION)
    def test_different_seeds_still_vary(self, task_id):
        """Guard against a fix that over-corrects into always-identical
        resets regardless of seed (e.g. accidentally hardcoding the draw).

        Checks ``self._rng`` directly (a draw taken immediately after
        reset) rather than downstream physical state: some tasks disable
        their only seed-dependent *physical* randomization by default,
        so qpos alone wouldn't prove anything
        either way for those. The property that actually matters here is
        that ``self._rng`` itself is seeded from the passed-in seed.
        """
        task = _make_task(task_id)
        try:
            task.reset(seed=7)
            draw7 = task._rng.uniform()
            task.reset(seed=9)
            draw9 = task._rng.uniform()
            assert draw7 != draw9, (
                f"{task_id}: reset(seed=7) and reset(seed=9) left "
                f"self._rng in the same state — it's no longer being "
                f"reseeded from the passed-in seed at all."
            )
        finally:
            task.close()


@pytest.mark.slow
@pytest.mark.parametrize("task_id", list_tasks())
def test_unseeded_reset_continues_the_task_rng(task_id):
    """reset() without a seed must not re-seed ``_rng``, or every episode repeats.

    The stream is advanced by hand between resets: a task that keeps its
    generator carries the extra draw into the next reset's state, one that
    re-seeds from ``self.seed`` lands on the same state as the seeded reset.
    """
    task = make(task_id, obs_mode="proprio", seed=3, max_episode_steps=10)
    try:
        task.reset(seed=3)
        rng = getattr(task, "_rng", None)
        if rng is None:
            pytest.skip("task has no task-level RNG")
        seeded = rng.get_state()
        rng.random_sample()
        task.reset()
        after = task._rng.get_state()
        assert not (np.array_equal(seeded[1], after[1]) and seeded[2] == after[2])
    finally:
        task.close()
