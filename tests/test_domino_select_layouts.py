"""Physical cascades and seed-composable variations for domino selection."""

import copy

import numpy as np
import pytest
import imbench as ib

from data_synthetic.single_arm_pipeline.domino_select_policy import detect_correct_chain
from imbench.tasks.physics import box_inertia
from tests.test_prehensile_regressions import _action, _finger_push, _step


@pytest.fixture(scope="module", params=[1, 2])
def task(request):
    task = ib.make(
        f"ib-p3-domino-select-v{request.param}", obs_mode="proprio",
        privileged_obs=True, max_episode_steps=400,
        config_overrides={"task": {"interconnected": False, "num_chains": 5}},
    )
    yield task
    task.close()


# Every variation subset, plus straight/curved/bent successes.
@pytest.mark.parametrize("seed", [0, 2, 3, 4, 15, 18, 24, 27, 33])
def test_sampled_correct_chain_physically_reaches_target(task, seed):
    task.reset(seed=seed)
    assert detect_correct_chain(task.sim) == task._correct_chain
    name = f"chain{task._correct_chain}_domino_0"
    try:
        (_, reward, terminated, _, info), ever = _step(
            task, 300, before_step=_finger_push(task, [name]))
        assert ever and terminated and info["is_success"]
        assert reward == info["task_score"] == 1.0
        assert task._target_displaced()
        assert (task._link_touch_step > 0).all()
        assert (np.diff(task._link_touch_step) >= 0).all()
        assert not task._decoy_toppled and not task._robot_touched_chain
    finally:
        task.sim.data.xfrc_applied[:] = 0


@pytest.mark.parametrize("seed", [2, 3, 7])
def test_decoys_really_stop_at_the_interruption(task, seed):
    task.reset(seed=seed)
    decoys = [k for k in range(task._num_chains) if k != task._correct_chain]
    for k in decoys:
        task.reset(seed=seed)
        spec = task._chain_specs[k]
        push = _finger_push(task, [f"chain{k}_domino_0"])
        failed = False
        try:
            # Deliberately continue physics beyond the task's early failure:
            # the break itself, rather than its decoy label, must stop the wave.
            for step in range(200):
                push(step)
                _, reward, terminated, _, info = task.step(_action(task))
                failed |= terminated
                assert reward == 0.0 and not info["is_success"]
            assert failed and task._decoy_toppled
            assert task._domino_toppled(k, 0), (seed, k, spec)
            assert not task._domino_toppled(k, task._dominoes_per_chain - 1), (seed, k, spec)
        finally:
            task.sim.data.xfrc_applied[:] = 0


def test_seed_controls_subsets_and_episode_stays_fixed(task):
    subsets, winners, successful_shapes, failed_shapes = set(), set(), set(), set()
    for seed in range(34):
        task.reset(seed=seed)
        subsets.add(tuple(task._episode_variations.values()))
        winners.add(task._correct_chain)
        successful_shapes.add(task._chain_specs[task._correct_chain]["pattern"])
        failed_shapes.update(s["pattern"] for k, s in enumerate(task._chain_specs)
                             if k != task._correct_chain)
    assert len(subsets) == 8
    assert winners == set(range(task._num_chains))
    assert successful_shapes == failed_shapes == {"line", "arc", "bend"}
    before = copy.deepcopy((task._episode_variations, task._chain_specs))
    sizes = task.sim.model.geom_size.copy()
    _step(task, 10)
    assert before == (task._episode_variations, task._chain_specs)
    np.testing.assert_array_equal(sizes, task.sim.model.geom_size)


@pytest.mark.parametrize("seed", [2, 24, 27])
def test_reset_is_reproducible_including_sizes_and_inertia(task, seed):
    task.reset(seed=seed)
    before = [task.sim.data.qpos.copy(), task.sim.model.geom_size.copy(),
              task.sim.model.body_mass.copy(), task.sim.model.body_inertia.copy()]
    specs = copy.deepcopy(task._chain_specs)
    task.reset(seed=seed + 1)
    task.reset(seed=seed)
    after = [task.sim.data.qpos, task.sim.model.geom_size,
             task.sim.model.body_mass, task.sim.model.body_inertia]
    for expected, actual in zip(before, after):
        np.testing.assert_array_equal(expected, actual)
    assert task._chain_specs == specs
    for k in range(task._num_chains):
        for i in range(task._dominoes_per_chain):
            body = task.sim.model.body_name2id(f"chain{k}_domino_{i}_main")
            half_sizes = task._chain_sizes[k, i][[2, 1, 0]]
            mass = 800 * 8 * np.prod(half_sizes)
            assert task.sim.model.body_mass[body] == pytest.approx(mass)
            np.testing.assert_allclose(task.sim.model.body_inertia[body], box_inertia(mass, half_sizes))


def test_correct_label_without_physical_chain_never_succeeds(task):
    task.reset(seed=2)
    k = task._correct_chain
    for i in (2, 3, 4):
        task._place_object(f"chain{k}_domino_{i}", [100, 100, 100])
    task.sim.forward()
    try:
        (_, reward, _, _, info), ever = _step(
            task, 200, before_step=_finger_push(task, [f"chain{k}_domino_0"]))
        assert task._first_domino_toppled
        assert not ever and not info["is_success"] and reward == 0.0
        assert not task._target_displaced()
    finally:
        task.sim.data.xfrc_applied[:] = 0


def test_short_decoy_has_no_total_length_cue(task):
    task.reset(seed=0)
    correct = task._chain_specs[task._correct_chain]
    short = next(s for s in task._chain_specs if s["defect"] == "short")
    def length(spec):
        return np.linalg.norm(np.diff(np.array(spec["layout"])[:, :2], axis=0), axis=1).sum()
    assert correct["pattern"] == short["pattern"] == "line"
    assert length(correct) == pytest.approx(length(short))
