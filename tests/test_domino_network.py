"""Real cross-chain propagation, failed straight entries and causal scoring."""
import copy
import numpy as np
import pytest
import imbench as ib
from data_synthetic.single_arm_pipeline.domino_select_policy import detect_correct_chain
from tests.test_prehensile_regressions import _finger_push, _action

# v1 draws randomized layouts (tests/test_domino_patterns.py); v2 is the merge.
MERGE = {"layout": "merge", "num_chains": 5}


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("defect", ["gap", "missing", "short"])
def test_every_candidate_start_has_the_expected_physical_outcome(variant, defect):
    task = ib.make(f"ib-p3-domino-select-v{variant}", obs_mode="proprio", max_episode_steps=500,
        config_overrides={"task": MERGE, "randomization": {"interconnections": {"defects": [defect]},
                                                           "layout_variations": {"interruption_probability": 1}}})
    try:
        for starter in range(task._num_chains):
            task.reset(seed=0)
            assert detect_correct_chain(task.sim, interconnected=True) == task._correct_chain
            assert len({name.split("_domino_")[0] for name in task._solution_route}) >= 3
            push = _finger_push(task, [f"chain{starter}_domino_0"])
            ever = False
            for step in range(350):
                push(step)
                _, reward, _, _, info = task.step(_action(task))
                ever |= info["is_success"]
            # Continue well past a decoy's scorer failure: its physical
            # cascade must actually stop, rather than merely lose credit.
            correct = starter == task._correct_chain
            assert bool(ever) == correct, (variant, defect, starter, task._link_touch_step)
            assert bool(task._target_displaced()) == correct
            if correct:
                assert (task._link_touch_step > 0).all()
                assert (np.diff(task._link_touch_step) >= 0).all()
                assert not task._decoy_toppled
    finally:
        task.sim.data.xfrc_applied[:] = 0
        task.close()


def test_network_is_seeded_and_fixed_within_the_episode():
    t = ib.make("ib-p3-domino-select-v1", obs_mode="proprio", config_overrides={"task": MERGE})
    try:
        samples=[]
        for seed in (15,24,27,15):
            t.reset(seed=seed)
            snapshot=copy.deepcopy((t._network_layout,t._solution_route,t._episode_variations))
            samples.append(snapshot)
            for _ in range(5):t.step(_action(t))
            assert snapshot == (t._network_layout,t._solution_route,t._episode_variations)
        assert samples[0] == samples[-1]
        assert len({tuple(s[2].values()) for s in samples[:-1]}) > 1
    finally:t.close()


def test_broken_relay_and_sweeping_starters_cannot_earn_success():
    t=ib.make("ib-p3-domino-select-v1",obs_mode="proprio",max_episode_steps=500,config_overrides={"task":MERGE})
    try:
        for sweep in (False,True):
            t.reset(seed=0)
            if not sweep:
                for name in t._solution_route[5:8]:
                    t._place_object(name,[100,100,100])
                t.sim.forward()
            names=[f"chain{k}_domino_0" for k in range(t._num_chains)] if sweep else [t._solution_route[0]]
            push=_finger_push(t,names)
            for step in range(350):
                push(step);_,reward,_,_,info=t.step(_action(t))
                assert not info["is_success"] and reward==0
            if not sweep:assert not t._target_displaced()
    finally:
        t.sim.data.xfrc_applied[:]=0;t.close()
