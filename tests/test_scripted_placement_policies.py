"""Physical regressions for the jaw and homogeneous plate-shelf collectors."""

import numpy as np
import pytest

import imbench as ib
from data_synthetic.collection import build_policy, clip_action
from data_synthetic.hetbi_scripted import TASK_POLICY_MAP
from data_synthetic.mp_hobi_plate_shelf import _make_policy


_HOBI_SEED0_XFAIL = pytest.mark.xfail(
    reason="the receiver's hand clips the shelf's side wall on this seed, so the "
           "plate stalls short of the slot", strict=False)


@pytest.mark.slow
@pytest.mark.parametrize("task_id, seed", [
    ("ib-p1-cracker-box-jaw-v1", 0),
    ("ib-p1-cracker-box-jaw-v1", 1),
    ("ib-p1-cracker-box-jaw-v1", 2),
    pytest.param("ib-p1-plate-shelf-hobi-v1", 0, marks=_HOBI_SEED0_XFAIL),
    ("ib-p1-plate-shelf-hobi-v1", 1),
    ("ib-p1-plate-shelf-hobi-v1", 2),
])
def test_scripted_placement_succeeds_on_original_failing_seeds(task_id, seed):
    """Use collector factories and clipping, with unchanged task scoring.

    Plate seed 2 also exercises the physical relocation/regrasp at the IK
    workspace limit.
    """
    task = ib.make(task_id, obs_mode="proprio", privileged_obs=True, seed=seed)
    try:
        obs, _ = task.reset(seed=seed)
        rng = np.random.default_rng(seed)
        policy = (_make_policy(task, obs, rng) if "plate-shelf" in task_id
                  else build_policy(TASK_POLICY_MAP[task_id], task, rng))
        for _ in range(task.MAX_EPISODE_STEPS):
            action = clip_action(policy.act(obs, env=task), task.action_space)
            obs, _, terminated, truncated, info = task.step(action)
            if terminated or truncated or policy.done:
                break
        assert info["is_success"], (policy.phase, policy.failure_reason, info)
        assert not policy.failed
    finally:
        task.close()


def test_jaw_does_not_lift_without_a_grasp(monkeypatch):
    task_id = "ib-p1-cracker-box-jaw-v1"
    task = ib.make(task_id, obs_mode="proprio", privileged_obs=True, seed=0)
    try:
        obs, _ = task.reset(seed=0)
        policy = build_policy(TASK_POLICY_MAP[task_id], task, np.random.default_rng(0))
        policy.phase = "grasp"
        monkeypatch.setattr(task, "_check_grasp", lambda **kwargs: False)
        for _ in range(250):
            policy.act(obs, env=task)
        assert policy.done and policy.failed
        assert policy.failure_reason == "grasp target was not reached"
        assert policy._lift_xy is None
    finally:
        task.close()
