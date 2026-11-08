"""``privileged_obs=False`` must strip simulator ground truth from every observation.

Several tasks hide their decision variable on purpose (the red cube under a lid
in occluder-push, the occluded cuboid in sheltered-grasp). By default
(``privileged_obs=False``) a task returns only proprioception, tactile, cameras
and force/torque; the scripted oracles and demo collectors pass
``privileged_obs=True`` to also get ``object_state`` / object poses.
"""

import re

import numpy as np
import pytest

from imbench.registry import list_tasks, make

POLICY_KEY = re.compile(r"robot\d+_\w+|rgb_\w+|depth_\w+|force_torque(_world)?")

# One task per way of building observations: single-arm base, the four task
# resets that rebuild obs themselves (edge-slide, packing, plate-shelf-hobi),
# hetbi, and tasks that add their own ground-truth keys (balance, keyboard).
CANONICAL = [
    "ib-p5-occluder-push-v1",
    "ib-p1-sheltered-grasp-v1",
    "ib-p3-domino-select-v1",
    "ib-p1-edge-slide-v1",
    "ib-p7-packing-v1",
    "ib-p1-plate-shelf-hobi-v1",
    "ib-p1-cup-inversion-v1",
    "ib-p7-balance-medium-v1",
    "ib-m-keyboard-typing-v1",
]


def _check_policy_obs(task_id):
    task = make(task_id, obs_mode="proprio", privileged_obs=False, max_episode_steps=5)
    try:
        obs, _ = task.reset(seed=0)
        step_obs, *_ = task.step(np.zeros(task.action_space.shape))
        for o in (obs, step_obs):
            leaked = [k for k in o if not POLICY_KEY.fullmatch(k)]
            assert not leaked, f"{task_id} leaks {leaked}"
            assert any(k.startswith("robot0_") for k in o), f"{task_id} lost proprio"
        assert set(task.observation_space.spaces) == set(step_obs)
    finally:
        task.close()


@pytest.mark.parametrize("task_id", CANONICAL)
def test_policy_obs_has_no_ground_truth(task_id):
    _check_policy_obs(task_id)


@pytest.mark.slow
@pytest.mark.parametrize("task_id", list_tasks())
def test_policy_obs_has_no_ground_truth_all_tasks(task_id):
    _check_policy_obs(task_id)


def test_default_strips_ground_truth():
    task = make("ib-p5-occluder-push-v1", obs_mode="proprio", max_episode_steps=5)
    try:
        obs, _ = task.reset(seed=0)
        assert not {"object_state", "object_pos", "object_quat"} & set(obs)
    finally:
        task.close()


def test_gymnasium_make_strips_ground_truth():
    import gymnasium

    env = gymnasium.make("ib-p5-occluder-push-v1", obs_mode="proprio", disable_env_checker=True)
    try:
        obs, _ = env.reset(seed=0)
        assert not {"object_state", "object_pos", "object_quat"} & set(obs)
    finally:
        env.close()


def test_privileged_obs_keeps_ground_truth():
    task = make("ib-p5-occluder-push-v1", obs_mode="proprio", privileged_obs=True,
                max_episode_steps=5)
    try:
        obs, _ = task.reset(seed=0)
        assert {"object_state", "object_pos"} <= set(obs)
    finally:
        task.close()


def test_policy_obs_keeps_cameras():
    task = make("ib-p5-occluder-push-v1", obs_mode="combo", privileged_obs=False,
                max_episode_steps=5)
    try:
        obs, _ = task.reset(seed=0)
        assert any(k.startswith("rgb_") for k in obs)
        assert "force_torque" in obs
        assert "object_state" not in obs
    finally:
        task.close()
