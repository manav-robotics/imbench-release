"""Episode mirror mapping applied to every camera."""
import copy

import numpy as np
import pytest

import imbench as ib
from imbench.tasks.prehensile.mirror_pick_place_v1 import MirrorPickPlaceV1

TASK = "ib-m-mirror-pick-place-v1"


@pytest.mark.parametrize("direction,axes", [
    ("horizontal", (1,)), ("vertical", (0,)), ("both", (0, 1)),
])
def test_rgb_and_depth_mapping_and_teleop_inverse(direction, axes):
    task = object.__new__(MirrorPickPlaceV1)
    task._mirror_direction = direction
    task.camera_cfgs = {
        "front": {"robosuite_name": "frontview"},
        "top": {"robosuite_name": "birdview"},
    }
    rgb = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    depth = np.arange(20, dtype=np.float32).reshape(4, 5)
    original = {"rgb_front": rgb, "depth_front": depth,
                "rgb_top": rgb, "depth_top": depth}
    obs = dict(original)
    task._flip_cameras(obs)
    # No camera is exempt, including one that renders MuJoCo's birdview.
    for label in ("front", "top"):
        np.testing.assert_array_equal(obs["rgb_" + label], np.flip(rgb, axes))
        np.testing.assert_array_equal(obs["depth_" + label], np.flip(depth, axes))
        assert obs["rgb_" + label].flags.c_contiguous
        assert obs["depth_" + label].flags.c_contiguous
    # collect_demos uses a separate dict and this same inverse mapping.
    view = dict(obs)
    task._flip_cameras(view)
    for key in original:
        np.testing.assert_array_equal(view[key], original[key])
    np.testing.assert_array_equal(obs["rgb_front"], np.flip(rgb, axes))


def test_seeded_episode_sequence_and_observation_space_are_stable():
    task = ib.make(TASK, obs_mode="proprio", seed=3)
    try:
        sequences = []
        for _ in range(2):
            sequence = []
            for episode in range(9):
                task.reset(seed=3 if episode == 0 else None)
                direction = task._mirror_direction
                sequence.append(direction)
                _ = task.observation_space
                task._rebuild_obs()
                for _ in range(3):
                    task.step(np.zeros(task.action_space.shape))
                    assert task._mirror_direction == direction
            sequences.append(sequence)
        assert sequences[0] == sequences[1]
        assert set(sequences[0]) == {"horizontal", "vertical", "both"}
    finally:
        task.close()


def test_shipped_config_has_no_bird_eye_camera():
    task = ib.make(TASK, obs_mode="combo", seed=0)
    try:
        assert set(task.camera_cfgs) == {"gripper_cam", "exo_front_cam"}
        assert all(cfg["robosuite_name"] != "birdview" for cfg in task.camera_cfgs.values())
        obs, _ = task.reset(seed=0)
        assert not any("bird" in key for key in obs)
    finally:
        task.close()


def test_real_cameras_cache_and_render_are_all_mirrored():
    cameras = {name: {"width": 120, "height": 90, "depth": True}
               for name in ("gripper_cam", "exo_front_cam")}
    task = ib.make(TASK, obs_mode="combo", render_mode="rgb_array", seed=0,
                   privileged_obs=True, config_overrides={"obs": {"cameras": cameras}})
    try:
        seen = set()
        for seed in (0, 2, 3):
            obs, _ = task.reset(seed=seed)
            direction = task._mirror_direction
            seen.add(direction)
            raw_cache = copy.deepcopy(task._camera_cache)
            sim = task._env.sim
            axes = {"horizontal": (1,), "vertical": (0,), "both": (0, 1)}[direction]
            for label in task.camera_cfgs:
                for prefix in ("rgb_", "depth_"):
                    key = prefix + label
                    np.testing.assert_array_equal(obs[key], np.flip(raw_cache[key], axes))
                    assert obs[key].flags.c_contiguous
            # Repeated forced observations and render agree with reset,
            # without alternating the cache between mirrored and raw.
            for _ in range(2):
                rebuilt = task._rebuild_obs()
                for key in raw_cache:
                    np.testing.assert_array_equal(rebuilt[key], obs[key])
                    np.testing.assert_array_equal(task._camera_cache[key], raw_cache[key])
                np.testing.assert_array_equal(task.render(), obs["rgb_gripper_cam"])
                assert task._mirror_direction == direction
            np.testing.assert_allclose(obs["object_pos"], sim.data.body_xpos[sim.model.body_name2id("cube_main")])
            # The next step is throttled: exactly the same cached image.
            stepped, *_ = task.step(np.zeros(task.action_space.shape))
            for key in raw_cache:
                np.testing.assert_array_equal(stepped[key], obs[key])
            assert task.observation_space.contains(stepped)
            assert task._mirror_direction == direction
        assert seen == {"horizontal", "vertical", "both"}
    finally:
        task.close()
