"""Reflect the complete sheltered layout, keeping its crossing arrangement."""
import numpy as np
import pytest

import imbench as ib
from imbench.tasks.prehensile.contacts import object_geom_ids

TASK = "ib-p1-sheltered-grasp-v1"


@pytest.mark.parametrize("story", ["bottom", "top"])
def test_rack_cube_and_tray_are_geometric_reflections(story):
    snapshots = []
    for side in ("left", "right"):
        task = ib.make(TASK, obs_mode="proprio", config_overrides={"randomization": {
            "rack_side": {"choices": [side]},
            "rack_orientation": {"enabled": True},
            "cube_story": {"enabled": False, "default": story}}})
        try:
            task.reset(seed=7)
            sim = task._env.sim
            positions = {name: sim.data.body_xpos[sim.model.body_name2id(name+"_main")].copy()
                         for name in ("cube", "rack", "tray")}
            positions["walls"] = np.array([sim.data.geom_xpos[g].copy()
                                             for g in sorted(object_geom_ids(task._env, "rack"))])
            snapshots.append(positions)
            assert task._rack_opening_side == side
            assert task._cube_story == story
            assert task._rack_pos[1] * task._tray_pos[1] < 0
            assert task._rack_containment_margin(positions["cube"][:2]) > 0
            # Opening's outward normal points back across the robot's
            # centreline; reflecting must not expose a solid back panel.
            outward = np.array([np.sin(task._rack_yaw), -np.cos(task._rack_yaw)])
            assert np.dot(outward, [0, -task._rack_pos[1]]) > 0
        finally:
            task.close()
    left, right = snapshots
    reflection = np.array([1, -1, 1])
    for name in ("cube", "rack", "tray"):
        np.testing.assert_allclose(right[name], left[name]*reflection, atol=1e-9)
    def rows(a):
        return sorted(map(tuple, np.round(a, 8)))
    assert rows(right["walls"]) == rows(left["walls"]*reflection)


def test_seed_stream_and_episode_layout_are_stable():
    task = ib.make(TASK, obs_mode="proprio")
    try:
        sequences = []
        for _ in range(2):
            sequence = []
            for episode in range(8):
                task.reset(seed=0 if episode == 0 else None)
                side = task._rack_opening_side
                rack, tray, yaw = task._rack_pos.copy(), task._tray_pos.copy(), task._rack_yaw
                sequence.append((side, rack.tolist(), tray.tolist(), yaw, str(task._cube_story)))
                task._rebuild_obs()
                _ = task.observation_space
                for _ in range(5):
                    task.step(np.array([0, 0, 0, 0, 0, 0, -1.]))
                assert task._rack_opening_side == side
                np.testing.assert_array_equal(task._rack_pos, rack)
                np.testing.assert_array_equal(task._tray_pos, tray)
                assert task._rack_yaw == yaw
            sequences.append(sequence)
        assert sequences[0] == sequences[1]
        assert {r[0] for r in sequences[0]} == {"left", "right"}
    finally:
        task.close()
