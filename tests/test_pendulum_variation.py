"""Seeded pendulum geometry, reachable swing envelopes and actual dynamics."""

import numpy as np
import pytest
from scipy.special import ellipk

from tests.conftest import cached_task
from tests.test_prehensile_regressions import _action, _make, _set_hinge


PROFILES = {
    "mild": (0.80, 0.90, 0.015, 12),
    "medium": (0.65, 0.78, 0.018, 24),
    "extreme": (0.50, 0.65, 0.022, 35),
    "long_wide": (0.80, 0.95, 0.022, 21),
}


def profile_overrides(name):
    length, height, radius, amplitude = PROFILES[name]
    return {
        "randomization": {"pendulum": {
            "pendulum_length": [length, length],
            "frame_height": [height, height],
            "bob_size": [radius, radius],
        }},
        "task": {"swing_amplitude_range": [amplitude, amplitude]},
    }


@pytest.fixture(params=[1, 2])
def task(request):
    return cached_task(f"ib-p2-pendulum-grasp-v{request.param}", obs_mode="proprio",
                       privileged_obs=True, max_episode_steps=600)


def test_seeded_geometry_and_release_are_compiled_and_reproducible(task):
    samples = []
    for seed in range(16):
        task.reset(seed=seed)
        sim = task.sim
        rod = sim.model.geom_name2id("pendulum_rod")
        bob = sim.model.geom_name2id("pendulum_bob")
        jid = sim.model.joint_name2id("pendulum_hinge")
        angle = sim.data.qpos[sim.model.jnt_qposadr[jid]]
        assert sim.model.geom_size[rod, 1] * 2 == pytest.approx(task._arm_length)
        assert sim.model.geom_size[bob, 0] == task._bob_size
        assert sim.model.geom_pos[bob, 2] == pytest.approx(-task._arm_length - task._bob_size)
        assert angle == pytest.approx(task._release_side * task._swing_amplitude)
        assert sim.data.qvel[sim.model.jnt_dofadr[jid]] == 0
        assert (task._arm_length + task._bob_size) * np.sin(abs(angle)) <= task._max_swing_offset
        snapshot = np.r_[task._arm_length, task._stand_height, task._bob_size,
                         task._swing_amplitude, task._release_side, task._get_bob_pos()]
        samples.append(snapshot)
        task.reset(seed=seed)
        repeated = np.r_[task._arm_length, task._stand_height, task._bob_size,
                         task._swing_amplitude, task._release_side, task._get_bob_pos()]
        np.testing.assert_array_equal(snapshot, repeated)
        _set_hinge(task, "pendulum_hinge", 0)
        clearance = task._get_bob_pos()[2] - task._bob_size - task._table_height
        assert task.BOB_CLEARANCE_RANGE[0] <= clearance <= task.BOB_CLEARANCE_RANGE[1]
    samples = np.array(samples)
    assert np.ptp(samples[:, 0]) > 0.15
    assert np.ptp(np.degrees(samples[:, 3])) > 10
    assert set(samples[:, 4]) == {-1, 1}


@pytest.mark.parametrize("variant", [1, 2])
@pytest.mark.parametrize("profile", PROFILES)
def test_motion_has_the_correct_period_energy_and_apex(variant, profile):
    task = _make(f"ib-p2-pendulum-grasp-v{variant}", max_episode_steps=600,
                 config_overrides=profile_overrides(profile))
    try:
        task.reset(seed=0)
        sim = task.sim
        jid = sim.model.joint_name2id("pendulum_hinge")
        qaddr, vaddr = sim.model.jnt_qposadr[jid], sim.model.jnt_dofadr[jid]
        body = sim.model.jnt_bodyid[jid]
        mass = sim.model.body_mass[body]
        radius = task._arm_length + task._bob_size
        inertia = mass * radius**2 + sim.model.body_inertia[body, 0]
        amplitude = task._swing_amplitude
        expected_period = 4 * np.sqrt(inertia / (mass * 9.81 * radius)) * ellipk(np.sin(amplitude / 2)**2)
        params = (task._arm_length, task._stand_height, task._bob_size,
                  task._swing_amplitude, task._release_side)
        energies, crossings, heights, angles = [], [], [], []
        previous = float(sim.data.qpos[qaddr])
        for step in range(500):
            _, reward, terminated, _, info = task.step(_action(task))
            theta, velocity = float(sim.data.qpos[qaddr]), float(sim.data.qvel[vaddr])
            energy = mass * 9.81 * radius * (1 - np.cos(theta)) + 0.5 * inertia * velocity**2
            energies.append(energy)
            angles.append(theta)
            heights.append(task._get_bob_pos()[2])
            if previous < 0 <= theta:
                crossings.append((step + (-previous / (theta - previous))) / task.control_freq)
            previous = theta
            assert reward == 0 and not info["is_success"] and not terminated
            assert not task._robot_touches_bob()
        assert len(crossings) >= 2
        np.testing.assert_allclose(np.diff(crossings), expected_period, rtol=0.01)
        assert np.ptp(energies) / np.mean(energies) < 0.01
        assert min(heights) - task._bob_size >= task._table_height + task.BOB_CLEARANCE_RANGE[0] - 0.001
        assert max(heights) == pytest.approx(task._apex_z, abs=0.002)
        assert max(angles) == pytest.approx(amplitude, abs=0.005)
        assert min(angles) == pytest.approx(-amplitude, abs=0.005)
        assert params == (task._arm_length, task._stand_height, task._bob_size,
                          task._swing_amplitude, task._release_side)
    finally:
        task.close()
