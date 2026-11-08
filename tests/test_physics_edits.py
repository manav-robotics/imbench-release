"""Runtime model edits must reach the dynamics MuJoCo integrates."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import imbench as ib
from imbench.tasks.physics import commit_model_edits, set_body_mass

# Tasks that change a free body's mass at reset, and that body.
MASS_EDITS = [
    ("ib-p5-mass-sort-v1", "cube_1_main"),
    ("ib-p3-seesaw-balance-v1", "weight_0_main"),
    ("ib-p1-edge-slide-v1", "cube_main"),
    ("ib-p1-edge-slide-v2", "cube_main"),
    ("ib-p7-balance-medium-v1", "rod_main"),
    ("ib-p7-balance-hard-v2", "rod_main"),
]


def _free_dofs(model, body: int) -> slice:
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    return slice(dof, dof + 6)


@pytest.mark.parametrize("task_id,body_name", MASS_EDITS)
def test_reset_mass_edits_reach_the_mass_matrix(task_id, body_name):
    t = ib.make(task_id, obs_mode="proprio")
    try:
        t.reset(seed=1)
        model = t._env.sim.model
        body = model.body_name2id(body_name)
        m0 = model._model.dof_M0[_free_dofs(model, body)]
        np.testing.assert_allclose(m0[:3], model.body_mass[body])
        np.testing.assert_allclose(m0[3:], model.body_inertia[body])
    finally:
        t.close()


def test_commit_model_edits_keeps_state_and_updates_inertial_mass():
    t = ib.make("ib-p5-mass-sort-v1", obs_mode="proprio")
    try:
        t.reset(seed=0)
        sim = t._env.sim
        z = np.zeros(t.action_space.shape)
        for _ in range(5):
            t.step(z)
        body = sim.model.body_name2id("cube_1_main")
        before = (sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.time)
        set_body_mass(sim, body, 4.0 * sim.model.body_mass[body])
        commit_model_edits(sim)
        np.testing.assert_array_equal(sim.data.qpos, before[0])
        np.testing.assert_array_equal(sim.data.qvel, before[1])
        assert sim.data.time == before[2]
        assert sim.model._model.dof_M0[_free_dofs(sim.model, body)][0] == pytest.approx(sim.model.body_mass[body])
    finally:
        t.close()


def test_multiccd_enabled_after_every_reset():
    t = ib.make("ib-p1-edge-slide-v2", obs_mode="proprio")
    try:
        for seed in (0, 1):
            t.reset(seed=seed)
            assert t._env.sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD
    finally:
        t.close()
