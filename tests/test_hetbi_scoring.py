"""Tests for HetBI task scoring, termination, and domain randomization.

Covers:
- imbench.tasks.hetbi.* (_compute_task_score, _check_termination)
- Visual/color randomization, payload fall-off detection, suction step cycles
"""

import numpy as np
import pytest

from imbench.registry import TASK_REGISTRY, make


def _step_until_done(task, max_steps=60):
    """Step zero actions until the episode terminates; return the last step tuple."""
    zero = np.zeros(task.action_space.shape)
    for _ in range(max_steps):
        step = task.step(zero)
        if step[2]:
            break
    return step


# =====================================================================
#  HetBI task lifecycle
# =====================================================================

class TestHetbiTasks:

    def _hetbi_ids(self):
        import inspect
        from imbench.tasks.hetbi.base_hetbi import HetBiBaseTask
        return sorted(
            tid for tid, factory in TASK_REGISTRY.items()
            if inspect.isclass(
                factory.func if hasattr(factory, "func") else factory
            ) and issubclass(
                factory.func if hasattr(factory, "func") else factory, HetBiBaseTask
            )
        )

    def test_at_least_5_hetbi_tasks(self):
        assert len(self._hetbi_ids()) >= 5

    @pytest.mark.parametrize("task_id", sorted(
        tid for tid in TASK_REGISTRY if "hetbi" in tid or tid in {
            "ib-p1-bowl-table-v1", "ib-p1-cup-inversion-v1",
            "ib-p1-cracker-box-jaw-v1", "ib-p1-cracker-box-suction-v1",
        }
    ))
    def test_reset_and_step(self, task_id):
        task = make(task_id, obs_mode="proprio", render_mode=None, seed=42,
                     max_episode_steps=5)
        try:
            obs, info = task.reset()
            assert isinstance(obs, dict)
            lo, _ = task.action_space.low, task.action_space.high
            task.step(np.zeros(lo.shape[0]))
        finally:
            task.close()


# =====================================================================
#  HetBI scoring & termination paths
# =====================================================================

class TestHetbiScoring:

    @pytest.fixture(scope="class")
    @classmethod
    def hetbi_task(cls):
        task = make("ib-p1-bowl-table-v1", obs_mode="proprio",
                     render_mode=None, seed=42, max_episode_steps=20)
        task.reset()
        yield task
        task.close()

    def test_initial_score(self, hetbi_task):
        hetbi_task.reset()
        raw = hetbi_task._get_sim_observations()
        score = hetbi_task._compute_task_score(raw)
        assert 0.0 <= score <= 1.0

    def test_termination_on_fall(self, hetbi_task):
        """Payload below table should trigger termination."""
        hetbi_task.reset()
        pf = hetbi_task._payload_prefix or "payload_"
        raw = {f"{pf}pos": np.array([0.0, 0.0, 0.01])}
        term = hetbi_task._check_termination(raw)
        assert term

    def test_termination_missing_key(self, hetbi_task):
        raw = {}
        term = hetbi_task._check_termination(raw)
        assert not term

    def test_gripper_contacts_payload_false(self, hetbi_task):
        hetbi_task.reset()
        result = hetbi_task._gripper_contacts_payload()
        assert isinstance(result, bool)

    def test_score_missing_pos_key(self, hetbi_task):
        score = hetbi_task._compute_task_score({})
        assert score == 0.0

    def test_has_suction_weld(self, hetbi_task):
        assert hasattr(hetbi_task, "_suction_weld")

    def test_step_with_suction(self, hetbi_task):
        hetbi_task.reset()
        lo, _ = hetbi_task.action_space.low, hetbi_task.action_space.high
        action = np.zeros(lo.shape[0])
        action[-1] = 1.0
        obs, r, t, tr, info = hetbi_task.step(action)
        assert isinstance(obs, dict)


# =====================================================================
#  HetBI domain randomization paths
# =====================================================================

class TestHetbiRandomization:

    def test_visual_randomization(self):
        """HetBI task with visual randomization enabled should not crash."""
        task = make("ib-p1-bowl-table-v1", obs_mode="proprio",
                     render_mode=None, seed=42, max_episode_steps=10)
        try:
            for _ in range(3):
                obs, info = task.reset()
                assert isinstance(obs, dict)
        finally:
            task.close()

    def test_payload_fell_off_recovery(self):
        """After payload falls, score stays 0 and task terminates."""
        task = make("ib-p1-bowl-table-v1", obs_mode="proprio",
                     render_mode=None, seed=42, max_episode_steps=10)
        try:
            task.reset()
            pf = task._payload_prefix or "payload_"
            raw = {f"{pf}pos": np.array([0.0, 0.0, 0.01])}
            score = task._compute_task_score(raw)
            assert score == 0.0
            assert task._payload_fell_off

            raw2 = {f"{pf}pos": np.array([0.0, 0.0, 0.9])}
            score2 = task._compute_task_score(raw2)
            assert score2 == 0.0

            term = task._check_termination(raw)
            assert term
        finally:
            task.close()

    def test_random_object_color(self):
        task = make("ib-p1-bowl-table-v1", obs_mode="proprio",
                     render_mode=None, seed=42, max_episode_steps=5)
        try:
            task.reset()
            c = task._random_object_color()
            assert c.shape == (4,)
            assert c[3] == 1.0
        finally:
            task.close()

    def test_random_table_color(self):
        task = make("ib-p1-bowl-table-v1", obs_mode="proprio",
                     render_mode=None, seed=42, max_episode_steps=5)
        try:
            task.reset()
            c = task._random_table_color()
            assert c.shape == (4,)
        finally:
            task.close()


# =====================================================================
class TestBowlTableScoring:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = make("ib-p1-bowl-table-v1", obs_mode="proprio",
                 render_mode=None, seed=42, max_episode_steps=50)
        t.reset()
        yield t
        t.close()

    # ── helpers ──────────────────────────────────────────────────────

    def _prefix(self, task):
        return task._payload_prefix or "payload_"

    def _table_pos(self, task):
        """A position on the table, well outside the rack."""
        return np.array([0.1, -0.15, task._table_height + 0.02])

    def _rest_pos(self, task):
        """Upright bowl resting on the table outside the rack."""
        z = task._table_height + abs(task._env._payload.bottom_offset[2]) + 0.002
        return np.array([0.1, -0.15, z])

    def _rack_centre_pos(self, task):
        fb = task._env._fixture_bounds
        return np.array([
            (fb["x"][0] + fb["x"][1]) / 2,
            (fb["y"][0] + fb["y"][1]) / 2,
            (fb["z"][0] + fb["z"][1]) / 2,
        ])

    def _upright_quat_xyzw(self):
        return np.array([0.0, 0.0, 0.0, 1.0])

    def _inverted_quat_xyzw(self):
        return np.array([1.0, 0.0, 0.0, 0.0])

    def _tilted_quat_xyzw(self):
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        return np.array([s, 0.0, 0.0, c])

    def _teleport_payload(self, task, pos, quat_wxyz, zero_vel=True):
        """Teleport the payload to *pos* with orientation *quat_wxyz*."""
        sim = task._env.sim
        payload = task._env._payload
        jn = payload.joints[0]
        sim.data.set_joint_qpos(jn, np.concatenate([pos, quat_wxyz]))
        if zero_vel:
            vel_addr = sim.model.get_joint_qvel_addr(jn)
            if isinstance(vel_addr, tuple):
                vel_addr = vel_addr[0]
            sim.data.qvel[vel_addr:vel_addr + 6] = 0.0
        sim.forward()

    # ── score = 0 cases ─────────────────────────────────────────────

    def test_score_zero_missing_pos(self, task):
        task.reset()
        assert task._compute_task_score({}) == 0.0

    def test_score_zero_still_in_rack(self, task):
        """Bowl inside the rack must not register as complete (it must be
        on the table). Other milestones (e.g. upright) can still contribute
        partial credit, so this checks "not success" rather than exactly 0.0."""
        task.reset()
        pf = self._prefix(task)
        pos = self._rack_centre_pos(task)
        raw = {
            f"{pf}pos": pos,
            f"{pf}quat": self._upright_quat_xyzw(),
        }
        assert task._compute_task_score(raw) < 1.0

    def test_score_zero_inverted_on_table(self, task):
        """Bowl on the table but upside-down must not register as complete.
        Partial credit from other milestones is expected under the staged
        reward, so this checks "not success" rather than an exact 0.0."""
        task.reset()
        pos = self._table_pos(task)
        self._teleport_payload(task, pos, np.array([0.0, 1.0, 0.0, 0.0]))
        raw = task._get_sim_observations(force_update=True)
        assert task._compute_task_score(raw) < 1.0

    def test_score_zero_tilted_on_table(self, task):
        """Bowl on the table but on its side must not register as complete
        (see test_score_zero_inverted_on_table for why this is now a
        "not success" bound rather than an exact 0.0)."""
        task.reset()
        pos = self._table_pos(task)
        # 90° around X in wxyz = [cos45, sin45, 0, 0]
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        self._teleport_payload(task, pos, np.array([c, s, 0.0, 0.0]))
        raw = task._get_sim_observations(force_update=True)
        assert task._compute_task_score(raw) < 1.0

    def test_score_zero_above_table(self, task):
        """Bowl floating high above the table must not register as complete
        (see test_score_zero_inverted_on_table for why this is now a
        "not success" bound rather than an exact 0.0)."""
        task.reset()
        pf = self._prefix(task)
        raw = {
            f"{pf}pos": np.array([0.1, 0.0, task._table_height + 0.25]),
            f"{pf}quat": self._upright_quat_xyzw(),
        }
        assert task._compute_task_score(raw) < 1.0

    def test_upright_in_rack_not_success(self, task):
        """Bowl upright at table height but still inside the rack footprint."""
        task.reset()
        pf = self._prefix(task)
        rack = self._rack_centre_pos(task)
        raw = {
            f"{pf}pos": np.array([rack[0], rack[1], task._table_height + 0.02]),
            f"{pf}quat": self._upright_quat_xyzw(),
        }
        assert task._compute_task_score(raw) < 1.0
        assert task._stage_debug["_success"] is False

    # ── score = 1 case ────────────────────────────────────────────

    def test_score_one_upright_on_table(self, task):
        """Bowl resting upright on the table outside the rack → success, score 1."""
        task.reset()
        self._teleport_payload(task, self._rest_pos(task), np.array([1.0, 0.0, 0.0, 0.0]))
        _, _, _, _, info = _step_until_done(task)
        assert info["is_success"] is True
        assert info["task_score"] == 1.0

    def test_upright_on_table_while_touched_not_success(self, task, monkeypatch):
        """A correctly posed bowl still touched by a gripper is not placed."""
        task.reset()
        monkeypatch.setattr(task, "_gripper_contacts_payload", lambda: True)
        self._teleport_payload(task, self._rest_pos(task), np.array([1.0, 0.0, 0.0, 0.0]))
        _, _, terminated, _, info = _step_until_done(task, max_steps=20)
        assert terminated is False
        assert info["is_success"] is False

    def test_taxel_only_contact_counts_as_gripper_contact(self, task):
        """A payload touched only by jaw taxels is touched by that gripper, not released."""
        task.reset()
        sim = task._env.sim
        model = sim.model
        payload_id = task._env._payload_body_id
        payload_geoms = {
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] == payload_id and model.geom_contype[g]
        }
        named = [g for g in range(model.ngeom)
                 if (model.geom_id2name(g) or "").startswith("gripper")]
        saved = model.geom_contype[named].copy(), model.geom_conaffinity[named].copy()
        try:
            # Only the taxels can collide; centre a payload hull on one of them.
            model.geom_contype[named] = 0
            model.geom_conaffinity[named] = 0
            taxel = model.geom_name2id("robot0_taxel_left_0_0")
            hull = min(payload_geoms)
            hull_offset = sim.data.geom_xpos[hull] - sim.data.body_xpos[payload_id]
            _teleport(task, sim.data.geom_xpos[taxel] - hull_offset,
                      np.array(sim.data.body_xquat[payload_id]))
            touching = {
                int(c.geom1) if int(c.geom2) in payload_geoms else int(c.geom2)
                for c in sim.data.contact[:sim.data.ncon]
                if int(c.geom1) in payload_geoms or int(c.geom2) in payload_geoms
            }
            assert any("taxel" in model.geom_id2name(g) for g in touching)
            assert task._gripper_contacts_payload_idx(0) is True
            assert task._gripper_contacts_payload_idx(1) is False
            assert task._payload_released() is False
        finally:
            model.geom_contype[named], model.geom_conaffinity[named] = saved

    # ── fall-off / termination ──────────────────────────────────────

    def test_fall_terminates(self, task):
        task.reset()
        pf = self._prefix(task)
        raw = {f"{pf}pos": np.array([0.0, 0.0, 0.01])}
        assert task._check_termination(raw) is True
        assert task._payload_fell_off is True

    def test_fall_sets_flag(self, task):
        """After fall, _payload_fell_off flag is set."""
        task.reset()
        pf = self._prefix(task)
        # Trigger fall
        task._compute_task_score({f"{pf}pos": np.array([0.0, 0.0, 0.01])})
        assert task._payload_fell_off

    def test_no_termination_missing_key(self, task):
        task.reset()
        assert task._check_termination({}) is False

    # ── success termination ─────────────────────────────────────────

    def test_success_terminates(self, task):
        """Bowl resting upright on the table should trigger termination."""
        task.reset()
        self._teleport_payload(task, self._rest_pos(task), np.array([1.0, 0.0, 0.0, 0.0]))
        _, _, terminated, _, _ = _step_until_done(task)
        assert terminated is True

    def test_upright_on_table_needs_rest(self, task):
        """A single frame of the success pose is not success: the bowl must be at rest."""
        task.reset()
        pf = self._prefix(task)
        raw = {
            f"{pf}pos": self._table_pos(task),
            f"{pf}quat": self._upright_quat_xyzw(),
        }
        assert task._compute_task_score(raw) < 1.0
        assert task._check_termination(raw) is False


# =====================================================================
#  Shared helpers for fixture- and table-target tasks
# =====================================================================

def _teleport(task, pos, quat_wxyz, zero_vel=True):
    """Teleport the payload to *pos* with orientation *quat_wxyz*."""
    sim = task._env.sim
    payload = task._env._payload
    jn = payload.joints[0]
    sim.data.set_joint_qpos(jn, np.concatenate([pos, quat_wxyz]))
    if zero_vel:
        vel_addr = sim.model.get_joint_qvel_addr(jn)
        if isinstance(vel_addr, tuple):
            vel_addr = vel_addr[0]
        sim.data.qvel[vel_addr:vel_addr + 6] = 0.0
    sim.forward()


def _fixture_centre(task):
    fb = task._env._fixture_bounds
    return np.array([
        (fb["x"][0] + fb["x"][1]) / 2,
        (fb["y"][0] + fb["y"][1]) / 2,
        (fb["z"][0] + fb["z"][1]) / 2,
    ])


def _table_pos(task):
    return np.array([0.1, 0.0, task._table_height + 0.02])


# =====================================================================
#  Plate-Shelf scoring & termination
# =====================================================================

class TestPlateShelfScoring:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = make("ib-p1-plate-shelf-hetbi-v1", obs_mode="proprio",
                 render_mode=None, seed=42, max_episode_steps=200)
        t.reset()
        yield t
        t.close()

    def test_score_zero_missing_pos(self, task):
        task.reset()
        assert task._compute_task_score({}) == 0.0

    def test_score_zero_on_table(self, task):
        """Plate on the table (outside shelf) must not register as complete.
        Partial credit from other milestones (e.g. lift) is expected under
        the staged reward, so this checks "not success" rather than
        exactly 0.0."""
        task.reset()
        pf = task._payload_prefix or "payload_"
        raw = {f"{pf}pos": _table_pos(task)}
        assert task._compute_task_score(raw) < 1.0

    def test_score_zero_flat_in_shelf(self, task):
        """Plate flat (identity quat) in shelf bounds must NOT score 1.

        A position-only check would count a plate lying flat inside the
        shelf footprint as a success; the `inserted_in_bounds` stage also
        requires the plate to be on-edge.
        """
        task.reset()
        pos = _fixture_centre(task)
        _teleport(task, pos, np.array([1.0, 0.0, 0.0, 0.0]))
        raw = task._get_sim_observations(force_update=True)
        assert task._compute_task_score(raw) < 1.0

    def test_score_one_in_shelf_on_edge(self, task, monkeypatch):
        """Plate on-edge inside shelf bounds and at rest → score 1."""
        task.reset()
        task._handoff_complete = True  # placement-only probe after an earned handover
        monkeypatch.setattr(task, "_payload_is_stable_multistep", lambda *a, **k: True)
        pf = task._payload_prefix or "payload_"
        pos = _fixture_centre(task)
        # 90° around X in xyzw → face normal along the narrow centre slot
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        raw = {
            f"{pf}pos": pos,
            f"{pf}quat": np.array([s, 0.0, 0.0, c]),
        }
        assert task._compute_task_score(raw) == 1.0

    def test_in_shelf_on_edge_while_touched_not_success(self, task, monkeypatch):
        """An on-edge plate at rest in the shelf still touched by a gripper is not settled."""
        task.reset()
        task._handoff_complete = True  # placement-only probe after an earned handover
        monkeypatch.setattr(task, "_payload_is_stable_multistep", lambda *a, **k: True)
        monkeypatch.setattr(task, "_gripper_contacts_payload", lambda: True)
        pf = task._payload_prefix or "payload_"
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        raw = {
            f"{pf}pos": _fixture_centre(task),
            f"{pf}quat": np.array([s, 0.0, 0.0, c]),
        }
        assert task._compute_task_score(raw) < 1.0
        assert task._check_termination(raw) is False

    def test_fall_terminates(self, task):
        task.reset()
        pf = task._payload_prefix or "payload_"
        raw = {f"{pf}pos": np.array([0.0, 0.0, 0.01])}
        assert task._check_termination(raw) is True
        assert task._payload_fell_off is True

    def test_no_termination_missing_key(self, task):
        task.reset()
        assert task._check_termination({}) is False

    def test_success_terminates(self, task, monkeypatch):
        task.reset()
        task._handoff_complete = True  # placement-only probe after an earned handover
        monkeypatch.setattr(task, "_payload_is_stable_multistep", lambda *a, **k: True)
        pf = task._payload_prefix or "payload_"
        pos = _fixture_centre(task)
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        raw = {
            f"{pf}pos": pos,
            f"{pf}quat": np.array([s, 0.0, 0.0, c]),
        }
        assert task._check_termination(raw) is True

    def test_in_shelf_on_edge_moving_not_success(self, task):
        """The on-edge in-shelf pose only counts once the plate is at rest."""
        task.reset()
        task._handoff_complete = True  # placement-only probe after an earned handover
        pf = task._payload_prefix or "payload_"
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        raw = {
            f"{pf}pos": _fixture_centre(task),
            f"{pf}quat": np.array([s, 0.0, 0.0, c]),
        }
        assert task._compute_task_score(raw) < 1.0
        assert task._check_termination(raw) is False

    def test_fixture_bounds_are_shelf_specific(self, task):
        """Shelf bounds z-range should be taller than rack (>= 0.20)."""
        task.reset()
        fb = task._env._fixture_bounds
        assert fb is not None
        z_span = fb["z"][1] - fb["z"][0]
        assert z_span >= 0.20, f"Shelf z-span {z_span:.3f} looks too short"


# =====================================================================
#  Cracker-box slot scoring & termination (suction-only task)
# =====================================================================

class TestCrackerBoxScoring:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = make("ib-p1-cracker-box-suction-v1", obs_mode="proprio",
                 render_mode=None, seed=42, max_episode_steps=100)
        t.reset()
        yield t
        t.close()

    def test_score_zero_missing_pos(self, task):
        task.reset()
        assert task._compute_task_score({}) == 0.0

    def test_score_zero_on_table(self, task):
        """Box on the table (outside slot) must not register as complete.
        Partial credit from other milestones (e.g. lift) is expected under
        the staged reward, so this checks "not success" rather than
        exactly 0.0."""
        task.reset()
        pf = task._payload_prefix or "payload_"
        raw = {f"{pf}pos": _table_pos(task)}
        assert task._compute_task_score(raw) < 1.0

    # Box lying flat: thin body x vertical, long body z along the channel.
    _FLAT_WXYZ = np.array([0.7071068, 0.0, -0.7071068, 0.0])
    # Box on its long edge (166 mm side vertical) — protrudes from the slot.
    _ON_EDGE_WXYZ = np.array([0.5, -0.5, -0.5, 0.5])

    @pytest.mark.parametrize("quat_wxyz", [
        np.array([1.0, 0.0, 0.0, 0.0]),   # standing upright
        _ON_EDGE_WXYZ,
    ], ids=["upright", "on_long_edge"])
    def test_wrong_orientation_in_slot_not_success(self, task, quat_wxyz):
        """A box settled at the slot centre in the wrong orientation is not inserted."""
        task.reset()
        _teleport(task, _fixture_centre(task), quat_wxyz)
        _, _, terminated, _, info = _step_until_done(task)
        assert terminated is False
        assert info["is_success"] is False
        assert info["task_score"] < 1.0

    def test_score_one_in_slot(self, task):
        """Box lying flat in the slot, released and at rest → score 1."""
        task.reset()
        _teleport(task, _fixture_centre(task), self._FLAT_WXYZ)
        _, _, _, _, info = _step_until_done(task)
        assert info["is_success"] is True
        assert info["task_score"] == 1.0

    def test_in_slot_while_touched_not_success(self, task, monkeypatch):
        """A correctly posed box still touched by a gripper is not inserted."""
        task.reset()
        monkeypatch.setattr(task, "_gripper_contacts_payload", lambda: True)
        _teleport(task, _fixture_centre(task), self._FLAT_WXYZ)
        _, _, terminated, _, info = _step_until_done(task)
        assert terminated is False
        assert info["is_success"] is False

    def test_fall_terminates(self, task):
        task.reset()
        pf = task._payload_prefix or "payload_"
        raw = {f"{pf}pos": np.array([0.0, 0.0, 0.01])}
        assert task._check_termination(raw) is True

    def test_no_termination_missing_key(self, task):
        task.reset()
        assert task._check_termination({}) is False

    def test_success_terminates(self, task):
        """Box lying flat in the slot should trigger termination."""
        task.reset()
        _teleport(task, _fixture_centre(task), self._FLAT_WXYZ)
        _, _, terminated, _, _ = _step_until_done(task)
        assert terminated is True


# =====================================================================
#  Cup-Inversion scoring & termination
# =====================================================================

class TestCupInversionScoring:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = make("ib-p1-cup-inversion-v1", obs_mode="proprio",
                 render_mode=None, seed=42, max_episode_steps=200)
        t.reset()
        yield t
        t.close()

    def test_score_zero_missing_pos(self, task):
        task.reset()
        assert task._compute_task_score({}) == 0.0

    def test_score_zero_inverted_on_table(self, task):
        """Cup still inverted on the table must not register as complete.
        Partial credit from other milestones (e.g. on_table) is expected
        under the staged reward, so this checks "not success" rather than
        exactly 0.0."""
        task.reset()
        pf = task._payload_prefix or "payload_"
        # 180° around X in xyzw → upside-down
        raw = {
            f"{pf}pos": np.array([0.0, 0.0, task._table_height + 0.02]),
            f"{pf}quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        assert task._compute_task_score(raw) < 1.0

    def test_score_zero_upright_above_table(self, task):
        """Cup upright but floating above the table region must not
        register as complete (see test_score_zero_inverted_on_table for
        why this is now a "not success" bound rather than an exact 0.0)."""
        task.reset()
        pf = task._payload_prefix or "payload_"
        raw = {
            f"{pf}pos": np.array([0.0, 0.0, task._table_height + 0.30]),
            f"{pf}quat": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        assert task._compute_task_score(raw) < 1.0

    def _rest_in_goal(self, task):
        goal = task._goal_pos
        z = task._table_height + abs(task._env._payload.bottom_offset[2]) + 0.002
        return np.array([goal[0], goal[1], z])

    def test_score_one_upright_on_table(self, task):
        """Cup resting upright on the table inside the goal region → score 1."""
        task.reset()
        _teleport(task, self._rest_in_goal(task), np.array([1.0, 0.0, 0.0, 0.0]))
        _, _, _, _, info = _step_until_done(task)
        assert info["is_success"] is True
        assert info["task_score"] == 1.0

    def test_upright_in_goal_while_touched_not_success(self, task, monkeypatch):
        """A correctly posed cup still touched by a gripper is not placed."""
        task.reset()
        monkeypatch.setattr(task, "_gripper_contacts_payload", lambda: True)
        _teleport(task, self._rest_in_goal(task), np.array([1.0, 0.0, 0.0, 0.0]))
        _, _, terminated, _, info = _step_until_done(task)
        assert terminated is False
        assert info["is_success"] is False

    def test_success_terminates(self, task):
        task.reset()
        _teleport(task, self._rest_in_goal(task), np.array([1.0, 0.0, 0.0, 0.0]))
        _, _, terminated, _, _ = _step_until_done(task)
        assert terminated is True

    def test_upright_outside_goal_not_success(self, task):
        task.reset()
        pos = self._rest_in_goal(task)
        pos[0] -= 0.3
        _teleport(task, pos, np.array([1.0, 0.0, 0.0, 0.0]))
        _, _, terminated, _, info = _step_until_done(task)
        assert terminated is False
        assert info["is_success"] is False

    def test_fall_terminates(self, task):
        task.reset()
        pf = task._payload_prefix or "payload_"
        raw = {f"{pf}pos": np.array([0.0, 0.0, 0.01])}
        assert task._check_termination(raw) is True

    def test_no_termination_missing_key(self, task):
        task.reset()
        assert task._check_termination({}) is False

    def test_description_says_starts_inverted(self, task):
        """Config description should state the cup starts inverted."""
        desc = task.DESCRIPTION.lower()
        assert "inverted" in desc or "upside-down" in desc, (
            f"Description should mention cup starts inverted: {task.DESCRIPTION}"
        )


# =====================================================================
#  Orientation at rest (stepped: success also needs a released, still payload)
# =====================================================================

def _settle_payload(task, pos, quat_wxyz, max_steps=100):
    """Teleport the payload to *pos* / *quat_wxyz*, step zero actions; return whether success ever fired."""
    sim = task._env.sim
    joint = task._env._payload.joints[0]
    sim.data.set_joint_qpos(joint, np.concatenate([pos, quat_wxyz]))
    sim.data.set_joint_qvel(joint, np.zeros(6))
    sim.forward()
    zero = np.zeros(task.action_space.shape)
    for _ in range(max_steps):
        *_, terminated, _, info = task.step(zero)
        if info["is_success"]:
            return True
        if terminated:
            break
    return False


@pytest.mark.parametrize("upright", [True, False], ids=["upright", "inverted"])
@pytest.mark.parametrize("task_id", ["ib-p1-cup-inversion-v1", "ib-p1-bowl-table-v1"])
def test_settled_payload_succeeds_only_upright(task_id, upright):
    task = make(task_id, obs_mode="proprio", max_episode_steps=400)
    try:
        task.reset(seed=0)
        if task_id == "ib-p1-cup-inversion-v1":
            pos = np.array([*task._goal_pos[:2], task._table_height + 0.06])
        else:  # on the table, clear of the rack
            pos = np.array([0.1, -0.15, task._table_height + 0.08])
        quat = np.array([1.0, 0, 0, 0]) if upright else np.array([0.0, 1.0, 0, 0])
        assert _settle_payload(task, pos, quat) is upright
    finally:
        task.close()
