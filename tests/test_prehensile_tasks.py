"""Tests for single-arm prehensile tasks (ib-p1/p2/p3/p4/p5/p6/p7-*) using SingleArmTableEnv.

Uses parametrize to run a common test suite against every prehensile
task, ensuring each one initialises, resets, steps, and produces valid
observations and scores.

Each task is created, tested, and closed within a single parametrized
test to avoid keeping many MuJoCo simulations alive simultaneously.

Also tests programmatic success for tasks that can be driven to
score==1.0 by teleporting objects to goal states.
"""

import numpy as np
import pytest

from imbench.registry import TASK_REGISTRY, make
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.base_task import BaseTask
from imbench.tasks.physics import commit_model_edits, set_body_mass
from imbench.tasks.prehensile.domino_patterns import (
    ALL_PATTERNS,
    ARC_MAX_TURN_PER_LINK,
    generate_pattern,
    heading_to_quat,
    rotate_layout,
    translate_layout,
)


# ── Discover all prehensile task IDs dynamically ────────────────────

def _prehensile_task_ids():
    """Return all registered single-arm prehensile tasks that use SingleArmBase."""
    ids = []
    for tid, cls in TASK_REGISTRY.items():
        actual = cls.func if hasattr(cls, "func") else cls
        try:
            if issubclass(actual, SingleArmBase):
                ids.append(tid)
        except TypeError:
            pass
    return sorted(ids)


PREHENSILE_IDS = _prehensile_task_ids()
# Canonical subset run by default; full matrix gated behind @pytest.mark.slow.
PREHENSILE_SMOKE_IDS = [
    tid for tid in ("ib-p1-edge-slide-v1", "ib-p2-cube-toss-v1") if tid in PREHENSILE_IDS
] or PREHENSILE_IDS[:2]


# ── Registration ────────────────────────────────────────────────────


class TestPrehensileRegistration:

    def test_at_least_12_prehensile_tasks(self):
        assert len(PREHENSILE_IDS) >= 12

    @pytest.mark.parametrize("task_id", PREHENSILE_IDS)
    def test_in_registry(self, task_id):
        assert task_id in TASK_REGISTRY

    @pytest.mark.parametrize("task_id", PREHENSILE_IDS)
    def test_is_single_arm_base(self, task_id):
        cls = TASK_REGISTRY[task_id]
        actual = cls.func if hasattr(cls, "func") else cls
        assert issubclass(actual, SingleArmBase)


# ── Comprehensive per-task tests ────────────────────────────────────
# Each parametrized test creates the env, runs all checks, then closes.


class TestPrehensileTaskFull:
    """Full lifecycle, step, obs, config, and env internals for each task."""

    @pytest.mark.parametrize("task_id", PREHENSILE_SMOKE_IDS)
    def test_smoke_lifecycle(self, task_id):
        """Light lifecycle check on canonical tasks — always runs."""
        task = make(task_id, obs_mode="combo", render_mode=None, seed=0, max_episode_steps=3)
        try:
            obs, info = task.reset()
            assert isinstance(obs, dict) and "task_score" in info
            lo, _ = task.action_space.low, task.action_space.high
            task.step(np.zeros(lo.shape[0]))
        finally:
            task.close()

    @pytest.mark.slow
    @pytest.mark.parametrize("task_id", PREHENSILE_IDS)
    def test_full_lifecycle(self, task_id):
        task = make(task_id, obs_mode="combo", render_mode=None, seed=42, max_episode_steps=10,
                    privileged_obs=True)
        try:
            # ── isinstance checks ───────────────────────────────────
            assert isinstance(task, BaseTask)

            # ── MRO checks ──────────────────────────────────────────
            mro = [c.__name__ for c in type(task._env).__mro__]
            assert "ManipulationEnv" in mro
            assert "Lift" not in mro

            # ── Reset ───────────────────────────────────────────────
            obs, info = task.reset()
            assert isinstance(obs, dict)
            assert isinstance(info, dict)
            assert "task_score" in info
            assert "step" in info
            assert info["step"] == 0

            # ── Proprio keys ────────────────────────────────────────
            for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                assert key in obs, f"Missing proprio key: {key}"
                assert isinstance(obs[key], np.ndarray)

            # ── Object state ────────────────────────────────────────
            assert "object_pos" in obs
            assert obs["object_pos"].shape == (3,)

            # ── Camera obs (combo mode) ─────────────────────────────
            rgb_keys = [k for k in obs if k.startswith("rgb_")]
            assert len(rgb_keys) > 0, "combo mode should have camera obs"
            for k in rgb_keys:
                assert obs[k].ndim == 3, f"{k} should be 3D (H,W,C)"

            # ── Action space ────────────────────────────────────────
            lo, hi = task.action_space.low, task.action_space.high
            assert lo.shape == (7,), "Single Panda: 6 OSC + 1 gripper"
            assert hi.shape == (7,)
            assert np.all(lo <= hi)
            assert np.all(lo >= -1.0)
            assert np.all(hi <= 1.0)

            # ── Step ────────────────────────────────────────────────
            action = np.zeros(lo.shape[0])
            action[-1] = -1  # open gripper
            obs2, reward, terminated, truncated, info2 = task.step(action)
            assert isinstance(obs2, dict)
            assert isinstance(reward, (int, float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))
            assert isinstance(truncated, (bool, np.bool_))
            assert "task_score" in info2
            assert 0.0 <= info2["task_score"] <= 1.0
            assert info2["step"] == 1

            # ── Config ──────────────────────────────────────────────
            assert hasattr(task, "cfg")
            assert task.cfg["task_id"] == task_id
            assert task.MAX_EPISODE_STEPS == task.cfg["max_episode_steps"]
            assert task.MAX_EPISODE_STEPS > 0
            tc = task.cfg.get("task_conditions", {})
            assert "success" in tc
            assert "failure" in tc

            # ── Env internals ───────────────────────────────────────
            env = task._env
            assert len(env._obj_body_ids) == len(env._dynamic_objects)
            for obj in env._dynamic_objects:
                assert obj.name in env._obj_body_ids
            assert len(env.robots) == 1

            # ── Raw observables ─────────────────────────────────────
            raw = env._get_observations()
            if env._dynamic_objects:
                assert env.cube is not None
                first_name = env._dynamic_objects[0].name
                assert f"{first_name}_pos" in raw
                assert raw[f"{first_name}_pos"].shape == (3,)
                assert f"{first_name}_quat" in raw
                assert raw[f"{first_name}_quat"].shape == (4,)

                # ── Gripper-to-object sensor ────────────────────────────
                grip_keys = [k for k in raw if "gripper_to" in k]
                assert len(grip_keys) > 0

            # ── Step count increments ───────────────────────────────
            task.reset()
            for i in range(3):
                _, _, _, _, si = task.step(np.zeros(lo.shape[0]))
                assert si["step"] == i + 1

        finally:
            task.close()


# ── Proprio-only mode test (one representative task) ────────────────


class TestPrehensileProprioMode:

    def test_proprio_mode_no_cameras(self):
        task = make(PREHENSILE_IDS[0], obs_mode="proprio", render_mode=None, seed=0, max_episode_steps=5)
        try:
            obs, _ = task.reset()
            assert not any(k.startswith("rgb_") for k in obs)
        finally:
            task.close()


# =====================================================================
#  Programmatic success — teleport objects to goal states
# =====================================================================

def _make_task(task_id, **kw):
    defaults = dict(obs_mode="combo", render_mode=None, seed=42, max_episode_steps=200)
    defaults.update(kw)
    return make(task_id, **defaults)


class TestToolRetrieveSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p4-tool-retrieve-v1")
        t.reset()
        yield t
        t.close()

    @staticmethod
    def _lift_cube(task):
        lift_z = task._table_height + task._cube_size + task._lift_height + 0.01
        task._place_object("cube", [0.0, 0.0, lift_z])
        task._set_object_velocity("cube", [0, 0, 0])
        task._env.sim.forward()

    def test_teleport_cube_above_lift_threshold(self, task, monkeypatch):
        task.reset()
        self._lift_cube(task)
        raw_obs = task._get_sim_observations(force_update=True)
        assert task._compute_task_score(raw_obs) < 1.0  # not grasped
        monkeypatch.setattr(task, "_gripper_contacts_object", lambda obj_name="cube": True)
        assert task._compute_task_score(raw_obs) == 1.0
        assert task._check_termination(raw_obs) is True

    def test_step_returns_terminated(self, task, monkeypatch):
        task.reset()
        self._lift_cube(task)
        monkeypatch.setattr(task, "_gripper_contacts_object", lambda obj_name="cube": True)
        lo, _ = task.action_space.low, task.action_space.high
        _, _, terminated, _, info = task.step(np.zeros(lo.shape[0]))
        assert terminated is True
        assert info["task_score"] == 1.0


class TestOccluderPushSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p5-occluder-push-v1")
        t.reset()
        yield t
        t.close()

    def test_teleport_cube_to_patch(self, task):
        task.reset()
        patch_pos = task._patch_pos
        # Rest the cube on the patch (top at table + 2 mm); placement only
        # counts once it is in contact with the surface, so let it settle.
        cube_z = task._table_height + 0.002 + task._cube_size + 0.001
        task._place_object("cube", [patch_pos[0], patch_pos[1], cube_z])
        task._set_object_velocity("cube", [0, 0, 0])
        task._env.sim.forward()

        action = np.zeros(task.action_space.shape)
        for _ in range(10):
            _, _, terminated, _, info = task.step(action)
            if terminated:
                break
        assert info["task_score"] == 1.0
        assert info["is_success"] is True
        assert terminated is True


class TestShelteredGraspSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p1-sheltered-grasp-v1")
        t.reset()
        yield t
        t.close()

    def test_teleport_cube_to_tray(self, task):
        task.reset()
        tray_pos = task._tray_pos
        # Rest the cube on the tray (top at table + 2 mm); placement only
        # counts once it is in contact with the surface, so let it settle.
        cube_z = task._table_height + 0.002 + task._cube_size + 0.001
        task._place_object("cube", [tray_pos[0], tray_pos[1], cube_z])
        task._set_object_velocity("cube", [0, 0, 0])
        task._env.sim.forward()

        action = np.zeros(task.action_space.shape)
        for _ in range(10):
            _, _, terminated, _, info = task.step(action)
            if terminated:
                break
        assert info["task_score"] == 1.0
        assert info["is_success"] is True
        assert terminated is True


# Straight chains cascade reliably; curved patterns can jam.
_DOMINO_LINE_LAYOUT = {"randomization": {"pattern": {"enabled": False}}, "task": {"interconnected": False}}


def _push_first_domino(task, first, steps=300):
    """Tip domino *first* along its heading with a fingertip-like torque and step until done.

    Success needs a real cascade, so the scene cannot be teleported into its end state.
    Returns the last step's ``(terminated, info)``.
    """
    sim = task._env.sim
    body = sim.model.body_name2id(f"{first}_main")
    action = np.zeros(task.action_space.shape)
    try:
        for i in range(steps):
            heading = sim.data.body_xmat[body].reshape(3, 3)[:, 0]
            sim.data.xfrc_applied[body, 3:] = 0.05 * np.cross([0, 0, 1], heading) if i < 25 else 0.0
            _, _, terminated, truncated, info = task.step(action)
            if terminated or truncated:
                break
    finally:
        sim.data.xfrc_applied[:] = 0.0
    return terminated, info


class TestDominoSelectSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p3-domino-select-v1", obs_mode="proprio", max_episode_steps=300,
                       config_overrides=_DOMINO_LINE_LAYOUT)
        t.reset()
        yield t
        t.close()

    def test_first_domino_push_cascades_to_target(self, task):
        task.reset()
        terminated, info = _push_first_domino(task, f"chain{task._correct_chain}_domino_0")
        assert info["task_score"] == 1.0
        assert info["is_success"] is True
        assert terminated is True


class TestCubeTossSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p2-cube-toss-v1", max_episode_steps=500)
        t.reset()
        yield t
        t.close()

    def test_place_on_goal_and_rest(self, task):
        task.reset()
        goal_xy = task._goal_pos
        task._place_object("cube", [goal_xy[0], goal_xy[1], task._table_height + 0.025])
        task._set_object_velocity("cube", [0, 0, 0])
        task._env.sim.forward()

        lo, _ = task.action_space.low, task.action_space.high
        action = np.zeros(lo.shape[0])
        terminated = False
        # Allow extra steps for the cube to fall (it starts 5 mm above the
        # table surface) and settle before the at-rest counter accumulates.
        for _ in range(task._rest_steps + 50):
            _, _, terminated, _, info = task.step(action)
            if terminated:
                break

        assert task.task_score() == 1.0
        assert info["is_success"] is True
        assert terminated is True


class TestSeesawSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p3-seesaw-balance-v1", max_episode_steps=250)
        t.reset()
        yield t
        t.close()

    def test_place_counterweight_balances_seesaw(self, task):
        """Place a matching counterweight at the ideal balance position."""
        task.reset()
        sim = task._env.sim
        fulcrum_x = task._fulcrum_x if task._fulcrum_x is not None else 0.10

        # Use the task's computed ideal counter arm for perfect balance:
        #   target_mass * target_arm == counter_mass * counter_arm
        plank_top_z = (task._table_height + task._fulcrum_height
                       + task._seesaw_thickness + 0.005)
        counter_x = fulcrum_x - task._ideal_counter_arm
        task._place_object("weight_0", [counter_x, 0.0, plank_top_z + 0.02])
        sim.forward()

        # Step enough for physics to settle AND the stability hold
        # counter to fill (_balance_hold_steps defaults to 100)
        import numpy as np
        action = np.zeros(task.action_space.low.shape[0])
        for _ in range(task._balance_hold_steps + 20):
            _, _, terminated, _, info = task.step(action)
            if terminated:
                break

        angle = task._get_seesaw_angle_deg()
        assert angle <= task._tip_angle_threshold, (
            f"Seesaw should be balanced but angle={angle:.1f}°")
        assert task.task_score() == 1.0
        assert info["is_success"] is True

    def test_overbalance_does_not_terminate(self, task):
        """Place a much heavier weight so the seesaw tips the other way.

        Score should be < 1.0 (penalised) and the task should NOT terminate.
        """
        task.reset()
        sim = task._env.sim
        half_len = task._seesaw_length / 2
        fulcrum_x = task._fulcrum_x if task._fulcrum_x is not None else 0.10

        # Place the counterweight on the near end (-x side)
        plank_top_z = (task._table_height + task._fulcrum_height
                       + task._seesaw_thickness + 0.005)
        near_x = fulcrum_x - half_len + 0.04
        task._place_object("weight_0", [near_x, 0.0, plank_top_z + 0.02])
        # 5× the actual counter mass — seesaw should tip heavily
        set_body_mass(sim, sim.model.body_name2id("weight_0_main"), task._counter_mass_actual * 5.0)
        commit_model_edits(sim)

        import numpy as np
        action = np.zeros(task.action_space.low.shape[0])
        terminated = False
        for _ in range(60):
            obs, r, term, trunc, info = task.step(action)
            if term:
                terminated = True
                break

        angle = task._get_seesaw_angle_deg()
        score = task.task_score()

        assert angle > task._tip_angle_threshold, (
            f"Seesaw should be overbalanced but angle={angle:.1f}°")
        assert score < 1.0, f"Overbalanced score should be < 1.0 but got {score:.3f}"
        assert info["is_success"] is False
        assert terminated is False, "Task should NOT terminate on overbalance"


class TestDominoSingleSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p3-domino-single-v1", obs_mode="proprio", max_episode_steps=300,
                       config_overrides=_DOMINO_LINE_LAYOUT)
        t.reset()
        yield t
        t.close()

    def test_first_domino_push_cascades_to_target(self, task):
        task.reset()
        terminated, info = _push_first_domino(task, "domino_0")
        assert info["task_score"] == 1.0
        assert info["is_success"] is True
        assert terminated is True


class TestMassSortSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def task(cls):
        t = _make_task("ib-p5-mass-sort-v1")
        t.reset()
        yield t
        t.close()

    def test_place_cubes_in_correct_zones(self, task):
        task.reset()
        sim = task._env.sim

        # For each cube, find its target zone and teleport there
        for i in range(task._num_cubes):
            assigned_mass = task._masses[task._mass_order[i]]
            sorted_masses = sorted(
                [task._masses[task._mass_order[j]] for j in range(task._num_cubes)]
            )
            target_zone = sorted_masses.index(assigned_mass)

            zone_bid = sim.model.body_name2id(f"zone_{target_zone}_main")
            zone_pos = sim.data.body_xpos[zone_bid].copy()
            obj_name = "cube" if i == 0 else f"cube_{i}"
            task._place_object(obj_name, [zone_pos[0], zone_pos[1], task._table_height + 0.03])
            task._set_object_velocity(obj_name, [0, 0, 0])

        sim.forward()

        raw_obs = task._get_sim_observations(force_update=True)
        score = task._compute_task_score(raw_obs)
        assert score == 1.0, f"Expected 1.0, got {score}"
        assert task._check_termination(raw_obs) is True


# ── Domino pattern generator tests ──────────────────────────────────

_PATTERN_RNG_SEED = 123
_PATTERN_N = 6
_PATTERN_SPACING = 0.045


class TestHeadingToQuat:

    def test_zero_heading(self):
        q = heading_to_quat(0.0)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_90_deg(self):
        q = heading_to_quat(np.pi / 2)
        expected = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        np.testing.assert_allclose(q, expected, atol=1e-10)


class TestLayoutHelpers:

    def test_translate(self):
        layout = [(0, 0, 0), (1, 0, 0)]
        out = translate_layout(layout, 10, 20)
        assert out[0][:2] == pytest.approx((10, 20))
        assert out[1][:2] == pytest.approx((11, 20))

    def test_rotate_90(self):
        layout = [(1, 0, 0)]
        out = rotate_layout(layout, np.pi / 2)
        assert out[0][0] == pytest.approx(0, abs=1e-10)
        assert out[0][1] == pytest.approx(1, abs=1e-10)


class TestPatternGenerators:
    """Verify every pattern returns correct count, start at origin, and spacing."""

    @pytest.fixture(params=ALL_PATTERNS)
    def layout(self, request):
        rng = np.random.RandomState(_PATTERN_RNG_SEED)
        return request.param, generate_pattern(
            request.param, _PATTERN_N, _PATTERN_SPACING, rng
        )

    def test_correct_count(self, layout):
        _name, pts = layout
        assert len(pts) == _PATTERN_N

    def test_starts_near_origin(self, layout):
        _name, pts = layout
        x0, y0, _ = pts[0]
        assert abs(x0) < 1e-10
        assert abs(y0) < 1e-10

    def test_consecutive_spacing(self, layout):
        """Euclidean distance between consecutive dominoes ≈ SPACING."""
        name, pts = layout
        for i in range(len(pts) - 1):
            dx = pts[i + 1][0] - pts[i][0]
            dy = pts[i + 1][1] - pts[i][1]
            dist = np.sqrt(dx * dx + dy * dy)
            assert dist == pytest.approx(_PATTERN_SPACING, rel=0.25), (
                f"{name}: segment {i} dist={dist:.5f}, expected ~{_PATTERN_SPACING}"
            )

    def test_heading_direction(self, layout):
        """Heading of each domino should roughly point toward the next."""
        _name, pts = layout
        for i in range(len(pts) - 1):
            x0, y0, h = pts[i]
            x1, y1, _ = pts[i + 1]
            dx, dy = x1 - x0, y1 - y0
            expected = np.arctan2(dy, dx)
            diff = abs((h - expected + np.pi) % (2 * np.pi) - np.pi)
            assert diff < 0.01, f"heading mismatch at {i}: {h:.3f} vs {expected:.3f}"

    def test_generates_in_plus_x(self, layout):
        """Average x of the chain should be positive (progresses forward)."""
        _name, pts = layout
        avg_x = sum(p[0] for p in pts) / len(pts)
        assert avg_x > 0


class TestArcPattern:

    def test_turn_between_neighbours_capped(self):
        """A sharper turn between neighbouring dominoes jams the cascade."""
        for n in range(2, 11):
            for seed in range(20):
                pts = generate_pattern("arc", n, _PATTERN_SPACING, np.random.RandomState(seed))
                turns = np.abs(np.diff(np.unwrap([h for _, _, h in pts])))
                assert turns.max() <= ARC_MAX_TURN_PER_LINK + 1e-9, (n, seed)


class TestGeneratePatternFallback:

    def test_unknown_pattern_falls_back_to_line(self):
        rng = np.random.RandomState(0)
        out = generate_pattern("nonexistent", 4, 0.05, rng)
        assert len(out) == 4
        # Should be a line along +x
        for i, (x, y, _) in enumerate(out):
            assert x == pytest.approx(i * 0.05, abs=1e-10)
            assert y == pytest.approx(0.0, abs=1e-10)

    def test_single_domino(self):
        rng = np.random.RandomState(0)
        for name in ALL_PATTERNS:
            out = generate_pattern(name, 1, 0.05, rng)
            assert len(out) == 1

    def test_two_dominoes(self):
        rng = np.random.RandomState(0)
        for name in ALL_PATTERNS:
            out = generate_pattern(name, 2, 0.05, rng)
            assert len(out) == 2
