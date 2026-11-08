"""Regression tests for puzzle, balance and freeform task scoring.

Each test drives a specific scenario (teleported objects, a few physics
steps) and checks the scorer, termination or reset behaviour.
"""

import numpy as np
import pytest

from imbench.registry import make
from imbench.tasks.puzzle._stack_scoring import (
    body_subtree_geom_ids,
    count_table_and_gripper_contacts,
    object_body_id,
)


def _make_task(task_id, **kw):
    defaults = dict(obs_mode="proprio", render_mode=None, seed=0, privileged_obs=True,
                    max_episode_steps=400)
    defaults.update(kw)
    return make(task_id, **defaults)


def _zero(task):
    return np.zeros(task.action_space.shape[0])


def _teleport(task, name, pos, quat=None):
    task._place_object(name, pos, quat=quat)
    task._set_object_velocity(name, [0, 0, 0], [0, 0, 0])
    task._env.sim.forward()


def _body_pos(task, body):
    sim = task._env.sim
    return sim.data.body_xpos[sim.model.body_name2id(body)].copy()


# =====================================================================
#  Fixed observation shape across episodes
# =====================================================================

class TestObservationShapeStable:

    @pytest.mark.parametrize("task_id", [
        "ib-p7-shape-stack-v1",
        "ib-p2-ramp-sort-v1",
        "ib-p3-indirect-push-v1",
        "ib-p6-pick-place-ycb-v1",
    ])
    def test_object_state_shape_constant_and_obs_in_space(self, task_id):
        task = _make_task(task_id)
        try:
            space = task.observation_space
            shapes = set()
            for seed in range(6):
                obs, _ = task.reset(seed=seed)
                shapes.add(obs["object_state"].shape)
                assert space.contains(obs), f"seed={seed}: reset obs outside observation_space"
            obs, *_ = task.step(_zero(task))
            assert space.contains(obs)
            assert len(shapes) == 1, f"object_state shape varies across seeds: {shapes}"
        finally:
            task.close()


# =====================================================================
#  Stack collapse recovery
# =====================================================================

class TestStackCollapse:
    _CUBES = ["cube_small", "cube_medium", "cube_large"]

    def _cubes_on_table_after_collapse(self, both_prob, seeds):
        task = _make_task("ib-p6-stack-collapse-recovery-v1",
                          config_overrides={"task": {"collapse": {"both_prob": both_prob}}})
        try:
            counts = []
            for seed in seeds:
                task.reset(seed=seed)
                for _ in range(120):
                    _, _, term, _, _ = task.step(_zero(task))
                    assert not term, f"seed={seed}: a cube left the table during the collapse"
                n_touching, _ = count_table_and_gripper_contacts(task._env, self._CUBES)
                counts.append(n_touching)
                assert task._collapse_settled, f"seed={seed}: collapse never settled"
            return counts
        finally:
            task.close()

    def test_both_mode_knocks_both_upper_cubes_onto_table(self):
        assert self._cubes_on_table_after_collapse(1.0, range(3)) == [3, 3, 3]

    def test_top_mode_leaves_middle_cube_on_base(self):
        assert self._cubes_on_table_after_collapse(0.0, range(2)) == [2, 2]

    @pytest.mark.parametrize("both_prob", [0.0, 1.0])
    def test_no_credit_before_collapse_settles(self, both_prob):
        """Neither the intact tower nor a cube in flight counts as a recovered stack.

        Zero actions may earn at most the 0.25 of a middle cube left on the
        base once the collapse has settled.
        """
        task = _make_task("ib-p6-stack-collapse-recovery-v1",
                          config_overrides={"task": {"collapse": {"both_prob": both_prob}}})
        try:
            task.reset(seed=0)
            for _ in range(30):
                _, _, _, _, info = task.step(_zero(task))
                assert info["task_score"] <= 0.25 and task._stable_count == 0
        finally:
            task.close()

    def test_restack_after_collapse_succeeds(self):
        task = _make_task("ib-p6-stack-collapse-recovery-v1",
                          config_overrides={"task": {"collapse": {"both_prob": 1.0}}})
        try:
            task.reset(seed=0)
            for _ in range(120):
                task.step(_zero(task))
            x, y, z = _body_pos(task, "cube_large_main")
            _teleport(task, "cube_medium", [x, y, z + 0.030 + 0.025 + 0.001])
            _teleport(task, "cube_small", [x, y, z + 0.030 + 0.050 + 0.020 + 0.002])
            for _ in range(task._stable_steps_required + 20):
                _, _, term, _, info = task.step(_zero(task))
                if term:
                    break
            assert info["is_success"] and term
        finally:
            task.close()

    def test_object_color_flag_respected(self):
        from imbench.tasks.puzzle.stack_collapse_recovery_v1 import _CUBES
        task = _make_task("ib-p6-stack-collapse-recovery-v1")
        try:
            assert not task._rand_object_color
            task.reset(seed=3)
            sim = task._env.sim
            for c in _CUBES:
                gid = sim.model.geom_name2id(f"{c['name']}_g0_vis")
                np.testing.assert_allclose(sim.model.geom_rgba[gid], c["rgba"])
        finally:
            task.close()


# =====================================================================
#  Ramp sort
# =====================================================================

@pytest.fixture(scope="module")
def ramp_task():
    t = _make_task("ib-p2-ramp-sort-v1")
    yield t
    t.close()


class TestRampSort:

    def test_barrier_top_blocks_balls_dropped_from_above(self, ramp_task):
        ramp_task.reset(seed=0)
        model = ramp_task._env.sim.model
        top_z = model.body_pos[model.body_name2id("barrier_top_main")][2]
        _teleport(ramp_task, "sphere_0", [*ramp_task._sphere_bin_world_xy, top_z + 0.05])
        _teleport(ramp_task, "box_0", [*ramp_task._cube_bin_world_xy, top_z + 0.05])
        for _ in range(100):
            _, _, term, _, info = ramp_task.step(_zero(ramp_task))
        assert info["task_score"] == 0.0
        assert not info["is_success"]
        assert _body_pos(ramp_task, "sphere_0_main")[2] > top_z

    def test_fence_walls_block_balls(self, ramp_task):
        """A ball thrown at a bin through a fence wall bounces off it; only the ramp side is open."""
        ramp_task.reset(seed=0)
        model = ramp_task._env.sim.model
        ball = model.geom_name2id(ramp_task._env.get_object("sphere_0").contact_geoms[0])

        def collides(panel):
            g = model.geom_name2id(ramp_task._env.get_object(panel).contact_geoms[0])
            return bool((model.geom_contype[g] & model.geom_conaffinity[ball])
                        | (model.geom_contype[ball] & model.geom_conaffinity[g]))

        assert [collides(p) for p in ("barrier_front", "barrier_left", "barrier_right",
                                      "barrier_top", "barrier_back")] == [True] * 4 + [False]
        wall_x = min(_body_pos(ramp_task, f"barrier_{s}_main")[0] for s in ("left", "right"))
        bin_xy = ramp_task._sphere_bin_world_xy
        start = np.array([wall_x - 0.03, bin_xy[1], ramp_task._table_height + 0.20])
        fall_time = np.sqrt(2 * (start[2] - ramp_task._table_height - 0.028) / 9.81)
        _teleport(ramp_task, "sphere_0", start)
        ramp_task._set_object_velocity("sphere_0", [(bin_xy[0] - start[0]) / fall_time, 0, 0])
        for _ in range(150):
            _, _, term, _, info = ramp_task.step(_zero(ramp_task))
            if term:
                break
        assert info["task_score"] == 0.0
        assert _body_pos(ramp_task, "sphere_0_main")[0] < wall_x

    def test_ball_above_wrong_bin_is_not_in_it(self, ramp_task):
        ramp_task.reset(seed=0)
        bin_xy = ramp_task._sphere_bin_world_xy
        _teleport(ramp_task, "box_0", [*bin_xy, ramp_task._table_height + 0.40])
        raw = ramp_task._get_sim_observations(force_update=True)
        assert not ramp_task._any_ball_in_wrong_bin()
        assert not ramp_task._check_termination(raw)

    def test_ball_in_wrong_bin_terminates(self, ramp_task):
        ramp_task.reset(seed=0)
        bin_xy = ramp_task._sphere_bin_world_xy
        _teleport(ramp_task, "box_0", [*bin_xy, ramp_task._table_height + 0.03])
        raw = ramp_task._get_sim_observations(force_update=True)
        assert ramp_task._compute_task_score(raw) == 0.0
        assert ramp_task._check_termination(raw)

    def test_visual_geoms_stay_non_colliding(self, ramp_task):
        ramp_task.reset(seed=0)
        m = ramp_task._env.sim.model
        colliding_vis = [
            m.geom_id2name(g) for g in range(m.ngeom)
            if (m.geom_id2name(g) or "").endswith(("_vis", "_visual"))
            and (m.geom_contype[g] or m.geom_conaffinity[g])
        ]
        assert colliding_vis == []


# =====================================================================
#  Pick-place YCB
# =====================================================================

@pytest.fixture(scope="module")
def ycb_task():
    t = _make_task("ib-p6-pick-place-ycb-v1")
    yield t
    t.close()


class TestPickPlaceYCB:

    def test_object_still_held_at_goal_is_not_success(self, ycb_task, monkeypatch):
        import imbench.tasks.puzzle.pick_place_ycb_v1 as ycb_module

        ycb_task.reset(seed=0)
        name = ycb_task._active_obj_name()
        ycb_task._was_lifted = True
        ycb_task._mg_remaining = 0
        _teleport(ycb_task, name, [*ycb_task._goal_pos, ycb_task._rest_z])
        monkeypatch.setattr(ycb_module, "robot_contacts", lambda sim, geoms: True)
        assert ycb_task.task_score(recompute=True) < 1.0

    def test_doing_nothing_scores_zero(self, ycb_task):
        ycb_task.reset(seed=0)
        for _ in range(60):
            _, _, _, _, info = ycb_task.step(_zero(ycb_task))
        assert info["task_score"] == 0.0

    def test_inactive_objects_parked(self, ycb_task):
        ycb_task.reset(seed=0)
        for _ in range(20):
            ycb_task.step(_zero(ycb_task))
        parked = ycb_task._parked_objects()
        assert len(parked) == len(ycb_task._ycb_pool) - 1
        for name in parked:
            assert _body_pos(ycb_task, f"{name}_main")[2] > 50.0

    def test_recompute_scores_active_object(self, ycb_task):
        ycb_task.reset(seed=0)
        name = ycb_task._active_obj_name()
        # The env's cube_pos alias points at the first pool object.
        assert name != ycb_task._env._dynamic_objects[0].name
        ycb_task._was_lifted = True
        ycb_task._mg_remaining = 0
        _teleport(ycb_task, name, [*ycb_task._goal_pos, ycb_task._rest_z])
        assert ycb_task.task_score(recompute=True) == 1.0

    def test_placement_before_final_relocation_scores_nothing(self, ycb_task):
        ycb_task.reset(seed=0)
        name = ycb_task._active_obj_name()
        goal = ycb_task._goal_pos.copy()
        remaining = ycb_task._mg_remaining
        assert remaining > 0
        # Lift away from the goal, then set the object down at rest in the
        # patch without a lifted approach.
        xy = _body_pos(ycb_task, f"{name}_main")[:2]
        _teleport(ycb_task, name, [*xy, ycb_task._rest_z + ycb_task._mg_min_lift + 0.01])
        ycb_task.step(_zero(ycb_task))
        assert ycb_task._was_lifted
        _teleport(ycb_task, name, [*goal, ycb_task._rest_z])
        for _ in range(30):
            _, _, terminated, _, info = ycb_task.step(_zero(ycb_task))
            assert not info["is_success"] and not terminated
        stages = ycb_task._stage_debug
        assert stages["xy_over_goal"] == stages["z_placement"] == stages["settled_at_rest"] == 0.0
        assert ycb_task._mg_remaining == remaining
        np.testing.assert_array_equal(ycb_task._goal_pos, goal)

    def test_placement_succeeds_once_every_relocation_fired(self, ycb_task):
        ycb_task.reset(seed=0)
        name = ycb_task._active_obj_name()
        lift_z = ycb_task._rest_z + ycb_task._mg_min_lift + 0.01
        # Each lifted approach relocates the goal once.
        for _ in range(ycb_task._mg_total):
            _teleport(ycb_task, name, [*ycb_task._goal_pos, lift_z])
            ycb_task.step(_zero(ycb_task))
        assert ycb_task._mg_remaining == 0
        # Carry the object over the final patch, then set it down there.
        _teleport(ycb_task, name, [*ycb_task._goal_pos, lift_z])
        ycb_task.step(_zero(ycb_task))
        _teleport(ycb_task, name, [*ycb_task._goal_pos, ycb_task._rest_z])
        for _ in range(30):
            _, _, terminated, _, info = ycb_task.step(_zero(ycb_task))
            if terminated:
                break
        assert info["is_success"]

    def test_lift_measured_from_rest_height(self, ycb_task):
        ycb_task.reset(seed=0)
        name = ycb_task._active_obj_name()
        xy = _body_pos(ycb_task, f"{name}_main")[:2]
        _teleport(ycb_task, name, [*xy, ycb_task._rest_z + ycb_task._mg_min_lift - 0.01])
        ycb_task.task_score(recompute=True)
        assert not ycb_task._was_lifted
        _teleport(ycb_task, name, [*xy, ycb_task._rest_z + ycb_task._mg_min_lift + 0.01])
        ycb_task.task_score(recompute=True)
        assert ycb_task._was_lifted


# =====================================================================
#  Gap funnel
# =====================================================================

@pytest.fixture(scope="module")
def gap_task():
    t = _make_task("ib-p3-gap-funnel-v1")
    yield t
    t.close()


class TestGapFunnel:

    def test_airborne_ball_over_goal_does_not_succeed(self, gap_task):
        gap_task.reset(seed=0)
        _teleport(gap_task, "cube", [gap_task._goal_x_min + 0.04, gap_task._gap_center_y + 0.2,
                                 gap_task._table_height + 0.2])
        _, _, term, _, info = gap_task.step(_zero(gap_task))
        assert not info["is_success"]
        assert not term

    def test_ball_resting_in_goal_succeeds_after_hold(self, gap_task):
        gap_task.reset(seed=0)
        _teleport(gap_task, "cube", [gap_task._goal_x_min + 0.10, gap_task._gap_center_y,
                                 gap_task._table_height + gap_task._ball_radius])
        successes = []
        for _ in range(gap_task._goal_hold_steps + 5):
            _, _, term, _, info = gap_task.step(_zero(gap_task))
            successes.append(info["is_success"])
            if term:
                break
        assert term and successes[-1]
        assert not any(successes[:gap_task._goal_hold_steps - 1])

    def test_ball_cannot_roll_around_barrier_end(self, gap_task):
        gap_task.reset(seed=0)
        _teleport(gap_task, "cube", [0.10, 0.395, gap_task._table_height + gap_task._ball_radius])
        gap_task._set_object_velocity("cube", [0.8, -0.15, 0])
        for _ in range(150):
            _, _, term, _, info = gap_task.step(_zero(gap_task))
            assert not info["is_success"]
            if term:
                break

    def test_no_alignment_credit_at_spawn(self, gap_task):
        for seed in range(4):
            gap_task.reset(seed=seed)
            _, _, _, _, info = gap_task.step(_zero(gap_task))
            assert info["task_score"] == 0.0, f"seed={seed}"

    def test_scripted_push_reaches_goal(self):
        # The policy relies on the gripper-contact velocity boost firing.
        from data_synthetic.single_arm_pipeline.gap_funnel_policy import GapFunnelPolicy

        t = _make_task("ib-p3-gap-funnel-v1", max_episode_steps=1300)
        try:
            obs, _ = t.reset(seed=0)
            policy = GapFunnelPolicy(rng=np.random.default_rng(0))
            policy.reset()
            obs, *_ = t.step(_zero(t))
            for _ in range(1300):
                obs, _, term, trunc, info = t.step(np.clip(policy.act(obs, env=t), -1, 1))
                if term or trunc:
                    break
            assert info["is_success"]
        finally:
            t.close()


# =====================================================================
#  Ball cage escape
# =====================================================================

@pytest.fixture(scope="module")
def cage_task():
    t = _make_task("ib-p3-ball-cage-escape-v1")
    yield t
    t.close()


class TestBallCageEscape:

    def test_ball_released_when_cage_pushed_off_stool(self, cage_task):
        cage_task.reset(seed=0)
        sim = cage_task._env.sim
        cage = _body_pos(cage_task, "cage_root")
        quat = sim.data.body_xquat[sim.model.body_name2id("cage_root")].copy()
        _teleport(cage_task, "cage", [cage[0] + 0.25, cage[1], cage_task._table_height + 0.03],
                  quat=quat)
        for _ in range(100):
            cage_task.step(_zero(cage_task))
        assert cage_task._ball_released and not cage_task._cage_picked_up
        assert _body_pos(cage_task, "cube_main")[2] < cage_task._table_height + 0.05

    @pytest.mark.parametrize("seed", [0, 1])
    def test_caged_ball_touching_table_is_not_success(self, cage_task, seed):
        """Cage set flat on the table: the ball rests in the floor hole on the table, still caged."""
        cage_task.reset(seed=seed)
        ci = np.asarray(cage_task._cage_inner) / 2
        wt = cage_task._wall_thickness
        cage_xy = np.array([0.10, 0.25])
        _teleport(cage_task, "cage", [*cage_xy, cage_task._table_height + ci[2] + wt / 2 + 0.001])
        hole = cage_xy + cage_task._hole_xy
        _teleport(cage_task, "cube", [*hole, cage_task._table_height + wt + cage_task._ball_radius + 0.002])
        for _ in range(60):
            _, _, term, _, info = cage_task.step(_zero(cage_task))
            assert not info["is_success"] and not term
        assert cage_task._ball_touches_table()

    def test_ball_escaped_onto_table_succeeds(self, cage_task):
        cage_task.reset(seed=0)
        cage = _body_pos(cage_task, "cage_root")
        _teleport(cage_task, "cube", [cage[0], cage[1] + 0.20,
                                      cage_task._table_height + cage_task._ball_radius])
        cage_task._ball_released = True
        successes = []
        for _ in range(60):
            _, _, term, _, info = cage_task.step(_zero(cage_task))
            successes.append(info["is_success"])
            if term:
                break
        assert any(successes[:5]) and term and info["is_success"]

    def test_score_drops_with_success_when_ball_leaves_table(self, cage_task):
        cage_task.reset(seed=0)
        sim = cage_task._env.sim
        cage = _body_pos(cage_task, "cage_root")
        quat = sim.data.body_xquat[sim.model.body_name2id("cage_root")].copy()
        # Every stage reached: cage lifted clear, ball out of it on the table.
        _teleport(cage_task, "cage", cage + [0.0, 0.0, 0.10], quat=quat)
        ball_xy = [cage[0], cage[1] + 0.20]
        _teleport(cage_task, "cube", [*ball_xy, cage_task._table_height + cage_task._ball_radius])
        cage_task._ball_released = True
        for _ in range(5):
            _, _, _, _, info = cage_task.step(_zero(cage_task))
        assert info["is_success"] and info["task_score"] == 1.0
        assert all(v == 1.0 for k, v in cage_task._stage_debug.items() if k != "_success")

        # Lifted clear of the table: no longer a success, so no full score.
        _teleport(cage_task, "cube", [*ball_xy, cage_task._table_height + 0.10])
        _, _, term, _, info = cage_task.step(_zero(cage_task))
        assert not info["is_success"] and not term
        assert info["task_score"] < 1.0

    def test_ball_off_table_terminates(self, cage_task):
        cage_task.reset(seed=0)
        _teleport(cage_task, "cube", [0.10, 0.70, 0.02])
        cage_task._ball_released = True
        _, _, term, _, info = cage_task.step(_zero(cage_task))
        assert term and not info["is_success"]

    def test_ball_stays_frozen_while_cage_undisturbed(self, cage_task):
        cage_task.reset(seed=1)
        ball0 = _body_pos(cage_task, "cube_main")
        for _ in range(50):
            cage_task.step(_zero(cage_task))
        assert not cage_task._ball_released
        np.testing.assert_allclose(_body_pos(cage_task, "cube_main"), ball0, atol=1e-9)

    def test_hole_always_over_stool(self, cage_task):
        stool_half = np.asarray(cage_task._stool_size[:2]) / 2
        for seed in range(8):
            cage_task.reset(seed=seed)
            assert np.all(np.abs(cage_task._hole_xy) + cage_task._hole_half <= stool_half + 1e-9)

    def test_reset_settle_does_not_move_arm(self, cage_task):
        robot = cage_task._env.robots[0]
        captured = {}
        orig_reset = cage_task._env.reset

        def reset_and_capture():
            obs = orig_reset()
            captured["q"] = cage_task._env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
            return obs

        cage_task._env.reset = reset_and_capture
        try:
            cage_task.reset(seed=0)
        finally:
            del cage_task._env.reset
        np.testing.assert_allclose(
            cage_task._env.sim.data.qpos[robot._ref_joint_pos_indexes], captured["q"], atol=1e-12)


# =====================================================================
#  Shape stack
# =====================================================================

@pytest.fixture(scope="module")
def shape_task():
    t = _make_task("ib-p7-shape-stack-v1")
    yield t
    t.close()


class TestShapeStack:

    def test_reset_reuses_env(self, shape_task):
        env = shape_task._env
        for seed in range(4):
            shape_task.reset(seed=seed)
            assert shape_task._env is env

    def test_objects_spawn_resting_on_table(self, shape_task):
        seen = set()
        for seed in range(30):
            shape_task.reset(seed=seed)
            sim = shape_task._env.sim
            for arch in shape_task._selected_archetypes:
                seen.add(arch["name"])
                gids = body_subtree_geom_ids(sim, object_body_id(sim, arch["name"]))
                bottom = min(
                    sim.data.geom_xpos[g][2]
                    - (sim.model.geom_size[g][2] if sim.model.geom_type[g] == 6
                       else sim.model.geom_size[g][1])
                    for g in gids
                )
                assert bottom == pytest.approx(shape_task._table_height, abs=1e-4), arch["name"]
        assert "cone_frustum" in seen

    def test_no_free_credit_at_spawn(self, shape_task):
        shape_task.reset(seed=0)
        for _ in range(3):
            _, _, _, _, info = shape_task.step(_zero(shape_task))
            assert info["task_score"] < 0.02


# =====================================================================
#  Indirect push
# =====================================================================

class TestIndirectPush:

    def test_target_must_rest_in_goal(self):
        """One slow sample in the goal is not success; a sustained rest is, until it moves."""
        task = _make_task("ib-p3-indirect-push-v1")
        try:
            task.reset(seed=0)
            goal = task._goal_pos
            _teleport(task, "target", [goal[0], goal[1], task._table_height + 0.026])
            task.task_score(recompute=True)
            assert not task._target_in_goal()
            for _ in range(50):
                if task._target_in_goal():
                    break
                task.task_score(recompute=True)
            assert task._target_in_goal()
            task._set_object_velocity("target", [1.2, 0.0, 0.0], [0, 0, 0])
            task.task_score(recompute=True)
            assert not task._target_in_goal()
        finally:
            task.close()

    def test_no_chain_credit_at_spawn(self):
        task = _make_task("ib-p3-indirect-push-v1")
        try:
            for seed in range(8):
                task.reset(seed=seed)
                _, _, _, _, info = task.step(_zero(task))
                assert info["task_score"] < 1e-3, f"seed={seed}"
        finally:
            task.close()

    def test_target_knocked_off_table_terminates(self):
        task = _make_task("ib-p3-indirect-push-v1")
        try:
            task.reset(seed=0)
            _teleport(task, "target", [0.9, 0.0, 0.03])
            _, _, term, _, info = task.step(_zero(task))
            assert term
            assert not info["is_success"]
        finally:
            task.close()
