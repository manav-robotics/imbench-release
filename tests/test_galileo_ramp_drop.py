"""Galileo's right-side entry, covered goal, first-swing and binary reward regressions."""

import numpy as np
import pytest

from tests.conftest import cached_task
from data_synthetic.single_arm_pipeline.galileo_ramp_drop_policy import RELEASE_ANGLE_POLY
from tests.test_prehensile_regressions import (
    _action, _calibrated_release_point, _place_at_rest, _ramp_arc_point,
)


@pytest.fixture(params=[1, 2])
def task(request):
    return cached_task(
        f"ib-p2-galileo-ramp-drop-v{request.param}", obs_mode="proprio",
        privileged_obs=True, max_episode_steps=150,
    )


@pytest.mark.parametrize("seed", range(6))
def test_right_side_roll_succeeds_with_binary_reward(task, seed):
    task.reset(seed=seed)
    # Energy-matched release on the open arm: the first apex lands on the goal.
    _place_at_rest(task, "cube", _calibrated_release_point(task))
    rewards = []
    for _ in range(100):
        _, reward, terminated, truncated, info = task.step(_action(task))
        rewards.append(reward)
        assert reward in (0.0, 1.0)
        assert info["task_score"] == reward == float(info["is_success"])
        if terminated or truncated:
            break
    assert info["is_success"] and terminated
    assert task._entered_from_right and task._crossed_trough
    assert rewards[-1] == 1.0 and all(r == 0.0 for r in rewards[:-1])


@pytest.mark.parametrize("seed", [0, 4, 5])
@pytest.mark.parametrize("entry", ["goal_inside", "goal_above", "trough"])
def test_shortcut_never_scores_even_after_later_oscillations(task, seed, entry):
    task.reset(seed=seed)
    theta = 0 if entry == "trough" else task._goal_theta
    pos = _ramp_arc_point(task, theta)
    if entry == "goal_above":
        pos[2] += 0.12
    _place_at_rest(task, "cube", pos)
    for _ in range(100):
        _, reward, terminated, truncated, info = task.step(_action(task))
        assert reward == info["task_score"] == 0.0
        assert not info["is_success"]
        if terminated or truncated:
            break
    assert not task._crossed_trough


def test_cover_physically_blocks_vertical_goal_drop(task):
    task.reset(seed=0)
    pos = _ramp_arc_point(task, task._goal_theta)
    pos[2] += 0.12
    _place_at_rest(task, "cube", pos)
    hit_cover = False
    for _ in range(35):
        task.step(_action(task))
        for contact in task.sim.data.contact[:task.sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & task._ball_geoms:
                hit_cover |= any(
                    (task.sim.model.geom_id2name(g) or "").startswith("ramp_cover_")
                    for g in pair - task._ball_geoms
                )
    assert hit_cover, "the ball passed through the goal-side cover"
    assert not task._task_done


def test_reset_forgets_previous_valid_route(task):
    task.reset(seed=0)
    task._entered_from_right = task._crossed_trough = task._task_done = True
    task.reset(seed=0)
    assert not task._entered_from_right and not task._crossed_trough
    assert not task._task_done and not task._invalid_entry


@pytest.mark.parametrize("axis", [0, 2])
def test_lateral_or_vertical_motion_is_not_a_stopped_turn(task, monkeypatch, axis):
    task.reset(seed=0)
    task._entered_from_right = task._crossed_trough = True
    task._untouched_steps = task._UNTOUCHED_STEPS
    rot = task._goal_rot_world
    _place_at_rest(task, "cube", task._goal_center_world + rot[:, 2] * task._ball_radius)
    module = type(task).__module__
    monkeypatch.setattr(f"{module}.geoms_touch", lambda *args: True)
    raw = task._get_sim_observations(force_update=True)
    for sign in (1, -1):
        monkeypatch.setattr(task, "_ball_velocity", lambda: sign * 0.3 * rot[:, 1] + 0.3 * rot[:, axis])
        assert task._compute_task_score(raw) == 0.0


def test_robot_recontact_cancels_valid_route(task, monkeypatch):
    task.reset(seed=0)
    task._entered_from_right = task._crossed_trough = True
    module = type(task).__module__
    monkeypatch.setattr(f"{module}.robot_contacts", lambda *args: True)
    task._compute_task_score(task._get_sim_observations(force_update=True))
    assert not task._entered_from_right and not task._crossed_trough
    assert task._untouched_steps == 0


def test_reversal_at_goal_edge_does_not_imply_stop_inside(task, monkeypatch):
    task.reset(seed=0)
    task._entered_from_right = task._crossed_trough = True
    task._untouched_steps = task._UNTOUCHED_STEPS
    rot = task._goal_rot_world
    pos = (task._goal_center_world + rot[:, 2] * task._ball_radius
           + rot[:, 1] * (task._goal_patch_half_y - 0.001))
    _place_at_rest(task, "cube", pos)
    monkeypatch.setattr(f"{type(task).__module__}.geoms_touch", lambda *args: True)
    raw = task._get_sim_observations(force_update=True)
    for sign in (1, -1):
        monkeypatch.setattr(task, "_ball_velocity", lambda: sign * 0.3 * rot[:, 1])
        assert task._compute_task_score(raw) == 0.0


def _run_untouched(task, steps=150):
    """Step with the arm still; return whether the task ever scored."""
    for _ in range(steps):
        _, reward, terminated, truncated, info = task.step(_action(task))
        if reward == 1.0:
            return True
        if terminated or truncated:
            break
    return False


@pytest.mark.parametrize("seed", range(12))
def test_goal_blind_high_release_cannot_score_on_a_later_swing(task, seed):
    """A fixed release high on the open arm overshoots on the first swing.

    The oscillation decays, so a later, lower swing would turn around inside
    any goal; only the first goal-side swing may count.
    """
    task.reset(seed=seed)
    if np.rad2deg(task._goal_theta) > 45:
        pytest.skip("goal within reach of the first swing from this release")
    _place_at_rest(task, "cube", _ramp_arc_point(task, -np.deg2rad(72)))
    assert not _run_untouched(task)
    assert task._crossed_trough and task._first_swing_spent


@pytest.mark.parametrize("seed", [0, 3, 4, 6, 8, 10])
def test_release_below_goal_height_fails(task, seed):
    task.reset(seed=seed)
    _place_at_rest(task, "cube", _ramp_arc_point(task, -(task._goal_theta - np.deg2rad(18))))
    assert not _run_untouched(task)
    assert task._first_swing_spent


def test_regrasp_after_a_missed_swing_starts_a_new_attempt(task, monkeypatch):
    task.reset(seed=0)
    task._crossed_trough = task._first_swing_spent = True
    module = type(task).__module__
    monkeypatch.setattr(f"{module}.robot_contacts", lambda *args: True)
    task._compute_task_score(task._get_sim_observations(force_update=True))
    assert not task._first_swing_spent and not task._crossed_trough


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("offset_deg", [-3.0, 3.0])
def test_release_three_degrees_off_misses_the_goal(task, seed, offset_deg):
    """The goal is +/-5 mm along the arc: a release 3 deg off the energy match misses it."""
    task.reset(seed=seed)
    phi = np.polyval(RELEASE_ANGLE_POLY, np.rad2deg(task._goal_theta)) + offset_deg
    _place_at_rest(task, "cube", _ramp_arc_point(task, -np.deg2rad(phi)))
    assert not _run_untouched(task)
    assert task._first_swing_spent and task._last_apex_local is not None
    assert abs(task._last_apex_local[1]) > task._goal_patch_half_y


@pytest.mark.parametrize("seed", range(6))
def test_naive_mirror_release_falls_short(task, seed):
    """Releasing at the goal's mirror angle ignores the channel losses: the apex lands short."""
    task.reset(seed=seed)
    _place_at_rest(task, "cube", _ramp_arc_point(task, -task._goal_theta))
    assert not _run_untouched(task)
    assert task._last_apex_local[1] < -task._goal_patch_half_y


@pytest.mark.parametrize("seed", range(4))
def test_scripted_policy_lands_first_apex_on_goal(seed):
    """The calibrated scripted policy picks the ball up and lands its first apex on the goal."""
    from data_synthetic.single_arm_pipeline.galileo_ramp_drop_policy import GalileoRampDropPolicy

    task = cached_task("ib-p2-galileo-ramp-drop-v1", obs_mode="proprio", privileged_obs=True)
    obs, _ = task.reset(seed=seed)
    policy = GalileoRampDropPolicy()
    for _ in range(task.MAX_EPISODE_STEPS):
        obs, _, terminated, truncated, info = task.step(policy.act(obs, env=task))
        if terminated or truncated:
            break
    assert info["is_success"], task._last_apex_local
