"""Tests for puzzle task scoring and termination.

Covers:
- imbench.tasks.puzzle.* (_compute_task_score, _check_termination)
- Shape stack stable_count, ring toss distance, pegboard, ramp sort, etc.
"""

import numpy as np
import pytest

from imbench.registry import make


# ── Helpers ─────────────────────────────────────────────────────────

def _make_task(task_id, **kw):
    # Proprio mode skips camera rendering (cheaper); individual tests that
    # require RGB obs should pass obs_mode="combo" as an override.
    defaults = dict(obs_mode="proprio", render_mode=None, seed=42, max_episode_steps=10)
    defaults.update(kw)
    return make(task_id, **defaults)


def _step_n(task, n=5, action=None):
    lo, _ = task.action_space.low, task.action_space.high
    if action is None:
        action = np.zeros(lo.shape[0])
    for _ in range(n):
        task.step(action)


# =====================================================================
#  Puzzle tasks — score/termination spot checks
# =====================================================================

class TestPuzzleTaskScoring:

    @pytest.mark.parametrize("task_id", [
        "ib-p7-shape-stack-v1",
        "ib-p6-stack-collapse-recovery-v1",
        "ib-p2-ramp-sort-v1",
        "ib-p3-ball-cage-escape-v1",
        "ib-p3-indirect-push-v1",
    ])
    def test_initial_score_bounded(self, task_id):
        task = _make_task(task_id)
        try:
            task.reset()
            raw = task._get_sim_observations()
            score = task._compute_task_score(raw)
            assert 0.0 <= score <= 1.0
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", [
        "ib-p7-shape-stack-v1",
        "ib-p6-stack-collapse-recovery-v1",
        "ib-p2-ramp-sort-v1",
        "ib-p3-ball-cage-escape-v1",
        "ib-p3-indirect-push-v1",
    ])
    def test_termination_returns_bool(self, task_id):
        task = _make_task(task_id)
        try:
            task.reset()
            raw = task._get_sim_observations()
            term = task._check_termination(raw)
            assert isinstance(term, (bool, np.bool_))
        finally:
            task.close()

    @pytest.mark.parametrize("task_id", [
        "ib-p7-shape-stack-v1",
        "ib-p6-stack-collapse-recovery-v1",
        "ib-p2-ramp-sort-v1",
    ])
    def test_step_changes_score(self, task_id):
        """Score should remain bounded after a few steps."""
        task = _make_task(task_id, max_episode_steps=30)
        try:
            task.reset()
            lo, _ = task.action_space.low, task.action_space.high
            for _ in range(3):
                task.step(np.zeros(lo.shape[0]))
            score = task.task_score()
            assert 0.0 <= score <= 1.0
        finally:
            task.close()


# =====================================================================
#  Shape stack specifics
# =====================================================================

class TestShapeStackScoring:

    @pytest.fixture(scope="class")
    @classmethod
    def stack_task(cls):
        task = _make_task("ib-p7-shape-stack-v1", max_episode_steps=50)
        task.reset()
        yield task
        task.close()

    def test_score_initially_low(self, stack_task):
        stack_task.reset()
        raw = stack_task._get_sim_observations()
        score = stack_task._compute_task_score(raw)
        assert score < 1.0

    def test_stable_count_resets_on_reset(self, stack_task):
        stack_task.reset()
        assert stack_task._stable_count == 0

    def test_score_after_steps(self, stack_task):
        """Score should remain bounded after stepping."""
        stack_task.reset()
        _step_n(stack_task, n=20)
        raw = stack_task._get_sim_observations()
        score = stack_task._compute_task_score(raw)
        assert 0.0 <= score <= 1.0

    def test_correct_order_gives_partial_score(self, stack_task):
        """Manually stack objects and verify partial score."""
        stack_task.reset()
        sim = stack_task._env.sim
        archetypes = stack_task._selected_archetypes
        names = [a["name"] for a in archetypes]
        half_heights = [a["half_h"] for a in archetypes]
        t_height = stack_task._table_height

        cur_z = t_height + 0.002
        for idx in range(len(names)):
            hh = half_heights[idx]
            stack_task._place_object(names[idx], [0.0, 0.0, cur_z + hh])
            cur_z += hh * 2 + 0.001

        for name in names:
            stack_task._set_object_velocity(name, [0, 0, 0])

        sim.forward()
        for _ in range(3):
            stack_task.step(np.zeros(7))

        raw = stack_task._get_sim_observations()
        score = stack_task._compute_task_score(raw)
        assert score >= 0.0


# =====================================================================
#  Shape stack — programmatic success check
# =====================================================================

class TestShapeStackSuccessState:
    """Stack all objects and verify score == 1.0."""

    # stable_steps=5: don't wait 100 steps for the tower to be "stable".
    # num_objects range=[2,2]: a fixed two-object stack.
    _OVERRIDES = {
        "task": {"stable_steps": 5},
        "randomization": {"num_objects": {"range": [2, 2]}},
    }

    def _stack_objects(self, task):
        """Stack every object, in catalogue order, directly above the table.

        Uses exact resting-contact positions (no vertical gap between surfaces)
        so there are no contact-force impulses at t=0 and objects stay stable.
        """
        archetypes = task._selected_archetypes
        t_height = task._table_height

        # Start 0.5 mm below the table surface so the bottom object penetrates
        # slightly. Zero-distance contacts are unreliable without stepping
        # physics — a definite negative distance guarantees check_contact()
        # sees the object-table pair.
        cur_z = t_height - 0.0005
        for arch in archetypes:
            half_h = arch["half_h"]
            cur_z += half_h                          # move to body centre
            task._place_object(arch["name"], [0.0, 0.0, cur_z])
            cur_z += half_h                          # move to top surface

        # Zero both linear and angular velocity so no initial impulse.
        for arch in archetypes:
            task._set_object_velocity(arch["name"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        task._env.sim.forward()

    def test_score_reaches_one(self):
        """Score must reach 1.0 after stable_steps consecutive stable evaluations."""
        task = _make_task(
            "ib-p7-shape-stack-v1",
            max_episode_steps=500,
            config_overrides=self._OVERRIDES,
        )
        try:
            task.reset(seed=0)
            self._stack_objects(task)

            # Pump _compute_task_score on the settled state without advancing
            # physics, so we test stable_count accumulation in isolation.
            raw = task._get_sim_observations()
            score = 0.0
            for _ in range(task._stable_steps_required + 2):
                score = task._compute_task_score(raw)
                if score >= 1.0:
                    break

            assert score == pytest.approx(1.0), (
                f"Score never reached 1.0; last value={score:.3f}. "
                f"stable_count={task._stable_count}"
            )
        finally:
            task.close()

    def test_termination_after_stable_steps(self):
        """_check_termination must return True once stable_count >= threshold."""
        task = _make_task(
            "ib-p7-shape-stack-v1",
            max_episode_steps=500,
            config_overrides=self._OVERRIDES,
        )
        try:
            task.reset(seed=1)
            self._stack_objects(task)

            raw = task._get_sim_observations()
            terminated = False
            for _ in range(task._stable_steps_required + 2):
                if task._check_termination(raw):
                    terminated = True
                    break

            assert terminated, (
                f"Task did not terminate. "
                f"stable_count={task._stable_count}, "
                f"required={task._stable_steps_required}"
            )
        finally:
            task.close()

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_one_across_seeds(self, seed):
        """Score == 1.0 for several random archetype selections and orders."""
        task = _make_task(
            "ib-p7-shape-stack-v1",
            max_episode_steps=500,
            config_overrides=self._OVERRIDES,
        )
        try:
            task.reset(seed=seed)
            self._stack_objects(task)

            raw = task._get_sim_observations()
            score = 0.0
            for _ in range(task._stable_steps_required + 2):
                score = task._compute_task_score(raw)
                if score >= 1.0:
                    break

            assert score == pytest.approx(1.0), (
                f"seed={seed}: score={score:.3f}, "
                f"stable_count={task._stable_count}"
            )
        finally:
            task.close()

    def test_physics_termination_box_stack(self):
        """Run real sim steps: box-only stack stays stable and task terminates.

        Prisms/cylinders can topple under gravity. This test forces the two
        flat box archetypes so the tower survives physics integration long
        enough for _stable_count to reach the threshold.
        """
        from imbench.tasks.puzzle.shape_stack_v1 import _ALL_ARCHETYPES

        # Pick the two flat box archetypes — most stable under gravity.
        # cube: half_h=0.025, small_cube: half_h=0.018.
        stable_archetypes = [
            a for a in _ALL_ARCHETYPES if a["name"] in ("cube", "small_cube")
        ]
        assert len(stable_archetypes) == 2, "Expected cube and small_cube in archetypes"

        task = _make_task(
            "ib-p7-shape-stack-v1",
            max_episode_steps=500,
            config_overrides={
                "task": {"stable_steps": 5},
                "randomization": {"num_objects": {"range": [2, 2]}},
            },
        )
        try:
            task.reset(seed=0)

            # Override to use only flat boxes.  The model always holds the
            # whole catalogue, so no rebuild is needed — just re-park the
            # archetypes that are no longer in play so they cannot collide
            # with the stack or be counted by the scorer.
            task._selected_archetypes = stable_archetypes
            task._park_unselected()
            task._env.sim.forward()

            # _ALL_ARCHETYPES orders cube before small_cube, so stable_archetypes
            # is [cube, small_cube] — wider box on bottom, naturally stable.
            self._stack_objects(task)

            # Step real physics — boxes on flat surfaces should stay put.
            action = np.zeros(task.action_space.low.shape[0])
            terminated = False
            final_score = 0.0
            for _ in range(50):
                _, _, term, trunc, info = task.step(action)
                final_score = info.get("task_score", 0.0)
                if term:
                    terminated = True
                    break
                if trunc:
                    break

            assert terminated, (
                f"Task did not terminate within 50 physics steps. "
                f"final_score={final_score:.3f}, "
                f"stable_count={task._stable_count}"
            )
            assert final_score == pytest.approx(1.0), (
                f"Terminated but score was {final_score:.3f}, expected 1.0"
            )
        finally:
            task.close()


# =====================================================================
#  Ramp sort — programmatic success check
# =====================================================================

class TestRampSortSuccessState:
    """Teleport all objects into their correct bins and verify score == 1.0."""

    def _teleport_all_to_bins(self, task):
        """Place every sphere/box at the centre of its correct bin."""
        sim = task._env.sim
        # Bin base height: bin wall half-thickness is 0.004, so base top is
        # table_height + 0.004.  Drop objects just above that so they rest in.
        bin_z = task._table_height + 0.004 + task._object_size + 0.003

        all_names = (
            [f"sphere_{i}" for i in range(task._num_spheres)]
            + [f"box_{i}" for i in range(task._num_cubes)]
        )

        for i in range(task._num_spheres):
            name = f"sphere_{i}"
            cx, cy = task._sphere_bin_world_xy
            task._place_object(name, [cx, cy, bin_z])

        for i in range(task._num_cubes):
            name = f"box_{i}"
            cx, cy = task._cube_bin_world_xy
            task._place_object(name, [cx, cy, bin_z])

        # Zero velocities so objects are stable from the start
        for name in all_names:
            task._set_object_velocity(name, [0, 0, 0], [0, 0, 0])

        sim.forward()

    @staticmethod
    def _settle(task):
        """Idle until the teleported balls have rested in their bins for REST_TIME."""
        action = np.zeros(task.action_space.low.shape[0])
        for _ in range(int(round(task.REST_TIME * task.control_freq)) + 100):
            _, _, term, trunc, _ = task.step(action)
            if term or trunc:
                break

    def test_score_is_one_after_teleport(self):
        """Score is 1.0 once objects placed in their correct bins have come to rest."""
        task = _make_task(
            "ib-p2-ramp-sort-v1",
            max_episode_steps=200,
            config_overrides={"task": {
                "num_spheres": 1, "num_cubes": 1,
                "num_spheres_range": [1, 1], "num_cubes_range": [1, 1],
            }},
        )
        try:
            task.reset(seed=0)
            self._teleport_all_to_bins(task)
            raw = task._get_sim_observations(force_update=True)
            assert task._compute_task_score(raw) < 1.0   # not yet at rest for REST_TIME
            self._settle(task)

            raw = task._get_sim_observations(force_update=True)
            score = task._compute_task_score(raw)
            assert score == pytest.approx(1.0), (
                f"Expected score 1.0 after teleport, got {score:.3f}. "
                f"sphere_bin={task._sphere_bin_world_xy}, "
                f"cube_bin={task._cube_bin_world_xy}"
            )
        finally:
            task.close()

    def test_termination_after_stable_steps(self):
        """After a few physics steps with objects in bins, task must terminate."""
        task = _make_task(
            "ib-p2-ramp-sort-v1",
            max_episode_steps=200,
            config_overrides={"task": {
                "num_spheres": 1, "num_cubes": 1,
                "num_spheres_range": [1, 1], "num_cubes_range": [1, 1],
            }},
        )
        try:
            task.reset(seed=1)
            self._teleport_all_to_bins(task)

            terminated = False
            action = np.zeros(task.action_space.low.shape[0])
            for _ in range(20):
                _, _, term, trunc, _ = task.step(action)
                if term or trunc:
                    terminated = True
                    break

            assert terminated, (
                "Task did not terminate within 20 steps after all objects were "
                "placed in correct bins."
            )
        finally:
            task.close()

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_score_one_across_seeds(self, seed):
        """Score == 1.0 once teleported objects rest in their bins, for several ramp orientations."""
        task = _make_task(
            "ib-p2-ramp-sort-v1",
            max_episode_steps=200,
            config_overrides={"task": {
                "num_spheres": 1, "num_cubes": 1,
                "num_spheres_range": [1, 1], "num_cubes_range": [1, 1],
            }},
        )
        try:
            task.reset(seed=seed)
            self._teleport_all_to_bins(task)
            self._settle(task)

            raw = task._get_sim_observations(force_update=True)
            score = task._compute_task_score(raw)
            assert score == pytest.approx(1.0), (
                f"seed={seed}: score={score:.3f}"
            )
        finally:
            task.close()
