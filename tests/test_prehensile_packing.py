"""Tests for the single-arm suction packing task (ib-p7-packing-v1)."""

import numpy as np
import pytest

import imbench as ib


# ── Packing V1 ──────────────────────────────────────────────────────


class TestPackingV1:
    """Tests for the Packing puzzle task."""

    TASK_ID = "ib-p7-packing-v1"

    @pytest.fixture(scope="class")
    @classmethod
    def env(cls):
        env = ib.make(cls.TASK_ID, obs_mode="combo", render_mode=None, seed=0)
        yield env
        env.close()

    def test_in_registry(self):
        assert self.TASK_ID in ib.list_tasks()

    def test_make_and_reset(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert info["task_score"] == 0.0

    def test_obs_keys_single_arm(self, env):
        obs, _ = env.reset()
        for suffix in ["eef_pos", "eef_quat", "gripper_qpos",
                       "joint_pos", "joint_vel"]:
            key = f"robot0_{suffix}"
            assert key in obs, f"Missing key: {key}"

    def test_action_space_7dof(self, env):
        lo, hi = env.action_space.low, env.action_space.high
        assert lo.shape[0] == 7

    def test_max_episode_steps(self, env):
        assert env.MAX_EPISODE_STEPS == env.cfg["max_episode_steps"]

    def test_task_conditions_parsed(self, env):
        tc = env.cfg["task_conditions"]
        assert env._success_score_threshold == tc["success"]["score_threshold"]
        assert env._failure_floor_threshold == tc["failure"]["floor_z_threshold"]

    def test_table_config(self, env):
        cfg_table = env.cfg["scene"]["table"]
        assert env._table_height == cfg_table["height"]
        assert env._table_full_size == tuple(cfg_table["full_size"])
        np.testing.assert_allclose(env._env.table_full_size, cfg_table["full_size"], atol=1e-6)

    def test_l_pieces_in_configured_range(self, env):
        env.reset()
        lo, hi = env._piece_count_range
        assert lo <= len(env._blocks) <= hi
        assert len(env._block_names) == len(env._blocks)

    def test_box_fixture_created(self, env):
        env.reset()
        assert env._box is not None

    def test_task_score_bounded(self, env):
        env.reset()
        score = env.task_score()
        assert 0.0 <= score <= 1.0

    def test_step_random_action(self, env):
        env.reset()
        lo, hi = env.action_space.low, env.action_space.high
        action = np.random.uniform(lo, hi)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert 0.0 <= info["task_score"] <= 1.0

    def test_block_positions_available(self, env):
        env.reset()
        positions = env._get_block_positions()
        lo, hi = env._piece_count_range
        assert lo <= len(positions) <= hi
        for name, pos in positions.items():
            assert pos.shape == (3,)

    def test_termination_checks_floor_drop(self, env):
        env.reset()
        raw_obs = env._get_sim_observations()
        # Blocks should be on the table, no termination
        assert env._check_termination(raw_obs) is False

    def test_light_randomization_constrained(self, env):
        env._rand_light_color = True
        env.reset(seed=10)
        sim = env._env.sim
        for i in range(sim.model.nlight):
            diffuse = sim.model.light_diffuse[i]
            assert all(c >= 0.6 for c in diffuse)


# ── Cross-task tests ────────────────────────────────────────────────


@pytest.mark.slow
class TestPackingCrossTask:
    """Lifecycle / sanity checks for the packing task across obs modes."""

    TASK_IDS = [
        "ib-p7-packing-v1",
    ]

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_make_reset_step_close(self, task_id):
        """Basic lifecycle: make → reset → step → close."""
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=0)
        try:
            obs, info = env.reset()
            assert isinstance(obs, dict)
            lo, _ = env.action_space.low, env.action_space.high
            obs, reward, terminated, truncated, info = env.step(np.zeros(lo.shape[0]))
            assert isinstance(obs, dict)
            assert "task_score" in info
        finally:
            env.close()

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_obs_mode_combo_has_cameras(self, task_id):
        """combo obs_mode should include RGB camera images."""
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=0)
        try:
            obs, _ = env.reset()
            rgb_keys = [k for k in obs if k.startswith("rgb_")]
            assert len(rgb_keys) > 0, f"No RGB keys in combo mode for {task_id}"
            for key in rgb_keys:
                assert obs[key].ndim == 3  # (H, W, C)
        finally:
            env.close()

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_task_score_in_valid_range_after_steps(self, task_id):
        """task_score stays in [0, 1] over several steps."""
        env = ib.make(task_id, obs_mode="combo", render_mode=None, seed=0)
        try:
            env.reset()
            lo, hi = env.action_space.low, env.action_space.high
            for _ in range(5):
                action = np.random.uniform(lo, hi)
                _, _, _, _, info = env.step(action)
                assert 0.0 <= info["task_score"] <= 1.0
        finally:
            env.close()


# =====================================================================
#  Programmatic success — teleport objects to goal states
# =====================================================================


class TestPackingSuccess:

    @pytest.fixture(scope="class")
    @classmethod
    def env(cls):
        e = ib.make("ib-p7-packing-v1", obs_mode="combo", render_mode=None, seed=42)
        e.reset()
        yield e
        e.close()

    def test_place_all_blocks_flat_in_box(self, env):
        env.reset()
        sim = env._env.sim

        box_bid = sim.model.body_name2id(env._box.root_body)
        box_pos = sim.data.body_xpos[box_bid].copy()

        # Use the BPP centroid positions so pieces are guaranteed non-overlapping.
        # Each piece's centroid is in the bin's 2D frame (relative to bin centre).
        # The block body-frame origin coincides with the piece's bottom face (the
        # L-piece geometry is defined with geom centre at z = _PIECE_THICKNESS/2
        # above the origin), so setting body_z = floor_z rests the piece on the
        # bin floor.
        from imbench.tasks.prehensile.packing_v1 import (
            _BOX_BASE_THICK, _PIECE_SETTLED_CONTACT_STEPS,
        )
        floor_z = box_pos[2] + _BOX_BASE_THICK
        piece_z = floor_z + 0.001  # 1 mm clearance to avoid initial penetration

        for block, piece in zip(env._blocks, env._pieces):
            cx, cy = piece["centroid"]
            pos = np.array([box_pos[0] + cx, box_pos[1] + cy, piece_z])
            quat = np.array([1.0, 0.0, 0.0, 0.0])  # flat, identity rotation
            sim.data.set_joint_qpos(block.joints[0], np.concatenate([pos, quat]))
        sim.forward()

        # _compute_task_score requires _bin_contact_streak >= _PIECE_SETTLED_CONTACT_STEPS.
        # Step the simulation long enough for the contact streaks to accumulate
        # while the blocks rest at their BPP-solution positions inside the bin.
        lo, _ = env.action_space.low, env.action_space.high
        action = np.zeros(lo.shape[0])
        for _ in range(_PIECE_SETTLED_CONTACT_STEPS + 10):
            env.step(action)

        raw_obs = env._get_sim_observations(force_update=True)
        score = env._compute_task_score(raw_obs)
        assert score == 1.0, f"Expected 1.0, got {score}"
        assert env._check_termination(raw_obs) is True


class TestPackingSingleLayer:
    """A piece counts as packed only resting on the bin floor."""

    def test_pieces_stacked_two_high_are_not_packed(self):
        from imbench.tasks.prehensile.packing_v1 import (
            _BOX_BASE_THICK, _BOX_INNER_HALF_X, _PIECE_SETTLED_CONTACT_STEPS, _PIECE_THICKNESS,
        )
        env = ib.make("ib-p7-packing-v1", obs_mode="proprio", render_mode=None, seed=0)
        try:
            # A 4-piece seed where both stacked pieces keep settled contact
            # with a wall in the snug bin (on some seeds a small top piece
            # is nudged off the wall and the contact check below is moot).
            env.reset(seed=10)
            assert len(env._blocks) == 4
            sim = env._env.sim
            box_pos = sim.data.body_xpos[sim.model.body_name2id(env._box.root_body)].copy()
            floor_z = box_pos[2] + _BOX_BASE_THICK
            # Pieces 0 and 1 on the floor, pressed against the -x and +x walls;
            # pieces 2 and 3 lying flat on top of them.
            for k, (block, piece) in enumerate(zip(env._blocks, env._pieces)):
                left, right, down, up = env._piece_bbox_directional(piece)
                if k % 2 == 0:
                    x = -_BOX_INNER_HALF_X + left - 0.001
                else:
                    x = _BOX_INNER_HALF_X - right + 0.001
                # Centre the bounding box, not the centroid, across the bin:
                # an L's centroid is off-centre and the bin is snug in y.
                y = (down - up) / 2
                z = floor_z + 0.0005 + (k // 2) * (_PIECE_THICKNESS + 0.0005)
                sim.data.set_joint_qpos(block.joints[0],
                                        np.r_[box_pos[0] + x, box_pos[1] + y, z, 1.0, 0.0, 0.0, 0.0])
            sim.forward()

            for _ in range(_PIECE_SETTLED_CONTACT_STEPS + 20):
                *_, info = env.step(np.zeros(env.action_space.shape))
                assert info["is_success"] is False

            heights = [pos[2] - floor_z for pos in env._get_block_positions().values()]
            assert max(heights[:2]) < 0.005 < _PIECE_THICKNESS - 0.005 < min(heights[2:])
            assert all(env._bin_contact_streak[name] >= _PIECE_SETTLED_CONTACT_STEPS
                       for name in env._block_names)
            placed = [env._piece_placed_flat_progress(name) for name in env._block_names]
            assert [env._piece_in_box_flat(name) for name in env._block_names] == [True, True, False, False]
            # Even the floor-level pieces are against walls, away from
            # their generated target slots: placement reward is now zero.
            assert placed == [0.0, 0.0, 0.0, 0.0]
        finally:
            env.close()


# ── Grasp anchor (BPP-side metadata) ───────────────────────────────────
# These tests exercise the piece-generation logic directly without
# spinning up a robosuite env — fast, deterministic, and they cover many
# more seeds than the env-level fixtures can.


class TestGraspAnchor:
    """The ``grasp_anchor`` field on each piece must point inside one of
    its sub-boxes with enough boundary clearance for a circular suction
    snout, and must NOT land on the L-piece centroid (which sits at the
    inside corner — empty space)."""

    # Suction snout radius (panda_suction.xml: geo_snout cylinder size
    # "0.015 0.001"). The anchor must sit at least this far from any
    # boundary of its sub-box for the snout to fully cover material.
    SNOUT_RADIUS = 0.015

    @staticmethod
    def _anchor_inside_subbox(anchor, subbox, margin=0.0):
        """True iff ``anchor`` lies inside ``subbox`` (cx, cy, lx, ly)
        with at least ``margin`` clearance to each edge."""
        ax, ay = anchor
        cx, cy, lx, ly = subbox
        return (abs(ax - cx) <= lx / 2 - margin
                and abs(ay - cy) <= ly / 2 - margin)

    def test_grasp_anchor_is_a_subbox_centre(self):
        """Anchor must coincide with one of the two sub-box centres."""
        from imbench.tasks.prehensile.packing_v1 import _generate_l_pieces

        for seed in range(50):
            for n_pieces in (3, 4):
                rng = np.random.RandomState(seed)
                pieces = _generate_l_pieces(rng, n_pieces)
                for k, piece in enumerate(pieces):
                    assert "grasp_anchor" in piece, (
                        f"seed={seed} n={n_pieces} piece={k}: missing grasp_anchor")
                    ax, ay = piece["grasp_anchor"]
                    sub_a, sub_b = piece["subboxes"]
                    matches_a = abs(ax - sub_a[0]) < 1e-9 and abs(ay - sub_a[1]) < 1e-9
                    matches_b = abs(ax - sub_b[0]) < 1e-9 and abs(ay - sub_b[1]) < 1e-9
                    assert matches_a or matches_b, (
                        f"seed={seed} n={n_pieces} piece={k}: anchor "
                        f"({ax:.4f},{ay:.4f}) is neither sub-box centre "
                        f"a=({sub_a[0]:.4f},{sub_a[1]:.4f}) "
                        f"b=({sub_b[0]:.4f},{sub_b[1]:.4f})")

    def test_grasp_anchor_is_not_centroid(self):
        """Anchor must not coincide with the (empty-space) L centroid."""
        from imbench.tasks.prehensile.packing_v1 import _generate_l_pieces

        for seed in range(50):
            for n_pieces in (3, 4):
                rng = np.random.RandomState(seed)
                pieces = _generate_l_pieces(rng, n_pieces)
                for k, piece in enumerate(pieces):
                    cx, cy = piece["centroid"]
                    ax, ay = piece["grasp_anchor"]
                    # Two sub-boxes of different size always have at
                    # least one centre offset from the area-weighted
                    # centroid by more than 1 mm. Use that as the
                    # threshold; equality would only happen if both
                    # sub-boxes were identical *and* concentric, which
                    # the BPP guillotine split can't produce.
                    dist = np.hypot(ax - cx, ay - cy)
                    assert dist > 1e-3, (
                        f"seed={seed} n={n_pieces} piece={k}: anchor "
                        f"sits on the centroid (dist={dist:.6f}) — "
                        f"would be empty space inside the L")

    def test_grasp_anchor_clears_snout_radius(self):
        """Anchor must have ≥ snout_radius clearance to its sub-box edges."""
        from imbench.tasks.prehensile.packing_v1 import _generate_l_pieces

        failures = []
        total = 0
        for seed in range(50):
            for n_pieces in (3, 4):
                rng = np.random.RandomState(seed)
                pieces = _generate_l_pieces(rng, n_pieces)
                for k, piece in enumerate(pieces):
                    total += 1
                    ax, ay = piece["grasp_anchor"]
                    # Find which sub-box the anchor sits on.
                    sub_a, sub_b = piece["subboxes"]
                    if abs(ax - sub_a[0]) < 1e-9 and abs(ay - sub_a[1]) < 1e-9:
                        host = sub_a
                    else:
                        host = sub_b
                    # Anchor at sub-box centre → margin to each edge is
                    # half-extent. Both half-extents must clear snout.
                    hx, hy = host[2] / 2, host[3] / 2
                    if hx < self.SNOUT_RADIUS or hy < self.SNOUT_RADIUS:
                        failures.append(
                            f"seed={seed} n={n_pieces} piece={k}: "
                            f"host sub-box half-extents=({hx:.4f},{hy:.4f}) "
                            f"< snout_radius={self.SNOUT_RADIUS}")
        # Allow a small fraction of failures because _MIN_EDGE (= 0.036
        # full / 0.018 half) is below SNOUT_RADIUS — some BPP splits
        # produce sub-boxes too thin in one dim for a perfectly-centred
        # snout. The "max-min-edge" heuristic picks whichever sub-box
        # is blockier, but if BOTH are thin in one dim, neither clears.
        # Acceptable rate: < 5% of generated pieces.
        rate = len(failures) / total
        assert rate < 0.05, (
            f"{len(failures)}/{total} pieces ({rate:.1%}) had a "
            f"grasp anchor that didn't clear the snout radius. "
            f"First few: {failures[:5]}")

    def test_grasp_anchor_picks_blockier_subbox(self):
        """Heuristic check: chosen sub-box has ≥ the min-edge of the
        other (i.e. we picked the blockier one or they tied)."""
        from imbench.tasks.prehensile.packing_v1 import _generate_l_pieces

        for seed in range(50):
            for n_pieces in (3, 4):
                rng = np.random.RandomState(seed)
                pieces = _generate_l_pieces(rng, n_pieces)
                for k, piece in enumerate(pieces):
                    ax, ay = piece["grasp_anchor"]
                    sub_a, sub_b = piece["subboxes"]
                    if abs(ax - sub_a[0]) < 1e-9 and abs(ay - sub_a[1]) < 1e-9:
                        chosen, other = sub_a, sub_b
                    else:
                        chosen, other = sub_b, sub_a
                    chosen_min = min(chosen[2], chosen[3])
                    other_min = min(other[2], other[3])
                    assert chosen_min >= other_min - 1e-12, (
                        f"seed={seed} n={n_pieces} piece={k}: chosen "
                        f"min-edge={chosen_min:.4f} < other "
                        f"min-edge={other_min:.4f}")


@pytest.mark.parametrize("seed", [0, 1])
def test_scripted_policy_packs_every_piece_in_the_snug_bin(seed):
    """The scripted policy aligns each held piece on its slot and packs all of them (3 pieces on seed 0, 4 on seed 1)."""
    import contextlib
    import io
    from data_synthetic.single_arm_pipeline.packing_policy import PackingPolicy

    env = ib.make("ib-p7-packing-v1", obs_mode="proprio", privileged_obs=True, render_mode=None)
    try:
        obs, _ = env.reset(seed=seed)
        policy = PackingPolicy()
        info = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(env.MAX_EPISODE_STEPS):
                obs, _, terminated, truncated, info = env.step(policy.act(obs, env=env))
                if terminated or truncated:
                    break
        assert info.get("is_success"), [env._piece_pose_errors(name) for name in env._block_names]
    finally:
        env.close()
