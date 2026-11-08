"""Stack Collapse Recovery — restack a tower of three cubes after it collapses.

Task ID: ``ib-p6-stack-collapse-recovery-v1``

Three cubes of slightly increasing side length spawn already stacked
largest-on-bottom. The stack is **always** knocked down a few control
steps into the episode: a fair coin flip decides whether the kick hits
only the top cube (the middle stays put) or both upper cubes
(middle + top fan apart).

Success criteria are borrowed exactly from ``ShapeStackV1``: exactly one
of the three cubes touches the table, no cube touches the gripper, and
the configuration holds for ``stable_steps`` consecutive ticks. The
restacking order is not scored. Nothing is scored until the collapse
has fired and every cube has come to rest.
"""

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward
from imbench.tasks.puzzle._stack_scoring import (
    object_body_id,
    count_table_and_gripper_contacts,
    stacking_progress,
    hold_duration_ramp,
)


# Three cubes, slightly increasing side length. ``half_h`` is the cube's
# half-side (cubes are isotropic, so it doubles as half-width).
_CUBES = [
    {"name": "cube_small",  "half_h": 0.020, "rgba": [0.85, 0.30, 0.25, 1.0]},
    {"name": "cube_medium", "half_h": 0.025, "rgba": [0.30, 0.70, 0.40, 1.0]},
    {"name": "cube_large",  "half_h": 0.030, "rgba": [0.25, 0.45, 0.85, 1.0]},
]


class StackCollapseRecoveryV1(SingleArmBase):
    TASK_ID = "ib-p6-stack-collapse-recovery-v1"
    DESCRIPTION = (
        "Puzzle: three cubes of increasing size start stacked on the "
        "table. The tower always collapses shortly after the episode "
        "begins — sometimes only the top cube is knocked off, sometimes "
        "both upper cubes. Restack the cubes into a single tower (any "
        "order) and hold it stable."
    )

    # ── Config parsing ──────────────────────────────────────────────

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._stable_steps_required = int(task_cfg.get("stable_steps", 100))

        collapse_cfg = task_cfg.get("collapse", {})
        # Probability (per episode) that the collapse hits **both** upper
        # cubes; the complement only kicks the top cube. The collapse
        # itself fires every episode — there is no "no collapse" branch.
        self._collapse_both_prob = float(collapse_cfg.get("both_prob", 0.5))
        # When (in control-steps) the collapse fires after reset. A short
        # delay lets the cubes settle visually before the kick, so the
        # collapse looks like a perturbation rather than spawn jitter.
        self._collapse_step = int(collapse_cfg.get("step", 5))
        # Horizontal kick speeds (m/s). ``top_speed`` knocks the top cube
        # off in either mode; ``middle_speed`` must be high enough to slide
        # the middle cube (with the top one riding on it) off the bottom
        # cube in "both" mode without launching it off the table.
        self._collapse_top_speed = float(collapse_cfg.get("top_speed", 0.9))
        self._collapse_middle_speed = float(collapse_cfg.get("middle_speed", 1.1))

        # Per-episode placement jitter for the base of the tower.
        place_cfg = task_cfg.get("placement", {})
        self._tower_xy_range = place_cfg.get("xy_range", [0.08, 0.10])

        # Bookkeeping reset in reset()
        self._stable_count = 0
        self._failed = False
        self._collapse_mode = "top"   # "top" or "both" — chosen each reset
        self._collapse_fired = False
        self._collapse_settled = False
        self._tower_xy = np.zeros(2)
        self._num_touching_table = len(_CUBES)
        self._gripper_contact_now = False

        # Same base + ramp pattern as the hold-style tasks
        # (medium_base_balance_*): dense stacking progress plus a
        # gripper-clear check are both instantaneous (non-sticky), and
        # the hold-duration ramp is what actually requires sustaining the
        # recovered arrangement for `_stable_steps_required` ticks. Nothing
        # is scored before the collapse has fired and settled: the intact
        # tower and a cube in flight would otherwise count as stacked.
        self._staged_reward = StagedReward(
            stages=[
                Stage("recovery_progress", 0.5, lambda t, o: t._recovery_progress(), sticky=False),
                # Paid only for a released, recovered stack, never for an untouched scene.
                Stage("gripper_clear", 0.1, lambda t, o: 1.0 if t._is_recovered() else 0.0, sticky=False),
                Stage("hold_duration_ramp", 0.4,
                      lambda t, o: hold_duration_ramp(t._stable_count, t._stable_steps_required)
                      if t._is_recovered() else 0.0,
                      sticky=False),
            ],
            success_fn=lambda t, o: t._stable_count >= t._stable_steps_required,
        )

    # ── Object specs ────────────────────────────────────────────────

    def _make_objects(self):
        objects = []
        for c in _CUBES:
            h = c["half_h"]
            objects.append({
                "type": "box",
                "name": c["name"],
                "size": [h, h, h],
                "rgba": c["rgba"],
                "density": 500,
                "friction": [0.85, 0.005, 0.0001],
                "joints": "default",
            })
        return objects

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._stable_count = 0
        self._failed = False
        self._collapse_fired = False
        self._collapse_settled = False
        self._num_touching_table = len(_CUBES)
        self._gripper_contact_now = False
        self._staged_reward.reset()
        # Decide once per episode which cubes the kick will hit. The
        # collapse always fires; the only randomness is "top only" vs
        # "both upper cubes", split by ``collapse_both_prob``.
        self._collapse_mode = (
            "both" if self._rng.rand() < self._collapse_both_prob else "top"
        )

        # Tighten the MuJoCo solver — same rationale as ShapeStackV1:
        # robosuite defaults are loose for multi-body stacks and the
        # contacts buzz noticeably without these.
        mj_opt = self._env.sim.model.opt
        mj_opt.timestep = 0.002
        mj_opt.iterations = 150
        mj_opt.noslip_iterations = 3
        mj_opt.noslip_tolerance = 1e-6

        # Sample a single XY for the bottom cube; the upper two stack
        # directly on top so the whole tower starts aligned.
        xr, yr = self._tower_xy_range
        if isinstance(xr, (int, float)):
            xr = [-xr, xr]
        if isinstance(yr, (int, float)):
            yr = [-yr, yr]
        tx = float(self._rng.uniform(xr[0], xr[1]))
        ty = float(self._rng.uniform(yr[0], yr[1]))
        self._tower_xy = np.array([tx, ty])

        # Stack largest-on-bottom. Cube i sits with its centre at
        # (tower_z_so_far + half_h_i); after placement, advance the
        # cursor by 2 * half_h_i so the next cube rests on top.
        if self._rand_object_color:
            self._randomize_cube_colors()

        order = sorted(range(len(_CUBES)), key=lambda i: -_CUBES[i]["half_h"])
        z_cursor = self._table_height + 0.001
        for idx in order:
            c = _CUBES[idx]
            h = c["half_h"]
            self._place_object(c["name"], [tx, ty, z_cursor + h])
            self._set_object_velocity(c["name"], [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0])
            z_cursor += 2.0 * h

        self._env.sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ── Collapse perturbation ───────────────────────────────────────

    def _trigger_collapse(self):
        """Knock the tower over by velocity-kicking the upper cube(s).

        The bottom cube (``cube_large``) is always left alone so the
        recovery target stays anchored on the table. ``_collapse_mode``
        chooses what gets hit:

        - ``"top"``  — only ``cube_small`` (the top of the tower) is
          kicked off; the middle cube stays seated on the bottom one.
        - ``"both"`` — ``cube_medium`` and ``cube_small`` are both kicked
          off the bottom cube, in directions ~120° apart so they diverge
          instead of moving as a unit.
        """
        theta = float(self._rng.uniform(0.0, 2.0 * np.pi))

        if self._collapse_mode == "top":
            # Only the top cube moves. Direction is fully random.
            vx_top = self._collapse_top_speed * np.cos(theta)
            vy_top = self._collapse_top_speed * np.sin(theta)
            self._set_object_velocity(
                "cube_small", [vx_top, vy_top, 0.10], [0.0, 0.0, 0.0]
            )
            return

        # "both" — middle cube along theta, top cube ~120° offset so
        # the kicks diverge.
        vx_mid = self._collapse_middle_speed * np.cos(theta)
        vy_mid = self._collapse_middle_speed * np.sin(theta)
        theta_top = theta + (2.0 * np.pi / 3.0)
        vx_top = self._collapse_top_speed * np.cos(theta_top)
        vy_top = self._collapse_top_speed * np.sin(theta_top)
        self._set_object_velocity(
            "cube_medium", [vx_mid, vy_mid, 0.05], [0.0, 0.0, 0.0]
        )
        self._set_object_velocity(
            "cube_small", [vx_top, vy_top, 0.10], [0.0, 0.0, 0.0]
        )

    def _randomize_cube_colors(self):
        """Paint every cube a fresh random colour each reset.

        Each of the three cubes gets an independent draw so the policy
        can visually distinguish them (otherwise they'd all share one
        random colour, which carries less information than the fixed
        red/green/blue defaults). Walks each cube body's geom range
        and overwrites ``geom_rgba`` directly — robosuite ignores the
        per-spec rgba once the model is built.
        """
        sim = self._env.sim
        for c in _CUBES:
            try:
                bid = sim.model.body_name2id(c["name"] + "_main")
            except Exception:
                continue
            geom_start = sim.model.body_geomadr[bid]
            geom_count = sim.model.body_geomnum[bid]
            rgba = self._random_object_color()
            for gi in range(geom_start, geom_start + geom_count):
                sim.model.geom_rgba[gi] = rgba

    def step(self, action: np.ndarray):
        self._check_open()
        # Fire the collapse exactly once, on the configured step. Use
        # the pre-step counter so the kick is applied *before* the next
        # physics step rolls forward — robosuite advances the sim inside
        # super().step(), so velocities set here are integrated by it.
        if (not self._collapse_fired
                and self._step_count >= self._collapse_step):
            self._trigger_collapse()
            self._collapse_fired = True

        obs, reward, terminated, truncated, info = super().step(action)
        info["collapse_mode"] = self._collapse_mode
        info["collapse_fired"] = self._collapse_fired
        return obs, reward, terminated, truncated, info

    # ── Model introspection ────────────────────────────────────────
    # These are direct ports of the helpers in ShapeStackV1: needed
    # because the scorer walks each cube's geom subtree to count
    # table contacts and gripper contacts.

    def _object_body_id(self, name: str) -> int | None:
        return object_body_id(self._env.sim, name)

    def _any_object_fell(self) -> bool:
        sim = self._env.sim
        for c in _CUBES:
            bid = self._object_body_id(c["name"])
            if bid is None:
                continue
            if sim.data.body_xpos[bid][2] < self._failure_floor_threshold:
                return True
        return False

    # ── Scoring ──────────────────────────────────────────────────────
    # Stacking-state signals (num_touching_table / gripper contact) are
    # shared with ShapeStackV1 via ``imbench.tasks.puzzle._stack_scoring``;
    # the staged weighting on top is new (see class docstring /
    # notes/staged_rewards_design.md §3.1 P6).

    def _update_stack_contact_state(self):
        """Refresh ``_num_touching_table`` / ``_gripper_contact_now`` for
        this tick, latch ``_collapse_settled`` once every cube is at rest
        after the collapse, and advance ``_stable_count`` accordingly."""
        self._num_touching_table, self._gripper_contact_now = (
            count_table_and_gripper_contacts(self._env, [c["name"] for c in _CUBES])
        )
        if self._collapse_fired and not self._collapse_settled:
            self._collapse_settled = all(
                self._object_speed(c["name"]) < self.REST_SPEED for c in _CUBES
            )
        if self._is_recovered():
            self._stable_count += 1
        else:
            self._stable_count = 0

    def _is_recovered(self) -> bool:
        """True iff the collapse has settled, exactly one cube touches the
        table and the gripper is clear — the instantaneous (pre-hold)
        success precondition."""
        return (self._collapse_settled and self._num_touching_table == 1
                and not self._gripper_contact_now)

    def _recovery_progress(self) -> float:
        """Dense stacking progress: 0 before the collapse has settled or
        when every cube still touches the table (fully collapsed/scattered),
        1 when only one does (fully restacked)."""
        if not self._collapse_settled:
            return 0.0
        return stacking_progress(self._num_touching_table, len(_CUBES))

    def _compute_task_score(self, raw_obs):
        if getattr(self, "_failed", False):
            return 0.0

        self._update_stack_contact_state()
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if getattr(self, "_failed", False):
            return True
        if self._any_object_fell():
            self._failed = True
            return True
        score = self._compute_task_score(raw_obs)
        return score >= self._success_score_threshold
