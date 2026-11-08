"""Pick-and-place a medium YCB object onto a green goal patch on the table.

Task ID: ``ib-p6-pick-place-ycb-v1``

One random YCB object (from a fresh pool of 8 medium, graspable shapes)
spawns on the table along with a 0.15 m x 0.15 m green goal patch. Both
the object spawn position and the goal patch position are sampled
uniformly within the robot's reachable workspace, with a 0.1 m slack
margin so the arm never has to fully extend. Success requires the
object to be placed inside the patch in x, y, AND z (resting on or
just above the patch) while in contact with the table, at rest and
released by the robot.

Moving-goal caveat: at reset we also sample N in [n_min, n_max]
(defaults 1-3). The goal patch then relocates N times during the
episode -- each relocation fires when the object has been lifted and
is approaching the current patch, forcing the policy to replan.
After the Nth relocation the goal stays put and success is evaluated
against that final position.
"""

import numpy as np

from imbench.tasks.prehensile.contacts import object_geom_ids, robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


def _quat_mul(q1, q2):
    """Hamilton product of two [w, x, y, z] quaternions -> q1 o q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# Per-object spawn-orientation overrides: rotate so the shortest local
# axis becomes vertical, giving the gripper a stable top-down grasp.
_AXIS_Y_90 = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]
_AXIS_X_90 = [float(np.cos(np.pi / 4)), float(np.sin(np.pi / 4)), 0.0, 0.0]

DEFAULT_SPAWN_POSE = {
    # large_marker: long thin cylinder (0.019 x 0.121 x 0.019). Lay it
    # flat by rotating 90 deg about y so the short axis is vertical.
    "large_marker": _AXIS_Y_90,
    # medium_clamp: flat profile (0.077 x 0.085 x 0.026) with its
    # shortest axis already z -> no rotation.
}


# Fresh YCB pool (7 medium, well-graspable objects not used elsewhere
# in the codebase). Verified via mesh bounding-box: shortest bounding
# axis of each is <= 0.056 m, well inside the Franka parallel-jaw
# gripper's ~0.08 m max opening.
#
# `b_colored_wood_blocks` is intentionally excluded: its mesh is a
# bundle of 5 small coloured cubes welded into one rigid body, which
# renders as a cluster of cubes (notably a blue one) and reads to the
# user as multiple objects spawning at once.
DEFAULT_YCB_POOL = [
    "foam_brick",             # brick, min axis 0.051 m
    "large_marker",           # cylinder, min axis 0.019 m
    "padlock",                # compact, min axis 0.029 m
    "racquetball",            # sphere, 0.056 m
    "medium_clamp",           # tool, min axis 0.026 m
    "plum",                   # sphere, 0.053 m
    "golf_ball",              # sphere, 0.043 m
]


class PickPlaceYCBV1(SingleArmBase):
    TASK_ID = "ib-p6-pick-place-ycb-v1"
    DESCRIPTION = (
        "Single arm: a random medium YCB object and a 0.15 m x 0.15 m "
        "green goal patch spawn at random positions within the robot's "
        "reachable workspace. Pick the object and place it inside the "
        "patch (x, y, and z). The goal relocates 1-3 times mid-episode "
        "each time the object approaches it, forcing the policy to "
        "replan."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._ycb_pool = list(task_cfg.get("ycb_pool", DEFAULT_YCB_POOL))

        # Green patch is a square of side `goal_side`; success tolerance
        # in z is `goal_z_tol` above the patch surface.
        self._goal_side = float(task_cfg.get("goal_side", 0.15))
        self._goal_half = self._goal_side / 2.0
        self._goal_z_tol = float(task_cfg.get("goal_z_tol", 0.05))

        # Reachable workspace bounds (table-center frame). The Franka on
        # this table reaches roughly x in [-0.30, +0.35], y in [-0.32,
        # +0.32] comfortably. `slack` trims that inward so neither the
        # object nor the goal ever land where the arm has to fully
        # extend.
        self._reach_x = tuple(task_cfg.get("reach_x_range", [-0.30, 0.35]))
        self._reach_y = tuple(task_cfg.get("reach_y_range", [-0.32, 0.32]))
        self._slack = float(task_cfg.get("slack", 0.10))

        # Minimum horizontal separation between object spawn and goal
        # center. Keeps the task non-trivial (the object never starts
        # already inside the patch).
        self._min_obj_goal_dist = float(
            task_cfg.get("min_obj_goal_dist", 0.15)
        )

        # ── Moving-goal caveat ──────────────────────────────────────
        # Once per episode we sample N in [n_min, n_max] (inclusive).
        # The goal patch then relocates N times during the episode.
        # Each relocation fires when the object has been lifted and is
        # approaching the current goal (within `approach_radius` of it)
        # -- forcing the policy to replan mid-transport. After the Nth
        # relocation the goal stays put; only then do placement and
        # success count.
        mg_cfg = task_cfg.get("moving_goal", {})
        self._mg_n_min = int(mg_cfg.get("n_min", 1))
        self._mg_n_max = int(mg_cfg.get("n_max", 3))
        self._mg_approach_radius = float(mg_cfg.get("approach_radius", 0.08))
        # Minimum xy distance the new goal center must sit from the old
        # one, so the relocation is noticeable and demands replanning.
        self._mg_min_shift = float(mg_cfg.get("min_shift", 0.15))
        # Lift (m above the object's resting height) required for an
        # approach to count -- avoids firing when the object is just
        # sliding past. Also the lift that sets ``_was_lifted``.
        self._mg_min_lift = float(mg_cfg.get("min_lift", 0.03))

        self._goal_pos = np.zeros(2)
        self._mg_total = 0          # total relocations this episode (sampled each reset)
        self._mg_remaining = 0      # relocations left before goal is final
        self._task_done = False
        self._was_lifted = False
        # Active object's resting centre height, recorded at reset.
        self._rest_z = self._table_height

        # Placement and settling only count inside the final goal patch,
        # so an object left resting where it spawned, or set down before
        # every relocation has fired, earns nothing for them.
        self._staged_reward = StagedReward(
            stages=[
                Stage("object_lifted", 0.2, lambda t, o: 1.0 if t._was_lifted else 0.0),
                Stage("xy_over_goal", 0.3, lambda t, o: float(t._in_xy(o))),
                Stage("z_placement", 0.3, lambda t, o: float(t._in_xy(o) and t._in_z(o))),
                Stage("settled_at_rest", 0.2,
                      lambda t, o: float(t._in_xy(o) and t._in_z(o) and t._placed_and_released())),
            ],
            success_fn=lambda t, o: (
                t._in_xy(o) and t._in_z(o) and t._placed_and_released() and t._was_lifted
            ),
        )

        self._selected_ycb = self._ycb_pool[
            self._rng.randint(len(self._ycb_pool))
        ]

    # ── Naming helpers ──────────────────────────────────────────────

    def _ycb_obj_name(self, ycb_name: str) -> str:
        return f"ycb_{ycb_name}"

    def _active_obj_name(self) -> str:
        return self._ycb_obj_name(self._selected_ycb)

    def _inject_active_pose(self, raw_obs):
        """Point ``cube_pos`` / ``cube_quat`` at the active YCB's pose.

        The env's default alias points to the FIRST dynamic object,
        which isn't necessarily the active one once we bake the entire
        pool into the scene. Idempotent; the scorers call it too so
        ``task_score(recompute=True)`` scores the active object.
        """
        active = self._active_obj_name()
        pos_key = f"{active}_pos"
        quat_key = f"{active}_quat"
        if pos_key in raw_obs:
            raw_obs["cube_pos"] = raw_obs[pos_key]
        if quat_key in raw_obs:
            raw_obs["cube_quat"] = raw_obs[quat_key]

    def _get_obs(self, raw_obs):
        self._inject_active_pose(raw_obs)
        return super()._get_obs(raw_obs)

    # ── Object / static factories ───────────────────────────────────

    def _make_objects(self):
        # Bake every YCB in the pool once. On each reset we teleport the
        # chosen one onto the table and park the rest off-scene -- avoids
        # the slow close-and-rebuild cycle.
        return [
            {"type": "ycb", "name": self._ycb_obj_name(ycb),
             "ycb_name": ycb,
             "density": 200,
             "joints": "default"}
            for ycb in self._ycb_pool
        ]

    def _make_static_objects(self):
        return [
            {"type": "box", "name": "goal_marker",
             "size": [self._goal_half, self._goal_half, 0.001],
             "rgba": [0.0, 0.8, 0.0, 0.4],
             "density": 1,
             "friction": [0.1, 0.005, 0.0001],
             "joints": None},
        ]

    # ── Helpers ─────────────────────────────────────────────────────

    def _parked_objects(self):
        active = self._active_obj_name()
        return {self._ycb_obj_name(y) for y in self._ycb_pool} - {active}

    def _swap_active_ycb(self):
        """Pick a new active YCB and park all the others off-scene."""
        self._selected_ycb = self._ycb_pool[
            self._rng.randint(len(self._ycb_pool))
        ]
        self._park_objects(self._parked_objects())

    def _sampling_bounds(self):
        """Shrink reachable bounds by `slack` to keep the arm comfortable."""
        xlo = self._reach_x[0] + self._slack
        xhi = self._reach_x[1] - self._slack
        ylo = self._reach_y[0] + self._slack
        yhi = self._reach_y[1] - self._slack
        return xlo, xhi, ylo, yhi

    def _relocate_goal(self):
        """Move the goal patch to a new random spot at least
        ``_mg_min_shift`` away from its current location (and still
        inside the trimmed reach box). Decrements the remaining-
        relocation counter."""
        xlo, xhi, ylo, yhi = self._sampling_bounds()
        old = self._goal_pos.copy()
        new_xy = old
        for _ in range(40):
            cx = self._rng.uniform(xlo, xhi)
            cy = self._rng.uniform(ylo, yhi)
            if np.hypot(cx - old[0], cy - old[1]) >= self._mg_min_shift:
                new_xy = np.array([cx, cy])
                break
        self._goal_pos = new_xy
        self._place_static(
            "goal_marker",
            [float(new_xy[0]), float(new_xy[1]), self._table_height + 0.001],
        )
        self._mg_remaining -= 1

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Refresh controller goal poses so OSC doesn't carry last
        # episode's target into step 1 of the new episode.
        cc = self._env.robots[0].composite_controller
        for ctl in cc.part_controllers.values():
            if hasattr(ctl, "update"):
                try:
                    ctl.update(force=True)
                except TypeError:
                    try:
                        ctl.update()
                    except Exception:
                        pass
                except Exception:
                    pass
            if hasattr(ctl, "reset_goal"):
                try:
                    ctl.reset_goal()
                except Exception:
                    pass

        self._task_done = False
        self._was_lifted = False
        self._staged_reward.reset()

        # Use the base class's `self._rng` for all per-episode sampling
        # below: seeded from `self.seed` only when `reset(seed=...)` is
        # called with an explicit seed, so `reset(seed=7)` reproduces
        # identical draws every call — required by the eval harness's
        # "K repeats per seed, identical reset" protocol — while repeated `reset(seed=None)`
        # calls (the normal RL rollout pattern) still vary, since the RNG
        # keeps advancing instead of being pinned to one draw forever.
        rng = self._rng

        # Sample how many times the goal relocates this episode.
        self._mg_total = int(self._rng.randint(self._mg_n_min,
                                                self._mg_n_max + 1))
        self._mg_remaining = self._mg_total

        self._swap_active_ycb()
        active_name = self._ycb_obj_name(self._selected_ycb)

        xlo, xhi, ylo, yhi = self._sampling_bounds()

        # Sample goal patch center anywhere in the trimmed reach box.
        goal_x = rng.uniform(xlo, xhi)
        goal_y = rng.uniform(ylo, yhi)
        self._goal_pos = np.array([goal_x, goal_y])

        self._place_static(
            "goal_marker",
            [goal_x, goal_y, self._table_height + 0.001],
        )

        # Goal marker is visual-only: disable its collisions so the
        # object doesn't rest on top of the 1 mm patch geom (which
        # would hide the real table-contact signal and also shift the
        # object's z 1 mm high).
        sim = self._env.sim
        goal_body_id = sim.model.body_name2id("goal_marker_main")
        for g_idx in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g_idx] == goal_body_id:
                sim.model.geom_contype[g_idx] = 0
                sim.model.geom_conaffinity[g_idx] = 0

        # Sample object spawn in the same trimmed box, at least
        # `min_obj_goal_dist` from the goal center so the task is
        # non-trivial. Fall back to the last candidate after 30 tries.
        ox, oy = goal_x, goal_y
        for _ in range(30):
            cx = rng.uniform(xlo, xhi)
            cy = rng.uniform(ylo, yhi)
            if np.hypot(cx - goal_x, cy - goal_y) >= self._min_obj_goal_dist:
                ox, oy = cx, cy
                break

        # Place the active object resting on the table.
        active_obj = self._env.get_object(active_name)
        bottom_dz = abs(active_obj.bottom_offset[2]) if active_obj is not None else 0.0
        yaw = rng.uniform(0, 2 * np.pi)
        yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        base_quat = np.array(DEFAULT_SPAWN_POSE.get(
            self._selected_ycb, [1.0, 0.0, 0.0, 0.0]
        ))
        spawn_quat = _quat_mul(yaw_quat, base_quat)
        air_gap = 0.015 if self._selected_ycb in DEFAULT_SPAWN_POSE else 0.005
        self._rest_z = self._table_height + bottom_dz
        obj_z = self._rest_z + air_gap
        self._place_object(active_name, [ox, oy, obj_z], quat=spawn_quat)
        self._set_object_velocity(active_name, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        self._env.sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ── Success check ───────────────────────────────────────────────

    def _in_xy(self, raw_obs) -> bool:
        """True iff the goal is final and the object's xy center is inside it.

        While relocations remain the patch is not the target, so an object
        that reaches it without triggering one (slid in, or carried below
        ``min_lift`` or outside ``approach_radius``) earns no placement
        credit and cannot succeed.
        """
        if self._mg_remaining > 0:
            return False
        obj_xy = raw_obs["cube_pos"][:2]
        return (
            abs(obj_xy[0] - self._goal_pos[0]) < self._goal_half
            and abs(obj_xy[1] - self._goal_pos[1]) < self._goal_half
        )

    def _in_z(self, raw_obs) -> bool:
        """True iff the object's z is within `goal_z_tol` of the table surface.

        Lower bound allows some mesh-penetration / object-bottom offset;
        upper bound is the placement tolerance from config.
        """
        obj_z = raw_obs["cube_pos"][2]
        return (
            self._table_height - 0.02
            <= obj_z
            <= self._table_height + self._goal_z_tol
        )

    def _compute_task_score(self, raw_obs):
        self._inject_active_pose(raw_obs)
        if "cube_pos" not in raw_obs:
            return 0.0

        obj_xy = raw_obs["cube_pos"][:2]
        obj_z = raw_obs["cube_pos"][2]

        # Moving-goal caveat: while relocations remain, trigger one as
        # soon as the object is lifted and approaching the current
        # goal. Guard with `_mg_remaining` so once the caveat budget is
        # spent the goal stays put and success can be scored normally.
        lifted = obj_z > self._rest_z + self._mg_min_lift
        if self._mg_remaining > 0:
            dist_to_goal = float(np.hypot(
                obj_xy[0] - self._goal_pos[0],
                obj_xy[1] - self._goal_pos[1],
            ))
            if lifted and dist_to_goal <= self._mg_approach_radius:
                self._relocate_goal()
                # The goal just moved -- any xy/z staged progress credited
                # toward the old goal position is no longer meaningful, so
                # drop it along with the score for this step.
                self._staged_reward.reset()
                return 0.0

        # Anti-slide guard: the object must have been lifted at least
        # `_mg_min_lift` above its resting height at some point this
        # episode. Rejects policies that just slide the object into the patch.
        if lifted:
            self._was_lifted = True

        # At-rest check happens inside the staged-reward stage fn below
        # (uses velocity, not table-contact, to avoid brittleness from
        # YCB collision meshes that occasionally sit a hair above the
        # table surface).
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        # Terminate on success only: sticky partial credit can sum to 1.0
        # without the success condition ever holding.
        if self._stage_debug["_success"]:
            self._task_done = True
        return score

    def _placed_and_released(self) -> bool:
        """The active object is at rest and the robot no longer touches it."""
        name = self._active_obj_name()
        return self._object_at_rest() and not robot_contacts(self._env.sim, object_geom_ids(self._env, name))

    def _object_at_rest(self, lin_tol: float = 0.05, ang_tol: float = 0.5) -> bool:
        """Return True if the active object's linear+angular speed are
        both below tolerance. Used as a proxy for 'placed and settled'.
        """
        sim = self._env.sim
        body_id = sim.model.body_name2id(f"{self._active_obj_name()}_main")
        jnt_addr = sim.model.body_jntadr[body_id]
        if jnt_addr < 0:
            return False
        dof_addr = sim.model.jnt_dofadr[jnt_addr]
        lin = sim.data.qvel[dof_addr:dof_addr + 3]
        ang = sim.data.qvel[dof_addr + 3:dof_addr + 6]
        return (float(np.linalg.norm(lin)) < lin_tol
                and float(np.linalg.norm(ang)) < ang_tol)

    def _check_termination(self, raw_obs):
        if self._task_done:
            return True
        self._inject_active_pose(raw_obs)
        return self._object_fell_off_table(raw_obs)
