"""Pick-and-place a small YCB object onto a green goal region on the table.

Task ID: ``ib-p6-slip-recovery-v2``

OOD variant: double slip
    The object is dropped **twice** per episode instead of once.
    The policy must re-grasp and resume transport two times before it
    can successfully place the object in the goal region.
"""

import mujoco
import numpy as np

from imbench.tasks.prehensile.contacts import geoms_touch, object_geom_ids, robot_contacts
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


def _quat_mul(q1, q2):
    """Hamilton product of two [w, x, y, z] quaternions → q1 ∘ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


_AXIS_Y_90 = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]

DEFAULT_SPAWN_POSE = {
    "gelatin_box": _AXIS_Y_90,
    # The lemon mesh's default pose is not a resting one: it drops and rolls
    # ~4 cm. This is the pose it settles into, on its side.
    "lemon": [0.7504, 0.6207, -0.2245, 0.0346],
}


DEFAULT_YCB_POOL = [
    "apple",
    "banana",
    "lemon",
    "orange",
    "peach",
    "pear",
    "strawberry",
    "tomato_soup_can",
    "tuna_fish_can",
    "mustard_bottle",
]

# Maximum number of slip events per episode.
_MAX_SLIPS = 2


class SlipRecoveryV2(SingleArmBase):
    TASK_ID = "ib-p6-slip-recovery-v2"
    DESCRIPTION = (
        "Single arm: pick a small YCB object from the table and place it "
        "inside a green goal region on the table (object must be released "
        "onto the table inside the region). "
        "[OOD] The gripper will slip and drop the object exactly twice "
        "per episode; the policy must re-grasp and recover both times."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._ycb_pool = list(task_cfg.get("ycb_pool", DEFAULT_YCB_POOL))
        self._goal_half = float(task_cfg.get("goal_half_extent", 0.10))
        self._spawn_distance_range = tuple(
            task_cfg.get("spawn_distance_range", [0.15, 0.25])
        )
        self._goal_xy_range = tuple(task_cfg.get("goal_xy_range", [0.12, 0.12]))

        slip_cfg = task_cfg.get("slippage", {})
        self._slip_enabled = slip_cfg.get("enabled", True)
        # Must exceed the goal square's half-diagonal so the object is never
        # dropped inside the goal.
        self._slip_min_goal_dist = float(slip_cfg.get("min_goal_dist", 0.20))
        if self._slip_min_goal_dist <= self._goal_half * np.sqrt(2.0):
            raise ValueError(
                f"slippage.min_goal_dist ({self._slip_min_goal_dist}) must exceed the "
                f"goal half-diagonal ({self._goal_half * np.sqrt(2.0):.3f})"
            )
        self._slip_min_lift = float(slip_cfg.get("min_lift", 0.05))
        self._slip_hold_range = tuple(
            slip_cfg.get("hold_steps_range", [30, 200])
        )
        self._slip_open_window = int(slip_cfg.get("open_window", 20))

        self._goal_pos = np.zeros(2)
        self._task_done = False

        # ── Double-slip state ──────────────────────────────────────
        # Number of slips that have already fired this episode (0, 1, or 2).
        self._slip_count: int = 0
        # Per-slip hold thresholds (list of _MAX_SLIPS ints), sampled fresh
        # each reset so the two windows are independent.
        self._slip_hold_targets: list = [int(self._slip_hold_range[0])] * _MAX_SLIPS
        # Steps the object has been continuously lifted-and-far since the
        # last slip (or since episode start).
        self._grasp_hold_count: int = 0
        # Countdown for how many more steps the gripper is forced open.
        self._slip_force_open_remaining: int = 0
        # ──────────────────────────────────────────────────────────

        self._was_lifted = False
        self._released_slips = [False] * _MAX_SLIPS
        self._recovered_slips = [False] * _MAX_SLIPS
        self._object_spawn_xy = np.zeros(2)
        self._parked = set()
        self._object_geoms = set()
        self._support_geoms = set()
        # Refreshed every scoring call (see _compute_task_score).
        self._touching_table = False
        self._held = False

        self._selected_ycb = self._ycb_pool[
            self._rng.randint(len(self._ycb_pool))
        ]

        self._staged_reward = StagedReward(
            stages=[
                Stage("initial_lift", 0.15, lambda t, o: 1.0 if t._was_lifted else 0.0),
                Stage("recover_after_slip_1", 0.20, lambda t, o: t._recovery_progress(o, 0)),
                Stage("recover_after_slip_2", 0.20, lambda t, o: t._recovery_progress(o, 1)),
                Stage("final_placement_in_goal", 0.45,
                      lambda t, o: 1.0 if t._placed_in_goal(o) else 0.0, sticky=False),
            ],
            success_fn=lambda t, o: (
                t._placed_in_goal(o) and (not t._slip_enabled or all(t._recovered_slips))
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ycb_obj_name(self, ycb_name: str) -> str:
        return f"ycb_{ycb_name}"

    def _active_obj_name(self) -> str:
        return self._ycb_obj_name(self._selected_ycb)

    def _inject_active_pose(self, raw_obs):
        """Point ``cube_pos`` / ``cube_quat`` at the active YCB's pose.

        Idempotent; the scorers call it too so ``task_score(recompute=True)``
        scores the active object.
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

    def _make_objects(self):
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
             "rgba": [0.0, 0.8, 0.0, 0.35],
             "density": 1,
             "friction": [0.1, 0.005, 0.0001],
             "joints": None},
        ]

    def _swap_active_ycb(self):
        self._selected_ycb = self._ycb_pool[
            self._rng.randint(len(self._ycb_pool))
        ]
        self._parked = {
            self._ycb_obj_name(ycb) for ycb in self._ycb_pool
            if ycb != self._selected_ycb
        }
        self._park_objects(self._parked)

    def _parked_objects(self):
        return self._parked

    # ------------------------------------------------------------------
    # Step: mask gripper while a slip is active
    # ------------------------------------------------------------------

    def step(self, action):
        if self._slip_force_open_remaining > 0:
            action = np.asarray(action, dtype=float).copy()
            action[-1] = -1.0  # fully open
            self._slip_force_open_remaining -= 1
        return super().step(action)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Native GJK/EPA convex collision: under libccd's MPR a tumbling YCB
        # hull can gain energy on impact with the table and bounce off it.
        self._env.sim.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_NATIVECCD

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
        self._released_slips = [False] * _MAX_SLIPS
        self._recovered_slips = [False] * _MAX_SLIPS
        self._grasp_hold_count = 0
        self._slip_force_open_remaining = 0
        self._staged_reward.reset()

        # ── Double-slip reset ──────────────────────────────────────
        self._slip_count = 0
        lo, hi = self._slip_hold_range
        # Sample two independent hold targets for the two slip events.
        self._slip_hold_targets = [
            int(self._rng.randint(int(lo), int(hi) + 1))
            for _ in range(_MAX_SLIPS)
        ]
        # ──────────────────────────────────────────────────────────

        # Use the base class's `self._rng` (seeded from `self.seed` only
        # when `reset(seed=...)` is called with an explicit seed) so
        # `reset(seed=7)` reproduces identical draws every call — required
        # by the eval harness's "K repeats per seed, identical reset"
        # protocol — while repeated `reset(seed=None)` calls still vary,
        # since the RNG keeps advancing between episodes.
        rng = self._rng

        self._swap_active_ycb()
        active_name = self._ycb_obj_name(self._selected_ycb)

        sim = self._env.sim

        gx_half, gy_half = self._goal_xy_range
        goal_x = rng.uniform(-gx_half, gx_half)
        goal_y = rng.uniform(-gy_half, gy_half)
        self._goal_pos = np.array([goal_x, goal_y])

        self._place_static(
            "goal_marker",
            [goal_x, goal_y, self._table_height + 0.001],
        )

        d_min, d_max = self._spawn_distance_range
        for _ in range(20):
            theta = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(d_min, d_max)
            ox = goal_x + dist * np.cos(theta)
            oy = goal_y + dist * np.sin(theta)
            if abs(ox) < 0.32 and abs(oy) < 0.32:
                break

        yaw = rng.uniform(0, 2 * np.pi)
        yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        base_quat = np.array(DEFAULT_SPAWN_POSE.get(
            self._selected_ycb, [1.0, 0.0, 0.0, 0.0]
        ))
        spawn_quat = _quat_mul(yaw_quat, base_quat)
        # Lowest point of the posed collision meshes 5 mm above the table
        # (``bottom_offset`` is in the unposed mesh frame, so it is wrong for
        # objects re-oriented by DEFAULT_SPAWN_POSE).
        self._place_object(active_name, [ox, oy, self._table_height], quat=spawn_quat)
        sim.forward()
        obj_z = self._table_height + (
            self._table_height + 0.005 - self._lowest_collision_z(active_name))
        self._place_object(active_name, [ox, oy, obj_z], quat=spawn_quat)
        self._set_object_velocity(active_name, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self._object_spawn_xy = np.array([ox, oy])
        self._object_geoms = object_geom_ids(self._env, active_name)
        self._support_geoms = object_geom_ids(self._env, "goal_marker") | {
            sim.model.geom_name2id("table_collision")}
        self._touching_table = False
        self._held = False

        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ------------------------------------------------------------------
    # Double-slip logic
    # ------------------------------------------------------------------

    def _lowest_collision_z(self, name):
        """World z of the lowest point of *name*'s collision geoms (mesh
        vertices; bounding-box corners for primitives)."""
        sim = self._env.sim
        model = sim.model
        lowest = np.inf
        for g in object_geom_ids(self._env, name):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            rot_z = sim.data.geom_xmat[g].reshape(3, 3)[2]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                start = model.mesh_vertadr[mesh]
                verts = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
                offset = (verts @ rot_z).min()
            else:
                centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
                offset = rot_z @ centre - np.abs(rot_z) @ half
            lowest = min(lowest, sim.data.geom_xpos[g][2] + offset)
        return float(lowest)

    def _maybe_trigger_slip(self, raw_obs):
        """Drop the object up to _MAX_SLIPS (2) times per episode.

        Each slip event is independent:
          - After the first slip fires, _slip_count becomes 1 and
            _grasp_hold_count resets to 0 so the second event has its
            own hold window (_slip_hold_targets[1]).
          - After the second slip fires, _slip_count becomes 2 and no
            further slips are triggered.

        The approaching_goal early-fire guard applies to whichever slip
        is currently armed (indexed by _slip_count). A slip is armed only
        once the previous one has been recovered, so success (which needs
        every slip recovered) stays reachable.
        """
        if not self._slip_enabled:
            return
        # All slips already consumed.
        if self._slip_count >= _MAX_SLIPS:
            return
        # A slip is already in progress (gripper forced open); don't
        # start another one until the window expires.
        if self._slip_force_open_remaining > 0:
            return
        if self._slip_count > 0 and not self._recovered_slips[self._slip_count - 1]:
            return
        if "cube_pos" not in raw_obs:
            return

        obj_pos = raw_obs["cube_pos"]
        lifted = obj_pos[2] > self._table_height + self._slip_min_lift
        goal_dist = np.linalg.norm(obj_pos[:2] - self._goal_pos)
        far_from_goal = goal_dist > self._slip_min_goal_dist

        # Current hold target for whichever slip is next.
        current_target = self._slip_hold_targets[self._slip_count]

        # Lifted inside the near-goal zone fires at once, so the object
        # cannot be carried to the goal without slipping.
        approaching_goal = lifted and not far_from_goal

        if lifted and far_from_goal:
            self._grasp_hold_count += 1
        else:
            # Object not in the air or already near goal — reset counter
            # so only sustained mid-transport lift counts.
            self._grasp_hold_count = 0

        reached_target = self._grasp_hold_count >= current_target

        if reached_target or approaching_goal:
            # Fire this slip.
            self._slip_force_open_remaining = self._slip_open_window
            self._slip_count += 1
            # Reset counter so the next slip event starts fresh.
            self._grasp_hold_count = 0

    # ------------------------------------------------------------------
    # Table-contact detection
    # ------------------------------------------------------------------

    def _object_touches_table(self):
        """Whether the active object touches the table top or the goal marker on it."""
        return geoms_touch(self._env.sim, self._object_geoms, self._support_geoms)

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _in_goal_region(self, raw_obs) -> bool:
        obj_xy = raw_obs["cube_pos"][:2]
        return (
            abs(obj_xy[0] - self._goal_pos[0]) < self._goal_half
            and abs(obj_xy[1] - self._goal_pos[1]) < self._goal_half
        )

    def _placed_in_goal(self, raw_obs) -> bool:
        """Lifted at some point, now released onto the table inside the goal."""
        return (self._was_lifted and self._touching_table and not self._held
                and self._in_goal_region(raw_obs))

    def _update_recovery(self, raw_obs):
        """Latch the latest slip's release and the subsequent re-grasp.

        After slip ``i`` fires, the object counts as released once it touches
        the table untouched by the robot, and slip ``i`` as recovered once the
        robot then holds it lifted past the slip's lift threshold again.
        """
        i = self._slip_count - 1
        if i < 0 or self._recovered_slips[i]:
            return
        if not self._released_slips[i]:
            self._released_slips[i] = self._touching_table and not self._held
            return
        lifted_again = raw_obs["cube_pos"][2] > self._table_height + self._slip_min_lift
        if lifted_again and self._held:
            self._recovered_slips[i] = True

    def _recovery_progress(self, raw_obs, slip_index: int) -> float:
        """1.0 once the object has been re-grasped and re-lifted after the
        ``slip_index``-th scripted slip (0 → first slip, 1 → second)."""
        return 1.0 if self._recovered_slips[slip_index] else 0.0

    def _compute_task_score(self, raw_obs):
        self._inject_active_pose(raw_obs)
        self._maybe_trigger_slip(raw_obs)
        if "cube_pos" not in raw_obs:
            return 0.0

        obj_z = raw_obs["cube_pos"][2]
        self._touching_table = self._object_touches_table()
        self._held = robot_contacts(self._env.sim, self._object_geoms)
        lifted = obj_z > self._table_height + self._slip_min_lift

        if lifted and not self._touching_table:
            self._was_lifted = True
        self._update_recovery(raw_obs)

        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        if score >= 1.0:
            self._task_done = True
        return score

    def _check_termination(self, raw_obs):
        if self._task_done:
            return True
        self._inject_active_pose(raw_obs)
        return self._object_fell_off_table(raw_obs)