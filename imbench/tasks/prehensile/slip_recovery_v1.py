"""Pick-and-place a small YCB object onto a green goal region on the table.

Task ID: ``ib-p6-slip-recovery-v1``
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


# Per-object spawn-orientation overrides. Some YCB meshes have default
# poses whose vertical axis isn't their shortest — tippy and hard to
# grasp. Quaternions below (w, x, y, z) rotate the mesh so the shortest
# local axis becomes vertical (largest face down).
#
# gelatin_box body-frame dims (x, y, z) ≈ (0.073, 0.082, 0.090): shortest
# is x, so rotate 90° about y to bring x to world-z.
_AXIS_Y_90 = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]

DEFAULT_SPAWN_POSE = {
    "gelatin_box": _AXIS_Y_90,
    # pudding_box default is already flat (x is its shortest axis).
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


class SlipRecoveryV1(SingleArmBase):
    TASK_ID = "ib-p6-slip-recovery-v1"
    DESCRIPTION = (
        "Single arm: pick a small YCB object from the table and place it "
        "inside a green goal region on the table (object must be released "
        "onto the table inside the region)."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._ycb_pool = list(task_cfg.get("ycb_pool", DEFAULT_YCB_POOL))
        self._goal_half = float(task_cfg.get("goal_half_extent", 0.10))
        self._spawn_distance_range = tuple(
            task_cfg.get("spawn_distance_range", [0.15, 0.25])
        )
        self._goal_xy_range = tuple(task_cfg.get("goal_xy_range", [0.12, 0.12]))

        # ── Slippage simulation ────────────────────────────────────
        # While the object is grasped and lifted mid-transport, force a
        # single slip event so the policy must re-grasp and retry.
        slip_cfg = task_cfg.get("slippage", {})
        self._slip_enabled = slip_cfg.get("enabled", True)
        # Minimum horizontal distance from the goal for a slip to trigger.
        # Must exceed the goal square's half-diagonal so the object is
        # never dropped inside the goal.
        self._slip_min_goal_dist = float(slip_cfg.get("min_goal_dist", 0.20))
        if self._slip_min_goal_dist <= self._goal_half * np.sqrt(2.0):
            raise ValueError(
                f"slippage.min_goal_dist ({self._slip_min_goal_dist}) must exceed the "
                f"goal half-diagonal ({self._goal_half * np.sqrt(2.0):.3f})"
            )
        # Minimum lift above the table (m) before a slip may trigger.
        self._slip_min_lift = float(slip_cfg.get("min_lift", 0.05))
        # Slip fires after a variable number of consecutive lifted-and-far
        # steps sampled uniformly from this range on each reset. If the
        # object crosses ``min_goal_dist`` before the counter reaches its
        # threshold, the slip is forced immediately so the object never
        # slips inside the goal approach.
        self._slip_hold_range = tuple(
            slip_cfg.get("hold_steps_range", [30, 200])
        )

        # Number of control steps the gripper action is forced open (= -1)
        # after a slip triggers; decremented each step. While > 0, any
        # "close" command from the policy is masked out so the object can
        # fall clear of the fingers.
        self._slip_open_window = int(slip_cfg.get("open_window", 20))

        self._goal_pos = np.zeros(2)
        self._task_done = False
        self._slipped = False
        self._grasp_hold_count = 0
        self._slip_hold_target = int(self._slip_hold_range[0])
        self._slip_force_open_remaining = 0
        # Reward state machine: success requires a clean "pick and place"
        # trajectory — the object must be airborne inside the goal region
        # before it touches the table there. Sliding the object into the
        # region (touching-table the whole time) disqualifies the episode.
        self._was_lifted = False
        self._released_after_slip = False
        self._recovered_after_slip = False
        self._object_spawn_xy = np.zeros(2)
        self._parked = set()
        self._object_geoms = set()
        self._support_geoms = set()
        # Refreshed every scoring call (see _compute_task_score).
        self._touching_table = False
        self._held = False

        self._staged_reward = StagedReward(
            stages=[
                Stage("initial_lift", 0.20, lambda t, o: 1.0 if t._was_lifted else 0.0),
                Stage("transport_progress_toward_goal", 0.20, lambda t, o: t._transport_progress(o)),
                Stage("recover_after_slip", 0.25, lambda t, o: t._recovery_progress(o)),
                Stage("final_placement_in_goal", 0.35,
                      lambda t, o: 1.0 if t._placed_in_goal(o) else 0.0, sticky=False),
            ],
            success_fn=lambda t, o: (
                t._placed_in_goal(o) and (not t._slip_enabled or t._recovered_after_slip)
            ),
        )

        self._selected_ycb = self._ycb_pool[
            self._rng.randint(len(self._ycb_pool))
        ]

    def _ycb_obj_name(self, ycb_name: str) -> str:
        return f"ycb_{ycb_name}"

    def _active_obj_name(self) -> str:
        return self._ycb_obj_name(self._selected_ycb)

    def _inject_active_pose(self, raw_obs):
        """Overwrite ``cube_pos`` / ``cube_quat`` with the active YCB's
        pose so reward and slip logic always see the one object that
        matters. The env's default alias points to the FIRST dynamic
        object, which isn't necessarily the active one. Idempotent; the
        scorers call it too so ``task_score(recompute=True)`` scores the
        active object."""
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
        # Bake every YCB in the pool into the scene once. On each reset we
        # teleport the chosen one onto the table and park the rest
        # off-scene, avoiding the slow close/build rebuild cycle.
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
        """Pick a new active YCB and park all others off-scene.

        Every YCB in the pool is already in the sim; we just teleport the chosen one onto
        the table and park the rest. Reset is effectively instant, so the
        dashboard never stalls.
        """
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

    def step(self, action):
        # Mask the gripper action open while a slip is active so the
        # object actually falls out of the fingers even if the policy
        # keeps commanding close.
        if self._slip_force_open_remaining > 0:
            action = np.asarray(action, dtype=float).copy()
            action[-1] = -1.0  # fully open
            self._slip_force_open_remaining -= 1
        return super().step(action)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Native GJK/EPA convex collision: under libccd's MPR a tumbling YCB
        # hull can gain energy on impact with the table and bounce off it.
        self._env.sim.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_NATIVECCD
        # Robosuite's env reset doesn't clear each part-controller's
        # internal goal pose. Without this, the OSC controller carries
        # last episode's target and drives the arm toward it on step 1
        # (visible as the arm "moving backwards"). Refresh each
        # controller's reference pose from the sim, then snap goals to
        # the current state.
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
        self._slipped = False
        self._grasp_hold_count = 0
        self._slip_force_open_remaining = 0
        self._was_lifted = False
        self._released_after_slip = False
        self._recovered_after_slip = False
        self._staged_reward.reset()
        lo, hi = self._slip_hold_range
        self._slip_hold_target = int(self._rng.randint(int(lo), int(hi) + 1))

        # Use the base class's `self._rng` for all per-episode sampling
        # below: seeded from `self.seed` only when `reset(seed=...)` is
        # called with an explicit seed, so `reset(seed=7)` reproduces
        # identical draws on every call — required by the eval harness's
        # "K repeats per seed, identical reset" protocol — while repeated `reset(seed=None)`
        # calls (the normal RL rollout pattern) still vary, since the RNG
        # keeps advancing instead of being pinned to one draw forever.
        rng = self._rng

        # Pick a new active YCB and hide the others.
        self._swap_active_ycb()
        active_name = self._ycb_obj_name(self._selected_ycb)

        sim = self._env.sim

        # Sample goal region within reachable bounds.
        gx_half, gy_half = self._goal_xy_range
        goal_x = rng.uniform(-gx_half, gx_half)
        goal_y = rng.uniform(-gy_half, gy_half)
        self._goal_pos = np.array([goal_x, goal_y])

        self._place_static(
            "goal_marker",
            [goal_x, goal_y, self._table_height + 0.001],
        )

        # Spawn the object at a random direction `d` away from the goal,
        # with d sampled in [spawn_min, spawn_max]. Clamp to stay on table.
        d_min, d_max = self._spawn_distance_range
        for _ in range(20):
            theta = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(d_min, d_max)
            ox = goal_x + dist * np.cos(theta)
            oy = goal_y + dist * np.sin(theta)
            # Stay within a reachable box on the table.
            if abs(ox) < 0.32 and abs(oy) < 0.32:
                break
        # Place the active object resting on the table.
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

    # ── Slippage simulation ─────────────────────────────────────────

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
        """Drop the object once, mid-transport, to force a re-grasp.

        Triggers on the "mid-transport" signature: the object is airborne
        (``> _slip_min_lift`` above the table) and the goal is still far
        (``> _slip_min_goal_dist`` away), or at once if the airborne object
        is already within ``_slip_min_goal_dist``. We purposely do not require the
        ``_check_grasp`` flag — with many YCB shapes the robosuite grasp
        heuristic is flaky, and "lifted" is a cleaner transport signal.
        Fires exactly once per episode.
        """
        if self._slipped or not self._slip_enabled:
            return
        if "cube_pos" not in raw_obs:
            return

        obj_pos = raw_obs["cube_pos"]
        lifted = obj_pos[2] > self._table_height + self._slip_min_lift
        goal_dist = np.linalg.norm(obj_pos[:2] - self._goal_pos)
        far_from_goal = goal_dist > self._slip_min_goal_dist

        # Safety check BEFORE updating the counter. If the object is
        # lifted inside the near-goal zone, force the slip immediately.
        # This guarantees the slip always fires before the policy can
        # place the object into the goal.
        approaching_goal = lifted and not far_from_goal

        if lifted and far_from_goal:
            self._grasp_hold_count += 1
        else:
            self._grasp_hold_count = 0

        reached_target = self._grasp_hold_count >= self._slip_hold_target
        if reached_target or approaching_goal:
            self._slip_force_open_remaining = self._slip_open_window
            self._slipped = True

    # ── Success: object released onto the table inside the goal ───────

    def _object_touches_table(self):
        """Whether the active object touches the table top or the goal marker on it."""
        return geoms_touch(self._env.sim, self._object_geoms, self._support_geoms)

    def _in_goal_region(self, raw_obs) -> bool:
        obj_xy = raw_obs["cube_pos"][:2]
        return (
            abs(obj_xy[0] - self._goal_pos[0]) < self._goal_half
            and abs(obj_xy[1] - self._goal_pos[1]) < self._goal_half
        )

    def _transport_progress(self, raw_obs) -> float:
        """Shrinkage of the object-to-goal xy distance from its spawn.

        Counts only once the object has been lifted at some point
        (``_was_lifted``), so pushing it along the table from the start earns
        no transport credit; after a lift, sliding it closer does.
        """
        if not self._was_lifted:
            return 0.0
        obj_xy = raw_obs["cube_pos"][:2]
        dist = float(np.linalg.norm(obj_xy - self._goal_pos))
        initial_dist = float(np.linalg.norm(self._object_spawn_xy - self._goal_pos))
        if initial_dist <= 1e-6:
            return 0.0
        return 1.0 - dist / initial_dist

    def _placed_in_goal(self, raw_obs) -> bool:
        """Lifted at some point, now released onto the table inside the goal."""
        return (self._was_lifted and self._touching_table and not self._held
                and self._in_goal_region(raw_obs))

    def _update_recovery(self, raw_obs):
        """Latch the slip's release and the subsequent re-grasp.

        The object counts as released once it touches the table untouched by
        the robot after the slip fired, and as recovered once the robot then
        holds it lifted past the slip's lift threshold again.
        """
        if not self._slipped or self._recovered_after_slip:
            return
        if not self._released_after_slip:
            self._released_after_slip = self._touching_table and not self._held
            return
        lifted_again = raw_obs["cube_pos"][2] > self._table_height + self._slip_min_lift
        if lifted_again and self._held:
            self._recovered_after_slip = True

    def _recovery_progress(self, raw_obs) -> float:
        """1.0 once the object has been re-grasped and re-lifted after the
        scripted slip dropped it."""
        return 1.0 if self._recovered_after_slip else 0.0

    def _compute_task_score(self, raw_obs):
        self._inject_active_pose(raw_obs)
        self._maybe_trigger_slip(raw_obs)
        if "cube_pos" not in raw_obs:
            return 0.0

        obj_z = raw_obs["cube_pos"][2]
        self._touching_table = self._object_touches_table()
        self._held = robot_contacts(self._env.sim, self._object_geoms)
        lifted = obj_z > self._table_height + self._slip_min_lift

        # Anti-slide guard (relaxed): the object must have been picked up
        # (lifted above the threshold) at some point during the episode —
        # anywhere, not necessarily inside the goal region. This rejects
        # policies that only push/slide the object along the table.
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
