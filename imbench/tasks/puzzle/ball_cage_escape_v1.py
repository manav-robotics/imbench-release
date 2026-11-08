"""Ball Cage Escape — tilt a thin cuboid to let a trapped ball drop through a hole in its floor.

Task ID: ``ib-p3-ball-cage-escape-v1``

A thin rectangular cuboid stands upright on the table with a ball trapped
inside. The cuboid's bottom face has a rectangular hole (random XY, away
from corners) just wide enough to let the ball through. With the cuboid
sitting flat, the table blocks the hole and the ball stays in. The agent
must lift and/or tilt the cuboid so the hole clears the table, then the
ball drops out. Success: the ball touches the table outside the cuboid.
"""

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward


# Number of additional sim steps to run after the score first reaches
# 1.0 before terminating. Lets the ball come to rest on the table so
# the recorded data captures the settled scene, not the moment of
# first contact.
POST_SUCCESS_DWELL_STEPS = 30


class BallCageEscapeV1(SingleArmBase):
    TASK_ID = "ib-p3-ball-cage-escape-v1"
    DESCRIPTION = (
        "Puzzle: a ball is trapped inside a thin cuboid whose floor has a "
        "hole. Tilt or lift the cuboid so the ball drops out through the hole."
    )

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._cage_inner = task_cfg.get("cage_inner_size", [0.08, 0.08, 0.05])
        self._wall_thickness = task_cfg.get("wall_thickness", 0.004)
        self._ball_radius = task_cfg.get("ball_radius", 0.016)
        self._hole_scale = task_cfg.get("hole_scale", 1.15)
        self._corner_margin = task_cfg.get("corner_margin", 0.010)
        self._ball_density = task_cfg.get("ball_density", 400.0)
        self._cage_density = task_cfg.get("cage_density", 500.0)
        self._cage_rgba = task_cfg.get("cage_rgba", [0.55, 0.65, 0.80, 0.25])
        # Narrow-but-tall platform ("stool") that elevates the cage so the
        # gripper can approach from the side.
        self._stool_size = task_cfg.get("stool_size", [0.06, 0.06, 0.10])
        self._stool_rgba = task_cfg.get("stool_rgba", [0.35, 0.35, 0.35, 1.0])
        self._hole_xy = (0.0, 0.0)
        self._hole_half = self._ball_radius * self._hole_scale
        self._hole_rest_z = self._table_height
        if min(self._stool_size[:2]) / 2 < self._hole_half:
            raise ValueError(
                f"stool_size {self._stool_size} is too small to cover the "
                f"floor hole (half-width {self._hole_half:.3f} m)"
            )

        # hole_cleared tracks how far the hole (in the cage's bottom face)
        # has risen above its resting height — via the cage's own rotation,
        # so tilting counts just as much as a straight vertical lift.
        # ball_separated_from_cage fires once the ball has broken contact
        # with the cage AND dropped below the hole's current height, i.e.
        # it actually fell out through the hole rather than merely losing
        # contact in place. ball_settled_on_table is the success condition
        # and tracks the current state, so a ball that rolls away loses
        # its credit and the score drops below 1 along with is_success.
        self._staged_reward = StagedReward(
            stages=[
                Stage("cage_lifted", 0.2, lambda t, o: 1.0 if t._cage_picked_up else 0.0),
                Stage("hole_cleared", 0.3, lambda t, o: t._hole_cleared_progress(o)),
                Stage("ball_separated_from_cage", 0.2,
                      lambda t, o: 1.0 if t._ball_separated_from_cage() else 0.0),
                Stage("ball_settled_on_table", 0.3,
                      lambda t, o: 1.0 if t._ball_escaped_onto_table() else 0.0,
                      sticky=False),
            ],
            success_fn=lambda t, o: t._ball_escaped_onto_table(),
        )

    def _make_objects(self):
        ci = [s / 2 for s in self._cage_inner]
        wt = self._wall_thickness
        cage_rgba = list(self._cage_rgba)
        cage_friction = [0.4, 0.005, 0.0001]
        dens = self._cage_density

        # Bottom-face frame: 4 bars (x-neg, x-pos, y-neg, y-pos) surrounding
        # a rectangular hole. Sizes + local positions are placeholders —
        # re-laid every reset via `_relayout_bottom_frame()`.
        sub_objects = [
            # Top wall (solid)
            {"type": "box", "name": "top",
             "size": [ci[0], ci[1], wt / 2],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            # Side walls
            {"type": "box", "name": "left",
             "size": [wt / 2, ci[1], ci[2]],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            {"type": "box", "name": "right",
             "size": [wt / 2, ci[1], ci[2]],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            {"type": "box", "name": "front",
             "size": [ci[0], wt / 2, ci[2]],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            {"type": "box", "name": "back",
             "size": [ci[0], wt / 2, ci[2]],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            # Bottom frame (four bars around the hole)
            {"type": "box", "name": "bxn",
             "size": [ci[0], ci[1], wt / 2],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            {"type": "box", "name": "bxp",
             "size": [ci[0], ci[1], wt / 2],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            {"type": "box", "name": "byn",
             "size": [ci[0], ci[1], wt / 2],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
            {"type": "box", "name": "byp",
             "size": [ci[0], ci[1], wt / 2],
             "rgba": cage_rgba, "density": dens, "friction": cage_friction},
        ]
        sub_locations = [
            (0, 0, ci[2] - wt / 2),        # top
            (-(ci[0] - wt / 2), 0, 0),     # left
            (ci[0] - wt / 2, 0, 0),        # right
            (0, -(ci[1] - wt / 2), 0),     # front
            (0, ci[1] - wt / 2, 0),        # back
            (0, 0, -(ci[2] - wt / 2)),     # bxn placeholder
            (0, 0, -(ci[2] - wt / 2)),     # bxp placeholder
            (0, 0, -(ci[2] - wt / 2)),     # byn placeholder
            (0, 0, -(ci[2] - wt / 2)),     # byp placeholder
        ]

        return [
            {"type": "composite", "name": "cage",
             "sub_objects": sub_objects,
             "sub_locations": sub_locations,
             "total_size": [ci[0] * 2 + wt,
                            ci[1] * 2 + wt,
                            ci[2] * 2 + wt],
             "density": dens,
             "joints": "default"},
            {"type": "ball", "name": "cube",
             "size": [self._ball_radius],
             "rgba": [0.85, 0.35, 0.20, 1.0],
             "density": self._ball_density,
             "friction": [0.5, 0.005, 0.0001],
             "joints": "default"},
        ]

    def _make_static_objects(self):
        ss = [s / 2 for s in self._stool_size]
        return [
            {"type": "box", "name": "stool",
             "size": ss,
             "rgba": self._stool_rgba,
             "density": 5000.0,
             "friction": [0.8, 0.005, 0.0001],
             "joints": None},
        ]

    # ── Bottom-frame resize / relayout ──────────────────────────────

    def _relayout_bottom_frame(self, hole_cx, hole_cy):
        """Resize+reposition the 4 bottom-frame bars to encircle a hole at
        local (hole_cx, hole_cy) in the composite's frame.

        Uses sim.model.geom_size for half-extents and sim.model.body_pos for
        the welded sub-body's offset inside the composite.
        """
        sim = self._env.sim
        ci = [s / 2 for s in self._cage_inner]
        wt = self._wall_thickness
        hh = self._hole_half  # hole half-width

        # Inner bottom extent is ±ci[0] in x, ±ci[1] in y. Bottom sits at
        # local z = -(ci[2] - wt/2), which is inside the side walls. We use
        # the same z for all 4 bars.
        bz = -(ci[2] - wt / 2)

        # x-strip bars (run full width in x, clipped in y)
        # bxn: y ∈ [-ci[1], hole_cy - hh]
        bxn_len_y = max(((hole_cy - hh) - (-ci[1])) / 2.0, 1e-4)
        bxn_cy = (-ci[1] + (hole_cy - hh)) / 2.0
        # bxp: y ∈ [hole_cy + hh, ci[1]]
        bxp_len_y = max((ci[1] - (hole_cy + hh)) / 2.0, 1e-4)
        bxp_cy = ((hole_cy + hh) + ci[1]) / 2.0
        # y-strip bars (span only the hole x-band in y, fill x on either side)
        # byn: x ∈ [-ci[0], hole_cx - hh], y ∈ [hole_cy - hh, hole_cy + hh]
        byn_len_x = max(((hole_cx - hh) - (-ci[0])) / 2.0, 1e-4)
        byn_cx = (-ci[0] + (hole_cx - hh)) / 2.0
        byp_len_x = max((ci[0] - (hole_cx + hh)) / 2.0, 1e-4)
        byp_cx = ((hole_cx + hh) + ci[0]) / 2.0

        self._resize_sub_box(sim, "cage_bxn",
                             half_size=[ci[0], bxn_len_y, wt / 2],
                             local_pos=[0.0, bxn_cy, bz])
        self._resize_sub_box(sim, "cage_bxp",
                             half_size=[ci[0], bxp_len_y, wt / 2],
                             local_pos=[0.0, bxp_cy, bz])
        self._resize_sub_box(sim, "cage_byn",
                             half_size=[byn_len_x, hh, wt / 2],
                             local_pos=[byn_cx, hole_cy, bz])
        self._resize_sub_box(sim, "cage_byp",
                             half_size=[byp_len_x, hh, wt / 2],
                             local_pos=[byp_cx, hole_cy, bz])

    @staticmethod
    def _resize_sub_box(sim, body_name, half_size, local_pos):
        # Composite sub-bodies are registered as "<composite>_<sub>_main".
        body_id = sim.model.body_name2id(f"{body_name}_main")
        sim.model.body_pos[body_id] = np.array(local_pos, dtype=float)
        for g_idx in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g_idx] == body_id:
                sim.model.geom_size[g_idx] = np.array(half_size, dtype=float)

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._staged_reward.reset()
        self._failed = False
        # Post-success dwell: once score hits 1.0 we don't terminate
        # immediately. _check_termination sets this to 0 on the first
        # success tick and increments it each subsequent step until it
        # reaches POST_SUCCESS_DWELL_STEPS, then terminates. Lets the
        # ball fully settle on the table before the episode ends.
        self._post_success_ticks: int | None = None
        # Ball-freeze: until the cage is disturbed (moved or rotated
        # beyond a small threshold, or lifted) the ball is force-pinned to
        # zero velocity / zero acceleration every step, so micro-jitter
        # from contact noise can't pre-roll it before the policy starts
        # moving the cage. The rest pose is captured below, after the cage
        # has been placed on the stool and settled.
        self._cage_rest_z = self._table_height + 0.10
        self._cage_rest_pos = np.zeros(3)
        self._cage_rest_xmat = np.eye(3)
        self._cage_picked_up = False
        self._ball_released = False
        sim = self._env.sim

        ci = [s / 2 for s in self._cage_inner]
        wt = self._wall_thickness
        hh = self._hole_half

        # Hole XY in the cuboid's local frame, kept away from corners and
        # entirely over the stool top, so the stool blocks the hole until
        # the cage is moved. Bounds per axis:
        # ±min(ci - hh - corner_margin, stool_half - hh).
        margin = self._corner_margin
        ss = [s / 2 for s in self._stool_size]
        bx = max(min(ci[0] - hh - margin, ss[0] - hh), 1e-4)
        by = max(min(ci[1] - hh - margin, ss[1] - hh), 1e-4)
        hole_cx = float(self._rng.uniform(-bx, bx))
        hole_cy = float(self._rng.uniform(-by, by))
        self._hole_xy = (hole_cx, hole_cy)

        # Rebuild the bottom frame around the chosen hole.
        self._relayout_bottom_frame(hole_cx, hole_cy)

        # Place the stool first; the cage rests on top of it so the gripper
        # can approach from the side without colliding with the tabletop.
        cx, cy = 0.10, 0.0
        stool_z = self._table_height + ss[2]
        self._place_static("stool", [cx, cy, stool_z])

        # Cage sits centered on the stool top.
        stool_top_z = self._table_height + self._stool_size[2]
        cz = stool_top_z + ci[2] + wt / 2
        self._place_object("cage", [cx, cy, cz])
        # The ball stays parked until the cage has settled, so it cannot bias
        # the settling; it is placed below.
        self._park_objects(["cube"])
        self._set_object_velocity("cage", [0, 0, 0])
        sim.forward()

        # Let the empty cage settle on the stool so the bottom face is truly
        # flat before we drop the ball in — otherwise the ball would roll
        # downhill on the micro-tilt left over from the initial placement.
        # Only the cage keeps its settled state: the uncontrolled arm sags
        # under gravity during these raw steps, so everything else is
        # restored afterwards.
        qpos0 = sim.data.qpos.copy()
        qvel0 = sim.data.qvel.copy()
        for _ in range(60):
            sim.step()
        cage_jnt = sim.model.joint_name2id(self._env.get_object("cage").joints[0])
        cage_qadr = int(sim.model.jnt_qposadr[cage_jnt])
        cage_settled = sim.data.qpos[cage_qadr:cage_qadr + 7].copy()
        sim.data.qpos[:] = qpos0
        sim.data.qvel[:] = qvel0
        sim.data.qpos[cage_qadr:cage_qadr + 7] = cage_settled
        sim.forward()

        # Pick a random spawn spot for the ball on the inner floor, well
        # clear of the hole so it rests on the frame (not above the hole).
        # Ball must stay inside the walls and at least `clear` away from
        # the hole edge.
        inset = self._ball_radius + 0.003
        clear = self._ball_radius + self._hole_half + 0.005
        for _ in range(50):
            bx_off = self._rng.uniform(-(ci[0] - inset), (ci[0] - inset))
            by_off = self._rng.uniform(-(ci[1] - inset), (ci[1] - inset))
            if max(abs(bx_off - hole_cx), abs(by_off - hole_cy)) > clear:
                break
        # Re-read the cage's settled pose so we place the ball on the ACTUAL
        # equilibrium floor (not the nominal target that sim.forward used).
        cage_bid = sim.model.body_name2id("cage_root")
        cage_world = sim.data.body_xpos[cage_bid].copy()
        self._cage_rest_z = float(cage_world[2])
        self._cage_rest_pos = cage_world
        self._cage_rest_xmat = sim.data.body_xmat[cage_bid].reshape(3, 3).copy()

        # Place the ball exactly on the bottom-frame top surface (no drop
        # gap) so there is no settling impact when the episode starts.
        frame_top_z = cage_world[2] - (ci[2] - wt)
        self._hole_rest_z = frame_top_z
        bz = frame_top_z + self._ball_radius
        self._place_object("cube",
                           [cage_world[0] + bx_off, cage_world[1] + by_off, bz])

        # Fully freeze both dynamic bodies: clear linear + angular velocity
        # and any accumulated acceleration state, for ball and cage.
        for name in ("cube", "cage"):
            self._set_object_velocity(name, [0, 0, 0], angvel=[0, 0, 0])
        sim.data.qvel[:] = 0.0
        sim.data.qacc[:] = 0.0
        sim.data.qacc_warmstart[:] = 0.0

        sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ── Step override: freeze ball until the cage is disturbed ─────

    def step(self, action: np.ndarray):
        self._check_open()
        # Snapshot the ball's free-joint qpos BEFORE stepping so we can
        # restore it (along with zero qvel/qacc) if the cage hasn't
        # been disturbed yet. This guarantees the ball is exactly
        # stationary -- not just "low velocity" -- until the cage moves.
        if not self._ball_released:
            ball_qpos_addr, _ball_qvel_addr = self._ball_joint_addrs()
            if ball_qpos_addr is not None:
                ball_qpos_locked = self._env.sim.data.qpos[
                    ball_qpos_addr:ball_qpos_addr + 7
                ].copy()
            else:
                ball_qpos_locked = None
        else:
            ball_qpos_locked = None

        out = super().step(action)

        sim = self._env.sim
        if not self._cage_picked_up:
            self._update_cage_picked_up()
        if not self._ball_released and (self._cage_picked_up or self._cage_disturbed()):
            self._ball_released = True

        # Until the cage is disturbed, re-pin the ball.
        if not self._ball_released and ball_qpos_locked is not None:
            ball_qpos_addr, ball_qvel_addr = self._ball_joint_addrs()
            if ball_qpos_addr is not None:
                sim.data.qpos[ball_qpos_addr:ball_qpos_addr + 7] = ball_qpos_locked
            if ball_qvel_addr is not None:
                sim.data.qvel[ball_qvel_addr:ball_qvel_addr + 6] = 0.0
                sim.data.qacc[ball_qvel_addr:ball_qvel_addr + 6] = 0.0
                sim.data.qacc_warmstart[ball_qvel_addr:ball_qvel_addr + 6] = 0.0
            sim.forward()

        return out

    def _ball_joint_addrs(self):
        """Return (qpos_addr, qvel_addr) for the ball's free joint, or
        (None, None) if the ball has no free joint."""
        try:
            sim = self._env.sim
            body_id = sim.model.body_name2id("cube_main")
            jnt_addr = sim.model.body_jntadr[body_id]
            if jnt_addr < 0:
                return (None, None)
            return (
                int(sim.model.jnt_qposadr[jnt_addr]),
                int(sim.model.jnt_dofadr[jnt_addr]),
            )
        except Exception:
            return (None, None)

    # Cage-lifted threshold: cage z must rise this far above its rest
    # baseline before we consider it picked up. Tuned to be larger than
    # any plausible numerical jitter (a few mm) but smaller than the
    # smallest deliberate lift the policy commands.
    _CAGE_LIFT_THRESHOLD = 0.015   # 1.5 cm
    # Any cage displacement or rotation beyond these releases the ball.
    _CAGE_DISTURB_POS = 0.003      # m
    _CAGE_DISTURB_ANGLE_DEG = 2.0

    def _cage_disturbed(self):
        """True once the cage has moved or rotated beyond the freeze thresholds."""
        sim = self._env.sim
        cage_bid = sim.model.body_name2id("cage_root")
        shift = np.linalg.norm(sim.data.body_xpos[cage_bid] - self._cage_rest_pos)
        if shift > self._CAGE_DISTURB_POS:
            return True
        R = sim.data.body_xmat[cage_bid].reshape(3, 3)
        cos_angle = np.clip((np.trace(self._cage_rest_xmat.T @ R) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle)) > self._CAGE_DISTURB_ANGLE_DEG

    def _update_cage_picked_up(self):
        """Latch ``_cage_picked_up`` to True the first time the cage's
        z exceeds its at-rest baseline by more than the threshold."""
        try:
            sim = self._env.sim
            cage_bid = sim.model.body_name2id("cage_root")
            cage_z = float(sim.data.body_xpos[cage_bid][2])
            if cage_z - self._cage_rest_z > self._CAGE_LIFT_THRESHOLD:
                self._cage_picked_up = True
        except Exception:
            pass

    # ── Scoring / termination ───────────────────────────────────────

    def _check_termination(self, raw_obs):
        # Failures terminate immediately -- the post-success dwell is
        # specifically for the success path.
        if self._ball_fell_off_table():
            return True
        if self._cage_fell_off_table():
            self._failed = True
            return True

        score = self._compute_task_score(raw_obs)
        if score >= self._success_score_threshold:
            # First success tick: arm the dwell counter and keep
            # stepping. Subsequent ticks increment it; we only
            # terminate once the dwell window has elapsed.
            if self._post_success_ticks is None:
                self._post_success_ticks = 0
            else:
                self._post_success_ticks += 1
            if self._post_success_ticks >= POST_SUCCESS_DWELL_STEPS:
                return True
            return False

        # If the score dropped back below threshold (ball bounced off
        # the table briefly, etc.), reset the dwell counter so the
        # next sustained success has to dwell again.
        if self._post_success_ticks is not None:
            self._post_success_ticks = None
        return False

    def _cage_fell_off_table(self):
        """True if the cage has dropped below the failure floor threshold."""
        try:
            sim = self._env.sim
            cage_bid = sim.model.body_name2id("cage_root")
            return sim.data.body_xpos[cage_bid][2] < self._failure_floor_threshold
        except Exception:
            return False

    def _ball_fell_off_table(self):
        """True if the ball has dropped below the failure floor threshold.

        Reads the ball's body: ``cube_pos`` aliases the cage, the first
        dynamic object.
        """
        sim = self._env.sim
        return sim.data.body_xpos[sim.model.body_name2id("cube_main")][2] < self._failure_floor_threshold

    def _hole_world_pos(self):
        """World position of the hole centre, following the cage's current
        rotation (so tilting the cage moves the hole even without lift)."""
        try:
            sim = self._env.sim
            cage_bid = sim.model.body_name2id("cage_root")
            pos = np.array(sim.data.body_xpos[cage_bid], dtype=float)
            R = np.array(sim.data.body_xmat[cage_bid], dtype=float).reshape(3, 3)
            ci = [s / 2 for s in self._cage_inner]
            local_hole = np.array([self._hole_xy[0], self._hole_xy[1], -ci[2]])
            return pos + R @ local_hole
        except Exception:
            return None

    def _hole_cleared_progress(self, raw_obs):
        hole_pos = self._hole_world_pos()
        if hole_pos is None:
            return 0.0
        rise = hole_pos[2] - self._hole_rest_z
        clearance_needed = 2.0 * self._ball_radius
        return float(np.clip(rise / max(clearance_needed, 1e-6), 0.0, 1.0))

    def _ball_separated_from_cage(self):
        try:
            ball = self._env.get_object("cube")
            cage = self._env.get_object("cage")
            if ball is None or cage is None:
                return False
            if self._env.check_contact(ball.contact_geoms, cage.contact_geoms):
                return False
            hole_pos = self._hole_world_pos()
            if hole_pos is None:
                return False
            ball_bid = self._env.sim.model.body_name2id("cube_main")
            ball_z = float(self._env.sim.data.body_xpos[ball_bid][2])
            return ball_z < hole_pos[2] - self._ball_radius
        except Exception:
            return False

    def _ball_touches_table(self):
        try:
            ball = self._env.get_object("cube")
            if ball is None:
                return False
            return bool(self._env.check_contact(ball.contact_geoms, ["table_collision"]))
        except Exception:
            return False

    def _ball_outside_cage(self):
        """True iff the ball centre is at least one ball radius outside the
        cage's interior box, so a ball resting in the floor hole or pressed
        into a wall does not count."""
        sim = self._env.sim
        cage_bid = sim.model.body_name2id("cage_root")
        ball_bid = sim.model.body_name2id("cube_main")
        R = sim.data.body_xmat[cage_bid].reshape(3, 3)
        local = R.T @ (sim.data.body_xpos[ball_bid] - sim.data.body_xpos[cage_bid])
        inner_half = np.asarray(self._cage_inner) / 2 - self._wall_thickness
        return float(np.linalg.norm(np.maximum(np.abs(local) - inner_half, 0.0))) >= self._ball_radius

    def _ball_escaped_onto_table(self):
        """Ball outside the cage and touching the table: the success condition."""
        return self._ball_touches_table() and self._ball_outside_cage()

    def _compute_task_score(self, raw_obs):
        if getattr(self, "_failed", False):
            return 0.0
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score
