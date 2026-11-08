"""Peg Insertion — push a rod through a horizontal rectangular slot.

Task ID: ``ib-p1-recover-peg-insert-v1``

Scene:
- ``RectSlot`` static fixture (4 stacked boxes: a wide base plate, two
  narrow pillars sandwiching an empty gap, and a wide top plate) sits on
  the table. The gap between the pillars is a horizontal through-slot —
  open on both ±X ends of the fixture, with no geometry inside so it's
  collision-free.
- A cuboidal rod stands upright on the table in front of the slot.

The robot must grasp the rod, tip it onto its side so the long axis is
horizontal, align it with the slot's axis, and push it through the gap.
"""

import logging

import numpy as np
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward

logger = logging.getLogger(__name__)


# Baked-in geometry of the RectSlot fixture (4 legs + 4 stacked boxes,
# bounds: x,y ∈ [-0.10, 0.10], z ∈ [0, 0.35]; legs rest at local z=0).
# The rectangular through-slot runs along local ±X between the two
# middle pillars: its centre is at local (0, 0, 0.325), and its inner
# cross-section spans y ∈ [-0.017, 0.017], z ∈ [0.31, 0.34].
_SLOT_LENGTH = 0.20           # fixture depth along the slot axis
_SLOT_CENTRE_Z = 0.325        # height of the slot centre above the legs' base
_SLOT_HALF_OPENING_Y = 0.017  # gap half-width between the two pillars
_SLOT_HALF_OPENING_Z = 0.015  # slot height half-extent
_SLOT_HALF_FOOTPRINT = 0.10   # base-plate half-extent in fixture-local XY


class RecoverPegInsertV1(SingleArmBase):
    TASK_ID = "ib-p1-recover-peg-insert-v1"
    DESCRIPTION = (
        "Single arm: a cuboidal rod stands upright on the table and a "
        "rectangular through-slot fixture is mounted nearby with its "
        "opening facing the robot. Grasp the rod, tip it horizontal, "
        "align its long axis with the slot, and push it through."
    )

    # ── Config parsing ──────────────────────────────────────────────

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        # Peg geometry (cuboidal rod). The peg's long axis is its local +Z.
        # ``peg_half_cross`` is the half-extent of the two short sides
        # (X and Y in the peg's local frame).
        self._peg_half_cross = task_cfg.get("peg_half_cross", 0.012)
        self._peg_half_length = task_cfg.get("peg_half_length", 0.05)

        # Slot XY center on the table and per-reset jitter. The fixture
        # yaw is fixed (handled in reset()); only the centre is jittered.
        self._slot_center_xy = task_cfg.get("slot_center_xy", [0.18, 0.0])
        self._slot_center_jitter = task_cfg.get(
            "slot_center_jitter", [0.03, 0.05]
        )

        # Success thresholds. Insertion depth is expressed as the fraction
        # of the peg's length that must lie inside the slot. Default
        # 0.95 → 95% of the peg inside the slot. Converted to the
        # axial-distance window used by the scorer below.
        self._insert_fraction = task_cfg.get("insert_fraction", 0.95)
        self._lateral_tolerance = task_cfg.get("lateral_tolerance", 0.015)
        self._alignment_threshold = task_cfg.get("alignment_threshold", 0.92)
        # axial = (peg_centre − entrance) · n; leading-edge distance past
        # the entrance is axial + peg_half_length. To have `insert_fraction`
        # of the 2*peg_half_length-long peg past the entrance, we need
        # axial ≥ (2 * insert_fraction − 1) * peg_half_length.
        self._insert_depth_threshold = (
            (2.0 * self._insert_fraction - 1.0) * self._peg_half_length
        )
        # The same fraction must still be short of the exit plane, one slot
        # length past the entrance.
        self._max_insert_depth = _SLOT_LENGTH - self._insert_depth_threshold

        # Peg spawn XY range (table-relative)
        self._peg_spawn_xy_range = task_cfg.get(
            "peg_spawn_xy_range", [[-0.10, -0.02], [-0.15, 0.15]]
        )
        # Buffer added to the slot's half-footprint when keeping the peg
        # spawn out of the slot-board's XY region (stops peg from clipping
        # the legs at spawn time).
        self._slot_keepout = task_cfg.get("slot_keepout", 0.03)

        # Recovery perturbation — once per episode, the peg is rotated
        # about world +Z by ``yaw_deg`` total, applied gradually over
        # ``ramp_steps`` control-steps (so the tilt looks smooth, not a
        # snap). Firing is triggered by *alignment* rather than step
        # count: the push begins when the peg has entered the approach
        # corridor (within ``trigger_xz_tolerance`` of the hole's X and Z
        # coordinates), which is the moment the operator is actively
        # trying to insert and a misalignment is most disruptive.
        perturb_cfg = task_cfg.get("perturb", {})
        self._perturb_yaw_deg = float(perturb_cfg.get("yaw_deg", 15.0))
        self._perturb_ramp_steps = int(perturb_cfg.get("ramp_steps", 100))
        # Per-axis trigger tolerances — the hole is a horizontal
        # rectangle, so the axes aren't interchangeable. Each value is
        # the half-width of the trigger window around the hole centre
        # in that axis (i.e. |peg - hole| ≤ tol to count as aligned).
        # Y is the approach axis and uses a looser bound so the ramp
        # fires while the peg is still approaching the slot.
        self._perturb_trigger_x_tol = float(
            perturb_cfg.get("trigger_x_tolerance", 0.03)
        )
        self._perturb_trigger_y_tol = float(
            perturb_cfg.get("trigger_y_tolerance", 0.25)
        )
        self._perturb_trigger_z_tol = float(
            perturb_cfg.get("trigger_z_tolerance", 0.02)
        )

        # State populated in reset()
        self._hole_center_world = None
        self._hole_normal_world = None
        self._inserted = False
        # Perturbation bookkeeping. ``_perturb_fired`` is True as soon as
        # the ramp has started; it stays True for the rest of the episode.
        # ``_perturb_remaining_steps`` decrements from ramp_steps → 0
        # while the yaw is being applied.
        self._perturb_fired = False
        self._perturb_remaining_steps = 0
        self._perturb_yaw_per_step = 0.0

        self._staged_reward = StagedReward(
            stages=[
                Stage("grasp_peg", 0.15, lambda t, o: 1.0 if t._peg_is_grasped() else 0.0),
                Stage("lateral_alignment", 0.20, lambda t, o: t._lateral_alignment_score(o)),
                Stage("axis_alignment", 0.20, lambda t, o: t._axis_alignment_score(o)),
                Stage("insertion_depth", 0.45, lambda t, o: t._insertion_depth_score(o)),
            ],
            success_fn=lambda t, o: t._is_inserted(o),
        )

    # ── Object specs ────────────────────────────────────────────────

    def _make_objects(self):
        return [
            {
                "type": "box",
                "name": "cube",  # first dynamic obj aliased as "cube_*"
                # BoxObject takes full half-sizes [hx, hy, hz]; the peg's
                # long axis is its local +Z so length goes in the Z slot.
                "size": [self._peg_half_cross, self._peg_half_cross,
                         self._peg_half_length],
                "rgba": [0.85, 0.55, 0.15, 1.0],
                "density": 600,
                "friction": [0.8, 0.005, 0.0001],
                "joints": "default",
            },
        ]

    def _make_static_objects(self):
        return [
            {
                "type": "extras",
                "name": "slot",
                "extras_name": "RectSlot",
                "joints": None,
            },
        ]

    # ── Geometry helpers ────────────────────────────────────────────

    @staticmethod
    def _quat_zrot(angle_rad: float) -> np.ndarray:
        """MuJoCo [w, x, y, z] quat for a rotation about world +Z by angle_rad."""
        return np.array([np.cos(angle_rad / 2), 0.0, 0.0,
                         np.sin(angle_rad / 2)])

    # ── Reset ───────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._inserted = False
        self._staged_reward.reset()

        # Perturbation bookkeeping: fresh for each episode. The sign of
        # the yaw is sampled on the tick the ramp starts.
        self._perturb_fired = False
        self._perturb_remaining_steps = 0
        self._perturb_yaw_per_step = 0.0

        # Slot placement. RectSlot sits upright on its four legs (base at
        # local z=0) with its through-slot running along local +X, so no
        # pitch or roll is needed. The fixture lives to the robot's left
        # at world +Y, fixed at yaw = -π/2: local +X → world -Y so the
        # entrance opens toward the robot (-Y) and the exit is on the
        # far side (+Y). XY centre jitter adds per-reset variety without
        # rotating the fixture.
        jitter_x = self._rng.uniform(-self._slot_center_jitter[0],
                                     self._slot_center_jitter[0])
        jitter_y = self._rng.uniform(-self._slot_center_jitter[1],
                                     self._slot_center_jitter[1])
        slot_cx = float(self._slot_center_xy[0] + jitter_x)
        slot_cy = float(self._slot_center_xy[1] + jitter_y)
        slot_cz = float(self._table_height)

        yaw = -np.pi / 2
        q_slot = self._quat_zrot(yaw)
        self._place_static("slot", [slot_cx, slot_cy, slot_cz],
                           quat=q_slot.tolist())

        # Hole axis in world: local +X after yaw rotation.
        hole_axis_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        # Hole centre is the entrance face — one half-slot-length along
        # hole_axis_world from the fixture origin, lifted to the slot's
        # mid-height above the table.
        self._hole_center_world = np.array([
            slot_cx + 0.5 * _SLOT_LENGTH * hole_axis_world[0],
            slot_cy + 0.5 * _SLOT_LENGTH * hole_axis_world[1],
            slot_cz + _SLOT_CENTRE_Z,
        ])
        # Insertion-direction normal points from entrance toward exit so
        # ``axial = (peg - hole_center) · n`` grows positive as the peg
        # pushes deeper in.
        self._hole_normal_world = -hole_axis_world

        # ── Peg (cuboidal rod) spawn, standing upright on the table ──
        # The peg's local +Z is its long axis; leaving it upright makes the
        # height equal the full length (2 * peg_half_length) — the longest
        # dimension. The robot must tip it on its side to push it through
        # the horizontal slot. Fit check: peg_half_cross must be less than
        # both the slot's Y-gap half-width and Z-opening half-height.
        if self._peg_half_cross >= _SLOT_HALF_OPENING_Y or \
                self._peg_half_cross >= _SLOT_HALF_OPENING_Z:
            raise ValueError(
                f"peg_half_cross={self._peg_half_cross} too large: must fit "
                f"inside the slot opening "
                f"({_SLOT_HALF_OPENING_Y=}, {_SLOT_HALF_OPENING_Z=})."
            )
        # Peg spawns with a fixed orientation (identity quat) — no yaw
        # randomization, only position randomization below.
        q_peg = np.array([1.0, 0.0, 0.0, 0.0])

        # Rejection-sample the peg's XY position inside the spawn bounds,
        # staying outside the slot-board footprint (a 2*_SLOT_HALF_FOOTPRINT
        # square around (slot_cx, slot_cy)) plus a keep-out margin so the
        # peg can't clip the legs. Falls back to the farthest-valid corner
        # after many failed attempts (shouldn't happen with sane configs).
        xr = self._peg_spawn_xy_range[0]
        yr = self._peg_spawn_xy_range[1]
        keep = _SLOT_HALF_FOOTPRINT + self._slot_keepout
        peg_x = peg_y = None
        for _ in range(200):
            cand_x = float(self._rng.uniform(xr[0], xr[1]))
            cand_y = float(self._rng.uniform(yr[0], yr[1]))
            if abs(cand_x - slot_cx) >= keep or abs(cand_y - slot_cy) >= keep:
                peg_x, peg_y = cand_x, cand_y
                break
        if peg_x is None:
            peg_x = float(xr[0] if abs(xr[0] - slot_cx) > abs(xr[1] - slot_cx)
                          else xr[1])
            peg_y = float(yr[0] if abs(yr[0] - slot_cy) > abs(yr[1] - slot_cy)
                          else yr[1])
        peg_z = self._table_height + self._peg_half_length + 0.005

        self._place_object("cube", [peg_x, peg_y, peg_z],
                           quat=q_peg.tolist())

        self._env.sim.forward()
        obs = self._rebuild_obs()
        return obs, info

    # ── Peg orientation helpers ─────────────────────────────────────

    def _peg_world_axis(self, raw_obs) -> np.ndarray:
        """Return the peg's long axis direction in world frame (unit vector).

        Robosuite cylinders have their long axis along local +Z; rotate
        that by the peg's current body orientation.
        """
        sim = self._env.sim
        try:
            bid = sim.model.body_name2id("cube_main")
            xmat = sim.data.body_xmat[bid].reshape(3, 3)
            return xmat @ np.array([0.0, 0.0, 1.0])
        except Exception:
            return np.array([0.0, 0.0, 1.0])

    # ── Recovery perturbation ──────────────────────────────────────

    def _peg_body_id(self):
        try:
            return int(self._env.sim.model.body_name2id("cube_main"))
        except Exception:
            return None

    def _peg_is_grasped(self):
        """Permissive grasp detector: True if the peg's lowest point is
        clearly above the table AND it is in contact with any gripper geom
        (either fingerpad). Avoids robosuite's `_check_grasp`, which
        requires simultaneous contact on both fingerpads and often misses
        valid holds on a narrow cuboidal peg.
        """
        bid = self._peg_body_id()
        if bid is None:
            return False
        xmat = self._env.sim.data.body_xmat[bid].reshape(3, 3)
        half_extents = np.array([self._peg_half_cross, self._peg_half_cross,
                                 self._peg_half_length])
        lowest_z = float(self._env.sim.data.body_xpos[bid][2]
                         - np.abs(xmat[2]) @ half_extents)
        if lowest_z < self._table_height + 0.02:
            return False
        sim = self._env.sim
        gripper_geoms = [
            sim.model.geom_id2name(g) for g in range(sim.model.ngeom)
            if (sim.model.geom_id2name(g)
                and ("finger" in sim.model.geom_id2name(g)
                     or "gripper" in sim.model.geom_id2name(g)
                     or "taxel" in sim.model.geom_id2name(g)))
        ]
        if not gripper_geoms:
            return False
        peg_obj = self._env.get_object("cube")
        peg_geoms = peg_obj.contact_geoms if peg_obj is not None else ["cube"]
        return self._env.check_contact(peg_geoms, gripper_geoms)

    @staticmethod
    def _quat_mul_wxyz(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _peg_aligned_with_hole(self):
        """True when the peg is within per-axis tolerances of the hole on
        all three world axes. X/Z tolerances are tight (lateral alignment
        with the slot opening); the Y tolerance is looser since Y is the
        approach axis and we want the ramp to fire while the peg is still
        closing in on the slot.
        """
        if self._hole_center_world is None:
            return False
        bid = self._peg_body_id()
        if bid is None:
            return False
        peg_pos = self._env.sim.data.body_xpos[bid]
        dx = abs(float(peg_pos[0] - self._hole_center_world[0]))
        dy = abs(float(peg_pos[1] - self._hole_center_world[1]))
        dz = abs(float(peg_pos[2] - self._hole_center_world[2]))
        return (dx <= self._perturb_trigger_x_tol
                and dy <= self._perturb_trigger_y_tol
                and dz <= self._perturb_trigger_z_tol)

    def _apply_yaw_step(self, yaw_rad: float):
        """Rotate the peg's current orientation by ``yaw_rad`` about
        world +Z, leaving its position untouched.
        """
        q_dz = self._quat_zrot(yaw_rad)
        sim = self._env.sim
        peg_obj = self._env.get_object("cube")
        joint_name = peg_obj.joints[0]
        jid = sim.model.joint_name2id(joint_name)
        qposadr = int(sim.model.jnt_qposadr[jid])
        current = sim.data.qpos[qposadr:qposadr + 7].copy()
        pos = current[:3]
        quat = current[3:7]  # [w, x, y, z]
        new_quat = self._quat_mul_wxyz(q_dz, quat)
        sim.data.set_joint_qpos(
            joint_name, np.concatenate([pos, new_quat])
        )
        sim.forward()

    def _maybe_fire_perturbation(self):
        """Kick off the yaw ramp once the peg is grasped AND aligned with
        the hole (X/Z within tolerance). Once started, keep applying a
        per-step yaw increment until the total has been delivered. The
        ramp is smooth — each step contributes ``yaw_deg / ramp_steps``.
        """
        # If the ramp is active, apply one increment per tick.
        if self._perturb_remaining_steps > 0:
            self._apply_yaw_step(self._perturb_yaw_per_step)
            self._perturb_remaining_steps -= 1
            if self._perturb_remaining_steps == 0:
                logger.info(
                    "[recover_peg_insert]   ramp complete at step %s",
                    self._step_count + 1,
                )
            return

        if self._perturb_fired:
            return

        next_step = self._step_count + 1
        grasped = self._peg_is_grasped()
        aligned = self._peg_aligned_with_hole()

        if not (grasped and aligned):
            return

        # Arm the ramp.
        sign = 1.0 if self._rng.rand() < 0.5 else -1.0
        total_yaw = np.radians(self._perturb_yaw_deg) * sign
        self._perturb_yaw_per_step = total_yaw / self._perturb_ramp_steps
        self._perturb_remaining_steps = self._perturb_ramp_steps
        self._perturb_fired = True
        logger.info(
            "\n[recover_peg_insert] YAW RAMP STARTED step=%s "
            "total_yaw=%+.2f deg over %s ticks (%+.3f deg/tick)",
            next_step, np.degrees(total_yaw),
            self._perturb_ramp_steps, np.degrees(self._perturb_yaw_per_step),
        )
        # Apply the first increment this tick.
        self._apply_yaw_step(self._perturb_yaw_per_step)
        self._perturb_remaining_steps -= 1

    def step(self, action: np.ndarray):
        self._check_open()
        self._maybe_fire_perturbation()
        obs, reward, terminated, truncated, info = super().step(action)
        info["perturb_fired"] = self._perturb_fired
        info["perturb_ramp_remaining"] = self._perturb_remaining_steps
        return obs, reward, terminated, truncated, info

    # ── Scoring ─────────────────────────────────────────────────────

    # ── Staged-scoring helpers ───────────────────────────────────────

    def _peg_axial_lateral(self, raw_obs):
        """Peg centre's (axial, lateral) offset from the slot entrance.

        ``axial`` is measured along the insertion direction, ``lateral``
        is the distance from the slot's centre line.
        """
        rel = np.asarray(raw_obs["cube_pos"], dtype=float) - self._hole_center_world
        n = self._hole_normal_world
        axial = float(np.dot(rel, n))
        lateral = float(np.linalg.norm(rel - axial * n))
        return axial, lateral

    def _lateral_alignment_score(self, raw_obs):
        if "cube_pos" not in raw_obs or self._hole_center_world is None:
            return 0.0
        _, lateral = self._peg_axial_lateral(raw_obs)
        return max(0.0, 1.0 - lateral / 0.05)

    def _axis_alignment_score(self, raw_obs):
        # Only a held peg earns alignment credit (a peg spawned lying along
        # the slot axis would otherwise score it for free).
        if self._hole_normal_world is None or not self._peg_is_grasped():
            return 0.0
        peg_axis = self._peg_world_axis(raw_obs)
        align = abs(float(
            np.dot(peg_axis / (np.linalg.norm(peg_axis) + 1e-9), self._hole_normal_world)
        ))
        return max(0.0, (align - 0.5) / 0.5)

    def _insertion_depth_score(self, raw_obs):
        """Progress of the peg centre toward the insertion depth, credited
        only while the peg is on the slot's centre line and not pushed out
        past the far end."""
        if "cube_pos" not in raw_obs or self._hole_center_world is None:
            return 0.0
        axial, lateral = self._peg_axial_lateral(raw_obs)
        if lateral >= self._lateral_tolerance or axial > self._max_insert_depth:
            return 0.0
        return float(np.clip(
            (axial + self._peg_half_length * 0.3) /
            (self._insert_depth_threshold + self._peg_half_length * 0.3),
            0.0, 1.0,
        ))

    def _is_inserted(self, raw_obs):
        """True once the peg centre projects inside the hole (lateral
        distance < tolerance), the peg's long axis is aligned with the
        hole normal (|dot| > threshold), and ``insert_fraction`` of the
        peg lies between the slot's entrance and exit planes.
        """
        if "cube_pos" not in raw_obs or self._hole_center_world is None:
            return False
        axial, lateral = self._peg_axial_lateral(raw_obs)
        n = self._hole_normal_world
        peg_axis = self._peg_world_axis(raw_obs)
        align = abs(float(
            np.dot(peg_axis / (np.linalg.norm(peg_axis) + 1e-9), n)
        ))
        inserted = (lateral < self._lateral_tolerance
                    and align > self._alignment_threshold
                    and self._insert_depth_threshold <= axial <= self._max_insert_depth)
        if inserted:
            self._inserted = True
        return bool(inserted)

    def _compute_task_score(self, raw_obs):
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        if self._inserted:
            return True
        return self._object_fell_off_table(raw_obs, pos_key="cube_pos")
