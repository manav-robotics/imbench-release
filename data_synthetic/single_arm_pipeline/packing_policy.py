"""Scripted packing policy for ``ib-p7-packing-v1``.

The task spawns N (3-4) procedurally generated L-shaped pieces scattered
around a rectangular bin. Each piece has a known target slot inside the
bin (the centroid of its sub-boxes in the bin's inner-rect frame).

This policy is a clean rectilinear pick-and-place:

  * No wrist rotations for travel — the suction cup can grab a flat
    piece anywhere on its top face, so we always approach top-down with
    the default orientation. The pieces' yaw is identity at spawn (the
    task only randomises yaw when ``randomization.block_yaw.enabled`` is
    true, which it isn't by default). The only turn is the few-degree
    yaw correction while aligning a held piece over its slot.

  * Pick-side motion is decomposed into pure axis legs:
        Y first → X → Z descend → suck → Z lift.
    Place-side travel is the mirror (X first → Y), then closed loop on
    the held piece's measured pose, because the bin leaves only ~4 mm of
    slack per piece: lower until the piece is just above its packed
    neighbours, servo its centroid and yaw onto the slot (the suction
    weld can leave it several mm and degrees off), lower it straight in
    and release a few mm above the floor → Z lift.

  * Each piece is processed in turn; the policy walks a fixed phase
    machine and increments ``self._piece_idx`` once a piece is settled
    in its slot.

The whole thing assumes the env exposes the task internals: ``_blocks``
(MujocoObject list), ``_pieces`` (BPP layout dicts), ``_box`` (the bin
fixture), ``_box_pos`` (bin XY in world), and ``_table_height``.
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw — gripper Z down

# Wrist yaw the policy holds throughout the episode.
#
# Joint 6 (Panda wrist pitch) has a *highly asymmetric* limit range of
# [-0.0175, 3.7525] rad — i.e. it can't swing negative. Stock init_qpos
# puts it near pi - 0.2 ≈ 2.94 rad (well inside range, but with very
# little slack toward the upper limit). Rolling the cup by ±90 deg about
# world Z does NOT move joint 6 — it shifts joint 7 (axis = local Z at
# the wrist tip), which has a symmetric ±2.897 range with plenty of
# headroom either way. So a yaw offset doesn't help joint 6 at all and
# only burns rotational error budget on a setpoint that the OSC then has
# to slew through — which couples into position drift and made the arm
# go haywire on the first attempt.
#
# Hold the baseline orientation. Joint 7 absorbs the +pi/2 base rotation
# automatically and lands near -pi/4, which is centred in its range.
COMFORT_YAW = 0.0

GAIN_XY = 4.0
GAIN_Z = 4.0
ROT_GAIN = 0.5             # gentle pull back to COMFORT_QUAT — too high
                           # causes joint 7 to wind to its limit during
                           # cross-table travel because a world-fixed
                           # target orientation isn't compatible with
                           # constant joint angles as the EE moves.
MAX_ACTION_STEP = 0.5
MAX_ROT_STEP = 0.05        # tight cap on rotation deltas per tick.
                           # Larger values starve the linear axes when
                           # the wrist is mid-leg and let joint 7 race
                           # toward its upper limit.

POS_TOL_AXIS = 0.005       # per-axis arrival tolerance
POS_TOL_DESCEND = 0.004    # tighter on the final z-down leg
ORI_TOL = 0.10             # axis-angle norm

HOVER_HEIGHT = 0.20        # m above table for travel
# Pick descent target = body_z + thickness/2 (mid-height of the piece).
# robot0_eef_pos reports the grip_site (= suction snout) world position,
# and the suction weld activates on contact within a small tolerance —
# pressing the snout slightly into the piece is fine and gives a more
# reliable weld than hovering above the top face.

# Closed-loop placement for the snug bin (~4 mm of slack per piece). The
# held piece is lowered to PREPLACE_CLEARANCE above the floor (clear of
# its packed neighbours, which are _PIECE_THICKNESS tall), its measured
# centroid and yaw are servoed onto the slot, then it is lowered straight
# in to RELEASE_CLEARANCE and released. Heights are measured on the piece
# itself, so the cup-to-piece offset of the suction weld drops out.
PREPLACE_CLEARANCE = 0.025
RELEASE_CLEARANCE = 0.003
ALIGN_XY_TOL = 0.001
ALIGN_YAW_TOL = np.deg2rad(0.5)
ALIGN_HOLD = 8             # consecutive ticks inside both tolerances
ALIGN_TIMEOUT = 500
ALIGN_GAIN = 5.0
ALIGN_INT_GAIN = 0.02       # per-tick integral gain: removes the held piece's sag
YAW_GAIN = 2.0
YAW_MAX_STEP = 0.15         # the wrist drifts up to ~20 deg in transit; turn it back quickly
INSERT_TIMEOUT = 400

SETTLE_STEPS = 10
ROTATE_WRIST_MAX_STEPS = 120  # cap on the initial wrist slew before travel
SUCK_STEPS = 5             # ticks to hold while the weld engages
RELEASE_STEPS = 15         # ticks to hold while the weld disengages

# Each travel leg is broken into N intermediate waypoints between the
# start pose and the leg target. The policy commands waypoint i as the
# OSC setpoint for ~WAYPOINT_HOLD_TICKS ticks before stepping to i+1.
# Smaller per-step setpoint changes keep the OSC from saturating its
# control output and avoid winding the wrist against joint limits when
# the orientation is also being tracked. With 50 waypoints and a tight
# hold, a 0.5 m cross-table sweep happens in ~500-1000 ticks total but
# every intermediate goal is well within the controller's tracking
# envelope, so the EE actually arrives at each one rather than chasing
# a far-away setpoint.
WAYPOINTS_PER_SEGMENT = 35
WAYPOINT_HOLD_TICKS = 3          # ticks per waypoint
WAYPOINT_ARRIVE_TOL = 0.012      # m, per-axis tolerance to skip ahead
# Final-arrival tolerance on the leg target (after the last waypoint).
LEG_FINAL_TOL = 0.005

# Hard ceiling so a stuck phase can't hang the episode. Sized to cover
# all 50 waypoints × WAYPOINT_HOLD_TICKS plus headroom.
PHASE_TIMEOUT = 800


# Robot base yaw relative to world. The packing task rotates the Panda
# base +pi/2 about world Z (see packing_v1._build_env), and the OSC
# controller is configured with input_ref_frame="base" + input_type
# ="delta" — so any action delta we publish is interpreted in the
# rotated base frame, not world. We compute world-frame setpoints (from
# piece + bin world XY) and rotate the resulting world delta into the
# base frame here. Without this rotation the arm goes haywire because
# every commanded "+y" turns into world "-x" inside the controller.
_BASE_YAW = np.pi / 2  # world → base via -_BASE_YAW Z-rotation


# Geometry constants mirrored from the task module so the policy doesn't
# need to import private internals — these match the inner half-extents
# and base/piece thicknesses used by ``packing_v1.py`` (all 2× scaled).
_BOX_BASE_THICK = 0.005 * 2.0
_PIECE_THICKNESS = 0.01 * 2.0


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _yaw_quat(world_yaw: float) -> np.ndarray:
    """Top-down cup quat rotated by ``world_yaw`` about world Z (xyzw)."""
    half = 0.5 * world_yaw
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


COMFORT_QUAT = _yaw_quat(COMFORT_YAW)


def _world_to_base(vec_world: np.ndarray) -> np.ndarray:
    """Rotate a world-frame 3-vector into the rotated robot base frame.

    Base = world rotated by +_BASE_YAW about Z, so world → base is a
    -_BASE_YAW rotation about Z.
    """
    c = np.cos(-_BASE_YAW)
    s = np.sin(-_BASE_YAW)
    x, y, z = vec_world
    return np.array([c * x - s * y, s * x + c * y, z])


def _quat_err_axis_angle(cur_xyzw: np.ndarray, tgt_xyzw: np.ndarray) -> np.ndarray:
    cx, cy, cz, cw = cur_xyzw
    tx, ty, tz, tw = tgt_xyzw
    ccx, ccy, ccz, ccw = -cx, -cy, -cz, cw
    ex = tw * ccx + tx * ccw + ty * ccz - tz * ccy
    ey = tw * ccy - tx * ccz + ty * ccw + tz * ccx
    ez = tw * ccz + tx * ccy - ty * ccx + tz * ccw
    ew = tw * ccw - tx * ccx - ty * ccy - tz * ccz
    if ew < 0.0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    sin_half = float(np.sqrt(ex * ex + ey * ey + ez * ez))
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, ew)
    axis = np.array([ex, ey, ez]) / sin_half
    return axis * angle


class PackingPolicy(PolicyInterface):
    """Pick each L-piece and drop it onto its target slot inside the bin."""

    def __init__(self):
        self.reset()

    # ── Reset ───────────────────────────────────────────────────────
    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0
        self._piece_idx = 0
        self._n_pieces = 0
        self._initialised = False

        # Per-piece cached targets (filled once the piece phase begins).
        self._pick_xy: np.ndarray | None = None
        self._pick_mid_z: float = 0.0
        self._place_xy: np.ndarray | None = None
        self._place_z: float = 0.0
        self._table_z: float = 0.9
        self._anchor_geom_id: int | None = None
        # XY offset from the cup snout to the anchor sub-box centre at
        # the moment the suction weld engages. The weld locks the piece
        # to the cup with whatever sub-mm offset the contact resolved at
        # — usually the cup overshoots the geom centre by 5-15 mm. We
        # snapshot it after suck so the place phases can subtract it
        # and land the *piece* (not the cup) on the target slot.
        self._cup_to_anchor_xy: np.ndarray | None = None

        # Waypoint planner state. _leg_start, _leg_target define the
        # current straight-line segment; _wp_idx tracks which of the
        # WAYPOINTS_PER_SEGMENT intermediate goals we've published; and
        # _wp_ticks counts how long we've held the current waypoint as
        # the active OSC setpoint. ``_leg_phase`` is the phase string
        # this leg belongs to (used to detect a new leg starting).
        self._leg_phase: str | None = None
        self._leg_start: np.ndarray | None = None
        self._leg_target: np.ndarray | None = None
        self._leg_target_snapshot: np.ndarray | None = None
        self._wp_idx: int = 0
        self._wp_ticks: int = 0
        self._leg_finished: bool = False

        self._align_ticks = 0
        self._hold_pos: np.ndarray | None = None
        self._servo_int = np.zeros(3)

        self._last_action = np.zeros(7)
        self._last_action[6] = -1.0

    # ── Phase machinery ─────────────────────────────────────────────
    def _advance(self, next_phase: str):
        print(f"  [piece {self._piece_idx}] [{self.step_in_phase:4d} ticks] "
              f"{self.phase} -> {next_phase}")
        self.phase = next_phase
        self.step_in_phase = 0
        self._servo_int = np.zeros(3)
        # Force the leg planner to re-snapshot from the current EE pose.
        self._leg_phase = None

    # ── Leg planner: discretise a straight-line segment ─────────────
    def _leg_waypoint(self, eef_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the current intermediate waypoint along the segment
        from ``_leg_start`` to ``target``, advancing through up to
        ``WAYPOINTS_PER_SEGMENT`` evenly-spaced sub-goals.

        On a new leg (phase changed since last call OR the target moved
        materially), the start is reset to the current EE position so
        the segment is always anchored to where the arm actually is —
        not where we wished it had ended up last leg. After the final
        waypoint, we stay at ``target`` for ``LEG_FINAL_HOLD`` ticks to
        let OSC converge tightly before the phase reads its arrival
        check.
        """
        # New leg? Reset the planner.
        new_target = (
            self._leg_phase != self.phase
            or self._leg_target is None
            or np.linalg.norm(self._leg_target - target) > 1e-6
        )
        if new_target:
            self._leg_phase = self.phase
            self._leg_start = eef_pos.copy()
            self._leg_target = target.copy()
            self._wp_idx = 0
            self._wp_ticks = 0
            self._leg_finished = False

        # If we're already at the last waypoint and have held it long
        # enough, stay at target.
        if self._wp_idx >= WAYPOINTS_PER_SEGMENT:
            self._leg_finished = True
            return self._leg_target.copy()

        # Compute the i-th waypoint on the segment.
        t = (self._wp_idx + 1) / WAYPOINTS_PER_SEGMENT
        wp = self._leg_start + t * (self._leg_target - self._leg_start)

        # Advance the waypoint counter once we've held it long enough
        # OR the EE has actually arrived at it (whichever first).
        self._wp_ticks += 1
        wp_arrived = bool(np.all(np.abs(wp - eef_pos) < WAYPOINT_ARRIVE_TOL))
        if self._wp_ticks >= WAYPOINT_HOLD_TICKS or wp_arrived:
            self._wp_idx += 1
            self._wp_ticks = 0
        return wp

    def _leg_done(self, eef_pos: np.ndarray, axis_idx: int | None = None,
                  tol: float = LEG_FINAL_TOL) -> bool:
        """True iff the leg planner has run through every waypoint and
        the EE is within ``tol`` of the final target on the relevant
        axis (or all axes if ``axis_idx`` is None)."""
        if not self._leg_finished or self._leg_target is None:
            return False
        diff = np.abs(self._leg_target - eef_pos)
        if axis_idx is None:
            return bool(np.all(diff < tol))
        return bool(diff[axis_idx] < tol)

    def _cup_place_xy(self) -> np.ndarray:
        """World XY the cup must reach so the anchor sub-box lands on
        ``self._place_xy``. Subtracts the cup→anchor offset captured at
        weld engagement; falls back to the raw place_xy before suck.
        """
        if self._cup_to_anchor_xy is None:
            return self._place_xy
        return self._place_xy - self._cup_to_anchor_xy

    def _slot_error(self, env):
        """Current piece vs its slot: (world XY to move, yaw to turn [rad], bottom height above the floor)."""
        sim = env._env.sim
        name = env._block_names[self._piece_idx]
        box = sim.model.body_name2id(env._box.root_body)
        box_rot = sim.data.body_xmat[box].reshape(3, 3)
        cx, cy = env._piece_by_name[name]["centroid"]
        target = sim.data.body_xpos[box] + box_rot @ np.array([cx, cy, _BOX_BASE_THICK + _PIECE_THICKNESS / 2])
        centroid = env._piece_centroid(name)
        body = sim.model.body_name2id(env._block_by_name[name].root_body)
        rel = box_rot.T @ sim.data.body_xmat[body].reshape(3, 3)
        yaw = float(np.arctan2(rel[1, 0], rel[0, 0]))
        return target[:2] - centroid[:2], -yaw, float(centroid[2] - target[2])

    def _servo_piece(self, eef_pos, eef_quat, env, clearance, gain_xy=ALIGN_GAIN, turn=True):
        """Action moving the held piece's centroid onto its slot at ``clearance`` above the floor (PI on position)."""
        dxy, dyaw, height = self._slot_error(env)
        err = np.array([dxy[0], dxy[1], clearance - height])
        self._servo_int = np.clip(self._servo_int + err, -0.5, 0.5)
        target = eef_pos + err + ALIGN_INT_GAIN * self._servo_int
        action = self._axis_move(eef_pos, eef_quat, target, gripper=1.0, gain_xy=gain_xy, gain_z=6.0,
                                 lock_roll_pitch=True)
        if turn:
            action[5] = self._yaw_command(dyaw)
        return action, dxy, dyaw, height

    @staticmethod
    def _yaw_command(dyaw):
        return float(np.clip(dyaw * YAW_GAIN, -YAW_MAX_STEP, YAW_MAX_STEP))

    # ── Per-piece target computation ────────────────────────────────
    def _setup_piece(self, env):
        """Cache pick + place world coordinates for the current piece.

        Queries the sim directly for the L-piece's collision geoms — one
        ``col_*`` geom per sub-box — and picks the anchor whose half-
        extents give the most boundary clearance for the suction snout.
        The geom world position from ``sim.data.geom_xpos`` is the
        ground-truth current pose, independent of any cached metadata
        in ``piece['centroid']`` etc. that could go out of sync if the
        piece settles or drifts after spawn.
        """
        i = self._piece_idx
        sim = env._env.sim

        block = env._blocks[i]
        # New piece — discard any offset captured for the previous one.
        self._cup_to_anchor_xy = None

        # Collect the piece's collision sub-boxes from the sim. Each
        # geom name is "<naming_prefix>col_<k>" per the L-piece XML.
        prefix = block.naming_prefix
        sub_geoms = []  # list of (gid, world_xpos, body_local_pos, half_extents)
        for gid in range(sim.model.ngeom):
            gname = sim.model.geom_id2name(gid)
            if not gname or not gname.startswith(prefix) or "col_" not in gname:
                continue
            world = np.array(sim.data.geom_xpos[gid])
            local = np.array(sim.model.geom_pos[gid])
            half = np.array(sim.model.geom_size[gid])
            sub_geoms.append((gid, world, local, half))
        if not sub_geoms:
            raise RuntimeError(
                f"No col_* geoms found for piece {i} (prefix={prefix!r})")

        # Pick the geom with the largest minimum half-extent — i.e. the
        # blockiest sub-box, where the round suction snout has the most
        # margin to every edge. min(half_x, half_y) is the limiting
        # dimension for a circular snout of radius 15 mm.
        anchor = max(sub_geoms, key=lambda g: min(g[3][0], g[3][1]))
        anchor_gid, anchor_world, anchor_local, anchor_half = anchor
        # Cache the anchor geom id so we can re-query its world XY
        # later (e.g. to compute the cup→anchor offset that the suction
        # weld locks in at activation time).
        self._anchor_geom_id = anchor_gid

        # Pick XY: the geom's current world centre. Pick mid-Z: same.
        # geom_xpos points at the geom centre, which is already the
        # mid-height of the sub-box (body origin is at piece bottom and
        # the geom sits at z = thickness/2 in body-local frame).
        self._pick_xy = anchor_world[:2].copy()
        self._pick_mid_z = float(anchor_world[2])

        # Place: translate the anchor sub-box to its slot inside the
        # bin. Three frames are at play:
        #   * world           — sim global
        #   * inner-rect      — bin centre is origin (BPP layout frame)
        #   * body-local      — piece centroid is origin
        # The anchor sub-box's correct slot is its **inner-rect**
        # coordinate ``(sx, sy)`` (that's what makes the L's tile the
        # bin's interior). The geom's body-local pos from MuJoCo is
        # ``(sx - cx, sy - cy)`` — the offset from the piece centroid.
        # So the slot in inner-rect coords is body_local + centroid,
        # and the slot in world coords is bin_world_xy + that.
        # The pieces are pre-shrunk in geometry (see _PIECE_XY_SHRINK in
        # packing_v1.py), so the anchor lands in its true tile slot —
        # the gap from neighbouring pieces is built into the sub-boxes,
        # not added at placement time.
        box_bid = sim.model.body_name2id(env._box.root_body)
        box_pos = np.array(sim.data.body_xpos[box_bid])
        piece = env._pieces[i]
        cx, cy = piece["centroid"]
        anchor_inner_rect = anchor_local[:2] + np.array([cx, cy])
        self._place_xy = box_pos[:2] + anchor_inner_rect

        # Place mid-Z: floor + thickness/2 (matches the pick mid-height
        # convention so the cup-to-piece offset stays consistent).
        floor_z = float(box_pos[2]) + _BOX_BASE_THICK
        self._place_z = floor_z + _PIECE_THICKNESS / 2

        print(f"  [piece {i}] anchor_local=({anchor_local[0]:+.3f},"
              f"{anchor_local[1]:+.3f}) "
              f"half=({anchor_half[0]:.3f},{anchor_half[1]:.3f}) "
              f"pick_xy={np.round(self._pick_xy, 3)} "
              f"pick_mid_z={self._pick_mid_z:.3f} "
              f"place_xy={np.round(self._place_xy, 3)} "
              f"place_z={self._place_z:.3f}")

    # ── Action helper: rectilinear axis-only move ───────────────────
    def _axis_move(self, eef_pos, eef_quat, target_pos, gripper,
                   gain_xy=GAIN_XY, gain_z=GAIN_Z, max_step=MAX_ACTION_STEP,
                   track_ori=False, lock_roll_pitch=False):
        """Servo to ``target_pos`` (world frame).

        ``target_pos`` and ``eef_pos`` are both world-frame; the resulting
        delta is rotated into the base frame because the OSC config in
        this task is ``input_ref_frame=base``. Without this rotation the
        action gets misinterpreted as a base-frame delta and the arm
        moves at a 90° angle to the intended direction.

        Orientation tracking is OFF by default. Full orientation
        tracking pins joint 7 against its upper limit during long
        cross-table sweeps because a world-fixed top-down target isn't
        a constant-joint solution.

        Two opt-ins for finer control:
        * ``track_ori=True`` — full top-down orientation servo. Use only
          on short legs where joint 7 has slack.
        * ``lock_roll_pitch=True`` — servo only the roll & pitch
          components of the orientation error, leaving yaw (about
          world-Z, i.e. joint 7) untouched. This stops the cup from
          tilting on contact during the pick descent without winding
          joint 7. Critical during ``pick_z_down`` and ``suck``: when
          the snout touches the piece off-centre, the contact reaction
          tilts the wrist a few degrees, and that tilt locks into the
          weld — so the held piece sits skewed relative to the cup
          and tips on placement.
        """
        action = np.zeros(7)
        delta_world = target_pos - eef_pos
        delta_base = _world_to_base(delta_world)
        action[0] = float(np.clip(delta_base[0] * gain_xy, -max_step, max_step))
        action[1] = float(np.clip(delta_base[1] * gain_xy, -max_step, max_step))
        action[2] = float(np.clip(delta_base[2] * gain_z, -max_step, max_step))
        if track_ori or lock_roll_pitch:
            ori_err_world = _quat_err_axis_angle(eef_quat, COMFORT_QUAT)
            ori_err_base = _world_to_base(ori_err_world)
            if lock_roll_pitch and not track_ori:
                # Zero out the world-Z (= base-Z) component so we only
                # correct roll & pitch. Joint 7 stays free.
                ori_err_base = ori_err_base.copy()
                ori_err_base[2] = 0.0
            action[3:6] = np.clip(ori_err_base * ROT_GAIN,
                                  -MAX_ROT_STEP, MAX_ROT_STEP)
        # else: action[3:6] stays zero — no orientation delta commanded.
        action[6] = gripper
        return action

    # ── Main act() ──────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", np.array([0.0, 0.0, 0.0, 1.0]))

        if env is not None and not self._initialised:
            self._n_pieces = len(env._blocks)
            self._table_z = float(getattr(env, "_table_height", 0.9))
            self._initialised = True
            print(f"[packing] {self._n_pieces} pieces to pack")

        hover_z = self._table_z + HOVER_HEIGHT

        # ── 0. Settle: let initial controller transients dissipate ──
        # We hold the *baseline* TOP_DOWN_QUAT here so the OSC controller
        # doesn't fight the boot-up state; the wrist slew happens in the
        # next phase, where we hold position and only command rotation.
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = -1.0
            ori_err = _world_to_base(
                _quat_err_axis_angle(eef_quat, TOP_DOWN_QUAT))
            action[3:6] = np.clip(ori_err * ROT_GAIN, -MAX_ROT_STEP, MAX_ROT_STEP)
            self.step_in_phase += 1
            if self.step_in_phase >= SETTLE_STEPS:
                self._advance("rotate_wrist")
            self._last_action = action
            return action

        # ── 0b. Rotate wrist to the comfort yaw before any travel ──
        # Pure rotation: position target = current EE position, so the
        # arm stays put while joint 6 slews to mid-range. This keeps the
        # wrist out of its rotation limits during the subsequent
        # rectilinear travel and descent legs.
        if self.phase == "rotate_wrist":
            target = eef_pos.copy()
            action = self._axis_move(eef_pos, eef_quat, target, gripper=-1.0,
                                     track_ori=True)
            ori_err_norm = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, COMFORT_QUAT)))
            self.step_in_phase += 1
            if (ori_err_norm < ORI_TOL
                    or self.step_in_phase >= ROTATE_WRIST_MAX_STEPS):
                if env is not None:
                    self._setup_piece(env)
                self._advance("pick_y")
            self._last_action = action
            return action

        # All remaining phases require env access.
        if env is None:
            return self._last_action.copy()

        # If we've packed all pieces, hold open.
        if self._piece_idx >= self._n_pieces:
            action = np.zeros(7)
            action[6] = -1.0
            self.step_in_phase += 1
            self.phase = "done"
            self._last_action = action
            return action

        # ─────────────────────────────────────────────────────────────
        #  PICK side: Y → X → Z (descend) → suck → lift
        # ─────────────────────────────────────────────────────────────

        # ── 1a. Move along Y to the pick column, holding current X/Z ──
        # The leg target snapshots once at the start of the phase (when
        # _leg_phase changes); after that, _leg_start is fixed and we
        # walk through 50 evenly-spaced waypoints to it. We don't keep
        # rebuilding ``target`` from a live ``eef_pos[0]`` because that
        # would chase the EE's own X drift during the Y travel.
        if self.phase == "pick_y":
            if self._leg_phase != "pick_y":
                self._leg_target_snapshot = np.array(
                    [eef_pos[0], self._pick_xy[1], hover_z])
            wp = self._leg_waypoint(eef_pos, self._leg_target_snapshot)
            action = self._axis_move(eef_pos, eef_quat, wp, gripper=-1.0)
            self.step_in_phase += 1
            if self._leg_done(eef_pos, axis_idx=1):
                self._advance("pick_x")
            elif self.step_in_phase >= PHASE_TIMEOUT:
                self._advance("pick_x")
            self._last_action = action
            return action

        # ── 1b. Move along X to the pick row, holding Y at pick line ──
        if self.phase == "pick_x":
            if self._leg_phase != "pick_x":
                self._leg_target_snapshot = np.array(
                    [self._pick_xy[0], self._pick_xy[1], hover_z])
            wp = self._leg_waypoint(eef_pos, self._leg_target_snapshot)
            action = self._axis_move(eef_pos, eef_quat, wp, gripper=-1.0)
            self.step_in_phase += 1
            if self._leg_done(eef_pos, axis_idx=0):
                self._advance("pick_z_down")
            elif self.step_in_phase >= PHASE_TIMEOUT:
                self._advance("pick_z_down")
            self._last_action = action
            return action

        # ── 1c. Descend straight down to the piece's mid-height ──
        # Pressing into the piece by half a thickness is fine — the
        # suction weld activates on contact and a small overshoot just
        # firms up the seal. The descent is short enough that the
        # waypoint chunking still helps avoid OSC overshoot but the
        # arrival check is what matters.
        if self.phase == "pick_z_down":
            if self._leg_phase != "pick_z_down":
                self._leg_target_snapshot = np.array(
                    [self._pick_xy[0], self._pick_xy[1], self._pick_mid_z])
            wp = self._leg_waypoint(eef_pos, self._leg_target_snapshot)
            # Lock roll & pitch so contact with the piece can't tilt
            # the cup; leave yaw (joint 7) free.
            action = self._axis_move(eef_pos, eef_quat, wp, gripper=-1.0,
                                     gain_z=6.0, lock_roll_pitch=True)
            self.step_in_phase += 1
            if self._leg_done(eef_pos, axis_idx=2, tol=POS_TOL_DESCEND):
                self._advance("suck")
            elif self.step_in_phase >= PHASE_TIMEOUT:
                self._advance("suck")
            self._last_action = action
            return action

        # ── 2. Engage suction, hold pose ──
        # Same roll/pitch lock — the weld constraint snapshots the cup
        # orientation at activation, so any tilt induced during this
        # phase persists for the rest of the transport.
        if self.phase == "suck":
            target = np.array([
                self._pick_xy[0], self._pick_xy[1], self._pick_mid_z,
            ])
            action = self._axis_move(eef_pos, eef_quat, target, gripper=1.0,
                                     gain_z=6.0, lock_roll_pitch=True)
            self.step_in_phase += 1
            if self.step_in_phase >= SUCK_STEPS:
                # Capture the cup→anchor XY offset right before we lift.
                # The weld is now established and the piece moves
                # rigidly with the cup; whatever XY mismatch the contact
                # resolved at is locked in for the rest of the trip.
                # Subtracting this from the place target lets us land
                # the *anchor sub-box* (and therefore the whole tile)
                # exactly on its slot, instead of the cup snout.
                if self._anchor_geom_id is not None:
                    sim = env._env.sim
                    anchor_xy = np.array(
                        sim.data.geom_xpos[self._anchor_geom_id][:2])
                    self._cup_to_anchor_xy = anchor_xy - eef_pos[:2]
                    print(f"  [piece {self._piece_idx}] "
                          f"cup→anchor offset = "
                          f"({self._cup_to_anchor_xy[0]:+.4f}, "
                          f"{self._cup_to_anchor_xy[1]:+.4f})")
                self._advance("lift")
            self._last_action = action
            return action

        # ── 3. Lift straight up to hover height ──
        # Keep the lock during the lift too: while the cup pulls the
        # piece off the table, asymmetric mass distribution would
        # otherwise pendulum-tilt the wrist.
        if self.phase == "lift":
            if self._leg_phase != "lift":
                self._leg_target_snapshot = np.array(
                    [eef_pos[0], eef_pos[1], hover_z])
            wp = self._leg_waypoint(eef_pos, self._leg_target_snapshot)
            action = self._axis_move(eef_pos, eef_quat, wp, gripper=1.0,
                                     lock_roll_pitch=True)
            self.step_in_phase += 1
            if self._leg_done(eef_pos, axis_idx=2):
                self._advance("place_x")
            elif self.step_in_phase >= PHASE_TIMEOUT:
                self._advance("place_x")
            self._last_action = action
            return action

        # ─────────────────────────────────────────────────────────────
        #  PLACE side: X → Y → Z (descend) → release → lift → next
        # ─────────────────────────────────────────────────────────────

        # ── 4a. Move along X to the place column ──
        # All place targets use _cup_place_xy() — the cup goes to
        # (place_xy − cup_to_anchor_offset) so the anchor (and the
        # piece welded to it) ends up on place_xy.
        if self.phase == "place_x":
            cup_xy = self._cup_place_xy()
            if self._leg_phase != "place_x":
                self._leg_target_snapshot = np.array(
                    [cup_xy[0], eef_pos[1], hover_z])
            wp = self._leg_waypoint(eef_pos, self._leg_target_snapshot)
            action = self._axis_move(eef_pos, eef_quat, wp, gripper=1.0)
            self.step_in_phase += 1
            if self._leg_done(eef_pos, axis_idx=0):
                self._advance("place_y")
            elif self.step_in_phase >= PHASE_TIMEOUT:
                self._advance("place_y")
            self._last_action = action
            return action

        # ── 4b. Move along Y to the place row ──
        if self.phase == "place_y":
            cup_xy = self._cup_place_xy()
            if self._leg_phase != "place_y":
                self._leg_target_snapshot = np.array(
                    [cup_xy[0], cup_xy[1], hover_z])
            wp = self._leg_waypoint(eef_pos, self._leg_target_snapshot)
            action = self._axis_move(eef_pos, eef_quat, wp, gripper=1.0)
            # Start turning the held piece back to its slot yaw on the way.
            action[5] = self._yaw_command(self._slot_error(env)[1])
            self.step_in_phase += 1
            if self._leg_done(eef_pos, axis_idx=1):
                self._advance("place_z_down")
            elif self.step_in_phase >= PHASE_TIMEOUT:
                self._advance("place_z_down")
            self._last_action = action
            return action

        # ── 4c. Descend straight down to the bin floor ──
        # Lock roll & pitch so the held piece stays flat while it
        # approaches the floor — releasing a tilted piece guarantees
        # the success check fails (>5° tilt threshold).
        if self.phase == "place_z_down":
            # Lower until the piece's bottom is PREPLACE_CLEARANCE above
            # the floor, above its packed neighbours, still turning it.
            action, _, _, height = self._servo_piece(eef_pos, eef_quat, env, PREPLACE_CLEARANCE,
                                                     gain_xy=GAIN_XY)
            self.step_in_phase += 1
            if abs(height - PREPLACE_CLEARANCE) < 0.003 or self.step_in_phase >= PHASE_TIMEOUT:
                self._align_ticks = 0
                self._advance("align")
            self._last_action = action
            return action

        # ── 4d. Servo the piece's measured centroid and yaw onto its slot ──
        if self.phase == "align":
            action, dxy, dyaw, _ = self._servo_piece(eef_pos, eef_quat, env, PREPLACE_CLEARANCE)
            inside = np.linalg.norm(dxy) < ALIGN_XY_TOL and abs(dyaw) < ALIGN_YAW_TOL
            self._align_ticks = self._align_ticks + 1 if inside else 0
            self.step_in_phase += 1
            if self._align_ticks >= ALIGN_HOLD or self.step_in_phase >= ALIGN_TIMEOUT:
                self._advance("insert")
            self._last_action = action
            return action

        # ── 4e. Lower straight into the slot, still holding XY and yaw ──
        if self.phase == "insert":
            action, _, _, height = self._servo_piece(eef_pos, eef_quat, env, RELEASE_CLEARANCE)
            self.step_in_phase += 1
            if height < RELEASE_CLEARANCE + 0.001 or self.step_in_phase >= INSERT_TIMEOUT:
                self._hold_pos = eef_pos.copy()
                self._advance("release")
            self._last_action = action
            return action

        # ── 5. Release suction, hold pose so the piece settles ──
        if self.phase == "release":
            # Hold the pose the piece was inserted at while the weld lets go.
            action = self._axis_move(eef_pos, eef_quat, self._hold_pos, gripper=-1.0,
                                     gain_z=6.0)
            self.step_in_phase += 1
            if self.step_in_phase >= RELEASE_STEPS:
                # Commit the piece count here. The piece is in its slot
                # and the cup is releasing — if the env terminates on
                # success during retract, _piece_idx already reflects
                # reality (vs. incrementing in retract, which never
                # finishes when the score-check fires mid-phase).
                self._piece_idx += 1
                self._advance("retract")
            self._last_action = action
            return action

        # ── 6. Retract straight up so the next pick-Y leg has clearance ──
        if self.phase == "retract":
            if self._leg_phase != "retract":
                self._leg_target_snapshot = np.array(
                    [eef_pos[0], eef_pos[1], hover_z])
            wp = self._leg_waypoint(eef_pos, self._leg_target_snapshot)
            action = self._axis_move(eef_pos, eef_quat, wp, gripper=-1.0)
            self.step_in_phase += 1
            done = self._leg_done(eef_pos, axis_idx=2)
            if done or self.step_in_phase >= PHASE_TIMEOUT:
                if self._piece_idx >= self._n_pieces:
                    self._advance("done")
                else:
                    self._setup_piece(env)
                    self._advance("pick_y")
            self._last_action = action
            return action

        # ── 7. Done: hold open at current pose ──
        action = np.zeros(7)
        action[6] = -1.0
        self.step_in_phase += 1
        self._last_action = action
        return action


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env = ib.make("ib-p7-packing-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = PackingPolicy()

    for ep in range(3):
        obs, info = env.reset(seed=ep * 17 + 3)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===  pieces={len(env._blocks)}")
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated
        print(f"Episode {ep}  score={info.get('task_score', 0.0):.3f}  "
              f"phase={policy.phase}  packed={policy._piece_idx}/{policy._n_pieces}")
    env.close()
