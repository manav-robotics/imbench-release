"""Scripted policy for ``ib-p7-shape-stack-v1``.

Strategy
--------
Each reset spawns 3-5 differently-shaped objects from a fixed catalogue
(cube, small_cube, cylinder, hex_prism, cone_frustum). Success = exactly
one object touches the table, nothing touches the gripper, and that
state holds for 100 ticks.

Plan: pick the widest, flat-topped object as the BASE and leave it
where it is; pick up every other object and stack it on top of the base.
The stack centre xy = base xy. Each new placement z is derived by
adding the previous-top z to the new object's half-height plus a small
clearance.

Object body names are ``<archetype>_main`` for primitives, ``<archetype>_root``
for composites. Half-heights and footprint radii come from the env's
selected_archetypes list.

Phase machine (per object i to stack):
  settle -> plan
  for each i in stack_order:
    pre_grasp_i  -> descend_i -> close_i -> lift_i
    -> transport_i -> lower_i -> release_i -> retract_i
  retreat -> wait_stable -> done
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw -- gripper Z down

# Vertical clearances above table / current stack top.
PRE_GRASP_HEIGHT_ABOVE_TBL = 0.20    # hover height before descending onto an obj
TRANSPORT_CLEARANCE        = 0.12    # clearance above the stack during transport
STACK_PLACE_CLEARANCE      = 0.003   # hover 3 mm above natural rest before
                                     # release. NO press-to-seat — pressing
                                     # was crushing the support stack with
                                     # accumulated OSC force.
RETREAT_HEIGHT_ABOVE_TBL   = 0.30    # final retract height
RETREAT_AWAY_XY            = np.array([-0.10, -0.20])  # bias retreat off the stack

# Tolerances.
POS_TOL_XY   = 0.012
POS_TOL_Z    = 0.010
ORI_TOL      = 0.10
LIFT_Z_TOL   = 0.015

# Position-control gains.
GAIN_XY  = 4.0
GAIN_Z   = 4.0
ROT_GAIN = 2.5
MAX_ACTION_STEP = 0.6

RAMP_TICKS        = 20
RAMP_START_FRAC   = 0.2
TARGET_CHANGE_TOL = 0.015

# Phase budgets (ticks at 100 Hz). Per-object pick & place needs ~120
# ticks total; with up to 4 stack-ons + 100 stable tick window we want
# ~700 ticks. The default YAML horizon (500) is too tight; the runner
# bumps the env's MAX_EPISODE_STEPS at construction time.
SETTLE_STEPS         = 15
PRE_GRASP_MAX_STEPS  = 80
APPROACH_MAX_STEPS   = 80    # drop to just above object, re-align xy
DESCEND_MAX_STEPS    = 90    # final drop — exits early on z_reached, capped here
CLOSE_STEPS          = 25
LIFT_MAX_STEPS       = 60
TRANSPORT_MAX_STEPS  = 220   # full xy traverse from spawn to base; under-budget
                             # causes the placement to land off the support
HOVER_ALIGN_MAX_STEPS = 90   # before lowering, hover above target, lock xy+ori
LOWER_MAX_STEPS      = 160   # pure-z descent at hovering xy
RELEASE_STEPS        = 50    # open gripper + dwell so the placed object
                             # settles onto the support before we retract
RETRACT_MAX_STEPS    = 50
RETREAT_MAX_STEPS    = 60
WAIT_STABLE_STEPS    = 110   # just over task's 100-tick stable window

# Composite descent compensation: composite half_h is conservative
# (e.g. cone_frustum reports 0.030 but actually spans ~0.080). Use a
# table of measured top-of-object z offsets relative to body centre so
# placement heights are correct.
ARCHETYPE_TOP_OVER_CENTER = {
    "cube":         0.025,
    "small_cube":   0.018,
    "cylinder":     0.020,
    "hex_prism":    0.022,
    # cone_frustum: 3 stacked cylinders span body z [-0.040, +0.040] so
    # the body-frame top is +0.040 above body centre.
    "cone_frustum": 0.040,
}

ARCHETYPE_BOTTOM_BELOW_CENTER = {
    "cube":         0.025,
    "small_cube":   0.018,
    "cylinder":     0.020,
    "hex_prism":    0.022,
    "cone_frustum": 0.040,
}

# Footprint radius — used to choose the base (largest) and grasp width
# safety check (must be < gripper opening of ~0.08 m).
ARCHETYPE_FOOTPRINT = {
    "cube":         0.025,
    "small_cube":   0.018,
    "cylinder":     0.025,
    "hex_prism":    0.030,   # outer radius of rotated slabs
    "cone_frustum": 0.030,   # base radius
}

# Absolute grasp z (metres above the table top). The Panda's eef site
# sits well above the actual fingertip pads, so for short objects on
# the table the eef needs to descend nearly to the table to bracket
# the body. Values here match the empirical pattern in
# pick_place_ycb_policy.GRASP_CATALOG.
GRASP_Z_ABOVE_TBL = {
    # Targets are tuned to land just above the natural descent floor
    # the OSC reaches when fingertips reach the object. Aiming below the
    # floor wastes ~150 ticks of futile pushing without going lower.
    "cube":         0.018,
    "small_cube":   0.012,
    "cylinder":     0.015,
    "hex_prism":    0.016,
    # cone_frustum is ~8 cm tall and stepped. Grip the MIDDLE cylinder
    # (radius 0.0225, smaller and clear of the wide base) so the
    # gripper closes cleanly without swallowing the flange.
    "cone_frustum": 0.045,
}

# Order of preference for choosing the base object: widest flat top
# wins. cone_frustum is pyramidal -> bad to stack on, but with a 0.030
# base it's stable as the bottom-most piece itself; we'll only pick it
# as base if no flat-top is available.
BASE_PREFERENCE = ["cube", "hex_prism", "cylinder", "small_cube", "cone_frustum"]


# ── Quaternion helpers (xyzw) ───────────────────────────────────────

def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _grasp_quat_for_yaw(world_yaw: float) -> np.ndarray:
    half = 0.5 * world_yaw
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


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


# ── Policy ──────────────────────────────────────────────────────────

class ShapeStackPolicy(PolicyInterface):
    """Stack all but one of N differently-shaped objects on top of a base."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8

        # Plan-time state.
        self._archetypes: list[dict] = []          # filled in plan
        self._base_name: str | None = None
        self._base_xy: np.ndarray | None = None
        self._stack_order: list[str] = []
        # current top-of-stack z (world frame) -- updated after each release
        self._stack_top_z: float = 0.8

        # Per-object scratch.
        self._cur_idx: int = 0
        self._cur_name: str | None = None
        self._cur_archetype: dict | None = None
        self._cur_obj_xy: np.ndarray | None = None
        self._cur_grasp_z: float = 0.8
        self._cur_grasp_offset: float = 0.0  # gripper z minus body z at grasp
        self._cur_place_z: float = 0.8       # eef z at release

        # Setpoint-ramp state.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

    # ── Phase helpers ────────────────────────────────────────────────
    def _advance(self, next_phase: str):
        print(f"  [{self.step_in_phase:4d} ticks] {self.phase} -> {next_phase}",
              flush=True)
        self.phase = next_phase
        self.step_in_phase = 0

    def _ramp_factor(self, target_pos: np.ndarray) -> float:
        if (self._last_target_pos is None
                or float(np.linalg.norm(target_pos - self._last_target_pos))
                   > TARGET_CHANGE_TOL):
            self._ticks_since_target_change = 0
        self._last_target_pos = target_pos.copy()
        self._ticks_since_target_change += 1
        t = min(self._ticks_since_target_change / max(RAMP_TICKS, 1), 1.0)
        s = t * t * (3.0 - 2.0 * t)
        return RAMP_START_FRAC + (1.0 - RAMP_START_FRAC) * s

    def _move_to(self, eef_pos, eef_quat, target_pos, gripper,
                 gain_xy=GAIN_XY, gain_z=GAIN_Z, max_step=MAX_ACTION_STEP,
                 target_quat=None):
        ramp = self._ramp_factor(target_pos)
        action = np.zeros(7)
        delta = target_pos - eef_pos
        action[0] = float(np.clip(delta[0] * gain_xy * ramp, -max_step, max_step))
        action[1] = float(np.clip(delta[1] * gain_xy * ramp, -max_step, max_step))
        action[2] = float(np.clip(delta[2] * gain_z  * ramp, -max_step, max_step))
        tgt = TOP_DOWN_QUAT if target_quat is None else target_quat
        ori_err = _quat_err_axis_angle(eef_quat, tgt)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
        action[6] = gripper
        return action

    # ── Sim-introspection helpers ────────────────────────────────────
    @staticmethod
    def _archetype_body_id(env, name: str) -> int | None:
        sim = env._env.sim
        for suffix in ("_main", "_root", ""):
            try:
                return sim.model.body_name2id(name + suffix)
            except Exception:
                continue
        return None

    def _archetype_pos(self, env, name: str) -> np.ndarray:
        bid = self._archetype_body_id(env, name)
        if bid is None:
            return np.array([0.0, 0.0, self._table_z])
        return np.asarray(env._env.sim.data.body_xpos[bid], dtype=float).copy()

    # ── Plan ─────────────────────────────────────────────────────────
    def _build_plan(self, env):
        archetypes = list(getattr(env, "_selected_archetypes", []))
        self._archetypes = archetypes
        self._table_z = float(getattr(env, "_table_height", self._table_z))

        # Pick base: first archetype name in BASE_PREFERENCE that's present.
        present_names = [a["name"] for a in archetypes]
        base_name = None
        for pref in BASE_PREFERENCE:
            if pref in present_names:
                base_name = pref
                break
        if base_name is None:
            base_name = present_names[0]
        self._base_name = base_name

        # Stack order: every archetype except the base, sorted by
        # descending footprint so the widest sit lowest.
        others = [n for n in present_names if n != base_name]
        others.sort(
            key=lambda n: ARCHETYPE_FOOTPRINT.get(n, 0.025),
            reverse=True,
        )
        self._stack_order = others

        base_pos = self._archetype_pos(env, base_name)
        self._base_xy = base_pos[:2].copy()
        # Stack top z = base body centre + base "top over centre"
        top_over = ARCHETYPE_TOP_OVER_CENTER.get(base_name, 0.025)
        self._stack_top_z = float(base_pos[2]) + float(top_over)

        print(f"  [plan] archetypes={present_names}  base={base_name}  "
              f"base_xy={np.round(self._base_xy, 3)}  "
              f"stack_top_z={self._stack_top_z:.3f}  "
              f"order={self._stack_order}",
              flush=True)

    def _start_object(self, env, name: str):
        """Latch the live xy and grasp z for the next object to pick."""
        self._cur_name = name
        for a in self._archetypes:
            if a["name"] == name:
                self._cur_archetype = a
                break

        pos = self._archetype_pos(env, name)
        self._cur_obj_xy = pos[:2].copy()
        # Grasp z: absolute height above table, calibrated per archetype.
        # The Panda eef site sits well above the actual fingertip pads,
        # so for short tabletop objects we need to dive nearly to the
        # table top to bracket the body.
        z_above = GRASP_Z_ABOVE_TBL.get(name, 0.010)
        self._cur_grasp_z = self._table_z + z_above
        bottom_below = ARCHETYPE_BOTTOM_BELOW_CENTER.get(name, 0.020)
        # Grasp offset: how far the gripper sat ABOVE the body's centre
        # when we closed on it (we usually grasp slightly above centre,
        # especially for the cone where we grip the middle cylinder).
        # The same offset is preserved while the object is held, so when
        # we lower to place_z the body centre will be at place_z minus
        # this offset. Without this correction the formula effectively
        # presses every object 5+ mm into the support stack.
        body_z_when_grasped = float(pos[2])
        self._cur_grasp_offset = self._cur_grasp_z - body_z_when_grasped
        # Placement: gripper z = (desired body centre) + grasp_offset.
        #   desired body bottom  = stack_top + clearance
        #   desired body centre  = stack_top + bottom_below + clearance
        self._cur_place_z = (
            self._stack_top_z
            + bottom_below
            + STACK_PLACE_CLEARANCE
            + self._cur_grasp_offset
        )
        print(f"  [obj] name={name}  obj_xy={np.round(self._cur_obj_xy, 3)}  "
              f"grasp_z={self._cur_grasp_z:.3f}  place_z={self._cur_place_z:.3f}  "
              f"stack_top_z={self._stack_top_z:.3f}",
              flush=True)

    def _finish_object(self, name: str):
        """After a successful release, advance the stack top."""
        top_over = ARCHETYPE_TOP_OVER_CENTER.get(name, 0.020)
        bottom_below = ARCHETYPE_BOTTOM_BELOW_CENTER.get(name, 0.020)
        # Object body centre after release = stack_top + bottom_below
        new_centre = self._stack_top_z + bottom_below
        self._stack_top_z = new_centre + top_over
        print(f"  [stack] placed {name}  new_top_z={self._stack_top_z:.3f}",
              flush=True)

    # ── Main act() ────────────────────────────────────────────────────
    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0.0, 0.0, 0.0, 1.0])))

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))

        # ── 1. Settle ────────────────────────────────────────────────
        if self.phase == "settle":
            action = np.zeros(7)
            action[6] = -1.0
            if self.step_in_phase >= SETTLE_STEPS and env is not None:
                self._build_plan(env)
                if self._stack_order:
                    self._cur_idx = 0
                    self._start_object(env, self._stack_order[0])
                    self._advance("pre_grasp")
                else:
                    self._advance("retreat")
            self.step_in_phase += 1
            return action

        # ── 2. Pre-grasp: hover above current obj xy ─────────────────
        if self.phase == "pre_grasp":
            # Re-read live object xy so a tiny sim drift since plan-time
            # doesn't bake in a stale offset.
            if env is not None and self._cur_name is not None:
                live = self._archetype_pos(env, self._cur_name)
                self._cur_obj_xy = live[:2].copy()
            target = np.array([
                self._cur_obj_xy[0],
                self._cur_obj_xy[1],
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=TOP_DOWN_QUAT)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_ok = abs(target[2] - eef_pos[2]) < POS_TOL_Z
            ori_ok = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, TOP_DOWN_QUAT))) < ORI_TOL
            if (xy_err < 0.005 and z_ok and ori_ok) \
                    or self.step_in_phase >= PRE_GRASP_MAX_STEPS:
                print(f"  [pre_grasp] xy_err={xy_err:.3f}  obj={self._cur_name}",
                      flush=True)
                self._advance("approach")
            self.step_in_phase += 1
            return action

        # ── 3a. Approach: drop to just-above object & lock xy ───────
        # Splitting descend in two prevents the "bump-and-miss" failure
        # where OSC steady-state error in xy is masked by an aggressive z
        # drop. We come down to obj_top + safety, refresh xy from the
        # live sim, and only commit to grasp depth once xy is tight.
        if self.phase == "approach":
            if env is not None and self._cur_name is not None:
                live = self._archetype_pos(env, self._cur_name)
                self._cur_obj_xy = live[:2].copy()
            top_over = ARCHETYPE_TOP_OVER_CENTER.get(self._cur_name, 0.025)
            obj_pos = (self._archetype_pos(env, self._cur_name)
                       if env is not None else None)
            obj_top_z = (float(obj_pos[2]) + top_over) if obj_pos is not None \
                        else self._cur_grasp_z + 0.04
            # Hover 3 cm above the object's top — clear of the surface so
            # we can shuffle xy without contact.
            approach_z = obj_top_z + 0.03
            target = np.array([
                self._cur_obj_xy[0],
                self._cur_obj_xy[1],
                approach_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_xy=6.0, gain_z=6.0,
                                   target_quat=TOP_DOWN_QUAT)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_ok = abs(target[2] - eef_pos[2]) < 0.008
            # Tight xy gate: we MUST be lined up before committing the
            # final drop. 3 mm is safely inside any object's footprint.
            if (xy_err < 0.004 and z_ok) \
                    or self.step_in_phase >= APPROACH_MAX_STEPS:
                print(f"  [approach] xy_err={xy_err:.4f}  "
                      f"eef_z=table+{eef_pos[2]-self._table_z:.3f}  "
                      f"approach_z=table+{approach_z-self._table_z:.3f}",
                      flush=True)
                self._advance("descend")
            self.step_in_phase += 1
            return action

        # ── 3b. Descend: slow drop into grasp depth ─────────────────
        if self.phase == "descend":
            # Don't refresh xy here — we already locked it in approach.
            # If the object has moved at this point, it's because we just
            # touched it; chasing the new xy would only make things worse.
            target = np.array([
                self._cur_obj_xy[0],
                self._cur_obj_xy[1],
                self._cur_grasp_z,
            ])
            # Saturate the action clip so descent runs at the OSC's
            # achievable peak speed (~1 mm/tick). Lower gains were
            # leaving action[2] well below MAX_ACTION_STEP and stalling.
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_xy=6.0, gain_z=6.0,
                                   target_quat=TOP_DOWN_QUAT)
            z_reached = abs(target[2] - eef_pos[2]) < 0.005
            if z_reached or self.step_in_phase >= DESCEND_MAX_STEPS:
                xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
                print(f"  [descend] eef_z=table+{eef_pos[2]-self._table_z:+.4f}  "
                      f"target=table+{self._cur_grasp_z-self._table_z:+.4f}  "
                      f"xy_err={xy_err:.4f}",
                      flush=True)
                self._advance("close_gripper")
            self.step_in_phase += 1
            return action

        # ── 4. Close gripper ─────────────────────────────────────────
        if self.phase == "close_gripper":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, TOP_DOWN_QUAT)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase >= CLOSE_STEPS:
                self._advance("lift")
            self.step_in_phase += 1
            return action

        # ── 5. Lift to safe transport altitude ───────────────────────
        if self.phase == "lift":
            transport_z = max(
                self._stack_top_z + TRANSPORT_CLEARANCE,
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            )
            target = np.array([eef_pos[0], eef_pos[1], transport_z])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=TOP_DOWN_QUAT)
            reached = abs(target[2] - eef_pos[2]) < LIFT_Z_TOL
            if reached or self.step_in_phase >= LIFT_MAX_STEPS:
                self._advance("transport")
            self.step_in_phase += 1
            return action

        # ── 6. Transport to base xy at transport altitude ────────────
        if self.phase == "transport":
            transport_z = max(
                self._stack_top_z + TRANSPORT_CLEARANCE,
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            )
            target = np.array([
                self._base_xy[0],
                self._base_xy[1],
                transport_z,
            ])
            # Higher xy gain so a 30 cm cross-table traverse actually
            # converges within budget. Under-saturated gain was leaving
            # the gripper 14 cm off the base when lower started.
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=6.0, gain_z=4.0,
                                   target_quat=TOP_DOWN_QUAT)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            # Tight gate: anything more than 8 mm off-base will overhang
            # the support stack and topple it on release.
            if xy_err < 0.008 or self.step_in_phase >= TRANSPORT_MAX_STEPS:
                print(f"  [transport] xy_err={xy_err:.3f}  base_xy={np.round(self._base_xy, 3)}",
                      flush=True)
                self._advance("hover_align")
            self.step_in_phase += 1
            return action

        # ── 7a. Hover-align: lock xy & top-down orientation BEFORE the
        # vertical drop. Without this, the lower phase corrects xy and
        # ori mid-descent, which lands the object at an angle and
        # topples the stack on release. Stay at transport altitude.
        if self.phase == "hover_align":
            transport_z = max(
                self._stack_top_z + TRANSPORT_CLEARANCE,
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            )
            target = np.array([
                self._base_xy[0],
                self._base_xy[1],
                transport_z,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=6.0, gain_z=4.0,
                                   target_quat=TOP_DOWN_QUAT)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            ori_err = float(np.linalg.norm(
                _quat_err_axis_angle(eef_quat, TOP_DOWN_QUAT)))
            # Tight gates — straight-down drop only after both xy and
            # orientation are locked.
            if (xy_err < 0.004 and ori_err < 0.05) \
                    or self.step_in_phase >= HOVER_ALIGN_MAX_STEPS:
                print(f"  [hover_align] xy_err={xy_err:.4f}  ori_err={ori_err:.3f}",
                      flush=True)
                self._advance("lower")
            self.step_in_phase += 1
            return action

        # ── 7b. Lower onto stack top — pure vertical drop ────────────
        if self.phase == "lower":
            # Hold xy at base_xy and orientation at TOP_DOWN_QUAT with
            # high gains so the descent is straight down. Modulate ONLY
            # the z gain so the touchdown is gentle.
            target = np.array([
                self._base_xy[0],
                self._base_xy[1],
                self._cur_place_z,
            ])
            dz_remaining = eef_pos[2] - self._cur_place_z
            if dz_remaining > 0.030:
                gz = 4.0
            elif dz_remaining > 0.012:
                gz = 1.2
            else:
                gz = 0.4   # very low: gripper coasts onto place_z, no
                           # accumulating contact force on the support
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=8.0, gain_z=gz,
                                   target_quat=TOP_DOWN_QUAT)
            reached = abs(target[2] - eef_pos[2]) < 0.005
            if reached or self.step_in_phase >= LOWER_MAX_STEPS:
                xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
                print(f"  [lower] eef_z=table+{eef_pos[2]-self._table_z:.3f}  "
                      f"target_place_z=table+{self._cur_place_z-self._table_z:.3f}  "
                      f"final_dz={eef_pos[2]-self._cur_place_z:+.3f}  "
                      f"xy_err={xy_err:.4f}",
                      flush=True)
                self._advance("release")
            self.step_in_phase += 1
            return action

        # ── 8. Release: open gripper, hold pose ──────────────────────
        # Open gripper while holding the press-to-seat pose so the object
        # is already in contact with the support when fingers part. Then
        # dwell for the rest of RELEASE_STEPS letting physics settle
        # before retract — yanking up too soon was tipping the stack.
        if self.phase == "release":
            target = np.array([
                self._base_xy[0],
                self._base_xy[1],
                self._cur_place_z,
            ])
            # Hold gripper z gently — once released, the held object's
            # weight is no longer borne by the gripper, so we don't need
            # to push down hard.
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_z=1.5, target_quat=TOP_DOWN_QUAT)
            if self.step_in_phase >= RELEASE_STEPS:
                self._finish_object(self._cur_name)
                self._advance("retract")
            self.step_in_phase += 1
            return action

        # ── 9. Retract: lift away to current transport altitude ──────
        if self.phase == "retract":
            transport_z = max(
                self._stack_top_z + TRANSPORT_CLEARANCE,
                self._table_z + PRE_GRASP_HEIGHT_ABOVE_TBL,
            )
            target = np.array([eef_pos[0], eef_pos[1], transport_z])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=TOP_DOWN_QUAT)
            reached = abs(target[2] - eef_pos[2]) < LIFT_Z_TOL
            if reached or self.step_in_phase >= RETRACT_MAX_STEPS:
                self._cur_idx += 1
                if self._cur_idx < len(self._stack_order):
                    self._start_object(env, self._stack_order[self._cur_idx])
                    self._advance("pre_grasp")
                else:
                    self._advance("retreat")
            self.step_in_phase += 1
            return action

        # ── 10. Retreat off the stack so the gripper isn't in contact ─
        if self.phase == "retreat":
            target = np.array([
                self._base_xy[0] + RETREAT_AWAY_XY[0],
                self._base_xy[1] + RETREAT_AWAY_XY[1],
                self._table_z + RETREAT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=TOP_DOWN_QUAT)
            xy_err = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_ok = abs(target[2] - eef_pos[2]) < LIFT_Z_TOL
            if (xy_err < POS_TOL_XY * 2 and z_ok) \
                    or self.step_in_phase >= RETREAT_MAX_STEPS:
                self._advance("wait_stable")
            self.step_in_phase += 1
            return action

        # ── 11. Wait for the stack to register `stable_steps` ticks ──
        if self.phase == "wait_stable":
            target = np.array([
                self._base_xy[0] + RETREAT_AWAY_XY[0],
                self._base_xy[1] + RETREAT_AWAY_XY[1],
                self._table_z + RETREAT_HEIGHT_ABOVE_TBL,
            ])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=TOP_DOWN_QUAT)
            if self.step_in_phase >= WAIT_STABLE_STEPS:
                self._advance("done")
            self.step_in_phase += 1
            return action

        # ── 12. Done: hold pose ─────────────────────────────────────
        action = np.zeros(7)
        action[6] = -1.0
        ori_err = _quat_err_axis_angle(eef_quat, TOP_DOWN_QUAT)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
        self.step_in_phase += 1
        return action


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = ib.make("ib-p7-shape-stack-v1", privileged_obs=True, render_mode=render_mode)
    policy = ShapeStackPolicy()

    scores = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 13)
        policy.reset()
        names = [a["name"] for a in env._selected_archetypes]
        print(f"\n=== Episode {ep}  seed={args.seed + ep * 13} ===  "
              f"archetypes={names}",
              flush=True)
        done = False
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            done = terminated or truncated
        score = float(info.get("task_score", 0.0))
        scores.append(score)
        print(f"Episode {ep}  score={score:.3f}  phase={policy.phase}",
              flush=True)
    print(f"\nMean score over {args.episodes} eps: {np.mean(scores):.3f}",
          flush=True)
    env.close()
