"""Scripted tool-retrieval policy for ``ib-p4-tool-retrieve-v1``.

The task spawns an L-shaped hook tool within reach and a cube out of
reach.  The policy:
  1. Grasps the hook at the handle (−X shaft end in local frame).
  2. Flies the hook tip to the far side of the cube and lowers.
  3. Drags the hook toward the robot until the cube is in reach.
  4. Releases the hook, then picks and lifts the cube.

Phase machine:
  settle
  → pre_grasp_hook → pre_grasp_settle → descend_hook → close_hook
  → lift_hook
  → approach_drag → lower_to_drag → drag
  → release_hook → retract
  → pre_grasp_cube → descend_cube → close_cube → lift_cube
  → done

Obs-key notes
─────────────
The filtered obs dict for this task contains:
  object_pos   (3,)   — cube position (first object alias)
  object_quat  (4,)   — cube quaternion xyzw
  object_state (17,)  — [cube_pos(3) | cube_quat(4) | hook_pos(3) | hook_quat(4) | gripper_to_cube(3)]
Hook and tip positions are read directly from env.sim (body_xpos) for accuracy.
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Hook geometry (must match task YAML / _make_objects defaults) ────
# hook_length=0.3  →  shaft X half-extent = hl/2 = 0.15 m
# tip sub_location: [hl/2, tip_half, 0] = [0.15, 0.04, 0]  (local frame)
# hook_width=0.01  →  shaft cross-section half-width = 0.01 m
HOOK_SHAFT_HALF = 0.15
HOOK_TIP_Y      = 0.04
HOOK_TIP_LOCAL  = np.array([HOOK_SHAFT_HALF,  HOOK_TIP_Y, 0.0])
# Grasp near handle end for max reach extension.
# At table level, robot stalls ~3cm short of full handle (x=-0.155).
# Clamped to x=-0.12 in settle if hook is too far left.
HANDLE_LOCAL    = np.array([-0.15, 0.0, 0.0])


CUBE_HALF   = 0.02   # cube_size from YAML (half-side)
LIFT_TARGET = 0.25   # m above table; task threshold is 0.20

# ── MuJoCo body names for sim reads ─────────────────────────────────
BODY_CUBE     = "cube_main"
BODY_HOOK_ROOT = "hook_tool_root"
BODY_HOOK_TIP  = "hook_tool_tip_main"

# ── Gripper orientation ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])   # xyzw, EEF Z → world −Z

# ── Heights above table (m) ──────────────────────────────────────────
PRE_GRASP_H       = 0.18   # hover height before hook grasp descent
HOOK_GRASP_H      = 0.003  # height above table = shaft center (hook_width=0.01)
TRANSPORT_H       = 0.10   # hook transport / approach altitude (tip clears cube top)
DRAG_H            = 0.003  # EEF at table level so shaft rests on table → max side-face contact
CUBE_PRE_GRASP_H  = 0.18   # hover before cube grasp descent
CUBE_GRASP_H      = 0.012  # descent target above table for cube grasp

# ── Phase step budgets (ticks @ 100 Hz) ─────────────────────────────
SETTLE_STEPS        = 10
PRE_GRASP_MAX       = 200
PRE_GRASP_SETTLE    = 10
DESCEND_HOOK_MAX    = 200
CLOSE_HOOK_STEPS    = 30
LIFT_HOOK_MAX       = 130
APPROACH_DRAG_MAX   = 380
LOWER_DRAG_MAX      = 150
DRAG_MAX            = 1400
SWEEP_OUT_MAX       = 35   # ticks to sweep hook sideways after drag
RELEASE_HOOK_STEPS  = 30
RETRACT_MAX         = 60
PRE_GRASP_CUBE_MAX  = 180
DESCEND_CUBE_MAX    = 180
CLOSE_CUBE_STEPS    = 35
LIFT_CUBE_MAX       = 180

# ── Drag parameters ──────────────────────────────────────────────────
CUBE_IN_REACH_X    = 0.28   # drag cube well into workspace before releasing
DRAG_SPEED         = 0.25   # −X action magnitude per tick

# Ratchet parameters: drag a stroke, re-engage, repeat until cube is in workspace.
DRAG_STROKE_TICKS  = 650    # ticks per stroke before re-engaging (~130mm per stroke)
MAX_DRAG_STROKES   = 5      # at most 5 strokes before giving up
SLIP_MARGIN        = 0.02   # m; re-engage when tip is this close to passing cube (near-slip)
STUCK_WINDOW       = 50     # ticks between stuck-check samples
STUCK_TOL          = 0.003  # m; if cube moved less than this over STUCK_WINDOW → stuck

# How far past the cube's back face to position the tip before each drag stroke.
TIP_CLEARANCE  = 0.06       # m; clearance past cube far face before starting stroke
SIDE_CLEARANCE = -0.10      # m; approach from -Y side so tip clears cube during descent

# ── Tolerances ───────────────────────────────────────────────────────
POS_TOL_XY = 0.012
POS_TOL_Z  = 0.010
ORI_TOL    = 0.12

# ── Control gains ────────────────────────────────────────────────────
GAIN_XY         = 4.0
GAIN_Z          = 5.0
ROT_GAIN        = 2.0
MAX_ACTION_STEP = 0.50

# ── Setpoint ramp (smoothstep over RAMP_TICKS from RAMP_START_FRAC) ─
RAMP_TICKS        = 20
RAMP_START_FRAC   = 0.2
TARGET_CHANGE_TOL = 0.015


# ── Quaternion / rotation helpers (xyzw convention) ─────────────────

def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion → 3×3 rotation matrix."""
    x, y, z, w = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _quat_err_axis_angle(cur: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    cx, cy, cz, cw = cur
    tx, ty, tz, tw = tgt
    ccx, ccy, ccz, ccw = -cx, -cy, -cz, cw
    ex = tw*ccx + tx*ccw + ty*ccz - tz*ccy
    ey = tw*ccy - tx*ccz + ty*ccw + tz*ccx
    ez = tw*ccz + tx*ccy - ty*ccx + tz*ccw
    ew = tw*ccw - tx*ccx - ty*ccy - tz*ccz
    if ew < 0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    sin_half = float(np.sqrt(ex*ex + ey*ey + ez*ez))
    if sin_half < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, ew)
    return np.array([ex, ey, ez]) / sin_half * angle


def _grasp_quat_for_yaw(yaw: float) -> np.ndarray:
    """Top-down grasp quat with gripper rotated by yaw about world Z (xyzw)."""
    half = 0.5 * yaw
    qz = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


def _hook_shaft_world_yaw(hook_quat: np.ndarray) -> float:
    """World yaw of hook shaft (local X axis projected onto world XY)."""
    R = _quat_to_rot(hook_quat)
    bx = R[:, 0]           # world direction of hook local-X
    if abs(bx[2]) > 0.9:   # nearly vertical — fall back to local-Y
        bx = R[:, 1]
    return float(np.arctan2(bx[1], bx[0]))


def _cube_world_yaw(quat: np.ndarray) -> float:
    x, y, z, w = quat
    return float(np.arctan2(2*(x*y + z*w), 1 - 2*(y*y + z*z)))


def _shortest_cube_grasp_yaw(yaw: float) -> float:
    """Cube has 4-fold symmetry: wrap to [−π/4, π/4]."""
    return float((yaw + np.pi/4) % (np.pi/2) - np.pi/4)


# ── Policy ───────────────────────────────────────────────────────────

class ToolRetrievePolicy(PolicyInterface):
    """Scripted policy for ib-p4-tool-retrieve-v1.

    All geometry derived from live obs and env.sim each tick —
    agnostic to hook / cube position and orientation randomisation.
    """

    def __init__(self, rng=None, verbose: bool = False):
        self.rng = np.random.default_rng() if rng is None else rng
        self.verbose = verbose  # print [TR] phase diagnostics
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8

        # MuJoCo body ID cache (filled on first _read_positions call).
        self._bid_cube:     int | None = None
        self._bid_hook_root: int | None = None
        self._bid_hook_tip:  int | None = None

        # Locked at end of settle.
        self._hook_handle_world: np.ndarray | None = None
        self._hook_grasp_quat: np.ndarray = TOP_DOWN_QUAT.copy()

        # Locked at first tick of pre_grasp_cube.
        self._cube_xy: np.ndarray | None = None
        self._cube_grasp_quat: np.ndarray = TOP_DOWN_QUAT.copy()

        # Drag ratchet state. The stuck-check reference and timer carry over
        # between strokes, so a new stroke is judged only once it has run past
        # the previous stroke's last check; the first window always passes.
        self._drag_stroke_count: int = 0
        self._drag_prev_cube_x: float = np.inf
        self._drag_prev_check_t: int = 0

        # Hook grasp retry state.
        self._hook_grasp_retries: int = 0
        self._hook_z_at_grasp: float = 0.0

        # Lift slip-recovery counter.
        self._lift_retry: int = 0

        # Setpoint ramp state.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

    # ── Phase transition helper ───────────────────────────────────────

    def _advance(self, next_phase: str, **kw):
        extra = "  ".join(f"{k}={v}" for k, v in kw.items())
        if self.verbose:
            print(
                f"  [TR] [{self.step_in_phase:4d}t] {self.phase} → {next_phase}"
                + (f"  {extra}" if extra else ""),
                flush=True,
        )
        self.phase = next_phase
        self.step_in_phase = 0

    # ── Body-ID cache ─────────────────────────────────────────────────

    def _cache_body_ids(self, env) -> bool:
        """Populate MuJoCo body IDs once.  Returns True on success."""
        if self._bid_cube is not None:
            return True
        try:
            sim = env.sim
            self._bid_cube      = sim.model.body_name2id(BODY_CUBE)
            self._bid_hook_root = sim.model.body_name2id(BODY_HOOK_ROOT)
            self._bid_hook_tip  = sim.model.body_name2id(BODY_HOOK_TIP)
            if self.verbose:
                print(
                    f"  [TR] body IDs cached: cube={self._bid_cube}  "
                    f"hook_root={self._bid_hook_root}  hook_tip={self._bid_hook_tip}",
                    flush=True,
            )
            return True
        except (ValueError, KeyError) as e:
            if self.verbose: print(f"  [TR] WARNING: body ID cache failed: {e}", flush=True)
            return False

    # ── Position reader ───────────────────────────────────────────────

    def _read_positions(self, obs: dict, env):
        """Return (cube_pos, cube_quat, hook_pos, hook_quat, tip_pos).

        cube / hook: from env.sim (authoritative) when available;
                     object_state fallback otherwise.
        tip:         hook_tool_tip_main body from sim (direct, no matrix math).
        """
        # ── cube ─────────────────────────────────────────────────────
        cube_pos  = obs.get("object_pos",  obs.get("cube_pos",  np.zeros(3))).copy()
        cube_quat = obs.get("object_quat", obs.get("cube_quat",
                            np.array([0., 0., 0., 1.]))).copy()

        # ── hook from object_state fallback (indices 7:14) ───────────
        obj_state = obs.get("object_state", np.zeros(17))
        hook_pos  = obj_state[7:10].copy()
        hook_quat = obj_state[10:14].copy()
        # Derive tip from rotation matrix (fallback)
        tip_pos   = hook_pos + _quat_to_rot(hook_quat) @ HOOK_TIP_LOCAL

        # ── sim override (preferred) ──────────────────────────────────
        if env is not None and self._cache_body_ids(env):
            sim = env.sim
            hook_pos  = sim.data.body_xpos[self._bid_hook_root].copy()
            wxyz      = sim.data.body_xquat[self._bid_hook_root]   # MuJoCo: wxyz
            hook_quat = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])  # → xyzw

            # Tip body position read directly — no rotation-matrix math needed.
            tip_pos = sim.data.body_xpos[self._bid_hook_tip].copy()

            # Cube from sim as well (consistent coordinate origin).
            cube_pos  = sim.data.body_xpos[self._bid_cube].copy()
            wxyz      = sim.data.body_xquat[self._bid_cube]
            cube_quat = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

        return cube_pos, cube_quat, hook_pos, hook_quat, tip_pos

    # ── Setpoint ramp ─────────────────────────────────────────────────

    def _ramp(self, target: np.ndarray) -> float:
        if (self._last_target_pos is None
                or np.linalg.norm(target - self._last_target_pos) > TARGET_CHANGE_TOL):
            self._ticks_since_target_change = 0
        self._last_target_pos = target.copy()
        self._ticks_since_target_change += 1
        t = min(self._ticks_since_target_change / max(RAMP_TICKS, 1), 1.0)
        s = t * t * (3.0 - 2.0 * t)   # smoothstep
        return RAMP_START_FRAC + (1.0 - RAMP_START_FRAC) * s

    # ── Move-to helper ────────────────────────────────────────────────

    def _move_to(self, eef_pos, eef_quat, target_pos, gripper,
                 gain_xy=GAIN_XY, gain_z=GAIN_Z, max_step=MAX_ACTION_STEP,
                 target_quat=None):
        ramp = self._ramp(target_pos)
        d = target_pos - eef_pos
        action = np.zeros(7)
        action[0] = float(np.clip(d[0] * gain_xy * ramp, -max_step, max_step))
        action[1] = float(np.clip(d[1] * gain_xy * ramp, -max_step, max_step))
        action[2] = float(np.clip(d[2] * gain_z  * ramp, -max_step, max_step))
        tgt_q = TOP_DOWN_QUAT if target_quat is None else target_quat
        ori_err = _quat_err_axis_angle(eef_quat, tgt_q)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
        action[6] = float(gripper)
        return action

    # ── Tip-servo helper ──────────────────────────────────────────────

    def _move_tip_to(self, eef_pos, eef_quat, tip_pos, desired_tip,
                     gripper, servo_z=True, target_quat=None):
        delta = desired_tip - tip_pos           # how much tip needs to move
        eef_target = eef_pos + delta
        if not servo_z:
            eef_target[2] = self._table_z + TRANSPORT_H
        return self._move_to(eef_pos, eef_quat, eef_target, gripper,
                             target_quat=target_quat)

    # ── Main act() ────────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos  = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", np.array([0., 0., 0., 1.]))

        if env is not None:
            self._table_z = float(getattr(env, "_table_height", self._table_z))

        cube_pos, cube_quat, hook_pos, hook_quat, tip_pos = \
            self._read_positions(obs, env)

        # ────────────────────────────────────────────────────────────
        # 1. Settle: let controller transients die, read object states
        # ────────────────────────────────────────────────────────────
        if self.phase == "settle":
            action = np.zeros(7); action[6] = -1.0
            if self.step_in_phase >= SETTLE_STEPS:
                shaft_yaw = _hook_shaft_world_yaw(hook_quat)
                grip_yaw  = (shaft_yaw + np.pi/2) % np.pi - np.pi/2
                self._hook_grasp_quat = _grasp_quat_for_yaw(grip_yaw)

                R = _quat_to_rot(hook_quat)
                handle_full = hook_pos + R @ HANDLE_LOCAL
                # Clamp handle to reachable X (~−0.12 at pre-grasp height).
                # Slide grasp point along shaft toward tip if handle is too far.
                REACH_X_LIMIT = -0.12
                if handle_full[0] < REACH_X_LIMIT:
                    shaft_dir = R[:, 0]  # world direction of local +X (toward tip)
                    if abs(shaft_dir[0]) > 0.01:
                        dx = REACH_X_LIMIT - handle_full[0]   # positive
                        handle_full = handle_full + shaft_dir * (dx / shaft_dir[0])
                self._hook_handle_world = handle_full

                if self.verbose:
                    print(
                        f"  [TR] settle done  "
                        f"hook_pos={np.round(hook_pos,3)}  "
                        f"cube_pos={np.round(cube_pos,3)}  "
                        f"tip_pos={np.round(tip_pos,3)}  "
                        f"shaft_yaw={np.degrees(shaft_yaw):+.1f}°  "
                        f"grip_yaw={np.degrees(grip_yaw):+.1f}°  "
                        f"handle_world={np.round(self._hook_handle_world,3)}",
                        flush=True,
                )
                self._advance("pre_grasp_hook")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 2. Pre-grasp hook: hover above handle end, yaw-aligned
        # ────────────────────────────────────────────────────────────
        if self.phase == "pre_grasp_hook":
            hx, hy = self._hook_handle_world[:2]
            target  = np.array([hx, hy, self._table_z + PRE_GRASP_H])
            action  = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                    target_quat=self._hook_grasp_quat)
            xy_err  = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_err   = abs(target[2] - eef_pos[2])
            ori_err = float(np.linalg.norm(_quat_err_axis_angle(eef_quat,
                                                                  self._hook_grasp_quat)))
            if self.step_in_phase % 40 == 0:
                if self.verbose:
                    print(
                        f"  [TR] pre_grasp_hook t={self.step_in_phase}  "
                        f"eef={np.round(eef_pos,3)}  target={np.round(target,3)}  "
                        f"xy_err={xy_err:.4f}  z_err={z_err:.4f}  ori_err={ori_err:.4f}",
                        flush=True,
                )
            if (xy_err < POS_TOL_XY and z_err < POS_TOL_Z and ori_err < ORI_TOL) \
                    or self.step_in_phase >= PRE_GRASP_MAX:
                self._advance("pre_grasp_settle",
                              xy_ok=(xy_err < POS_TOL_XY),
                              z_ok=(z_err < POS_TOL_Z),
                              ori_ok=(ori_err < ORI_TOL),
                              timeout=(self.step_in_phase >= PRE_GRASP_MAX))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 3. Pre-grasp settle: hold hover so OSC converges before descent
        # ────────────────────────────────────────────────────────────
        if self.phase == "pre_grasp_settle":
            hx, hy = self._hook_handle_world[:2]
            target  = np.array([hx, hy, self._table_z + PRE_GRASP_H])
            action  = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                    target_quat=self._hook_grasp_quat)
            if self.step_in_phase >= PRE_GRASP_SETTLE:
                self._advance("descend_hook")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 4. Descend onto hook shaft handle end
        # ────────────────────────────────────────────────────────────
        if self.phase == "descend_hook":
            hx, hy = self._hook_handle_world[:2]
            target  = np.array([hx, hy, self._table_z + HOOK_GRASP_H])
            action  = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                    gain_z=5.0, target_quat=self._hook_grasp_quat)
            z_err = abs(target[2] - eef_pos[2])
            if self.step_in_phase % 30 == 0:
                if self.verbose:
                    print(
                        f"  [TR] descend_hook t={self.step_in_phase}  "
                        f"eef_z={eef_pos[2]:.4f}  target_z={target[2]:.4f}  "
                        f"z_err={z_err:.4f}  hook_z={hook_pos[2]:.4f}",
                        flush=True,
                )
            if z_err < 0.012 or self.step_in_phase >= DESCEND_HOOK_MAX:
                self._advance("close_hook",
                              reached=(z_err < 0.012),
                              timeout=(self.step_in_phase >= DESCEND_HOOK_MAX),
                              eef_z=round(float(eef_pos[2]), 4),
                              hook_z=round(float(hook_pos[2]), 4))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 5. Close gripper on hook shaft
        # ────────────────────────────────────────────────────────────
        if self.phase == "close_hook":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._hook_grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase == CLOSE_HOOK_STEPS - 1:
                g = obs.get("robot0_gripper_qpos", np.array([0., 0.]))
                if self.verbose:
                    print(
                        f"  [TR] close_hook done  "
                        f"gripper_qpos={np.round(g,4)}  "
                        f"hook_pos={np.round(hook_pos,3)}  eef_pos={np.round(eef_pos,3)}",
                        flush=True,
                )
            if self.step_in_phase >= CLOSE_HOOK_STEPS:
                self._hook_z_at_grasp = float(hook_pos[2])
                self._advance("lift_hook")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 6. Lift hook to transport height
        # ────────────────────────────────────────────────────────────
        if self.phase == "lift_hook":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + TRANSPORT_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._hook_grasp_quat)
            z_err = abs(target[2] - eef_pos[2])
            if self.step_in_phase % 30 == 0:
                if self.verbose:
                    print(
                        f"  [TR] lift_hook t={self.step_in_phase}  "
                        f"eef_z={eef_pos[2]:.4f}  hook_z={hook_pos[2]:.4f}  "
                        f"tip_pos={np.round(tip_pos,3)}  z_err={z_err:.4f}",
                        flush=True,
                )
            if self.step_in_phase == 25:
                hook_rose = hook_pos[2] > self._hook_z_at_grasp + 0.008
                if self.verbose:
                    print(
                        f"  [TR] lift_hook grasp_check  hook_z={hook_pos[2]:.4f}  "
                        f"at_grasp={self._hook_z_at_grasp:.4f}  rose={hook_rose}  "
                        f"retries={self._hook_grasp_retries}",
                        flush=True,
                    )
                if not hook_rose:
                    self._hook_grasp_retries += 1
                    if self._hook_grasp_retries >= 1:
                        # Shift grasp point 3 cm toward tip along shaft for better contact.
                        R = _quat_to_rot(hook_quat)
                        shaft_dir = R[:, 0]
                        self._hook_handle_world = self._hook_handle_world + shaft_dir * 0.03
                        if self.verbose:
                            print(
                                f"  [TR] grasp retry {self._hook_grasp_retries}: "
                                f"shifting grasp +3cm along shaft",
                                flush=True,
                            )
                    if self._hook_grasp_retries <= 3:
                        self._advance("pre_grasp_hook")
                    elif self.verbose:
                        print(
                            f"  [TR] GRASP_FAIL after {self._hook_grasp_retries} retries — proceeding",
                            flush=True,
                        )
                    self.step_in_phase += 1
                    return action
            if z_err < 0.012 or self.step_in_phase >= LIFT_HOOK_MAX:
                self._advance("approach_drag",
                              reached=(z_err < 0.012),
                              timeout=(self.step_in_phase >= LIFT_HOOK_MAX),
                              hook_pos=np.round(hook_pos,3),
                              tip_pos=np.round(tip_pos,3),
                              cube_pos=np.round(cube_pos,3))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 7. Approach drag (XY only):
        #    Servo tip to (cube_x + CUBE_HALF + TIP_CLEARANCE, cube_y)
        #    at transport height.  Z is held; only XY is servoed.
        #
        #    Key: eef_target = eef_pos + (desired_tip - tip_pos)
        #    This works for any hook orientation since tip_pos comes
        #    directly from hook_tool_tip_main body in the sim.
        # ────────────────────────────────────────────────────────────
        if self.phase == "approach_drag":
            # Approach from the +Y side (SIDE_CLEARANCE extra) so the tip passes
            # beside the cube, not over it, preventing top-face collision on descent.
            # Shaft alignment (desired_tip_y = cube_y + HOOK_TIP_Y) is achieved
            # in the lower_to_drag phase where Y and Z are servoed simultaneously.
            desired_tip = np.array([
                cube_pos[0] + CUBE_HALF + TIP_CLEARANCE,
                cube_pos[1] + HOOK_TIP_Y + SIDE_CLEARANCE,
                self._table_z + TRANSPORT_H,
            ])
            action = self._move_tip_to(eef_pos, eef_quat, tip_pos, desired_tip,
                                       gripper=1.0, servo_z=False,
                                       target_quat=self._hook_grasp_quat)
            xy_err = float(np.linalg.norm(desired_tip[:2] - tip_pos[:2]))
            if self.step_in_phase % 40 == 0:
                if self.verbose:
                    print(
                        f"  [TR] approach_drag t={self.step_in_phase}  "
                        f"eef={np.round(eef_pos,3)}  "
                        f"tip_pos={np.round(tip_pos,3)}  desired_tip={np.round(desired_tip,3)}  "
                        f"tip_xy_err={xy_err:.4f}",
                        flush=True,
                )
            # Also exit when tip_x has cleared the cube's far face (robot workspace
            # limit may prevent full convergence but dragging can still work).
            tip_past_cube = tip_pos[0] >= cube_pos[0] + CUBE_HALF
            if (xy_err < POS_TOL_XY or tip_past_cube
                    or self.step_in_phase >= APPROACH_DRAG_MAX):
                self._advance("lower_to_drag",
                              reached=(xy_err < POS_TOL_XY),
                              tip_past_cube=tip_past_cube,
                              timeout=(self.step_in_phase >= APPROACH_DRAG_MAX),
                              tip_pos=np.round(tip_pos,3),
                              cube_pos=np.round(cube_pos,3))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 8. Lower EEF to drag height; hook shaft follows.
        #    Direct EEF Z target (not tip servo) so EEF doesn't
        #    saturate below table level trying to push tip down.
        # ────────────────────────────────────────────────────────────
        if self.phase == "lower_to_drag":
            # Descend to drag height AND slide tip Y to shaft-aligned position
            # (shaft_center_y = cube_y → desired tip_y = cube_y + HOOK_TIP_Y).
            desired_tip_y = cube_pos[1] + HOOK_TIP_Y
            eef_y_target = eef_pos[1] + (desired_tip_y - tip_pos[1])
            target = np.array([eef_pos[0], eef_y_target, self._table_z + DRAG_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._hook_grasp_quat)
            z_err = abs(target[2] - eef_pos[2])
            y_err = abs(desired_tip_y - tip_pos[1])
            if self.step_in_phase % 30 == 0:
                if self.verbose:
                    print(
                        f"  [TR] lower_to_drag t={self.step_in_phase}  "
                        f"eef_z={eef_pos[2]:.4f}  target_z={target[2]:.4f}  "
                        f"z_err={z_err:.4f}  y_err={y_err:.4f}  tip_pos={np.round(tip_pos,3)}",
                        flush=True,
                )
            if (z_err < 0.010 and y_err < 0.015) or self.step_in_phase >= LOWER_DRAG_MAX:
                self._advance("drag",
                              reached=(z_err < 0.010 and y_err < 0.015),
                              timeout=(self.step_in_phase >= LOWER_DRAG_MAX),
                              tip_pos=np.round(tip_pos,3),
                              eef_z=round(float(eef_pos[2]),4),
                              cube_pos=np.round(cube_pos,3))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 9. Drag: ratchet — short strokes, re-engage if stuck or tip
        #    gets too close to cube, until cube_x < CUBE_IN_REACH_X.
        # ────────────────────────────────────────────────────────────
        if self.phase == "drag":
            action = np.zeros(7)
            action[0] = -DRAG_SPEED
            z_err = (self._table_z + DRAG_H) - eef_pos[2]
            action[2] = float(np.clip(z_err * GAIN_Z, -MAX_ACTION_STEP, MAX_ACTION_STEP))
            ori_err = _quat_err_axis_angle(eef_quat, self._hook_grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
            action[6] = 1.0

            # ── Diagnostics every 50 ticks ──────────────────────────
            tip_lead = tip_pos[0] - cube_pos[0]
            if self.step_in_phase % 50 == 0:
                if self.verbose:
                    print(
                        f"  [TR] drag stroke={self._drag_stroke_count} t={self.step_in_phase}  "
                        f"cube=[{cube_pos[0]:.3f},{cube_pos[1]:.3f}]  "
                        f"tip=[{tip_pos[0]:.3f},{tip_pos[1]:.3f}]  "
                        f"tip_lead={tip_lead:.3f}  "
                        f"eef_z_above_tbl={eef_pos[2]-self._table_z:.4f}",
                        flush=True,
                )

            # ── Stuck check every STUCK_WINDOW ticks ────────────────
            stuck = False
            if self.step_in_phase - self._drag_prev_check_t >= STUCK_WINDOW:
                moved = self._drag_prev_cube_x - cube_pos[0]
                stuck = moved < STUCK_TOL
                if stuck:
                    if self.verbose:
                        print(
                            f"  [TR] drag STUCK  moved={moved:.4f} over {STUCK_WINDOW}t  "
                            f"cube_x={cube_pos[0]:.4f}",
                            flush=True,
                    )
                self._drag_prev_cube_x = cube_pos[0]
                self._drag_prev_check_t = self.step_in_phase

            # ── Exit conditions ──────────────────────────────────────
            cube_in_reach  = cube_pos[0] < CUBE_IN_REACH_X
            slip_imminent  = tip_lead < SLIP_MARGIN          # tip too close → re-engage
            stroke_done    = self.step_in_phase >= DRAG_STROKE_TICKS
            strokes_maxed  = self._drag_stroke_count >= MAX_DRAG_STROKES

            if cube_in_reach or strokes_maxed:
                reason = "in_reach" if cube_in_reach else "max_strokes"
                if self.verbose:
                    print(
                        f"  [TR] drag DONE  cube_x={cube_pos[0]:.4f}  reason={reason}  "
                        f"strokes={self._drag_stroke_count}",
                        flush=True,
                )
                self._drag_stroke_count = 0
                self._advance("sweep_out",
                              cube_x=round(float(cube_pos[0]), 4),
                              in_reach=cube_in_reach)
            elif slip_imminent or stroke_done or stuck:
                reason = ("slip" if slip_imminent else
                          "stuck" if stuck else "stroke_done")
                self._drag_stroke_count += 1
                if self.verbose:
                    print(
                        f"  [TR] drag re-engage #{self._drag_stroke_count}  "
                        f"cube_x={cube_pos[0]:.4f}  reason={reason}  "
                        f"tip_lead={tip_lead:.3f}",
                        flush=True,
                )
                self._advance("approach_drag")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 10. Sweep hook sideways (−Y) so it clears the cube before release
        # ────────────────────────────────────────────────────────────
        if self.phase == "sweep_out":
            # Move hook −Y while rising to transport height; keep gripper closed.
            target = np.array([eef_pos[0] - 0.03,
                               eef_pos[1] - 0.12,
                               self._table_z + TRANSPORT_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   gain_xy=8.0, max_step=0.5,
                                   target_quat=self._hook_grasp_quat)
            y_moved = abs(eef_pos[1] - target[1])
            if self.step_in_phase % 20 == 0:
                if self.verbose:
                    print(
                        f"  [TR] sweep_out t={self.step_in_phase}  "
                        f"eef={np.round(eef_pos,3)}  cube={np.round(cube_pos,3)}",
                        flush=True,
                )
            reached = (np.linalg.norm(eef_pos - target) < 0.020)
            if reached or self.step_in_phase >= SWEEP_OUT_MAX:
                self._advance("release_hook")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 11. Release hook
        # ────────────────────────────────────────────────────────────
        if self.phase == "release_hook":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._hook_grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = -1.0
            if self.step_in_phase >= RELEASE_HOOK_STEPS:
                if self.verbose:
                    print(
                        f"  [TR] hook released  "
                        f"cube_pos={np.round(cube_pos,3)}  hook_pos={np.round(hook_pos,3)}",
                        flush=True,
                )
                self._advance("retract")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 11. Retract: lift EEF clear of hook/cube area
        # ────────────────────────────────────────────────────────────
        if self.phase == "retract":
            target = np.array([eef_pos[0] - 0.10,
                               eef_pos[1],
                               self._table_z + TRANSPORT_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   target_quat=TOP_DOWN_QUAT)
            reached = (abs(target[2] - eef_pos[2]) < 0.015
                       and np.linalg.norm(target[:2] - eef_pos[:2]) < 0.015)
            if self.step_in_phase % 20 == 0:
                if self.verbose:
                    print(
                        f"  [TR] retract t={self.step_in_phase}  "
                        f"eef={np.round(eef_pos,3)}  cube_pos={np.round(cube_pos,3)}",
                        flush=True,
                )
            if reached or self.step_in_phase >= RETRACT_MAX:
                self._advance("pre_grasp_cube", cube_pos=np.round(cube_pos,3))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 12. Pre-grasp cube: hover above cube with yaw-aligned gripper
        # ────────────────────────────────────────────────────────────
        if self.phase == "pre_grasp_cube":
            if self._cube_xy is None:
                self._cube_xy = cube_pos[:2].copy()
                cube_yaw  = _cube_world_yaw(np.asarray(cube_quat))
                grip_yaw  = _shortest_cube_grasp_yaw(cube_yaw)
                self._cube_grasp_quat = _grasp_quat_for_yaw(grip_yaw)
                if self.verbose:
                    print(
                        f"  [TR] pre_grasp_cube locked  "
                        f"cube_pos={np.round(cube_pos,3)}  "
                        f"cube_yaw={np.degrees(cube_yaw):+.1f}°  "
                        f"grip_yaw={np.degrees(grip_yaw):+.1f}°",
                        flush=True,
                )

            target  = np.array([self._cube_xy[0], self._cube_xy[1],
                                self._table_z + CUBE_PRE_GRASP_H])
            action  = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                    target_quat=self._cube_grasp_quat)
            xy_err  = float(np.linalg.norm(target[:2] - eef_pos[:2]))
            z_err   = abs(target[2] - eef_pos[2])
            ori_err = float(np.linalg.norm(_quat_err_axis_angle(eef_quat,
                                                                  self._cube_grasp_quat)))
            if self.step_in_phase % 40 == 0:
                if self.verbose:
                    print(
                        f"  [TR] pre_grasp_cube t={self.step_in_phase}  "
                        f"eef={np.round(eef_pos,3)}  target={np.round(target,3)}  "
                        f"xy_err={xy_err:.4f}  z_err={z_err:.4f}  ori_err={ori_err:.4f}",
                        flush=True,
                )
            if (xy_err < POS_TOL_XY and z_err < POS_TOL_Z and ori_err < ORI_TOL) \
                    or self.step_in_phase >= PRE_GRASP_CUBE_MAX:
                self._advance("descend_cube",
                              reached=(xy_err < POS_TOL_XY and z_err < POS_TOL_Z),
                              timeout=(self.step_in_phase >= PRE_GRASP_CUBE_MAX))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 13. Descend onto cube
        # ────────────────────────────────────────────────────────────
        if self.phase == "descend_cube":
            target = np.array([self._cube_xy[0], self._cube_xy[1],
                               self._table_z + CUBE_GRASP_H])
            action = self._move_to(eef_pos, eef_quat, target, gripper=-1.0,
                                   gain_z=5.0, target_quat=self._cube_grasp_quat)
            z_err = abs(target[2] - eef_pos[2])
            if self.step_in_phase % 30 == 0:
                if self.verbose:
                    print(
                        f"  [TR] descend_cube t={self.step_in_phase}  "
                        f"eef_z={eef_pos[2]:.4f}  target_z={target[2]:.4f}  "
                        f"cube_z={cube_pos[2]:.4f}  z_err={z_err:.4f}",
                        flush=True,
                )
            if z_err < 0.005 or self.step_in_phase >= DESCEND_CUBE_MAX:
                self._advance("close_cube",
                              reached=(z_err < 0.005),
                              timeout=(self.step_in_phase >= DESCEND_CUBE_MAX),
                              eef_z=round(float(eef_pos[2]),4),
                              cube_z=round(float(cube_pos[2]),4))
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 14. Close gripper on cube
        # ────────────────────────────────────────────────────────────
        if self.phase == "close_cube":
            action = np.zeros(7)
            ori_err = _quat_err_axis_angle(eef_quat, self._cube_grasp_quat)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)
            action[6] = 1.0
            if self.step_in_phase == CLOSE_CUBE_STEPS - 1:
                g = obs.get("robot0_gripper_qpos", np.array([0., 0.]))
                if self.verbose:
                    print(
                        f"  [TR] close_cube done  "
                        f"gripper_qpos={np.round(g,4)}  "
                        f"cube_pos={np.round(cube_pos,3)}  eef_pos={np.round(eef_pos,3)}",
                        flush=True,
                )
            if self.step_in_phase >= CLOSE_CUBE_STEPS:
                self._advance("lift_cube")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # 15. Lift cube past success threshold
        # ────────────────────────────────────────────────────────────
        if self.phase == "lift_cube":
            target = np.array([eef_pos[0], eef_pos[1],
                               self._table_z + LIFT_TARGET])
            action = self._move_to(eef_pos, eef_quat, target, gripper=1.0,
                                   target_quat=self._cube_grasp_quat)
            z_err = abs(target[2] - eef_pos[2])
            if self.step_in_phase % 25 == 0:
                if self.verbose:
                    print(
                        f"  [TR] lift_cube t={self.step_in_phase}  "
                        f"eef_z={eef_pos[2]:.4f}  cube_z={cube_pos[2]:.4f}  "
                        f"cube_above_tbl={cube_pos[2]-self._table_z:.4f}  "
                        f"target_z={target[2]:.4f}",
                        flush=True,
                )
            # Slip recovery: if cube didn't rise after 60 ticks, re-close once.
            if (self.step_in_phase > 60
                    and self._lift_retry == 0
                    and cube_pos[2] - self._table_z < 0.04):
                if self.verbose:
                    print(
                        f"  [TR] lift_cube SLIP  "
                        f"cube_above_tbl={cube_pos[2]-self._table_z:.4f}  "
                        f"t={self.step_in_phase}  → re-closing",
                        flush=True,
                )
                self._lift_retry = 1
                action[6] = 1.0

            if z_err < 0.012 or self.step_in_phase >= LIFT_CUBE_MAX:
                if self.verbose:
                    print(
                        f"  [TR] lift_cube exit  "
                        f"cube_z={cube_pos[2]:.4f}  "
                        f"cube_above_tbl={cube_pos[2]-self._table_z:.4f}  "
                        f"reached={z_err < 0.012}  timeout={self.step_in_phase >= LIFT_CUBE_MAX}",
                        flush=True,
                )
                self._advance("done")
            self.step_in_phase += 1
            return action

        # ────────────────────────────────────────────────────────────
        # Done: hold closed, let scorer see the lifted cube
        # ────────────────────────────────────────────────────────────
        if self.step_in_phase == 0:
            if self.verbose:
                print(
                    f"  [TR] DONE  cube_above_tbl={cube_pos[2]-self._table_z:.4f}",
                    flush=True,
            )
        action = np.zeros(7)
        action[6] = 1.0
        self.step_in_phase += 1
        return action


# ── Stand-alone runner ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes",  type=int, default=5)
    parser.add_argument("--seed",      type=int, default=7)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--quiet",     action="store_true", help="suppress [TR] diagnostic prints")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env    = ib.make("ib-p4-tool-retrieve-v1", privileged_obs=True, render_mode=render_mode, obs_mode="combo")
    policy = ToolRetrievePolicy(verbose=not args.quiet)

    for ep in range(args.episodes):
        seed = args.seed + ep * 13
        obs, info = env.reset(seed=seed)
        policy.reset()
        done = False
        print(
            f"\n=== Episode {ep} ===  seed={seed}",
            flush=True,
        )
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            done = terminated or truncated
        print(
            f"Episode {ep}  score={info.get('task_score',0.0):.3f}  "
            f"phase={policy.phase}  t={policy.step_in_phase}",
            flush=True,
        )
    env.close()
