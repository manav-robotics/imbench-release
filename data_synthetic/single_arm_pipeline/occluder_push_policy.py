"""Scripted occluder-push policy for ``ib-p5-occluder-push-v1``.

The task lays out 3–4 covers of several kinds (plates, upside-down cups,
hinged flaps, spring flaps, sliding-lid boxes); one hides the red cube.
The policy visits the covers in a random order, removing each the way its
kind allows, until it has uncovered the red cube; it then picks the cube up
and places it on the green patch.

* plate  — closed fingers behind it, push it along its clear lane;
* cup    — grasp the knob, lift, set it down on its free spot;
* flap   — fingertip under the overhanging lip, then follow the lip's arc
  over the hinge past the spring latch (plain and spring flaps are handled
  the same way, so the policy never needs to know which is which);
* box    — fingertip behind the lid's knob, slide the lid to its stop.

Layout facts (cover kinds, poses, each plate's clear lane, each cup's
set-down spot) come from the task's host-side ``_covers`` list and body
poses from ``env.sim``; this is a privileged data-collection oracle.

``max_action`` caps every motion channel (0.3 matches the gamepad's default
teleop speed) so a run can check that the scene is doable by teleoperation.
"""

from __future__ import annotations

import numpy as np

from imbench.policy.base_policy import PolicyInterface
from imbench.tasks.prehensile.occluder_push_v1 import (
    BOX_H, BOX_LID_T, BOX_TRAVEL, CUP_H, CUP_TOP, FLAP_HALF, FLAP_LIP, FLAP_RAIL_H,
    FLAP_RAIL_W, KNOB_HH, PLATE_HALF, PLATE_PUSH,
)

TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw, EEF Z → world −Z
GRIP_SITE = "gripper0_right_grip_site"
FINGER_JOINT = "gripper0_right_finger_joint1"
HOLD = 0.0             # gripper action that keeps the jaw where it is
GRASP_FINGER_Q = 0.025 # finger travel for a cube grasp: pads 5 cm apart, jaw 11.4 cm wide

OPEN, CLOSED = -1.0, 1.0
SAFE_H = 0.24          # transit height above the table (m), clear of open flaps
CARRY_H = 0.20         # grip height carrying a cup: its rim 13 cm up, above closed covers;
                       # the layout keeps open flaps off its path
PLATE_PUSH_Z = 0.045   # grip-site height for pushing a plate (its edge is 4-5 cm up)
FLAP_UNDER_Z = 0.030   # fingertip height under a flap's lip
FLAP_PUSH_DEG = 125    # follow the lip's arc this far (the stop is at 110 deg, the latch at 100)
FLAP_LATCHED_DEG = 103
NEUTRAL = (-0.05, 0.0)  # between covers the hand returns above here, so the arm never stays stretched
KNOB_Z = BOX_H + BOX_LID_T + 0.0005 + 0.012    # box knob centre above the table
CUP_KNOB_Z = CUP_H + CUP_TOP + KNOB_HH          # cup knob centre above its rim
GAIN = 8.0
ROT_GAIN = 2.0
MAX_ACTION_STEP = 0.5


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _quat_err_axis_angle(cur, tgt):
    cx, cy, cz, cw = cur
    tx, ty, tz, tw = tgt
    ccx, ccy, ccz, ccw = -cx, -cy, -cz, cw
    ex = tw * ccx + tx * ccw + ty * ccz - tz * ccy
    ey = tw * ccy - tx * ccz + ty * ccw + tz * ccx
    ez = tw * ccz + tx * ccy - ty * ccx + tz * ccw
    ew = tw * ccw - tx * ccx - ty * ccy - tz * ccz
    if ew < 0:
        ex, ey, ez, ew = -ex, -ey, -ez, -ew
    sin_half = float(np.sqrt(ex * ex + ey * ey + ez * ez))
    if sin_half < 1e-8:
        return np.zeros(3)
    return np.array([ex, ey, ez]) / sin_half * 2.0 * np.arctan2(sin_half, ew)


def _grasp_quat_for_yaw(yaw):
    """Top-down gripper turned by ``yaw`` about world z (fingers close
    along world y at yaw 0)."""
    qz = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
    return _quat_mul_xyzw(qz, TOP_DOWN_QUAT)


def _wrap_half(yaw):
    """Equivalent gripper yaw in [-pi/2, pi/2) (the jaw is symmetric)."""
    return (yaw + np.pi / 2) % np.pi - np.pi / 2


def _quat_yaw_wxyz(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class OccluderPushPolicy(PolicyInterface):
    """Scripted oracle for ib-p5-occluder-push-v1 (see module docstring)."""

    def __init__(self, verbose: bool = False, rng=None, max_action: float = MAX_ACTION_STEP,
                 rot_max: float = 0.3):
        self.verbose = verbose
        self.rng = np.random.default_rng() if rng is None else rng
        self.max_action = float(max_action)
        self.rot_max = float(rot_max)
        self.reset()

    def reset(self):
        self.phase = "settle"
        self._script = None
        self._obs = None
        self._env = None
        self.log = []
        self.timeouts = []

    # ── Plumbing ──────────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        self._obs, self._env = obs, env
        if self._script is None:
            self._script = self._run()
        try:
            return next(self._script)
        except StopIteration:
            self.phase = "done"
            a = np.zeros(7)
            a[6] = OPEN
            return a

    @property
    def _sim(self):
        env = self._env
        return env._env.sim if hasattr(env, "_env") else env.sim

    def _eef(self):
        sim = self._sim
        return np.array(sim.data.site_xpos[sim.model.site_name2id(GRIP_SITE)])

    def _eef_quat(self):
        return np.asarray(self._obs.get("robot0_eef_quat", TOP_DOWN_QUAT), dtype=float)

    def _pos(self, name):
        return self._env._body_pos(name)

    def _body_yaw(self, name):
        sim = self._sim
        body = f"{name}_root" if name.startswith("cup_") else f"{name}_main"
        return _quat_yaw_wxyz(sim.data.body_xquat[sim.model.body_name2id(body)])

    def _note(self, phase, **kw):
        self.phase = phase
        self.log.append((phase, kw))
        if self.verbose:
            extra = "  ".join(f"{k}={v}" for k, v in kw.items())
            print(f"  [OP] {phase}" + (f"  {extra}" if extra else ""), flush=True)

    def _go(self, target, grip, yaw=0.0, tol=0.006, max_ticks=1500, cap=None, settle=0):
        """Move the fingertips to ``target`` (world xyz) at gripper ``yaw``."""
        cap = self.max_action if cap is None else min(cap, self.max_action)
        q = _grasp_quat_for_yaw(yaw)
        target = np.asarray(target, dtype=float)
        hold = 0
        for _ in range(max_ticks):
            err = target - self._eef()
            ori = _quat_err_axis_angle(self._eef_quat(), q)
            a = np.zeros(7)
            a[:3] = np.clip(GAIN * err, -cap, cap)
            a[3:6] = np.clip(ROT_GAIN * ori, -self.rot_max, self.rot_max)
            a[6] = grip
            if np.linalg.norm(err) < tol and np.linalg.norm(ori) < 0.06:
                hold += 1
                if hold > settle:
                    return
            yield a
        self.timeouts.append((self.phase, np.round(target, 3).tolist(), np.round(self._eef(), 3).tolist(),
                              round(float(np.linalg.norm(_quat_err_axis_angle(self._eef_quat(), q))), 2)))

    def _wait(self, ticks, grip, yaw=0.0):
        """Hold the current position for ``ticks`` while setting the jaw."""
        target = self._eef()
        q = _grasp_quat_for_yaw(yaw)
        cap = self.max_action
        for _ in range(ticks):
            a = np.zeros(7)
            a[:3] = np.clip(GAIN * (target - self._eef()), -cap, cap)
            a[3:6] = np.clip(ROT_GAIN * _quat_err_axis_angle(self._eef_quat(), q), -self.rot_max, self.rot_max)
            a[6] = grip
            yield a

    def _finger_q(self):
        sim = self._sim
        return float(sim.data.qpos[sim.model.jnt_qposadr[sim.model.joint_name2id(FINGER_JOINT)]])

    def _open_partly(self, yaw, q=GRASP_FINGER_Q):
        """Open the jaw only until each finger is ``q`` from centre, then hold
        (the gripper moves while the action is nonzero and holds at 0)."""
        target = self._eef()
        qq = _grasp_quat_for_yaw(yaw)
        for _ in range(300):
            a = np.zeros(7)
            a[:3] = np.clip(GAIN * (target - self._eef()), -self.max_action, self.max_action)
            a[3:6] = np.clip(ROT_GAIN * _quat_err_axis_angle(self._eef_quat(), qq), -self.rot_max, self.rot_max)
            if self._finger_q() >= q:
                return
            a[6] = OPEN
            yield a

    def _via(self, xy, z, grip, yaw):
        """Go up to the transit height, across, then down to ``z``."""
        tz = self._table_z
        here = self._eef()
        top = max(here[2], tz + SAFE_H)
        yield from self._go([here[0], here[1], top], grip, yaw, tol=0.02)
        yield from self._go([xy[0], xy[1], top], grip, yaw, tol=0.01)
        yield from self._go([xy[0], xy[1], tz + z], grip, yaw)

    # ── Script ────────────────────────────────────────────────────────

    def _run(self):
        env = self._env
        self._table_z = float(env._table_height)
        self._note("settle")
        for _ in range(20):
            a = np.zeros(7)
            a[6] = CLOSED
            yield a
        covers = list(env._covers)
        # A random order among those whose hand paths stay clear of what
        # earlier removals leave standing (open flaps, cups); it stops at the
        # red cube's cover.
        clear = env.clear_orders()
        pick = clear[int(self.rng.integers(len(clear)))] if clear else self.rng.permutation(len(covers))
        order = [covers[i] for i in pick]
        for cover in order:
            yield from self._neutral()
            self._note(f"remove_{cover['kind']}", cover=cover["name"])
            yield from getattr(self, f"_remove_{cover['kind'].replace('spring_', '')}")(cover)
            if cover.get("cube") == "cube":
                break
        yield from self._neutral()
        yield from self._pick_and_place()

    def _neutral(self):
        here = self._eef()
        top = max(here[2], self._table_z + SAFE_H)
        yield from self._go([here[0], here[1], top], CLOSED, 0.0, tol=0.03, max_ticks=400)
        yield from self._go([NEUTRAL[0], NEUTRAL[1], self._table_z + SAFE_H], CLOSED, 0.0, tol=0.03, max_ticks=600)

    def _local(self, cover, lx, ly):
        c, s = np.cos(cover["yaw"]), np.sin(cover["yaw"])
        return np.asarray(cover["pos"]) + np.array([c * lx - s * ly, s * lx + c * ly])

    def _remove_plate(self, cover):
        tz = self._table_z
        d = np.array(cover["push_dir"], dtype=float)
        c, s = np.cos(cover["yaw"]), np.sin(cover["yaw"])
        dw = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        p = self._pos(cover["name"])[:2]
        # Fingers side by side across the lane: only their 1.5 cm thickness
        # sticks out ahead.
        yaw = _wrap_half(np.arctan2(dw[1], dw[0]))
        start = p - dw * (PLATE_HALF + 0.03)
        # Jaw open, fingers side by side across the lane: they push the
        # plate's edge either side of its middle and the cube passes between.
        yield from self._via(start, PLATE_PUSH_Z, OPEN, yaw)
        end = p - dw * PLATE_HALF + dw * (PLATE_PUSH - 0.01)
        yield from self._go([end[0], end[1], tz + PLATE_PUSH_Z], OPEN, yaw, cap=0.3, tol=0.01)
        here = self._eef()
        yield from self._go([here[0], here[1], tz + SAFE_H], CLOSED, yaw, tol=0.02)

    def _remove_cup(self, cover):
        tz = self._table_z
        name = cover["name"]
        root = self._pos(name)
        yaw = _wrap_half(self._body_yaw(name))
        knob_z = root[2] - tz + CUP_KNOB_Z
        yield from self._via(root[:2], knob_z + 0.08, OPEN, yaw)
        yield from self._go([root[0], root[1], tz + knob_z - 0.004], OPEN, yaw, tol=0.004)
        yield from self._wait(40, CLOSED, yaw)
        here = self._eef()
        yield from self._go([here[0], here[1], tz + CARRY_H], CLOSED, yaw, tol=0.02)
        spot = cover["setdown"]
        # Carried by its knob, the cup is moved gently so it does not swing loose.
        yield from self._go([spot[0], spot[1], tz + CARRY_H], CLOSED, yaw, tol=0.01, cap=0.25)
        yield from self._go([spot[0], spot[1], tz + CUP_KNOB_Z + 0.006], CLOSED, yaw, tol=0.005)
        yield from self._wait(30, OPEN, yaw)
        yield from self._go([spot[0], spot[1], tz + SAFE_H], OPEN, yaw, tol=0.02)

    def _remove_flap(self, cover):
        for attempt in range(2):
            yield from self._flap_arc(cover)
            qa = self._sim.model.jnt_qposadr[self._sim.model.joint_name2id(f"{cover['name']}_flap_hinge")]
            if np.rad2deg(self._sim.data.qpos[qa]) >= FLAP_LATCHED_DEG:
                return
            self._note("flap_retry", angle=round(float(np.rad2deg(self._sim.data.qpos[qa])), 1))

    def _flap_arc(self, cover):
        tz = self._table_z
        yaw = _wrap_half(cover["yaw"])
        lip_x = FLAP_HALF + FLAP_LIP
        out = self._local(cover, lip_x + 0.035, 0.0)
        yield from self._via(out, FLAP_UNDER_Z, CLOSED, yaw)
        # The fingertip is 3 cm thick across the lip and stops against the
        # frame's end about 1 cm out from the lip's edge; from there, rising
        # catches the lip.
        under = self._local(cover, lip_x + 0.009, 0.0)
        yield from self._go([under[0], under[1], tz + FLAP_UNDER_Z], CLOSED, yaw, cap=0.3, tol=0.006, max_ticks=200)
        r = 2 * FLAP_HALF + FLAP_LIP - 0.004
        for deg in list(range(10, FLAP_PUSH_DEG, 10)) + [FLAP_PUSH_DEG]:
            th = np.deg2rad(deg)
            xy = self._local(cover, -FLAP_HALF + r * np.cos(th) + 0.005, 0.0)
            z = tz + FLAP_RAIL_H + r * np.sin(th) - 0.012
            yield from self._go([xy[0], xy[1], z], CLOSED, yaw, tol=0.008, max_ticks=80, cap=0.4)
        # The flap now rests on its open stop, past the spring latch; leave
        # straight up so as not to drag it back.
        here = self._eef()
        yield from self._go([here[0], here[1], tz + SAFE_H + 0.04], CLOSED, yaw, tol=0.02, max_ticks=400)

    def _remove_box(self, cover):
        tz = self._table_z
        yaw = _wrap_half(cover["yaw"])
        behind = self._local(cover, -0.032, 0.0)
        yield from self._via(behind, KNOB_Z + 0.06, CLOSED, yaw)
        yield from self._go([behind[0], behind[1], tz + KNOB_Z], CLOSED, yaw, tol=0.004)
        for k in range(1, 9):
            p = self._local(cover, -0.032 + (BOX_TRAVEL + 0.012) * k / 8, 0.0)
            yield from self._go([p[0], p[1], tz + KNOB_Z], CLOSED, yaw, tol=0.006, max_ticks=150, cap=0.3)
        here = self._eef()
        yield from self._go([here[0], here[1], tz + SAFE_H], CLOSED, yaw, tol=0.02)

    def _hand_clash(self, xy, yaw):
        """How much of the hand's footprint at ``xy``/``yaw`` overlaps cups
        and open flaps (things taller than the palm's height)."""
        from imbench.tasks.prehensile.occluder_push_v1 import (
            CUP_OUT, HAND_HALF_LONG, HAND_HALF_SHORT, _boxes_overlap, _rot)
        # yaw 0: the jaw closes along world y, so the hand's long side is y.
        hand = (np.asarray(xy, dtype=float), _rot(yaw), np.array([HAND_HALF_SHORT, HAND_HALF_LONG]))
        n = 0
        for cv in self._env._covers:
            if cv["kind"] == "cup":
                p = self._pos(cv["name"])
                n += _boxes_overlap(hand, (p[:2], np.eye(2), np.full(2, CUP_OUT)), 0.01)
            elif "flap" in cv["kind"]:
                rects, _, _ = self._env._cover_rects(cv["kind"], cv["pos"], cv["yaw"])
                n += any(_boxes_overlap(hand, b, 0.01) for b, cls, _ in rects if cls == "rest")
        return n

    def _pick_and_place(self):
        env, tz = self._env, self._table_z
        cover = env._red_cover
        for attempt in range(3):
            self._note("grasp_cube", attempt=attempt)
            c = self._pos("cube")
            cube_yaw = self._body_yaw("cube")
            if cover["kind"] in ("plate", "cup"):
                # In the open: of the two jaw headings square to the cube,
                # the one whose hand keeps clear of cups and open flaps.
                yaw = min((_wrap_half(cube_yaw + k * np.pi / 2) for k in (0, 1)),
                          key=lambda y: self._hand_clash(c[:2], y))
            else:
                # In a flap frame or box: the jaw closes along the cover's y
                # axis (the layout keeps that hand footprint clear).
                k = np.round((cover["yaw"] - cube_yaw) / (np.pi / 2))
                yaw = _wrap_half(cube_yaw + k * np.pi / 2)
            # A partly open jaw fits inside a box or a flap frame.
            yield from self._via(c[:2], c[2] - tz + 0.12, CLOSED, yaw)
            yield from self._open_partly(yaw)
            yield from self._go([c[0], c[1], c[2] + 0.002], HOLD, yaw, tol=0.004)
            yield from self._wait(45, CLOSED, yaw)
            here = self._eef()
            yield from self._go([here[0], here[1], tz + SAFE_H], CLOSED, yaw, tol=0.02)
            if self._pos("cube")[2] > tz + 0.08:
                break
        self._note("transport")
        patch = np.asarray(env._patch_pos, dtype=float)
        # Near the robot's base the hand only gets down with the wrist near
        # its home yaw; the cube may turn with it, its heading on the patch
        # does not matter.
        yaw = yaw - np.pi / 2 * np.round(yaw / (np.pi / 2))
        yield from self._go([patch[0], patch[1], tz + SAFE_H], CLOSED, yaw, tol=0.01)
        # A cube picked up off-centre (say after a plate push turned it)
        # hangs off the fingertips: aim the cube, not the hand, at the patch.
        off = self._pos("cube") - self._eef()
        if np.linalg.norm(off[:2]) > 0.04:
            off = np.array([0.0, 0.0, -0.002])     # not in the hand after all
        aim = patch - off[:2]
        yield from self._go([aim[0], aim[1], tz + SAFE_H], CLOSED, yaw, tol=0.006)
        drop_z = tz + env._cube_size + 0.006 - off[2]
        yield from self._go([aim[0], aim[1], drop_z], CLOSED, yaw, tol=0.004)
        self._note("release")
        yield from self._wait(30, OPEN, yaw)
        here = self._eef()
        yield from self._go([here[0], here[1], tz + SAFE_H], OPEN, yaw, tol=0.02)
