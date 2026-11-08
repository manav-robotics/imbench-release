"""Occluder Push — uncover the red cube under one of several covers and place it on the green patch.

Task ID: ``ib-p5-occluder-push-v1``

A quarter of seeds keep the original scene: three flat plates side by side,
each resting on a cube (red, blue, yellow). The rest lay out 3–4 covers of
at least two kinds on the table, at random positions and headings:

* **plate** — a flat plate resting on a cube; push it aside;
* **cup** — an upside-down cup with a knob on top; a sideways push drags the
  cube along inside it, so it has to be lifted off (by the knob) and set down;
* **flap** — a flap hinged along one edge of a fixed walled frame; lift its
  overhanging lip and swing it past upright to its open stop;
* **spring flap** — looks exactly like a flap, but a spring pulls it shut
  unless it is swung past a latch angle, a little beyond upright;
* **box** — a fixed walled box whose lid slides along one axis only; slide
  it off by its knob.

The red cube is under one cover; coloured distractor cubes (never red,
orange or pink) are under others, and 0–2 covers are empty (plates always
rest on a cube, since an empty plate would lie visibly lower). The layout is
checked so that every cover can still be removed: each plate has a clear
push lane, each flap room to swing and to reach its lip, each box room for
its lid to slide, each cup somewhere to be set down, and nothing sits on
the green patch.

The episode fails if any cube or loose cover leaves the table, and is not a
success while a distractor cube rests on the patch.
"""

import numpy as np

from imbench.tasks.prehensile.contacts import (
    body_geom_ids, geoms_touch, object_geom_ids, robot_contacts,
)
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward

#: Cover kinds, and the body pool each one draws from (both flap kinds
#: share one pool: a spring flap is an ordinary flap with a spring applied).
COVER_KINDS = ("plate", "cup", "flap", "spring_flap", "box")
_POOL_OF = {"plate": "plate", "cup": "cup", "flap": "flap", "spring_flap": "flap", "box": "box"}
_POOL_SIZE = {"plate": 3, "cup": 2, "flap": 3, "box": 2}
#: Most covers of one pool in a mixed scene (three plates only in the original scene).
_MAX_PER_SEED = {"plate": 2, "cup": 2, "flap": 3, "box": 2}

#: Distractor colours: none of them red, orange or pink.
DISTRACTOR_COLORS = {
    "blue": [0.15, 0.4, 0.9, 1.0],
    "yellow": [0.95, 0.8, 0.1, 1.0],
    "purple": [0.5, 0.2, 0.75, 1.0],
    "cyan": [0.1, 0.75, 0.85, 1.0],
    "white": [0.92, 0.92, 0.92, 1.0],
}
_MAX_DISTRACTORS = 4

COVER_RGBA = [0.3, 0.3, 0.3, 1.0]
FRAME_RGBA = [0.25, 0.25, 0.25, 1.0]

# ── Cover geometry (m) ────────────────────────────────────────────────
PLATE_HALF, PLATE_HALF_THICK = 0.06, 0.005
PLATE_FRICTION = 0.15
CUP_IN, CUP_WALL, CUP_H, CUP_TOP = 0.033, 0.004, 0.055, 0.004
KNOB_R, KNOB_HH = 0.012, 0.015
CUP_OUT = CUP_IN + CUP_WALL
FLAP_HALF, FLAP_HALF_Y, FLAP_RAIL_W, FLAP_RAIL_H, FLAP_LIP, FLAP_THICK = 0.06, 0.07, 0.012, 0.045, 0.02, 0.006
BOX_IN, BOX_WALL, BOX_H, BOX_LID_T, BOX_TRAVEL = 0.065, 0.006, 0.05, 0.006, 0.16
BOX_OUT = BOX_IN + BOX_WALL

#: How far a plate's push lane runs past its footprint, and how much room a
#: fingertip needs behind it.
PLATE_PUSH, PLATE_APPROACH = 0.13, 0.05
#: Room an open flap needs past its hinge, and a fingertip past its lip.
FLAP_SWING, FLAP_REACH = 0.06, 0.05

#: Footprints in each cover's own frame: (x0, x1, y0, y1). A flap's hinge
#: runs along y at x = -FLAP_HALF and its lip sticks out past x = +FLAP_HALF;
#: a box lid slides toward +x.
_BODY_RECT = {
    "plate": (-PLATE_HALF, PLATE_HALF, -PLATE_HALF, PLATE_HALF),
    "cup": (-CUP_OUT, CUP_OUT, -CUP_OUT, CUP_OUT),
    "flap": (-FLAP_HALF, FLAP_HALF + FLAP_LIP, -FLAP_HALF_Y - FLAP_RAIL_W, FLAP_HALF_Y + FLAP_RAIL_W),
    "box": (-BOX_OUT, BOX_OUT, -BOX_OUT, BOX_OUT),
}
_SETDOWN_HALF = CUP_OUT + 0.01
_PLACE_ORDER = {"box": 0, "flap": 1, "spring_flap": 1, "plate": 2, "cup": 3}
_PLATE_DIRS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

# Layout rectangles carry a class that decides what they may overlap:
#   body   a cover's footprint;
#   rest   where something ends up once removed (a pushed plate, an open
#          flap, a set-down cup) -- tall or in the way, overlaps nothing;
#   reach  room a fingertip passes through (behind a plate, past a flap's
#          lip) -- may overlap other reach zones;
#   lid    a box lid's slide; it runs 5.6 cm up, above plates (5 cm with
#          their cube) and closed flap frames (5.1 cm), so it may pass over
#          those and over a pushed plate's resting place.
_LOW_KINDS = ("plate", "flap", "spring_flap")


#: Hand footprint (m) when grasping a cube, top-down: its long side runs
#: along the jaw's closing axis; the palm is ~7 cm above the table, above
#: plates, closed flap frames, box walls and slid lids, but not cups or
#: open flaps.
HAND_HALF_LONG, HAND_HALF_SHORT = 0.115, 0.04


def _may_overlap(a, b):
    (ca, ka), (cb, kb) = a, b
    if ca == "reach" and cb == "reach":
        return True
    if ca == "hand" or cb == "hand":
        other_cls, other_kind = (cb, kb) if ca == "hand" else (ca, ka)
        if other_cls in ("hand", "reach", "lid"):
            return True
        if other_cls == "body":
            return other_kind != "cup"
        return other_kind in ("plate", "patch")
    if ca == "lid" or cb == "lid":
        other_cls, other_kind = (cb, kb) if ca == "lid" else (ca, ka)
        return (other_cls == "body" and other_kind in _LOW_KINDS) or (
            other_cls == "rest" and other_kind == "plate")
    return False


def _yaw_quat(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _world_rect(pos, yaw, rect):
    """A local (x0, x1, y0, y1) rectangle as a world box (centre, axes, half sizes)."""
    x0, x1, y0, y1 = rect
    R = _rot(yaw)
    centre = np.asarray(pos[:2]) + R @ np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return centre, R, np.array([(x1 - x0) / 2, (y1 - y0) / 2])


def _boxes_overlap(a, b, gap):
    """Separating-axis test for two world boxes, each grown by ``gap / 2``."""
    (ca, Ra, ha), (cb, Rb, hb) = a, b
    d = cb - ca
    if d[0] * d[0] + d[1] * d[1] > (np.hypot(*ha) + np.hypot(*hb) + gap) ** 2:
        return False          # bounding circles apart
    ha, hb = ha + gap / 2, hb + gap / 2
    for axis in (Ra[:, 0], Ra[:, 1], Rb[:, 0], Rb[:, 1]):
        ra = abs(axis @ Ra[:, 0]) * ha[0] + abs(axis @ Ra[:, 1]) * ha[1]
        rb = abs(axis @ Rb[:, 0]) * hb[0] + abs(axis @ Rb[:, 1]) * hb[1]
        if abs(axis @ d) > ra + rb:
            return False
    return True


def _corners(box):
    c, R, h = box
    return [c + R @ np.array([sx * h[0], sy * h[1]]) for sx in (-1, 1) for sy in (-1, 1)]


class OccluderPushV1(SingleArmBase):
    TASK_ID = "ib-p5-occluder-push-v1"
    DESCRIPTION = (
        "Single arm: either three flat plates in a row, or 3-4 covers of "
        "different kinds (flat plates, upside-down "
        "cups, hinged flaps, spring-loaded flaps that look like plain flaps, "
        "and boxes with sliding lids) hide the red cube, coloured distractor "
        "cubes, or nothing. Remove covers the way each kind allows until the "
        "red cube shows, then place it on the green patch. Knocking a cube or "
        "cover off the table fails the episode, and a distractor left on the "
        "patch blocks success."
    )

    #: Max linear speed (m/s) of a cube that counts as resting on the patch.
    _REST_SPEED = 0.05
    #: Movement (m) a cover may make while settling before it counts as opening.
    _SETTLE_SLACK = 0.01
    #: Static covers not used this episode are moved here (out of every view).
    _STATIC_PARK = (10.0, 10.0, 0.0)

    # ── Config ──────────────────────────────────────────────────────

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._patch_pos = np.array(task_cfg.get("patch_pos", [-0.20, 0.0]))
        self._patch_half = float(task_cfg.get("patch_half", 0.05))
        self._patch_tolerance = task_cfg.get("placement_threshold", 0.04)
        self._cube_size = task_cfg.get("cube_size", 0.02)
        cov = task_cfg.get("covers", {})
        self._count_range = tuple(cov.get("count", [3, 4]))
        self._kinds = tuple(cov.get("kinds", COVER_KINDS))
        self._min_kinds = int(cov.get("min_kinds", 2))
        self._empty_range = tuple(cov.get("empty", [0, 2]))
        self._layout_tries = int(cov.get("layout_tries", 5000))
        self._redraw_every = int(cov.get("redraw_kinds_every", 500))
        self._classic_frac = float(cov.get("original_fraction", 0.25))
        self._body_gap = float(cov.get("body_gap", 0.02))
        self._zone_gap = float(cov.get("zone_gap", 0.01))
        self._reach = float(cov.get("reach", 0.76))
        self._reach_min = float(cov.get("reach_min", 0.27))
        self._reach_high = float(cov.get("reach_high", 0.66))
        self._setdown_min = float(cov.get("setdown_min_reach", 0.36))
        self._patch_keepout = float(cov.get("patch_keepout", 0.09))
        spring = cov.get("spring", {})
        self._spring_k = float(spring.get("stiffness", 0.08))
        self._spring_latch = np.deg2rad(float(spring.get("latch_deg", 100.0)))
        self._flap_open_deg = float(cov.get("flap_open_limit_deg", 110.0))
        self._flap_damping = float(cov.get("flap_damping", 0.01))
        self._distractor_colors = list(task_cfg.get("distractor_colors", list(DISTRACTOR_COLORS)))
        for name in self._distractor_colors:
            if name not in DISTRACTOR_COLORS:
                raise ValueError(f"unknown distractor colour {name!r}")
        for kind in self._kinds:
            if kind not in COVER_KINDS:
                raise ValueError(f"unknown cover kind {kind!r}")

        self._cube_placed = False
        self._cube_spawn_xy = None
        self._covers = []
        self._active_cubes = ["cube"]
        self._parked = set()
        self._spring_joints = []
        self._layout_draws = 0
        self._classic = False

        self._staged_reward = StagedReward(
            stages=[
                Stage("reveal_target", 0.25, lambda t, o: t._reveal_target_progress(o)),
                Stage("grasp_target_cube", 0.25,
                      lambda t, o: 1.0 if t._gripper_contacts_object("cube") else 0.0),
                Stage("transport_toward_patch", 0.20, lambda t, o: t._transport_progress(o)),
                Stage("place_on_patch", 0.30, lambda t, o: t._place_on_patch_progress(o)),
            ],
            success_fn=lambda t, o: (t._place_on_patch_progress(o) >= 1.0
                                     and not t._distractor_on_patch()),
        )

    # ── Model ───────────────────────────────────────────────────────

    def _cube_spec(self, name, rgba):
        return {"type": "box", "name": name, "size": [self._cube_size] * 3, "rgba": rgba,
                "density": 100, "friction": [0.8, 0.005, 0.0001], "joints": "default"}

    def _make_objects(self):
        """Red cube first (``cube_pos`` aliases it), then the distractor,
        plate and cup pools; unused ones are parked each episode."""
        objects = [self._cube_spec("cube", [0.8, 0.2, 0.2, 1.0])]
        for i in range(_MAX_DISTRACTORS):
            objects.append(self._cube_spec(f"distractor_{i}", DISTRACTOR_COLORS["blue"]))
        for i in range(_POOL_SIZE["plate"]):
            objects.append({"type": "box", "name": f"plate_{i}",
                            "size": [PLATE_HALF, PLATE_HALF, PLATE_HALF_THICK],
                            "rgba": COVER_RGBA, "density": 200,
                            "friction": [0.4, 0.005, 0.0001], "joints": "default"})
        o = CUP_IN + CUP_WALL / 2
        for i in range(_POOL_SIZE["cup"]):
            objects.append({
                "type": "composite", "name": f"cup_{i}", "rgba": COVER_RGBA, "density": 300,
                "friction": [1.0, 0.005, 0.0001], "joints": "default",
                "sub_objects": [
                    {"type": "box", "name": "w_px", "size": [CUP_WALL / 2, CUP_OUT, CUP_H / 2]},
                    {"type": "box", "name": "w_nx", "size": [CUP_WALL / 2, CUP_OUT, CUP_H / 2]},
                    {"type": "box", "name": "w_py", "size": [CUP_IN, CUP_WALL / 2, CUP_H / 2]},
                    {"type": "box", "name": "w_ny", "size": [CUP_IN, CUP_WALL / 2, CUP_H / 2]},
                    {"type": "box", "name": "top", "size": [CUP_OUT, CUP_OUT, CUP_TOP / 2]},
                    {"type": "cylinder", "name": "knob", "size": [KNOB_R, KNOB_HH]}],
                "sub_locations": [[o, 0, CUP_H / 2], [-o, 0, CUP_H / 2], [0, o, CUP_H / 2],
                                  [0, -o, CUP_H / 2], [0, 0, CUP_H + CUP_TOP / 2],
                                  [0, 0, CUP_H + CUP_TOP + KNOB_HH]],
                "total_size": [CUP_OUT, CUP_OUT, (CUP_H + CUP_TOP + 2 * KNOB_HH) / 2]})
        return objects

    def _make_static_objects(self):
        objects = [
            {"type": "box", "name": "goal_patch",
             "size": [self._patch_half, self._patch_half, 0.001],
             "rgba": [0.2, 0.7, 0.2, 0.5],
             "density": 1000, "friction": [0.5, 0.005, 0.0001], "joints": None},
        ]
        for i in range(_POOL_SIZE["flap"]):
            objects.append({"type": "hinged_flap", "name": f"flap_{i}",
                            "half_x": FLAP_HALF, "half_y": FLAP_HALF_Y, "thickness": FLAP_THICK,
                            "rail_height": FLAP_RAIL_H, "rail_width": FLAP_RAIL_W, "lip": FLAP_LIP,
                            "flap_rgba": COVER_RGBA, "rail_rgba": FRAME_RGBA,
                            "open_limit_deg": self._flap_open_deg, "damping": self._flap_damping,
                            "end_walls": True})
        for i in range(_POOL_SIZE["box"]):
            objects.append({"type": "sliding_lid_box", "name": f"box_{i}",
                            "inner_half": BOX_IN, "wall": BOX_WALL, "height": BOX_H,
                            "lid_thickness": BOX_LID_T, "travel": BOX_TRAVEL,
                            "box_rgba": FRAME_RGBA, "lid_rgba": COVER_RGBA})
        return objects

    # ── Per-seed scene ──────────────────────────────────────────────

    def _draw_kinds(self, n=None):
        lo, hi = self._count_range
        if n is None:
            n = int(self._rng.randint(lo, hi + 1))
        for _ in range(200):
            kinds = [str(self._rng.choice(self._kinds)) for _ in range(n)]
            used = {}
            for k in kinds:
                used[_POOL_OF[k]] = used.get(_POOL_OF[k], 0) + 1
            if (len(set(kinds)) >= min(self._min_kinds, len(self._kinds))
                    and all(used[p] <= _MAX_PER_SEED[p] for p in used)):
                return kinds
        raise RuntimeError("could not draw cover kinds within the body pools")

    def _cover_rects(self, kind, pos, yaw, push_dir=None):
        """The cover's layout rectangles as ``(world box, class, kind)`` plus
        the points a fingertip must reach to remove it."""
        local = [(_BODY_RECT[_POOL_OF[kind]], "body")]
        pts = [(0.0, 0.0)]
        high = []
        if kind == "plate":
            dx, dy = push_dir
            h, a, b = PLATE_HALF, PLATE_PUSH, PLATE_APPROACH
            if dx:
                s = dx
                local += [((min(s * h, s * (h + a)), max(s * h, s * (h + a)), -h, h), "rest"),
                          ((min(-s * h, -s * (h + b)), max(-s * h, -s * (h + b)), -0.035, 0.035), "reach")]
                pts += [(-s * (h + 0.03), 0.0), (s * (a - h), 0.0)]
            else:
                s = dy
                local += [((-h, h, min(s * h, s * (h + a)), max(s * h, s * (h + a))), "rest"),
                          ((-0.035, 0.035, min(-s * h, -s * (h + b)), max(-s * h, -s * (h + b))), "reach")]
                pts += [(0.0, -s * (h + 0.03)), (0.0, s * (a - h))]
        elif kind in ("flap", "spring_flap"):
            lip = FLAP_HALF + FLAP_LIP
            wide = FLAP_HALF_Y + FLAP_RAIL_W
            local += [((lip, lip + FLAP_REACH, -0.045, 0.045), "reach"),
                      ((-FLAP_HALF - FLAP_SWING, -FLAP_HALF, -wide, wide), "rest")]
            pts += [(lip + 0.035, 0.0), (lip, 0.0)]
            high = [(-FLAP_HALF - 0.08, 0.0)]   # the hand passes here, ~15 cm up, opening it
        elif kind == "box":
            local += [((BOX_OUT, BOX_OUT + BOX_TRAVEL + 0.01, -BOX_OUT, BOX_OUT), "lid")]
        if kind in ("flap", "spring_flap", "box"):
            # Inside a frame or box the jaw closes along the cover's y axis.
            local += [((-HAND_HALF_SHORT, HAND_HALF_SHORT, -HAND_HALF_LONG, HAND_HALF_LONG), "hand")]
            pts += [(-0.032, 0.0), (BOX_TRAVEL - 0.02, 0.0)]
        R = _rot(yaw)
        rects = [(_world_rect(pos, yaw, r), cls, kind) for r, cls in local]
        world = lambda ps: [np.asarray(pos[:2]) + R @ np.array(p) for p in ps]
        return rects, world(pts), world(high)

    def _on_table(self, box):
        tx, ty = self._table_full_size[0] / 2 - 0.01, self._table_full_size[1] / 2 - 0.01
        return all(-tx <= p[0] <= tx and -ty <= p[1] <= ty for p in _corners(box))

    def _reachable(self, p, high=False):
        r = np.linalg.norm(np.asarray(p) - self._robot_base_xy)
        return self._reach_min <= r <= (self._reach_high if high else self._reach)

    def _carry_clear(self, a, b, placed):
        """The cup, carried 13 cm up from ``a`` to ``b``, passes no open flap."""
        flaps = [box for box, cls, kind in placed if cls == "rest" and kind in ("flap", "spring_flap")]
        n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / 0.02))
        for t in np.linspace(0.0, 1.0, n):
            p = (1 - t) * np.asarray(a) + t * np.asarray(b)
            sq = (p, np.eye(2), np.full(2, CUP_OUT))
            if any(_boxes_overlap(sq, f, 0.0) for f in flaps):
                return False
        return True

    def _cup_reachable(self, xy):
        """A lifted cup is set down away from the arm's folded-up zone near
        the base and from its stretched-out reach."""
        r = np.linalg.norm(np.asarray(xy) - self._robot_base_xy)
        return self._setdown_min <= r <= self._reach_high

    def _fits(self, rects, pts, placed, high=()):
        """Every rectangle on the table, every body and fingertip point in
        reach, and no overlap the classes do not allow."""
        for box, cls, _ in rects:
            if not self._on_table(box):
                return False
            if cls == "body" and not all(self._reachable(c) for c in _corners(box)):
                return False
        if not all(self._reachable(p) for p in pts) or not all(self._reachable(p, True) for p in high):
            return False
        if any(kind == "cup" and cls == "rest" and not self._cup_reachable(box[0])
               for box, cls, kind in rects):
            return False
        for box, cls, kind in rects:
            for obox, ocls, okind in placed:
                if _may_overlap((cls, kind), (ocls, okind)):
                    continue
                gap = self._body_gap if cls == ocls == "body" else self._zone_gap
                if _boxes_overlap(box, obox, gap):
                    return False
        return True

    def _sample_layout(self, kinds):
        """Place covers one by one, mostly near one already placed so the
        layout is cluttered; returns a list of cover dicts or None."""
        keep_out = (self._patch_pos.copy(), np.eye(2), np.full(2, self._patch_keepout))
        placed = [(keep_out, "rest", "patch")]
        covers = []
        tx, ty = self._table_full_size[0] / 2, self._table_full_size[1] / 2
        for kind in kinds:
            ok = False
            for _ in range(80):
                yaw = float(self._rng.uniform(-np.pi, np.pi))
                if covers and self._rng.uniform() < 0.7:
                    anchor = covers[self._rng.randint(len(covers))]["pos"]
                    ang = self._rng.uniform(-np.pi, np.pi)
                    pos = anchor + self._rng.uniform(0.11, 0.28) * np.array([np.cos(ang), np.sin(ang)])
                else:
                    pos = np.array([self._rng.uniform(-tx, tx), self._rng.uniform(-ty, ty)])
                push = None
                if kind == "plate":
                    for k in self._rng.permutation(4):
                        rects, pts, high = self._cover_rects(kind, pos, yaw, _PLATE_DIRS[k])
                        if self._fits(rects, pts, placed, high):
                            push = _PLATE_DIRS[k]
                            break
                    if push is None:
                        continue
                else:
                    rects, pts, high = self._cover_rects(kind, pos, yaw)
                    if not self._fits(rects, pts, placed, high):
                        continue
                placed += rects
                covers.append({"kind": kind, "pos": np.asarray(pos, dtype=float), "yaw": yaw,
                               "push_dir": push})
                ok = True
                break
            if not ok:
                return None
        # Somewhere to set each cup down, clear of everything else.
        for cover in covers:
            if cover["kind"] != "cup":
                continue
            spot = None
            for _ in range(400):
                # Drawn inside the band of reach a set-down spot must lie in.
                r = self._rng.uniform(self._setdown_min, self._reach_high)
                a = self._rng.uniform(-np.pi / 2, np.pi / 2)
                p = self._robot_base_xy + r * np.array([np.cos(a), np.sin(a)])
                box = (p, np.eye(2), np.full(2, _SETDOWN_HALF))
                if self._fits([(box, "rest", "cup")], [p], placed) and self._carry_clear(cover["pos"], p, placed):
                    spot = p
                    break
            if spot is None:
                return None
            placed.append(((spot, np.eye(2), np.full(2, _SETDOWN_HALF)), "rest", "cup"))
            cover["setdown"] = spot
        return covers

    # ── Removal order: keep the hand clear of tall things ───────────

    def _hand_zones(self, c):
        """Where the palm passes while removing cover ``c``: ``(box, low)``;
        ``low`` zones (palm ~7 cm up) are blocked by cups as well as open
        flaps, the rest only by open flaps."""
        L, S = HAND_HALF_LONG, HAND_HALF_SHORT
        kind, pos, yaw = c["kind"], c["pos"], c["yaw"]
        if kind in ("flap", "spring_flap"):
            lip = FLAP_HALF + FLAP_LIP
            return [(_world_rect(pos, yaw, (0.02, lip + 0.05, -L, L)), True),
                    (_world_rect(pos, yaw, (-FLAP_HALF - 0.09, 0.02, -L, L)), False)]
        if kind == "box":
            return [(_world_rect(pos, yaw, (-0.07, BOX_TRAVEL + 0.03, -L, L)), False)]
        if kind == "plate":
            dx, dy = c["push_dir"]
            h, a, b = PLATE_HALF, PLATE_PUSH, PLATE_APPROACH
            if dx:
                r = (min(-dx * (h + b), dx * (h + a)), max(-dx * (h + b), dx * (h + a)), -L, L)
            else:
                r = (-L, L, min(-dy * (h + b), dy * (h + a)), max(-dy * (h + b), dy * (h + a)))
            return [(_world_rect(pos, yaw, r), False)]
        return []          # a cup is lifted by its knob, palm well above

    def _grasp_zones(self, c):
        """Hand footprints that could grasp the cube under ``c`` (any one will do)."""
        L, S = HAND_HALF_LONG, HAND_HALF_SHORT
        if c["kind"] in ("plate", "cup"):
            return [_world_rect(c["pos"], c["yaw"] + k * np.pi / 2, (-S, S, -L, L)) for k in (0, 1)]
        return [_world_rect(c["pos"], c["yaw"], (-S, S, -L, L))]

    def _blockers(self, covers, opened, skip):
        """``(tall, low)`` boxes standing after ``opened`` covers were removed."""
        tall, low = [], []
        for i, c in enumerate(covers):
            if i == skip:
                continue
            if c["kind"] in ("flap", "spring_flap") and i in opened:
                rects, _, _ = self._cover_rects(c["kind"], c["pos"], c["yaw"])
                tall += [b for b, cls, _ in rects if cls == "rest"]
            elif c["kind"] == "cup":
                p = c["setdown"] if i in opened else c["pos"]
                low.append((np.asarray(p, dtype=float), np.eye(2), np.full(2, CUP_OUT)))
        return tall, low

    def _order_clear(self, covers, order, red):
        opened = set()
        for i in order:
            tall, low = self._blockers(covers, opened, i)
            for box, is_low in self._hand_zones(covers[i]):
                if any(_boxes_overlap(box, b, 0.0) for b in tall + (low if is_low else [])):
                    return False
            opened.add(i)
            if i == red:
                tall, low = self._blockers(covers, opened, i)
                return any(not any(_boxes_overlap(z, b, 0.0) for b in tall + low)
                           for z in self._grasp_zones(covers[i]))
        return False

    def clear_orders(self, covers=None, red=None, limit=None):
        """Removal orders (cover indices, up to the red cube's cover) whose
        hand paths stay clear; host-side, for the layout check and the
        scripted policy."""
        import itertools
        covers = self._covers if covers is None else covers
        red = self._red_index if red is None else red
        out, seen = [], set()
        for perm in itertools.permutations(range(len(covers))):
            prefix = perm[:perm.index(red) + 1]
            if prefix in seen:
                continue
            seen.add(prefix)
            if self._order_clear(covers, prefix, red):
                out.append(prefix)
                if limit and len(out) >= limit:
                    break
        return out

    def _mixed_layout(self):
        """3-4 covers of at least two kinds; returns (covers, red, filled, draws)."""
        # The cover count and kinds are drawn once, so layouts that are hard
        # to fit are not quietly traded for easier ones.
        kinds = self._draw_kinds()
        covers = None
        draws = 0
        while covers is None:
            draws += 1
            if draws > self._layout_tries:
                raise RuntimeError(f"no valid occluder layout in {self._layout_tries} draws")
            if draws % self._redraw_every == 0:
                # A rare mix of large covers that will not fit: same count, new kinds.
                kinds = self._draw_kinds(n=len(kinds))
            # Largest first: they are the hardest to fit once others are down.
            covers = self._sample_layout(sorted(self._rng.permutation(kinds), key=lambda k: _PLACE_ORDER[k]))
            if covers is None:
                continue
            # Which covers hide a cube: plates always do; up to the drawn
            # number of the others stay empty; the red cube goes under any.
            n = len(covers)
            lo, hi = self._empty_range
            n_empty = int(self._rng.randint(lo, hi + 1))
            others = [i for i, c in enumerate(covers) if c["kind"] != "plate"]
            n_empty = min(n_empty, len(others), n - 1)
            empty = set(self._rng.choice(others, size=n_empty, replace=False).tolist()) if n_empty else set()
            filled = [i for i in range(n) if i not in empty]
            red = int(self._rng.choice(filled))
            for i, c in enumerate(covers):
                c["has_cube"] = i in filled
            # Some order of removing covers must keep the hand clear of what
            # earlier removals left standing (open flaps, cups).
            if not self.clear_orders(covers, red, limit=1):
                covers = None
        return covers, red, filled, draws

    def _classic_layout(self):
        """The original scene: three plates side by side along y, each on a
        cube (red, blue, yellow), jittered as in the first version of the
        task; each plate's push lane runs toward or away from the robot,
        whichever fits."""
        pitch = 2 * PLATE_HALF + 0.03
        keep_out = (self._patch_pos.copy(), np.eye(2), np.full(2, self._patch_keepout))
        for draws in range(1, self._layout_tries + 1):
            placed = [(keep_out, "rest", "patch")]
            covers = []
            for k in range(3):
                pos = np.array([0.12 + self._rng.uniform(-0.06, 0.06),
                                (k - 1) * pitch + self._rng.uniform(-0.004, 0.004)])
                push = None
                for d in [((-1.0, 0.0), (1.0, 0.0))[j] for j in self._rng.permutation(2)]:
                    rects, pts, high = self._cover_rects("plate", pos, 0.0, d)
                    if self._fits(rects, pts, placed, high):
                        push = d
                        placed += rects
                        break
                if push is None:
                    break
                covers.append({"kind": "plate", "pos": pos, "yaw": 0.0, "push_dir": push, "has_cube": True})
            if len(covers) == 3:
                return covers, int(self._rng.randint(3)), [0, 1, 2], draws
        raise RuntimeError(f"no valid original occluder layout in {self._layout_tries} draws")

    def _cube_offset(self, kind):
        """Random cube position inside a cover, in the cover's frame. In a
        flap frame or a box the cube stays centred across the cover's y axis,
        so a partly opened jaw closing along that axis clears the walls."""
        r = self._rng.uniform
        if kind == "plate":
            return np.array([r(-0.008, 0.008), r(-0.008, 0.008)])
        if kind == "cup":
            return np.array([r(-0.006, 0.006), r(-0.006, 0.006)])
        if kind in ("flap", "spring_flap"):
            # Between the end walls, which are inset by a rail width.
            return np.array([r(-0.018, 0.018), r(-0.004, 0.004)])
        return np.array([r(-0.03, 0.03), r(-0.004, 0.004)])

    def _cube_yaw_jitter(self, kind):
        lim = 0.08 if kind in ("flap", "spring_flap", "box") else 0.2
        return self._rng.uniform(-lim, lim)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._cube_placed = False
        self._staged_reward.reset()
        sim = self._env.sim
        self._robot_base_xy = np.array(sim.data.body_xpos[sim.model.body_name2id("robot0_base")][:2])

        # A quarter of seeds keep the original scene: three plates in a row.
        self._classic = bool(self._rng.uniform() < self._classic_frac)
        if self._classic:
            covers, red, filled, draws = self._classic_layout()
        else:
            covers, red, filled, draws = self._mixed_layout()
        self._layout_draws = draws
        self._red_index = red

        colours = (["blue", "yellow"] if self._classic
                   else list(self._rng.permutation(self._distractor_colors)))
        cube_of, k = {}, 0
        for i in filled:
            if i == red:
                cube_of[i] = "cube"
            else:
                cube_of[i] = f"distractor_{k}"
                self._set_body_rgba(cube_of[i], DISTRACTOR_COLORS[colours[k % len(colours)]])
                k += 1
        self._active_cubes = ["cube"] + [f"distractor_{j}" for j in range(k)]

        # Bodies from each pool, in a random order so names carry nothing.
        free = {p: list(self._rng.permutation(_POOL_SIZE[p])) for p in _POOL_SIZE}
        table_z = self._table_height
        cube_z = table_z + self._cube_size + 0.001
        used = set()
        self._spring_joints = []
        for i, cover in enumerate(covers):
            pool = _POOL_OF[cover["kind"]]
            name = f"{pool}_{free[pool].pop()}"
            used.add(name)
            cover["name"] = name
            cover["cube"] = cube_of.get(i)
            quat = _yaw_quat(cover["yaw"])
            pos = cover["pos"]
            if cover["cube"] is not None:
                off = _rot(cover["yaw"]) @ self._cube_offset(cover["kind"])
                cyaw = cover["yaw"] + self._cube_yaw_jitter(cover["kind"])
                self._place_object(cover["cube"], [pos[0] + off[0], pos[1] + off[1], cube_z],
                                   _yaw_quat(cyaw))
                self._set_object_velocity(cover["cube"], [0, 0, 0], [0, 0, 0])
            if pool == "plate":
                z = table_z + 2 * self._cube_size + PLATE_HALF_THICK + 0.002
                self._place_object(name, [pos[0], pos[1], z], quat)
                self._set_object_velocity(name, [0, 0, 0], [0, 0, 0])
            elif pool == "cup":
                self._place_object(name, [pos[0], pos[1], table_z + 0.001], quat)
                self._set_object_velocity(name, [0, 0, 0], [0, 0, 0])
            else:
                self._place_static(name, [pos[0], pos[1], table_z], quat)
        self._covers = covers

        # Park everything this episode does not use.
        parked = {f"distractor_{j}" for j in range(k, _MAX_DISTRACTORS)}
        parked |= {f"{p}_{j}" for p in ("plate", "cup") for j in range(_POOL_SIZE[p])
                   if f"{p}_{j}" not in used}
        self._parked = parked
        self._park_objects(parked)
        for j, name in enumerate(f"{p}_{i}" for p in ("flap", "box") for i in range(_POOL_SIZE[p])):
            if name not in used:
                x0, y0, z0 = self._STATIC_PARK
                self._place_static(name, [x0 + j, y0, z0])

        # Joints closed and still; springs only on this episode's spring flaps.
        for i in range(_POOL_SIZE["flap"]):
            qa, da = self._joint_adr(f"flap_{i}_flap_hinge")
            sim.data.qpos[qa] = 0.0
            sim.data.qvel[da] = 0.0
            sim.data.qfrc_applied[da] = 0.0
        for i in range(_POOL_SIZE["box"]):
            qa, da = self._joint_adr(f"box_{i}_lid_slide")
            sim.data.qpos[qa] = 0.0
            sim.data.qvel[da] = 0.0
        for cover in covers:
            if cover["kind"] == "spring_flap":
                self._spring_joints.append(self._joint_adr(f"{cover['name']}_flap_hinge"))

        # Plates slide: their low friction wins (geom priority) over the
        # cube's grippy surface, so a pushed plate does not drag its cube.
        for i in range(_POOL_SIZE["plate"]):
            bid = sim.model.body_name2id(f"plate_{i}_main")
            for g in range(sim.model.ngeom):
                if sim.model.geom_bodyid[g] == bid:
                    sim.model.geom_friction[g][0] = PLATE_FRICTION
                    sim.model.geom_priority[g] = 1

        self._place_static("goal_patch", [self._patch_pos[0], self._patch_pos[1], table_z + 0.001])
        red_cover = next(c for c in covers if c["cube"] == "cube")
        self._red_cover = red_cover
        sim.forward()
        self._cube_spawn_xy = np.array(sim.data.body_xpos[sim.model.body_name2id("cube_main")][:2])
        # Opening is measured from where a loose cover starts relative to its
        # cube (a cube sits up to ~1 cm off a cover's centre).
        for c in covers:
            if c["kind"] in ("plate", "cup") and c["cube"]:
                c["rest_offset"] = self._body_pos(c["name"])[:2] - self._body_pos(c["cube"])[:2]
        obs = self._rebuild_obs()
        return obs, info

    def _parked_objects(self):
        return set(self._parked)

    def _joint_adr(self, joint):
        sim = self._env.sim
        j = sim.model.joint_name2id(joint)
        return int(sim.model.jnt_qposadr[j]), int(sim.model.jnt_dofadr[j])

    def _set_body_rgba(self, name, rgba):
        sim = self._env.sim
        body_id = sim.model.body_name2id(f"{name}_main")
        for g in range(sim.model.ngeom):
            if sim.model.geom_bodyid[g] == body_id:
                sim.model.geom_rgba[g] = np.asarray(rgba, dtype=float)

    # ── Physics: the spring flaps ───────────────────────────────────

    def _apply_springs(self):
        sim = self._env.sim
        for qa, da in self._spring_joints:
            theta = sim.data.qpos[qa]
            sim.data.qfrc_applied[da] = -self._spring_k * theta if theta < self._spring_latch else 0.0

    def step(self, action):
        self._apply_springs()
        return super().step(action)

    # ── State helpers (host side; never in observations) ────────────

    def _body_pos(self, name):
        sim = self._env.sim
        body = f"{name}_root" if name.startswith("cup_") else f"{name}_main"
        return np.array(sim.data.body_xpos[sim.model.body_name2id(body)])

    def cover_opening(self, cover):
        """How far a cover is from hiding its spot, in [0, 1]: 1 once whatever
        it covered is exposed from above."""
        sim = self._env.sim
        kind, name = cover["kind"], cover["name"]
        spot = self._body_pos(cover["cube"]) if cover.get("cube") else np.array([*cover["pos"], 0.0])
        if kind == "plate":
            d = np.linalg.norm(self._body_pos(name)[:2] - spot[:2] - cover.get("rest_offset", 0.0)) - self._SETTLE_SLACK
            return float(np.clip(d / (PLATE_HALF + self._cube_size), 0.0, 1.0))
        if kind == "cup":
            p = self._body_pos(name)
            lift = p[2] - self._table_height - self._SETTLE_SLACK
            d = np.linalg.norm(p[:2] - spot[:2] - cover.get("rest_offset", 0.0)) - self._SETTLE_SLACK
            return float(np.clip(max(lift / (2 * self._cube_size + 0.01),
                                     d / (CUP_OUT + self._cube_size)), 0.0, 1.0))
        if kind in ("flap", "spring_flap"):
            qa, _ = self._joint_adr(f"{name}_flap_hinge")
            return float(np.clip((sim.data.qpos[qa] - 0.05) / (np.pi / 2 - 0.05), 0.0, 1.0))
        qa, _ = self._joint_adr(f"{name}_lid_slide")
        local_x = (_rot(cover["yaw"]).T @ (spot[:2] - cover["pos"]))[0]
        need = max(1e-3, local_x + self._cube_size + BOX_OUT)
        return float(np.clip((sim.data.qpos[qa] - self._SETTLE_SLACK) / need, 0.0, 1.0))

    # ── Scoring ─────────────────────────────────────────────────────

    def _reveal_target_progress(self, raw_obs):
        if not self._covers:
            return 0.0
        return self.cover_opening(self._red_cover)

    def _transport_progress(self, raw_obs):
        if "cube_pos" not in raw_obs or self._cube_spawn_xy is None:
            return 0.0
        cube_xy = raw_obs["cube_pos"][:2]
        dist = float(np.linalg.norm(cube_xy - self._patch_pos))
        init_dist = float(np.linalg.norm(self._cube_spawn_xy - self._patch_pos))
        if init_dist <= self._SETTLE_SLACK:
            return 1.0
        # Settling nudges the cube a little: no credit for the first centimetre.
        return float(np.clip((init_dist - dist - self._SETTLE_SLACK) / (init_dist - self._SETTLE_SLACK), 0.0, 1.0))

    def _cube_speed(self):
        """Linear speed (m/s) of the red cube's free joint."""
        sim = self._env.sim
        dof = sim.model.jnt_dofadr[sim.model.body_jntadr[sim.model.body_name2id("cube_main")]]
        return float(np.linalg.norm(sim.data.qvel[dof:dof + 3]))

    def _place_on_patch_progress(self, raw_obs):
        """1.0 iff the red cube is at rest on the patch (or the table inside the
        placement band) and untouched by the robot, else 0.0."""
        if "cube_pos" not in raw_obs:
            return 0.0
        cube_xy = raw_obs["cube_pos"][:2]
        cube_z = raw_obs["cube_pos"][2]
        dist_to_patch = np.linalg.norm(cube_xy - self._patch_pos)
        if not (dist_to_patch < self._patch_tolerance
                and cube_z < self._table_height + 0.05
                and self._cube_speed() < self._REST_SPEED):
            return 0.0
        sim = self._env.sim
        cube_geoms = object_geom_ids(self._env, "cube")
        support = (body_geom_ids(sim, "goal_patch_main")
                   | {sim.model.geom_name2id("table_collision")})
        if robot_contacts(sim, cube_geoms) or not geoms_touch(sim, cube_geoms, support):
            return 0.0
        return 1.0

    def _distractor_on_patch(self):
        """Whether a distractor cube rests on (or overlaps) the green patch."""
        reach = self._patch_half + self._cube_size
        for name in self._active_cubes[1:]:
            p = self._body_pos(name)
            if (np.all(np.abs(p[:2] - self._patch_pos) < reach)
                    and p[2] < self._table_height + 0.06):
                return True
        return False

    def _something_fell(self):
        """Whether any cube or loose cover in play has left the table."""
        names = list(self._active_cubes) + [c["name"] for c in self._covers
                                            if c["kind"] in ("plate", "cup")]
        return any(self._body_pos(n)[2] < self._failure_floor_threshold for n in names)

    def _compute_task_score(self, raw_obs):
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        if self._stage_debug.get("_success"):
            self._cube_placed = True
        return score

    def _check_termination(self, raw_obs):
        if self._cube_placed:
            return True
        return self._something_fell()
