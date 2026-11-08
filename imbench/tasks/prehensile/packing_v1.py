"""Packing Problem (Tetris-style) — single-arm suction puzzle task.

The flat region just inside the bin is decomposed by a Bin-Packing-Problem
generator (Algorithm 2 from the BPP literature: recursively split an item
along a random axis at a random position, weighted by volume / edge length).
The decomposition produces 6 or 8 axis-aligned boxes (``piece_count_range``)
that tile the inner shape, which are then fused pairwise into 3 or 4 L-shaped
pieces, each shrunk slightly about its centroid. A single Panda arm fitted
with a suction gripper picks and places each L-piece flat on the bin floor,
in a single layer.

Task ID: ``ib-p7-packing-v1``
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

import imbench.envs  # noqa: F401
import imbench.grippers  # noqa: F401  — registers PandaSuctionGripper
from imbench.grippers.suction_weld import SuctionWeld
from imbench.tasks.base_task import BaseTask, load_task_config
from imbench.tasks.prehensile.contacts import body_subtree_ids
from imbench.tasks.staged_reward import Stage, StagedReward

# Minimum XY displacement from a piece's spawn position (metres) before it
# counts as "moved" for the picked_and_moved stage — distinguishes an
# intentional pick-and-carry from a piece merely jostled in place.
_PIECE_MOVED_THRESHOLD = 0.03


# Global scale factor applied to every XYZ dimension below (bin and pieces).
_SCALE = 2.0

# Packing box inner half-extents (drives both the generated bin XML and the
# in-bin success check). Sized so the packed pieces leave ~4 mm of slack
# to each other and to the walls: a piece that fits its slot is then
# roughly within the 5 mm target tolerance, and most wrong arrangements
# do not fit.
_BOX_INNER_HALF_X = 0.08 * _SCALE
_BOX_INNER_HALF_Y = 0.0525 * _SCALE
_BOX_WALL_THICK = 0.005 * _SCALE
_BOX_WALL_HEIGHT = 0.015 * _SCALE
_BOX_BASE_THICK = 0.005 * _SCALE   # = 2 × 0.0025

# Inner flat shape — slightly smaller than the bin interior so pieces fit.
_INNER_MARGIN = 0.0015 * _SCALE
_INNER_L = 2 * (_BOX_INNER_HALF_X - _INNER_MARGIN)   # along x
_INNER_W = 2 * (_BOX_INNER_HALF_Y - _INNER_MARGIN)   # along y
_PIECE_THICKNESS = 0.01 * _SCALE                      # along z

# Default range for the number of L-shaped pieces. The actual count is
# sampled per-episode (see ``randomization.piece_count_range`` in the YAML).
_DEFAULT_PIECE_COUNT_RANGE = (3, 4)
_MAX_PIECES = 4  # upper bound so we can pre-allocate dynamic-object slots

# Minimum sub-box edge length so pieces remain graspable.
_MIN_EDGE = 0.018 * _SCALE

# XY shrink factor applied to each L-piece about its centroid just before
# MJCF generation, so neighbours do not share full faces. 0.975 leaves
# ~4 mm between packed pieces.
_PIECE_XY_SHRINK = 0.975

# A piece counts as "settled" only after it has held contact with the bin
# (any wall / base) for this many consecutive sim steps while NOT being
# held by suction. Stops the success check from firing while the cup is
# still descending — the welded piece is geometrically inside the bin
# and flat at that point, but it isn't really packed yet.
_PIECE_SETTLED_CONTACT_STEPS = 50

# Color palette for the 4 L-pieces.
_PIECE_COLORS = [
    (0.10, 0.80, 0.85, 1.0),  # cyan
    (0.95, 0.55, 0.20, 1.0),  # orange
    (0.45, 0.85, 0.30, 1.0),  # green
    (0.85, 0.30, 0.55, 1.0),  # magenta
]

# Asset name for the scaled bin (regenerated on import — idempotent).
_BIN_ASSET = "packing_box_2x"


def _piece_symmetries(piece):
    """Proper 3D rotations preserving this extruded rectangle-union shape.

    Work about the geometric centroid, not an individual sub-box centre.
    The eight planar signed permutations cover rotations and reflections
    of the axis-aligned footprint. A footprint reflection also reverses Z,
    making a proper 3D rotation of the uniform extrusion (turning it over).
    Only transforms with zero footprint symmetric-difference area survive.
    Sub-boxes in generated pieces have disjoint interiors, so pairwise
    intersection areas give the exact union-overlap area.
    """
    boxes = np.asarray(piece["subboxes"], dtype=float)
    centres = boxes[:, :2] - np.asarray(piece["centroid"])
    halves = boxes[:, 2:] / 2
    area = float(np.sum(boxes[:, 2] * boxes[:, 3]))
    rotations = (np.eye(2), np.array([[0, -1], [1, 0]]),
                 -np.eye(2), np.array([[0, 1], [-1, 0]]))
    symmetries = []
    for rotation in rotations:
        for reflection in (np.eye(2), np.diag([1, -1])):
            q = rotation @ reflection
            moved_centres = centres @ q.T
            moved_halves = halves @ np.abs(q).T
            lo = np.maximum((centres-halves)[:, None],
                            (moved_centres-moved_halves)[None])
            hi = np.minimum((centres+halves)[:, None],
                            (moved_centres+moved_halves)[None])
            overlap = float(np.prod(np.maximum(hi-lo, 0), axis=-1).sum())
            if abs(area-overlap) <= area * 1e-10:
                symmetry = np.zeros((3, 3))
                symmetry[:2, :2] = q
                symmetry[2, 2] = np.linalg.det(q)
                symmetries.append(symmetry)
    return np.asarray(symmetries)


def _bin_xml() -> str:
    """Render the open-top rectangular bin as MJCF, scaled to 2×."""
    hx = _BOX_INNER_HALF_X + _BOX_WALL_THICK            # outer half-x
    hy = _BOX_INNER_HALF_Y + _BOX_WALL_THICK            # outer half-y
    base_h = _BOX_BASE_THICK / 2                        # base half-thickness
    wall_hh = _BOX_WALL_HEIGHT / 2                      # wall half-height
    wall_z = _BOX_BASE_THICK + wall_hh                  # wall centre z (above base)
    color = "0.55 0.45 0.30 1"
    return f"""<mujoco model="{_BIN_ASSET}">
  <compiler angle="radian"/>
  <worldbody>
    <body>
      <body name="object" pos="0 0 0">
        <!-- Base plate -->
        <geom name="vis" type="box" size="{hx} {hy} {base_h}"
              pos="0 0 {base_h}" rgba="{color}"
              group="1" contype="0" conaffinity="0"/>
        <geom name="col_base" type="box" size="{hx} {hy} {base_h}"
              pos="0 0 {base_h}" rgba="{color}" group="0"/>
        <!-- Front wall (+Y) -->
        <geom name="col_front_v" type="box"
              size="{hx} {_BOX_WALL_THICK / 2} {wall_hh}"
              pos="0 {_BOX_INNER_HALF_Y + _BOX_WALL_THICK / 2} {wall_z}"
              rgba="{color}" group="1" contype="0" conaffinity="0"/>
        <geom name="col_front" type="box"
              size="{hx} {_BOX_WALL_THICK / 2} {wall_hh}"
              pos="0 {_BOX_INNER_HALF_Y + _BOX_WALL_THICK / 2} {wall_z}"
              rgba="{color}" group="0"/>
        <!-- Back wall (-Y) -->
        <geom name="col_back_v" type="box"
              size="{hx} {_BOX_WALL_THICK / 2} {wall_hh}"
              pos="0 {-(_BOX_INNER_HALF_Y + _BOX_WALL_THICK / 2)} {wall_z}"
              rgba="{color}" group="1" contype="0" conaffinity="0"/>
        <geom name="col_back" type="box"
              size="{hx} {_BOX_WALL_THICK / 2} {wall_hh}"
              pos="0 {-(_BOX_INNER_HALF_Y + _BOX_WALL_THICK / 2)} {wall_z}"
              rgba="{color}" group="0"/>
        <!-- Left wall (-X) -->
        <geom name="col_left_v" type="box"
              size="{_BOX_WALL_THICK / 2} {hy} {wall_hh}"
              pos="{-(_BOX_INNER_HALF_X + _BOX_WALL_THICK / 2)} 0 {wall_z}"
              rgba="{color}" group="1" contype="0" conaffinity="0"/>
        <geom name="col_left" type="box"
              size="{_BOX_WALL_THICK / 2} {hy} {wall_hh}"
              pos="{-(_BOX_INNER_HALF_X + _BOX_WALL_THICK / 2)} 0 {wall_z}"
              rgba="{color}" group="0"/>
        <!-- Right wall (+X) -->
        <geom name="col_right_v" type="box"
              size="{_BOX_WALL_THICK / 2} {hy} {wall_hh}"
              pos="{_BOX_INNER_HALF_X + _BOX_WALL_THICK / 2} 0 {wall_z}"
              rgba="{color}" group="1" contype="0" conaffinity="0"/>
        <geom name="col_right" type="box"
              size="{_BOX_WALL_THICK / 2} {hy} {wall_hh}"
              pos="{_BOX_INNER_HALF_X + _BOX_WALL_THICK / 2} 0 {wall_z}"
              rgba="{color}" group="0"/>
        <inertial pos="0 0 {_BOX_WALL_HEIGHT / 2}" mass="2.0"
                  diaginertia="0.01 0.01 0.01"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005"
            pos="0 0 {_BOX_BASE_THICK + _BOX_WALL_HEIGHT}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{hx} {hy} 0"
            name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""


# The scaled bin XML is written per-instance into a private tempdir (see
# PackingV1.__init__) rather than once at import time into the package's
# own deps/extras/ directory -- writing into the install directory breaks
# on read-only installs (system site-packages, containers, shared HPC) and
# isn't safe across parallel workers.


# ---------------------------------------------------------------------------
#  BPP decomposition
# ---------------------------------------------------------------------------

def _bpp_split_rectangles(L: float, W: float, n: int, rng: np.random.RandomState):
    """Split a flat rectangle of size (L, W) into ``n`` axis-aligned rects.

    Implements Algorithm 2 (BPP generator) restricted to 2D guillotine splits:
    while we have fewer than n items, pop one weighted by area, choose an axis
    weighted by edge length, and split it at a random position weighted by
    distance to the edge centre. ``_MIN_EDGE`` keeps every child graspable.

    Returns a list of ``(cx, cy, lx, ly)`` tuples (centre + full extents).
    """
    # Each item: [cx, cy, lx, ly]
    items = [[0.0, 0.0, L, W]]
    while len(items) < n:
        areas = np.array([it[2] * it[3] for it in items])
        # Weighted pop by area. Skip items that can't be split further.
        splittable = np.array([
            (it[2] >= 2 * _MIN_EDGE) or (it[3] >= 2 * _MIN_EDGE)
            for it in items
        ])
        weights = areas * splittable
        if weights.sum() <= 0:
            raise RuntimeError(
                "BPP split: no item large enough to split further. "
                "Decrease _MIN_EDGE or the number of pieces."
            )
        idx = rng.choice(len(items), p=weights / weights.sum())
        cx, cy, lx, ly = items.pop(idx)

        # Choose axis weighted by edge length, restricted to splittable axes.
        axis_weights = np.array([
            lx if lx >= 2 * _MIN_EDGE else 0.0,
            ly if ly >= 2 * _MIN_EDGE else 0.0,
        ])
        axis = rng.choice(2, p=axis_weights / axis_weights.sum())

        edge = lx if axis == 0 else ly
        # Split position weighted by distance to centre (Algorithm 2). Use a
        # triangular density peaked at the edges, clamped to leave both
        # children >= _MIN_EDGE.
        lo = _MIN_EDGE
        hi = edge - _MIN_EDGE
        # Sample with bias toward the centre via a Beta(1.5, 1.5)-ish dist;
        # this matches "distance to centre" weighting from the paper.
        u = rng.beta(1.5, 1.5)
        pos = lo + u * (hi - lo)

        if axis == 0:
            l1, l2 = pos, edge - pos
            c1 = cx - lx / 2 + l1 / 2
            c2 = cx - lx / 2 + l1 + l2 / 2
            items.append([c1, cy, l1, ly])
            items.append([c2, cy, l2, ly])
        else:
            l1, l2 = pos, edge - pos
            c1 = cy - ly / 2 + l1 / 2
            c2 = cy - ly / 2 + l1 + l2 / 2
            items.append([cx, c1, lx, l1])
            items.append([cx, c2, lx, l2])
    return items


def _are_adjacent(a, b, tol: float = 1e-6):
    """Two axis-aligned rects are adjacent if they share a face segment.

    Returns the shared axis (0 = vertical edge, 1 = horizontal edge) and the
    overlap length, or ``None`` if not adjacent.
    """
    ax_lo, ax_hi = a[0] - a[2] / 2, a[0] + a[2] / 2
    ay_lo, ay_hi = a[1] - a[3] / 2, a[1] + a[3] / 2
    bx_lo, bx_hi = b[0] - b[2] / 2, b[0] + b[2] / 2
    by_lo, by_hi = b[1] - b[3] / 2, b[1] + b[3] / 2

    # Shared vertical edge (touching along x)
    if abs(ax_hi - bx_lo) < tol or abs(bx_hi - ax_lo) < tol:
        overlap = min(ay_hi, by_hi) - max(ay_lo, by_lo)
        if overlap > tol:
            return 0, overlap
    # Shared horizontal edge (touching along y)
    if abs(ay_hi - by_lo) < tol or abs(by_hi - ay_lo) < tol:
        overlap = min(ax_hi, bx_hi) - max(ax_lo, bx_lo)
        if overlap > tol:
            return 1, overlap
    return None


def _union_is_rectangle(a, b, tol: float = 1e-6):
    """Two adjacent boxes union into a rectangle iff they share the full edge."""
    res = _are_adjacent(a, b, tol)
    if res is None:
        return False
    axis, overlap = res
    if axis == 0:
        return abs(overlap - min(a[3], b[3])) < tol and abs(a[3] - b[3]) < tol
    return abs(overlap - min(a[2], b[2])) < tol and abs(a[2] - b[2]) < tol


def _pair_into_l_shapes(boxes, rng: np.random.RandomState):
    """Match adjacent sub-boxes into pairs whose union is L-shaped.

    Backtracking search over all valid perfect matchings of the adjacency
    graph that uses only L-shape edges (adjacent + non-rectangular union).
    Falls back to allowing rectangular unions if no all-L matching exists.

    Returns a list of (box_a, box_b) pairs.
    """
    n = len(boxes)
    assert n % 2 == 0
    # Precompute edge sets.
    l_neighbors = {i: [] for i in range(n)}
    any_neighbors = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if _are_adjacent(boxes[i], boxes[j]) is None:
                continue
            any_neighbors[i].append(j)
            any_neighbors[j].append(i)
            if not _union_is_rectangle(boxes[i], boxes[j]):
                l_neighbors[i].append(j)
                l_neighbors[j].append(i)

    def _search(neighbors):
        order = list(range(n))
        rng.shuffle(order)
        used = [False] * n
        pairs_idx: list[tuple[int, int]] = []

        def recurse():
            i = next((k for k in order if not used[k]), None)
            if i is None:
                return True
            cands = [j for j in neighbors[i] if not used[j]]
            rng.shuffle(cands)
            for j in cands:
                used[i] = used[j] = True
                pairs_idx.append((i, j))
                if recurse():
                    return True
                pairs_idx.pop()
                used[i] = used[j] = False
            return False

        return pairs_idx if recurse() else None

    matched = _search(l_neighbors)
    if matched is None:
        raise RuntimeError("BPP pairing: no all-L matching for this split.")
    return [(boxes[i], boxes[j]) for i, j in matched]


def _generate_l_pieces(rng: np.random.RandomState, num_pieces: int):
    """Build ``num_pieces`` L-shaped pieces tiling the inner rectangle.

    Strategy: BPP-split the inner rectangle into ``2 * num_pieces`` axis-
    aligned boxes, then greedily pair adjacent boxes (preferring L-unions).
    Retries with fresh splits if a pairing has no L-shaped candidates left.

    Returns a list of dicts, each describing one L-piece:
        {
          "subboxes": [(cx, cy, lx, ly), (cx, cy, lx, ly)],  # in inner-rect frame
          "centroid": (cx, cy),                               # area-weighted centroid
          "color":    (r, g, b, a),
        }
    """
    if num_pieces < 1:
        raise ValueError(f"num_pieces must be >= 1, got {num_pieces}")
    num_subboxes = 2 * num_pieces
    last_err = None
    for _ in range(50):
        try:
            boxes = _bpp_split_rectangles(_INNER_L, _INNER_W, num_subboxes, rng)
            pairs = _pair_into_l_shapes(boxes, rng)
            break
        except RuntimeError as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"Failed to generate L-pieces: {last_err}")

    pieces = []
    for k, (a, b) in enumerate(pairs):
        # Shrink the whole L around its own centroid, not each sub-box around
        # its own centre. Scaling each sub-box independently opens a gap at
        # the shared interior edge — the L develops a notch and stops being
        # an L. Scaling positions AND sizes uniformly about the centroid
        # keeps the shared edge glued (both sub-boxes' touching faces map to
        # the same line under a homothety) while the outer perimeter pulls
        # inward, giving the desired gap from neighbouring pieces.
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]
        l_cx = (a[0] * area_a + b[0] * area_b) / (area_a + area_b)
        l_cy = (a[1] * area_a + b[1] * area_b) / (area_a + area_b)
        s = _PIECE_XY_SHRINK
        a = (
            l_cx + (a[0] - l_cx) * s,
            l_cy + (a[1] - l_cy) * s,
            a[2] * s, a[3] * s,
        )
        b = (
            l_cx + (b[0] - l_cx) * s,
            l_cy + (b[1] - l_cy) * s,
            b[2] * s, b[3] * s,
        )
        # Recompute centroid from the shrunk boxes (it equals the homothety
        # centre by construction, but recomputing keeps the data consistent
        # with subboxes for downstream consumers).
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]
        cx = (a[0] * area_a + b[0] * area_b) / (area_a + area_b)
        cy = (a[1] * area_a + b[1] * area_b) / (area_a + area_b)
        # Authoritative grasp anchor for scripted suction-pick policies:
        # the centre of the sub-box with the **largest minimum edge
        # length** — i.e. the most "blocky" of the two. A circular
        # suction snout (radius ~15 mm) needs both the x- and y-half-
        # extents of its target sub-box to clear the snout radius, so
        # picking by max(min(lx, ly)) avoids long thin bars where the
        # snout would overhang the short edge. The area-weighted
        # centroid above sits at the inside corner of the L — empty
        # space — so it's not itself a usable grasp point.
        anchor_sub = a if min(a[2], a[3]) >= min(b[2], b[3]) else b
        grasp_x, grasp_y = anchor_sub[0], anchor_sub[1]
        pieces.append({
            "subboxes": [tuple(a), tuple(b)],
            "centroid": (cx, cy),
            "grasp_anchor": (grasp_x, grasp_y),
            "color": _PIECE_COLORS[k % len(_PIECE_COLORS)],
        })
    return pieces


# ---------------------------------------------------------------------------
#  MJCF generation
# ---------------------------------------------------------------------------

def _l_piece_xml(model_name: str, piece: dict) -> str:
    """Render an L-piece as MuJoCo XML (two box geoms in a free body).

    Sub-box positions are expressed relative to the piece centroid so the
    piece's body origin sits at the geometric centre of mass.
    """
    cx, cy = piece["centroid"]
    r, g, b, a = piece["color"]
    geoms = []
    for k, (sx, sy, lx, ly) in enumerate(piece["subboxes"]):
        # Relative to centroid, lift by half-thickness so body Z=0 is bottom.
        gx, gy = sx - cx, sy - cy
        gz = _PIECE_THICKNESS / 2
        hx, hy, hz = lx / 2, ly / 2, _PIECE_THICKNESS / 2
        geoms.append(
            f'<geom name="vis_{k}" type="box" size="{hx} {hy} {hz}" '
            f'pos="{gx} {gy} {gz}" rgba="{r} {g} {b} {a}" '
            f'group="1" contype="0" conaffinity="0"/>'
        )
        geoms.append(
            f'<geom name="col_{k}" type="box" size="{hx} {hy} {hz}" '
            f'pos="{gx} {gy} {gz}" rgba="{r} {g} {b} {a}" group="0"/>'
        )
    geoms_xml = "\n        ".join(geoms)

    # Half-extents of the bounding box around both sub-boxes (for sites).
    xs_lo = min(sx - lx / 2 for (sx, sy, lx, ly) in piece["subboxes"])
    xs_hi = max(sx + lx / 2 for (sx, sy, lx, ly) in piece["subboxes"])
    ys_lo = min(sy - ly / 2 for (sx, sy, lx, ly) in piece["subboxes"])
    ys_hi = max(sy + ly / 2 for (sx, sy, lx, ly) in piece["subboxes"])
    bbox_hx = max(abs(xs_lo - cx), abs(xs_hi - cx))
    bbox_hy = max(abs(ys_lo - cy), abs(ys_hi - cy))

    return f"""<mujoco model="{model_name}">
  <compiler angle="radian"/>
  <worldbody>
    <body>
      <body name="object" pos="0 0 0">
        {geoms_xml}
        <inertial pos="0 0 {_PIECE_THICKNESS / 2}" mass="0.05"
                  diaginertia="0.00001 0.00001 0.00002"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {_PIECE_THICKNESS}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{bbox_hx} {bbox_hy} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""


def _write_l_piece_assets(pieces: list[dict], out_dir: Path) -> list[str]:
    """Write an XML file per piece into *out_dir* and return their paths.

    ``out_dir`` is a per-instance private tempdir (see PackingV1.__init__),
    so concurrent PackingV1 instances / worker processes never share a
    file, and repeated reset() calls just overwrite the same per-index
    filename in place -- nothing accumulates or needs cross-reset cleanup.
    """
    xml_paths = []
    for k, piece in enumerate(pieces):
        path = out_dir / f"lpiece_{k}.xml"
        path.write_text(_l_piece_xml(f"lpiece_{k}", piece))
        xml_paths.append(str(path))
    return xml_paths


# ---------------------------------------------------------------------------
#  Task class
# ---------------------------------------------------------------------------

class PackingV1(BaseTask):
    """Packing puzzle: pack 4 procedurally-generated L-pieces into a box."""

    TASK_ID = "ib-p7-packing-v1"
    DESCRIPTION = (
        "Pack 3-4 procedurally-generated L-shaped flat blocks (decomposed from "
        "the bin's inner shape via a BPP generator) into the rectangular bin."
    )

    def __init__(self, task_id: str = None, **kwargs):
        if task_id:
            self.TASK_ID = task_id
        self.CONFIG_FILE = f"prehensile/{self.TASK_ID}"

        _cfg = load_task_config(self.TASK_ID, config_file=self.CONFIG_FILE,
                                overrides=kwargs.get("config_overrides"))
        self.DESCRIPTION = _cfg.get("description", self.DESCRIPTION)
        if "max_episode_steps" in _cfg:
            self.MAX_EPISODE_STEPS = _cfg["max_episode_steps"]

        self._rng = np.random.RandomState(kwargs.get("seed", 0))

        # Randomization config
        rand_cfg = _cfg.get("randomization", {})
        place_cfg = rand_cfg.get("block_placement", {})
        self._rand_block_placement = place_cfg.get("enabled", False)
        self._place_x_range = place_cfg.get("x_range", 0.10)
        self._place_y_range = place_cfg.get("y_range", 0.10)

        yaw_cfg = rand_cfg.get("block_yaw", {})
        self._rand_block_yaw = yaw_cfg.get("enabled", False)

        # Per-episode L-shape count, sampled from this inclusive range. The
        # env is rebuilt on each reset() so a different number / shape of
        # L-pieces is presented every episode.
        pc_range = rand_cfg.get("piece_count_range", _DEFAULT_PIECE_COUNT_RANGE)
        self._piece_count_range = (int(pc_range[0]), int(pc_range[-1]))
        if not (1 <= self._piece_count_range[0] <= self._piece_count_range[1] <= _MAX_PIECES):
            raise ValueError(
                f"piece_count_range {self._piece_count_range} must satisfy "
                f"1 <= lo <= hi <= {_MAX_PIECES}"
            )

        # Scene config
        scene_cfg = _cfg.get("scene", {})
        self._table_color = scene_cfg.get("table_color", [1.0, 1.0, 1.0, 1.0])
        # Task-specific parameters live under `task:` per the prehensile
        # YAML convention; fall back to root for backwards compatibility.
        task_cfg = _cfg.get("task", {})
        self._box_pos = task_cfg.get(
            "fixture_pos", _cfg.get("fixture_pos", [0.0, 0.0]),
        )
        self._centroid_position_threshold = float(task_cfg.get("centroid_position_threshold", 0.005))
        self._orientation_threshold = np.deg2rad(float(task_cfg.get("orientation_threshold_deg", 5.0)))
        if not (np.isfinite(self._centroid_position_threshold) and self._centroid_position_threshold > 0
                and np.isfinite(self._orientation_threshold) and 0 < self._orientation_threshold < np.pi):
            raise ValueError("Packing pose thresholds must be positive and orientation must be below 180 degrees")

        table_cfg = scene_cfg.get("table", {})
        self._table_height = table_cfg.get("height", 0.9)
        _full_size = table_cfg.get("full_size")
        self._table_full_size = tuple(_full_size) if _full_size else (1.0, 0.75, 0.05)
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)

        _robot_offset = scene_cfg.get("robot_offset")
        self._robot_offset = [list(r) for r in _robot_offset] if _robot_offset else None

        cond_cfg = _cfg.get("task_conditions", {})
        success_cfg = cond_cfg.get("success", {})
        self._success_score_threshold = success_cfg.get("score_threshold", 1.0)
        failure_cfg = cond_cfg.get("failure", {})
        self._failure_floor_threshold = failure_cfg.get("floor_z_threshold", 0.05)

        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position = light_cfg.get("position", False)
        self._rand_light_color = light_cfg.get("color", False)

        # Private per-instance directory for all generated MJCF assets
        # (the bin + every L-piece). Never written into the package/repo's
        # own deps/extras/ directory -- see the module-level comment above
        # _write_l_piece_assets for why.
        self._asset_tmpdir = Path(tempfile.mkdtemp(prefix="imbench_packing_"))
        self._bin_asset_path = str(self._asset_tmpdir / f"{_BIN_ASSET}.xml")
        Path(self._bin_asset_path).write_text(_bin_xml())

        # Generate the initial L-pieces from the configured seed. The env
        # is rebuilt on each reset() with a fresh BPP split (and possibly
        # a different piece count), so this initial generation just decides
        # how many dynamic-object slots the first env build gets.
        self._gen_seed = kwargs.get("seed", 0)
        gen_rng = np.random.RandomState(self._gen_seed)
        n0 = self._sample_piece_count(gen_rng)
        self._pieces, self._block_asset_paths = self._generate_pieces(
            self._gen_seed, n0,
        )

        # Suction weld is contact-driven; reset cache between episodes.
        self._suction_weld = SuctionWeld()

        super().__init__(**kwargs)

    def _sample_piece_count(self, rng: np.random.RandomState) -> int:
        lo, hi = self._piece_count_range
        return int(rng.randint(lo, hi + 1))

    @staticmethod
    def _piece_bbox_half(piece: dict) -> tuple[float, float]:
        """Return (hx, hy) bounding-box half-extents around the piece centroid."""
        cx, cy = piece["centroid"]
        xs_lo = min(sx - lx / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        xs_hi = max(sx + lx / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        ys_lo = min(sy - ly / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        ys_hi = max(sy + ly / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        return max(abs(xs_lo - cx), abs(xs_hi - cx)), max(abs(ys_lo - cy), abs(ys_hi - cy))

    @staticmethod
    def _piece_bbox_directional(piece: dict) -> tuple[float, float, float, float]:
        """Return signed half-extents (h_left, h_right, h_down, h_up) around centroid.

        Unlike :meth:`_piece_bbox_half` (which returns symmetric ``max(abs(…))``),
        this gives the *directional* reach so the spawn logic can place a piece
        right against an obstacle on one side without leaving extra slack on
        the other.
        """
        cx, cy = piece["centroid"]
        xs_lo = min(sx - lx / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        xs_hi = max(sx + lx / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        ys_lo = min(sy - ly / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        ys_hi = max(sy + ly / 2 for (sx, sy, lx, ly) in piece["subboxes"])
        return cx - xs_lo, xs_hi - cx, cy - ys_lo, ys_hi - cy

    def _generate_pieces(self, seed: int, num_pieces: int):
        gen_rng = np.random.RandomState(seed)
        pieces = _generate_l_pieces(gen_rng, num_pieces)
        asset_paths = _write_l_piece_assets(pieces, self._asset_tmpdir)
        return pieces, asset_paths

    def _build_env(self):
        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        self._block_names = [f"lpiece_{k}" for k in range(len(self._block_asset_paths))]
        block_specs = [
            {"type": "extras", "name": name, "xml_path": path,
             "joints": "default"}
            for name, path in zip(self._block_names, self._block_asset_paths)
        ]
        box_spec = {"type": "extras", "name": "packing_box",
                    "xml_path": self._bin_asset_path, "joints": None}

        common_kwargs = dict(
            robots="Panda",
            gripper_types="PandaSuctionGripper",
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            has_renderer=(self.render_mode == "human"),
            has_offscreen_renderer=use_cameras,
            render_camera=None,
            use_camera_obs=False,  # BaseTask renders camera frames itself
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            horizon=self.MAX_EPISODE_STEPS,
        )

        # Rotate the arm 90° clockwise (around table centre) so its workspace
        # spans the table's long axis (X) — pieces scattered on both sides
        # of the bin along ±X are then within reach. Without this the arm
        # sits on the -X side and can only reach the -X column.
        if self._robot_offset is not None:
            robot_offset = self._robot_offset[0]
        else:
            import robosuite.utils.transform_utils as T
            from robosuite.models.robots import Panda
            default_xpos = np.array(
                Panda().base_xpos_offset["table"](self._table_full_size[0])
            )
            rot = np.array([0.0, 0.0, np.pi / 2])
            rotated_xpos = T.euler2mat(rot) @ default_xpos
            robot_offset = list(rotated_xpos) + list(rot)

        env = self._make_robosuite_env(
            "SingleArmTableEnv",
            objects=block_specs,
            static_objects=[box_spec],
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            robot_offset=robot_offset,
            placement_config={
                "x_range": [0.0, 0.0],
                "y_range": [0.0, 0.0],
                "z_offset": 0.01,
                "reference_pos": [0.0, 0.0, self._table_height],
                "ensure_valid_placement": False,
            },
            **common_kwargs,
        )

        # Cache MujocoObject handles for downstream qpos / scoring lookups.
        self._blocks = list(env._dynamic_objects)
        self._box = env.get_object("packing_box")
        return env

    def _get_block_positions(self):
        positions = {}
        for name, block in zip(self._block_names, self._blocks):
            bid = self._env.sim.model.body_name2id(block.root_body)
            positions[name] = np.array(self._env.sim.data.body_xpos[bid])
        return positions

    def _block_in_box(self, block_pos, box_pos):
        dx = abs(block_pos[0] - box_pos[0])
        dy = abs(block_pos[1] - box_pos[1])
        # Vertical containment: the piece's body origin (its bottom face) must
        # be within half a piece thickness of the bin floor, i.e. the piece
        # rests on the floor in a single layer. A piece stacked on another
        # sits a full thickness up and pokes out above the rim.
        floor_z = box_pos[2] + _BOX_BASE_THICK
        in_z = abs(block_pos[2] - floor_z) < _PIECE_THICKNESS / 2
        return dx < _BOX_INNER_HALF_X and dy < _BOX_INNER_HALF_Y and in_z

    def _block_is_flat(self, block):
        bid = self._env.sim.model.body_name2id(block.root_body)
        quat = self._env.sim.data.body_xquat[bid]
        w, x, y, z = quat
        up_z = 1 - 2 * (x * x + y * y)
        tilt_angle = np.arccos(np.clip(abs(up_z), 0, 1))
        return tilt_angle < np.deg2rad(5.0)

    def reset(self, *, seed=None, options=None):
        self._check_open()
        self._suction_weld.reset()

        # ── Per-episode regeneration ─────────────────────────────────
        # Re-run BPP with a new piece count and split seed, then rebuild the
        # robosuite env so the new geometry takes effect. Draws come from
        # ``self._rng`` (reseeded only on an explicit seed), so seeded resets
        # are reproducible and unseeded resets vary. Done BEFORE
        # super().reset() so the new env flows through the obs pipeline.
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        n_pieces = self._sample_piece_count(self._rng)
        split_seed = int(self._rng.randint(0, 2**31 - 1))
        self._pieces, self._block_asset_paths = self._generate_pieces(
            split_seed, n_pieces,
        )
        self._rebuild_env()

        obs, info = super().reset(seed=seed, options=options)

        sim = self._env.sim
        table_z = self._env.table_offset[2]

        # Suction may only weld to the pieces, never to the table, bin or
        # robot. Body ids change with every rebuild, so recompute each reset.
        piece_bodies = set().union(*(body_subtree_ids(sim, b.root_body) for b in self._blocks))
        self._suction_weld.excluded_body_ids = set(range(sim.model.nbody)) - piece_bodies

        # Place box fixture
        box_bid = self._env.sim.model.body_name2id(self._box.root_body)
        box_pos = np.array([
            self._box_pos[0], self._box_pos[1],
            table_z + abs(self._box.bottom_offset[2]) + 0.001,
        ])
        sim.model.body_pos[box_bid] = box_pos

        # Scatter L-pieces on the table around the bin. Clearance is
        # computed per-piece from its directional bounding box so the
        # piece's edge nearest the bin sits a controlled distance away
        # from the bin wall — no spawn-time intersection or visual
        # overlap, regardless of which way the centroid is biased.
        bin_half_x = _BOX_INNER_HALF_X + _BOX_WALL_THICK
        bin_half_y = _BOX_INNER_HALF_Y + _BOX_WALL_THICK
        gap = 0.06  # clearance between piece's near edge and bin wall

        self._piece_spawn_xy: dict[str, tuple] = {}

        for i, (block, piece) in enumerate(zip(self._blocks, self._pieces)):
            phx, phy = self._piece_bbox_half(piece)
            hl, hr, hd, hu = self._piece_bbox_directional(piece)
            if self._rand_block_placement:
                bx = self._rng.uniform(-self._place_x_range, self._place_x_range)
                by = self._rng.uniform(-self._place_y_range, self._place_y_range)
                while (abs(bx - self._box_pos[0]) < bin_half_x + phx + gap
                       and abs(by - self._box_pos[1]) < bin_half_y + phy + gap):
                    bx = self._rng.uniform(-self._place_x_range, self._place_x_range)
                    by = self._rng.uniform(-self._place_y_range, self._place_y_range)
            else:
                # Alternate +X / -X side; rows step along Y on each side.
                # Use *directional* extents so the piece's near edge (the
                # one facing the bin) sits exactly ``gap`` from the wall.
                side = 1 if (i % 2 == 0) else -1
                row = i // 2
                near_extent = hl if side == 1 else hr  # extent toward bin
                bx = self._box_pos[0] + side * (bin_half_x + gap + near_extent)
                # Stagger rows along Y; centre the row stack on the bin.
                by = self._box_pos[1] + (row - 0.5) * (2 * phy + gap)

            bz = table_z + abs(block.bottom_offset[2]) + 0.002
            block_pos = np.array([bx, by, bz])

            if self._rand_block_yaw:
                yaw = self._rng.uniform(0, 2 * np.pi)
                quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0])

            sim.data.set_joint_qpos(
                block.joints[0],
                np.concatenate([block_pos, quat]),
            )
            self._piece_spawn_xy[self._block_names[i]] = (bx, by)

        if self._rand_light_position or self._rand_light_color:
            for i in range(sim.model.nlight):
                if self._rand_light_position:
                    sim.model.light_pos[i] = [
                        self._rng.uniform(0.0, 2.0),
                        self._rng.uniform(-1.0, 2.0),
                        self._rng.uniform(1.0, 2.5),
                    ]
                if self._rand_light_color:
                    base = self._rng.uniform(0.75, 1.0)
                    sim.model.light_diffuse[i] = np.clip(
                        base + self._rng.uniform(-0.05, 0.05, size=3), 0.6, 1.0,
                    )

        self._block_fell_off = False

        # Per-piece bin-contact streaks (reset to zero whenever the piece
        # loses contact with the bin or is welded to suction). A piece
        # only counts as "settled in the bin" once its streak reaches
        # _PIECE_SETTLED_CONTACT_STEPS — the success check requires this
        # for every piece on top of the geometric in-box / flat checks.
        self._bin_contact_streak: dict[str, int] = {
            name: 0 for name in self._block_names
        }
        # Cache body ids used for contact filtering. Populated lazily in
        # step() the first time it runs (after env build).
        self._packing_box_body_id: int | None = None
        self._block_body_ids: dict[str, int] = {}

        # Per-piece staged reward: picked_and_moved -> placed_in_box_flat ->
        # settled_contact_streak. Rebuilt fresh every reset (piece names may
        # vary in count across episodes since piece_count is re-sampled).
        self._block_by_name = dict(zip(self._block_names, self._blocks))
        self._piece_by_name = dict(zip(self._block_names, self._pieces))
        self._piece_symmetry_rotations = {
            name: _piece_symmetries(piece) for name, piece in self._piece_by_name.items()
        }
        self._piece_ever_picked: dict[str, bool] = {
            name: False for name in self._block_names
        }
        self._staged_rewards: dict[str, StagedReward] = {
            name: self._make_piece_staged_reward(name)
            for name in self._block_names
        }
        self._stage_debug: dict[str, dict] = {}
        self._all_packed = False

        sim.forward()

        self._camera_cache.clear()
        raw_obs = self._get_sim_observations(force_update=True)
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._observe(raw_obs)
        info = {"task_score": 0.0, "is_success": False, "step": 0}
        return obs, info

    def step(self, action):
        action = self._validate_action(action)
        raw_obs, _rs_reward, _rs_done, info = self._env.step(action)
        # Drive suction-weld activation/release after the physics step so
        # contact data reflects the post-step state, and before scoring so
        # the score sees this tick's suction state rather than the last one.
        self._suction_weld.update(self._env, action)
        # Update per-piece bin-contact streaks. Done after the suction
        # update so the welded_body_id check reflects this tick's state.
        self._update_bin_contact_streaks()
        return self._finish_step(raw_obs, info)

    def _welded_body_ids(self) -> set[int]:
        """Body ids currently held by any suction weld this episode."""
        ids: set[int] = set()
        for handles in self._suction_weld._handles.values():
            if handles.welded_body_id is not None:
                ids.add(int(handles.welded_body_id))
        return ids

    def _update_bin_contact_streaks(self) -> None:
        """Increment each piece's bin-contact streak if it has at least
        one MuJoCo contact with the packing box AND is not held by
        suction; otherwise reset its streak to zero.
        """
        sim = self._env.sim

        # Lazy-cache body ids on the first call after a fresh env build.
        if self._packing_box_body_id is None:
            self._packing_box_body_id = sim.model.body_name2id(
                self._box.root_body)
            self._block_body_ids = {
                name: sim.model.body_name2id(block.root_body)
                for name, block in zip(self._block_names, self._blocks)
            }

        box_bid = self._packing_box_body_id
        welded = self._welded_body_ids()
        # Block-id → still-touching-bin? (default False, flipped True on
        # first matching contact this step)
        in_contact = {name: False for name in self._block_names}

        ncon = int(sim.data.ncon)
        for i in range(ncon):
            c = sim.data.contact[i]
            b1 = int(sim.model.geom_bodyid[int(c.geom1)])
            b2 = int(sim.model.geom_bodyid[int(c.geom2)])
            if b1 != box_bid and b2 != box_bid:
                continue
            other = b2 if b1 == box_bid else b1
            for name, bbid in self._block_body_ids.items():
                if bbid == other:
                    in_contact[name] = True
                    break

        for name, bbid in self._block_body_ids.items():
            if bbid in welded:
                self._piece_ever_picked[name] = True
            if in_contact[name] and bbid not in welded:
                self._bin_contact_streak[name] += 1
            else:
                self._bin_contact_streak[name] = 0

    # ------------------------------------------------------------------
    #  Solution preview
    # ------------------------------------------------------------------

    def solution_image(self, width: int = 480, height: int = 360) -> np.ndarray:
        """Render the goal layout (4 L-pieces tiling the bin) as an RGB image.

        Used by ``scripts/collect_demos.py`` to show the operator what the
        target configuration looks like, since the L-shapes are procedurally
        generated and the operator can't infer the layout otherwise.

        Returns
        -------
        np.ndarray
            ``(H, W, 3)`` uint8 RGB array suitable for cv2.imshow / writeText.
        """
        # We do everything with numpy + a tiny rasteriser so this works
        # headless and without a matplotlib import (matplotlib's heavy state
        # would clash with the inline dashboard's window manager).
        img = np.full((height, width, 3), 245, dtype=np.uint8)

        # World→pixel mapping. Fit the bin's outer footprint with 8 % padding.
        outer_x = _BOX_INNER_HALF_X + _BOX_WALL_THICK
        outer_y = _BOX_INNER_HALF_Y + _BOX_WALL_THICK
        pad = 0.08
        sx = (1 - 2 * pad) * width / (2 * outer_x)
        sy = (1 - 2 * pad) * height / (2 * outer_y)
        scale = min(sx, sy)
        cx_px, cy_px = width // 2, height // 2

        def w2p(x, y):
            # Flip y so +y is up in the image (visual convention).
            return (
                int(round(cx_px + x * scale)),
                int(round(cy_px - y * scale)),
            )

        def fill_rect(cx, cy, lx, ly, rgb):
            x0, y1 = w2p(cx - lx / 2, cy - ly / 2)
            x1, y0 = w2p(cx + lx / 2, cy + ly / 2)
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, width), min(y1, height)
            img[y0:y1, x0:x1] = rgb

        def stroke_rect(cx, cy, lx, ly, rgb, thickness=2):
            x0, y1 = w2p(cx - lx / 2, cy - ly / 2)
            x1, y0 = w2p(cx + lx / 2, cy + ly / 2)
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, width), min(y1, height)
            t = thickness
            img[y0:y0 + t, x0:x1] = rgb
            img[y1 - t:y1, x0:x1] = rgb
            img[y0:y1, x0:x0 + t] = rgb
            img[y0:y1, x1 - t:x1] = rgb

        # Bin walls (outer outline + inner outline).
        wood = np.array([141, 116, 77], dtype=np.uint8)  # warm wood
        stroke_rect(0, 0, 2 * outer_x, 2 * outer_y, wood, thickness=4)
        stroke_rect(0, 0, 2 * _BOX_INNER_HALF_X, 2 * _BOX_INNER_HALF_Y, wood,
                    thickness=2)

        # Each L-piece: fill its 2 sub-boxes, then a thin border around them.
        for piece in self._pieces:
            r, g, b, _ = piece["color"]
            rgb = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
            for (sxc, syc, lx, ly) in piece["subboxes"]:
                fill_rect(sxc, syc, lx, ly, rgb)
            for (sxc, syc, lx, ly) in piece["subboxes"]:
                stroke_rect(sxc, syc, lx, ly, (rgb // 2).astype(np.uint8),
                            thickness=1)
        return img

    def close(self) -> None:
        """Tear down the env and remove this instance's private asset tempdir.

        The primary cleanup path (deterministic, unlike ``__del__``); the
        ``__del__`` override below is only a best-effort backstop for
        callers that never explicitly close the task.
        """
        try:
            super().close()
        finally:
            self._cleanup_asset_tmpdir()

    def _cleanup_asset_tmpdir(self):
        tmpdir = getattr(self, "_asset_tmpdir", None)
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def __del__(self):
        # Best-effort backstop: garbage collection timing/ordering isn't
        # guaranteed, so callers should prefer close(). Swallow everything
        # -- __del__ running during interpreter teardown can't rely on
        # module globals (e.g. shutil) still being bound.
        try:
            self._cleanup_asset_tmpdir()
        except Exception:
            pass

    def _get_obs(self, raw_obs):
        """Obs with a layout fixed across piece counts.

        Every slot ``lpiece_0`` .. ``lpiece_{_MAX_PIECES - 1}`` is present;
        slots of pieces absent this episode are zero. ``object_state`` holds
        those slots' pos/quat in order, then the remaining object
        observables in robosuite order.
        """
        obs = self._build_obs_from_config(raw_obs)
        if not self.object_state_enabled:
            return obs

        pieces = {}
        for k in range(_MAX_PIECES):
            for key, dim in ((f"lpiece_{k}_pos", 3), (f"lpiece_{k}_quat", 4)):
                pieces[key] = np.asarray(raw_obs[key]) if key in raw_obs else np.zeros(dim)
        obs.update(pieces)

        if "object_state" in obs:
            rest = [
                np.atleast_1d(raw_obs[k]) for k, o in self._env._observables.items()
                if o.modality == "object" and o.is_enabled() and o.is_active()
                and k in raw_obs and k not in pieces
            ]
            obs["object_state"] = np.concatenate([*pieces.values(), *rest])
        return obs

    def _piece_moved_from_spawn(self, name):
        """XY distance (metres) of piece *name* from its spawn position."""
        spawn = self._piece_spawn_xy.get(name)
        if spawn is None:
            return 0.0
        pos = self._get_block_positions()[name]
        return float(np.hypot(pos[0] - spawn[0], pos[1] - spawn[1]))

    def _piece_picked_and_moved_progress(self, name):
        picked = self._piece_ever_picked.get(name, False)
        moved = self._piece_moved_from_spawn(name) > _PIECE_MOVED_THRESHOLD
        return 1.0 if (picked and moved) else 0.0

    def _piece_centroid(self, name):
        """Actual volume centroid; body origin is the bottom of the piece."""
        sim = self._env.sim
        bid = sim.model.body_name2id(self._block_by_name[name].root_body)
        rotation = sim.data.body_xmat[bid].reshape(3, 3)
        return sim.data.body_xpos[bid] + rotation @ np.array([0, 0, _PIECE_THICKNESS / 2])

    def _piece_pose_errors(self, name):
        """3D centroid distance and SO(3) angle to the nearest shape symmetry."""
        sim = self._env.sim
        box_bid = sim.model.body_name2id(self._box.root_body)
        box_rotation = sim.data.body_xmat[box_bid].reshape(3, 3)
        cx, cy = self._piece_by_name[name]["centroid"]
        target = sim.data.body_xpos[box_bid] + box_rotation @ np.array([
            cx, cy, _BOX_BASE_THICK + _PIECE_THICKNESS / 2])
        distance = float(np.linalg.norm(self._piece_centroid(name) - target))
        bid = sim.model.body_name2id(self._block_by_name[name].root_body)
        relative = box_rotation.T @ sim.data.body_xmat[bid].reshape(3, 3)
        traces = np.einsum("ij,kij->k", relative, self._piece_symmetry_rotations[name])
        angle = float(np.min(np.arccos(np.clip((traces-1)/2, -1, 1))))
        return distance, angle

    def _piece_at_target(self, name):
        distance, angle = self._piece_pose_errors(name)
        return distance <= self._centroid_position_threshold and angle <= self._orientation_threshold

    def _piece_in_box_flat(self, name):
        box_bid = self._env.sim.model.body_name2id(self._box.root_body)
        box_pos = self._env.sim.data.body_xpos[box_bid]
        # Use the geometric bottom-plane height, including equivalent
        # upside-down poses, rather than the body's arbitrary origin.
        pos = self._piece_centroid(name) - np.array([0, 0, _PIECE_THICKNESS / 2])
        block = self._block_by_name[name]
        return bool(self._block_in_box(pos, box_pos) and self._block_is_flat(block))

    def _piece_placed_flat_progress(self, name):
        return float(self._piece_in_box_flat(name) and self._piece_at_target(name))

    def _piece_settled_progress(self, name):
        return min(1.0, self._bin_contact_streak.get(name, 0)
                   / _PIECE_SETTLED_CONTACT_STEPS)

    def _piece_fully_packed(self, name):
        """Each piece must match its own target AND be flat, in-bin and released/settled."""
        return bool(
            self._piece_in_box_flat(name)
            and self._piece_at_target(name)
            and self._bin_contact_streak.get(name, 0)
                >= _PIECE_SETTLED_CONTACT_STEPS
        )

    def _make_piece_staged_reward(self, name):
        """Build the 3-stage StagedReward for one piece (see
        notes/staged_rewards_design.md §3.1 P7 / ib-p7-packing-v1): picked
        and carried off its spawn spot, placed flat at its target pose, then
        held settled contact with the bin long enough to count as packed.
        The two placement stages track the current state, so a piece taken
        back out of the bin loses their credit.
        """
        return StagedReward(
            stages=[
                Stage("picked_and_moved", 0.20,
                      lambda t, o, _n=name: t._piece_picked_and_moved_progress(_n)),
                Stage("placed_in_box_flat", 0.35,
                      lambda t, o, _n=name: t._piece_placed_flat_progress(_n),
                      sticky=False),
                Stage("settled_contact_streak", 0.45,
                      lambda t, o, _n=name: t._piece_settled_progress(_n),
                      sticky=False),
            ],
            success_fn=lambda t, o, _n=name: t._piece_fully_packed(_n),
        )

    def _compute_task_score(self, raw_obs):
        # Per-piece staged score, averaged across all pieces. Each piece's
        # own StagedReward saturates to 1.0 once it is fully packed (at its
        # own target pose, in the bin, flat, and settled), so the
        # overall average only reaches 1.0 once every piece is packed —
        # requiring every target pose simultaneously — while
        # giving partial credit for picking/placing progress along the way.
        if not self._block_names:
            return 0.0
        debug = {}
        total = 0.0
        for name in self._block_names:
            score, stage_values = self._staged_rewards[name].compute(self, raw_obs)
            debug[name] = stage_values
            total += score
        self._stage_debug = debug
        self._all_packed = all(v["_success"] for v in debug.values())
        return total / len(self._block_names)

    def _is_success(self, task_score):
        """Every piece is fully packed right now."""
        return self._all_packed

    def _check_termination(self, raw_obs):
        self._compute_task_score(raw_obs)
        if self._all_packed:
            return True
        # Failure: any piece dropped off the table.  We trigger as soon as
        # the piece centre is below the table top by more than a small
        # margin (it can't recover from there) instead of waiting for it
        # to hit the floor — fail fast, save the operator's time.
        table_z = self._env.table_offset[2]
        drop_margin = 0.05  # metres below table top
        positions = self._get_block_positions()
        for pos in positions.values():
            if pos[2] < table_z - drop_margin or pos[2] < self._failure_floor_threshold:
                self._block_fell_off = True
                return True
        return False
