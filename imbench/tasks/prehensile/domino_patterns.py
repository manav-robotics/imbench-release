"""Domino chain pattern generators.

Each generator produces positions starting at (0, 0), proceeding
generally in the +x direction.  The caller may rotate / translate
the result to place the chain anywhere on the table.

Every function returns ``[(x, y, heading), ...]`` where *heading*
(radians, CCW from +x) is the cascade direction for each domino.
"""

import numpy as np
from typing import List, Tuple

DominoLayout = List[Tuple[float, float, float]]

ALL_PATTERNS = [
    "line", "arc", "spiral", "zigzag",
    "l_shape", "s_curve", "polygon",
]

#: Largest heading change (rad) between neighbouring dominoes of an arc.
ARC_MAX_TURN_PER_LINK = np.radians(20.0)


# ── Helpers ─────────────────────────────────────────────────────────

def heading_to_quat(heading: float) -> List[float]:
    """Heading angle → MuJoCo *wxyz* quaternion (z-rotation only)."""
    return [float(np.cos(heading / 2)), 0.0, 0.0, float(np.sin(heading / 2))]


def _compute_headings(points):
    headings = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        headings.append(float(np.arctan2(dy, dx)))
    headings.append(headings[-1] if headings else 0.0)
    return headings


def _attach(points):
    headings = _compute_headings(points)
    return [(x, y, h) for (x, y), h in zip(points, headings)]


# ── Pattern generators ──────────────────────────────────────────────

def _line(n, spacing, rng):
    """Straight line along +x."""
    return _attach([(i * spacing, 0.0) for i in range(n)])


def _arc(n, spacing, rng):
    """Smooth arc — portion of a circle, curving up or down.

    The heading turns by at most ``ARC_MAX_TURN_PER_LINK`` between
    neighbouring dominoes; a sharper turn jams the cascade.
    """
    if n < 2:
        return _line(n, spacing, rng)
    max_angle = min(3 * np.pi / 4, (n - 1) * ARC_MAX_TURN_PER_LINK)
    total_angle = rng.uniform(min(np.pi / 4, max_angle), max_angle)
    arc_length = (n - 1) * spacing
    radius = arc_length / total_angle
    curves_up = bool(rng.randint(0, 2))

    if curves_up:
        cx, cy = 0.0, radius
        start = -np.pi / 2
        da = total_angle / (n - 1)
    else:
        cx, cy = 0.0, -radius
        start = np.pi / 2
        da = -total_angle / (n - 1)

    points = [
        (cx + radius * np.cos(start + i * da),
         cy + radius * np.sin(start + i * da))
        for i in range(n)
    ]
    return _attach(points)


def _spiral(n, spacing, rng):
    """Euler-spiral (clothoid): curvature increases with arc length."""
    if n < 3:
        return _line(n, spacing, rng)
    total_turn = rng.uniform(np.pi / 2, 2 * np.pi)
    sign = rng.choice([-1, 1])
    denom = n * (n - 1) / 2
    points = [(0.0, 0.0)]
    heading = 0.0
    for i in range(1, n):
        heading += sign * total_turn * i / denom
        x = points[-1][0] + spacing * np.cos(heading)
        y = points[-1][1] + spacing * np.sin(heading)
        points.append((x, y))
    return _attach(points)


def _zigzag(n, spacing, rng):
    """Zigzag — alternating lateral offsets from a central line."""
    if n < 3:
        return _line(n, spacing, rng)
    # Cap amplitude so that max |dy| between consecutive points (= 2*amp)
    # stays below spacing, preserving correct cascade distance.
    amplitude = spacing * rng.uniform(0.15, 0.45)
    points = [(0.0, 0.0)]
    for i in range(1, n):
        target_y = (1 if i % 2 else -1) * amplitude
        dy = target_y - points[-1][1]
        dx = np.sqrt(max(spacing ** 2 - dy ** 2, 0.01 * spacing ** 2))
        points.append((points[-1][0] + dx, target_y))
    return _attach(points)


def _l_shape(n, spacing, rng):
    """Two perpendicular line segments (L-shape)."""
    if n < 3:
        return _line(n, spacing, rng)
    split = rng.randint(max(1, n // 3), min(n - 1, 2 * n // 3) + 1)
    turn = rng.choice([-1, 1]) * np.pi / 2
    points = [(i * spacing, 0.0) for i in range(split)]
    lx, ly = points[-1]
    for j in range(1, n - split + 1):
        points.append(
            (lx + j * spacing * np.cos(turn),
             ly + j * spacing * np.sin(turn))
        )
    return _attach(points)


def _s_curve(n, spacing, rng):
    """Smooth S-curve (sinusoidal path)."""
    if n < 4:
        return _line(n, spacing, rng)
    amplitude = spacing * rng.uniform(0.5, 1.5)
    total_length = (n - 1) * spacing
    N = 500
    raw = [
        (total_length * k / N, amplitude * np.sin(2 * np.pi * k / N))
        for k in range(N + 1)
    ]
    cum = [0.0]
    for k in range(1, N + 1):
        dx = raw[k][0] - raw[k - 1][0]
        dy = raw[k][1] - raw[k - 1][1]
        cum.append(cum[-1] + np.sqrt(dx * dx + dy * dy))
    cum = np.asarray(cum)
    points = [raw[0]]
    for i in range(1, n):
        idx = min(int(np.searchsorted(cum, i * spacing)), N)
        points.append(raw[idx])
    return _attach(points)


def _polygon(n, spacing, rng):
    """Partial regular polygon (square / pentagon / hexagon edges)."""
    if n < 3:
        return _line(n, spacing, rng)
    sides = int(rng.choice([4, 5, 6]))
    turn_dir = rng.choice([-1, 1])
    ext = turn_dir * 2 * np.pi / sides
    per_side = max(1, n // sides)
    rem = n - per_side * sides

    points = [(0.0, 0.0)]
    heading = 0.0
    count = 1
    for s in range(sides):
        seg_n = per_side + (1 if s < rem else 0)
        for _ in range(seg_n):
            if count >= n:
                break
            points.append(
                (points[-1][0] + spacing * np.cos(heading),
                 points[-1][1] + spacing * np.sin(heading))
            )
            count += 1
        heading += ext
        if count >= n:
            break
    return _attach(points)


# ── Public API ──────────────────────────────────────────────────────

_REGISTRY = {
    "line": _line,
    "arc": _arc,
    "spiral": _spiral,
    "zigzag": _zigzag,
    "l_shape": _l_shape,
    "s_curve": _s_curve,
    "polygon": _polygon,
}


def generate_pattern(name: str, n: int, spacing: float, rng) -> DominoLayout:
    """Return ``[(x, y, heading), ...]`` for the requested pattern."""
    fn = _REGISTRY.get(name, _line)
    return fn(n, spacing, rng)


def layout_has_collisions(
    layout: DominoLayout,
    footprint_half_extents: Tuple[float, float],
) -> bool:
    """Check if any non-consecutive dominos overlap.

    Uses the circumscribed circle of the oriented domino footprint as a
    conservative collision check.
    """
    hx, hy = footprint_half_extents
    min_dist = 2 * np.sqrt(hx ** 2 + hy ** 2)
    for i in range(len(layout)):
        xi, yi, _ = layout[i]
        for j in range(i + 2, len(layout)):
            xj, yj, _ = layout[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 < min_dist ** 2:
                return True
    return False


def generate_safe_pattern(
    name: str,
    n: int,
    spacing: float,
    rng,
    footprint_half_extents: Tuple[float, float],
    max_retries: int = 3,
) -> DominoLayout:
    """Generate a pattern, falling back to 'line' if collisions exist."""
    for _attempt in range(max_retries):
        layout = generate_pattern(name, n, spacing, rng)
        if not layout_has_collisions(layout, footprint_half_extents):
            return layout
    # Guaranteed collision-free
    return generate_pattern("line", n, spacing, rng)


def rotate_layout(layout: DominoLayout, angle: float) -> DominoLayout:
    """Rotate an entire layout around the origin."""
    c, s = np.cos(angle), np.sin(angle)
    return [(c * x - s * y, s * x + c * y, h + angle) for x, y, h in layout]


def translate_layout(layout: DominoLayout, dx: float, dy: float) -> DominoLayout:
    """Shift an entire layout."""
    return [(x + dx, y + dy, h) for x, y, h in layout]
