"""Randomized Domino select layouts: one true route to red among look-alike decoys.

Each pattern is drawn in a local frame with the red target at the origin,
falling toward +x, and the route's last piece one spacing behind it. A true
route turns at most ``MAX_TURN`` per link, the sharpest turn measured to
propagate at every piece size. A decoy either stops at a break measured to
stop at every size (a gap of 16 cm or more, three missing pieces, three low
blocks, two pieces twisted 60-80 degrees, a sharp corner, a sideways jog of
7.5 cm or more) or can never reach red (it meets another chain side-on, or
ends beside red or short of it). A decoy aimed at red has its break as its
own last pieces, so nothing full-height past the break points at red. The
layout is then mirrored, rotated to any heading and shifted onto the table;
:func:`check` rejects layouts whose route would brush a decoy, or where any
decoy piece, pushed either way, could reach red or drive the route (a second
solution), and the generator draws again.
"""

import numpy as np

MAX_TURN = np.deg2rad(18.0)
BREAKS = ("gap", "missing", "low", "twist", "sharp", "jog")
SHAPES = ("zigzag", "arc", "s_curve", "line")
SHAPE_WEIGHTS = (0.4, 0.2, 0.2, 0.2)
#: World-frame bounds for piece centres: clear of the robot base at x = -0.56
#: and of the table edges (half size 0.4).
REGION = np.array([[-0.31, 0.32], [-0.33, 0.33]])
TABLE_HALF = 0.38
#: Every piece stays within REACH of the robot base. The Panda's grip site
#: reaches 0.84 m at pushing height (table + 3 cm); the margin leaves room to
#: get a fingertip behind any piece, decoys included, so reach gives no hint.
ROBOT_BASE_XY = np.array([-0.56, 0.0])
REACH = 0.72
#: Chain lengths drawn at random are scaled by this much (floors keep each
#: pattern's structure).
LENGTH_SCALE = 0.75
#: Line-orientation difference below which a falling piece drives the piece
#: it lands on along its own line; side-on hits (60 degrees or more) stop.
SIDE_ON = np.deg2rad(55.0)
LOW_HEIGHT_FRACTION = 0.23
#: Link after a sharp corner, in spacings: the turned piece's ends clear the
#: piece two back.
SHARP_LINK = 1.2
#: Least gap (m) between the footprints of neighbouring pieces in a chain. A
#: twisted piece lies across its line, so the links on each side of it grow
#: until both neighbours clear it by this much.
NEIGHBOUR_CLEAR = 0.003


def _deg(rng, lo, hi):
    return np.deg2rad(rng.uniform(lo, hi))


def _len(rng, lo, hi, floor=1):
    """Chain length drawn like ``rng.randint(lo, hi)``, scaled by LENGTH_SCALE and kept at or above *floor*."""
    a = max(floor, int(round(lo * LENGTH_SCALE)))
    b = max(a, int(round((hi - 1) * LENGTH_SCALE)))
    return int(rng.randint(a, b + 1))


def _sign(rng):
    return float(rng.choice([-1, 1]))


def _wrap(h):
    return float((h + np.pi) % (2 * np.pi) - np.pi)


def _line_diff(a, b):
    d = abs(_wrap(a - b))
    return min(d, np.pi - d)


class Geometry:
    """Spacing and largest piece/red dimensions (full extents, metres)."""

    def __init__(self, spacing, piece_half_max, red_half):
        self.S = float(spacing)
        self.H, self.W, self.T = (2 * float(v) for v in piece_half_max)
        self.red_T, self.red_W, self.red_H = (2 * float(v) for v in red_half)


class Chain:
    """Forward-ordered poses ``(x, y, heading)``; heading is the fall direction.

    ``stop`` is the index of the first piece a wave started at ``poses[0]``
    cannot topple (``None``: every piece can fall). ``both_ways`` marks a
    chain that may be driven from its far end.
    """

    def __init__(self, poses, low=(), defect="none", stop=None, both_ways=False):
        self.poses = [(float(x), float(y), _wrap(h)) for x, y, h in poses]
        self.low = set(low)
        self.defect = defect
        self.stop = stop
        self.both_ways = both_ways

    def live(self):
        return range(len(self.poses) if self.stop is None else self.stop)


class Layout:
    def __init__(self, route, decoys, red=(0.0, 0.0, 0.0), yaws=None, clearance=0.008, allow=()):
        self.route, self.decoys, self.red = route, list(decoys), red
        self.pattern = ""
        self.yaws = yaws
        self.clearance = clearance
        self.allow = {frozenset(p) for p in allow}

    @property
    def chains(self):
        return [self.route] + self.decoys

    def count(self):
        return sum(len(c.poses) for c in self.chains)

    def transformed(self, mirror, yaw, shift):
        c, s = np.cos(yaw), np.sin(yaw)

        def tf(p):
            x, y, h = p[0], mirror * p[1], mirror * p[2]
            return (c * x - s * y + shift[0], s * x + c * y + shift[1], h + yaw)

        out = Layout(None, [], tf(self.red), clearance=self.clearance)
        out.allow, out.pattern = self.allow, self.pattern
        chains = [Chain([tf(p) for p in ch.poses], ch.low, ch.defect, ch.stop, ch.both_ways)
                  for ch in self.chains]
        out.route, out.decoys = chains[0], chains[1:]
        return out


class Pen:
    """Lays pieces link by link from a start pose.

    ``dirs[i]`` is the travel heading of the link that placed piece ``i``.
    :meth:`turn` spreads a turn over the fewest links that keep each within
    ``MAX_TURN``; :meth:`snap` turns on the spot, which is a break.
    """

    def __init__(self, g, x, y, h):
        self.g = g
        self.at = np.array([x, y], float)
        self.h = float(h)
        self.xy, self.dirs = [self.at.copy()], [self.h]
        self.low, self.twist = set(), {}

    @property
    def n(self):
        return len(self.xy)

    def _unit(self):
        return np.array([np.cos(self.h), np.sin(self.h)])

    def _clear_link(self, link, d, side):
        """*link*, lengthened until the next piece clears a twisted neighbour (or is cleared, if it is twisted)."""
        prev, new = self.n - 1, self.n
        if not (self.twist.get(prev) or self.twist.get(new)):
            return link
        before = footprint(self.g, (*self.xy[prev], self.dirs[prev] + self.twist.get(prev, 0.0)))
        while True:
            at = self.at + link * d + side
            if not _overlap(before, footprint(self.g, (*at, self.h + self.twist.get(new, 0.0))), NEIGHBOUR_CLEAR):
                return link
            link += 0.002

    def go(self, n=1, link=None, lateral=0.0):
        for _ in range(int(n)):
            d = self._unit()
            side = lateral * np.array([-d[1], d[0]])
            self.at = self.at + self._clear_link(self.g.S if link is None else link, d, side) * d + side
            self.xy.append(self.at.copy())
            self.dirs.append(self.h)
            link, lateral = None, 0.0
        return self

    def skip(self, n):
        """Advance *n* links without placing pieces."""
        self.at = self.at + n * self.g.S * self._unit()
        return self

    def turn(self, angle, links=None):
        k = links or max(1, int(np.ceil(abs(angle) / MAX_TURN - 1e-9)))
        for _ in range(k):
            self.h += angle / k
            self.go()
        return self

    def snap(self, angle):
        self.h += angle
        return self

    def chain(self, backward, defect="none", stop=None, both_ways=False):
        """Forward-ordered chain; a backward pen walked from the chain's end.

        *stop* is a pen index; it is converted to the forward order.
        """
        n = self.n
        order = list(range(n - 1, -1, -1)) if backward else list(range(n))
        if backward:
            heads = [self.dirs[i] + np.pi for i in order]
        else:
            heads = [self.dirs[i + 1] if i + 1 < n else self.dirs[i] for i in order]
        index = {old: new for new, old in enumerate(order)}
        poses = [(self.xy[i][0], self.xy[i][1], h + self.twist.get(i, 0.0)) for i, h in zip(order, heads)]
        return Chain(poses, {index[i] for i in self.low if i < n}, defect,
                     None if stop is None else index[stop], both_ways)


# ── Shapes and breaks ───────────────────────────────────────────────

def shape(rng, n, kind=None, zig_max=45, legs=None):
    """Walk about *n* links of *kind* (random by weight); zig-zags swing up to *zig_max* degrees each way.

    With *legs*, a zig-zag ignores *n* and lays exactly that many straight
    legs, alternating sides, joined by rounded corners.
    """
    kind = kind or str(rng.choice(SHAPES, p=SHAPE_WEIGHTS))
    sign = _sign(rng)
    if kind == "s_curve" and n < 8:
        kind = "arc"
    if kind == "line":
        return lambda pen: pen.go(n)
    if kind == "arc":
        total = sign * _deg(rng, 25, min(100, 17 * n))

        def arc(pen):
            for _ in range(n):
                pen.h += total / n
                pen.go()
        return arc
    if kind == "s_curve":
        amp = _deg(rng, 22, min(50, 17 * n / (2 * np.pi)))

        def s_curve(pen):
            base = pen.h
            for i in range(1, n + 1):
                pen.h = base + sign * amp * np.sin(2 * np.pi * i / n)
                pen.go()
        return s_curve
    a, run = _deg(rng, 25 if legs is None else 30, zig_max), int(rng.randint(1, 3)) if legs is None else 1

    def zigzag(pen):
        base, s, start, laid = pen.h, sign, pen.n, 1
        pen.turn(s * a).go(run)
        while (pen.n - start < n) if legs is None else (laid < legs):
            s = -s
            pen.turn(base + s * a - pen.h).go(run)
            laid += 1
        pen.turn(base - pen.h)
    return zigzag


def gentle(rng, n, sign=None):
    """A straight run or a shallow arc (15-40 degrees in total, turning *sign*), for lanes side by side."""
    if rng.rand() < 0.5:
        return _go(n)
    # At most MAX_TURN per link, so a short arc is shallower.
    total = (sign or _sign(rng)) * min(_deg(rng, 15, 40), n * MAX_TURN)

    def arc(pen):
        for _ in range(n):
            pen.h += total / n
            pen.go()
    return arc


def add_break(pen, rng, kind):
    """Insert break *kind* at a backward pen; returns the pen index of the first piece the wave cannot topple.

    Walking backward, that is the piece placed last before the break, except
    for a twist, where it is the first twisted piece the wave meets.
    """
    stop = pen.n - 1
    if kind == "gap":
        pen.go(link=rng.uniform(0.16, 0.19))
        return stop
    if kind == "missing":
        pen.skip(3).go()
        return stop
    if kind == "low":
        # Three low blocks: a full-height piece can tumble over a pair.
        pen.low.update((pen.n, pen.n + 1, pen.n + 2))
        pen.go(3)
        return stop
    if kind == "twist":
        # Past 80 degrees the leaning wave can shove the piece over along its
        # width; a single twisted piece can itself be shoved into the next, so
        # two in a row.
        for _ in range(2):
            pen.twist[pen.n] = _sign(rng) * _deg(rng, 60, 80)
            pen.go()
        return pen.n - 1
    if kind == "sharp":
        pen.snap(_sign(rng) * _deg(rng, 75, 95)).go(link=SHARP_LINK * pen.g.S)
        return stop
    if kind == "jog":
        pen.go(lateral=_sign(rng) * rng.uniform(0.075, 0.09))
        return stop
    raise ValueError(kind)


END_BREAKS = ("low", "twist")


def add_end_break(pen, rng, kind):
    """Make the first pieces of a backward pen (the decoy's end) the break; returns the stop index.

    Nothing full-height then stands past the break, so no piece of the decoy
    can be pushed on toward red from beyond it.
    """
    if kind == "low":
        pen.low.update((0, 1, 2))
        pen.go(2)
        return 2
    if kind == "twist":
        for i in (0, 1):
            pen.twist[i] = _sign(rng) * _deg(rng, 60, 80)
        pen.go(1)
        return 1
    raise ValueError(kind)


def decoy_into(g, rng, end, n=None, brk=None, body=None, tail=2, both_ways=False, at_end=False):
    """Decoy walked backward from its end pose ``(x, y, heading)``: a straight *tail*, the break, then *body*.

    With *at_end*, the break is the decoy's last pieces instead (no tail), for
    decoys aimed at red.
    """
    pen = Pen(g, end[0], end[1], end[2] + np.pi)
    if brk and at_end:
        stop = add_end_break(pen, rng, brk)
    else:
        # At least three pieces past a break, so the end still looks connected.
        pen.go((max(tail, 3) if brk else tail) - 1)
        stop = add_break(pen, rng, brk) if brk else None
    (body or shape(rng, _len(rng, 3, 6, 2) if n is None else n))(pen)
    return pen.chain(True, brk or "none", stop, both_ways)


def route_into_red(g, rng, program, lead=2):
    """True route walked backward from one spacing behind red: *lead* straight pieces, then *program*."""
    pen = Pen(g, -g.S, 0.0, np.pi)
    pen.go(lead - 1)
    program(pen)
    return pen.chain(True)


def straight_spots(chain, lo, hi):
    """Indices in [lo, hi) whose neighbours share their heading, where a side-on join has room."""
    h = [p[2] for p in chain.poses]
    return [i for i in range(max(lo, 1), min(hi, len(h) - 1))
            if _line_diff(h[i - 1], h[i]) < 0.05 and _line_diff(h[i], h[i + 1]) < 0.05]


def _seq(*steps):
    def run(pen):
        for step in steps:
            step(pen)
    return run


def _go(n):
    return lambda pen: pen.go(n)


def _turn(angle):
    return lambda pen: pen.turn(angle)


# ── Decoy anchors (end poses) ───────────────────────────────────────

def aim_at_red(rng, side=None, front=False, r=None):
    """End pose pointing at red: behind it on a diagonal, or in front beyond its fall."""
    # 17 cm or more: a piece knocked loose past the break still cannot reach red.
    if front:
        phi = _sign(rng) * _deg(rng, 0, 30)
        r = r or rng.uniform(0.19, 0.22)
    else:
        phi = np.pi + (side or _sign(rng)) * _deg(rng, 40, 60)
        r = r or rng.uniform(0.17, 0.20)
    return (r * np.cos(phi), r * np.sin(phi), phi + np.pi)


def beside_red(rng, side):
    """End pose of a lane that passes 8-9.5 cm beside red (a near miss)."""
    return (rng.uniform(-0.03, 0.03), side * rng.uniform(0.08, 0.095), _deg(rng, -6, 6))


def side_on(g, rng, pose, side):
    """End pose of a chain that meets *pose*'s piece side-on (a T join)."""
    x, y, h = pose
    v = np.array([-np.sin(h), np.cos(h)]) * side
    d = g.W / 2 + g.T / 2 + rng.uniform(0.015, 0.022)
    return (x + v[0] * d, y + v[1] * d, float(np.arctan2(-v[1], -v[0])) + _deg(rng, -8, 8))


def _aimed(g, rng, end, brk=None, body=None, n=None):
    """Decoy aimed at red whose break is its own last pieces (low or twisted)."""
    return decoy_into(g, rng, end, n=n, brk=brk or str(rng.choice(END_BREAKS)), body=body, at_end=True)


def _pick_break(rng, kinds=BREAKS):
    return str(rng.choice(kinds))


# ── Patterns ────────────────────────────────────────────────────────

def fan_in(g, rng):
    route = route_into_red(g, rng, shape(rng, _len(rng, 5, 9, 3)), lead=2)
    side = _sign(rng)
    anchors = [aim_at_red(rng, side), aim_at_red(rng, -side), aim_at_red(rng, front=True)]
    keep = anchors[:int(rng.randint(2, 4))]
    return Layout(route, [_aimed(g, rng, a) for a in keep])


def near_miss_lanes(g, rng):
    route = route_into_red(g, rng, gentle(rng, _len(rng, 3, 6, 2)), lead=3)
    # Walking back from red, turning by -side bends a lane away from the route.
    decoys = [decoy_into(g, rng, beside_red(rng, side), tail=3, body=gentle(rng, _len(rng, 2, 5, 2), -side))
              for side in (-1, 1)]
    if rng.rand() < 0.5:
        decoys[int(rng.randint(2))] = decoy_into(g, rng, aim_at_red(rng, front=True, r=rng.uniform(0.19, 0.22)),
                                                 body=_go(_len(rng, 3, 5, 2)))
    return Layout(route, decoys)


def collision_course(g, rng):
    route = route_into_red(g, rng, shape(rng, _len(rng, 5, 8, 3)), lead=2)
    side = _sign(rng)
    p = np.array([rng.uniform(-0.06, 0.02), side * rng.uniform(0.13, 0.17)])
    u = _deg(rng, -20, 20)
    d = np.array([np.cos(u), np.sin(u)]) * 0.03
    a = decoy_into(g, rng, (*(p - d), u), n=_len(rng, 3, 6, 2), both_ways=True)
    b = decoy_into(g, rng, (*(p + d), u + np.pi), n=_len(rng, 3, 6, 2), both_ways=True)
    return Layout(route, [a, b], allow=[(1, 2)])


def edge_on_finish(g, rng):
    side = _sign(rng)
    route = route_into_red(g, rng, _seq(_turn(side * _deg(rng, 60, 100)), shape(rng, _len(rng, 2, 5, 2))), lead=2)
    end = aim_at_red(rng, front=True) if rng.rand() < 0.5 else aim_at_red(rng, -side)
    pen = Pen(g, end[0], end[1], end[2] + np.pi)
    last = int(rng.randint(2))
    # Twist set before the piece is laid, so its links clear its neighbours.
    pen.twist[last] = _sign(rng) * _deg(rng, 60, 80)
    pen.go(last)
    stop = pen.n - 1
    shape(rng, _len(rng, 3, 6, 2), "line" if rng.rand() < 0.5 else None)(pen)
    return Layout(route, [pen.chain(True, "twist", stop)])


def orbit(g, rng):
    route = route_into_red(g, rng, shape(rng, _len(rng, 3, 7, 2)), lead=3)
    decoys = []
    for side in ([-1, 1] if rng.rand() < 0.5 else [_sign(rng)]):
        radius = rng.uniform(0.15, 0.18)
        t0, t1 = side * _deg(rng, 125, 145), side * _deg(rng, 35, 55)
        pen = Pen(g, radius * np.cos(t0), radius * np.sin(t0), t0 - side * np.pi / 2)
        step = g.S / radius
        for _ in range(int(abs(t0 - t1) / step)):
            pen.h -= side * step
            pen.go()
        pen.go(int(rng.randint(1, 3)))
        decoys.append(pen.chain(False))
    return Layout(route, decoys)


def _entrance(g, rng, pose):
    """Straight chain in line behind *pose*'s piece, 26-30 cm back, so it looks like the way in."""
    x, y, h = pose
    back = rng.uniform(0.26, 0.30)
    end = (x - back * np.cos(h), y - back * np.sin(h), h)
    return decoy_into(g, rng, end, body=_go(int(rng.randint(0, 2))))


def double_merge(g, rng):
    first, middle = int(rng.randint(2, 4)), 2
    t1, t2 = _sign(rng) * _deg(rng, 50, 75), _sign(rng) * _deg(rng, 50, 75)
    route = route_into_red(g, rng, _seq(_turn(t1), _go(middle), _turn(t2), _go(int(rng.randint(2, 4)))),
                           lead=first)
    n = len(route.poses)
    k1 = int(np.ceil(abs(t1) / MAX_TURN - 1e-9))
    j1 = n - first                 # first piece of the final straight
    j2 = j1 - k1 - middle          # first piece of the middle straight
    decoys = [_entrance(g, rng, route.poses[j1]), _entrance(g, rng, route.poses[j2])]
    if rng.rand() < 0.5:
        decoys.append(_aimed(g, rng, aim_at_red(rng)))
    return Layout(route, decoys)


def side_joins(g, rng):
    trunk = _len(rng, 6, 9, 5)
    route = route_into_red(g, rng, _seq(_turn(_sign(rng) * _deg(rng, 60, 100)), shape(rng, _len(rng, 2, 4, 2))),
                           lead=trunk)
    n = len(route.poses)
    # Feeders join at least two pieces apart, on alternating sides, and run
    # three pieces straight out, so the layout stays compact enough to reach.
    spots = np.arange(n - trunk + 1, n - 1)[::2]
    spots = sorted(rng.choice(spots, size=min(len(spots), int(rng.randint(2, 4))), replace=False))
    side = _sign(rng)
    decoys = []
    for j in spots:
        decoys.append(decoy_into(g, rng, side_on(g, rng, route.poses[j], side), body=_go(1)))
        side = -side
    return Layout(route, decoys)


def comb(g, rng):
    # Not shortened: a shorter spine puts the stubs next to red, where a decoy can reach it.
    spine, stub, side = int(rng.randint(6, 9)), int(rng.randint(2, 4)), _sign(rng)
    route = route_into_red(g, rng, _seq(_turn(side * _deg(rng, 80, 100)), _go(stub)), lead=spine)
    n = len(route.poses)
    first_spine = n - spine
    spots = list(range(first_spine + 2, n - 1, 2))
    rng.shuffle(spots)
    decoys = [decoy_into(g, rng, side_on(g, rng, route.poses[j], -side), body=_go(stub + int(rng.randint(0, 2))))
              for j in sorted(spots[:int(rng.randint(2, 4))])]
    return Layout(route, decoys)


def crossing(g, rng):
    route = route_into_red(g, rng, shape(rng, _len(rng, 5, 9, 4)), lead=3)
    n = len(route.poses)
    x, y, h = route.poses[int(rng.randint(2, n - 3))]
    u = h + _sign(rng) * _deg(rng, 70, 90)
    half = rng.uniform(0.09, 0.10)
    d = np.array([np.cos(u), np.sin(u)])
    near = decoy_into(g, rng, (x - d[0] * half, y - d[1] * half, u), body=_go(int(rng.randint(2, 4))))
    pen = Pen(g, x + d[0] * half, y + d[1] * half, u).go(int(rng.randint(1, 3)))
    decoys = [near, pen.chain(False)]
    if rng.rand() < 0.5:
        decoys.append(_aimed(g, rng, aim_at_red(rng)))
    return Layout(route, decoys)


def relay(g, rng):
    segs = [int(rng.randint(3, 5)) for _ in range(3)]
    turns = [_sign(rng) * _deg(rng, 30, 60) for _ in range(2)]
    program = _seq(_turn(turns[0]), _go(segs[1] - 1), _turn(turns[1]), _go(segs[2] - 1))
    route = route_into_red(g, rng, program, lead=segs[0])
    side = _sign(rng)
    end = (-g.S + rng.uniform(-0.02, 0.02), side * rng.uniform(0.13, 0.16), 0.0)
    pen = Pen(g, end[0], end[1], np.pi).go(segs[0] - 1)
    at = int(rng.randint(2))
    kind = str(rng.choice(["jog", "sharp"]))
    stop = None
    for i, (t, seg) in enumerate(zip(turns, segs[1:])):
        if i == at:
            if kind == "sharp":
                stop = pen.n - 1
                pen.snap(t).go(link=SHARP_LINK * g.S).go(seg - 1)
            else:
                pen.turn(t)
                stop = pen.n - 1
                pen.go(lateral=_sign(rng) * rng.uniform(0.075, 0.09)).go(seg - 2)
        else:
            pen.turn(t).go(seg - 1)
    return Layout(route, [pen.chain(True, kind, stop)])


def false_extension(g, rng):
    route = route_into_red(g, rng, shape(rng, _len(rng, 5, 8, 3)), lead=2)
    decoys = [decoy_into(g, rng, side_on(g, rng, route.poses[0], _sign(rng)),
                         body=shape(rng, _len(rng, 2, 5, 2), "line" if rng.rand() < 0.5 else "arc"))]
    if rng.rand() < 0.6:
        decoys.append(_aimed(g, rng, aim_at_red(rng)))
    return Layout(route, decoys)


def zigzag_vs_straight(g, rng):
    # Two legs: three run about 0.66 m, past the robot's reach.
    route = route_into_red(g, rng, shape(rng, 0, "zigzag", zig_max=40, legs=2), lead=2)
    decoy = _aimed(g, rng, aim_at_red(rng), body=_go(_len(rng, 3, 5, 2)))
    return Layout(route, [decoy])


def _twin(g, rng, brk, legs):
    """A zig-zag route and a matching zig-zag 12-14 cm to one side that ends beside red.

    The break sits in the straight lead-in, so both zig-zags start level.
    """
    zig = shape(rng, 0, "zigzag", zig_max=35, legs=legs)
    route = route_into_red(g, rng, zig, lead=3)
    side = _sign(rng)
    pen = Pen(g, -g.S + rng.uniform(-0.015, 0.015), side * rng.uniform(0.12, 0.14), np.pi)
    stop = 0
    if brk == "jog":
        # Outward, away from the route (walking back, +lateral points to -y).
        pen.go().go(lateral=-side * rng.uniform(0.075, 0.09))
        stop = 1
    else:
        pen.low.update((1, 2, 3))
        pen.go(3)
    zig(pen)
    return Layout(route, [pen.chain(True, brk, stop)])


def twin_zigzags(g, rng):
    # Two legs: three run about 0.7 m, past the robot's reach.
    return _twin(g, rng, "low", 2)


def zigzag_jog(g, rng):
    return _twin(g, rng, "jog", 2)


def u_turn(g, rng):
    route = route_into_red(g, rng, _seq(_turn(_sign(rng) * np.pi), _go(_len(rng, 2, 5, 2))),
                           lead=int(rng.randint(2, 4)))
    short = decoy_into(g, rng, aim_at_red(rng, front=True, r=rng.uniform(0.19, 0.22)),
                       body=_go(int(rng.randint(2, 4))))
    return Layout(route, [short])


def spiral_in(g, rng):
    side, total = _sign(rng), _deg(rng, 180, 250) * LENGTH_SCALE
    steps = int(np.ceil(total / np.deg2rad(13)))
    turns = np.linspace(MAX_TURN, np.deg2rad(8), steps)
    turns *= total / turns.sum()

    def spiral(pen):
        for t in turns:
            pen.h += side * t
            pen.go()
    route = route_into_red(g, rng, spiral, lead=2)
    decoys = [_aimed(g, rng, aim_at_red(rng, front=True))]
    if rng.rand() < 0.5:
        decoys.append(decoy_into(g, rng, beside_red(rng, _sign(rng)), tail=3, n=_len(rng, 3, 5, 2)))
    return Layout(route, decoys)


def hook(g, rng):
    side, bend = _sign(rng), _deg(rng, 35, 50)
    route = route_into_red(g, rng, _seq(_turn(side * bend), _go(int(rng.randint(0, 2))), _turn(-side * bend),
                                        shape(rng, _len(rng, 3, 6, 2), "line" if rng.rand() < 0.5 else None)),
                           lead=int(rng.randint(1, 3)))
    decoy = _aimed(g, rng, aim_at_red(rng, -side), body=_go(_len(rng, 3, 5, 2)))
    return Layout(route, [decoy])


def s_curve_vs_line(g, rng):
    route = route_into_red(g, rng, shape(rng, _len(rng, 9, 13, 8), "s_curve"), lead=2)
    decoy = _aimed(g, rng, aim_at_red(rng), brk="low", body=_go(_len(rng, 3, 5, 2)))
    return Layout(route, [decoy])


def rounded_vs_sharp(g, rng):
    side = _sign(rng)
    route = route_into_red(g, rng, _seq(_turn(side * np.pi / 2), _go(_len(rng, 2, 5, 2))),
                           lead=int(rng.randint(2, 4)))
    # The sharp corner is the decoy's last piece, a little further out than
    # other aimed decoys, so nothing past the corner can fall on toward red.
    end = aim_at_red(rng, -side, r=rng.uniform(0.22, 0.25))
    pen = Pen(g, end[0], end[1], end[2] + np.pi)
    stop = pen.n - 1
    pen.snap(-side * np.pi / 2).go(link=SHARP_LINK * g.S).go(int(rng.randint(1, 4)))
    return Layout(route, [pen.chain(True, "sharp", stop)])


def serpentine(g, rng):
    side = _sign(rng)
    steps = []
    for _ in range(2 if rng.rand() < 0.7 else 3):
        steps += [_turn(side * _deg(rng, 40, 55)), _go(int(rng.randint(1, 3)))]
        side = -side
    route = route_into_red(g, rng, _seq(*steps), lead=3)
    straight = straight_spots(route, 1, len(route.poses) - 1) or list(range(1, len(route.poses) - 1))
    spots = rng.choice(straight, size=min(len(straight), int(rng.randint(1, 3))), replace=False)
    decoys = [decoy_into(g, rng, side_on(g, rng, route.poses[j], _sign(rng)), body=_go(int(rng.randint(1, 3))))
              for j in sorted(spots)]
    if rng.rand() < 0.5:
        decoys.append(decoy_into(g, rng, beside_red(rng, _sign(rng)), tail=3,
                                 body=gentle(rng, int(rng.randint(1, 3)))))
    return Layout(route, decoys)


def long_way_round(g, rng):
    side = _sign(rng)
    # Loop segments not shortened: the loop has to clear the decoy inside it.
    route = route_into_red(g, rng, _seq(_turn(side * _deg(rng, 90, 120)), _go(int(rng.randint(3, 5))),
                                        _turn(side * _deg(rng, 90, 120)), _go(int(rng.randint(1, 3)))),
                           lead=int(rng.randint(3, 5)))
    # The twist is the decoy's red end; its plain body is kept short because
    # the route loops round it.
    decoy = _aimed(g, rng, aim_at_red(rng, side), brk="twist", body=_go(int(rng.randint(1, 3))))
    return Layout(route, [decoy])


def ladder(g, rng):
    trunk, rung = _len(rng, 3, 6, 3), int(rng.randint(3, 5))
    route = route_into_red(g, rng, _seq(lambda pen: pen.go(link=rng.uniform(0.06, 0.075)), _go(rung - 1)), lead=trunk)
    start = route.poses[0]
    pitch = rng.uniform(0.085, 0.10)
    offsets = [-2, -1, 1, 2]
    rng.shuffle(offsets)
    decoys = []
    for k in offsets[:int(rng.randint(2, 4))]:
        x = start[0] + rng.uniform(-0.015, 0.015)
        pen = Pen(g, x, start[1] + k * pitch, 0.0).go(rung - 1)
        decoys.append(pen.chain(False))
    return Layout(route, decoys)


def y_merge_t(g, rng):
    """Two feeders curve into a trunk: the true one bends in end-on, the false one meets the trunk side-on (a T).

    The false feeder joins two pieces past the bend; the bend piece falls
    across the trunk line and would land on it any earlier.
    """
    side, trunk, feed = _sign(rng), int(rng.randint(4, 6)), int(rng.randint(1, 3))
    bend = side * _deg(rng, 35, 55)
    route = route_into_red(g, rng, _seq(_turn(bend), _go(feed)), lead=trunk)
    n = len(route.poses)
    end = side_on(g, rng, route.poses[n - trunk + 2], side)
    pen = Pen(g, end[0], end[1], end[2] + np.pi)
    pen.turn(-bend * 0.6)
    pen.go(max(1, feed - 1))
    return Layout(route, [pen.chain(True)])


def stub_field(g, rng):
    """Short stubs aimed at red from the front and front diagonals end short of it or beside it; the route comes
    in from behind."""
    route = route_into_red(g, rng, gentle(rng, _len(rng, 3, 6, 2)), lead=3)

    def front(phi):
        r = rng.uniform(0.19, 0.22)
        return (r * np.cos(phi), r * np.sin(phi), phi + np.pi)
    ends = [front(_deg(rng, -10, 10)), front(_deg(rng, 35, 50)), front(-_deg(rng, 35, 50))]
    ends += [beside_red(rng, s) for s in (-1, 1) if rng.rand() < 0.5]
    rng.shuffle(ends)
    k = min(len(ends), int(rng.randint(3, 6)))
    return Layout(route, [decoy_into(g, rng, e, tail=2, body=_go(int(rng.randint(0, 2)))) for e in ends[:k]])


def double_cross(g, rng):
    """Two decoys cross the route through gaps of 16 cm or more; sometimes a third, broken one is aimed at red."""
    route = route_into_red(g, rng, shape(rng, int(rng.randint(5, 7)), "line" if rng.rand() < 0.5 else "arc"), lead=2)
    n = len(route.poses)
    picks = sorted(rng.choice(np.arange(2, n - 2), size=2, replace=False))
    if picks[1] - picks[0] < 3:
        picks[1] = min(n - 3, picks[0] + 3)
    decoys = []
    for i in picks:
        x, y, h = route.poses[i]
        u = h + _sign(rng) * _deg(rng, 75, 90)
        half = rng.uniform(0.09, 0.10)
        d = np.array([np.cos(u), np.sin(u)])
        decoys.append(decoy_into(g, rng, (x - d[0] * half, y - d[1] * half, u), body=_go(int(rng.randint(1, 3)))))
        decoys.append(Pen(g, x + d[0] * half, y + d[1] * half, u).go(int(rng.randint(1, 3))).chain(False))
    if rng.rand() < 0.3:
        decoys.append(_aimed(g, rng, aim_at_red(rng, front=True)))
    return Layout(route, decoys)


def shadow_route(g, rng):
    """An unbroken copy of the route's shape 10-12 cm to one side that ends beside red instead of at it."""
    side = _sign(rng)
    prog = shape(rng, _len(rng, 4, 8, 3), str(rng.choice(["arc", "line", "zigzag"])))
    route = route_into_red(g, rng, prog, lead=2)
    pen = Pen(g, -g.S + rng.uniform(-0.01, 0.01), side * rng.uniform(0.10, 0.12), np.pi).go(1)
    prog(pen)
    return Layout(route, [pen.chain(True)])


def long_near_miss(g, rng):
    """A long smooth arc sweeps toward red and passes beside it; the true route turns in from the other side."""
    side = _sign(rng)
    route = route_into_red(g, rng, _seq(_turn(side * _deg(rng, 60, 90)), _go(_len(rng, 2, 4, 2))), lead=2)
    end = beside_red(rng, side)
    pen = Pen(g, end[0], end[1], end[2] + np.pi).go(2)
    total = -side * _deg(rng, 40, 70)
    k = int(rng.randint(5, 8))
    for _ in range(k):
        pen.h += total / k
        pen.go()
    return Layout(route, [pen.chain(True)])


PATTERNS = {f.__name__: f for f in (
    fan_in, near_miss_lanes, collision_course, edge_on_finish, orbit,
    y_merge_t, double_merge, side_joins, comb, crossing, relay, false_extension,
    zigzag_vs_straight, twin_zigzags, zigzag_jog, u_turn, spiral_in, hook,
    s_curve_vs_line, rounded_vs_sharp, serpentine, long_way_round, ladder,
    stub_field, double_cross, shadow_route, long_near_miss,
)}


# ── Geometry checks ─────────────────────────────────────────────────

def _box(center, heading, along, lateral):
    u = np.array([np.cos(heading), np.sin(heading)])
    return np.asarray(center, float), u, np.array([along, lateral])


def _overlap(a, b, clearance=0.0):
    (ca, ua, ea), (cb, ub, eb) = a, b
    d = cb - ca
    if d @ d > (np.hypot(*ea) + np.hypot(*eb) + clearance) ** 2:
        return False
    va, vb = np.array([-ua[1], ua[0]]), np.array([-ub[1], ub[0]])
    for axis in (ua, va, ub, vb):
        ra = ea[0] * abs(ua @ axis) + ea[1] * abs(va @ axis)
        rb = eb[0] * abs(ub @ axis) + eb[1] * abs(vb @ axis)
        if abs(d @ axis) > ra + rb + clearance:
            return False
    return True


def footprint(g, pose):
    return _box(pose[:2], pose[2], g.T / 2, g.W / 2)


def sweep(g, pose, reverse=False):
    """Area a piece covers as it falls along its heading (or against it)."""
    h = pose[2] + (np.pi if reverse else 0.0)
    u = np.array([np.cos(h), np.sin(h)])
    return _box(np.asarray(pose[:2]) + u * g.H / 2, h, g.H / 2 + g.T / 2, g.W / 2)


def problem(layout, g, capacity):
    """Why *layout* (world frame) is rejected, or ``None`` when it fits the table and its route can fall
    without touching a decoy."""
    if layout.count() > capacity:
        return "too many pieces"
    chains = layout.chains
    pieces = [(c, i, p) for c, ch in enumerate(chains) for i, p in enumerate(ch.poses)]
    xy = np.array([p[:2] for _, _, p in pieces] + [layout.red[:2]])
    if (xy[:, 0] < REGION[0, 0]).any() or (xy[:, 0] > REGION[0, 1]).any() \
            or (xy[:, 1] < REGION[1, 0]).any() or (xy[:, 1] > REGION[1, 1]).any():
        return "off the table"
    if _farthest(xy[:-1]) > REACH:
        return "out of the robot's reach"
    red_h = layout.red[2]
    red_end = np.asarray(layout.red[:2]) + 0.10 * np.array([np.cos(red_h), np.sin(red_h)])
    if (np.abs(red_end) > TABLE_HALF).any():
        return "red falls off the table"
    boxes = [footprint(g, p) for _, _, p in pieces]
    red_box = _box(layout.red[:2], red_h, g.red_T / 2, g.red_W / 2)
    red_fall = _box(np.asarray(layout.red[:2]) + 0.5 * (red_end - layout.red[:2]), red_h,
                    (g.red_H + g.red_T) / 2 + 0.01, g.red_W / 2 + 0.01)
    route_last = len(chains[0].poses) - 1
    for a in range(len(pieces)):
        ca, ia, pa = pieces[a]
        for b in range(a + 1, len(pieces)):
            cb, ib, pb = pieces[b]
            if ca == cb:
                if abs(ia - ib) > 1 and _overlap(boxes[a], boxes[b], 0.003):
                    return "chain overlaps itself"
                if abs(ia - ib) == 1 and _overlap(boxes[a], boxes[b], NEIGHBOUR_CLEAR - 1e-6):
                    return "neighbouring pieces overlap"
            elif _overlap(boxes[a], boxes[b], layout.clearance):
                return "chains overlap"
        if not (ca == 0 and ia == route_last) and _overlap(boxes[a], red_box, max(layout.clearance, 0.008)):
            return "piece crowds red"
    # The true cascade (and red's fall) must not brush any decoy, nor skip
    # ahead along its own route.
    route = chains[0].poses
    for i, p in enumerate(route):
        sw = sweep(g, p)
        for k, (c, j, _) in enumerate(pieces):
            if c and _overlap(sw, boxes[k]):
                return "route falls onto a decoy"
            if not c and j >= i + 4 and _overlap(sw, boxes[k]):
                return "route falls onto its own later piece"
    if any(c and _overlap(red_fall, boxes[k]) for k, (c, _, _) in enumerate(pieces)):
        return "red falls onto a decoy"
    # No decoy piece, even one jostled loose past its break, may fall onto red;
    # nor may a piece the wave topples land on the route pieces next to red,
    # which even a side-on knock would shove in.
    near_red = [footprint(g, p) for p in chains[0].poses
                if np.hypot(p[0] - layout.red[0], p[1] - layout.red[1]) < 0.10]
    for c, ch in enumerate(chains[1:], start=1):
        live = set(ch.live())
        for i, p in enumerate(ch.poses):
            for sw in [sweep(g, p)] + ([sweep(g, p, True)] if ch.both_ways else []):
                if _overlap(sw, red_box) or (i in live and any(_overlap(sw, b) for b in near_red)):
                    return "decoy reaches red"
    # A pushed decoy may knock others side-on, but never along their line
    # (that would carry the wave on) and never red.
    for c, ch in enumerate(chains[1:], start=1):
        for i in ch.live():
            p = ch.poses[i]
            for sw in [sweep(g, p)] + ([sweep(g, p, True)] if ch.both_ways else []):
                if _overlap(sw, red_box):
                    return "decoy reaches red"
                for k, (c2, _, q) in enumerate(pieces):
                    if c2 == c or frozenset((c, c2)) in layout.allow:
                        continue
                    if _line_diff(p[2], q[2]) < SIDE_ON and _overlap(sw, boxes[k]):
                        return "decoy drives another chain along its line"
    # One solution only: no decoy piece, pushed either way (past its break
    # too), may reach red or the route pieces next to it, or drive a route
    # piece along its line. Low blocks are too short to knock anything over.
    for c, ch in enumerate(chains[1:], start=1):
        for i, p in enumerate(ch.poses):
            if i in ch.low:
                continue
            for sw in (sweep(g, p), sweep(g, p, True)):
                if _overlap(sw, red_box) or any(_overlap(sw, b) for b in near_red):
                    return "a decoy piece reaches red"
                for k, (c2, _, q) in enumerate(pieces):
                    if c2 == 0 and _line_diff(p[2], q[2]) < SIDE_ON and _overlap(sw, boxes[k]):
                        return "a decoy piece drives the route"
    # Room for a fingertip behind the starter.
    x, y, h = route[0]
    u = np.array([np.cos(h), np.sin(h)])
    behind = _box(np.array([x, y]) - u * (g.T / 2 + 0.05), h, 0.05, 0.045)
    if any(k and _overlap(behind, boxes[k]) for k in range(len(pieces))) or _overlap(behind, red_box):
        return "no room behind the starter"
    return None


def _farthest(xy):
    return float(np.max(np.hypot(xy[:, 0] - ROBOT_BASE_XY[0], xy[:, 1] - ROBOT_BASE_XY[1])))


def check(layout, g, capacity):
    return problem(layout, g, capacity) is None




def shift_bounds(layout):
    """Range of shifts that keep every piece centre in REGION and red's fall on the table, or ``None``."""
    xy = np.array([p[:2] for ch in layout.chains for p in ch.poses] + [layout.red[:2]])
    h = layout.red[2]
    fall = np.asarray(layout.red[:2]) + 0.10 * np.array([np.cos(h), np.sin(h)])
    lo = np.maximum(REGION[:, 0] - xy.min(axis=0), -TABLE_HALF - fall)
    hi = np.minimum(REGION[:, 1] - xy.max(axis=0), TABLE_HALF - fall)
    return None if (lo > hi).any() else (lo, hi)


def sample_layout(rng, g, names, capacity, tries=300):
    """Draw a pattern from *names* and place it; a pattern that never fits gives way to another draw."""
    for _ in range(len(names) * 4):
        name = str(names[rng.randint(len(names))])
        for _ in range(tries):
            local = PATTERNS[name](g, rng)
            local.pattern = name
            mirror = _sign(rng)
            if local.yaws is not None:
                yaw = float(rng.choice(local.yaws)) + _deg(rng, -2, 2)
            else:
                yaw = rng.uniform(-np.pi, np.pi)
            bounds = shift_bounds(local.transformed(mirror, yaw, (0.0, 0.0)))
            if bounds is None:
                continue
            # Cheap reach test over a few shifts before the full check.
            for _ in range(8):
                placed = local.transformed(mirror, yaw, rng.uniform(*bounds))
                xy = np.array([p[:2] for ch in placed.chains for p in ch.poses])
                if _farthest(xy) <= REACH:
                    break
            if check(placed, g, capacity):
                return placed
    raise RuntimeError(f"No domino layout fits from patterns {list(names)}")


def apply_layout(task):
    """Sample a layout for *task*, assign its pieces to pool bodies at random and place them."""
    from imbench.tasks.prehensile.domino_patterns import heading_to_quat
    rng = task._rng
    g = Geometry(task._correct_spacing, np.asarray(task._domino_size) * task._SIZE_MAX, [0.005, 0.03, 0.04])
    n = task._dominoes_per_chain
    names = [f"chain{k}_domino_{j}" for k in range(task._num_chains) for j in range(n)]
    layout = sample_layout(rng, g, task._layout_patterns, len(names))
    pool = [names[i] for i in rng.permutation(len(names))]
    assigned, layout_map, specs = iter(pool), {}, []
    for c, ch in enumerate(layout.chains):
        chain_names = [next(assigned) for _ in ch.poses]
        for i, (name, pose) in enumerate(zip(chain_names, ch.poses)):
            layout_map[name] = pose
            if i in ch.low:
                k, j = (int(v) for v in name.replace("chain", "").split("_domino_"))
                task._chain_sizes[k, j, 0] = task._domino_size[0] * LOW_HEIGHT_FRACTION
        specs.append({"role": "route" if c == 0 else "decoy", "names": chain_names, "defect": ch.defect,
                      "stop": ch.stop})
    task._solution_route = specs[0]["names"]
    task._decoy_starts = [s["names"][0] for s in specs[1:]]
    task._layout_pattern = layout.pattern
    task._network_layout = layout_map
    task._parked = set(names) - set(layout_map)
    task._resize_dominoes()
    for name, (x, y, h) in layout_map.items():
        k, j = (int(v) for v in name.replace("chain", "").split("_domino_"))
        task._place_object(name, [x, y, task._table_height + task._chain_sizes[k, j, 0] + .002],
                           quat=heading_to_quat(h))
    task._park_objects(task._parked)
    task._chain_specs = [{"pattern": layout.pattern, **s,
                          "layout": [layout_map[name] for name in s["names"]]} for s in specs]
    return list(layout.route.poses)
