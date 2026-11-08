"""
Policy for ib-p3-domino-select-v1: traces which standing domino's cascade
reaches the red target (or, for independent lanes, scores each chain for
hidden, flat or misaligned dominoes), then strikes it: rise clear of every
piece, travel on a curved path, drop straight down behind the starter with
the wrist turned so the hand clears its neighbours, flick the starter's upper
part forward and pull back before the hand can reach the next piece.
"""

import numpy as np
import imbench as ib
from imbench.policy.base_policy import PolicyInterface
from data_synthetic.episode_length_stats import sample_target_length


# ---------------------------------------------------------------------------
# Human-likeness hyper-parameters
# ---------------------------------------------------------------------------

# Path curvature: peak lateral deviation of the travel path as a fraction of
# its length (a human wrist never moves in a straight line).
CURVE_ARC_FRAC = 0.25

# Episode-length distribution (Domino Chain Selection): N(mean, std) truncated to [min, max].
# Each reset draws a target step count; the travel segment's speed is chosen so
# the strike lands near it, but early enough for the detected route's cascade
# to reach red within the task's step limit.
EP_LEN_MEAN = 703
EP_LEN_STD  = 273
EP_LEN_MIN  = 287
# Below the teleop maximum (2517): the task budget is 1000 steps and the
# cascade needs another 150-250 after the strike.
EP_LEN_MAX  = 750


# ---------------------------------------------------------------------------
# Domino / geometry helpers
# ---------------------------------------------------------------------------

def get_domino_pos(sim, chain_idx, dom_idx):
    name = f"chain{chain_idx}_domino_{dom_idx}_main"
    bid  = sim.model.body_name2id(name)
    return sim.data.body_xpos[bid].copy()


def get_domino_upright(sim, chain_idx, dom_idx):
    name = f"chain{chain_idx}_domino_{dom_idx}_main"
    bid  = sim.model.body_name2id(name)
    quat = sim.data.body_xquat[bid]
    w, x, y, z = quat
    return 1 - 2 * (x * x + y * y)


# ---------------------------------------------------------------------------
# Chain scorer
# ---------------------------------------------------------------------------

def detect_correct_chain(sim, num_chains=5, dominoes_per_chain=6,
                         correct_spacing=0.045, table_height=0.8, interconnected=False):
    if interconnected:
        return detect_connected_starter(sim, num_chains, dominoes_per_chain, table_height)
    # Read actual sizes and local headings. A global straightness or fixed
    # spacing threshold would reject valid curves and miss low-block breaks.
    scores = []
    for c in range(num_chains):
        penalty = 0
        blocks = []
        for i in range(dominoes_per_chain):
            body = sim.model.body_name2id(f"chain{c}_domino_{i}_main")
            pos = sim.data.body_xpos[body].copy()
            axes = sim.data.body_xmat[body].reshape(3, 3)
            geom = sim.model.body_geomadr[body]
            sizes = sim.model.geom_size[geom].copy()
            active = table_height - 0.5 < pos[2] < table_height + 1.0
            if not active or abs(axes[2, 2]) < 0.5:
                penalty += 100
            blocks.append((pos, axes, sizes, active))
        for previous, following in zip(blocks, blocks[1:]):
            pos, axes, sizes, active = previous
            next_pos, next_axes, next_sizes, next_active = following
            if not (active and next_active):
                continue
            delta = next_pos[:2] - pos[:2]
            forward, sideways = axes[:2, 0], axes[:2, 1]
            next_forward_extent = np.sum(next_sizes[:2] * np.abs(next_axes[:2, :2].T @ forward))
            next_side_extent = np.sum(next_sizes[:2] * np.abs(next_axes[:2, :2].T @ sideways))
            along = float(delta @ forward)
            lateral = abs(float(delta @ sideways))
            if (along <= 0 or along > 2 * sizes[2] + next_forward_extent + 0.002
                    or lateral > sizes[1] + next_side_extent):
                penalty += 50
        scores.append(penalty)
    best = int(np.argmin(scores))
    print(f"[detect] chain scores: {scores}  → correct={best}")
    return best


def detect_connected_starter(sim, num_chains, dominoes_per_chain, table_height):
    """Pool index of :func:`detect_starter`'s domino."""
    name = detect_starter(sim, num_chains, dominoes_per_chain, table_height)
    return int(name.replace("chain", "").split("_domino_")[0])


#: Largest heading difference across which a falling domino carries the wave
#: on; side-on hits, twisted pieces and sharp corners exceed it.
MAX_LINK_TURN = np.deg2rad(50)


def detect_starter(sim, num_chains, dominoes_per_chain, table_height):
    """Name of the domino that starts a continuous directed route to red (see :func:`detect_route`)."""
    return detect_route(sim, num_chains, dominoes_per_chain, table_height)[0]


def detect_route(sim, num_chains, dominoes_per_chain, table_height):
    """(starter name, number of pieces on its route) for the route that reaches red.

    Traces local contact possibilities only and never reads solution labels:
    each standing domino falls onto the nearest piece in its path, and the
    wave carries on only when that piece is within reach, stands at least
    half as tall, and faces within MAX_LINK_TURN of the same line. Among
    dominoes whose wave reaches red, the one nothing else knocks over wins.
    """
    names = [f"chain{k}_domino_{j}" for k in range(num_chains) for j in range(dominoes_per_chain)] + ["red_block"]
    bodies = {}
    for name in names:
        bid = sim.model.body_name2id(name+"_main")
        pos = sim.data.body_xpos[bid].copy()
        if not table_height-.1 < pos[2] < table_height+.3:
            continue
        gid = sim.model.body_geomadr[bid]
        bodies[name] = (pos, sim.data.body_xmat[bid].reshape(3, 3).copy(), sim.model.geom_size[gid].copy())
    edges = {}
    for name, (p, axes, sizes) in bodies.items():
        if name == "red_block":
            continue
        forward, sideways = axes[:2, 0], axes[:2, 1]
        candidates = []
        for other, (q, next_axes, next_sizes) in bodies.items():
            delta = q[:2]-p[:2]; along = float(delta @ forward)
            if along <= .005 or other == name:
                continue
            extent = np.abs(next_axes[:2, :2].T @ sideways) @ next_sizes[:2]
            if abs(delta @ sideways) > sizes[1]+extent+.0005:
                continue
            candidates.append((along, other))
        if candidates:
            along, following = min(candidates)
            _, next_axes, next_sizes = bodies[following]
            reach = 2*sizes[2] + np.abs(next_axes[:2, :2].T @ forward) @ next_sizes[:2]
            turn = np.arccos(min(1.0, abs(float(forward @ next_axes[:2, 0]) / np.linalg.norm(next_axes[:2, 0]))))
            tall = following == "red_block" or next_sizes[2] >= .5*sizes[2]
            if along <= reach+.002 and turn <= MAX_LINK_TURN and tall:
                edges[name] = following
    lengths = {}
    for name in edges:
        node, seen = name, set()
        while node in edges and node not in seen:
            seen.add(node); node = edges[node]
        if node == "red_block":
            lengths[name] = len(seen)
    knocked = set(edges.values())
    starters = sorted((-n, name) for name, n in lengths.items() if name not in knocked)
    if not starters:
        raise RuntimeError("No starter has a continuous directed route to the red target")
    return starters[0][1], -starters[0][0]


# ---------------------------------------------------------------------------
# Curved waypoint generation
# ---------------------------------------------------------------------------

def _perp3d(direction):
    """Return a unit vector perpendicular to `direction` in the XY plane."""
    dx, dy, dz = direction
    length_xy = np.sqrt(dx * dx + dy * dy)
    if length_xy < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return np.array([-dy / length_xy, dx / length_xy, 0.0])


def interpolate_waypoints_curved(start, end, n_steps, rng, arc_frac=CURVE_ARC_FRAC):
    """
    Cosine-eased waypoints from start → end over n_steps positions with a
    slight perpendicular arc to mimic natural wrist curvature.

    The peak lateral deviation is `arc_frac * segment_length`, placed at
    the midpoint of the trajectory.  A random sign is drawn each call so
    the arm sometimes curves left, sometimes right.
    """
    seg        = end - start
    seg_len    = np.linalg.norm(seg)
    perp       = _perp3d(seg)
    arc_sign   = rng.choice([-1, 1])
    peak_dev   = arc_sign * arc_frac * seg_len

    waypoints = []
    for i in range(n_steps):
        t        = (i + 1) / n_steps
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))

        # Linear interpolation along the chord
        pt = start + t_smooth * seg

        # Bell-shaped lateral arc: sin(π·t) peaks at t=0.5, zero at ends
        lateral_offset = peak_dev * np.sin(np.pi * t)
        pt = pt + lateral_offset * perp

        waypoints.append(pt)

    return waypoints


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DominoSelectPolicy(PolicyInterface):
    """Strike the detected starter: rise clear of every piece, travel on a curved
    human-like path, drop straight down behind the starter with the wrist turned
    so the hand clears its neighbours, flick the starter's upper part forward to
    a stop short of the next piece, then pull back and up.
    """

    #: Hand travels this far above the tallest piece (m). Higher costs reach
    #: close to the robot base, where the Panda cannot get far above the table.
    CLEAR_ABOVE = 0.05
    #: Fingertip contact height as a fraction of the starter's height, and
    #: the least distance (m) the fingertips stay below its top edge.
    CONTACT_FRAC = 0.75
    TOP_MARGIN = 0.023
    #: EEF site sits this far above the closed fingertips (m).
    TIP_BELOW_EEF = 0.011
    #: The hand body's underside sits this far above the EEF site (m).
    HAND_ABOVE_EEF = 0.022
    #: Gain on the wrist-yaw error (rad) for the rotation part of the action.
    YAW_GAIN = 2.0
    #: Hand-body footprint about the EEF in world x/y with the wrist in its
    #: reset orientation (the long finger-opening axis along world y), and
    #: the closed fingers' footprint (m).
    HAND_XY = ((-0.047, 0.028), (-0.104, 0.104))
    FINGER_XY = ((-0.017, 0.012), (-0.02, 0.02))
    #: Wrist turns tried (deg), and how far the fingertips stay short of any piece ahead (m).
    YAW_CHOICES = np.arange(-120, 121, 15)
    AHEAD_CLEAR = 0.008
    #: Gap between the starter's back face and the fingertip at the start of
    #: the push (m): the run-up that lets the strike reach speed. Layouts keep
    #: 10 cm clear behind every starter.
    BACK_GAP = 0.055
    #: The strike stops once the starter leans this far, or the fingertips are
    #: this far past its centre at most (less when a piece stands close ahead).
    TIP_DONE_DEG = 25.0
    PUSH_LIMIT = 0.03
    #: Steps the cascade needs per route piece, plus a margin, when budgeting the strike.
    CASCADE_STEPS_PER_PIECE = 10
    CASCADE_MARGIN = 120
    #: Typical steps for the descent and for the strike.
    DESCEND_STEPS = 130
    PUSH_STEPS = 120

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase = "detect"
        self.step_in_phase = 0
        self._correct_chain = None
        self._starter = None
        self._route_len = None
        self._start_pos = None
        self._end_pos = None
        self._travel_waypoints = []
        self._hold = None
        self._sim = None
        self._R0 = None
        self._yaw = 0.0
        self._over = False

        # Sample a target episode length (teleop-like spread); the travel
        # segment's speed is chosen to fill it. Detection re-budgets it once
        # the route length (and so the cascade time) is known.
        self._target_episode_steps = sample_target_length(
            self.rng, EP_LEN_MEAN, EP_LEN_STD, EP_LEN_MIN, EP_LEN_MAX,
        )
        self._ctrl_period = 4
        self._n_interp_travel = 8
        self._strike_gain = float(self.rng.uniform(16.0, 22.0))
        self._contact_jitter = float(self.rng.uniform(-0.05, 0.05))   # fraction of height
        self._back_jitter = float(self.rng.uniform(-0.005, 0.005))

        self._last_action = np.zeros(7)
        self._last_action[6] = 1.0
        self._ctrl_step = 0

    def _next(self, phase):
        self.phase = phase
        self.step_in_phase = 0

    # ------------------------------------------------------------------
    # Strike geometry
    # ------------------------------------------------------------------

    def _build_strike(self, sim, starter, table_height=0.8):
        bid = sim.model.body_name2id(f"{starter}_main")
        pos0 = sim.data.body_xpos[bid].copy()
        half = sim.model.geom_size[sim.model.body_geomadr[bid]]   # thickness, width, height halves
        # A domino's local x axis is the direction its cascade falls.
        direction = sim.data.body_xmat[bid].reshape(3, 3)[:2, 0].copy()
        direction /= np.linalg.norm(direction)
        self._starter_bid = bid
        self._dir = direction
        self._centre = pos0[:2]

        base_z = pos0[2] - half[2]
        top_z = base_z + 2 * half[2]
        back = half[0] + self.BACK_GAP + self._back_jitter
        start_xy = pos0[:2] - direction * back

        tops, others = [], []
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i) or ""
            if not name.endswith("_main") or not ("domino" in name or name == "red_block_main"):
                continue
            q = sim.data.body_xpos[i]
            if not table_height - 0.1 < q[2] < table_height + 0.3:
                continue
            h = sim.model.geom_size[sim.model.body_geomadr[i]]
            tops.append(q[2] + h[2])
            if i != bid and np.hypot(*(q[:2] - pos0[:2])) < 0.3:
                ax = sim.data.body_xmat[i].reshape(3, 3)[:2, 0]
                others.append((q[:2].copy(), ax / np.linalg.norm(ax), h.copy(), q[2] + h[2]))

        # Stop the push with the fingertips short of anything standing ahead.
        across = np.array([-direction[1], direction[0]])
        ahead = [0.08]
        for c, ax, h, top in others:
            d = c - pos0[:2]
            ext_along = abs(ax @ direction) * h[0] + abs(ax @ across) * h[1]
            ext_side = abs(ax @ across) * h[0] + abs(ax @ direction) * h[1]
            if d @ direction > 0 and abs(d @ across) < ext_side + 0.025:
                ahead.append(float(d @ direction - ext_along))
        finger_front = 0.02
        self._push_limit = float(np.clip(min(ahead) - self.AHEAD_CLEAR - finger_front, 0.012, self.PUSH_LIMIT))
        end_xy = pos0[:2] + direction * self._push_limit

        def corners(xr, yr, yaw, at):
            c_, s_ = np.cos(yaw), np.sin(yaw)
            R = np.array([[c_, -s_], [s_, c_]])
            pts = np.array([[x, y] for x in xr for y in yr])
            return at + pts @ R.T

        def overlaps(poly, c, ax, h):
            # Separating-axis test between a convex polygon and an oriented box.
            box = np.array([c + sx * ax * h[0] + sy * np.array([-ax[1], ax[0]]) * h[1]
                            for sx in (-1, 1) for sy in (-1, 1)])
            for axis in (ax, np.array([-ax[1], ax[0]]), *[np.array([-e[1], e[0]]) for e in
                         (poly[1] - poly[0], poly[2] - poly[0])]):
                n = axis / (np.linalg.norm(axis) + 1e-12)
                a, b = poly @ n, box @ n
                if a.max() < b.min() or b.max() < a.min():
                    return False
            return True

        def hits_for(contact_z, yaw):
            hand_bottom = contact_z + self.HAND_ABOVE_EEF
            hits = 0
            for k, at in enumerate((start_xy, end_xy)):
                hand = corners(self.HAND_XY[0], self.HAND_XY[1], yaw, at)
                fing = corners(self.FINGER_XY[0], self.FINGER_XY[1], yaw, at)
                for c, ax, h, top in others:
                    if top > hand_bottom - 0.005 and overlaps(hand, c, ax, h):
                        hits += 1
                    if overlaps(fing, c, ax, h):
                        hits += 1
                # The hand may not come down on the starter's top either.
                if k == 0 and top_z > hand_bottom - 0.005 and overlaps(hand, pos0[:2], direction, half):
                    hits += 1
            return hits

        # Contact height and wrist turn together: the fingertips push the
        # starter's upper part (never its top edge), and the hand body, turned
        # about the vertical, clears every other piece where it comes down and
        # where the push ends. Prefer the lowest contact and the smallest turn
        # with no overlaps; failing that, the fewest overlaps.
        nominal = base_z + 2 * half[2] * (self.CONTACT_FRAC + self._contact_jitter) + self.TIP_BELOW_EEF
        cap = top_z - self.TOP_MARGIN + self.TIP_BELOW_EEF
        heights = np.arange(min(nominal, cap), cap + 1e-9, 0.004)
        best = None
        for contact_z in heights:
            for yaw_deg in self.YAW_CHOICES:
                key = (hits_for(contact_z, np.radians(yaw_deg)), abs(yaw_deg), contact_z)
                if best is None or key < best:
                    best = key
            if best[0] == 0:
                break
        self._yaw_hits, yaw_abs, contact_z = best
        # Recover the signed turn for the chosen height.
        self._yaw = float(min((np.radians(y) for y in self.YAW_CHOICES if abs(y) == yaw_abs),
                              key=lambda y: hits_for(contact_z, y)))
        self._start_pos = np.array([*start_xy, contact_z])
        self._end_pos = np.array([*end_xy, contact_z])
        self._safe_z = max(tops + [contact_z]) + self.CLEAR_ABOVE
        self._R0 = sim.data.body_xmat[sim.model.body_name2id("gripper0_right_eef")].reshape(3, 3).copy()

        print(f"[strike] starter={starter}  route_len={self._route_len}  "
              f"start={np.round(self._start_pos, 3)}  safe_z={self._safe_z:.3f}")

    def _budget(self, env):
        """Pick the travel speed so the push lands by the deadline the cascade needs."""
        limit = getattr(env, "MAX_EPISODE_STEPS", 1000)
        cascade = self.CASCADE_STEPS_PER_PIECE * (self._route_len or 6) + self.CASCADE_MARGIN
        # Descent and push take about this many steps after travel ends.
        fixed = self.DESCEND_STEPS + self.PUSH_STEPS
        travel = int(np.clip(min(self._target_episode_steps, limit - cascade) - fixed, 60, 600))
        self._ctrl_period = int(self.rng.integers(4, 9))
        self._n_interp_travel = int(np.clip(travel // self._ctrl_period, 8, 150))

    def _starter_tilt_deg(self, sim):
        return float(np.degrees(np.arccos(np.clip(sim.data.body_xmat[self._starter_bid][8], -1.0, 1.0))))

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------

    def _p(self, eef, target, gain=8.0):
        a = np.zeros(7); a[6] = 1.0
        a[:3] = np.clip((target - eef) * gain, -1.0, 1.0)
        if self._sim is not None and self._R0 is not None:
            c, s_ = np.cos(self._yaw), np.sin(self._yaw)
            Rt = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ self._R0
            Rc = self._sim.data.body_xmat[self._eef_bid].reshape(3, 3)
            E = Rt @ Rc.T
            ang = np.arccos(np.clip((np.trace(E) - 1) / 2, -1.0, 1.0))
            if ang > 1e-6:
                axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) / (2 * np.sin(ang))
                a[3:6] = np.clip(axis * ang * self.YAW_GAIN, -1.0, 1.0)
        return a

    def act(self, obs, env=None):
        self._ctrl_step += 1
        eef = obs["robot0_eef_pos"]

        if self.phase == "detect":
            sim = env.sim
            self._sim = sim
            self._eef_bid = sim.model.body_name2id("gripper0_right_eef")
            if env._interconnected:
                self._starter, self._route_len = detect_route(
                    sim, env._num_chains, env._dominoes_per_chain, env._table_height)
            else:
                chain = detect_correct_chain(sim, env._num_chains, env._dominoes_per_chain,
                                             table_height=env._table_height)
                self._starter, self._route_len = f"chain{chain}_domino_0", env._dominoes_per_chain
            self._correct_chain = int(self._starter.replace("chain", "").split("_domino_")[0])
            self._build_strike(sim, self._starter, table_height=env._table_height)
            self._budget(env)
            self._lift_target = np.array([eef[0], eef[1], max(eef[2], self._safe_z)])
            self._next("lift")

        if self.phase == "lift":
            if np.linalg.norm(eef - self._lift_target) < 0.02 or self.step_in_phase > 150:
                above = np.array([*self._start_pos[:2], self._safe_z])
                self._travel_waypoints = interpolate_waypoints_curved(
                    self._lift_target, above, self._n_interp_travel, self.rng)
                self._next("travel")
            else:
                self._last_action = self._p(eef, self._lift_target)

        if self.phase == "travel":
            # Curved path well above every piece, followed on a fixed schedule:
            # the reference moves one waypoint every ctrl_period ticks.
            wp = self._travel_waypoints
            i = min(self.step_in_phase // self._ctrl_period, len(wp) - 1)
            self._last_action = self._p(eef, wp[i], 8.0)
            done = self.step_in_phase >= len(wp) * self._ctrl_period
            if done and (np.linalg.norm(eef - wp[-1]) < 0.02 or self.step_in_phase >= len(wp) * self._ctrl_period + 40):
                self._next("descend")

        if self.phase == "descend":
            # Straight down behind the starter, stopping at contact height.
            above = np.array([*self._start_pos[:2], eef[2]])
            over = np.linalg.norm(eef[:2] - self._start_pos[:2]) < 0.006
            self._over = self._over or over or self.step_in_phase > 60
            target = self._start_pos if self._over else above
            self._last_action = self._p(eef, target, 10.0)
            if np.linalg.norm(eef - self._start_pos) < 0.006 or self.step_in_phase > 200:
                self._next("push")

        if self.phase == "push":
            # A brisk strike: drive the fingertips to the stop point at the
            # controller's pace, so the starter leaves with real speed, then
            # pull straight back as soon as they get there.
            aim = self._end_pos + np.array([*self._dir, 0.0]) * 0.02
            self._last_action = self._p(eef, aim, self._strike_gain)
            past = float((eef[:2] - self._centre) @ self._dir)
            if (past >= self._push_limit - 0.002 or self._starter_tilt_deg(env.sim) > self.TIP_DONE_DEG
                    or self.step_in_phase > 150):
                back = eef - np.array([*self._dir, 0.0]) * 0.04
                self._retreat = np.array([back[0], back[1], self._safe_z])
                self._next("retreat")

        if self.phase == "retreat":
            # Back off along the strike line first, then up, away from the falling pieces.
            target = self._retreat if self.step_in_phase > 20 else np.array([*self._retreat[:2], eef[2] + 0.01])
            self._last_action = self._p(eef, target)
            if np.linalg.norm(eef - self._retreat) < 0.02 or self.step_in_phase > 200:
                self._hold = eef.copy()
                self._next("done")

        if self.phase == "done":
            self._last_action = self._p(eef, self._hold)

        self.step_in_phase += 1
        return self._last_action.copy()


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import cv2
    import os
    from pynput import keyboard as kb

    _quit = False

    def on_press(key):
        global _quit
        if hasattr(key, "char") and key.char == "q":
            _quit = True

    listener = kb.Listener(on_press=on_press)
    listener.start()

    env    = ib.make("ib-p3-domino-select-v1", privileged_obs=True, render_mode="human")
    policy = DominoSelectPolicy()
    quit_all = False

    for ep in range(5):
        if quit_all:
            break
        seed = ep * 42 + 7
        obs, info = env.reset(seed=seed)
        policy.reset()
        done   = False
        frames = []
        step   = 0

        print(f"\n=== Episode {ep}  seed={seed} ===")
        print(f"    ground truth correct_chain = {env._correct_chain}")

        while not done:
            if _quit:
                print("  Q pressed — stopping.")
                quit_all = True
                break

            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            step += 1

            frame = None
            for cam_key in ("rgb_exo_front_cam", "rgb_exo_right_cam",
                            "rgb_exo_left_cam", "rgb_gripper_cam"):
                if cam_key in obs and obs[cam_key] is not None:
                    frame = obs[cam_key]
                    break
            if frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            done = terminated or truncated

        if frames:
            os.makedirs("videos", exist_ok=True)
            out_path = f"videos/episode_{ep}.mp4"
            h, w     = frames[0].shape[:2]
            fourcc   = cv2.VideoWriter_fourcc(*"avc1")
            writer   = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            print(f"  saved → {out_path}")

        print(f"Ep {ep}  steps={step}  score={info['task_score']:.3f}  "
              f"phase={policy.phase}  "
              f"detected={policy._correct_chain}  "
              f"truth={env._correct_chain}")

    listener.stop()
    env.close()
