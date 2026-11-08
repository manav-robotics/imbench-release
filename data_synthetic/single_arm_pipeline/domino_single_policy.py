import numpy as np
import imbench as ib
from imbench.policy.base_policy import PolicyInterface
from data_synthetic.episode_length_stats import sample_target_length
from imbench.utils.obs_visualizer import ObsDashboard

STRIDE = 7

# ---------------------------------------------------------------------------
# Human-likeness hyper-parameters
# ---------------------------------------------------------------------------

# Path curvature: peak lateral deviation as a fraction of segment length.
# 0.0 = straight line, 0.12 = gentle arc (human wrist never moves in a line).
CURVE_ARC_FRAC = 0.2

# Overshoot: when approaching a waypoint the controller briefly overshoots by
# this fraction of the remaining distance, then corrects back.
OVERSHOOT_FRAC  = 0.8   # how far past the waypoint to aim on first approach
OVERSHOOT_STEPS = 3      # number of control ticks to hold the overshot target
CORRECTION_GAIN = 0.6    # gain multiplier during the pull-back correction tick

# Episode-length distribution (Domino Chain Trigger): N(mean, std) truncated to [min, max].
# Each reset draws a target step count and picks ctrl_period / n_interp so the
# strike naturally takes that many steps. Faster motion shortens the episode;
# slower motion lengthens it. Phase ordering (close_gripper → goto → push) is
# the same for every target.
EP_LEN_MEAN = 662
EP_LEN_STD  = 280
EP_LEN_MIN  = 252
# Below the teleop maximum (2532): the task budget is 1000 steps and the
# cascade needs another 150-250 after the strike.
EP_LEN_MAX  = 750

# Bounds for ctrl_period when scaling. Outside [5, 50] the OSC controller
# either jitters (too small) or fails to track (too large).
CTRL_PERIOD_MIN = 5
CTRL_PERIOD_MAX = 50

# Bounds for waypoints per segment.
N_INTERP_MIN = 10
N_INTERP_MAX = 80

# Steps spent in close_gripper before the strike starts.
WARMUP_STEPS = 40


# ---------------------------------------------------------------------------
# Object state helpers
# ---------------------------------------------------------------------------

def parse_obj(obs, idx):
    s = obs["object_state"]
    b = idx * STRIDE
    return s[b:b+3].copy(), s[b+3:b+7].copy()

def quat_to_yaw(q):
    return 2.0 * np.arctan2(q[2], q[3])


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
    the midpoint of the trajectory. A random sign is drawn each call so
    the arm sometimes curves left, sometimes right.
    """
    seg      = end - start
    seg_len  = np.linalg.norm(seg)
    perp     = _perp3d(seg)
    arc_sign = rng.choice([-1, 1])
    peak_dev = arc_sign * arc_frac * seg_len

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
# Overshoot tracker (per waypoint index)
# ---------------------------------------------------------------------------

class OvershootTracker:
    """
    Wraps waypoint-following with a brief overshoot + correction pattern
    that mimics human motor control inaccuracy near targets.
    """

    def __init__(self, overshoot_frac=OVERSHOOT_FRAC,
                 overshoot_steps=OVERSHOOT_STEPS):
        self.overshoot_frac  = overshoot_frac
        self.overshoot_steps = overshoot_steps
        self._os_wp          = None
        self._os_ticks       = 0
        self._os_target      = None

    def reset(self):
        self._os_wp     = None
        self._os_ticks  = 0
        self._os_target = None

    def get_target(self, waypoints, wp_idx, eef):
        """
        Returns (target_pos, gain_multiplier).

        gain_multiplier < 1 during correction so the arm gently pulls back,
        giving the characteristic human overshoot-and-settle motion.
        """
        if wp_idx >= len(waypoints):
            return waypoints[-1], 1.0

        nominal = waypoints[wp_idx]
        delta   = nominal - eef
        dist    = np.linalg.norm(delta)

        TRIGGER_DIST = 0.04
        if (dist < TRIGGER_DIST
                and self._os_wp != wp_idx
                and wp_idx < len(waypoints) - 1):
            direction        = delta / (dist + 1e-9)
            overshoot_amount = self.overshoot_frac * dist
            self._os_target  = nominal + direction * overshoot_amount
            self._os_wp      = wp_idx
            self._os_ticks   = 0

        if self._os_wp == wp_idx:
            self._os_ticks += 1
            if self._os_ticks <= self.overshoot_steps:
                return self._os_target, 1.0
            else:
                return nominal, CORRECTION_GAIN

        return nominal, 1.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class DominoPolicy(PolicyInterface):

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase         = "close_gripper"
        self.step_in_phase = 0
        self._start_pos    = None
        self._end_pos      = None

        self._goto_waypoints = []
        self._push_waypoints = []
        self._wp_idx         = 0

        # Overshoot trackers (one per motion segment)
        self._ot_goto = OvershootTracker()
        self._ot_push = OvershootTracker()

        # Sample target episode length, then factor it into ctrl_period and
        # n_interp values whose product (× 2 segments) plus warmup equals the
        # target. Pick ctrl_period first (within its viable range) and derive
        # n_interp to hit the budget; clamp both ends if needed.
        self._target_episode_steps = sample_target_length(
            self.rng, EP_LEN_MEAN, EP_LEN_STD, EP_LEN_MIN, EP_LEN_MAX,
        )
        strike_budget = max(self._target_episode_steps - WARMUP_STEPS, 2 * N_INTERP_MIN * CTRL_PERIOD_MIN)

        cp_lo = max(CTRL_PERIOD_MIN, int(np.ceil(strike_budget / (2 * N_INTERP_MAX))))
        cp_hi = min(CTRL_PERIOD_MAX, int(strike_budget / (2 * N_INTERP_MIN)))
        if cp_hi < cp_lo:
            cp_hi = cp_lo
        self._ctrl_period = int(self.rng.integers(cp_lo, cp_hi + 1))

        n_total = strike_budget // self._ctrl_period
        # Split between goto and push roughly evenly (±25 % each side).
        split = self.rng.uniform(0.4, 0.6)
        self._n_interp_goto = int(np.clip(round(n_total * split), N_INTERP_MIN, N_INTERP_MAX))
        self._n_interp_push = int(np.clip(n_total - self._n_interp_goto, N_INTERP_MIN, N_INTERP_MAX))

        self._last_action    = np.zeros(7)
        self._last_action[6] = 1.0
        self._ctrl_step      = 0

        est_steps = WARMUP_STEPS + (self._n_interp_goto + self._n_interp_push) * self._ctrl_period
        print(f"[reset] target_ep_steps={self._target_episode_steps}  "
              f"ctrl_period={self._ctrl_period}  "
              f"n_interp_goto={self._n_interp_goto}  "
              f"n_interp_push={self._n_interp_push}  "
              f"est_strike_steps≈{est_steps}")

    def _next(self, phase):
        self.phase         = phase
        self.step_in_phase = 0
        self._wp_idx       = 0

    def _build_strike(self, obs):
        d0_pos, d0_quat = parse_obj(obs, 0)
        d1_pos, _       = parse_obj(obs, 1)

        yaw  = quat_to_yaw(d0_quat)
        axis = np.array([np.cos(yaw), np.sin(yaw)])

        to_d1 = d1_pos[:2] - d0_pos[:2]
        if np.dot(axis, to_d1) < 0:
            axis = -axis

        z = 0.91

        BACK = 0.08
        self._start_pos = np.array([
            d0_pos[0] - axis[0] * BACK,
            d0_pos[1] - axis[1] * BACK,
            z,
        ])

        THRU = 0.1
        self._end_pos = np.array([
            d0_pos[0] + axis[0] * THRU,
            d0_pos[1] + axis[1] * THRU,
            z,
        ])

        eef_clean = obs["robot0_eef_pos"]
        self._goto_waypoints = interpolate_waypoints_curved(
            eef_clean, self._start_pos, self._n_interp_goto, self.rng)
        self._push_waypoints = interpolate_waypoints_curved(
            self._start_pos, self._end_pos, self._n_interp_push, self.rng,
            arc_frac=0.05)   # tighter arc for the strike itself

        self._ot_goto.reset()
        self._ot_push.reset()

        print(f"[strike] d0={np.round(d0_pos[:2],4)}  yaw={np.degrees(yaw):.1f}°  "
              f"axis={np.round(axis,3)}")
        print(f"[strike] start={np.round(self._start_pos,4)}")
        print(f"[strike] end  ={np.round(self._end_pos,4)}")

    def act(self, obs, env=None):
        self._ctrl_step += 1

        action    = np.zeros(7)
        action[6] = 1.0
        GAIN      = 5.0

        if self.phase == "close_gripper":
            if self.step_in_phase >= 40:
                self._build_strike(obs)
                self._next("go_to_start")
            self.step_in_phase += 1
            return action

        # On non-control ticks hold the last action
        if self._ctrl_step % self._ctrl_period != 0:
            return self._last_action.copy()

        eef = obs["robot0_eef_pos"]

        if self.phase == "go_to_start":
            target, gain_mult = self._ot_goto.get_target(
                self._goto_waypoints, self._wp_idx, eef)
            delta = target - eef
            action[:3] = np.clip(delta * GAIN * gain_mult, -1.0, 1.0)

            if np.linalg.norm(eef - self._goto_waypoints[
                    min(self._wp_idx, len(self._goto_waypoints) - 1)]) < 0.025:
                self._wp_idx += 1

            if self._wp_idx >= len(self._goto_waypoints):
                self._next("push")

        elif self.phase == "push":
            target, gain_mult = self._ot_push.get_target(
                self._push_waypoints, self._wp_idx, eef)
            delta = target - eef
            action[:3] = np.clip(delta * GAIN * gain_mult, -1.0, 1.0)

            if np.linalg.norm(eef - self._push_waypoints[
                    min(self._wp_idx, len(self._push_waypoints) - 1)]) < 0.025:
                self._wp_idx += 1

            if self._wp_idx >= len(self._push_waypoints):
                self._next("done")

        elif self.phase == "done":
            pass

        self.step_in_phase += 1
        self._last_action = action
        return self._last_action.copy()


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env    = ib.make("ib-p3-domino-single-v1", privileged_obs=True)
    policy = DominoPolicy()
    dash   = ObsDashboard()

    for ep in range(5):

        obs, info = env.reset(seed=ep*47+2)
        policy.reset()
        dash.reset()
        dash.update(obs, info)
        done = False

        print(f"\n=== Episode {ep} ===")
        for i in range(9):
            pos, quat = parse_obj(obs, i)
            print(f"  obj[{i}]  pos={np.round(pos,4)}  yaw={np.degrees(quat_to_yaw(quat)):.1f}°")

        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            key = dash.update(obs, info, reward=reward)
            if key == ord('q'):
                break
            done = terminated or truncated

        print(f"Ep {ep}  score={info['task_score']:.3f}  phase={policy.phase}")

    dash.close()
    env.close()