"""Scripted typing policy for ``ib-m-keyboard-typing-v1``.

Action layout (homogeneous bimanual): ``[arm0(7), arm1(7)]`` — 14 dims.

The task already runs its OWN scripted policy for arm 0 inside
``KeyboardTypingV1.step()``: ``action[:7]`` is overwritten with the
task's internal FSM, which types arm 0's letters on its turns. So this
external policy only needs to drive **arm 1** through its letters.

Turn schedule (matching the task): the word is typed left to right, and
each letter by the arm that owns it. While the next letter is arm 0's,
arm 1 waits at home; pressing any key out of turn is a typo that fails
the episode.

Per-arm-1 FSM:
  wait_turn (hold home until the next letter is arm 1's)
    -> handoff_settle (100 zero-action steps so arm 0 comes fully to
                       rest before arm 1 starts contributing forces;
                       skipped when arm 0 has not typed since arm 1's
                       last letter)
    -> for each consecutive arm-1 letter:
         lift_to_traverse  (rectilinear +z)
      -> traverse_xy       (curved cosine-eased waypoints at constant z,
                            with a perpendicular bell-shaped arc -- same
                            trick as the domino policies)
      -> descend_to_press  (rectilinear -z past the key surface; physics
                            blocks the EEF and detection flips)
      -> lift_off_key      (rectilinear +z)
    -> retreat (back to the reset joint pose)
    -> wait_turn (at home) ... -> done once the word is typed

So each transition between keys is decomposed into orthogonal +z /
xy / -z legs rather than a single diagonal sweep, and the xy leg
itself is a curved arc rather than a straight chord.

The retreat is in joint space. Pressing a far key (e.g. bottom-row N
with the keyboard jittered away) locks arm 1's elbow straight against
its joint limit. A Cartesian move home from there folds the wrist
instead of bending the elbow, which left the wrist resting on the
keyboard and against arm 0's fingers, so arm 0 could not reach its next
key. The retreat sends the EEF twist that drives the joints back to
their reset pose.

OSC actions are normalised to ``[-1, 1]`` per axis with the standard
robosuite output_max of 0.05 m/unit (position) and 0.5 rad/unit
(rotation).
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# OSC scaling: action * 0.05 = metres/step, action * 0.5 = rad/step.
_OSC_POS_SCALE = 0.05
_OSC_ROT_SCALE = 0.5

# Geometry constants (world-frame; matches the task's internal FSM).
HOVER_HEIGHT_ABOVE_KEY = 0.10   # m
PRESS_DEPTH_BELOW_KEY  = 0.05   # m below key top — physics blocks the EEF
HOVER_REACHED_TOL      = 0.008  # m
LIFT_REACHED_TOL       = 0.012  # m
HOME_REACHED_TOL       = 0.015  # m

# Action clip for traverse/lift (fast) and press (slow).
ACTION_CLIP_FAST = 0.6
ACTION_CLIP_SLOW = 0.3

# Joint-space retreat to the reset pose after arm 1's last letter of a turn.
ACTION_CLIP_RETREAT = 1.0
RETREAT_JOINT_STEP  = 0.3    # rad -- max joint-space offset per step
RETREAT_REACHED_TOL = 0.15   # rad -- joint-space distance to reset pose
RETREAT_MAX_STEPS   = 800

# Settle pause after arm 0 types, before arm 1 starts moving.
# Lets the first arm come to rest, dampens any residual oscillation,
# and gives the recorded data a clean handoff.
ARM0_HANDOFF_SETTLE_STEPS = 100

# ── Curved traverse parameters ──────────────────────────────────────
# Each lateral move between two hover positions is a *rectilinear*
# motion broken into two legs:
#   1) lift straight up to TRAVERSE_HEIGHT_ABOVE_KEY above the current
#      key (rectilinear -- pure +z motion).
#   2) curved XY traverse at constant z to above the next key (the
#      cosine-eased path with a perpendicular arc, mimicking natural
#      wrist curvature, as in the domino policies).
#   3) descend straight down to the press target (rectilinear -- pure
#      -z motion).
# Plus the press itself is straight down. So the full per-letter cycle
# is: lift -> curved-traverse-xy -> descend -> press -> lift -> ...
TRAVERSE_HEIGHT_ABOVE_KEY = 0.12   # higher than HOVER for safety clearance
CURVE_ARC_FRAC            = 0.18   # peak lateral deviation as a fraction
                                    # of the chord length
TRAVERSE_WAYPOINTS_MIN    = 18      # per-segment waypoint count is sampled
TRAVERSE_WAYPOINTS_MAX    = 32      # uniformly in [min, max] each traverse
WAYPOINT_REACHED_TOL      = 0.015   # m -- advance to next waypoint
WAYPOINT_FINAL_TOL        = 0.010   # m -- last waypoint of a leg is tighter

# ── Episode-length distribution (Keyboard Typing Bimanual teleop) ──
# At reset we draw a target horizon and stretch the per-letter cycle
# (waypoint counts, settle pause) so the realised episode length tracks
# this distribution across many runs.
HORIZON_MEAN     = 2799
HORIZON_STD      = 631
HORIZON_MIN      = 1951
HORIZON_MAX      = 6812
# Empirical horizon at scale=1.0 (baseline waypoint counts and settle).
# Tuned so scale=1.0 produces ~HORIZON_MEAN.
BASELINE_HORIZON = 2799


# ── Curved waypoint generation ──────────────────────────────────────
# Same as the domino policies: cosine-eased chord with a bell-shaped
# perpendicular arc in the XY plane so the trajectory doesn't read as a
# pure straight line.

def _perp_xy(direction: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to ``direction`` in the world XY plane."""
    dx, dy, _ = direction
    length_xy = float(np.sqrt(dx * dx + dy * dy))
    if length_xy < 1e-6:
        return np.array([1.0, 0.0, 0.0])
    return np.array([-dy / length_xy, dx / length_xy, 0.0])


def _curved_waypoints(start: np.ndarray, end: np.ndarray,
                      n_steps: int, rng,
                      arc_frac: float = CURVE_ARC_FRAC) -> list[np.ndarray]:
    """Cosine-eased waypoints from start to end with a perpendicular arc.

    The peak lateral deviation is ``arc_frac * |end - start|`` and is
    placed at the midpoint of the trajectory. A random sign is drawn so
    the arm sometimes curves left, sometimes right.
    """
    seg     = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    seg_len = float(np.linalg.norm(seg))
    perp    = _perp_xy(seg)
    arc_sign = float(rng.choice([-1, 1]))
    peak_dev = arc_sign * arc_frac * seg_len

    pts: list[np.ndarray] = []
    for i in range(n_steps):
        t = (i + 1) / n_steps
        t_smooth = 0.5 * (1.0 - np.cos(np.pi * t))   # cosine ease
        chord_pt = start + t_smooth * seg
        # Bell-shaped lateral arc (sin(pi t) peaks at t=0.5, zero at ends).
        lateral_offset = peak_dev * np.sin(np.pi * t)
        pts.append(chord_pt + lateral_offset * perp)
    return pts


class KeyboardTypingPolicy(PolicyInterface):
    """Scripted arm-1 driver for the keyboard typing task.

    Arm 0 is owned by the task's internal FSM; this policy emits a
    14-dim action where the first 7 dims are zeros (the task overwrites
    them anyway) and the last 7 dims drive arm 1 through its assigned
    letters, each on its turn in the word.
    """

    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    def reset(self):
        self.phase: str = "wait_turn"
        self.step_in_phase: int = 0

        # Cached at first act() call.
        self._cached: bool = False
        self._word: str = ""
        self._arm1_indices: list[int] = []
        self._press_hold_steps: int = 3
        self._key_positions: dict[str, tuple[float, float]] = {}
        self._key_top_z_local: float = 0.024
        self._keyboard_world_pos: np.ndarray | None = None
        self._robot1_base_R: np.ndarray = np.eye(3)

        # Per-arm-1 state.
        self._target_idx: int | None = None  # word position being typed
        self._arm1_home_pos: np.ndarray | None = None
        self._arm1_home_qpos: np.ndarray | None = None
        self._arm1_qpos_idx: list[int] = []
        self._arm1_qvel_idx: list[int] = []
        self._arm1_eef_site: str = ""

        # Curved-traverse state. Populated when entering ``traverse_xy``.
        self._traverse_waypoints: list[np.ndarray] = []
        self._traverse_idx: int = 0

        # Per-episode time scale: stretch arm-1 waypoint counts and the
        # arm-0 -> arm-1 settle pause so the realised total episode length
        # tracks the teleop horizon distribution across many runs.
        sampled = float(self.rng.normal(HORIZON_MEAN, HORIZON_STD))
        self._target_horizon = int(np.clip(sampled, HORIZON_MIN, HORIZON_MAX))
        self._time_scale = self._target_horizon / float(BASELINE_HORIZON)
        self._scaled_settle_steps = max(
            1, int(round(ARM0_HANDOFF_SETTLE_STEPS * self._time_scale))
        )
        self._scaled_wp_min = max(
            2, int(round(TRAVERSE_WAYPOINTS_MIN * self._time_scale))
        )
        self._scaled_wp_max = max(
            self._scaled_wp_min + 1,
            int(round(TRAVERSE_WAYPOINTS_MAX * self._time_scale)),
        )
        print(f"  [policy] target_horizon={self._target_horizon}  "
              f"time_scale={self._time_scale:.2f}  "
              f"settle={self._scaled_settle_steps}  "
              f"wp_range=[{self._scaled_wp_min},{self._scaled_wp_max}]",
              flush=True)

    def _advance(self, next_phase: str):
        print(f"  [arm1 {self.step_in_phase:4d} ticks] {self.phase} -> {next_phase}",
              flush=True)
        self.phase = next_phase
        self.step_in_phase = 0

    # ── Lazy env probe ───────────────────────────────────────────────

    def _cache_from_env(self, env, obs) -> None:
        if self._cached or env is None:
            return
        # Pull the task-level constants from the env. These are set up
        # by KeyboardTypingV1.reset().
        self._word = str(getattr(env, "_word", ""))
        self._arm1_indices = list(getattr(env, "_arm1_indices", []))
        self._press_hold_steps = int(getattr(env, "_press_hold_steps", 3))
        self._key_top_z_local = float(getattr(env, "_KEY_TOP_Z", 0.024))
        # The key-position dict is a module-level constant; pull it
        # from the task module directly so this policy stays in sync if
        # the layout ever changes.
        from imbench.tasks.hobi import keyboard_typing_v1 as _ktv1
        self._key_positions = dict(_ktv1._KEY_POSITIONS)
        kbd = getattr(env, "_keyboard_world_pos", None)
        if kbd is not None:
            self._keyboard_world_pos = np.asarray(kbd, dtype=float).copy()

        # Robot 1 base rotation -- needed to convert world deltas into
        # robot-1-base-frame deltas before sending to the OSC.
        sim = env._env.sim
        try:
            r1_bid = sim.model.body_name2id("robot1_base")
            quat_wxyz = sim.model.body_quat[r1_bid].copy()
            # Convert wxyz -> xyzw and build rotation matrix.
            w, x, y, z = quat_wxyz
            # 3x3 rotation matrix from quat (xyzw).
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            wx, wy, wz = w * x, w * y, w * z
            R = np.array([
                [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
                [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
                [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
            ])
            self._robot1_base_R = R
        except Exception:
            self._robot1_base_R = np.eye(3)

        # Arm 1's joint indices, EEF site and reset joint pose, for the
        # joint-space retreat.
        robot1 = env._env.robots[1]
        self._arm1_qpos_idx = list(robot1._ref_joint_pos_indexes)
        self._arm1_qvel_idx = list(robot1._ref_joint_vel_indexes)
        self._arm1_eef_site = sim.model.site_id2name(robot1.eef_site_id["right"])
        self._arm1_home_qpos = sim.data.qpos[self._arm1_qpos_idx].copy()

        # Capture arm 1's home (current EEF) so we can return to it
        # after the last letter.
        eef1 = obs.get("robot1_eef_pos")
        if eef1 is not None:
            self._arm1_home_pos = np.asarray(eef1, dtype=float).copy()

        self._cached = True
        print(f"  [policy] word={self._word!r}  arm1_indices={self._arm1_indices}  "
              f"hold_steps={self._press_hold_steps}", flush=True)

    # ── Geometry helpers ─────────────────────────────────────────────

    def _key_world_xyz(self, letter: str) -> np.ndarray:
        lx, ly = self._key_positions[letter.upper()]
        kp = self._keyboard_world_pos
        return np.array([
            kp[0] + lx,
            kp[1] + ly,
            kp[2] + self._key_top_z_local,
        ])

    def _world_to_arm1_action(self, world_delta: np.ndarray,
                              clip: float) -> np.ndarray:
        """Convert a world-frame delta into a robot-1-base-frame OSC action."""
        delta_base = self._robot1_base_R.T @ world_delta
        norm_delta = delta_base / _OSC_POS_SCALE
        return np.clip(norm_delta, -clip, clip)

    # ── Phase helpers ────────────────────────────────────────────────

    def _begin_traverse(self, eef_pos: np.ndarray, key_xyz: np.ndarray) -> None:
        """Sample a curved XY path from the current EEF (held at the
        traverse altitude) to a point above the next key."""
        traverse_z = key_xyz[2] + TRAVERSE_HEIGHT_ABOVE_KEY
        start = np.array([eef_pos[0], eef_pos[1], traverse_z])
        end   = np.array([key_xyz[0], key_xyz[1], traverse_z])
        n = int(self.rng.integers(self._scaled_wp_min,
                                  self._scaled_wp_max + 1))
        self._traverse_waypoints = _curved_waypoints(start, end, n, self.rng)
        self._traverse_idx = 0

    # ── Main act() ───────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        # 14-dim bimanual action: first 7 dims are arm 0 (overwritten
        # by the task's internal scripted FSM); last 7 dims drive arm 1.
        action = np.zeros(14, dtype=np.float64)
        # Keep both grippers closed (task expects fingertip presses).
        action[6] = 1.0
        action[13] = 1.0

        self._cache_from_env(env, obs)
        if env is None:
            return action

        typed_mask = list(getattr(env, "_typed_mask", []))
        next_idx = env._next_word_idx()
        my_turn = next_idx is not None and next_idx in self._arm1_indices

        eef_pos = np.asarray(obs.get("robot1_eef_pos",
                                     np.array([0.0, 0.0, 0.0])),
                             dtype=float)

        # ── wait_turn: hold home while arm 0 types ───────────────────
        if self.phase in ("wait_turn", "done"):
            if next_idx is None:
                if self.phase != "done":
                    self._advance("done")
            elif my_turn:
                # Let arm 0 come to rest if it typed the previous letter.
                arm0_typed_last = next_idx > 0 and next_idx - 1 not in self._arm1_indices
                self._target_idx = next_idx
                self._advance("handoff_settle" if arm0_typed_last else "lift_to_traverse")
            if self.phase in ("wait_turn", "done"):
                self._hold_arm1_home(action, obs)
                self.step_in_phase += 1
                return action

        # ── handoff_settle: settle pause -- let arm 0 come to rest ───
        # Send literal zero motion so any residual oscillation damps out
        # before arm 1 starts moving and adding contact forces. This is
        # also a clean break in the recorded trajectory.
        if self.phase == "handoff_settle":
            if self.step_in_phase >= self._scaled_settle_steps:
                self._advance("lift_to_traverse")
            self.step_in_phase += 1
            return action

        target_word_idx = self._target_idx
        key_xyz = self._key_world_xyz(self._word[target_word_idx])

        # ── lift_to_traverse: rectilinear +z lift from current XY ────
        # Keep XY pinned to the current EEF XY; only move along Z. After
        # reaching TRAVERSE_HEIGHT_ABOVE_KEY (relative to the next key
        # height -- they're all the same), build the curved XY path.
        if self.phase == "lift_to_traverse":
            traverse_z = key_xyz[2] + TRAVERSE_HEIGHT_ABOVE_KEY
            target = np.array([eef_pos[0], eef_pos[1], traverse_z])
            delta = target - eef_pos
            if abs(delta[2]) < LIFT_REACHED_TOL:
                self._begin_traverse(eef_pos, key_xyz)
                self._advance("traverse_xy")
            else:
                action[7:10] = self._world_to_arm1_action(delta, ACTION_CLIP_FAST)
            self.step_in_phase += 1
            return action

        # ── traverse_xy: follow the curved waypoint list at constant z ─
        if self.phase == "traverse_xy":
            if self._traverse_idx >= len(self._traverse_waypoints):
                self._advance("descend_to_press")
                self.step_in_phase += 1
                return action

            wp = self._traverse_waypoints[self._traverse_idx]
            delta = wp - eef_pos
            # Use the tighter tol on the final waypoint so the descent
            # starts directly above the key.
            is_last = self._traverse_idx == len(self._traverse_waypoints) - 1
            tol = WAYPOINT_FINAL_TOL if is_last else WAYPOINT_REACHED_TOL
            if float(np.linalg.norm(delta[:2])) < tol:
                self._traverse_idx += 1
            action[7:10] = self._world_to_arm1_action(delta, ACTION_CLIP_FAST)
            self.step_in_phase += 1
            return action

        # ── descend_to_press: rectilinear -z all the way past the key ─
        # A single straight downward motion past the key surface
        # (no separate hover). Physics blocks the EEF
        # when the fingertip contacts the key, creating the press; the
        # task detection then flips ``_typed_mask`` and we advance.
        if self.phase == "descend_to_press":
            press_target = key_xyz.copy()
            press_target[2] -= PRESS_DEPTH_BELOW_KEY
            # Pin XY so the descent is purely along Z (rectilinear).
            press_target[0] = key_xyz[0]
            press_target[1] = key_xyz[1]
            delta = press_target - eef_pos
            if typed_mask[target_word_idx]:
                self._advance("lift_off_key")
            else:
                # XY correction stays slow (small) and Z is the slow press.
                action[7:10] = self._world_to_arm1_action(delta, ACTION_CLIP_SLOW)
            self.step_in_phase += 1
            return action

        # ── lift_off_key: rectilinear +z back to traverse height ─────
        # Then straight on to the next letter if it is arm 1's too, else
        # retreat home to wait for arm 0.
        if self.phase == "lift_off_key":
            traverse_z = key_xyz[2] + TRAVERSE_HEIGHT_ABOVE_KEY
            target = np.array([eef_pos[0], eef_pos[1], traverse_z])
            delta = target - eef_pos
            if abs(delta[2]) < LIFT_REACHED_TOL:
                if my_turn:
                    self._target_idx = next_idx
                    self._advance("lift_to_traverse")
                else:
                    self._advance("retreat")
            else:
                action[7:10] = self._world_to_arm1_action(delta, ACTION_CLIP_FAST)
            self.step_in_phase += 1
            return action

        # ── retreat: back to the reset joint pose, then wait at home ──
        if self.phase == "retreat":
            if not self._retreat_arm1_joints(action, env):
                self._advance("wait_turn" if next_idx is not None else "done")
            self.step_in_phase += 1
            return action

        self.step_in_phase += 1
        return action

    def _retreat_arm1_joints(self, action: np.ndarray, env) -> bool:
        """Write into ``action[7:13]`` the EEF twist (position and
        rotation) that moves arm 1's joints toward their reset pose.
        Returns False once the joints are there or the retreat times out."""
        sim = env._env.sim
        dq = self._arm1_home_qpos - sim.data.qpos[self._arm1_qpos_idx]
        dist = float(np.linalg.norm(dq))
        if dist < RETREAT_REACHED_TOL or self.step_in_phase >= RETREAT_MAX_STEPS:
            return False
        dq *= min(1.0, RETREAT_JOINT_STEP / dist)
        jacp = np.asarray(sim.data.get_site_jacp(self._arm1_eef_site)).reshape(3, -1)
        jacr = np.asarray(sim.data.get_site_jacr(self._arm1_eef_site)).reshape(3, -1)
        R_T = self._robot1_base_R.T
        twist = np.concatenate([
            R_T @ (jacp[:, self._arm1_qvel_idx] @ dq) / _OSC_POS_SCALE,
            R_T @ (jacr[:, self._arm1_qvel_idx] @ dq) / _OSC_ROT_SCALE,
        ])
        # Scale uniformly so clipping keeps the twist's direction.
        peak = float(np.max(np.abs(twist)))
        if peak > ACTION_CLIP_RETREAT:
            twist *= ACTION_CLIP_RETREAT / peak
        action[7:13] = twist
        return True

    def _hold_arm1_home(self, action: np.ndarray, obs: dict) -> None:
        """Drive arm 1 toward its captured home pose. Writes into
        ``action[7:10]`` in-place."""
        if self._arm1_home_pos is None:
            return
        eef_pos = np.asarray(obs.get("robot1_eef_pos",
                                     np.array([0.0, 0.0, 0.0])),
                             dtype=float)
        delta = self._arm1_home_pos - eef_pos
        if float(np.linalg.norm(delta)) < HOME_REACHED_TOL:
            return
        action[7:10] = self._world_to_arm1_action(delta, ACTION_CLIP_FAST)


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env    = ib.make("ib-m-keyboard-typing-v1", privileged_obs=True, render_mode=render_mode)
    policy = KeyboardTypingPolicy()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 17)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} === word={info.get('word')}", flush=True)
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            done = terminated or truncated
        print(f"Episode {ep}  score={info.get('task_score', 0.0):.3f}  "
              f"typed={info.get('typed', '')!r}  phase={policy.phase}", flush=True)
    env.close()
