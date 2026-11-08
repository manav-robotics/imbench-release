"""Scripted catch policy for ``ib-p2-slide-catch-v1``.

The bin starts welded to the gripper (``bin_attached_to_gripper: true``)
and the task forces the gripper command closed every step, so the
policy never grasps -- it only positions the bin in space.

Strategy
--------
The incline is hinged along its low edge, which faces the robot; the task
shifts and yaws the incline per episode, and plays 2-3 cubes. Each cube
slides straight down the slope in its own lane, so the bin must move
along the edge to catch each one in turn. All geometry is read from the
task (privileged: this is a demo collector, not a benchmark entrant).

At every tick:

1. Skip cubes already in the bin or already fallen.
2. Target the next cube: one already sliding, nearest the edge first;
   otherwise the one the task's sliding model says leaves first.
3. Move the bin along the edge to the cube's lane (staying out past the
   edge), then out to the catch point just past the edge, below the
   table top (``task._catch_point``).

``max_action`` caps every motion channel (0.3 matches the gamepad's
default teleop speed) and ``obs_delay_steps`` makes the policy react to
cube positions that many steps old (a human's reaction time), so
together they check that a seed can be caught by teleoperation.

The gripper-to-bin offset is read live from ``bin_pos - eef_pos`` so
the policy is robust to handle slack and small swings of the welded
bin during fast motions.

Phase machine:
  settle -> catch -> done
"""

from __future__ import annotations

import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface


# ── Geometry constants ──────────────────────────────────────────────
TOP_DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw

# Where the bin waits for a cube comes from the task: ``_catch_point``
# (``feasibility.catch_out`` past the edge, ``catch_below`` under the
# table top), the same point the task's teleop feasibility check uses.

# Where along the bin (m from its centre, along the edge) the 1st, 2nd and
# 3rd caught cube lands, so caught cubes do not pile up.
CATCH_SPREAD = (-0.03, 0.03, 0.0)

# Fallback gripper-to-bin vertical offset, in case bin_pos isn't in
# the obs for some reason. The bin sits ~h below the gripper site,
# where h = bin_wall (0.012) + bin_wall_height (0.05) +
# bin_handle_height/2 (0.025) = 0.087 m. Used only as a safety net.
FALLBACK_BIN_BELOW_GRIPPER = 0.087

# ── Phase budgets ────────────────────────────────────────────────────
# Sampled-horizon distribution; only the settle phase uses it (the catch
# phase runs until every cube is caught or gone). Episode length is 3000
# ticks @ 100 Hz (30 s) per the task YAML.
SETTLE_TICKS = 20   # 0.2 s: cubes may leave the plate ~4 s in
CATCH_FRAC  = 0.95   # consumes the rest -- catch is the whole task

HORIZON_MEAN = 1800
HORIZON_STD  = 200
HORIZON_MIN  = 1200
HORIZON_MAX  = 2800   # task max_episode_steps is 3000, leave slack

# ── Tolerances ──────────────────────────────────────────────────────
CATCH_XY_TOL = 0.020   # bin within 2 cm of target xy is "ready"

# ── Position-control gains ──────────────────────────────────────────
GAIN_XY  = 10.0   # high: the bin must arrive fast (cubes leave as early as ~4 s)
GAIN_Z   = 10.0
ROT_GAIN = 2.0
MAX_ACTION_STEP = 0.5

# Setpoint-change ramp (smoothstep gain on freshly published targets).
RAMP_TICKS        = 15
RAMP_START_FRAC   = 0.25
TARGET_CHANGE_TOL = 0.020


# ── Quaternion helpers (xyzw) ───────────────────────────────────────

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

class SlideCatchPolicy(PolicyInterface):
    """Scripted policy that positions the welded bin under the
    next-falling cube. The gripper command is ignored by the task
    (forced closed), so the policy only servos linear/angular pose."""

    def __init__(self, rng=None, max_action=MAX_ACTION_STEP, obs_delay_steps=0):
        self.rng = np.random.default_rng() if rng is None else rng
        self.max_action = float(max_action)
        self.obs_delay_steps = int(obs_delay_steps)
        self.reset()

    def reset(self):
        self.phase = "settle"
        self.step_in_phase = 0

        self._table_z: float = 0.8
        self._num_cubes: int = 0
        self._bin_inner_half_x: float = 0.06
        self._bin_inner_half_y: float = 0.06

        # Sample horizon, slice into per-phase budgets.
        horizon = float(self.rng.normal(HORIZON_MEAN, HORIZON_STD))
        self.horizon = int(np.clip(round(horizon), HORIZON_MIN, HORIZON_MAX))
        self.budgets = {
            "settle": SETTLE_TICKS,
            "catch":  max(1, int(round(CATCH_FRAC  * self.horizon))),
        }
        print(f"  [horizon] sampled={self.horizon}  budgets={self.budgets}",
              flush=True)

        # Cube tracking state.
        self._cube_names: list[str] = []
        self._cube_body_ids: list[int] = []
        self._bin_body_id: int | None = None
        self._caught: set[str] = set()
        self._last_target_y: float | None = None
        self._cube_history: list[list[np.ndarray]] = []
        self._task = None

        # Setpoint-ramp state.
        self._last_target_pos: np.ndarray | None = None
        self._ticks_since_target_change: int = 10**6

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

    def _move_to(self, eef_pos, eef_quat, target_pos,
                 gain_xy=GAIN_XY, gain_z=GAIN_Z, max_step=None):
        max_step = self.max_action if max_step is None else max_step
        ramp = self._ramp_factor(target_pos)
        action = np.zeros(7)
        delta = target_pos - eef_pos
        action[0] = float(np.clip(delta[0] * gain_xy * ramp, -max_step, max_step))
        action[1] = float(np.clip(delta[1] * gain_xy * ramp, -max_step, max_step))
        action[2] = float(np.clip(delta[2] * gain_z  * ramp, -max_step, max_step))
        ori_err = _quat_err_axis_angle(eef_quat, TOP_DOWN_QUAT)
        action[3:6] = np.clip(ori_err * ROT_GAIN, -0.3, 0.3)
        # Gripper command is forced closed by the task; send +1.0 to
        # match what the task overwrites.
        action[6] = 1.0
        return action

    # ── Cube bookkeeping ─────────────────────────────────────────────

    def _discover_cubes(self, env) -> None:
        """Cache the cubes in play, the bin and task constants on the first
        call after reset(). Reads the task directly (privileged)."""
        if self._cube_names or env is None:
            return
        task = env.unwrapped if hasattr(env, "unwrapped") else env
        self._task = task
        sim = task._env.sim
        names = list(getattr(task, "_active_cubes", ["cube"]))
        self._cube_names = names
        self._cube_body_ids = [int(sim.model.body_name2id(f"{n}_main")) for n in names]
        self._num_cubes = len(names)
        self._bin_body_id = None
        for cand in ("bin_root", "bin_main"):
            try:
                self._bin_body_id = int(sim.model.body_name2id(cand))
                break
            except Exception:
                continue
        self._table_z = float(getattr(task, "_table_height", self._table_z))
        self._bin_inner_half_x = float(getattr(task, "_bin_inner_half_x", 0.06))
        self._bin_inner_half_y = float(getattr(task, "_bin_inner_half_y", 0.06))
        deps = getattr(task, "_departures", None) or [{"depart": None}] * len(names)
        self._predicted = [d["depart"] if d["depart"] is not None else 1e9 for d in deps]
        print(f"  [discover] cubes={names}  bin_bid={self._bin_body_id}  "
              f"yaw={np.rad2deg(getattr(task, '_incline_yaw', 0.0)):.1f}deg", flush=True)

    # ── Incline-edge frame ──────────────────────────────────────────

    def _edge_frame(self, p):
        """World point -> (u along the edge, w out past the edge toward the robot)."""
        t = self._task
        yaw = float(getattr(t, "_incline_yaw", 0.0))
        c, s = np.cos(yaw), np.sin(yaw)
        d = np.asarray(p[:2]) - np.asarray(t._anchor_world_pos[:2])
        return -s * d[0] + c * d[1], -(c * d[0] + s * d[1])

    def _cube_pos(self, env, idx: int) -> np.ndarray:
        """Cube position as the policy perceives it (``obs_delay_steps`` old)."""
        hist = self._cube_history
        k = max(0, len(hist) - 1 - self.obs_delay_steps)
        return hist[k][idx].copy() if hist else np.zeros(3)

    def _record_cubes(self) -> None:
        sim = self._task._env.sim
        self._cube_history.append([np.array(sim.data.body_xpos[b], dtype=float)
                                   for b in self._cube_body_ids])
        if len(self._cube_history) > self.obs_delay_steps + 2:
            self._cube_history.pop(0)

    def _bin_pos(self, env, fallback: np.ndarray) -> np.ndarray:
        if self._bin_body_id is None:
            return fallback
        sim = self._task._env.sim
        return np.array(sim.data.body_xpos[self._bin_body_id], dtype=float)

    def _cube_in_bin(self, cube_pos: np.ndarray, bin_pos: np.ndarray) -> bool:
        dx = abs(cube_pos[0] - bin_pos[0])
        dy = abs(cube_pos[1] - bin_pos[1])
        if dx > self._bin_inner_half_x or dy > self._bin_inner_half_y:
            return False
        # Cube must be at or above the bin's floor (allow 2 cm slack).
        if cube_pos[2] < bin_pos[2] - 0.02:
            return False
        return True

    def _select_target_cube(self, env, bin_pos: np.ndarray):
        """The cube to catch next: among cubes not yet caught or fallen, one
        already sliding (nearest the edge first), else the one predicted
        to leave the plate first."""
        sliding, waiting = [], []
        for i, name in enumerate(self._cube_names):
            if name in self._caught:
                continue
            cube_pos = self._cube_pos(env, i)
            if cube_pos[2] < self._table_z - 0.10:
                continue  # already fell past the bin
            if self._cube_in_bin(cube_pos, bin_pos):
                self._caught.add(name)
                print(f"  [caught] {name}", flush=True)
                continue
            hist = self._cube_history
            prev = hist[max(0, len(hist) - 2 - self.obs_delay_steps)][i] if len(hist) > 1 else cube_pos
            speed = float(np.linalg.norm(cube_pos - prev)) * 100.0
            _, w = self._edge_frame(cube_pos)
            if speed > 0.05:
                sliding.append((-w, i))      # -w: distance up-slope from the edge
            else:
                waiting.append((self._predicted[i], i))
        if sliding:
            return min(sliding)[1]
        if waiting:
            return min(waiting)[1]
        return None

    # ── Main act() ────────────────────────────────────────────────────

    def act(self, obs: dict, env=None) -> np.ndarray:
        self._discover_cubes(env)
        if self._task is not None:
            self._record_cubes()

        eef_pos  = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.asarray(obs.get("robot0_eef_quat",
                                      np.array([0.0, 0.0, 0.0, 1.0])))
        # Read bin position straight from the sim. Fallback assumes the
        # bin hangs below the gripper at the welded offset.
        bin_pos = self._bin_pos(
            env,
            fallback=eef_pos + np.array([0.0, 0.0, -FALLBACK_BIN_BELOW_GRIPPER]),
        ) if env is not None else (
            eef_pos + np.array([0.0, 0.0, -FALLBACK_BIN_BELOW_GRIPPER])
        )

        # Live offset from gripper to bin (world-frame). Used to convert
        # a desired bin position into a gripper setpoint.
        gripper_to_bin = bin_pos - eef_pos

        # ── 1. Settle: let bin attachment + initial transients dissipate ─
        if self.phase == "settle":
            # Hold pose: target = current eef.
            action = self._move_to(eef_pos, eef_quat, eef_pos.copy())
            if self.step_in_phase >= self.budgets["settle"]:
                self._advance("catch")
            self.step_in_phase += 1
            return action

        # ── 2. Catch: track the next-imminent cube ───────────────────
        if self.phase == "catch":
            target_idx = self._select_target_cube(env, bin_pos)

            if target_idx is None:
                # All cubes either caught or fallen off -- hold the
                # current position and wait out the episode.
                if len(self._caught) == self._num_cubes:
                    self._advance("done")
                action = self._move_to(eef_pos, eef_quat, eef_pos.copy())
                self.step_in_phase += 1
                return action

            cube_pos = self._cube_pos(env, target_idx)

            # ── Head for the catch point ──────────────────────────
            # Each catch lands at a different spot along the edge inside
            # the bin (CATCH_SPREAD), so caught cubes lie side by side
            # instead of piling up. While the bin's rim is above the table
            # top it stays the approach clearance out past the edge line:
            # a yawed incline's hinge line overhangs the table edge, and
            # the bin must not come down through it.
            lane, _ = self._edge_frame(cube_pos)
            yaw = float(getattr(self._task, "_incline_yaw", 0.0))
            along = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
            out = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])
            spread = CATCH_SPREAD[min(len(self._caught), len(CATCH_SPREAD) - 1)]
            bin_target = np.asarray(self._task._catch_point(lane), dtype=float) - spread * along
            rim_z = bin_pos[2] + self._task._bin_wall + self._task._bin_wall_height
            if rim_z > self._table_z - 0.005:
                # Above the table top: never move toward the edge, and stay
                # at least the approach clearance out; if starting closer,
                # first move out, holding height.
                _, w_bin = self._edge_frame(bin_pos)
                _, w_tgt = self._edge_frame(bin_target)
                w_safe = self._task._approach_clearance
                w_hold = max(w_bin, w_safe)
                if w_tgt < w_hold:
                    bin_target = bin_target + (w_hold - w_tgt) * out
                if w_bin < w_safe - 0.005:
                    bin_target[2] = bin_pos[2]

            # Convert bin target to gripper setpoint via the live offset.
            gripper_target = bin_target - gripper_to_bin

            action = self._move_to(eef_pos, eef_quat, gripper_target)
            # No catch budget: cubes may leave the plate as late as the
            # task's latest_departure_s, so keep catching until each cube
            # is caught or gone.
            self.step_in_phase += 1
            return action

        # ── 3. Done: hold pose ───────────────────────────────────────
        action = self._move_to(eef_pos, eef_quat, eef_pos.copy())
        self.step_in_phase += 1
        return action


# ────────────────────────────────────────────────────────────────────
# Stand-alone runner
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env    = ib.make("ib-p2-slide-catch-v1", privileged_obs=True, render_mode=render_mode)
    policy = SlideCatchPolicy()

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep * 17)
        policy.reset()
        done = False
        print(f"\n=== Episode {ep} ===", flush=True)
        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            if not args.no_render:
                env.render()
            done = terminated or truncated
        print(f"Episode {ep}  score={info.get('task_score', 0.0):.3f}  "
              f"phase={policy.phase}  caught={len(policy._caught)}/"
              f"{policy._num_cubes}", flush=True)
    env.close()
