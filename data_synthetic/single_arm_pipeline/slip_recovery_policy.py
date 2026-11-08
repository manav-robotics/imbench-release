"""Pick-and-place with straight-line waypoints between phase endpoints."""

import numpy as np
import imbench as ib

YAW_ALIGN_OBJECTS = {
    "gelatin_box",
    "potted_meat_can",
    "banana",
    "a_lego_duplo",
    "rubiks_cube"
}
YAW_EXTRA_OFFSET = {
    "gelatin_box":     np.pi / 2,
    "potted_meat_can": np.pi / 2,
    "banana":          np.pi / 2,
    "a_lego_duplo":    np.pi / 2,
    "rubiks_cube":     0.0
}

GELATIN_STANDING_QUAT_XYZW = np.array([-0.42059243, -0.56842062, 0.42059243, -0.56842062])

HOVER_ABOVE_OBJ       = 0.1
TRANSPORT_ABOVE_TABLE = 0.1
XY_TOL       = 0.013
Z_TOL        = 0.010
GAIN_XY      = 6.5
GAIN_Z       = 6.5
GAIN_FAST    = 10.0
YAW_GAIN     = 1.2
YAW_TOL      = 0.05
OPEN_HOLD    = 20
CLOSE_HOLD   = 40
LOWER_BUDGET = 180
RELEASE_HOLD = 40

ON_TABLE_Z_MARGIN = 0.04

# ── Waypoint config ───────────────────────────────────────────────────────────
WAYPOINT_STEPS   = 5
WAYPOINT_XY_TOL  = 0.02
WAYPOINT_Z_TOL   = 0.02


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _quat_multiply(q1_xyzw, q2_xyzw):
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _quat_inv(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([-x, -y, -z, w])


def _quat_to_yaw(q_xyzw: np.ndarray) -> float:
    x, y, z, w = q_xyzw / (np.linalg.norm(q_xyzw) + 1e-9)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_error(eef_quat: np.ndarray, target_yaw: float) -> float:
    eef_yaw = _quat_to_yaw(eef_quat)
    err = target_yaw - eef_yaw
    err = (err + np.pi) % (2 * np.pi) - np.pi
    if err >  np.pi / 2:
        err -= np.pi
    if err < -np.pi / 2:
        err += np.pi
    return err


def _target_yaw(obj_quat: np.ndarray, obj_name: str) -> float | None:
    if obj_name not in YAW_ALIGN_OBJECTS:
        return None
    if obj_name == "gelatin_box":
        base_inv   = _quat_inv(GELATIN_STANDING_QUAT_XYZW)
        q_yaw_only = _quat_multiply(base_inv, obj_quat)
        obj_yaw    = _quat_to_yaw(q_yaw_only)
    else:
        obj_yaw = _quat_to_yaw(obj_quat)
    extra = YAW_EXTRA_OFFSET.get(obj_name, 0.0)
    return obj_yaw + extra


# ── Action helpers ────────────────────────────────────────────────────────────

def _clip_action(dx, dy, dz, droll, dpitch, dyaw, gripper,
                 max_xy=1.0, max_z=1.0, max_rot=1.0):
    action = np.zeros(7)
    action[0] = np.clip(dx,     -max_xy,  max_xy)
    action[1] = np.clip(dy,     -max_xy,  max_xy)
    action[2] = np.clip(dz,     -max_z,   max_z)
    action[3] = np.clip(droll,  -max_rot, max_rot)
    action[4] = np.clip(dpitch, -max_rot, max_rot)
    action[5] = np.clip(dyaw,   -max_rot, max_rot)
    action[6] = gripper
    return action


def _no_move(gripper=1.0):
    return _clip_action(0, 0, 0, 0, 0, 0, gripper)


# ── Arc waypoints ─────────────────────────────────────────────────────────────

def _make_line_waypoints(
    start: np.ndarray,
    end: np.ndarray,
    n: int,
) -> list[np.ndarray]:
    """Linearly interpolated waypoints from start to end (n points, end-inclusive)."""
    return [(1.0 - i / n) * start + (i / n) * end for i in range(1, n + 1)]


# ── Policy ────────────────────────────────────────────────────────────────────

class SlipRecoveryPolicy:
    """
    INIT → OPEN → ALIGN_ABOVE → DESCEND → GRASP → LIFT
         → TRANSPORT → LOWER → RELEASE → DONE
    """

    N_WAYPOINTS = {
        "ALIGN_ABOVE": 8,
        "TRANSPORT":   12,
        "LOWER":       6,
    }

    def __init__(self):
        self.env = None
        self._reset_state()

    def reset(self):
        self._reset_state()

    def act(self, obs, env=None):
        if env is not None:
            self.env = env
        action = self._act_inner(obs)
        self.step_in_phase += 1
        return action

    # ── State ─────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.phase         = "INIT"
        self.step_in_phase = 0
        self._obj_name     = None
        self._table_z      = None
        self._transport_z  = None
        self._goal_xy      = None
        self._slip_count   = 0
        self._picked_up    = False

        self._waypoints = []
        self._wp_idx    = 0
        self._wp_hold   = 0

    def _table_height(self):
        if self._table_z is None:
            try:
                self._table_z = float(self.env._table_height)
            except Exception:
                self._table_z = 0.80
        return self._table_z

    def _read_obs(self, obs):
        eef_pos  = np.array(obs.get("robot0_eef_pos",  [0, 0, 1]),    dtype=float)
        eef_quat = np.array(obs.get("robot0_eef_quat", [1, 0, 0, 0]), dtype=float)
        obj_pos  = np.array(obs.get("object_pos",  [0, 0, self._table_height()]), dtype=float)
        obj_quat = np.array(obs.get("object_quat", [1, 0, 0, 0]),     dtype=float)
        return eef_pos, eef_quat, obj_pos, obj_quat

    def _hover_z(self, obj_pos):
        return obj_pos[2] + HOVER_ABOVE_OBJ

    def _slip_detected(self, obj_pos):
        if not self._picked_up:
            return False
        if self.phase in ("RELEASE", "DONE"):
            return False
        return obj_pos[2] < self._table_height() + ON_TABLE_Z_MARGIN

    def _advance(self, next_phase):
        self.phase = next_phase
        self.step_in_phase = 0
        self._waypoints = []
        self._wp_idx    = 0
        self._wp_hold   = 0

    def _dyaw(self, eef_quat, obj_quat):
        tgt = _target_yaw(obj_quat, self._obj_name)
        if tgt is None:
            return 0.0
        err = _yaw_error(eef_quat, tgt)
        return np.clip(err * YAW_GAIN, -1.0, 1.0)

    # ── Waypoint helpers ──────────────────────────────────────────────────────

    def _init_waypoints(self, eef_pos, end_pos, phase_key):
        n = self.N_WAYPOINTS.get(phase_key, 8)
        self._waypoints = _make_line_waypoints(eef_pos, end_pos, n)
        self._wp_idx    = 0
        self._wp_hold   = 0

    def _current_wp(self):
        if self._wp_idx < len(self._waypoints):
            return self._waypoints[self._wp_idx]
        return None

    def _tick_waypoint(self, eef_pos):
        wp = self._current_wp()
        if wp is None:
            return None
        xy_err = np.linalg.norm(eef_pos[:2] - wp[:2])
        z_err  = abs(eef_pos[2] - wp[2])
        if xy_err < WAYPOINT_XY_TOL and z_err < WAYPOINT_Z_TOL:
            self._wp_hold += 1
            if self._wp_hold >= WAYPOINT_STEPS:
                self._wp_idx += 1
                self._wp_hold = 0
        else:
            self._wp_hold = 0
        return self._current_wp()

    # ── State machine ─────────────────────────────────────────────────────────

    def _act_inner(self, obs):
        eef_pos, eef_quat, obj_pos, obj_quat = self._read_obs(obs)

        if self._obj_name is None:
            self._obj_name = getattr(self.env, "_selected_ycb", "unknown")
        if self._goal_xy is None:
            self._goal_xy = np.array(self.env._goal_pos, dtype=float)

        th      = self._table_height()
        goal_xy = self._goal_xy

        # ── Slip detection ────────────────────────────────────────────────────
        if self._slip_detected(obj_pos):
            self._slip_count += 1
            self._picked_up = False
            print(f"  [SLIP #{self._slip_count}] was in {self.phase} | "
                  f"obj_z={obj_pos[2]:.3f} table_z={th:.3f} | restarting pick")
            self._advance("OPEN")
            return _no_move(-1.0)

        # ── INIT ──────────────────────────────────────────────────────────────
        if self.phase == "INIT":
            if self.step_in_phase >= 5:
                self._advance("OPEN")
            return _no_move(-1.0)

        # ── OPEN ──────────────────────────────────────────────────────────────
        elif self.phase == "OPEN":
            if self.step_in_phase >= OPEN_HOLD:
                self._advance("ALIGN_ABOVE")
            return _no_move(-1.0)

        # ── ALIGN_ABOVE ───────────────────────────────────────────────────────
        elif self.phase == "ALIGN_ABOVE":
            target_z  = self._hover_z(obj_pos)
            final_tgt = np.array([obj_pos[0], obj_pos[1], target_z])

            if not self._waypoints:
                self._init_waypoints(eef_pos, final_tgt, "ALIGN_ABOVE")

            wp = self._tick_waypoint(eef_pos)
            if wp is None:
                wp = final_tgt

            delta = wp - eef_pos
            dist  = np.linalg.norm(delta)
            gain  = GAIN_FAST if dist > 0.12 else GAIN_XY
            dyaw  = self._dyaw(eef_quat, obj_quat)

            dx = delta[0] * gain
            dy = delta[1] * gain
            dz = delta[2] * gain

            xy_err  = np.linalg.norm(eef_pos[:2] - final_tgt[:2])
            z_err   = abs(eef_pos[2] - final_tgt[2])
            tgt_yaw = _target_yaw(obj_quat, self._obj_name)
            yaw_err = abs(_yaw_error(eef_quat, tgt_yaw)) if tgt_yaw is not None else 0.0
            yaw_ok  = (self._obj_name not in YAW_ALIGN_OBJECTS) or (yaw_err < YAW_TOL)

            if xy_err < XY_TOL and z_err < Z_TOL and yaw_ok:
                self._advance("DESCEND")

            return _clip_action(dx, dy, dz, 0, 0, dyaw, -1.0)

        # ── DESCEND ───────────────────────────────────────────────────────────
        elif self.phase == "DESCEND":
            dx   = (obj_pos[0] - eef_pos[0]) * GAIN_XY
            dy   = (obj_pos[1] - eef_pos[1]) * GAIN_XY
            dz   = (obj_pos[2] - eef_pos[2]) * GAIN_Z
            dyaw = self._dyaw(eef_quat, obj_quat)

            xy_err = np.linalg.norm([eef_pos[0]-obj_pos[0], eef_pos[1]-obj_pos[1]])
            z_err  = abs(eef_pos[2] - obj_pos[2])

            if (z_err < Z_TOL and xy_err < XY_TOL) or self.step_in_phase >= LOWER_BUDGET:
                self._advance("GRASP")

            return _clip_action(dx, dy, dz, 0, 0, dyaw, -1.0)

        # ── GRASP ─────────────────────────────────────────────────────────────
        elif self.phase == "GRASP":
            dz = max((obj_pos[2] - eef_pos[2]) * GAIN_Z, -0.2)

            if self.step_in_phase >= CLOSE_HOLD:
                self._transport_z = th + TRANSPORT_ABOVE_TABLE
                self._advance("LIFT")

            return _clip_action(0, 0, dz, 0, 0, 0, 1.0)

        # ── LIFT ──────────────────────────────────────────────────────────────
        elif self.phase == "LIFT":
            tz = self._transport_z
            dz = (tz - eef_pos[2]) * GAIN_Z
            if eef_pos[2] > tz - 0.015:
                self._picked_up = True
                self._advance("TRANSPORT")
            return _clip_action(0, 0, dz, 0, 0, 0, 1.0)

        # ── TRANSPORT ─────────────────────────────────────────────────────────
        elif self.phase == "TRANSPORT":
            tz        = self._transport_z
            final_tgt = np.array([goal_xy[0], goal_xy[1], tz])

            if not self._waypoints:
                self._init_waypoints(eef_pos, final_tgt, "TRANSPORT")

            wp = self._tick_waypoint(eef_pos)
            if wp is None:
                wp = final_tgt

            dx = (wp[0] - eef_pos[0]) * GAIN_FAST
            dy = (wp[1] - eef_pos[1]) * GAIN_FAST
            dz = (tz - eef_pos[2]) * GAIN_Z

            xy_err = np.linalg.norm(eef_pos[:2] - final_tgt[:2])
            if xy_err < XY_TOL:
                self._advance("LOWER")

            return _clip_action(dx, dy, dz, 0, 0, 0, 1.0)

        # ── LOWER ─────────────────────────────────────────────────────────────
        elif self.phase == "LOWER":
            place_z   = th + 0.005
            final_tgt = np.array([goal_xy[0], goal_xy[1], place_z])

            if not self._waypoints:
                self._init_waypoints(eef_pos, final_tgt, "LOWER")

            wp = self._tick_waypoint(eef_pos)
            if wp is None:
                wp = final_tgt

            dx = (wp[0] - eef_pos[0]) * GAIN_XY
            dy = (wp[1] - eef_pos[1]) * GAIN_XY
            dz = (wp[2] - eef_pos[2]) * GAIN_Z

            obj_near_table = obj_pos[2] < th + 0.045
            if obj_near_table or self.step_in_phase >= LOWER_BUDGET:
                self._picked_up = False
                self._advance("RELEASE")

            return _clip_action(dx, dy, dz, 0, 0, 0, 1.0)

        # ── RELEASE ───────────────────────────────────────────────────────────
        elif self.phase == "RELEASE":
            if self.step_in_phase >= RELEASE_HOLD:
                self._advance("DONE")
            return _no_move(-1.0)

        # ── DONE ──────────────────────────────────────────────────────────────
        elif self.phase == "DONE":
            return _no_move(-1.0)

        return np.zeros(7)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env = ib.make("ib-p6-slip-recovery-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")
    policy = SlipRecoveryPolicy()

    n_episodes = 5
    for ep in range(n_episodes):
        obs, info = env.reset()
        policy.reset()
        done       = False
        prev_phase = policy.phase
        step       = 0

        while not done:
            action = policy.act(obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if policy.phase != prev_phase:
                obj_pos = obs.get("object_pos", np.zeros(3))
                eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
                print(
                    f"  ep={ep}  step={step:4d}  "
                    f"{prev_phase:18s} → {policy.phase:18s}  "
                    f"obj_z={obj_pos[2]:.3f}  eef_z={eef_pos[2]:.3f}  "
                    f"score={info.get('task_score', 0):.3f}"
                )
                prev_phase = policy.phase

            done = terminated or truncated
            step += 1

        print(
            f"\nEpisode {ep}  steps={step}  slips={policy._slip_count}  "
            f"score={info.get('task_score', 0):.3f}  "
            f"final_phase={policy.phase}\n"
        )

    env.close()