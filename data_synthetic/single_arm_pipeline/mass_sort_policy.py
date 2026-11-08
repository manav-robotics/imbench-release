"""
mass_sort_policy.py — Pick every cube and place it in its correct zone.
"""

import numpy as np
import imbench as ib

# ── Tuning ────────────────────────────────────────────────────────────────────
HOVER_ABOVE_OBJ       = 0.10
TRANSPORT_ABOVE_TABLE = 0.10
XY_TOL       = 0.013
Z_TOL        = 0.010
GAIN_XY      = 6
GAIN_Z       = 6
GAIN_FAST    = 8        # used when dist > 0.12 m for long-haul transport
OPEN_HOLD    = 20       # gripper saturates quickly, no need to dwell long
CLOSE_HOLD   = 30
LOWER_BUDGET = 180
RELEASE_HOLD = 25
ON_TABLE_Z_MARGIN = 0.06

WAYPOINT_STEPS   = 10
WAYPOINT_XY_TOL  = 0.02
WAYPOINT_Z_TOL   = 0.02
ARC_Z_LIFT       = 0.1
OVERSHOOT_MAG    = 0.015   # fixed small overshoot in metres
OVERSHOOT_DECAY  = 0.55    # fast decay: magnitude halves every ~1.3 steps

# Waypoint counts per path phase. Directly set how many ticks each
# multi-segment motion takes — fewer = faster (steeper steps between
# waypoints) but less smooth. Tuned down from 8/12/6 to halve the path
# duration; bump back up if you see overshoot/jerk on the cube grasp.
N_WAYPOINTS = {"ALIGN_ABOVE": 5, "TRANSPORT": 7, "LOWER": 4}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clip_action(dx, dy, dz, gripper, max_xy=1.0, max_z=1.0):
    a = np.zeros(7)
    a[0] = np.clip(dx, -max_xy, max_xy)
    a[1] = np.clip(dy, -max_xy, max_xy)
    a[2] = np.clip(dz, -max_z,  max_z)
    a[6] = gripper
    return a

def _no_move(gripper=1.0):
    return _clip_action(0, 0, 0, gripper)

def _make_arc_waypoints(start, end, n, z_lift=0.0):
    wps = []
    for i in range(1, n + 1):
        t  = i / n
        pt = ((1.0 - t) * start + t * end).copy()
        pt[2] += z_lift * np.sin(np.pi * t)
        wps.append(pt)
    return wps


class OvershootOscillator:
    """
    Applies a single small nudge past the final waypoint then decays it away.
    magnitude is always OVERSHOOT_MAG (0.015 m) — never tied to OVERSHOOT_FRAC.
    """
    def __init__(self):
        self._offset = np.zeros(3)
        self._active = False

    def trigger(self, direction):
        n = np.linalg.norm(direction)
        if n < 1e-6:
            return
        self._offset = direction / n * OVERSHOOT_MAG   # fixed 1.5 cm nudge
        self._active = True

    def step(self):
        if not self._active:
            return np.zeros(3)
        out = self._offset.copy()
        self._offset *= OVERSHOOT_DECAY
        if np.linalg.norm(self._offset) < 1e-4:
            self._active = False
            self._offset = np.zeros(3)
        return out

    @property
    def active(self):
        return self._active

    def reset(self):
        self._active = False
        self._offset = np.zeros(3)


# ── Plan extraction ───────────────────────────────────────────────────────────

def build_placement_plan(env, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    task = env
    sim  = env._env.sim

    cube_masses = {}
    for i in range(task._num_cubes):
        mass_kg = task._masses[task._mass_order[i]]
        cube_masses[i] = mass_kg

    sorted_masses = sorted(cube_masses.values())

    plan = []
    for i in range(task._num_cubes):
        color     = task.CUBE_COLORS[i % len(task.CUBE_COLORS)][0]
        mass_kg   = cube_masses[i]
        body_name = "cube_main" if i == 0 else f"cube_{i}_main"

        target_zone = sorted_masses.index(mass_kg)
        zone_id     = sim.model.body_name2id(f"zone_{target_zone}_main")
        zone_pos    = sim.data.body_xpos[zone_id].copy()

        plan.append({
            "cube_idx":       i,
            "color":          color,
            "mass_g":         mass_kg * 1000,
            "cube_body_name": body_name,
            "target_zone":    target_zone,
            "target_pos":     zone_pos,
        })

    rng.shuffle(plan)
    return plan


# ── Waypoint tracker ──────────────────────────────────────────────────────────

class WaypointTracker:
    """
    Tracks progress through a list of arc waypoints.
    Call .init(start, end, phase_key) to load a new path.
    Call .tick(eef_pos) each step → returns (current_wp_xyz, overshoot_offset).
    .finished is True once the last waypoint is held AND overshoot has decayed.
    """

    def __init__(self):
        self._waypoints    = []
        self._wp_idx       = 0
        self._wp_hold      = 0
        self._approach_dir = np.zeros(3)
        self._overshoot    = OvershootOscillator()
        self._loaded       = False

    def init(self, start, end, phase_key):
        n      = N_WAYPOINTS.get(phase_key, 8)
        z_lift = ARC_Z_LIFT if phase_key == "TRANSPORT" else 0.0
        self._waypoints    = _make_arc_waypoints(start, end, n, z_lift=z_lift)
        self._wp_idx       = 0
        self._wp_hold      = 0
        self._overshoot.reset()
        diff               = end - start
        norm               = np.linalg.norm(diff)
        self._approach_dir = diff / norm if norm > 1e-6 else np.zeros(3)
        self._loaded       = True

    @property
    def loaded(self):
        return self._loaded

    @property
    def goal(self):
        """Final waypoint of the loaded path."""
        return self._waypoints[-1]

    @property
    def finished(self):
        """True only when past all waypoints AND overshoot has fully decayed."""
        return (
            self._loaded
            and self._wp_idx >= len(self._waypoints)
            and not self._overshoot.active
        )

    def tick(self, eef_pos):
        """
        Advance through waypoints.
        Returns (current_target_xyz, overshoot_offset).
        Once all waypoints are done, current_target_xyz is the final waypoint.
        """
        if not self._loaded or not self._waypoints:
            return np.zeros(3), np.zeros(3)

        idx = min(self._wp_idx, len(self._waypoints) - 1)
        wp  = self._waypoints[idx]

        if self._wp_idx < len(self._waypoints):
            xy_err = np.linalg.norm(eef_pos[:2] - wp[:2])
            z_err  = abs(eef_pos[2] - wp[2])
            if xy_err < WAYPOINT_XY_TOL and z_err < WAYPOINT_Z_TOL:
                self._wp_hold += 1
                if self._wp_hold >= WAYPOINT_STEPS:
                    self._wp_hold = 0
                    self._wp_idx += 1
                    # Trigger one small fixed overshoot when the last waypoint clears
                    if self._wp_idx >= len(self._waypoints) and not self._overshoot.active:
                        self._overshoot.trigger(self._approach_dir)
            else:
                self._wp_hold = 0

        overshoot = self._overshoot.step()
        return wp, overshoot

    def reset(self):
        self._waypoints    = []
        self._wp_idx       = 0
        self._wp_hold      = 0
        self._approach_dir = np.zeros(3)
        self._overshoot.reset()
        self._loaded       = False


# ── Single-cube pick-and-place ────────────────────────────────────────────────

class CubePickPlace:
    """
    Runs the full INIT→OPEN→ALIGN_ABOVE→DESCEND→GRASP→LIFT→
    TRANSPORT→LOWER→RELEASE→DONE sequence for one cube.
    .act(obs) → (action, done_bool)
    """

    def __init__(self, env, plan_entry):
        self.env            = env
        self.entry          = plan_entry
        self.phase          = "INIT"
        self.step_in_phase  = 0
        self._table_z       = float(env._table_height)
        self._picked_up     = False
        self._fully_lifted  = False
        self._tracker       = WaypointTracker()

    def _cube_pos(self):
        sim = self.env._env.sim
        bid = sim.model.body_name2id(self.entry["cube_body_name"])
        return sim.data.body_xpos[bid].copy()

    def _advance(self, nxt):
        print(f"    [{self.entry['color']}] {self.phase} → {nxt}  "
              f"(step {self.step_in_phase})")
        self.phase         = nxt
        self.step_in_phase = 0
        self._tracker.reset()

    def act(self, obs):
        eef_pos  = np.array(obs["robot0_eef_pos"], dtype=float)
        obj_pos  = self._cube_pos()
        th       = self._table_z
        goal_pos = self.entry["target_pos"]
        done     = False

        transport_z = self._table_z + TRANSPORT_ABOVE_TABLE
        if self._picked_up and eef_pos[2] > transport_z - 0.02:
            self._fully_lifted = True

        if self._fully_lifted and self.phase not in ("LOWER", "RELEASE", "DONE"):
            if obj_pos[2] < th + ON_TABLE_Z_MARGIN:
                print(f"    [{self.entry['color']}] SLIP — restarting pick")
                self._picked_up    = False
                self._fully_lifted = False
                self._advance("OPEN")
                return _no_move(-1.0), done

        # ── INIT ──────────────────────────────────────────────────────────────
        if self.phase == "INIT":
            if self.step_in_phase >= 5:
                self._advance("OPEN")
            self.step_in_phase += 1
            return _no_move(-1.0), done

        # ── OPEN ──────────────────────────────────────────────────────────────
        elif self.phase == "OPEN":
            if self.step_in_phase >= OPEN_HOLD:
                self._advance("ALIGN_ABOVE")
            self.step_in_phase += 1
            return _no_move(-1.0), done

        # ── ALIGN_ABOVE ───────────────────────────────────────────────────────
        elif self.phase == "ALIGN_ABOVE":
            hover_z   = obj_pos[2] + HOVER_ABOVE_OBJ
            final_tgt = np.array([obj_pos[0], obj_pos[1], hover_z])

            # Re-plan when the cube has moved since the path was built (e.g. it
            # was still dropping after a SLIP restart); otherwise the arm parks
            # at the stale final waypoint and the live exit check never passes.
            if (not self._tracker.loaded
                    or np.linalg.norm(self._tracker.goal - final_tgt) > WAYPOINT_XY_TOL):
                self._tracker.init(eef_pos, final_tgt, "ALIGN_ABOVE")

            wp, overshoot = self._tracker.tick(eef_pos)
            eff   = wp + overshoot
            delta = eff - eef_pos
            dist  = np.linalg.norm(delta)
            gain  = GAIN_FAST if dist > 0.12 else GAIN_XY

            xy_err = np.linalg.norm(eef_pos[:2] - final_tgt[:2])
            z_err  = abs(eef_pos[2] - final_tgt[2])
            if self._tracker.finished and xy_err < XY_TOL and z_err < Z_TOL:
                self._advance("DESCEND")

            self.step_in_phase += 1
            return _clip_action(delta[0]*gain, delta[1]*gain, delta[2]*gain, -1.0), done

        # ── DESCEND ───────────────────────────────────────────────────────────
        elif self.phase == "DESCEND":
            dx = (obj_pos[0] - eef_pos[0]) * GAIN_XY
            dy = (obj_pos[1] - eef_pos[1]) * GAIN_XY
            dz = (obj_pos[2] - eef_pos[2]) * GAIN_Z

            xy_err = np.linalg.norm(eef_pos[:2] - obj_pos[:2])
            z_err  = abs(eef_pos[2] - obj_pos[2])
            if (z_err < Z_TOL and xy_err < XY_TOL) or self.step_in_phase >= LOWER_BUDGET:
                self._advance("GRASP")

            self.step_in_phase += 1
            return _clip_action(dx, dy, dz, -1.0), done

        # ── GRASP ─────────────────────────────────────────────────────────────
        elif self.phase == "GRASP":
            dz = max((obj_pos[2] - eef_pos[2]) * GAIN_Z, -0.2)
            if self.step_in_phase >= CLOSE_HOLD:
                self._picked_up = True
                self._advance("LIFT")
            self.step_in_phase += 1
            return _clip_action(0, 0, dz, 1.0), done

        # ── LIFT ──────────────────────────────────────────────────────────────
        elif self.phase == "LIFT":
            tz = self._table_z + TRANSPORT_ABOVE_TABLE
            dz = (tz - eef_pos[2]) * GAIN_Z
            if eef_pos[2] > tz - 0.015:
                self._advance("TRANSPORT")
            self.step_in_phase += 1
            return _clip_action(0, 0, dz, 1.0), done

        # ── TRANSPORT ─────────────────────────────────────────────────────────
        elif self.phase == "TRANSPORT":
            # Arc trajectory with fixed-magnitude overshoot (0.015 m).
            # The tracker raises the path by ARC_Z_LIFT via a sine arc,
            # then applies one small nudge past the final waypoint that
            # decays within ~10 steps (OVERSHOOT_DECAY=0.55).
            tz        = self._table_z + TRANSPORT_ABOVE_TABLE
            final_tgt = np.array([goal_pos[0], goal_pos[1], tz])

            if not self._tracker.loaded:
                self._tracker.init(eef_pos, final_tgt, "TRANSPORT")

            wp, overshoot = self._tracker.tick(eef_pos)
            eff   = wp + overshoot
            delta = eff - eef_pos
            dist  = np.linalg.norm(delta)
            gain  = GAIN_FAST if dist > 0.12 else GAIN_XY

            xy_err = np.linalg.norm(eef_pos[:2] - final_tgt[:2])
            if self._tracker.finished and xy_err < XY_TOL:
                self._advance("LOWER")

            self.step_in_phase += 1
            return _clip_action(delta[0]*gain, delta[1]*gain, delta[2]*gain, 1.0), done

        # ── LOWER ─────────────────────────────────────────────────────────────
        elif self.phase == "LOWER":
            place_z   = th + 0.005
            final_tgt = np.array([goal_pos[0], goal_pos[1], place_z])

            if not self._tracker.loaded:
                self._tracker.init(eef_pos, final_tgt, "LOWER")

            wp, overshoot = self._tracker.tick(eef_pos)
            eff    = wp + overshoot
            eff[2] = wp[2]   # suppress Z overshoot on descent

            dx = (eff[0] - eef_pos[0]) * GAIN_XY
            dy = (eff[1] - eef_pos[1]) * GAIN_XY
            dz = (eff[2] - eef_pos[2]) * GAIN_Z

            if obj_pos[2] < th + 0.06 or self.step_in_phase >= LOWER_BUDGET:
                self._picked_up    = False
                self._fully_lifted = False
                self._advance("RELEASE")

            self.step_in_phase += 1
            return _clip_action(dx, dy, dz, 1.0), done

        # ── RELEASE ───────────────────────────────────────────────────────────
        elif self.phase == "RELEASE":
            if self.step_in_phase >= RELEASE_HOLD:
                self._advance("DONE")
                done = True
            self.step_in_phase += 1
            return _no_move(-1.0), done

        # ── DONE ──────────────────────────────────────────────────────────────
        elif self.phase == "DONE":
            return _no_move(-1.0), True

        return np.zeros(7), False


# ── Top-level policy ──────────────────────────────────────────────────────────

class MassSortPolicy:
    def __init__(self, env=None, rng=None):
        self.env           = env
        self.rng           = np.random.default_rng() if rng is None else rng
        self._plan         = []
        self._plan_idx     = 0
        self._current_task = None
        # The plan needs a live env (cube positions, masses); the driver
        # constructs the policy before the env exists and then passes
        # env on every act() call. Defer plan build to the first act().
        self._initialized  = False

    def _build_plan(self):
        self._plan     = build_placement_plan(self.env, rng=self.rng)
        self._plan_idx = 0

        print("\n=== MassSortPolicy: randomised placement plan ===")
        for i, entry in enumerate(self._plan):
            p = entry["target_pos"]
            print(f"  step {i}: {entry['color']:6s}  {entry['mass_g']:5.1f}g"
                  f"  → zone_{entry['target_zone']}"
                  f"  target=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
        print()

        if self._plan:
            self._current_task = CubePickPlace(self.env, self._plan[0])

    def reset(self, obs=None):
        self._plan         = []
        self._plan_idx     = 0
        self._current_task = None
        self._initialized  = False
        if self.env is None:
            return
        self._build_plan()
        self._initialized = True

    def act(self, obs, env=None):
        if env is not None:
            self.env = env
        if not self._initialized and self.env is not None:
            self._build_plan()
            self._initialized = True

        if self._current_task is None:
            return _no_move(-1.0)

        action, cube_done = self._current_task.act(obs)

        if cube_done:
            self._plan_idx += 1
            if self._plan_idx < len(self._plan):
                entry = self._plan[self._plan_idx]
                print(f"\n  → Next: cube_{entry['cube_idx']} ({entry['color']})"
                      f"  {entry['mass_g']:.0f}g → zone_{entry['target_zone']}\n")
                self._current_task = CubePickPlace(self.env, entry)
            else:
                print("\n  ✓ All cubes placed!\n")
                self._current_task = None

        return action

    @property
    def all_done(self):
        return self._current_task is None and self._plan_idx >= len(self._plan)

    @property
    def phase(self) -> str:
        """Human-readable phase label for the collect-demos failure log."""
        if not self._initialized:
            return "init"
        if self._current_task is None:
            return f"plan{self._plan_idx}/done"
        sub = type(self._current_task).__name__
        sub_phase = getattr(self._current_task, "phase", None)
        if sub_phase:
            return f"plan{self._plan_idx}/{sub}/{sub_phase}"
        return f"plan{self._plan_idx}/{sub}"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env = ib.make("ib-p5-mass-sort-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")

    n_episodes = 3
    for ep in range(n_episodes):
        obs, info = env.reset()
        policy    = MassSortPolicy(env)
        policy.reset(obs)

        step = 0
        done = False

        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated or policy.all_done
            step += 1

        print(
            f"\nEpisode {ep}  steps={step}  reward={reward:.1f}"
            f"  outcome={info.get('outcome', 'n/a')}"
            f"  score={info.get('task_score', 0):.3f}\n"
        )

    env.close()