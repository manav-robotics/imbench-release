"""
indirect_push_policy.py

Multi-wave pipeline:

  Wave k:
    Phase 1 — Y alignment:
        For each selected block, push it purely in Y until its y matches
        the target's current y.

    Phase 2 — X positioning with lazy Y guard:
        Before each X-push, re-check the block's actual Y (physics may have
        caused overshoot). Correct Y if needed, then push block into its slot
        directly behind the target.

    Phase 3 — Chain push in X:
        Push the whole chain purely in X to drive the target toward the goal.

  After Phase 3, check whether the target has reached the goal.
  If not, build a new wave behind the target's new position and repeat
  until the goal is reached or no usable blocks remain.

All pushes are strictly axis-aligned. Gripper always closed.
"""

import numpy as np
import imbench as ib

# ── Constants ─────────────────────────────────────────────────────────────────
TABLE_Z       = 0.800
BLOCK_SIZE    = 0.025
PUSH_Z        = TABLE_Z + BLOCK_SIZE + 0.003
HOVER_Z       = TABLE_Z + 0.10
PUSH_STANDOFF = 0.06
PUSH_OVER     = 0.005   # small overshoot is enough; a larger value makes
                        # AxisPush drive blocks well past dest_val.
# Vertical clearance above HOVER_Z to use during inter-block transit.
# Without this the EEF moves laterally at HOVER_Z right after pushing,
# and fingertips graze the just-aligned block (toppling it). The TRANSIT
# phase enforces lift-first-then-translate: rise to HOVER_Z + this, then
# move XY, then descend to PUSH_Z.
TRANSIT_LIFT      = 0.08
TRANSIT_LIFT_TOL  = 0.010   # eef z must be within this of (HOVER_Z + TRANSIT_LIFT) before lateral motion

# Per-phase tick budgets for AxisPush. If a phase exits condition never
# fires (e.g. block stuck against another block during PUSH; eef can't
# reach transit altitude due to joint limit), we force-advance after
# this many ticks to prevent the EEF from crawling indefinitely in one
# direction. Tuned generously — normal phases finish in 30-100 ticks.
AXIS_PUSH_PHASE_BUDGET = {
    "LIFT_TRANSIT": 200,
    "MOVE_TRANSIT": 400,
    "DESCEND":      200,
    "PUSH":         500,
    "ASCEND":       200,
}
CHAIN_SPACING = 0.055
# Distance (m) to nudge each just-placed chain block forward toward the
# target/goal *before* aligning the next block. Half of CHAIN_SPACING so
# the chain compacts as it forms — block 1 ends up ~half a slot ahead of
# its original placement, block 2 lands behind it but still close, and
# the final ChainPushX has a tighter chain to drive into the target.
NUDGE_FORWARD = CHAIN_SPACING * 0.5
XY_TOL        = 0.008
Z_TOL         = 0.006
GOAL_TOL      = 0.020
GAIN          = 5.0
# Per-tick step cap for the AxisPush "PUSH" sub-phase. Without this the
# controller commands the full ±1.0 action range whenever it's >0.2 m
# from push_end, which slams into the block at full speed and shoots it
# past the target. 0.20 ≈ 2 cm/tick at typical action scale, so the
# block decelerates smoothly over the last few cm instead of one big
# overshoot. Lower → gentler / longer; higher → faster / more overshoot.
PUSH_STEP_CAP = 0.20
# Distance (m) over which the push command ramps from PUSH_STEP_CAP down
# to a tiny crawl. At |err| ≤ this, action ≈ proportional with the cap;
# beyond it, action is held at the cap. Set to ~3× XY_TOL so the last
# few mm before dest_val are gentle.
PUSH_RAMP_DIST = 0.025


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clip(dx, dy, dz, g, lim=1.0):
    a = np.zeros(7)
    a[0] = np.clip(dx, -lim, lim)
    a[1] = np.clip(dy, -lim, lim)
    a[2] = np.clip(dz, -lim, lim)
    a[6] = g
    return a

def _eef(obs):
    return np.array(obs["robot0_eef_pos"], dtype=float)

def _body_pos(sim, name):
    bid = sim.model.body_name2id(name)
    return sim.data.body_xpos[bid].copy()

def _all_chain_blocks(env):
    """Chain blocks actually in play this episode.

    The task builds every block it could need into the model and parks
    the unused ones off-scene, so ask it which those are instead of
    inferring it from position — a parked block is still falling toward
    its resting place for the first few seconds and would otherwise pass
    a height test.  The z-check remains as a guard against blocks that
    have fallen off the table mid-episode.
    """
    sim = env._env.sim
    parked = env._parked_objects()
    n_max = getattr(env, "_max_chain", 6)
    blocks = []
    for i in range(n_max):
        if f"chain_{i}" in parked:
            continue
        try:
            pos = _body_pos(sim, f"chain_{i}_main")
        except Exception:
            break
        if pos[2] > 0.5:
            blocks.append((i, pos.copy()))
    return blocks

def _target_pos(sim):
    return _body_pos(sim, "target_main")

def _goal_pos(env, sim):
    gp = env._goal_pos
    tp = _target_pos(sim)
    if len(gp) == 2:
        return np.array([gp[0], gp[1], tp[2]])
    return np.array(gp, dtype=float)

def _target_reached(env, sim):
    tp = _target_pos(sim)[:2]
    gp = _goal_pos(env, sim)[:2]
    return np.linalg.norm(tp - gp) < GOAL_TOL


# ── Axis-aligned push primitive ───────────────────────────────────────────────

class AxisPush:
    """
    Push a block along a single axis (X or Y) to dest_val on that axis.
    axis: 0 = X,  1 = Y
    """

    def __init__(self, sim_getter, block_name, axis, dest_val, verbose=False):
        self._get_sim   = sim_getter
        self.block_name = block_name
        self.axis       = axis
        self.dest_val   = dest_val
        self.verbose    = verbose
        # Initial phase: lift straight up to TRANSIT z BEFORE any
        # lateral motion, so the EEF can't drag past the previous
        # just-aligned block on its way to this one. Transitions:
        # LIFT_TRANSIT → MOVE_TRANSIT → DESCEND → PUSH → ASCEND → DONE.
        self.phase             = "LIFT_TRANSIT"
        self._ticks_in_phase   = 0

    def _bpos(self):
        return _body_pos(self._get_sim(), self.block_name)

    def _advance_or_budget(self, next_phase: str) -> None:
        """Move to next_phase and reset the per-phase tick counter."""
        self.phase           = next_phase
        self._ticks_in_phase = 0

    def _budget_exceeded(self) -> bool:
        """Force-advance if the current phase has run too long without its
        natural exit condition firing — prevents the EEF from crawling in
        one direction forever when a block jams or a target is unreachable.
        """
        budget = AXIS_PUSH_PHASE_BUDGET.get(self.phase)
        if budget is not None and self._ticks_in_phase >= budget:
            if self.verbose:
                print(f"  [AxisPush] {self.block_name} {self.phase} budget "
                      f"({budget} ticks) exceeded — force advancing")
            return True
        return False

    def act(self, obs):
        eef  = _eef(obs)
        bpos = self._bpos()
        self._ticks_in_phase += 1

        delta_to_dest = self.dest_val - bpos[self.axis]
        if abs(delta_to_dest) < XY_TOL:
            return _clip(0, 0, 0, 1.0), True

        push_sign = np.sign(delta_to_dest)

        if self.phase == "LIFT_TRANSIT":
            # Lift straight up to a transit altitude clear of any
            # already-aligned blocks, with no lateral motion. Only after
            # the eef has reached the transit height do we begin moving
            # toward the next block.
            transit_z = HOVER_Z + TRANSIT_LIFT
            dz = (transit_z - eef[2]) * GAIN
            if (eef[2] >= transit_z - TRANSIT_LIFT_TOL
                    or self._budget_exceeded()):
                self._advance_or_budget("MOVE_TRANSIT")
            return _clip(0, 0, dz, 1.0), False

        elif self.phase == "MOVE_TRANSIT":
            # Translate at transit altitude (above the just-aligned
            # block) to a point directly above the standoff position
            # behind the next block. Z is held at transit_z; the next
            # phase (DESCEND) brings the eef down to PUSH_Z.
            transit_z = HOVER_Z + TRANSIT_LIFT
            behind_xy = bpos.copy()
            behind_xy[self.axis] -= push_sign * PUSH_STANDOFF
            target = np.array([behind_xy[0], behind_xy[1], transit_z])
            delta  = target - eef
            xy_ok = np.linalg.norm(eef[:2] - target[:2]) < XY_TOL
            # z-tight is enforced loosely (within TRANSIT_LIFT_TOL of
            # transit_z); the original `< Z_TOL` was tighter than the
            # LIFT_TRANSIT exit and could leave us stuck.
            z_ok  = abs(eef[2] - transit_z) < TRANSIT_LIFT_TOL
            if (xy_ok and z_ok) or self._budget_exceeded():
                self._advance_or_budget("DESCEND")
            return _clip(delta[0]*GAIN, delta[1]*GAIN, delta[2]*GAIN, 1.0), False

        elif self.phase == "DESCEND":
            dz = (PUSH_Z - eef[2]) * GAIN
            if abs(eef[2] - PUSH_Z) < Z_TOL or self._budget_exceeded():
                self._advance_or_budget("PUSH")
            return _clip(0, 0, dz, 1.0), False

        elif self.phase == "PUSH":
            # Drive based on the BLOCK's remaining distance to dest_val
            # (with a small PUSH_OVER bias to ensure full contact through
            # the last few mm). With contact, the block tracks the
            # gripper, so block-distance is the right error term and
            # naturally ramps the command down as the block approaches
            # its target — no more "single big shove" overshoot.
            push_end  = self.dest_val + push_sign * PUSH_OVER
            err       = push_end - bpos[self.axis]

            # Ramped P-controller: cap the per-tick step at PUSH_STEP_CAP,
            # then linearly fade that cap as the block enters PUSH_RAMP_DIST
            # of its target. Far away → full cap; close → gentle crawl.
            ramp = min(abs(err) / PUSH_RAMP_DIST, 1.0)
            move = float(np.clip(err * GAIN, -PUSH_STEP_CAP, PUSH_STEP_CAP)) * ramp
            dx   = move if self.axis == 0 else 0.0
            dy   = move if self.axis == 1 else 0.0

            if (abs(self.dest_val - bpos[self.axis]) < XY_TOL
                    or self._budget_exceeded()):
                self._advance_or_budget("ASCEND")
            return _clip(dx, dy, 0, 1.0), False

        elif self.phase == "ASCEND":
            # Lift past HOVER_Z all the way to transit altitude
            # (HOVER_Z + TRANSIT_LIFT) BEFORE signaling done, so the
            # next task starts with full vertical clearance over the
            # just-aligned block. Doing the lift here (while still
            # directly above the pushed block) is safe; doing it after
            # any lateral motion risks toppling.
            transit_z = HOVER_Z + TRANSIT_LIFT
            dz = (transit_z - eef[2]) * GAIN
            if (eef[2] >= transit_z - TRANSIT_LIFT_TOL
                    or self._budget_exceeded()):
                self.phase = "DONE"
                return _clip(0, 0, dz, 1.0), True
            return _clip(0, 0, dz, 1.0), False

        return _clip(0, 0, 0, 1.0), True  # DONE


# ── X push with lazy Y guard ──────────────────────────────────────────────────

class XPushWithYGuard:
    """
    Defers the Y check until just before the X-push begins, allowing the
    block to fully settle from the prior Y-push. If Y is off, a corrective
    AxisPush(Y) is issued (up to MAX_Y_ATTEMPTS times) before proceeding.
    """

    MAX_Y_ATTEMPTS = 4

    def __init__(self, sim_getter, block_name, target_y, dest_x, verbose=False):
        self._get_sim   = sim_getter
        self.block_name = block_name
        self.target_y   = target_y
        self.dest_x     = dest_x
        self.verbose    = verbose
        self.phase      = "Y_CHECK"
        self._y_corrector = None
        self._x_pusher    = None
        self._y_attempts  = 0

    def _bpos(self):
        return _body_pos(self._get_sim(), self.block_name)

    def act(self, obs):
        if self.phase == "Y_CHECK":
            bp    = self._bpos()
            err   = bp[1] - self.target_y
            if self.verbose:
                print(f"  [YGuard] {self.block_name} | actual_y={bp[1]:.5f} "
                      f"target_y={self.target_y:.5f} err={err:+.5f} "
                      f"| attempt {self._y_attempts}/{self.MAX_Y_ATTEMPTS}")

            if abs(err) <= XY_TOL or self._y_attempts >= self.MAX_Y_ATTEMPTS:
                if self.verbose and abs(err) > XY_TOL:
                    print(f"  [YGuard] {self.block_name} ✗ Y still wrong after "
                          f"{self.MAX_Y_ATTEMPTS} attempts — forcing X-push anyway")
                elif self.verbose:
                    print(f"  [YGuard] {self.block_name} ✓ Y ok — proceeding to X-push")
                self._x_pusher = AxisPush(self._get_sim, self.block_name, axis=0, dest_val=self.dest_x,
                                         verbose=self.verbose)
                self.phase = "X_PUSH"
            else:
                if self.verbose:
                    print(f"  [YGuard] {self.block_name} ✗ Y off by {err:+.5f} "
                          f"— spawning Y corrector (attempt {self._y_attempts + 1})")
                self._y_corrector = AxisPush(self._get_sim, self.block_name, axis=1, dest_val=self.target_y,
                                            verbose=self.verbose)
                self._y_attempts += 1
                self.phase = "Y_FIX"

            return _clip(0, 0, 0, 1.0), False

        elif self.phase == "Y_FIX":
            action, fix_done = self._y_corrector.act(obs)
            if fix_done:
                bp = self._bpos()
                if self.verbose:
                    print(f"  [YGuard] {self.block_name} Y corrector finished "
                          f"| post-fix actual_y={bp[1]:.5f} err={bp[1]-self.target_y:+.5f}")
                self.phase = "Y_CHECK"
            return action, False

        elif self.phase == "X_PUSH":
            action, x_done = self._x_pusher.act(obs)
            if x_done:
                bp = self._bpos()
                if self.verbose:
                    print(f"  [YGuard] {self.block_name} X-push done "
                          f"| final pos=({bp[0]:.5f}, {bp[1]:.5f}) "
                          f"dest_x={self.dest_x:.5f} x_err={bp[0]-self.dest_x:+.5f} "
                          f"y_err={bp[1]-self.target_y:+.5f}")
                self.phase = "DONE"
                return action, True
            return action, False

        return _clip(0, 0, 0, 1.0), True  # DONE


# ── Final chain push in X ─────────────────────────────────────────────────────

class ChainPushX:
    """
    Positions the EEF behind the furthest chain block and pushes purely in X
    until the target reaches the goal. Sets `goal_reached` accordingly.
    """

    def __init__(self, sim_getter, env_getter, furthest_block_name, push_sign, verbose=False):
        self._get_sim   = sim_getter
        self._get_env   = env_getter
        self.block_name = furthest_block_name
        self.push_sign  = push_sign
        self.verbose    = verbose
        self.phase      = "HOVER_BEHIND"
        self.goal_reached = False

    def _bpos(self):
        return _body_pos(self._get_sim(), self.block_name)

    def act(self, obs):
        eef = _eef(obs)

        if self.phase == "HOVER_BEHIND":
            bp     = self._bpos()
            behind = np.array([bp[0] - self.push_sign * PUSH_STANDOFF, bp[1], HOVER_Z])
            delta  = behind - eef
            if np.linalg.norm(eef[:2] - behind[:2]) < XY_TOL and abs(eef[2] - behind[2]) < Z_TOL:
                self.phase = "DESCEND"
            return _clip(delta[0]*GAIN, delta[1]*GAIN, delta[2]*GAIN, 1.0), False

        elif self.phase == "DESCEND":
            dz = (PUSH_Z - eef[2]) * GAIN
            if abs(eef[2] - PUSH_Z) < Z_TOL:
                self.phase = "PUSH"
            return _clip(0, 0, dz, 1.0), False

        elif self.phase == "PUSH":
            sim       = self._get_sim()
            env       = self._get_env()
            target_xy = _target_pos(sim)[:2]
            goal_xy   = _goal_pos(env, sim)[:2]
            goal_dist = np.linalg.norm(target_xy - goal_xy)

            push_end_x = goal_xy[0] + self.push_sign * 0.06
            err_x      = push_end_x - eef[0]
            # Ramped P-controller: cap per-tick step, then fade the cap
            # as goal_dist shrinks so the chain decelerates into the goal
            # rather than slamming the target past it. Use goal_dist for
            # the ramp (not err_x) because err_x can stay large while the
            # target is already nearly home.
            ramp = min(goal_dist / PUSH_RAMP_DIST, 1.0)
            dx   = float(np.clip(err_x * GAIN, -PUSH_STEP_CAP, PUSH_STEP_CAP)) * ramp

            if goal_dist < GOAL_TOL:
                self.goal_reached = True
                self.phase = "ASCEND"
                if self.verbose:
                    print(f"  ChainPushX: goal reached (dist={goal_dist:.4f})")
                return _clip(dx, 0, 0, 1.0), True
            elif abs(err_x) < XY_TOL:
                self.phase = "ASCEND"
                if self.verbose:
                    print(f"  ChainPushX: arm limit reached (goal_dist={goal_dist:.4f}) "
                          f"— handing off to next wave")
                return _clip(dx, 0, 0, 1.0), True
            return _clip(dx, 0, 0, 1.0), False

        elif self.phase == "ASCEND":
            dz = (HOVER_Z - eef[2]) * GAIN
            if abs(eef[2] - HOVER_Z) < Z_TOL:
                self.phase = "DONE"
                return _clip(0, 0, dz, 1.0), True
            return _clip(0, 0, dz, 1.0), False

        return _clip(0, 0, 0, 1.0), True  # DONE


# ── Top-level policy ──────────────────────────────────────────────────────────

class IndirectPushPolicy:
    """
    Multi-wave indirect push. Each wave Y-aligns free blocks, X-positions them
    into a chain behind the target, then drives the chain toward the goal.
    Repeats with remaining blocks until goal is reached or no blocks are left.
    """

    def __init__(self, env=None, rng=None, verbose: bool = False):
        self.env                 = env
        self.rng                 = np.random.default_rng() if rng is None else rng
        self.verbose             = verbose  # print per-wave / per-task progress
        self._tasks              = []
        self._idx                = 0
        self._cur                = None
        self._wave               = 0
        self._used_block_indices = set()
        # Wave building needs a live env (sim handle + goal pose); the
        # driver constructs policies before the env is created and then
        # passes env on every act() call. Defer the first build until
        # act() sees the env.
        self._initialized        = False

    def _sim(self):
        return self.env._env.sim

    @property
    def phase(self) -> str:
        """Human-readable phase label for the collect-demos failure log.
        Other scripted policies use a string state machine; we synthesise
        an equivalent label from wave index + active sub-task class."""
        if not self._initialized:
            return "init"
        if self._cur is None:
            return f"wave{self._wave}/done"
        sub = type(self._cur).__name__
        return f"wave{self._wave}/{sub}/task{self._idx}"

    def _build_wave(self):
        sim      = self._sim()
        all_blks = _all_chain_blocks(self.env)
        t_pos    = _target_pos(sim)
        g_pos    = _goal_pos(self.env, sim)

        target_y  = t_pos[1]
        target_x  = t_pos[0]
        push_sign = int(np.sign(g_pos[0] - target_x))

        self._wave += 1
        if self.verbose:
            print(f"\n=== Wave {self._wave} ===")
            print(f"  target : ({target_x:.4f}, {target_y:.4f})")
            print(f"  goal   : ({g_pos[0]:.4f}, {g_pos[1]:.4f})")
            print(f"  remaining dist: {abs(g_pos[0]-target_x):.4f}  push_sign: {push_sign:+d}")

        free_blks = [(bi, bp) for bi, bp in all_blks if bi not in self._used_block_indices]
        if self.verbose:
            print(f"  {len(free_blks)} free blocks (excluded: {self._used_block_indices})")

        if not free_blks:
            if self.verbose:
                print("  No free blocks left — cannot build another wave.")
            return False

        # Pick a desired chain length per wave (2 or 3, capped by what's
        # available). Free blocks are not deduplicated by X bucket, which
        # would usually collapse the chain to a single block; the slot
        # layout in phase 2 already spaces blocks out at CHAIN_SPACING,
        # so close-X starting positions are fine.
        desired_chain_len = int(self.rng.choice([2, 3]))
        desired_chain_len = max(1, min(desired_chain_len, len(free_blks)))

        # Rank by Y-distance to target (closer = less Y-push work),
        # break ties by X-distance behind the target so the closest
        # candidates fill the chain first.
        ranked = sorted(
            free_blks,
            key=lambda b: (abs(b[1][1] - target_y),
                           push_sign * (target_x - b[1][0])),
        )
        selected = ranked[:desired_chain_len]

        skipped = [bi for bi, _ in free_blks if bi not in {s[0] for s in selected}]
        if self.verbose:
            print(f"  desired_chain_len={desired_chain_len}  "
                  f"Using {len(selected)}/{len(free_blks)} free blocks "
                  f"(skipped: {skipped if skipped else 'none'})")

        slots_x = [target_x - push_sign * CHAIN_SPACING * (i + 1)
                   for i in range(len(selected))]

        # Per-block sequence (furthest-from-target first): align Y, push
        # to the slot in X, then nudge that block forward by
        # NUDGE_FORWARD before moving on to the next block. The nudge
        # progressively compacts the chain so the final ChainPushX has
        # less slack to absorb. The very last block skips the nudge —
        # ChainPushX will drive it (and everything behind the target)
        # the rest of the way.
        sorted_blocks = sorted(selected, key=lambda b: push_sign * b[1][0], reverse=True)
        last_idx = len(sorted_blocks) - 1
        for i, (bi, _) in enumerate(sorted_blocks):
            name = f"chain_{bi}_main"
            self._tasks.append(XPushWithYGuard(self._sim, name, target_y, slots_x[i],
                                               verbose=self.verbose))
            if self.verbose:
                print(f"    → XPushWithYGuard chain_{bi}: y≈{target_y:.4f} then x→{slots_x[i]:.4f}")
            if i < last_idx:
                nudge_dest_x = slots_x[i] + push_sign * NUDGE_FORWARD
                self._tasks.append(AxisPush(self._sim, name, axis=0, dest_val=nudge_dest_x,
                                            verbose=self.verbose))
                if self.verbose:
                    print(f"    → Nudge chain_{bi} forward to x={nudge_dest_x:.4f} "
                          f"(+{NUDGE_FORWARD:.3f} along push)")

        # Final chain push: drive the whole chain into the target/goal.
        furthest_name = f"chain_{sorted_blocks[-1][0]}_main"
        self._tasks.append(ChainPushX(self._sim, lambda: self.env, furthest_name, push_sign,
                                      verbose=self.verbose))
        if self.verbose:
            print(f"    → ChainPushX from behind {furthest_name}")

        for bi, _ in selected:
            self._used_block_indices.add(bi)

        return True

    def reset(self, obs=None):
        self._tasks              = []
        self._idx                = 0
        self._cur                = None
        self._wave               = 0
        self._used_block_indices = set()
        self._initialized        = False

        # If no env yet (driver pattern: reset() before first act()), defer
        # the wave build to the first act() call. Otherwise build now.
        if self.env is None:
            return

        if not self._build_wave():
            if self.verbose:
                print("  WARNING: no usable blocks found on reset.")
            self._initialized = True  # don't keep retrying on every act()
            return

        self._cur         = self._tasks[0]
        self._initialized = True

    def act(self, obs, env=None):
        if env is not None:
            self.env = env
        # Lazy init: build the first wave when we finally have an env.
        if not self._initialized and self.env is not None:
            if not self._build_wave():
                if self.verbose:
                    print("  WARNING: no usable blocks found on first act().")
            elif self._tasks:
                self._cur = self._tasks[0]
            self._initialized = True

        if self._cur is None:
            return _clip(0, 0, 0, 1.0)

        action, task_done = self._cur.act(obs)

        if task_done:
            if isinstance(self._cur, ChainPushX):
                if self._cur.goal_reached:
                    if self.verbose:
                        print("  ✓ Goal reached — all done.")
                    self._cur = None
                    return action
                elif self._idx + 1 >= len(self._tasks):
                    if self.verbose:
                        print("  Target not at goal — attempting next wave...")
                    if not self._build_wave():
                        if self.verbose:
                            print("  ✗ No more blocks available — stopping.")
                        self._cur = None
                        return action

            self._idx += 1
            if self._idx < len(self._tasks):
                self._cur = self._tasks[self._idx]
                task = self._cur
                if isinstance(task, AxisPush):
                    label = f"AxisPush axis={'X' if task.axis==0 else 'Y'} dest={task.dest_val:.4f}"
                else:
                    label = type(task).__name__
                if self.verbose:
                    print(f"  → Task {self._idx}: {label}")
            else:
                if self.verbose:
                    print("  ✓ All tasks complete")
                self._cur = None

        return action

    @property
    def all_done(self):
        return self._cur is None and self._idx >= len(self._tasks)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    env = ib.make("ib-p3-indirect-push-v1", privileged_obs=True, render_mode=None if "--no-render" in sys.argv else "human")

    for ep in range(3):
        ep = 45 * ep + 4
        obs, info = env.reset(seed=ep)
        policy    = IndirectPushPolicy(env)
        policy.reset(obs)

        done = False
        step = 0
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            done = terminated or truncated or policy.all_done
            step += 1

        print(f"\nEp {ep}  steps={step}  reward={reward:.2f}"
              f"  score={info.get('task_score', 0):.3f}\n")

    env.close()