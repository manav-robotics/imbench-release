"""Seesaw Balance — place a counterweight at the correct position to balance a seesaw.

Task ID: ``ib-p3-seesaw-balance-v1``
"""

import logging
import numpy as np
from imbench.tasks.physics import commit_model_edits, set_body_mass
from imbench.tasks.prehensile.contacts import (
    body_geom_ids, geoms_touch, object_geom_ids, robot_contacts,
)
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward

logger = logging.getLogger(__name__)


class SeesawBalanceV1(SingleArmBase):
    TASK_ID = "ib-p3-seesaw-balance-v1"
    DESCRIPTION = (
        "Single arm: a target block sits at a randomised position on "
        "one side of a wide seesaw. Pick the single counterweight and "
        "place it at exactly the right spot on the opposite side so "
        "the plank balances within the tip-angle threshold."
    )

    #: Max horizontal drift (m) of the target block from its spawn arm: the
    #: counterweight must balance the block where it was placed, not one
    #: moved toward the fulcrum.
    TARGET_ARM_TOL = 0.015

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        self._seesaw_length = task_cfg.get("seesaw_length", 0.45)
        self._seesaw_width = task_cfg.get("seesaw_width", 0.14)
        self._seesaw_thickness = task_cfg.get("seesaw_thickness", 0.008)
        self._fulcrum_height = task_cfg.get("fulcrum_height", 0.04)
        self._target_mass = task_cfg.get("target_mass", 0.10)
        self._tip_angle_threshold = task_cfg.get("tip_angle_threshold", 5)
        self._seesaw_damping = task_cfg.get("seesaw_damping", 0.01)
        self._seesaw_frictionloss = task_cfg.get("seesaw_frictionloss", 0.001)
        self._balance_hold_steps = task_cfg.get("balance_hold_steps", 100)
        self._max_seesaw_contact_steps = task_cfg.get(
            "max_seesaw_contact_steps", 100)

        # Randomization ranges
        rand_cfg = self.cfg.get("randomization", {})
        weight_rand = rand_cfg.get("weights", {})
        self._rand_target_mass = weight_rand.get(
            "target_mass_range", [0.08, 0.25])
        self._rand_counter_mass = weight_rand.get(
            "counter_mass_range", [0.08, 0.25])

        # Seesaw color randomization
        seesaw_color_cfg = rand_cfg.get("seesaw_color", {})
        self._randomize_seesaw_color = seesaw_color_cfg.get("enabled", False)

        self._fulcrum_x = None
        self._balance_counter = 0
        self._touched_seesaw = False
        self._seesaw_contact_steps = 0
        self._ideal_counter_arm = 0.0
        self._plank_geoms = set()
        self._object_geoms = {}
        # Refreshed every scoring call (see _compute_task_score).
        self._weight_on_plank = False
        self._weight_placed = False
        self._target_on_plank = False

        self._staged_reward = StagedReward(
            stages=[
                Stage("grasp_weight", 0.15,
                      lambda t, o: 1.0 if t._robot_touches("weight_0") else 0.0),
                Stage("place_near_ideal_position", 0.20,
                      lambda t, o: t._placement_progress()),
                Stage("angle_within_threshold_start", 0.25,
                      lambda t, o: 1.0 if t._balanced_with_counterweight() else 0.0),
                Stage("sustained_hold_complete", 0.40,
                      lambda t, o: t._balance_counter / max(1, t._balance_hold_steps)),
            ],
            success_fn=lambda t, o: t._balance_counter >= t._balance_hold_steps,
        )

    def _make_objects(self):
        objects = []
        # Target block (placed on one side of seesaw at reset)
        objects.append({
            "type": "box", "name": "cube",
            "size": [0.02, 0.02, 0.02],
            "rgba": [0.9, 0.2, 0.2, 1.0],
            "density": self._target_mass / (8 * 0.02 ** 3),
            "friction": [0.8, 0.005, 0.0001],
            "joints": "default",
        })
        # Single counterweight cube to be placed by the robot
        objects.append({
            "type": "box", "name": "weight_0",
            "size": [0.02, 0.02, 0.02],
            "rgba": [0.2, 0.5, 0.8, 1.0],
            "density": 400,  # placeholder, overridden at reset
            "friction": [0.8, 0.005, 0.0001],
            "joints": "default",
        })
        return objects

    def _make_static_objects(self):
        half_len = self._seesaw_length / 2
        half_w = self._seesaw_width / 2
        half_t = self._seesaw_thickness / 2
        return [
            {"type": "seesaw", "name": "seesaw",
             "plank_half_length": half_len,
             "plank_half_width": half_w,
             "plank_half_thickness": half_t,
             "fulcrum_height": self._fulcrum_height,
             "stopper_height": 0.015,
             "plank_rgba": [0.3, 0.7, 0.3, 1.0],
             "plank_friction": [0.5, 0.005, 0.0001],
             "damping": self._seesaw_damping,
             "frictionloss": self._seesaw_frictionloss,
             "joints": None},
        ]

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        sim = self._env.sim
        self._balance_counter = 0
        self._touched_seesaw = False
        self._seesaw_contact_steps = 0
        self._staged_reward.reset()

        # ── Randomize seesaw plank colour ───────────────────────
        if self._randomize_seesaw_color:
            r, g, b = self._rng.uniform(0.2, 0.9, size=3)
            for geom_name in ("seesaw_plank_beam", "seesaw_plank_beam_visual"):
                try:
                    geom_id = sim.model.geom_name2id(geom_name)
                    sim.model.geom_rgba[geom_id] = [r, g, b, 1.0]
                except Exception:
                    pass

        # ── Randomize object (cube / weight) colours ────────────
        if self._rand_object_color:
            for obj_name in ("cube", "weight_0"):
                rgba = self._random_object_color()
                try:
                    geom_id = sim.model.geom_name2id(f"{obj_name}_g0")
                    sim.model.geom_rgba[geom_id] = rgba
                except Exception:
                    pass
                try:
                    geom_id = sim.model.geom_name2id(f"{obj_name}_g0_vis")
                    sim.model.geom_rgba[geom_id] = rgba
                except Exception:
                    pass

        half_len = self._seesaw_length / 2
        half_t = self._seesaw_thickness / 2

        # ── Place seesaw ────────────────────────────────────────
        fulcrum_x = 0.10
        self._fulcrum_x = fulcrum_x
        self._place_static("seesaw", [
            fulcrum_x, 0.0, self._table_height,
        ])

        # ── Sample target arm position on the +x side ───────────
        plank_top_z = (self._table_height + self._fulcrum_height
                       + half_t * 2 + 0.005)
        min_target_arm = 0.04
        max_target_arm = half_len - 0.03
        max_counter_arm = max_target_arm  # symmetric plank
        target_arm = self._rng.uniform(min_target_arm, max_target_arm)

        # ── Sample masses so ideal_counter_arm fits on -x arm ───
        # Constraint: target_mass * target_arm / counter_mass ≤ max_counter_arm
        #   ⇒ target_mass / counter_mass ≤ max_counter_arm / target_arm
        tmin, tmax = self._rand_target_mass
        cmin, cmax = self._rand_counter_mass
        for _ in range(64):
            target_mass = self._rng.uniform(tmin, tmax)
            counter_mass = self._rng.uniform(cmin, cmax)
            ideal = target_mass * target_arm / counter_mass
            if min_target_arm <= ideal <= max_counter_arm:
                break
        else:
            # Fallback: force a counter_mass large enough to fit
            target_mass = self._rng.uniform(tmin, tmax)
            counter_mass = max(
                cmin, target_mass * target_arm / max_counter_arm)
            counter_mass = min(counter_mass, cmax)
            ideal = target_mass * target_arm / counter_mass

        set_body_mass(sim, sim.model.body_name2id("cube_main"), target_mass)
        set_body_mass(sim, sim.model.body_name2id("weight_0_main"), counter_mass)

        target_x = fulcrum_x + target_arm
        self._place_object("cube", [target_x, 0.0, plank_top_z + 0.02])

        # Store for scoring / debugging
        self._target_arm = target_arm
        self._target_mass_actual = target_mass
        self._counter_mass_actual = counter_mass
        self._ideal_counter_arm = ideal

        # ── Place counterweight off the seesaw, accessible to the
        #    robot, with per-episode position randomization ─────
        wz = self._table_height + 0.02 + 0.005
        base_x = fulcrum_x - half_len
        base_y = -(self._seesaw_width / 2 + 0.18)
        dx = self._rng.uniform(-0.08, 0.08)
        dy = self._rng.uniform(-0.05, 0.05)
        self._place_object("weight_0", [base_x + dx, base_y + dy, wz])

        commit_model_edits(sim)
        self._plank_geoms = body_geom_ids(sim, "seesaw_plank")
        self._object_geoms = {name: object_geom_ids(self._env, name) for name in ("cube", "weight_0")}

        obs = self._rebuild_obs()
        return obs, info

    def _get_seesaw_angle_deg(self):
        """Return the absolute tilt angle of the seesaw plank in degrees."""
        sim = self._env.sim
        try:
            jnt_id = sim.model.joint_name2id("seesaw_hinge")
            angle_rad = sim.data.qpos[sim.model.jnt_qposadr[jnt_id]]
            return float(np.degrees(np.abs(angle_rad)))
        except Exception:
            return 90.0

    def _robot_touches(self, name):
        """Whether any robot geom, tactile taxels included, touches object *name*."""
        return robot_contacts(self._env.sim, self._object_geoms[name])

    def _object_on_plank(self, name, far_side):
        """Whether *name* touches the plank on the fulcrum's far (-x,
        counterweight) side or, with ``far_side=False``, its +x side."""
        sim = self._env.sim
        if not geoms_touch(sim, self._object_geoms[name], self._plank_geoms):
            return False
        x = sim.data.body_xpos[sim.model.body_name2id(f"{name}_main")][0]
        return x < self._fulcrum_x if far_side else x > self._fulcrum_x

    def _balanced_with_counterweight(self):
        """Plank level with the target block (at its spawn arm) and the
        counterweight both resting on it, untouched by the robot, on
        opposite sides of the fulcrum."""
        return (self._get_seesaw_angle_deg() <= self._tip_angle_threshold
                and self._weight_placed and self._target_on_plank)

    def _placement_progress(self):
        """Progress toward resting ``weight_0`` at ``_ideal_counter_arm``
        on the fulcrum's far side; zero until it is on the plank there."""
        if not self._weight_on_plank:
            return 0.0
        sim = self._env.sim
        try:
            wid = sim.model.body_name2id("weight_0_main")
            actual_pos = sim.data.body_xpos[wid]
            actual_counter_arm = self._fulcrum_x - actual_pos[0]
            dist = abs(actual_counter_arm - self._ideal_counter_arm)
        except Exception:
            return 0.0
        return 1.0 - np.tanh(15.0 * dist)

    # ── Scoring ─────────────────────────────────────────────────────

    def _compute_task_score(self, raw_obs):
        """Staged reward via the shared ``StagedReward`` engine, saturating
        to 1.0 only when the plank has been held within
        ``_tip_angle_threshold`` of horizontal for ``_balance_hold_steps``
        consecutive steps. Robot-plank contact is allowed up to
        ``_max_seesaw_contact_steps`` cumulative steps; exceeding that
        budget latches failure (hard-zeroed, independent of stage progress).
        """
        if robot_contacts(self._env.sim, self._plank_geoms):
            self._seesaw_contact_steps += 1
            if (self._seesaw_contact_steps > self._max_seesaw_contact_steps
                    and not self._touched_seesaw):
                self._touched_seesaw = True
                logger.info(
                    "[seesaw reward] robot-plank contact budget (%d steps) exceeded → score=0.0",
                    self._max_seesaw_contact_steps,
                )
        if self._touched_seesaw:
            self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
            return 0.0

        # Only count "balanced" frames with the target block still on the
        # plank at its spawn arm and the counterweight released onto its far
        # side, neither touched by the robot — otherwise the robot could hold
        # the plank level through either block, or remove the target block or
        # move it toward the fulcrum.
        self._weight_on_plank = self._object_on_plank("weight_0", far_side=True)
        self._weight_placed = self._weight_on_plank and not self._robot_touches("weight_0")
        sim = self._env.sim
        cube_arm = sim.data.body_xpos[sim.model.body_name2id("cube_main")][0] - self._fulcrum_x
        self._target_on_plank = (
            self._object_on_plank("cube", far_side=False)
            and not self._robot_touches("cube")
            and abs(cube_arm - self._target_arm) <= self.TARGET_ARM_TOL)
        if self._balanced_with_counterweight():
            self._balance_counter += 1
        else:
            self._balance_counter = 0

        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        """Terminate on balance hold, either block falling to the floor, or
        exceeding the cumulative gripper-seesaw contact budget."""
        if self._touched_seesaw:
            return True

        # Counter is advanced in _compute_task_score (runs just before this).
        if self._balance_counter >= self._balance_hold_steps:
            return True

        return (self._object_fell_off_table(raw_obs)
                or self._object_fell_off_table(raw_obs, pos_key="weight_0_pos"))
