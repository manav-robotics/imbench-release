"""Pendulum Grasp — time a grasp to catch a swinging sphere at the apex.

Task ID: ``ib-p2-pendulum-grasp-v2``
"""

import logging

import numpy as np
from imbench.tasks.prehensile.contacts import body_geom_ids, geoms_touch, robot_contacts
from imbench.tasks.prehensile.pendulum_grasp_v1 import _geometry_rng
from imbench.tasks.prehensile.single_arm_base import SingleArmBase
from imbench.tasks.staged_reward import Stage, StagedReward, progress_from_reset

logger = logging.getLogger(__name__)


class PendulumGraspV2(SingleArmBase):
    TASK_ID = "ib-p2-pendulum-grasp-v2"
    DESCRIPTION = (
        "Single arm: a sphere swings on an ideal (undamped) pendulum. "
        "Time the grasp to catch it within 1 cm of the swing apex."
    )

    #: Max |bob z - apex_z| (m) at the catch and at every robot–bob contact
    #: before it: the bob must be intercepted at the apex, not stopped or
    #: grasped elsewhere on its arc and moved there.
    APEX_TOLERANCE = 0.01

    #: Consecutive control steps the bob must stay pinched between the jaws
    #: to count as grasped; a bob bouncing off the jaw tips touches both
    #: fingers for a step or two.
    PINCH_HOLD_STEPS = 10

    # ── "Resting on gripper" success branch ──────────────────────────
    # If the bob ended up balanced on top of the closed gripper (rather
    # than pinched between the jaws), the grasp check fails but we still
    # count it as a success when, for RESTING_HOLD_STEPS consecutive
    # steps (a swing's momentary stop at its apex does not count):
    #   - the bob touches the robot
    #   - the fingers are closed to less than the bob's diameter
    #   - bob's Z is above the EEF site by RESTING_Z_OVER_EEF_MIN
    #   - bob's horizontal offset from the EEF is within RESTING_XY_TOL
    #   - bob is approximately stationary (|v_hinge| < RESTING_VEL_TOL)
    # and every robot–bob contact so far is within APEX_TOLERANCE of apex_z.
    RESTING_Z_OVER_EEF_MIN = 0.0     # m, bob centroid must be ≥ EEF z
    RESTING_XY_TOL         = 0.06    # m, horizontal distance EEF↔bob
    RESTING_VEL_TOL        = 0.05    # rad/s, hinge angular velocity
    RESTING_HOLD_STEPS     = 5       # control steps

    #: Surface clearance at the bottom of the arc, not centre clearance.
    #: The bob centre is one radius below the rod tip, so two radii must
    #: be subtracted from frame_height - arm_length.
    BOB_CLEARANCE_RANGE = (0.03, 0.14)

    def _parse_task_config(self):
        task_cfg = self.cfg.get("task", {})
        rand_cfg = self.cfg.get("randomization", {}).get("pendulum", {})

        # The geometry is baked into the model XML, which robosuite
        # recompiles on every reset. reset() draws it before super().reset()
        # reseeds _rng, so it has its own stream, likewise reseeded only by
        # an explicit seed. The first model gets the first reset()'s draw.
        self._geometry_rng = _geometry_rng(self.seed)
        self._sample_geometry(_geometry_rng(self.seed))

        # Swing amplitude is sampled per-reset from this range.
        self._swing_amplitude_range = task_cfg.get(
            "swing_amplitude_range", [15, 45]
        )
        # Bound the complete swing envelope, including the opposite apex.
        self._max_swing_offset = task_cfg.get("max_swing_offset", 0.30)
        self._randomize_release_side = task_cfg.get("randomize_release_side", True)
        # Damping range is sampled per-reset (mutated on sim.model.dof_damping).
        self._damping_range = rand_cfg.get("pendulum_damping")
        self._bob_shape = task_cfg.get("bob_shape", "sphere")
        self._damping = float(task_cfg.get("pendulum_damping", 0.0))

        self._grasped = False
        self._pinched_steps = 0
        self._resting_steps = 0
        self._contact_apex_err = 0.0
        self._apex_z = None
        self._approach_at_reset = 0.0
        self._bob_geoms = set()
        self._finger_geoms = (set(), set())

        self._staged_reward = StagedReward(
            stages=[
                Stage("approach_pendulum", 0.15, lambda t, o: t._approach_pendulum_progress(o)),
                Stage("grasp_bob", 0.35, lambda t, o: t._grasp_only_progress(o)),
                Stage("apex_timing", 0.50, lambda t, o: t._apex_timing_progress(o)),
            ],
            success_fn=lambda t, o: t._bob_at_apex_grasped(o),
        )

    def _sample_geometry(self, rng):
        """Draw arm length, frame height and bob size from *rng*.

        Each comes from its ``randomization.pendulum`` range when one is
        given, else from ``task``; draws repeat until the bob's lowest point
        clears the table and still passes below the gripper's rest pose.
        """
        task_cfg = self.cfg.get("task", {})
        rand_cfg = self.cfg.get("randomization", {}).get("pendulum", {})

        def _sample_or_default(key, default):
            bounds = rand_cfg.get(key)
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                return float(rng.uniform(*bounds))
            return float(task_cfg.get(key, default))

        lo, hi = self.BOB_CLEARANCE_RANGE
        for _ in range(100):
            self._arm_length = _sample_or_default("pendulum_length", 0.70)
            self._stand_height = _sample_or_default("frame_height", 0.85)
            self._bob_size = _sample_or_default("bob_size", 0.025)
            if lo <= self._stand_height - self._arm_length - 2 * self._bob_size <= hi:
                return
        raise ValueError(
            "pendulum length / frame height / bob size cannot put the bob's "
            f"lowest point {lo}-{hi} m above the table"
        )

    def _sample_swing(self):
        """Sample a reachable amplitude and release side once per episode.

        The bob-centre radius is rod length plus bob radius. Limiting its
        lateral offset couples large angles to shorter rods without changing
        gravity, damping or the apex-grasp success condition.
        """
        low, high = np.radians(self._swing_amplitude_range)
        if not 0 < low <= high < np.pi / 2:
            raise ValueError("swing_amplitude_range must lie between 0 and 90 degrees")
        radius = self._arm_length + self._bob_size
        if self._max_swing_offset <= 0 or radius * np.sin(low) > self._max_swing_offset:
            raise ValueError("minimum swing amplitude exceeds max_swing_offset for this length")
        reachable_high = min(high, np.arcsin(min(1.0, self._max_swing_offset / radius)))
        self._swing_amplitude = float(self._rng.uniform(low, reachable_high))
        self._release_side = int(self._rng.choice([-1, 1])) if self._randomize_release_side else 1
        return self._swing_amplitude

    def _make_objects(self):
        # No free-joint dynamic objects — the bob is part of the pendulum
        return []

    def _make_static_objects(self):
        return [
            {"type": "pendulum", "name": "pendulum",
             "stand_height": self._stand_height,
             "arm_length": self._arm_length,
             "bob_size": self._bob_size,
             "bob_shape": self._bob_shape,
             "bob_rgba": [0.2, 0.2, 0.8, 1.0],
             "damping": self._damping,
             "joints": None},
        ]


    def reset(self, *, seed=None, options=None):
        # This episode's geometry, for the model super().reset() recompiles.
        if seed is not None:
            self._geometry_rng = _geometry_rng(seed)
        self._sample_geometry(self._geometry_rng)
        self._env._static_specs = self._make_static_objects()

        obs, info = super().reset(seed=seed, options=options)
        self._grasped = False
        self._pinched_steps = 0
        self._resting_steps = 0
        self._contact_apex_err = 0.0
        self._staged_reward.reset()

        sim = self._env.sim
        # The task is an ideal pendulum with a fixed energy-derived apex.
        # Robosuite's default air drag otherwise lowers that apex over time,
        # even when joint damping is zero.
        sim.model.opt.density = 0.0
        sim.model.opt.viscosity = 0.0

        stand_x = 0.17 / np.sqrt(2)   # ≈ 0.120
        stand_y = 0.17 / np.sqrt(2)   # ≈ 0.120
        stand_z = self._table_height
        self._place_static("pendulum", [stand_x, stand_y, stand_z])

        # --- Rotate hinge axis to 45° diagonal in XY plane ---
        try:
            joint_id = sim.model.joint_name2id("pendulum_hinge")
            # MuJoCo stores joint axis in jnt_axis as a flat array (n_joints * 3)
            sim.model.jnt_axis[joint_id] = [0.7071068, 0.7071068, 0.0]
        except Exception as e:
            logger.warning("Could not set hinge axis: %s", e)

        amplitude = self._sample_swing()
        # Release from rest at either apex; the energy-derived goal height
        # is identical for both signs.
        try:
            hinge_name = "pendulum_hinge"
            joint_id = sim.model.joint_name2id(hinge_name)
            qpos_addr = sim.model.jnt_qposadr[joint_id]
            dof_addr = sim.model.jnt_dofadr[joint_id]
            sim.data.qpos[qpos_addr] = self._release_side * amplitude
            sim.data.qvel[dof_addr] = 0.0
            if isinstance(self._damping_range, (list, tuple)) and len(self._damping_range) == 2:
                damp = float(self._rng.uniform(*self._damping_range))
                sim.model.dof_damping[dof_addr] = damp
        except Exception:
            pass

        pivot_z = stand_z + self._stand_height
        radius = self._arm_length + self._bob_size
        self._apex_z = pivot_z - radius * np.cos(amplitude)

        # Gripper collision geometry
        for pad_name in [
            "gripper0_right_finger1_pad_collision",
            "gripper0_right_finger2_pad_collision",
        ]:
            try:
                gid = sim.model.geom_name2id(pad_name)
                sim.model.geom_contype[gid] = 1
                sim.model.geom_conaffinity[gid] = 1
            except Exception:
                pass
        for body_name in [
            "gripper0_right_finger1_collision",
            "gripper0_right_finger2_collision",
            "gripper0_right_hand_collision",
        ]:
            try:
                gid = sim.model.geom_name2id(body_name)
                sim.model.geom_conaffinity[gid] = 0
            except Exception:
                pass

        if self._rand_object_color:
            # The rendered geom is the visual duplicate; the collision geom is hidden.
            try:
                gid = sim.model.geom_name2id("pendulum_bob_visual")
                sim.model.geom_rgba[gid] = self._random_object_color()
            except Exception:
                pass

        self._bob_geoms = {sim.model.geom_name2id("pendulum_bob")}
        # Each finger's subtree holds its pad and its tactile taxels.
        self._finger_geoms = tuple(
            body_geom_ids(sim, f"gripper0_right_{side}finger") for side in ("left", "right")
        )

        sim.forward()
        # Approach credit counts from the closest the swinging bob comes to
        # the gripper's reset pose, so holding still earns none.
        eef_pos = sim.data.site_xpos[self._env.robots[0].eef_site_id["right"]]
        closest = np.linalg.norm(self._bob_arc(amplitude) - eef_pos, axis=1).min()
        self._approach_at_reset = self._approach_shaping(closest)
        obs = self._rebuild_obs()
        return obs, info

    def _probe_obs(self):
        # The probe's reset() must not advance the geometry stream either.
        state = self._geometry_rng.bit_generator.state
        obs = super()._probe_obs()
        self._geometry_rng.bit_generator.state = state
        return obs

    def _bob_arc(self, amplitude, num=181):
        """Bob centre positions at *num* hinge angles across ``±amplitude``."""
        sim = self._env.sim
        jid = sim.model.joint_name2id("pendulum_hinge")
        down = np.array([0.0, 0.0, -1.0])
        swing = np.cross(sim.data.xaxis[jid], down)
        theta = np.linspace(-amplitude, amplitude, num)[:, None]
        radius = self._arm_length + self._bob_size
        return sim.data.xanchor[jid] + radius * (np.cos(theta) * down + np.sin(theta) * swing)

    def _get_bob_pos(self):
        """Get the bob geom center position from sim."""
        sim = self._env.sim
        try:
            gid = sim.model.geom_name2id("pendulum_bob")
            return sim.data.geom_xpos[gid].copy()
        except Exception:
            return None

    def _get_bob_quat(self):
        """Get the bob geom orientation as xyzw quat from sim."""
        sim = self._env.sim
        try:
            gid = sim.model.geom_name2id("pendulum_bob")
            xmat = sim.data.geom_xmat[gid].reshape(3, 3)
            from robosuite.utils.transform_utils import mat2quat
            return mat2quat(xmat)  # returns xyzw
        except Exception:
            return None

    def _get_obs(self, raw_obs):
        obs = super()._get_obs(raw_obs)
        bob_pos = self._get_bob_pos()
        if bob_pos is not None:
            obs["object_pos"] = bob_pos
        bob_quat = self._get_bob_quat()
        if bob_quat is not None:
            obs["object_quat"] = bob_quat
        return obs

    def _hinge_velocity(self):
        sim = self._env.sim
        try:
            joint_id = sim.model.joint_name2id("pendulum_hinge")
            return abs(float(sim.data.qvel[sim.model.jnt_dofadr[joint_id]]))
        except Exception:
            return float("inf")

    def _robot_touches_bob(self):
        return robot_contacts(self._env.sim, self._bob_geoms)

    def _bob_pinched(self):
        """Whether the bob is between the jaws, touching both fingers (pad or
        tactile taxels) and holding them at least a bob radius apart.

        Shut jaws that the bob bumps from outside touch it with both fingers
        too.
        """
        sim = self._env.sim
        if not all(geoms_touch(sim, finger, self._bob_geoms) for finger in self._finger_geoms):
            return False
        left, right = (
            sim.data.body_xpos[sim.model.body_name2id(f"gripper0_right_{side}finger")]
            for side in ("left", "right")
        )
        closing = left - right
        span = float(np.linalg.norm(closing))
        if span < self._bob_size:
            return False
        along = float(np.dot(self._get_bob_pos() - right, closing)) / span**2
        return 0.0 < along < 1.0

    def _bob_resting_on_gripper(self, raw_obs):
        """Instantaneous "resting on gripper" condition (see class constants)."""
        bob_pos = self._get_bob_pos()
        eef_pos = raw_obs.get("robot0_eef_pos")
        finger_qpos = raw_obs.get("robot0_gripper_qpos")
        if bob_pos is None or eef_pos is None or finger_qpos is None:
            return False
        return (
            float(finger_qpos[0] - finger_qpos[1]) < 2 * self._bob_size
            and float(bob_pos[2] - eef_pos[2]) >= self.RESTING_Z_OVER_EEF_MIN
            and float(np.linalg.norm(bob_pos[:2] - eef_pos[:2])) <= self.RESTING_XY_TOL
            and self._hinge_velocity() <= self.RESTING_VEL_TOL
            and self._robot_touches_bob()
        )

    @staticmethod
    def _approach_shaping(dist):
        return float(1.0 - np.tanh(10.0 * dist))

    def _approach_pendulum_progress(self, raw_obs):
        bob_pos = self._get_bob_pos()
        eef_pos = raw_obs.get("robot0_eef_pos")
        if bob_pos is None or eef_pos is None:
            return 0.0
        dist = float(np.linalg.norm(bob_pos - eef_pos))
        return progress_from_reset(self._approach_shaping(dist), self._approach_at_reset)

    def _grasp_only_progress(self, raw_obs):
        """Either grasp mode (pinched or resting) held, without the apex check."""
        grasped = (self._pinched_steps >= self.PINCH_HOLD_STEPS
                   or self._resting_steps >= self.RESTING_HOLD_STEPS)
        return 1.0 if grasped else 0.0

    def _apex_timing_progress(self, raw_obs):
        """While grasped, how close to the apex every robot–bob contact was."""
        if not self._grasp_only_progress(raw_obs):
            return 0.0
        return float(max(0.0, 1.0 - self._contact_apex_err / (4 * self.APEX_TOLERANCE)))

    def _bob_at_apex_grasped(self, raw_obs):
        """Bob held pinched or resting on the closed jaws, every robot contact
        with it so far within ``APEX_TOLERANCE`` of the apex height.

        Sets ``self._grasped`` as a side effect (read by
        ``_check_termination``).
        """
        if (self._contact_apex_err > self.APEX_TOLERANCE
                or not self._grasp_only_progress(raw_obs)):
            return False
        self._grasped = True
        return True

    def _compute_task_score(self, raw_obs):
        if self._robot_touches_bob():
            self._contact_apex_err = max(
                self._contact_apex_err, abs(float(self._get_bob_pos()[2]) - self._apex_z))
        self._pinched_steps = self._pinched_steps + 1 if self._bob_pinched() else 0
        if self._bob_resting_on_gripper(raw_obs):
            self._resting_steps += 1
        else:
            self._resting_steps = 0
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    def _check_termination(self, raw_obs):
        # A contact off the apex rules success out, but the episode runs on:
        # a later grasp still earns grasp_bob and partial apex_timing credit.
        return bool(self._grasped)
