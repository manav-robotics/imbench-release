"""Scripted single-arm policy for ``ib-p1-cracker-box-jaw-v1``.

Task
----
Jaw-only variant of the cracker-box slot-insertion task.  Both arms are in
the scene but the **suction (R1) arm is frozen** — only the finger arm (R0)
can move.

From YAML:
    grippers: [PandaGripper, PandaSuctionGripper]   (R0=jaw active, R1=suction frozen)
    payload: cracker_box
    fixture: slot

Strategy (only R0 acts; R1 action is always held in place and frozen by env)
--------
  raise             : R0 raises its wrist before moving toward the upright box
    → above         : R0 moves above the box, top-down, jaw open
    → descend       : R0 descends to the upper part of the box
    → grasp         : R0 closes its jaw and waits for a two-finger grasp
    → lift          : R0 lifts the box to safe height
    → rotate        : R0 aligns the long axis with the slot, held end tilted down
    → carry         : R0 centres the box above the slot opening
    → lower         : R0 lowers the box toward the slot
    → release       : R0 opens its jaw above the opening
    → withdraw      : R0 moves clear while the box settles

The task ends the episode when the released box settles inside the slot.
A lost grasp or an unreachable target ends the attempt as a failure.
R1 emits zero actions throughout.  CrackerBoxJawV1.step() freezes R1 anyway
(via the CrackerBoxSuctionV1 slice logic with _FROZEN_GRIPPER_KEYWORD = "suction").
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from imbench.policy.base_policy import PolicyInterface
from data_synthetic.hetbi_scripted._utils import (
    TABLE_Z, TOP_DOWN_QUAT, move_to, quat_err_axis_angle,
)


class CrackerBoxJawPolicy(PolicyInterface):
    """Scripted jaw-only pick-and-slot policy.

    Only R0 (finger) moves.  R1 emits zero actions (env freezes it anyway).
    Grasp and pose checks gate each phase; a timeout ends a failed attempt.
    """

    def __init__(self, table_z: float = TABLE_Z):
        self._tz = table_z
        self.reset()

    def reset(self):
        self.phase = "raise"
        self.tick = 0
        self.done = False
        self.failed = False
        self.failure_reason = None
        self._grasp_steps = 0
        self._lost_steps = 0
        self._quat = TOP_DOWN_QUAT.copy()
        self._lift_xy = None
        self._withdraw_from = None

    def _advance(self, phase):
        self.phase, self.tick = phase, 0

    def _fail(self, reason):
        self.failed = self.done = True
        self.failure_reason = reason
        self._advance("failed")

    def act(self, obs: dict, env=None) -> np.ndarray:
        if env is None:
            raise ValueError("CrackerBoxJawPolicy requires the task for object poses and grasp checks")
        arm = env._active_robot_idx
        base = 7 * arm
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        if self.done:
            return action
        self.tick += 1
        # ── Live object and grasp state ─────────────────────────────────────
        sim = env._env.sim
        box_id = env._env._payload_body_id
        box = np.array(sim.data.body_xpos[box_id])
        box_rot = Rotation.from_matrix(sim.data.body_xmat[box_id].reshape(3, 3))
        pos = np.asarray(obs[f"robot{arm}_eef_pos"])
        quat = np.asarray(obs[f"robot{arm}_eef_quat"])
        centre = env._slot_centre()
        grasped = env._check_grasp(
            gripper=env._arm_gripper(arm), object_geoms=env._env._payload.contact_geoms,
        )
        target = pos.copy()
        grip = -1.0
        next_phase = None
        # ── Targets for R0 ─────────────────────────────────────────────────
        p = self.phase
        if p == "raise":
            target[2] = self._tz + 0.38
            if abs(pos[2] - target[2]) < 0.012:
                next_phase = "above"
        elif p == "above":
            # The upright box extends 108 mm above its body origin.  Moving
            # sideways at centre + 100 mm hits its top edge.
            target = box + [0, 0, 0.24]
            if np.linalg.norm(pos - target) < 0.012:
                next_phase = "descend"
        elif p in ("descend", "grasp"):
            target = box + [0, 0, 0.085]
            if p == "descend" and np.linalg.norm(pos - target) < 0.015:
                next_phase = "grasp"
            if p == "grasp":
                grip = 1.0
                self._grasp_steps = self._grasp_steps + 1 if grasped else 0
                if self._grasp_steps >= 20:
                    self._lift_xy = pos[:2].copy()
                    next_phase = "lift"
        elif p in ("lift", "rotate"):
            grip = 1.0
            target = np.r_[self._lift_xy, self._tz + 0.48]
            if p == "lift" and abs(pos[2] - target[2]) < 0.01 and grasped:
                # Thin axis up, long axis along the channel.  Tilting the
                # held end downward avoids leaving the far end balanced
                # across a side wall when the fingers open.
                fixture = sim.data.body_xmat[env._env._fixture_body_id].reshape(3, 3)
                long_axis = fixture[:, 0]
                up = np.array([0., 0., 1.])
                flat = Rotation.from_matrix(np.column_stack((up, np.cross(long_axis, up), long_axis)))
                tilt = Rotation.from_rotvec(np.cross(up, long_axis) * np.deg2rad(30))
                self._quat = (tilt * flat * box_rot.inv() * Rotation.from_quat(quat)).as_quat()
                next_phase = "rotate"
            elif p == "rotate" and np.linalg.norm(quat_err_axis_angle(quat, self._quat)) < 0.04:
                next_phase = "carry"
        elif p in ("carry", "lower"):
            grip = 1.0
            goal = centre + [0, 0, 0.28 if p == "carry" else 0.24]
            target = pos + goal - box  # centre the box, rather than the wrist
            if np.linalg.norm(box - goal) < 0.01:
                next_phase = "lower" if p == "carry" else "release"
        elif p == "release":
            if self.tick >= 51:
                self._withdraw_from = pos.copy()
                next_phase = "withdraw"
        elif p == "withdraw":
            target = self._withdraw_from + [0.18, 0, 0.18]
            if self.tick >= 300:
                self._fail("box did not settle inside the slot")

        if p in ("lift", "rotate", "carry", "lower"):
            self._lost_steps = 0 if grasped else self._lost_steps + 1
            if self._lost_steps >= 20:
                self._fail("lost the box during transport")
        if self.tick >= (250 if p == "grasp" else 1000):
            self._fail(f"{p} target was not reached")
        if self.failed:
            return action
        action[base:base + 7] = move_to(
            pos, quat, target, self._quat, grip,
            gain_xy=12, gain_z=12, max_step=0.6,
        )
        if next_phase is not None:
            self._advance(next_phase)
        return action
