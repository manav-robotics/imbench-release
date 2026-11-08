"""Test script — Franka picks up the ib-p2-cube-toss-v1 cube where it spawns.

Policy phases:
    1. OPEN_GRIPPER — Open the gripper fully before moving.
    2. ALIGN_Z      — Move so TCP z aligns with the centre of the cube.
    3. APPROACH     — Move toward the cube in XY so TCP surrounds it.
    4. GRASP        — Close gripper to grab the cube.
    5. LIFT         — Move upward by 0.3m while holding the cube.
    6. DROP         — Open gripper to release/drop the cube.
    7. DONE         — Hold position.

Opens the MuJoCo viewer and the observation dashboard (needs a display).
Press ``q`` in the dashboard to quit.
"""

import os
import sys

import numpy as np
import imbench as ib
from imbench.policy.base_policy import PolicyInterface
from imbench.utils.obs_visualizer import ObsDashboard


# Slowdown factor — all gains divided by this, all durations multiplied
SLOW_FACTOR = 3.0

# Nominal cube centre height on the table; the policy tracks obs["object_pos"]
CUBE_CENTRE_Z = 0.82


class PickLiftPolicy(PolicyInterface):
    """Hard-coded cartesian policy: open → align_z → approach → grasp → lift → drop."""

    # Phase durations (steps), 3x slower
    OPEN_STEPS = int(20 * SLOW_FACTOR)
    ALIGN_Z_STEPS = int(60 * SLOW_FACTOR)
    APPROACH_STEPS = int(60 * SLOW_FACTOR)
    GRASP_STEPS = int(30 * SLOW_FACTOR)
    LIFT_STEPS = int(80 * SLOW_FACTOR)
    DROP_STEPS = int(20 * SLOW_FACTOR)

    # Lift offset: move up by 0.3m from grasp height
    LIFT_OFFSET = 0.3

    # Cartesian gain (slower by SLOW_FACTOR)
    GAIN = 5.0 / SLOW_FACTOR

    def __init__(self):
        self.phase = "open_gripper"
        self.step_in_phase = 0
        self._grasp_z = CUBE_CENTRE_Z

    def reset(self):
        self.phase = "open_gripper"
        self.step_in_phase = 0
        self._grasp_z = CUBE_CENTRE_Z

    def _advance(self, next_phase):
        self.phase = next_phase
        self.step_in_phase = 0

    def act(self, obs: dict) -> np.ndarray:
        # action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        # gripper: < 0 opens, > 0 closes (sign only)
        action = np.zeros(7)

        eef_pos = obs["robot0_eef_pos"]
        cube_pos = obs["object_pos"]

        if self.phase == "open_gripper":
            # Just open the gripper, no motion
            action[6] = -1.0
            if self.step_in_phase >= self.OPEN_STEPS:
                self._advance("align_z")

        elif self.phase == "align_z":
            # Move TCP z to match the cube centre z, keep XY, gripper open
            target_z = cube_pos[2]
            dz = target_z - eef_pos[2]
            action[2] = np.clip(dz * self.GAIN, -1.0, 1.0)
            action[6] = -1.0  # keep gripper open
            if self.step_in_phase >= self.ALIGN_Z_STEPS or abs(dz) < 0.005:
                self._advance("approach")

        elif self.phase == "approach":
            # Move toward cube in XY (z stays at cube centre), gripper open
            target = np.array([cube_pos[0], cube_pos[1], cube_pos[2]])
            delta = target - eef_pos
            action[:3] = np.clip(delta * self.GAIN, -1.0, 1.0)
            action[6] = -1.0  # keep gripper open
            if self.step_in_phase >= self.APPROACH_STEPS or np.linalg.norm(delta) < 0.005:
                self._advance("grasp")

        elif self.phase == "grasp":
            # Close gripper to grab the cube
            action[6] = 1.0
            # Record grasp height so we can lift relative to it
            if self.step_in_phase == 0:
                self._grasp_z = eef_pos[2]
            if self.step_in_phase >= self.GRASP_STEPS:
                self._advance("lift")

        elif self.phase == "lift":
            # Move upward by 0.3m from where we grasped
            target_z = self._grasp_z + self.LIFT_OFFSET
            dz = target_z - eef_pos[2]
            action[2] = np.clip(dz * self.GAIN, -1.0, 1.0)
            action[6] = 1.0  # keep gripper closed
            if self.step_in_phase >= self.LIFT_STEPS or abs(dz) < 0.005:
                self._advance("drop")

        elif self.phase == "drop":
            # Open gripper to release the cube
            action[6] = -1.0
            if self.step_in_phase >= self.DROP_STEPS:
                self._advance("done")

        elif self.phase == "done":
            action[6] = -1.0  # gripper stays open

        self.step_in_phase += 1
        return action


if __name__ == "__main__":
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        sys.exit("This script opens the MuJoCo viewer and needs a display, but DISPLAY is not set.")
    env = ib.make("ib-p2-cube-toss-v1", privileged_obs=True, render_mode="human")
    policy = PickLiftPolicy()
    dash = ObsDashboard()

    n_episodes = 3
    quit_requested = False
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        policy.reset()
        dash.reset()
        dash.update(obs, info)
        done = False

        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            key = dash.update(obs, info, reward=reward)
            if key == ord('q'):
                quit_requested = True
                break
            done = terminated or truncated

        print(f"Episode {ep}  steps={info['step']}  "
              f"score={info['task_score']:.3f}  "
              f"phase={policy.phase}")
        if quit_requested:
            break

    dash.close()
    env.close()
