import numpy as np
import imbench as ib
from imbench.policy.base_policy import PolicyInterface
from scipy.spatial.transform import Rotation


# ── Geometry ──────────────────────────────────────────────────────────
TABLE_HEIGHT     = 0.8
BOOK_HALF_HEIGHT = 0.0015

PUSH_Z               = TABLE_HEIGHT + 0.012
TABLE_EDGE_Y         = 0.25
BOOK_HALF_LEN        = 0.1
OVERHANG_FRAC        = 0.5
PUSH_TARGET_Y        = TABLE_EDGE_Y - BOOK_HALF_LEN * (1.0 - OVERHANG_FRAC)
PUSH_APPROACH_OFFSET = 0.15
RISE_HEIGHT          = 0.30   
RETRACE_PULLBACK     = 0.20

# ── Grasp geometry ────────────────────────────────────────────────────
# Approach from +Y toward -Y; fingers clamp the thin Z dimension of the book.
# eef_z → world -Y, eef_y → world +Z  →  Rx(+90°): [x,y,z,w] = [0.7071, 0, 0, 0.7071]
GRASP_QUAT = np.array([0.7071, 0.0, 0.0, 0.7071])

GRASP_APPROACH_Y_OFFSET = 0.14  
GRASP_Z                 = TABLE_HEIGHT + BOOK_HALF_HEIGHT
DESCEND_Y_OFFSET        = 0.05  
LIFT_TARGET_Z           = TABLE_HEIGHT + 0.25

# ── Control gains ─────────────────────────────────────────────────────
GAIN        = 4.0
ROT_GAIN    = 3.0
SLOW_FACTOR = 4.0


# ── Helpers ───────────────────────────────────────────────────────────

def quat_to_yaw(quat_xyzw: np.ndarray) -> float:
    return Rotation.from_quat(quat_xyzw).as_euler('xyz')[2]


def yaw_error_to_target(current_yaw: float, target_yaw: float) -> float:
    err = target_yaw - current_yaw
    return (err + np.pi) % (2 * np.pi) - np.pi


def orientation_error(current_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    r_err = Rotation.from_quat(target_quat) * Rotation.from_quat(current_quat).inv()
    return r_err.as_rotvec()


class EdgeSlidePushPolicy(PolicyInterface):

    ALIGN_ROT_STEPS  = int(60  * SLOW_FACTOR)
    ALIGN_PUSH_STEPS = int(100 * SLOW_FACTOR)
    PUSH_STEPS       = int(200 * SLOW_FACTOR)
    RETRACE_STEPS    = int(120 * SLOW_FACTOR)  # increased to accommodate the rise+retrace+descend
    RISE_STEPS       = int(60  * SLOW_FACTOR)
    PRE_GRASP_STEPS  = int(120 * SLOW_FACTOR)
    DESCEND_STEPS    = int(100 * SLOW_FACTOR)
    CLOSE_STEPS      = int(60  * SLOW_FACTOR)
    LIFT_STEPS       = int(100 * SLOW_FACTOR)
    # Add these step constants inside the class, alongside the others:
    LATERAL_ALIGN_STEPS = int(120 * SLOW_FACTOR)
    LATERAL_PUSH_STEPS  = int(200 * SLOW_FACTOR)
    LATERAL_RISE_STEPS  = int(60  * SLOW_FACTOR)

    MAX_RETRIES = 4

    def __init__(self):
        self.phase = "align_rotation"
        self.step_in_phase = 0
        self._retries = 0
        self._rise_z_start = PUSH_Z
        self._right_lateral_rise_z_start = PUSH_Z
        self._lateral_rise_z_start = PUSH_Z

    def reset(self):
        self.phase = "align_rotation"
        self.step_in_phase = 0
        self._retries = 0
        self._rise_z_start = PUSH_Z
        self._right_lateral_rise_z_start = PUSH_Z
        self._lateral_rise_z_start = PUSH_Z

    def _advance(self, next_phase: str):
        print(f"  [{self.step_in_phase:4d} steps]  {self.phase}  →  {next_phase}")
        self.phase = next_phase
        self.step_in_phase = 0

    def _overhang_confirmed(self, obj_pos: np.ndarray) -> bool:
        return obj_pos[1] >= PUSH_TARGET_Y

    def act(self, obs: dict, env=None) -> np.ndarray:
        action = np.zeros(7)
        action[6] = -1.0   # default: gripper open

        eef_pos  = obs["robot0_eef_pos"]
        eef_quat = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1]))
        obj_pos  = obs["object_pos"]
        obj_quat = obs["object_quat"]

        if self.phase == "align_rotation":
            book_yaw      = quat_to_yaw(obj_quat)
            book_yaw_norm = book_yaw % np.pi
            yaw_err       = yaw_error_to_target(book_yaw_norm, np.pi / 2)

            if abs(yaw_err) < 0.15 or self.step_in_phase >= self.ALIGN_ROT_STEPS:
                self._advance("align_to_push")
            else:
                side = np.sign(yaw_err)
                nudge_target = np.array([
                    obj_pos[0] + side * 0.06,
                    obj_pos[1] - 0.04,
                    PUSH_Z + 0.1,
                ])
                delta = nudge_target - eef_pos
                action[:3] = np.clip(delta * GAIN, -0.25, 0.25)
                action[1] += float(np.clip(0.15 * side, -0.15, 0.15))

        elif self.phase == "align_to_push":
            target = np.array([obj_pos[0], obj_pos[1] - PUSH_APPROACH_OFFSET - 0.05, PUSH_Z])
            delta  = target - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)
            action[6] = 1.0

            reached = (abs(delta[0]) < 0.012 and
                       abs(delta[1]) < 0.012 and
                       abs(delta[2]) < 0.008)
            if reached or self.step_in_phase >= self.ALIGN_PUSH_STEPS:
                self._advance("push_to_edge")

        elif self.phase == "push_to_edge":
            if self._overhang_confirmed(obj_pos):
                self._rise_z_start = eef_pos[2]
                self._advance("rise_and_stop")
            else:
                push_y_target = PUSH_TARGET_Y - PUSH_APPROACH_OFFSET + 0.30
                delta = np.array([obj_pos[0], push_y_target, PUSH_Z]) - eef_pos
                action[:3] = np.clip(delta * GAIN, -0.5, 0.5)
                action[6] = 1

                gripper_past_target = eef_pos[1] > (PUSH_TARGET_Y - 0.02)
                if gripper_past_target and not self._overhang_confirmed(obj_pos):
                    if self._retries < self.MAX_RETRIES:
                        self._retries += 1
                        self._advance("retrace")
                    else:
                        self._rise_z_start = eef_pos[2]
                        self._advance("rise_and_stop")
                elif self.step_in_phase >= self.PUSH_STEPS:
                    if self._overhang_confirmed(obj_pos):
                        self._rise_z_start = eef_pos[2]
                        self._advance("rise_and_stop")
                    elif self._retries < self.MAX_RETRIES:
                        self._retries += 1
                        self._advance("retrace")
                    else:
                        self._rise_z_start = eef_pos[2]
                        self._advance("rise_and_stop")

        elif self.phase == "retrace":
            # Step 1: rise up a little to clear the book
            # Step 2: at the risen height, move back to the approach XY position
            # Then hand off to retrace_descend to come back down before pushing again
            RETRACE_RISE_Z = PUSH_Z + 0.06
            action[6] = 1
            if eef_pos[2] < RETRACE_RISE_Z - 0.008:
                dz = RETRACE_RISE_Z - eef_pos[2]
                action[2] = float(np.clip(dz * GAIN, -0.5, 0.5))
            else:
                target_xy = np.array([obj_pos[0], obj_pos[1] - PUSH_APPROACH_OFFSET, RETRACE_RISE_Z])
                delta = target_xy - eef_pos
                action[:3] = np.clip(delta * GAIN, -0.5, 0.5)
                xy_reached = np.linalg.norm(delta[:2]) < 0.015
                if xy_reached or self.step_in_phase >= self.RETRACE_STEPS:
                    self._advance("retrace_descend")

        elif self.phase == "retrace_descend":
            # Descend from the risen retrace height back down to PUSH_Z before pushing again
            target = np.array([obj_pos[0], obj_pos[1] - PUSH_APPROACH_OFFSET, PUSH_Z])
            action[6] = 1
            delta  = target - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)
            if abs(delta[2]) < 0.008 or self.step_in_phase >= self.RETRACE_STEPS:
                self._advance("push_to_edge")

        elif self.phase == "rise_and_stop":
            target_z = self._rise_z_start + RISE_HEIGHT
            dz = target_z - eef_pos[2]
            action[2] = float(np.clip(dz * GAIN, -0.5, 0.5))
            action[6] = 1

            if abs(dz) < 0.005 or self.step_in_phase >= self.RISE_STEPS:
                # Branch: if object is on the +X side, push it in -X first
                if obj_pos[0] > 0.08:
                    self._advance("upper_lateral_align")
                # Branch: if object is still on the -X side, do the lateral push first
                elif obj_pos[0] < -0.05:
                    self._advance("lower_lateral_align")
                else:
                    self._advance("move_to_pre_grasp_xy")

        # ── Right lateral phases (mirror of lateral_*, pushing from +X toward -X) ──

        elif self.phase == "upper_lateral_align":
            # Position EEF at the +X side of the object, at risen height, ready to push in -X
            target = np.array([
                obj_pos[0] + PUSH_APPROACH_OFFSET,   # behind object in +X
                obj_pos[1],                            # aligned in Y
                eef_pos[2],                            # stay at risen height
            ])
            delta = target - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)

            reached = np.linalg.norm(delta) < 0.015
            if reached or self.step_in_phase >= self.LATERAL_ALIGN_STEPS:
                self._advance("upper_lateral_descend")

        elif self.phase == "upper_lateral_descend":
            # Descend to PUSH_Z while staying at the +X approach position
            target = np.array([
                obj_pos[0] + PUSH_APPROACH_OFFSET - 0.05,   # slightly closer in -X
                obj_pos[1],                                   # aligned in Y
                PUSH_Z,
            ])
            delta = target - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)

            reached = np.linalg.norm(delta) < 0.015
            if reached or self.step_in_phase >= self.LATERAL_ALIGN_STEPS:
                self._advance("upper_lateral_push")

        elif self.phase == "upper_lateral_push":
            # Push in -X until obj_pos[0] < 0.1
            push_x_target = eef_pos[0] - PUSH_APPROACH_OFFSET + 0.05
            delta = np.array([push_x_target, obj_pos[1], eef_pos[2]]) - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)

            obj_reached = obj_pos[0] <= 0.1
            eef_past    = eef_pos[0] <= push_x_target + 0.02
            if obj_reached or eef_past or self.step_in_phase >= self.LATERAL_PUSH_STEPS:
                self._right_lateral_rise_z_start = eef_pos[2]
                self._advance("upper_lateral_rise")

        elif self.phase == "upper_lateral_rise":
            # Rise again so we can approach from above cleanly for the grasp
            target_z = self._right_lateral_rise_z_start + RISE_HEIGHT + 0.05
            dz = target_z - eef_pos[2]
            action[2] = float(np.clip(dz * GAIN, -0.5, 0.5))

            if abs(dz) < 0.005 or self.step_in_phase >= self.LATERAL_RISE_STEPS:
                self._advance("move_to_pre_grasp_xy")

        # ── Lower lateral push (between rise_and_stop and move_to_pre_grasp_xy) ──

        elif self.phase == "lower_lateral_align":
            # Position the EEF at the -X side of the object, at current risen height, ready to push in +X
            target = np.array([
                obj_pos[0] - PUSH_APPROACH_OFFSET,   # behind object in -X
                obj_pos[1],                            # aligned in Y
                eef_pos[2],                            # stay at risen height
            ])
            delta = target - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)

            reached = np.linalg.norm(delta) < 0.015
            if reached or self.step_in_phase >= self.LATERAL_ALIGN_STEPS:
                self._advance("lower_lateral_descend")
        elif self.phase == "lower_lateral_descend":
            # Position the EEF at the -X side of the object, at current risen height, ready to push in +X
            target = np.array([
                obj_pos[0] - PUSH_APPROACH_OFFSET + 0.05,   # behind object in -X
                obj_pos[1],                            # aligned in Y
                PUSH_Z,                           
            ])
            delta = target - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)

            reached = np.linalg.norm(delta) < 0.015
            if reached or self.step_in_phase >= self.LATERAL_ALIGN_STEPS:
                self._advance("lower_lateral_push")
        elif self.phase == "lower_lateral_push":
            # Push in +X until object crosses X = 0
            push_x_target = eef_pos[0] + PUSH_APPROACH_OFFSET - 0.1   
            delta = np.array([push_x_target, obj_pos[1], eef_pos[2]]) - eef_pos
            action[:3] = np.clip(delta * GAIN, -0.5, 0.5)

            obj_reached = obj_pos[0] >= -0.05
            eef_past    = eef_pos[0] >= push_x_target - 0.02
            if obj_reached or eef_past or self.step_in_phase >= self.LATERAL_PUSH_STEPS:
                self._lateral_rise_z_start = eef_pos[2]
                self._advance("lower_lateral_rise")

        elif self.phase == "lower_lateral_rise":
            # Rise again so we can approach from above cleanly for the grasp
            target_z = self._lateral_rise_z_start + RISE_HEIGHT
            dz = target_z - eef_pos[2]
            action[2] = float(np.clip(dz * GAIN, -0.5, 0.5))

            if abs(dz) < 0.005 or self.step_in_phase >= self.LATERAL_RISE_STEPS:
                self._advance("move_to_pre_grasp_xy")

        elif self.phase == "move_to_pre_grasp_xy":
            # At the risen height, adopt grasp orientation and move to the XY approach position
            pre_grasp_xy_pos = np.array([
                obj_pos[0],
                obj_pos[1] + GRASP_APPROACH_Y_OFFSET,
                eef_pos[2],  # stay at current risen height
            ])
            pos_delta   = pre_grasp_xy_pos - eef_pos
            ori_err     = orientation_error(eef_quat, GRASP_QUAT)
            action[:3]  = np.clip(pos_delta * GAIN, -0.5, 0.5)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)

            pos_reached = np.linalg.norm(pos_delta) < 0.015
            ori_reached = np.linalg.norm(ori_err)   < 0.10
            if (pos_reached and ori_reached) or self.step_in_phase >= self.PRE_GRASP_STEPS:
                self._advance("pre_grasp")

        elif self.phase == "pre_grasp":
            # Descend straight down to GRASP_Z (XY is already correct)
            pre_grasp_pos = np.array([
                obj_pos[0],
                obj_pos[1] + GRASP_APPROACH_Y_OFFSET,
                GRASP_Z,
            ])
            pos_delta   = pre_grasp_pos - eef_pos
            ori_err     = orientation_error(eef_quat, GRASP_QUAT)
            action[:3]  = np.clip(pos_delta * GAIN, -0.5, 0.5)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.5, 0.5)

            pos_reached = np.linalg.norm(pos_delta) < 0.015
            ori_reached = np.linalg.norm(ori_err)   < 0.10
            if (pos_reached and ori_reached) or self.step_in_phase >= self.PRE_GRASP_STEPS:
                self._advance("descend_grasp")

        elif self.phase == "descend_grasp":
            descend_pos = np.array([obj_pos[0], obj_pos[1] + DESCEND_Y_OFFSET, GRASP_Z])
            pos_delta   = descend_pos - eef_pos
            ori_err     = orientation_error(eef_quat, GRASP_QUAT)
            action[:3]  = np.clip(pos_delta * GAIN, -0.4, 0.4)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.4, 0.4)

            if np.linalg.norm(pos_delta) < 0.012 or self.step_in_phase >= self.DESCEND_STEPS:
                self._advance("close_gripper")

        elif self.phase == "close_gripper":
            ori_err     = orientation_error(eef_quat, GRASP_QUAT)
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.25, 0.25)
            action[6]   = +1.0

            if self.step_in_phase >= self.CLOSE_STEPS:
                self._advance("lift")

        elif self.phase == "lift":
            dz          = LIFT_TARGET_Z - eef_pos[2]
            ori_err     = orientation_error(eef_quat, GRASP_QUAT)
            action[2]   = float(np.clip(dz * GAIN, -0.5, 0.5))
            action[3:6] = np.clip(ori_err * ROT_GAIN, -0.25, 0.25)
            action[6]   = +1.0

            if abs(dz) < 0.008 or self.step_in_phase >= self.LIFT_STEPS:
                self._advance("done")

        elif self.phase == "done":
            action[6] = +1.0

        self.step_in_phase += 1
        return action


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    import cv2
    from pynput import keyboard as kb

    _quit = False

    def on_press(key):
        global _quit
        if hasattr(key, "char") and key.char == "q":
            _quit = True

    listener = kb.Listener(on_press=on_press)
    listener.start()

    def check_q():
        return _quit

    env    = ib.make("ib-p1-edge-slide-v1", privileged_obs=True, render_mode="human")
    policy = EdgeSlidePushPolicy()
    quit_all = False

    for ep in range(3):
        if quit_all:
            break
        obs, info = env.reset(seed=ep)
        policy.reset()
        done = False
        frames = []
        print(f"\n=== Episode {ep} ===")
        print(f"  object_pos  = {obs['object_pos']}")
        print(f"  object_quat = {obs['object_quat']}")

        while not done:
            if check_q():
                print("  Q pressed — stopping.")
                quit_all = True
                break
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            frame = None
            for cam_key in ("rgb_exo_right_cam", "rgb_exo_left_cam",
                            "rgb_ego_cam", "rgb_gripper_cam"):
                if cam_key in obs and obs[cam_key] is not None:
                    frame = obs[cam_key]
                    break
            if frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            done = terminated or truncated

        if frames:
            os.makedirs("videos", exist_ok=True)
            out_path = f"videos/episode_{ep}.mp4"
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

        print(f"Episode {ep}  steps={info.get('step','?')}  "
              f"score={info.get('task_score', 0.0):.3f}  "
              f"phase={policy.phase}  "
              f"cube_y={obs['object_pos'][1]:.3f}  "
              f"target_y={PUSH_TARGET_Y:.3f}")

    env.close()