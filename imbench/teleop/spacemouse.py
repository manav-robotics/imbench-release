from __future__ import annotations

import numpy as np
try:
    import pyspacemouse
except ImportError as _err:
    raise ImportError(
        f"SpaceMouseTeleop requires pyspacemouse 1.x: pip install -e \".[teleop]\" ({_err})"
    ) from _err

from imbench.teleop.base_teleop import TeleopInterface


class SpaceMouseTeleop(TeleopInterface):
    """SpaceMouse teleoperation of one arm with continuous gripper control.

    Axes match the keyboard, gamepad and Quest (operator behind the robot):
    push the cap away / left / up for +X / +Y / +Z; tilt it right side down
    for +roll, front edge down for +pitch; twist it counter-clockwise (seen
    from above) for +yaw.  pyspacemouse reports x = right, y = away, z = up,
    and a clockwise-positive yaw.

    Gripper is controlled via buttons:
        Button 0 (left)  — close (ramp toward  1)
        Button 1 (right) — open  (ramp toward -1)
        Both / neither   — hold current value

    There is no discard input.  A failed read (e.g. the device was
    unplugged) pauses input with a warning instead of raising.
    """

    GRIPPER_SPEED = 0.1  # per-tick ramp rate for continuous gripper control

    def __init__(self, pos_scale: float = 0.5, rot_scale: float = 0.3,
                 num_arms: int = 1, **kwargs) -> None:
        if num_arms != 1:
            raise ValueError(
                f"SpaceMouseTeleop drives a single arm; this task has {num_arms}. "
                "Use the keyboard, gamepad or quest device."
            )
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self._gripper_value = -1.0  # continuous in [-1, 1]; -1 = fully open
        self._read_failed = False

    def start(self) -> None:
        try:
            device = pyspacemouse.open()
        except Exception as exc:  # pyspacemouse raises bare Exception
            raise RuntimeError(f"Could not open a SpaceMouse: {exc}") from exc
        if not device:
            raise RuntimeError(
                "Could not open a SpaceMouse: none found (is it plugged in "
                "and is /dev/hidraw* readable?)"
            )

    def stop(self) -> None:
        pyspacemouse.close()

    def reset(self) -> None:
        self._gripper_value = -1.0

    def get_action(self, obs: dict[str, np.ndarray]) -> dict:
        try:
            state = pyspacemouse.read()
            error = "device not open" if state is None else None
        except Exception as exc:  # easyhid.HIDException once unplugged
            state, error = None, exc
        if state is None:
            if not self._read_failed:
                print(f"[SpaceMouseTeleop] WARNING: read failed ({error}); "
                      "input paused. Replug and restart if it does not recover.")
                self._read_failed = True
            idle = np.zeros(self.ACTION_DIM)
            idle[6] = self._gripper_value
            return {"action": idle, "paused": True}
        self._read_failed = False

        # Continuous gripper: button 0 = close, button 1 = open
        buttons = list(state.buttons)
        btn_close = buttons[0] if len(buttons) > 0 else 0
        btn_open = buttons[1] if len(buttons) > 1 else 0
        has_button = bool(btn_close or btn_open)
        if btn_close and not btn_open:
            self._gripper_value += self.GRIPPER_SPEED
        elif btn_open and not btn_close:
            self._gripper_value -= self.GRIPPER_SPEED
        self._gripper_value = float(np.clip(self._gripper_value, -1.0, 1.0))

        axes = [state.x, state.y, state.z, state.roll, state.pitch, state.yaw]
        has_motion = any(abs(v) > 0.01 for v in axes)
        paused = not has_motion and not has_button

        action = np.array([
            state.y * self.pos_scale,      # away  -> +X
            -state.x * self.pos_scale,     # left  -> +Y
            state.z * self.pos_scale,      # up    -> +Z
            state.roll * self.rot_scale,
            state.pitch * self.rot_scale,
            -state.yaw * self.rot_scale,   # counter-clockwise -> +yaw
            self._gripper_value,
        ])
        return {"action": np.clip(action, -1, 1), "paused": paused}
