from __future__ import annotations

import numpy as np
from pynput import keyboard as kb
from imbench.teleop.base_teleop import TeleopInterface


def _key_id(key: object) -> object:
    """Normalise a pynput key to a hashable identifier."""
    if hasattr(key, 'char') and key.char is not None:  # type: ignore[union-attr]
        return key.char  # type: ignore[union-attr]
    return key  # kb.Key enum (e.g. kb.Key.up)


class KeyboardTeleop(TeleopInterface):
    """Keyboard teleoperation laid out on the numeric keypad.

    Keys are matched as pynput reports them, which hides whether a key came
    from the keypad.  Digits match both the keypad (NumLock on) and the
    number row; with NumLock off the keypad digits arrive as arrow and
    navigation keys and do nothing.  -Z is Delete, which the keypad ``.``
    sends only with NumLock off; yaw and gripper are F7-F10 (the keypad
    ``/ * + -`` are not bound, though some external keypads send F7-F10).

    The listener is global: keys typed into any window drive the robot, and
    Escape anywhere discards the episode.  If the listener stops (e.g. the
    X connection is lost), held keys are released and input stays paused.

    Translation:
        8 / 2          ±X  (forward / backward)
        4 / 6          ±Y  (left / right)
        0 / Delete     ±Z  (up / down)

    Rotation:
        7 / 9          ±Roll
        1 / 3          ±Pitch
        F7 / F8        ±Yaw

    Gripper:
        F9             Open  (ramp toward -1 while held)
        F10            Close (ramp toward  1 while held)
        Enter          Snap toggle between fully open (-1) and fully closed (1)

    Arm switching (bimanual):
        Tab            Switch active arm (robot0 ↔ robot1)

    Episode control:
        Escape         Discard the current episode
    """

    KEYS = {
        # ── Translation ──────────────────────────────────────────────
        '8':              np.array([ 1,  0,  0,  0,  0,  0,  0]),  # +X  fwd
        '2':              np.array([-1,  0,  0,  0,  0,  0,  0]),  # -X  back
        '4':              np.array([ 0,  1,  0,  0,  0,  0,  0]),  # +Y  left
        '6':              np.array([ 0, -1,  0,  0,  0,  0,  0]),  # -Y  right
        '0':              np.array([ 0,  0,  1,  0,  0,  0,  0]),  # +Z  up
        kb.Key.delete:    np.array([ 0,  0, -1,  0,  0,  0,  0]),  # -Z  down  (numpad .)
        # ── Rotation ─────────────────────────────────────────────────
        '7':              np.array([ 0,  0,  0,  1,  0,  0,  0]),  # +Roll
        '9':              np.array([ 0,  0,  0, -1,  0,  0,  0]),  # -Roll
        '1':              np.array([ 0,  0,  0,  0,  1,  0,  0]),  # +Pitch
        '3':              np.array([ 0,  0,  0,  0, -1,  0,  0]),  # -Pitch
        kb.Key.f7:        np.array([ 0,  0,  0,  0,  0,  1,  0]),  # +Yaw  (numpad /)
        kb.Key.f8:        np.array([ 0,  0,  0,  0,  0, -1,  0]),  # -Yaw  (numpad *)
    }
    SPEED = 0.05
    GRIPPER_SPEED = 0.1  # per-tick ramp rate for continuous gripper control

    # One-shot keys handled in _on_press
    _KEY_GRIPPER_SNAP   = kb.Key.enter # numpad Enter  (also fires for main Enter)
    _KEY_ARM_SWITCH     = kb.Key.tab
    _KEY_END_EPISODE    = kb.Key.esc

    # Keys that must be tracked as held (in addition to KEYS)
    _HELD_GRIPPER_OPEN  = kb.Key.f9
    _HELD_GRIPPER_CLOSE = kb.Key.f10

    def __init__(self, num_arms: int = 1) -> None:
        self._num_arms = num_arms
        self._active_arm = 0            # index of the arm being controlled
        self._pressed = set()           # currently held movement keys
        self._gripper_values = [-1.0] * num_arms  # per-arm gripper state
        self._gripper_toggled = False   # latched on press, consumed in get_action
        self._arm_switch = False        # latched on press, consumed in get_action
        self._end_episode = False       # latched on press, consumed in reset
        self._listener = None
        self._listener_lost = False

    def _on_press(self, key: object) -> None:
        kid = _key_id(key)
        self._pressed.add(kid)
        if kid == self._KEY_GRIPPER_SNAP:
            self._gripper_toggled = True
        elif kid == self._KEY_ARM_SWITCH:
            self._arm_switch = True
        elif kid == self._KEY_END_EPISODE:
            self._end_episode = True

    def _on_release(self, key: object) -> None:
        self._pressed.discard(_key_id(key))

    def start(self) -> None:
        self._listener_lost = False
        self._listener = kb.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        self._print_help()
        if self._num_arms > 1:
            print("[keyboard] Controlling arm 0.  Press Tab to switch arms.")

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def get_action(self, obs: dict[str, np.ndarray]) -> dict:
        # A dead listener never reports key releases: drop held keys.
        if self._listener is not None and not self._listener.is_alive():
            self._pressed.clear()
            if not self._listener_lost:
                self._listener_lost = True
                print("[keyboard] WARNING: keyboard listener stopped (display "
                      "connection lost?); input paused. Press Ctrl-C to stop.")

        # Handle arm switching
        if self._arm_switch and self._num_arms > 1:
            self._active_arm = (self._active_arm + 1) % self._num_arms
            print(f"[keyboard] Switched to arm {self._active_arm}")
            self._arm_switch = False

        single_action = np.zeros(7)
        has_motion = False
        for key, delta in self.KEYS.items():
            if key in self._pressed:
                single_action += delta * self.SPEED
                has_motion = True

        # Continuous gripper: ramp with F9 (open) / F10 (close)
        gv = self._gripper_values[self._active_arm]
        has_gripper = False
        if self._HELD_GRIPPER_OPEN in self._pressed:
            gv -= self.GRIPPER_SPEED
            has_gripper = True
        if self._HELD_GRIPPER_CLOSE in self._pressed:
            gv += self.GRIPPER_SPEED
            has_gripper = True

        # Enter snaps to fully open or fully closed
        if self._gripper_toggled:
            gv = -1.0 if gv > 0.0 else 1.0
            self._gripper_toggled = False
            has_gripper = True

        gv = float(np.clip(gv, -1.0, 1.0))
        self._gripper_values[self._active_arm] = gv
        single_action[6] = gv

        paused = not has_motion and not has_gripper

        # Build full action vector for all arms
        if self._num_arms == 1:
            return {"action": np.clip(single_action, -1, 1), "paused": paused}

        full_action = np.zeros(7 * self._num_arms)
        offset = self._active_arm * 7
        full_action[offset:offset + 7] = single_action
        # Set gripper values for inactive arms (hold their last gripper state)
        for arm_idx in range(self._num_arms):
            if arm_idx != self._active_arm:
                full_action[arm_idx * 7 + 6] = self._gripper_values[arm_idx]
        return {"action": np.clip(full_action, -1, 1), "paused": paused}

    def should_end_episode(self) -> bool:
        return self._end_episode

    def reset(self) -> None:
        self._gripper_values = [-1.0] * self._num_arms
        self._active_arm = 0
        self._gripper_toggled = False
        self._arm_switch = False
        self._end_episode = False

    # ── Help ─────────────────────────────────────────────────────────

    @staticmethod
    def _print_help():
        print(
            "\n"
            "  ┌────────────── Numpad Teleop ──────────────┐\n"
            "  │  Translation          Rotation            │\n"
            "  │  8 / 2    ±X fwd/back   7 / 9    ±Roll    │\n"
            "  │  4 / 6    ±Y left/right 1 / 3    ±Pitch   │\n"
            "  │  0 / Del  ±Z up/down    F7 / F8  ±Yaw     │\n"
            "  │                                           │\n"
            "  │  Gripper                                  │\n"
            "  │  [F9] Open (hold)   [F10] Close (hold)    │\n"
            "  │  [Enter] Snap toggle open/closed          │\n"
            "  │                                           │\n"
            "  │  [Tab] Switch arm   [Esc] Discard episode │\n"
            "  └───────────────────────────────────────────┘\n"
        )