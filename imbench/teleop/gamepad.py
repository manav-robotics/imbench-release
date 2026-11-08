from __future__ import annotations

import threading
import select as _sel
from pathlib import Path

import numpy as np
import yaml

try:
    import evdev
    from evdev import ecodes
except ImportError as _err:
    raise ImportError(
        f"GamepadTeleop requires evdev (Linux only): pip install -e \".[teleop]\" ({_err})"
    ) from _err

from imbench.teleop.base_teleop import TeleopInterface

_DEFAULT_CFG = Path(__file__).resolve().parent.parent.parent / "configs" / "teleop" / "gamepad.yaml"

# Actions that map to translation / rotation DOFs
_MOTION_ACTIONS = {"x", "y", "z", "roll", "pitch", "yaw"}
# Trigger-based gripper actions
_GRIPPER_AXIS_ACTIONS = {"gripper_open", "gripper_close"}
# Button actions
_BUTTON_ACTIONS = {"roll_neg", "roll_pos", "snap_toggle", "end_episode",
                   "switch_arm", "step_no_action"}


def _ecode(name: str) -> int:
    """Resolve an evdev ecode name (e.g. 'ABS_X', 'BTN_SOUTH') to its int."""
    code = getattr(ecodes, name, None)
    if code is None:
        raise ValueError(f"Unknown evdev ecode: {name}")
    return code


def _load_config(path: Path) -> dict:
    if path == _DEFAULT_CFG and not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: imbench must be installed in editable mode from a git "
            "checkout (pip install -e .), or pass config_path"
        )
    with open(path) as f:
        return yaml.safe_load(f)


class GamepadTeleop(TeleopInterface):
    """Config-driven gamepad teleoperation via evdev.

    All settings (axis/button mappings, speed, deadzones) are loaded from
    a YAML config file.  By default ``configs/teleop/gamepad.yaml`` is used;
    pass *config_path* to override.
    """

    def __init__(self, num_arms: int = 1, config_path: str | None = None) -> None:
        cfg = _load_config(Path(config_path) if config_path else _DEFAULT_CFG)

        # Tuning constants
        self.DEADZONE = cfg.get("deadzone", 0.25)
        self.TRIGGER_DEADZONE = cfg.get("trigger_deadzone", 0.10)
        self.SPEED = cfg.get("speed", 0.05)
        self.ROT_SPEED = cfg.get("rot_speed", self.SPEED)
        self.GRIPPER_SPEED = cfg.get("gripper_speed", 0.1)

        # Parse axis mappings
        self._axis_map: list[dict] = []
        for entry in cfg.get("axes", []):
            self._axis_map.append({
                "code": _ecode(entry["axis"]),
                "action": entry["action"],
                "invert": entry.get("invert", False),
                "type": entry.get("type", "stick"),
            })

        # Parse button mappings
        self._btn_map: dict[str, int] = {}  # action → ecode
        for entry in cfg.get("buttons", []):
            self._btn_map[entry["action"]] = _ecode(entry["button"])

        self.num_arms = num_arms
        self._active_arm = 0
        self._gripper_values = [-1.0] * num_arms

        # One-shot latch flags
        self._snap_latch = False
        self._arm_switch_latch = False
        self._end_episode = False

        # Shared state written by reader thread, read by get_action
        self._axes: dict[int, int] = {}
        self._buttons: dict[int, int] = {}
        self._axis_info: dict = {}
        self._lock = threading.Lock()

        self._devices: list = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()


    # ── lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        self._devices = self._find_devices()
        if not self._devices:
            raise RuntimeError(
                "No gamepad found. Make sure the controller is plugged in "
                "and you have permission to read /dev/input/* "
                "(add yourself to the 'input' group, see docs/info/teleop.md)."
            )

        # Gather axis calibration from all nodes
        for dev in self._devices:
            caps = dev.capabilities(verbose=False)
            if ecodes.EV_ABS in caps:
                for code, absinfo in caps[ecodes.EV_ABS]:
                    self._axis_info[code] = absinfo

        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

        arm_str = f"  ({self.num_arms} arm{'s' * (self.num_arms > 1)})"
        print(f"[GamepadTeleop] started{arm_str}")
        self._print_controls()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        for dev in self._devices:
            self._close(dev)
        self._devices.clear()

    def reset(self) -> None:
        self._gripper_values = [-1.0] * self.num_arms
        self._snap_latch = False
        self._arm_switch_latch = False
        self._end_episode = False
        self._active_arm = 0

    def should_end_episode(self) -> bool:
        return self._end_episode

    # ── device discovery ──────────────────────────────────────────

    @staticmethod
    def _find_devices():
        """Return evdev InputDevice handles for all Flydigi nodes (or any gamepad)."""
        devices = []

        # Pass 1: grab every node whose name contains "flydigi" or "direwolf"
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                if "flydigi" in dev.name.lower() or "direwolf" in dev.name.lower():
                    devices.append(dev)
                    print(f"[GamepadTeleop] found: {dev.name}  ({dev.path})")
                else:
                    dev.close()
            except (PermissionError, OSError):
                pass

        if devices:
            return devices

        # Pass 2: fall back to any device that has gamepad-range buttons
        _GAMEPAD_BTNS = {
            ecodes.BTN_SOUTH, ecodes.BTN_EAST, ecodes.BTN_NORTH,
            ecodes.BTN_WEST, ecodes.BTN_TL, ecodes.BTN_TR,
        }
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
                caps = dev.capabilities(verbose=False)
                key_caps = set(caps.get(ecodes.EV_KEY, []))
                if key_caps & _GAMEPAD_BTNS:
                    devices.append(dev)
                    print(f"[GamepadTeleop] found gamepad: {dev.name}  ({dev.path})")
                else:
                    dev.close()
            except (PermissionError, OSError):
                pass

        return devices

    # ── background reader ─────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """False before ``start()`` and once every gamepad node is lost."""
        return bool(self._devices)

    def _release_inputs(self) -> None:
        """Forget all axis and button states (motion stops, gripper holds)."""
        with self._lock:
            self._axes.clear()
            self._buttons.clear()

    def _reader_loop(self) -> None:
        while not self._stop.is_set() and self._devices:
            try:
                r, _, _ = _sel.select(self._devices, [], [], 0.05)
            except (ValueError, OSError):
                break
            for dev in r:
                try:
                    for ev in dev.read():
                        if ev.type == ecodes.EV_ABS:
                            with self._lock:
                                self._axes[ev.code] = ev.value
                        elif ev.type == ecodes.EV_KEY:
                            with self._lock:
                                self._buttons[ev.code] = ev.value
                            if ev.value == 1 and ev.code == self._btn_map.get("end_episode"):
                                # Latched on the press edge, so a tap between two
                                # polls is seen and a held button ends one episode.
                                self._end_episode = True
                except (BlockingIOError, InterruptedError):
                    pass
                except OSError as exc:
                    # Unplugged: the node stays readable and every read
                    # fails (ENODEV), so drop it instead of spinning.
                    print(f"[GamepadTeleop] WARNING: lost {dev.name} ({exc})")
                    self._devices.remove(dev)
                    self._release_inputs()
                    self._close(dev)
        if not self._stop.is_set():
            for dev in self._devices:
                self._close(dev)
            self._devices.clear()
            self._release_inputs()
            print("[GamepadTeleop] WARNING: no gamepad left; input released, "
                  "the robot stays paused. Reconnect and restart collection.")

    @staticmethod
    def _close(dev) -> None:
        try:
            dev.close()
        except Exception:
            pass

    # ── axis helpers ──────────────────────────────────────────────

    def _norm_stick(self, code: int) -> float:
        """Normalise a centered axis (stick / d-pad) to [-1, 1]."""
        with self._lock:
            raw = self._axes.get(code)
        if raw is None:
            return 0.0
        info = self._axis_info.get(code)
        if info is None:
            return 0.0
        mid = (info.max + info.min) / 2.0
        half = (info.max - info.min) / 2.0
        if half == 0:
            return 0.0
        return float(np.clip((raw - mid) / half, -1.0, 1.0))

    def _norm_trigger(self, code: int) -> float:
        """Normalise a one-sided axis (trigger) to [0, 1]."""
        with self._lock:
            raw = self._axes.get(code)
        if raw is None:
            return 0.0
        info = self._axis_info.get(code)
        if info is None:
            return 0.0
        span = info.max - info.min
        if span == 0:
            return 0.0
        return float(np.clip((raw - info.min) / span, 0.0, 1.0))

    def _btn(self, code: int) -> bool:
        with self._lock:
            return bool(self._buttons.get(code, 0))

    @staticmethod
    def _thresh(val: float, dz: float) -> float:
        """Binary threshold: beyond deadzone → ±1, else 0."""
        if val > dz:
            return 1.0
        if val < -dz:
            return -1.0
        return 0.0

    # ── action generation ─────────────────────────────────────────

    def get_action(self, obs: dict[str, np.ndarray]) -> dict:
        t = self._thresh
        dz = self.DEADZONE
        pos_spd = self.SPEED
        rot_spd = self.ROT_SPEED
        arm = self._active_arm

        has_button_input = False

        # ---- arm switch (bimanual only, one-shot) ----
        # Processed first so that axis / button handling below already
        # operates on the *new* active arm (gripper triggers, snap-toggle,
        # and the final action assembly all depend on ``arm``).
        if self.num_arms > 1:
            switch_code = self._btn_map.get("switch_arm")
            if switch_code is not None:
                if self._btn(switch_code):
                    has_button_input = True
                    if not self._arm_switch_latch:
                        self._arm_switch_latch = True
                        self._active_arm = (self._active_arm + 1) % self.num_arms
                        arm = self._active_arm
                        print(f"[GamepadTeleop] active arm → {self._active_arm}")
                else:
                    self._arm_switch_latch = False

        # ---- process axis mappings ----
        motion = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        has_trigger_input = False

        for mapping in self._axis_map:
            code = mapping["code"]
            action = mapping["action"]
            invert = mapping["invert"]

            if mapping["type"] == "trigger":
                val = self._norm_trigger(code)
                if val > self.TRIGGER_DEADZONE:
                    has_trigger_input = True
                    if action == "gripper_open":
                        self._gripper_values[arm] -= self.GRIPPER_SPEED
                    elif action == "gripper_close":
                        self._gripper_values[arm] += self.GRIPPER_SPEED
            else:  # stick
                val = self._norm_stick(code)
                if invert:
                    val = -val
                if action in _MOTION_ACTIONS:
                    s = pos_spd if action in ("x", "y", "z") else rot_spd
                    motion[action] = t(val, dz) * s

        # ---- process button mappings ----
        roll_neg_code = self._btn_map.get("roll_neg")
        roll_pos_code = self._btn_map.get("roll_pos")
        if roll_neg_code is not None and self._btn(roll_neg_code):
            motion["roll"] = -rot_spd
            has_button_input = True
        if roll_pos_code is not None and self._btn(roll_pos_code):
            motion["roll"] = rot_spd
            has_button_input = True

        # Snap-toggle gripper (one-shot latch)
        snap_code = self._btn_map.get("snap_toggle")
        if snap_code is not None:
            if self._btn(snap_code):
                has_button_input = True
                if not self._snap_latch:
                    self._snap_latch = True
                    g = self._gripper_values[arm]
                    self._gripper_values[arm] = -1.0 if g > 0.0 else 1.0
            else:
                self._snap_latch = False

        self._gripper_values[arm] = float(
            np.clip(self._gripper_values[arm], -1.0, 1.0)
        )

        # Step-with-no-action (hold to advance the sim one tick per tick
        # with zero motion and the gripper held at its current value).
        step_noop_code = self._btn_map.get("step_no_action")
        step_no_action = (step_noop_code is not None
                          and self._btn(step_noop_code))

        # ---- detect paused state ----
        has_motion = any(v != 0.0 for v in motion.values())
        paused = (not has_motion and not has_trigger_input
                  and not has_button_input and not step_no_action)

        # ---- assemble action ----
        # When step_no_action is held, force motion to zero and leave the
        # gripper unchanged from the previous tick. This takes priority
        # over any stick drift that might have nudged motion non-zero.
        if step_no_action:
            motion = {"x": 0.0, "y": 0.0, "z": 0.0,
                      "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

        single = np.array(
            [motion["x"], motion["y"], motion["z"],
             motion["roll"], motion["pitch"], motion["yaw"],
             self._gripper_values[arm]],
            dtype=np.float64,
        )
        single = np.clip(single, -1.0, 1.0)

        if self.num_arms == 1:
            return {"action": single, "paused": paused,
                    "step_no_action": step_no_action}

        # Bimanual: N×7 flat array, active arm gets motion, others hold gripper
        action = np.zeros(self.num_arms * 7, dtype=np.float64)
        action[arm * 7:(arm + 1) * 7] = single
        for a in range(self.num_arms):
            if a != arm:
                action[a * 7 + 6] = self._gripper_values[a]
        return {"action": np.clip(action, -1.0, 1.0), "paused": paused,
                "step_no_action": step_no_action}

    # ── pretty-print ──────────────────────────────────────────────

    def _print_controls(self):
        # Build readable names from config
        axis_lines = []
        for m in self._axis_map:
            name = m["code"]
            # reverse-lookup ecode name
            for attr in dir(ecodes):
                if getattr(ecodes, attr) == name and attr.startswith("ABS_"):
                    name = attr
                    break
            inv = " (inv)" if m["invert"] else ""
            axis_lines.append(f"  │  {name:<18} → {m['action']}{inv:<20}│")

        btn_lines = []
        for action, code in self._btn_map.items():
            name = code
            for attr in dir(ecodes):
                if getattr(ecodes, attr) == code and attr.startswith("BTN_"):
                    name = attr
                    break
            btn_lines.append(f"  │  {name:<18} → {action:<20}│")

        print(
            "\n"
            "  ┌─────────────────────────────────────────────────┐\n"
            "  │           GAMEPAD CONTROLS (from config)        │\n"
            "  ├─────────────────────────────────────────────────┤\n"
            "  │  Axes:                                          │"
        )
        for line in axis_lines:
            print(line)
        print(
            "  │  Buttons:                                       │"
        )
        for line in btn_lines:
            print(line)
        print(
            f"  │  speed={self.SPEED}  gripper_speed={self.GRIPPER_SPEED}       │\n"
            "  └─────────────────────────────────────────────────┘\n"
        )
