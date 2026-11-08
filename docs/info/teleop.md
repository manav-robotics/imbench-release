# Teleoperation

The teleoperation system provides a pluggable interface for controlling the robot with different input devices.

## Built-in Devices

| Device | Name | Requirements |
|--------|------|--------------|
| Keyboard | `keyboard` | `pynput` (install via `pip install -e ".[teleop]"`) and an X display |
| 3D SpaceMouse | `spacemouse` | `pyspacemouse` 1.x (install via `pip install -e ".[teleop]"`) |
| Gamepad (Flydigi Direwolf / any standard pad) | `gamepad` | `evdev` (install via `pip install -e ".[teleop]"`, Linux only) + Linux `input` group or a udev rule |
| Meta Quest VR | `quest` | UDP streaming from the companion Quest app (not yet publicly released; port 9090); Escape-to-discard needs `pynput` and a display |

`import imbench.teleop` always succeeds, even on a headless machine or without the `teleop` extra. `list_teleop_devices()` lists only the devices whose dependencies import here. Creating any other built-in device raises an `ImportError` naming the cause: `make_teleop("keyboard")` without `pynput` or a display, `make_teleop("spacemouse")` without `pyspacemouse`, `make_teleop("gamepad")` without `evdev`. A device whose library is present but whose hardware is missing fails in `start()`.

## Usage

```python
from imbench.teleop import make_teleop, list_teleop_devices

print(list_teleop_devices())  # usable here, e.g. ['keyboard', 'spacemouse', 'gamepad', 'quest']

teleop = make_teleop("keyboard")
teleop.start()

result = teleop.get_action(obs)         # {"action": (7,) array, "paused": bool}
action = result["action"]
paused = result["paused"]               # True when operator is idle (no inputs)
discard = teleop.should_end_episode()   # True after Escape / gamepad B

teleop.reset()                          # call between episodes
teleop.stop()                           # cleanup
```

### The `paused` flag

All devices detect when the operator is idle and return `paused=True`. The collection script uses this to skip `env.step()` and recording during idle periods so the dataset only contains intentional actions.

| Device | Paused when |
|--------|------------|
| Keyboard | No movement keys or gripper keys held, or the key listener stopped |
| Gamepad | All sticks inside deadzone, no triggers, no buttons, or the pad disconnected |
| SpaceMouse | All 6 axes below threshold, no buttons, or a read failed |
| Quest | No UDP packet for `stream_timeout` (0.25 s), `teleop_active` flag is False, or the driving hand is not tracked |

`collect_demos.py` treats `should_end_episode()` as "discard": the episode in progress is thrown away and the scene restarts. Episodes are saved only when the task terminates.

## Keyboard Controls

The layout follows the numeric keypad. Keys are matched as `pynput` reports them, which does not tell the keypad from the main keyboard:

- Keypad digits work only with NumLock **on**; they also match the number row. With NumLock off they arrive as arrow and navigation keys, which are not bound.
- -Z is Delete. The keypad `.` sends Delete only with NumLock off, so with NumLock on use the main-keyboard Delete.
- Yaw and gripper are F7–F10. The keypad `/ * + -` are not bound (some external keypads send F7–F10 for them).

Each held motion key contributes 0.05 per tick.

The key listener is global: it sees keys typed into any window. Digits typed into the dashboard or a terminal move the robot and are recorded, and Escape pressed anywhere discards the episode in progress, so avoid typing in other windows while collecting. If the listener stops (for example the X connection drops), held keys are released and input stays paused with a warning; press Ctrl-C to stop.

| Key | Action |
|-----|--------|
| `8` / `2` | +X / -X (forward / backward) |
| `4` / `6` | +Y / -Y (left / right) |
| `0` / Delete | +Z / -Z (up / down) |
| `7` / `9` | +Roll / -Roll |
| `1` / `3` | +Pitch / -Pitch |
| F7 / F8 | +Yaw / -Yaw |
| F9 | Open gripper (ramp toward -1 while held) |
| F10 | Close gripper (ramp toward 1 while held) |
| Enter | Snap toggle gripper fully open / fully closed |
| Tab | Switch active arm (bimanual tasks) |
| Escape | Discard the current episode |

## SpaceMouse Controls

Axes use the same robot frame as the other devices, seen by an operator standing behind the robot (X forward, Y left, Z up). Motion is proportional to the cap deflection (`pos_scale` 0.5, `rot_scale` 0.3).

| Input | Action |
|-------|--------|
| Push the cap away / pull it toward you | +X / -X (forward / backward) |
| Push the cap left / right | +Y / -Y (left / right) |
| Lift the cap / press it down | +Z / -Z (up / down) |
| Tilt right side down / left side down | +Roll / -Roll |
| Tilt front edge down / up | +Pitch / -Pitch |
| Twist counter-clockwise / clockwise (seen from above) | +Yaw / -Yaw |
| Left button (hold) | Close gripper (ramp toward 1) |
| Right button (hold) | Open gripper (ramp toward -1) |

The SpaceMouse drives one arm (`num_arms > 1` raises `ValueError`) and has no discard input: an episode ends when the task terminates and is then saved; Ctrl-C stops collection and drops the episode in progress. The gripper reopens on `reset()`. `start()` raises `RuntimeError` when no SpaceMouse can be opened (on Linux it needs read access to `/dev/hidraw*`). If a read fails mid-episode, for example because the device was unplugged, input pauses with a warning instead of stopping collection.

## Gamepad Controls

The gamepad driver works with the Flydigi Direwolf and any standard Linux gamepad. It auto-detects Flydigi controllers by name, falling back to any device that exposes gamepad button codes.

All inputs use **binary direction at constant speed** (no proportional control) -- only the direction of a stick deflection matters, not how far it is pushed. This matches the keyboard teleop behaviour and gives precise, controlled motion.

### Setup

```bash
# Install the teleop extra (includes evdev)
pip install -e ".[teleop]"

# Grant input device access (pick one):
#   Option A -- add yourself to the input group (permanent, needs re-login):
sudo usermod -aG input $USER
#   Option B -- a udev rule for the pad only (vendor ID from `lsusb`), then replug it:
echo 'SUBSYSTEM=="input", ATTRS{idVendor}=="<vid>", TAG+="uaccess"' | \
    sudo tee /etc/udev/rules.d/70-gamepad.rules && sudo udevadm control --reload
```

Do not run the collector with `sudo`: as root it uses `/root/.imbench`, a separate machine ID, run counter and session, and issues run IDs that collide with your user's.

### Single-arm controls

| Input | Action |
|-------|--------|
| Left stick X / Y | Y / X translation (strafe / forward-back) |
| D-pad Up / Down | +Z / -Z translation (up / down) |
| Right stick X / Y | Yaw / Pitch rotation |
| LB / RB | -Roll / +Roll |
| LT / RT (triggers) | Gripper open / close (continuous ramp) |
| A | Snap-toggle gripper fully open / fully closed |
| B | Discard the current episode |
| Y (hold) | Step the sim with zero motion, gripper held (recorded) |

### Bimanual controls

In bimanual tasks (e.g. `ib-p1-*-hetbi-*`) the gamepad outputs a 14-D action (7 per arm). The number of arms is auto-detected from the task config.

| Input | Action |
|-------|--------|
| X | Switch active arm (0 / 1) |
| *All other inputs* | Control the **active** arm only |

The inactive arm receives zero motion and holds its current gripper value.

### Disconnects

If the pad disconnects (cable pulled, wireless drop), its inputs are released: the robot stops and the gripper keeps its value. A warning is printed and `teleop.is_connected` becomes `False`. Input stays paused; reconnect the pad and restart collection.

### Testing the controller

A standalone diagnostic script is included to verify your controller is detected and to inspect raw axis/button codes:

```bash
python scripts/test_gamepad.py          # auto-detect & live monitor
python scripts/test_gamepad.py --list   # just list devices
```

### Configuration

All gamepad settings (axis/button mappings, speed, deadzones) are controlled by a YAML config file at `configs/teleop/gamepad.yaml`. Edit this file to remap controls or tune behaviour without touching Python code.

To use a custom config from the CLI:

```bash
python scripts/collect_demos.py --task ib-p2-cube-toss-v1 --teleop gamepad --teleop-config my_config.yaml
```

Or programmatically:

```python
teleop = make_teleop("gamepad", config_path="path/to/my_config.yaml")
```

Buttons are configured by evdev code: `BTN_NORTH` / `BTN_WEST` are the kernel aliases of `BTN_X` / `BTN_Y`, which xpad-style drivers report for the buttons labelled X and Y.

#### Tuning constants (in config)

| Key | Shipped value | Description |
|-----|---------------|-------------|
| `deadzone` | 0.25 | Normalised stick threshold before input registers |
| `trigger_deadzone` | 0.10 | Trigger threshold (0-1 range) |
| `speed` | 0.3 | Translation (x, y, z) action magnitude per tick |
| `rot_speed` | 0.2 | Rotation (roll, pitch, yaw) action magnitude per tick |
| `gripper_speed` | 0.10 | Gripper ramp rate per tick |

## Action Filter

Before actions reach the simulator, a low-pass filter + delta clamp is applied to smooth the 6 motion channels (dx, dy, dz, droll, dpitch, dyaw). The gripper channel passes through unfiltered so snap-toggle grasping stays responsive.

The filter is configured under `action_filter` in the teleop YAML passed with `--teleop-config` (for any device), defaulting to the repository's `configs/teleop/gamepad.yaml` regardless of the working directory:

```yaml
action_filter:
  alpha: 0.8        # EMA smoothing factor [0-1]. Higher = smoother, more lag. 0 = off.
  max_delta: 0.40   # Max per-step change on any motion channel.
```

The two stages run in sequence:

1. **Low-pass (EMA)**: `smoothed = alpha * prev + (1 - alpha) * raw`
2. **Delta clamp**: `delta = clip(smoothed - prev, -max_delta, max_delta)`

This prevents sudden spikes and jerky transitions while keeping the control responsive.

## Quest VR

The Quest driver receives hand-tracking data over UDP from a companion Meta Quest Unity app. The app has not been publicly released yet, and the device cannot be used without it; the driver and packet format are included for reference.

```python
teleop = make_teleop("quest", hand="right", udp_port=9090)
teleop.start()  # starts UDP listener
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hand` | `"right"` | Which hand drives the robot (ignored in bimanual mode) |
| `bimanual` | `False` | Track both hands, return 14-D action |
| `udp_port` | `9090` | UDP port (must match the Quest app) |
| `pos_scale` | `750.0` | Position delta scale factor |
| `rot_scale` | `35.0` | Rotation delta scale factor |
| `zero_when_inactive` | `True` | Pause while the app's `teleop_active` flag is off; `False` ignores the flag |
| `stream_timeout` | `0.25` | Seconds without a packet after which input pauses |

The coordinate transform maps Quest's left-handed system (X right, Y up, Z forward) to the robot's right-handed frame (X forward, Y left, Z up).

Input pauses when packets stop for `stream_timeout` (app closed, headset asleep, Wi-Fi drop), when `teleop_active` is off, and when the driving hand is flagged untracked or its pose is not finite. In bimanual mode an untracked hand gets zero motion and keeps its gripper value. When input resumes, motion is measured from the current hand pose, so moving the hand while paused (clutching) does not move the robot.

Pressing Escape on the host keyboard discards the current episode. This needs `pynput` and a display; without them the Quest driver still runs and prints a warning that Escape is unavailable. Like the keyboard device, this listener is global: Escape pressed in any window discards the episode.

Only one program can listen on the port: a second collector or `scripts/quest_sniff.py` on the same port stops with "already in use".

**Network trust.** The driver listens on every network interface and does not authenticate packets: any host that can reach the UDP port can stream poses, and the resulting actions are recorded. It follows the first host that streams and ignores other hosts while that stream is live, switching only after `stream_timeout` of silence (for example when the headset's address changes). This prevents cross-talk from a second headset but is not a security boundary, since UDP source addresses can be spoofed. Collect on a trusted network, or allow the port only from the headset; with `ufw` enabled (default deny incoming): `sudo ufw allow from <headset-ip> to any port 9090 proto udp`.

## Adding a Custom Device

Subclass `TeleopInterface` and register it:

```python
from imbench.teleop import TeleopInterface, register_teleop

class MyDevice(TeleopInterface):
    def get_action(self, obs):
        # read your device, return dict with "action" and "paused"
        action = ...  # (7,) array
        paused = ...  # bool: True when no input
        return {"action": action, "paused": paused}
    def start(self):
        ...
    def stop(self):
        ...

register_teleop("my_device", MyDevice)
```

It is then available everywhere -- `make_teleop("my_device")`, `--teleop my_device` in scripts, etc.
