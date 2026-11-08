#!/usr/bin/env python3
"""Test script for Flydigi Direwolf (or any gamepad) via evdev.

Usage:
    # Add yourself to the input group first (one-time, needs relogin):
    #   sudo usermod -aG input $USER
    python scripts/test_gamepad.py

    # If you know the device path:
    python scripts/test_gamepad.py --device /dev/input/event15
"""

import argparse
import sys
import time

try:
    import evdev
    from evdev import ecodes, categorize
except ImportError:
    sys.exit("evdev not installed. Run: pip install -e \".[teleop]\"")


def _ecode_name(val):
    """ecodes lookups can return str, list, or tuple — always return a str."""
    if isinstance(val, (list, tuple)):
        return val[0]
    return val


def list_devices():
    """List all input devices and highlight likely gamepads."""
    paths = evdev.list_devices()
    if not paths:
        print("\n[!] No input devices found.")
        print("    Likely cause: permission denied on /dev/input/*")
        print("    Fix (pick one):")
        print("      1. sudo usermod -aG input $USER   (then log out & back in)")
        print("      2. Run this script with sudo")
        return []

    devices = []
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║                   Available Input Devices                   ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for path in paths:
        try:
            dev = evdev.InputDevice(path)
            caps = dev.capabilities(verbose=True)
            has_abs = ("EV_ABS", ecodes.EV_ABS) in caps or ecodes.EV_ABS in dev.capabilities()
            has_key = ("EV_KEY", ecodes.EV_KEY) in caps or ecodes.EV_KEY in dev.capabilities()
            is_gamepad = has_abs and has_key

            marker = " <-- GAMEPAD?" if is_gamepad else ""
            print(f"║  {path:25s} {dev.name:30s}{marker}")
            devices.append((path, dev.name, is_gamepad))
            dev.close()
        except PermissionError:
            print(f"║  {path:25s} [permission denied]")
        except Exception as e:
            print(f"║  {path:25s} [error: {e}]")
    print("╚══════════════════════════════════════════════════════════════╝")
    return devices


def pick_devices(devices):
    """Auto-pick all Flydigi nodes (axes + buttons may be on separate nodes)."""
    # Grab ALL nodes whose name contains flydigi/direwolf
    flydigi = [(p, n) for p, n, _ in devices
               if "flydigi" in n.lower() or "direwolf" in n.lower()]
    if flydigi:
        print(f"\n[*] Auto-selected {len(flydigi)} Flydigi device node(s):")
        for path, name in flydigi:
            print(f"    {name} ({path})")
        return [p for p, _ in flydigi]

    gamepads = [(p, n) for p, n, is_gp in devices if is_gp]
    if len(gamepads) == 1:
        path, name = gamepads[0]
        print(f"\n[*] Auto-selected only gamepad: {name} ({path})")
        return [path]

    if gamepads:
        print(f"\n[*] Found {len(gamepads)} potential gamepads:")
        for i, (path, name) in enumerate(gamepads):
            print(f"    [{i}] {name} ({path})")
        choice = input("Pick one or more (comma-separated numbers): ").strip()
        try:
            indices = [int(c.strip()) for c in choice.split(",")]
            return [gamepads[i][0] for i in indices]
        except (ValueError, IndexError):
            sys.exit("Invalid choice.")

    print("\n[!] No gamepad-like devices detected.")
    print("    Make sure your Flydigi Direwolf is:")
    print("      - Plugged in via USB, or")
    print("      - Paired via Bluetooth and connected")
    print("    Then re-run this script.")
    sys.exit(1)


def describe_capabilities(dev):
    """Print detailed axis and button capabilities."""
    caps = dev.capabilities(verbose=False)

    print("\n┌─── Axes (Analog Sticks / Triggers) ─────────────────────────┐")
    if ecodes.EV_ABS in caps:
        for code, absinfo in caps[ecodes.EV_ABS]:
            name = _ecode_name(ecodes.ABS.get(code, f"ABS_{code}"))
            print(f"│  {name:20s}  min={absinfo.min:6d}  max={absinfo.max:6d}  "
                  f"flat={absinfo.flat:4d}  fuzz={absinfo.fuzz:3d}")
    else:
        print("│  (none)")
    print("└─────────────────────────────────────────────────────────────┘")

    print("\n┌─── Buttons ─────────────────────────────────────────────────┐")
    if ecodes.EV_KEY in caps:
        keys = caps[ecodes.EV_KEY]
        for code in keys:
            name = _ecode_name(ecodes.KEY.get(code, ecodes.BTN.get(code, f"KEY_{code}")))
            print(f"│  code={code:4d}  {name}")
    else:
        print("│  (none)")
    print("└─────────────────────────────────────────────────────────────┘")


def monitor_live(devices):
    """Live-print every event from one or more devices using select()."""
    import select as _sel

    print("\n" + "=" * 62)
    print(f"  LIVE INPUT MONITOR — reading {len(devices)} device(s)")
    print("  Move sticks, press buttons, pull triggers...")
    print("=" * 62 + "\n")

    fd_to_dev = {dev.fd: dev for dev in devices}
    axes = {}
    buttons = {}

    try:
        while True:
            r, _, _ = _sel.select(devices, [], [])
            for dev in r:
                for event in dev.read():
                    if event.type == ecodes.EV_ABS:
                        name = _ecode_name(ecodes.ABS.get(event.code, f"ABS_{event.code}"))
                        axes[name] = event.value
                        _print_state(axes, buttons)

                    elif event.type == ecodes.EV_KEY:
                        raw = ecodes.KEY.get(event.code, ecodes.BTN.get(event.code, f"BTN_{event.code}"))
                        name = _ecode_name(raw)
                        state = "PRESSED" if event.value == 1 else "RELEASED" if event.value == 0 else "HELD"
                        buttons[name] = state
                        _print_state(axes, buttons)

    except KeyboardInterrupt:
        print("\n\n[*] Stopped.")


def _print_state(axes, buttons):
    """Clear screen and print compact state overview."""
    # Build output lines
    lines = ["\033[2J\033[H"]  # clear screen + cursor home
    lines.append("═══ GAMEPAD STATE ═══════════════════════════════════════")
    lines.append("")

    # Axes in two columns
    lines.append("  AXES:")
    axis_items = sorted(axes.items())
    for i in range(0, len(axis_items), 2):
        left = f"    {axis_items[i][0]:16s} = {axis_items[i][1]:6d}"
        if i + 1 < len(axis_items):
            right = f"    {axis_items[i+1][0]:16s} = {axis_items[i+1][1]:6d}"
        else:
            right = ""
        lines.append(left + right)

    lines.append("")
    lines.append("  BUTTONS:")
    active = {k: v for k, v in buttons.items() if v != "RELEASED"}
    if active:
        for name, state in sorted(active.items()):
            lines.append(f"    {name:24s} {state}")
    else:
        lines.append("    (none pressed)")

    lines.append("")
    lines.append("─────────────────────────────────────────────────────────")
    lines.append("  Press Ctrl+C to quit")

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Test gamepad/controller input via evdev")
    parser.add_argument("--device", "-d", type=str, nargs="+", default=None,
                        help="Device path(s), e.g. /dev/input/event17 /dev/input/event18")
    parser.add_argument("--list", "-l", action="store_true",
                        help="Just list devices and exit")
    args = parser.parse_args()

    # List devices
    devices = list_devices()

    if args.list:
        return

    if not devices and not args.device:
        return

    # Pick device(s)
    if args.device:
        dev_paths = args.device
    else:
        dev_paths = pick_devices(devices)

    # Open all devices
    opened = []
    for dev_path in dev_paths:
        try:
            dev = evdev.InputDevice(dev_path)
        except PermissionError:
            sys.exit(f"[!] Permission denied on {dev_path}. Run with sudo.")
        except FileNotFoundError:
            sys.exit(f"[!] Device {dev_path} not found.")

        print(f"\n[*] Opened: {dev.name}")
        print(f"    Path:   {dev.path}")
        print(f"    Phys:   {dev.phys}")
        print(f"    Uniq:   {dev.uniq}")
        describe_capabilities(dev)
        opened.append(dev)

    # Go live
    monitor_live(opened)

    for dev in opened:
        dev.close()


if __name__ == "__main__":
    main()
