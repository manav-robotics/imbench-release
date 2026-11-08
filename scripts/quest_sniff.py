#!/usr/bin/env python3
"""Quest packet sniffer — diagnose left/right hand data from the Quest app.

Binds the same UDP port the teleop uses (9090) and prints, for every Nth
packet, the raw left/right wrist positions, grips, tracking flags, and the
per-hand movement since the previous packet. Use this to confirm whether the
Quest app is actually streaming valid, *changing* data for BOTH hands.

Run this with collect_demos.py NOT running (only one program can bind the
port; whichever starts second stops with "already in use"), point the Quest app
at this machine, then move each hand in turn and watch the deltas.

Usage:
    python scripts/quest_sniff.py            # port 9090
    python scripts/quest_sniff.py --port 9090 --every 10
"""

import argparse
import errno
import socket
import struct
import sys

import numpy as np

# Mirror the constants/layout the teleop parser uses so this reflects exactly
# what QuestTeleop sees.
_MAGIC = 0x51565254  # "QVRT"
_OFF_FLOATS = 4
_OFF_TELEOP = 134
_IDX = {"left": {"pos": 0, "rot": 3}, "right": {"pos": 14, "rot": 17}}
_IDX_GRIP = {"left": 28, "right": 29}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--every", type=int, default=15,
                    help="print 1 of every N packets (default 15)")
    ap.add_argument("--all", action="store_true",
                    help="dump all 30 floats and flag indices that changed "
                         "since the last printed packet (use to locate where "
                         "the new app puts left-hand data)")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", args.port))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            sys.exit(f"[sniff] UDP port {args.port} is already in use "
                     "(is collect_demos.py running?)")
        raise
    sock.settimeout(5.0)
    print(f"[sniff] listening on UDP :{args.port}. Move LEFT then RIGHT hand. Ctrl-C to stop.\n")

    prev = {"left": None, "right": None}
    prev_floats = None
    n = 0
    np.set_printoptions(precision=3, suppress=True)

    while True:
        try:
            data, addr = sock.recvfrom(512)
        except socket.timeout:
            print("[sniff] no packets in 5s — is the app streaming to this machine/port?")
            continue
        except KeyboardInterrupt:
            break

        if len(data) < 152:
            continue
        magic, = struct.unpack_from("<I", data, 0)
        if magic != _MAGIC:
            print(f"[sniff] bad magic 0x{magic:08x} from {addr[0]} (len={len(data)})")
            continue

        floats = struct.unpack_from("<30f", data, _OFF_FLOATS)
        teleop_active = data[_OFF_TELEOP] != 0
        left_tracked = data[132] != 0
        right_tracked = data[133] != 0

        n += 1
        if n % args.every:
            continue

        if args.all:
            cur = np.array(floats)
            changed = ""
            if prev_floats is not None:
                diff = np.abs(cur - prev_floats)
                idxs = [f"f{i}(Δ{diff[i]:.3f})" for i in range(30) if diff[i] > 0.01]
                changed = "  CHANGED: " + (", ".join(idxs) if idxs else "(none)")
            prev_floats = cur
            print(f"len={len(data)} active={int(teleop_active)} "
                  f"L_track={int(left_tracked)} R_track={int(right_tracked)}")
            print("  floats: " + " ".join(f"{i}:{v:.3f}" for i, v in enumerate(floats)))
            if changed:
                print(" " + changed)
            print()
            continue

        msg = [f"len={len(data)} active={int(teleop_active)} "
               f"L_track={int(left_tracked)} R_track={int(right_tracked)}"]
        for hand in ("left", "right"):
            pi = _IDX[hand]["pos"]
            pos = np.array(floats[pi:pi + 3])
            grip = floats[_IDX_GRIP[hand]]
            moved = 0.0 if prev[hand] is None else float(np.linalg.norm(pos - prev[hand]))
            prev[hand] = pos
            msg.append(f"{hand[0].upper()}: pos={pos} grip={grip:.2f} moved={moved:.4f}")
        print("  ".join(msg))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
