"""Quest VR hand tracking teleoperation — pure Python, no ROS required.

Listens on a UDP socket for binary packets (format below) sent by a
companion Meta Quest app and converts hand poses directly into IMBench
cartesian-delta actions.

Packet protocol  (QVRT, ≥152 bytes)
------------------------------------------
The Quest app may emit several wire-compatible variants — a 152-byte
hand-only packet, a 210-byte packet with controller poses, and a
239-byte packet that appends button / thumbstick / trigger
state in trailing bytes 210-238.  All share the layout below; only the
first 135 bytes are read here, so any packet ≥152 bytes is accepted and
trailing extension bytes are ignored.

Byte 0-3    : uint32 LE magic  0x51565254 ("QVRT")
Byte 4+     : 30 × float32 (little-endian)
  idx 0-2   : left  wrist position  (x, y, z) in Quest space
  idx 3-6   : left  wrist rotation  (qx, qy, qz, qw) in Quest space
  idx 7-9   : left  elbow position  (unused)
  idx 10-13 : left  elbow rotation  (unused)
  idx 14-16 : right wrist position  (x, y, z) in Quest space
  idx 17-20 : right wrist rotation  (qx, qy, qz, qw) in Quest space
  idx 21-23 : right elbow position  (unused)
  idx 24-27 : right elbow rotation  (unused)
  idx 28    : left  grip  [0, 1]
  idx 29    : right grip  [0, 1]
Byte 124-131: float64 timestamp
Byte 132    : left  is_tracked  (an untracked hand gives zero motion)
Byte 133    : right is_tracked
Byte 134    : teleop_active  (must be 1 for actions to be non-zero)
Byte 135    : calibration_request (unused here)
Byte 136-147: head position 3 × float32 (unused here)

Coordinate transform (Quest left-handed → Robot right-handed)
--------------------------------------------------------------
  position:   robot = [quest_z, -quest_x, quest_y]
  rotation:   R_robot = R_basis @ R_quest @ R_basis.T
  where R_basis = [[0, 0, 1], [-1, 0, 0], [0, 1, 0]]

Action format (7-D cartesian_delta)
-------------------------------------
  [dx, dy, dz, droll, dpitch, dyaw, gripper]
  gripper: +1 = close, -1 = open

Usage
-----
    from imbench.teleop import make_teleop
    teleop = make_teleop("quest")   # or make_teleop("quest", hand="left")
    teleop.start()
    result = teleop.get_action(obs)
    action = result["action"]
    teleop.stop()
"""

from __future__ import annotations

import errno
import socket
import struct
import threading
import time

import numpy as np

from imbench.teleop.base_teleop import TeleopInterface

# --------------------------------------------------------------------------- #
#  Protocol constants                                                          #
# --------------------------------------------------------------------------- #

_MAGIC          = 0x51565254        # "QVRT"
# Smallest valid packet (hand-only). The app also sends 210-byte (with
# controllers) and 239-byte (with full button/stick/trigger) variants; all
# share the first 135 bytes we read, so we accept anything at least this big
# and ignore trailing extension bytes. Rejecting on an exact size silently
# drops the current 239-byte packets and the robot never moves.
_PACKET_MIN     = 152

# Byte offsets into the packet
_OFF_FLOATS     = 4                 # floats start after 4-byte magic
_OFF_TRACKED    = {"left": 132, "right": 133}   # per-hand is_tracked flags
_OFF_TELEOP     = 134               # teleop_active flag

# Float indices (relative to _OFF_FLOATS) for each hand
_IDX = {
    "left":  {"pos": 0,  "rot": 3},
    "right": {"pos": 14, "rot": 17},
}
_IDX_GRIP = {"left": 28, "right": 29}

# --------------------------------------------------------------------------- #
#  Quaternion delta helpers                                                    #
# --------------------------------------------------------------------------- #

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[3] < 0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    s = np.sqrt(max(0.0, 1.0 - q[3] ** 2))
    if s < 1e-6:
        return np.zeros(3)
    return q[:3] / s * angle


# --------------------------------------------------------------------------- #
#  Coordinate transform: Quest (left-handed) → Robot (right-handed)           #
# --------------------------------------------------------------------------- #
#  Quest: X right, Y up, Z forward (left-handed)                              #
#  Robot: X forward, Y left, Z up  (right-handed)                             #
#  Mapping:  Quest Z → +Robot X                                               #
#            Quest X → -Robot Y                                               #
#            Quest Y → +Robot Z                                               #
# --------------------------------------------------------------------------- #

# Basis change matrix (columns implement the axis mapping above)
_R_BASIS = np.array([
    [0,  0,  1],
    [-1, 0,  0],
    [0,  1,  0],
], dtype=np.float64)
_R_BASIS_T = _R_BASIS.T


def _transform_pos(p: np.ndarray) -> np.ndarray:
    """Quest xyz → Robot xyz."""
    return np.array([p[2], -p[0], p[1]], dtype=np.float64)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(qx, qy, qz, qw) → 3×3 rotation matrix."""
    x, y, z, w = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → (qx, qy, qz, qw)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _transform_rot(q: np.ndarray) -> np.ndarray:
    """Quest quaternion (qx,qy,qz,qw) → Robot quaternion."""
    R_quest = _quat_to_matrix(q)
    R_robot = _R_BASIS @ R_quest @ _R_BASIS_T
    return _matrix_to_quat(R_robot)


# --------------------------------------------------------------------------- #
#  Packet parser                                                               #
# --------------------------------------------------------------------------- #

def _parse_packet(data: bytes) -> dict | None:
    """Return parsed hand data or None if the packet is invalid."""
    if len(data) < _PACKET_MIN:
        return None

    magic, = struct.unpack_from("<I", data, 0)
    if magic != _MAGIC:
        return None

    # All 30 floats starting at byte 4
    floats = struct.unpack_from("<30f", data, _OFF_FLOATS)

    teleop_active = data[_OFF_TELEOP] != 0

    result = {}
    for hand in ("left", "right"):
        pi = _IDX[hand]["pos"]
        ri = _IDX[hand]["rot"]
        # Quest quaternion in packet: (qx, qy, qz, qw)
        quest_pos  = np.array(floats[pi : pi + 3], dtype=np.float64)
        quest_quat = np.array(floats[ri : ri + 4], dtype=np.float64)  # qx,qy,qz,qw
        grip_raw   = float(floats[_IDX_GRIP[hand]])

        # Only a tracked hand with a finite, non-degenerate pose is used.
        tracked = bool(
            data[_OFF_TRACKED[hand]] != 0
            and np.isfinite(quest_pos).all() and np.isfinite(quest_quat).all()
            and np.isfinite(grip_raw) and np.linalg.norm(quest_quat) > 1e-6
        )
        result[hand] = {
            "tracked":    tracked,
            "pos":        _transform_pos(quest_pos) if tracked else None,
            "quat":       _transform_rot(quest_quat) if tracked else None,
            "grip":       np.clip(grip_raw, 0.0, 1.0) if tracked else None,
        }

    result["teleop_active"] = teleop_active
    return result


# --------------------------------------------------------------------------- #
#  QuestTeleop                                                                 #
# --------------------------------------------------------------------------- #

class QuestTeleop(TeleopInterface):
    """Drive an IMBench environment with a Meta Quest — pure Python UDP.

    No ROS 2 or workspace setup required.  Point the Quest Unity app at this
    machine's IP and port and call ``start()``.

    Parameters
    ----------
    hand : ``"right"`` | ``"left"``
        Which hand drives the robot in single-arm mode (default ``"right"``).
        Ignored when ``bimanual=True``.
    bimanual : bool
        When True, both hands are tracked and a 14-D action is returned:
        ``[right_arm(7), left_arm(7)]``.  Use for HetBi / two-arm tasks
        (default ``False``).
    udp_port : int
        UDP port to listen on (must match the Unity app setting, default 9090).
    pos_scale : float
        Scale applied to the position delta before clipping to [-1, 1].
        Increase if the robot feels sluggish (default 750.0).
    rot_scale : float
        Scale applied to the rotation-vector delta (default 35.0).
    zero_when_inactive : bool
        If True, return a zero action when the Quest ``teleop_active`` flag is
        False (operator has not pressed the activate button in Unity).
        If False, ``teleop_active`` is ignored (default True).
    stream_timeout : float
        Seconds without a packet after which input is paused (default 0.25).

    Deltas restart from the current pose whenever input resumes (after a
    pause, ``teleop_active`` toggle or tracking loss), so the robot never
    jumps by the hand motion made meanwhile.  The receiver follows the
    first host that streams and ignores other hosts until that stream has
    been silent for ``stream_timeout``.

    Pressing Escape on the host keyboard discards the current episode; this
    needs ``pynput`` and a display, and is disabled with a warning otherwise.
    """

    def __init__(
        self,
        hand: str = "right",
        bimanual: bool = False,
        num_arms: int = 1,
        udp_port: int = 9090,
        pos_scale: float = 750.0,
        rot_scale: float = 35.0,
        zero_when_inactive: bool = True,
        stream_timeout: float = 0.25,
    ):
        # num_arms mirrors the collect_demos convention; takes precedence over bimanual.
        if num_arms > 1:
            bimanual = True
        if hand not in ("left", "right"):
            raise ValueError(f"hand must be 'left' or 'right', got '{hand}'")

        self._hand = hand
        self._bimanual = bimanual
        self._udp_port = udp_port
        self._pos_scale = pos_scale
        self._rot_scale = rot_scale
        self._zero_when_inactive = zero_when_inactive
        self._stream_timeout = stream_timeout

        self._lock = threading.Lock()
        self._frame: dict | None = None   # latest parsed packet
        self._frame_time = float("-inf")  # time.monotonic() of _frame
        self._peer: str | None = None     # host whose packets are used
        self._ignored: set[str] = set()   # other hosts already reported
        self._stream_lost = False

        # Per-hand previous pose for delta computation
        self._prev: dict = {
            "left":  {"pos": None, "quat": None},
            "right": {"pos": None, "quat": None},
        }
        # Per-hand last gripper command, held while the hand is untracked
        self._grip = {"left": -1.0, "right": -1.0}

        self._sock: socket.socket | None = None
        self._udp_thread: threading.Thread | None = None
        self._running = False

        self._end_episode = False
        self._kb_listener = None
        self._esc_key = None

    # ------------------------------------------------------------------ #
    #  TeleopInterface lifecycle                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Open UDP socket and begin receiving Quest packets."""
        # No SO_REUSEADDR: a second listener on the port must fail loudly
        # rather than silently take every packet.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1.0)
        try:
            self._sock.bind(("", self._udp_port))
        except OSError as exc:
            self._sock.close()
            self._sock = None
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"UDP port {self._udp_port} is already in use (another "
                    "collector or scripts/quest_sniff.py?)"
                ) from exc
            raise
        self._running = True

        self._udp_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._udp_thread.start()

        try:
            from pynput import keyboard as kb
        except ImportError as exc:
            print(f"[QuestTeleop] WARNING: keyboard unavailable ({exc}); "
                  "Escape cannot discard episodes.")
        else:
            self._esc_key = kb.Key.esc
            self._kb_listener = kb.Listener(on_press=self._on_key_press)
            self._kb_listener.start()

        mode = "bimanual [right+left]" if self._bimanual else f"hand={self._hand}"
        print(
            f"[QuestTeleop] Listening on UDP port {self._udp_port} "
            f"({mode}). "
            "Ensure the Quest app is streaming to this machine."
        )

    def stop(self) -> None:
        """Close the UDP socket and stop the receiver thread."""
        self._running = False
        if self._sock:
            self._sock.close()
            self._sock = None
        if self._udp_thread:
            self._udp_thread.join(timeout=2.0)
            self._udp_thread = None
        if self._kb_listener:
            self._kb_listener.stop()
            self._kb_listener = None
        print("[QuestTeleop] Stopped.")

    def _on_key_press(self, key: object) -> None:
        if key == self._esc_key:
            self._end_episode = True

    def should_end_episode(self) -> bool:
        return self._end_episode

    def reset(self) -> None:
        """Clear previous-pose memory, gripper commands and episode-end flag."""
        self._end_episode = False
        self._grip = {"left": -1.0, "right": -1.0}
        self._forget("left", "right")

    def _forget(self, *hands: str) -> None:
        """Drop the previous pose so the next delta starts from zero."""
        with self._lock:
            for h in hands:
                self._prev[h] = {"pos": None, "quat": None}

    # ------------------------------------------------------------------ #
    #  UDP receiver (daemon thread)                                        #
    # ------------------------------------------------------------------ #

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break   # socket closed by stop()

            frame = _parse_packet(data)
            if frame is None:
                continue
            host, now = addr[0], time.monotonic()
            with self._lock:
                prev_peer = self._peer
                new_stream = now - self._frame_time > self._stream_timeout
                accepted = new_stream or host == prev_peer
                if accepted:
                    if new_stream:
                        # Measure motion from the new stream's first pose.
                        for h in self._prev:
                            self._prev[h] = {"pos": None, "quat": None}
                    self._peer = host
                    self._frame = frame
                    self._frame_time = now
            if not accepted and host not in self._ignored:
                self._ignored.add(host)
                print(f"[QuestTeleop] WARNING: ignoring packets from {host} "
                      f"while {prev_peer} is streaming")
            elif accepted and host != prev_peer:
                print(f"[QuestTeleop] receiving from {host}")

    # ------------------------------------------------------------------ #
    #  Action computation (main thread)                                   #
    # ------------------------------------------------------------------ #

    def get_action(self, obs: dict[str, np.ndarray]) -> dict:  # noqa: ARG002
        """Return a dict with the current Quest pose action.

        Returns dict with:
            ``"action"`` : (7,) for single-arm or (14,) for bimanual
                ([right_arm(7), left_arm(7)]).
            ``"paused"`` : True when no packet arrived for ``stream_timeout``,
                teleop_active is False, or no driving hand is tracked.
        """
        with self._lock:
            frame = self._frame
            age = time.monotonic() - self._frame_time

        hands = ("right", "left") if self._bimanual else (self._hand,)
        zero = np.zeros(7 * len(hands))

        if frame is None:
            return {"action": zero, "paused": True}

        if age > self._stream_timeout:
            if not self._stream_lost:
                self._stream_lost = True
                print(f"[QuestTeleop] WARNING: no packets for {age:.2f} s; paused")
            return {"action": zero, "paused": True}
        if self._stream_lost:
            self._stream_lost = False
            print("[QuestTeleop] stream resumed")

        if ((self._zero_when_inactive and not frame["teleop_active"])
                or not any(frame[h]["tracked"] for h in hands)):
            self._forget("left", "right")
            return {"action": zero, "paused": True}

        action = np.concatenate([self._hand_delta(h, frame) for h in hands])
        return {"action": np.clip(action, -1.0, 1.0), "paused": False}

    def _hand_delta(self, hand: str, frame: dict) -> np.ndarray:
        """Compute the 7-D delta action for one hand."""
        h    = frame[hand]
        action = np.zeros(7)
        if not h["tracked"]:
            # Zero motion, gripper held; re-anchor once tracking returns.
            self._forget(hand)
            action[6] = self._grip[hand]
            return action

        pos  = h["pos"]
        quat = h["quat"]
        grip = h["grip"]

        with self._lock:
            prev_pos  = self._prev[hand]["pos"]
            prev_quat = self._prev[hand]["quat"]

        if prev_pos is not None and prev_quat is not None:
            d_pos = (pos - prev_pos) * self._pos_scale
            q_rel = _quat_mul(quat, _quat_conjugate(prev_quat))
            d_rot = _quat_to_rotvec(q_rel) * self._rot_scale
            action[:3] = d_pos
            action[3:6] = d_rot

        with self._lock:
            self._prev[hand]["pos"]  = pos.copy()
            self._prev[hand]["quat"] = quat.copy()

        # Quest grip [0, 1]  →  IMBench [-1 open, +1 close]
        self._grip[hand] = 2.0 * grip - 1.0
        action[6] = self._grip[hand]
        return action