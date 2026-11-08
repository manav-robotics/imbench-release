"""Teleop device drivers against mocked hardware (no display, HID or headset)."""

import enum
import errno
import importlib
import os
import socket
import struct
import sys
import threading
import time
import types
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest


def _wait(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > end:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


@pytest.fixture
def fresh_import(monkeypatch):
    """Import a teleop module afresh against fake dependencies; undone after the test."""
    saved = []

    def _import(module, **fakes):
        for name, fake in fakes.items():
            monkeypatch.setitem(sys.modules, name, fake)
        pkg_name, _, attr = module.rpartition(".")
        pkg = sys.modules[pkg_name]
        saved.append((module, sys.modules.pop(module, None), pkg, attr, vars(pkg).pop(attr, None)))
        return importlib.import_module(module)

    yield _import
    for module, mod, pkg, attr, val in reversed(saved):
        sys.modules.pop(module, None)
        vars(pkg).pop(attr, None)
        if mod is not None:
            sys.modules[module] = mod
        if val is not None:
            setattr(pkg, attr, val)


# ── Gamepad ──────────────────────────────────────────────────────────


class _FakePad:
    """Pipe-backed evdev node; ``unplug()`` mimics the kernel removing it."""

    name = "Flydigi test pad"
    path = "/dev/input/event-test"

    def __init__(self, ecodes, absinfo):
        self._e = ecodes
        self._absinfo = absinfo
        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)
        self._events = []
        self._lock = threading.Lock()
        self.unplugged = False
        self.reads = 0

    def capabilities(self, verbose=False):
        e, info = self._e, self._absinfo
        stick = info(value=0, min=-32768, max=32767, fuzz=16, flat=128, resolution=0)
        trigger = info(value=0, min=0, max=255, fuzz=0, flat=0, resolution=0)
        hat = info(value=0, min=-1, max=1, fuzz=0, flat=0, resolution=0)
        return {
            e.EV_ABS: [(e.ABS_X, stick), (e.ABS_Y, stick), (e.ABS_RX, stick),
                       (e.ABS_RY, stick), (e.ABS_Z, trigger), (e.ABS_RZ, trigger),
                       (e.ABS_HAT0X, hat), (e.ABS_HAT0Y, hat)],
            e.EV_KEY: [e.BTN_SOUTH, e.BTN_EAST, e.BTN_NORTH, e.BTN_WEST, e.BTN_TL, e.BTN_TR],
        }

    def fileno(self):
        return self._r

    def read(self):
        self.reads += 1
        if self.unplugged:
            raise OSError(errno.ENODEV, "No such device")
        try:
            os.read(self._r, 4096)
        except BlockingIOError:
            pass
        with self._lock:
            events, self._events = self._events, []
        return iter(events)

    def close(self):
        pass

    def emit(self, etype, code, value):
        with self._lock:
            self._events.append(SimpleNamespace(type=etype, code=code, value=value))
        os.write(self._w, b"x")

    def unplug(self):
        self.unplugged = True
        os.close(self._w)  # EOF: select() now reports the node readable forever

    def release(self):
        os.close(self._r)
        if not self.unplugged:
            os.close(self._w)


@pytest.fixture
def gamepad_and_pad(monkeypatch):
    evdev = pytest.importorskip("evdev")
    from imbench.teleop import gamepad

    pad = _FakePad(evdev.ecodes, evdev.AbsInfo)
    monkeypatch.setattr(gamepad.evdev, "list_devices", lambda *a, **k: [pad.path])
    monkeypatch.setattr(gamepad.evdev, "InputDevice", lambda path: pad)
    teleop = gamepad.GamepadTeleop()
    teleop.start()
    yield teleop, pad
    teleop.stop()
    pad.release()


def test_gamepad_unplug_releases_input_and_stops_reader(gamepad_and_pad, capsys):
    teleop, pad = gamepad_and_pad
    e = pytest.importorskip("evdev").ecodes
    pad.emit(e.EV_ABS, e.ABS_Y, -32768)  # left stick forward
    pad.emit(e.EV_ABS, e.ABS_RZ, 255)    # RT: close the gripper
    _wait(lambda: teleop.get_action({})["action"][6] > -0.5)
    pad.emit(e.EV_ABS, e.ABS_RZ, 0)
    _wait(lambda: teleop._norm_trigger(e.ABS_RZ) == 0.0)
    before = teleop.get_action({})
    assert before["action"][0] > 0 and not before["paused"]

    reads = pad.reads
    pad.unplug()
    teleop._thread.join(timeout=1.0)
    after = teleop.get_action({})

    assert after["paused"]
    assert not after["action"][:6].any()
    assert after["action"][6] == before["action"][6]  # gripper keeps its state
    assert not teleop._thread.is_alive()
    assert pad.reads - reads <= 2
    assert not teleop.is_connected
    assert capsys.readouterr().out.count("no gamepad left") == 1


def test_gamepad_end_episode_latches_on_press(gamepad_and_pad):
    """A tap between two polls ends the episode; a button still held after reset() does not."""
    teleop, pad = gamepad_and_pad
    e = pytest.importorskip("evdev").ecodes
    end = teleop._btn_map["end_episode"]
    pad.emit(e.EV_KEY, end, 1)
    pad.emit(e.EV_KEY, end, 0)
    _wait(lambda: teleop.should_end_episode())
    pad.emit(e.EV_KEY, end, 1)
    _wait(lambda: teleop._btn(end))
    teleop.reset()
    teleop.get_action({})
    assert not teleop.should_end_episode()


# ── SpaceMouse ───────────────────────────────────────────────────────

_State = namedtuple("_State", "t x y z roll pitch yaw buttons")


def _from_hid(tx=0, ty=0, tz=0, rx=0, ry=0, rz=0, buttons=(0, 0)):
    """State as pyspacemouse 1.x decodes a raw 3Dconnexion report.

    Raw HID axes: X right, Y toward the user, Z down; rotations are
    right-handed about them.  pyspacemouse scales y, z, pitch (RX) and
    roll (RY) by -1, so x = right, y = away, z = up.
    """
    return _State(t=0.0, x=tx, y=-ty, z=-tz, roll=-ry, pitch=-rx, yaw=rz,
                  buttons=list(buttons))


class _HIDException(Exception):
    """Stand-in for easyhid.HIDException."""


@pytest.fixture
def spacemouse(fresh_import):
    fake = types.ModuleType("pyspacemouse")
    fake.state = _from_hid()
    fake.error = None

    def read():
        if fake.error is not None:
            raise fake.error
        return fake.state

    fake.read = read
    fake.open = lambda *a, **k: object()
    fake.close = lambda: None
    mod = fresh_import("imbench.teleop.spacemouse", pyspacemouse=fake)
    return mod, fake


@pytest.mark.parametrize("gesture, raw, expected", [
    ("push away", {"ty": -0.5}, [1, 0, 0, 0, 0, 0]),
    ("push right", {"tx": 0.5}, [0, -1, 0, 0, 0, 0]),
    ("lift", {"tz": -0.5}, [0, 0, 1, 0, 0, 0]),
    ("tilt right side down", {"ry": -0.5}, [0, 0, 0, 1, 0, 0]),
    ("tilt front edge down", {"rx": -0.5}, [0, 0, 0, 0, 1, 0]),
    ("twist counter-clockwise from above", {"rz": -0.5}, [0, 0, 0, 0, 0, 1]),
])
def test_spacemouse_axes_follow_robot_frame(spacemouse, gesture, raw, expected):
    """Same frame as keyboard / gamepad / Quest: X forward, Y left, Z up."""
    mod, fake = spacemouse
    teleop = mod.SpaceMouseTeleop(pos_scale=1.0, rot_scale=1.0)
    teleop.start()
    fake.state = _from_hid(**raw)
    np.testing.assert_allclose(teleop.get_action({})["action"][:6],
                               0.5 * np.array(expected), atol=1e-12, err_msg=gesture)


def test_spacemouse_reset_reopens_gripper(spacemouse):
    mod, fake = spacemouse
    teleop = mod.SpaceMouseTeleop()
    teleop.start()
    fake.state = _from_hid(buttons=(1, 0))
    for _ in range(25):
        teleop.get_action({})
    teleop.reset()
    fake.state = _from_hid(tx=0.5)
    assert teleop.get_action({})["action"][6] == -1.0


@pytest.mark.parametrize("opened", ["none", "raises"])
def test_spacemouse_start_fails_clearly_without_device(spacemouse, opened):
    mod, fake = spacemouse

    def open_(*a, **k):
        if opened == "raises":
            raise Exception("No found any connected or supported devices.")
        return None  # pyspacemouse prints and returns None on some paths

    fake.open = open_
    with pytest.raises(RuntimeError, match="SpaceMouse"):
        mod.SpaceMouseTeleop().start()


def test_spacemouse_read_error_pauses_instead_of_raising(spacemouse, capsys):
    mod, fake = spacemouse
    teleop = mod.SpaceMouseTeleop()
    teleop.start()
    fake.state = _from_hid(buttons=(1, 0))
    for _ in range(3):
        gripper = teleop.get_action({})["action"][6]

    fake.error = _HIDException("Failed to read from HID device")
    for _ in range(3):
        result = teleop.get_action({})
    assert result["paused"]
    assert not result["action"][:6].any()
    assert result["action"][6] == gripper
    assert capsys.readouterr().out.count("read failed") == 1

    fake.error = None
    fake.state = _from_hid(ty=-0.5)
    result = teleop.get_action({})
    assert not result["paused"] and result["action"][0] > 0


def test_spacemouse_rejects_more_than_one_arm(spacemouse):
    mod, _ = spacemouse
    with pytest.raises(ValueError, match="single arm"):
        mod.SpaceMouseTeleop(num_arms=2)


# ── Keyboard ─────────────────────────────────────────────────────────


def _fake_pynput():
    kb = types.ModuleType("pynput.keyboard")
    kb.Key = enum.Enum("Key", "delete f7 f8 f9 f10 enter tab esc")

    class Listener:
        def __init__(self, on_press=None, on_release=None):
            self.alive = False

        def start(self):
            self.alive = True

        def stop(self):
            self.alive = False

        def is_alive(self):
            return self.alive

    kb.Listener = Listener
    pkg = types.ModuleType("pynput")
    pkg.keyboard = kb
    return {"pynput": pkg, "pynput.keyboard": kb}


def test_keyboard_releases_held_keys_when_listener_dies(fresh_import, capsys):
    mod = fresh_import("imbench.teleop.keyboard", **_fake_pynput())
    teleop = mod.KeyboardTeleop()
    teleop.start()
    teleop._on_press(SimpleNamespace(char="8"))
    assert teleop.get_action({})["action"][0] > 0

    teleop._listener.alive = False  # e.g. the X connection dropped
    for _ in range(3):
        result = teleop.get_action({})
    assert result["paused"]
    assert not result["action"][:6].any()
    assert capsys.readouterr().out.count("listener stopped") == 1
    teleop.stop()


# ── Quest ────────────────────────────────────────────────────────────

_NAN = float("nan")


def _packet(right=(0.2, 1.0, 0.3), left=(-0.2, 1.0, 0.3), *, active=1,
            tracked=(1, 1), grip=(0.0, 0.0), right_quat=(0.0, 0.0, 0.0, 1.0)):
    """239-byte QVRT packet; positions in Quest space (x right, y up, z forward)."""
    f = [0.0] * 30
    f[0:3], f[3:7] = left, (0.0, 0.0, 0.0, 1.0)
    f[14:17], f[17:21] = right, right_quat
    f[28], f[29] = grip
    data = (struct.pack("<I", 0x51565254) + struct.pack("<30f", *f)
            + struct.pack("<d", 0.0) + bytes([tracked[0], tracked[1], active, 0]))
    return data + bytes(239 - len(data))


class _Sender:
    def __init__(self, teleop, host="127.0.0.1"):
        self.teleop = teleop
        self.port = teleop._sock.getsockname()[1]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, 0))

    def send(self, packet, wait=True):
        frame = self.teleop._frame
        self.sock.sendto(packet, ("127.0.0.1", self.port))
        if wait:
            _wait(lambda: self.teleop._frame is not frame)
        else:
            time.sleep(0.1)


@pytest.fixture
def quest(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)  # no Escape listener
    from imbench.teleop.quest import QuestTeleop

    made = []

    def _make(**kwargs):
        teleop = QuestTeleop(udp_port=0, **kwargs)
        teleop.start()
        made.append(teleop)
        return teleop

    yield _make
    for teleop in made:
        teleop.stop()


def test_quest_pauses_when_stream_stops_and_resumes_without_jump(quest):
    teleop = quest()
    sender = _Sender(teleop)
    sender.send(_packet(right=(0.2, 1.0, 0.300)))
    teleop.get_action({})
    sender.send(_packet(right=(0.2, 1.0, 0.301)))  # 1 mm forward
    assert teleop.get_action({})["action"][0] == pytest.approx(0.75, abs=1e-3)

    time.sleep(0.5)  # app closed / Wi-Fi drop
    result = teleop.get_action({})
    assert result["paused"] and not result["action"].any()

    sender.send(_packet(right=(0.2, 1.0, 0.401)))  # hand moved 10 cm meanwhile
    result = teleop.get_action({})
    assert not result["paused"] and not result["action"][:6].any()
    sender.send(_packet(right=(0.2, 1.0, 0.402)))
    assert teleop.get_action({})["action"][0] == pytest.approx(0.75, abs=1e-3)


@pytest.mark.parametrize("bad", [
    {"right": (_NAN, 1.0, 0.301)},
    {"right": (0.2, float("inf"), 0.301)},
    {"right_quat": (0.0, _NAN, 0.0, 1.0)},
    {"grip": (0.0, _NAN)},
])
def test_quest_non_finite_packet_never_reaches_the_action(quest, bad):
    teleop = quest()
    sender = _Sender(teleop)
    sender.send(_packet(right=(0.2, 1.0, 0.300)))
    teleop.get_action({})
    sender.send(_packet(**bad))
    result = teleop.get_action({})
    assert np.isfinite(result["action"]).all() and result["paused"]

    sender.send(_packet(right=(0.2, 1.0, 0.302)))
    result = teleop.get_action({})
    assert np.isfinite(result["action"]).all() and not result["action"][:6].any()
    sender.send(_packet(right=(0.2, 1.0, 0.303)))
    assert teleop.get_action({})["action"][0] == pytest.approx(0.75, abs=1e-3)


def test_quest_reengaging_starts_from_current_pose(quest):
    teleop = quest()
    sender = _Sender(teleop)
    sender.send(_packet(right=(0.2, 1.0, 0.3)))
    teleop.get_action({})
    sender.send(_packet(right=(0.2, 1.0, 0.3), active=0))
    assert teleop.get_action({})["paused"]
    sender.send(_packet(right=(0.25, 1.1, 0.4), active=0))  # hand repositioned
    teleop.get_action({})
    sender.send(_packet(right=(0.25, 1.1, 0.4)))
    result = teleop.get_action({})
    assert not result["paused"] and not result["action"][:6].any()


def test_quest_untracked_hand_gives_no_motion(quest):
    teleop = quest()
    sender = _Sender(teleop)
    sender.send(_packet(right=(0.2, 1.0, 0.3)))
    teleop.get_action({})
    sender.send(_packet(right=(0.0, 0.0, 0.0), tracked=(1, 0)))  # reported at origin
    result = teleop.get_action({})
    assert result["paused"] and not result["action"].any()
    sender.send(_packet(right=(0.2, 1.0, 0.4)))
    result = teleop.get_action({})
    assert not result["paused"] and not result["action"][:6].any()


def test_quest_bimanual_untracked_hand_holds_its_gripper(quest):
    teleop = quest(num_arms=2)
    sender = _Sender(teleop)
    sender.send(_packet(grip=(1.0, 0.0)))  # left hand closed
    teleop.get_action({})
    sender.send(_packet(right=(0.2, 1.0, 0.301), left=(0.0, 0.0, 0.0),
                        tracked=(0, 1), grip=(0.0, 0.0)))
    right, left = np.split(teleop.get_action({})["action"], 2)
    assert right[0] == pytest.approx(0.75, abs=1e-3)
    assert not left[:6].any() and left[6] == 1.0


def test_quest_follows_the_first_streaming_host(quest):
    teleop = quest()
    headset = _Sender(teleop)
    try:
        other = _Sender(teleop, host="127.0.0.2")
    except OSError:
        pytest.skip("no 127.0.0.2 loopback address")
    headset.send(_packet(right=(0.2, 1.0, 0.300)))
    teleop.get_action({})
    frame = teleop._frame
    other.send(_packet(right=(0.2, 1.0, 0.9)), wait=False)
    assert teleop._frame is frame
    headset.send(_packet(right=(0.2, 1.0, 0.301)))
    assert teleop.get_action({})["action"][0] == pytest.approx(0.75, abs=1e-3)

    time.sleep(0.5)  # headset silent: another host may take over
    other.send(_packet(right=(0.2, 1.0, 0.9)))
    result = teleop.get_action({})
    assert not result["paused"] and not result["action"][:6].any()


def test_quest_port_in_use_fails_loudly(quest):
    from imbench.teleop.quest import QuestTeleop

    port = quest()._sock.getsockname()[1]
    second = QuestTeleop(udp_port=port)
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            second.start()
    finally:
        if second._sock is not None:
            second.stop()
