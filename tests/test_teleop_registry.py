"""Teleop registry must import and degrade cleanly without optional deps or a display."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Run in a fresh interpreter with pynput made unimportable, as on a headless
# machine or one without the ``teleop`` extra.
_NO_PYNPUT = "import sys; sys.modules['pynput'] = None\n"


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _NO_PYNPUT + code],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )


def test_registry_imports_without_pynput():
    proc = _run(
        "import imbench.teleop as t\n"
        "assert 'keyboard' not in t.list_teleop_devices()\n"
        "try:\n"
        "    t.make_teleop('keyboard')\n"
        "except ImportError as e:\n"
        "    assert \"'keyboard' is unavailable\" in str(e)\n"
        "else:\n"
        "    raise AssertionError('expected ImportError')\n"
        "q = t.make_teleop('quest', udp_port=0)\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def test_collect_demos_help_without_pynput():
    proc = _run(
        "import runpy\n"
        "sys.argv = ['collect_demos.py', '--help']\n"
        "runpy.run_path('scripts/collect_demos.py', run_name='__main__')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert "--teleop" in proc.stdout


def test_unknown_device_is_value_error():
    from imbench.teleop import make_teleop

    with pytest.raises(ValueError, match="Unknown teleop device"):
        make_teleop("no-such-device")


@pytest.mark.parametrize("device, dependency", [
    ("gamepad", "evdev"),
    ("spacemouse", "pyspacemouse"),
])
def test_device_without_dependency_is_not_listed(device, dependency):
    proc = _run(
        f"sys.modules[{dependency!r}] = None\n"
        "import imbench.teleop as t\n"
        f"assert {device!r} not in t.list_teleop_devices(), t.list_teleop_devices()\n"
        "try:\n"
        f"    t.make_teleop({device!r})\n"
        "except ImportError as e:\n"
        f"    assert {dependency!r} in str(e), e\n"
        "else:\n"
        "    raise AssertionError('expected ImportError')\n"
        "print('ok')\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def test_pyspacemouse_has_module_level_api():
    """SpaceMouseTeleop uses pyspacemouse 1.x's module-level open/read/close."""
    try:
        import pyspacemouse
    except (ImportError, OSError):
        pytest.skip("pyspacemouse (teleop extra) not installed")
    assert all(hasattr(pyspacemouse, name) for name in ("open", "read", "close"))
