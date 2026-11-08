"""Teleoperation device registry.

Register new devices with ``register_teleop`` and instantiate them by name
with ``make_teleop``.  Built-in devices (keyboard, spacemouse, gamepad,
quest) are registered on import when their optional dependencies import
cleanly; otherwise the import error is kept and re-raised by
``make_teleop`` so that ``import imbench.teleop`` never fails on a headless
machine or without the ``teleop`` extra.

Usage::

    from imbench.teleop import make_teleop, list_teleop_devices

    teleop = make_teleop("keyboard")
    teleop.start()
    result = teleop.get_action(obs)  # {"action": np.ndarray, "paused": bool}
    action = result["action"]
    teleop.stop()
"""

import importlib

from imbench.teleop.base_teleop import TeleopInterface  # noqa: F401

_TELEOP_REGISTRY: dict[str, type] = {}
_UNAVAILABLE: dict[str, Exception] = {}


def register_teleop(name: str, cls: type):
    """Register a TeleopInterface subclass under *name*."""
    if not (isinstance(cls, type) and issubclass(cls, TeleopInterface)):
        raise TypeError(f"{cls} is not a TeleopInterface subclass")
    _TELEOP_REGISTRY[name] = cls
    _UNAVAILABLE.pop(name, None)


def get_teleop_class(name: str) -> type:
    """Return the class registered under *name*.

    Raises ``ImportError`` for a built-in device whose dependencies are
    missing or unusable here, ``ValueError`` for an unknown name.
    """
    if name in _TELEOP_REGISTRY:
        return _TELEOP_REGISTRY[name]
    if name in _UNAVAILABLE:
        err = _UNAVAILABLE[name]
        raise ImportError(f"Teleop device '{name}' is unavailable: {err}") from err
    raise ValueError(
        f"Unknown teleop device '{name}'. "
        f"Available: {list(_TELEOP_REGISTRY)}"
    )


def make_teleop(name: str, **kwargs) -> TeleopInterface:
    """Instantiate a registered teleop device by *name*."""
    return get_teleop_class(name)(**kwargs)


def list_teleop_devices() -> list[str]:
    """Return names of all registered (importable) teleop devices."""
    return list(_TELEOP_REGISTRY.keys())


# ── Auto-register built-in devices ──────────────────────────────────

_BUILTIN = {
    "keyboard": ("imbench.teleop.keyboard", "KeyboardTeleop"),
    "spacemouse": ("imbench.teleop.spacemouse", "SpaceMouseTeleop"),
    "gamepad": ("imbench.teleop.gamepad", "GamepadTeleop"),
    "quest": ("imbench.teleop.quest", "QuestTeleop"),
}

def _register_builtins() -> None:
    for name, (module, cls_name) in _BUILTIN.items():
        try:
            register_teleop(name, getattr(importlib.import_module(module), cls_name))
        except (ImportError, OSError, TypeError) as err:
            # pynput raises ImportError without an X display; evdev and
            # pyspacemouse raise ImportError when missing, easyhid OSError
            # when hidapi cannot be loaded, pyspacemouse TypeError on older
            # Pythons.
            _UNAVAILABLE[name] = err


_register_builtins()
