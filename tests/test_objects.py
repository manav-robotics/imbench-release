"""Tests for the object factory (objects.py + SingleArmTableEnv._make_object).

Covers:
- imbench.objects (SeesawObject, PendulumObject, YCBObject, ExtrasObject)
- imbench.envs.single_arm_table_env.SingleArmTableEnv._make_object
"""

import pytest

from imbench.objects import SeesawObject, PendulumObject, YCBObject, ExtrasObject
from imbench.envs.single_arm_table_env import SingleArmTableEnv


# =====================================================================
#  Object factory
# =====================================================================

class TestObjectFactory:

    def test_seesaw_object_creation(self):
        obj = SeesawObject(name="test_seesaw")
        assert obj is not None
        assert obj.name == "test_seesaw"

    def test_pendulum_object_creation(self):
        obj = PendulumObject(name="test_pend", stand_height=0.30, arm_length=0.15)
        assert obj is not None

    def test_seesaw_custom_params(self):
        obj = SeesawObject(
            name="custom_seesaw",
            plank_half_length=0.30,
            plank_half_width=0.04,
            fulcrum_height=0.05,
            stopper_height=0.02,
            damping=0.02,
        )
        assert obj is not None

    def test_ycb_object_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            YCBObject(name="x", ycb_name="does_not_exist_12345")

    def test_extras_object_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            ExtrasObject(name="x", extras_name="does_not_exist_12345")

    def test_make_object_seesaw_type(self):
        """Verify the env factory can create seesaw objects."""
        spec = {
            "type": "seesaw", "name": "test_sw",
            "plank_half_length": 0.20,
            "fulcrum_height": 0.04,
        }
        obj = SingleArmTableEnv._make_object(spec)
        assert obj is not None

    def test_make_object_pendulum_type(self):
        spec = {
            "type": "pendulum", "name": "test_pend",
            "stand_height": 0.25,
            "arm_length": 0.20,
        }
        obj = SingleArmTableEnv._make_object(spec)
        assert obj is not None

    def test_make_object_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown object type"):
            SingleArmTableEnv._make_object({"type": "alien", "name": "x"})


# =====================================================================
#  YCBObject / ExtrasObject — successful construction
# =====================================================================

class TestYCBObjectConstruction:

    def test_valid_ycb_object(self):
        obj = YCBObject(name="bowl", ycb_name="bowl")
        assert obj is not None
        assert hasattr(obj, "root_body")

    def test_valid_ycb_object_name_stored(self):
        obj = YCBObject(name="test_bowl", ycb_name="bowl")
        assert obj.name == "test_bowl"


class TestExtrasObjectConstruction:

    def test_valid_extras_object(self):
        obj = ExtrasObject(name="rack", extras_name="rack", joints=None)
        assert obj is not None
        assert hasattr(obj, "root_body")


# =====================================================================
#  deps/extras MJCF angles
# =====================================================================

def test_extras_euler_angles_are_radians():
    """robosuite compiles merged MJCF with ``angle="radian"``, so an ``euler``
    written in degrees silently misrotates the geom (the Galileo ramp and the
    open cup were built this way)."""
    import math
    import re
    from pathlib import Path

    extras = Path(__file__).resolve().parent.parent / "deps" / "extras"
    bad = []
    for xml in sorted(extras.glob("*.xml")):
        for m in re.finditer(r'euler="([^"]+)"', xml.read_text()):
            if any(abs(float(v)) > 2 * math.pi + 1e-6 for v in m.group(1).split()):
                bad.append(f"{xml.name}: {m.group(0)}")
    assert not bad, bad

