"""Tests for imbench.utils.tactile_setup.

These tests operate purely on ``xml.etree.ElementTree`` fixtures — no
MuJoCo sim is required — so they are fast and decoupled from robosuite
model compilation.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from imbench.utils import tactile_setup as ts
from imbench.utils.tactile_setup import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    DEFAULT_SPACING,
    FINGER_PAD_CONFIGS,
    find_finger_bodies,
    inject_tactile_for_robot,
    make_tactile_env_class,
)


# ── _make_tactile_sites ─────────────────────────────────────────────


class TestMakeTactileSites:

    def test_count_matches_rows_times_cols(self):
        sites = ts._make_tactile_sites(rows=3, cols=4, spacing=0.02)
        assert len(sites) == 12

    def test_grid_centered_on_origin(self):
        sites = ts._make_tactile_sites(rows=3, cols=3, spacing=0.01)
        xs = [x for _, _, x, _ in sites]
        ys = [y for _, _, _, y in sites]
        # Centered grid → sum of coords ~ 0.
        assert abs(sum(xs)) < 1e-9
        assert abs(sum(ys)) < 1e-9

    def test_even_dim_returns_correct_spacing(self):
        sites = ts._make_tactile_sites(rows=2, cols=2, spacing=0.01)
        # Expect offsets at ±0.005.
        xs = sorted({round(x, 6) for _, _, x, _ in sites})
        ys = sorted({round(y, 6) for _, _, _, y in sites})
        assert xs == [-0.005, 0.005]
        assert ys == [-0.005, 0.005]

    def test_row_col_indices_are_integers(self):
        sites = ts._make_tactile_sites(rows=2, cols=2, spacing=0.01)
        for r, c, _, _ in sites:
            assert isinstance(r, int)
            assert isinstance(c, int)


# ── find_finger_bodies ──────────────────────────────────────────────


def _make_finger_tree(robot_idx: int = 0, with_fingers: bool = True) -> ET.Element:
    """Build a minimal MJCF-like tree with optional finger bodies."""
    root = ET.Element("mujoco")
    worldbody = ET.SubElement(root, "worldbody")
    if with_fingers:
        ET.SubElement(
            worldbody, "body", attrib={"name": f"gripper{robot_idx}_rightfinger"}
        )
        ET.SubElement(
            worldbody, "body", attrib={"name": f"gripper{robot_idx}_leftfinger"}
        )
    # Noise body that should be ignored.
    ET.SubElement(worldbody, "body", attrib={"name": "unrelated_body"})
    return root


class TestFindFingerBodies:

    def test_finds_both_fingers(self):
        root = _make_finger_tree(robot_idx=0, with_fingers=True)
        out = find_finger_bodies(root, robot_idx=0)
        assert out["right"] == "gripper0_rightfinger"
        assert out["left"] == "gripper0_leftfinger"

    def test_returns_empty_when_no_fingers(self):
        root = _make_finger_tree(robot_idx=0, with_fingers=False)
        assert find_finger_bodies(root, robot_idx=0) == {}

    def test_returns_empty_for_wrong_robot_idx(self):
        root = _make_finger_tree(robot_idx=0, with_fingers=True)
        assert find_finger_bodies(root, robot_idx=1) == {}

    def test_handles_multi_robot(self):
        root = ET.Element("mujoco")
        wb = ET.SubElement(root, "worldbody")
        for idx in (0, 1):
            ET.SubElement(wb, "body", attrib={"name": f"gripper{idx}_rightfinger"})
            ET.SubElement(wb, "body", attrib={"name": f"gripper{idx}_leftfinger"})

        r0 = find_finger_bodies(root, 0)
        r1 = find_finger_bodies(root, 1)
        assert r0["right"] == "gripper0_rightfinger"
        assert r1["left"] == "gripper1_leftfinger"


# ── FINGER_PAD_CONFIGS sanity ───────────────────────────────────────


class TestFingerPadConfigs:

    def test_has_both_fingers(self):
        assert "left" in FINGER_PAD_CONFIGS
        assert "right" in FINGER_PAD_CONFIGS

    def test_each_has_pos_and_euler(self):
        for side in ("left", "right"):
            cfg = FINGER_PAD_CONFIGS[side]
            assert "pos" in cfg
            assert "euler" in cfg


# ── inject_tactile_for_robot ────────────────────────────────────────


class _FakeModel:
    """Stand-in for a robosuite MujocoModel that exposes .root and .sensor."""

    def __init__(self, root: ET.Element):
        self.root = root
        # Sensor element is typically separate in MJCF; tactile_setup only
        # appends to it, so any ET.Element works.
        self.sensor = ET.SubElement(root, "sensor")


def _build_model_with_fingers_and_pads(robot_idx: int = 0) -> _FakeModel:
    root = ET.Element("mujoco")
    wb = ET.SubElement(root, "worldbody")
    right = ET.SubElement(
        wb, "body", attrib={"name": f"gripper{robot_idx}_rightfinger"}
    )
    left = ET.SubElement(
        wb, "body", attrib={"name": f"gripper{robot_idx}_leftfinger"}
    )
    # Original collision pads that inject_tactile should disable.
    ET.SubElement(
        right, "geom",
        attrib={"name": f"gripper{robot_idx}_finger2_pad_collision",
                "contype": "1"},
    )
    ET.SubElement(
        left, "geom",
        attrib={"name": f"gripper{robot_idx}_finger1_pad_collision",
                "contype": "1"},
    )
    return _FakeModel(root)


class TestInjectTactileForRobot:

    def test_returns_false_when_no_fingers(self):
        model = _FakeModel(ET.Element("mujoco"))
        ET.SubElement(model.root, "worldbody")
        assert inject_tactile_for_robot(model, 0) is False

    def test_returns_true_when_fingers_present(self):
        model = _build_model_with_fingers_and_pads(0)
        assert inject_tactile_for_robot(model, 0, rows=2, cols=2) is True

    def test_adds_pad_bodies(self):
        model = _build_model_with_fingers_and_pads(0)
        inject_tactile_for_robot(model, 0, rows=2, cols=2)
        names = {b.get("name") for b in model.root.iter("body")}
        assert "robot0_tactile_pad_right" in names
        assert "robot0_tactile_pad_left" in names

    def test_adds_taxel_geoms(self):
        model = _build_model_with_fingers_and_pads(0)
        inject_tactile_for_robot(model, 0, rows=2, cols=3)
        taxels = [
            g for g in model.root.iter("geom")
            if (g.get("name") or "").startswith("robot0_taxel_")
        ]
        # 2*3 taxels × 2 fingers = 12 sphere geoms.
        assert len(taxels) == 12
        assert all(g.get("type") == "sphere" for g in taxels)

    def test_adds_backing_plates(self):
        model = _build_model_with_fingers_and_pads(0)
        inject_tactile_for_robot(model, 0, rows=2, cols=2)
        names = {g.get("name") for g in model.root.iter("geom")}
        assert "robot0_tactile_backing_right" in names
        assert "robot0_tactile_backing_left" in names

    def test_adds_touch_sites(self):
        model = _build_model_with_fingers_and_pads(0)
        inject_tactile_for_robot(model, 0, rows=2, cols=3)
        sites = [
            s for s in model.root.iter("site")
            if (s.get("name") or "").startswith("robot0_touch_")
        ]
        # 2*3 sites × 2 fingers = 12.
        assert len(sites) == 12

    def test_adds_touch_sensors(self):
        model = _build_model_with_fingers_and_pads(0)
        inject_tactile_for_robot(model, 0, rows=2, cols=3)
        sensors = list(model.sensor.iter("touch"))
        assert len(sensors) == 12
        for s in sensors:
            assert (s.get("name") or "").startswith("robot0_touch_sensor_")
            assert (s.get("site") or "").startswith("robot0_touch_")

    def test_disables_original_collision_pad(self):
        model = _build_model_with_fingers_and_pads(0)
        inject_tactile_for_robot(model, 0, rows=2, cols=2)
        for g in model.root.iter("geom"):
            name = g.get("name") or ""
            if name.endswith("_pad_collision"):
                assert g.get("contype") == "0", \
                    f"Original collision pad {name} should be disabled"

    def test_respects_robot_idx(self):
        model = _build_model_with_fingers_and_pads(1)
        assert inject_tactile_for_robot(model, 1, rows=2, cols=2) is True
        names = {b.get("name") for b in model.root.iter("body")}
        assert "robot1_tactile_pad_right" in names


# ── make_tactile_env_class ──────────────────────────────────────────


class _DummyBase:
    """Minimal stand-in for a robosuite env: enough surface for the
    generated TactileEnv subclass to declare overrides."""

    # Class-level flags so we can detect the calls without constructing.
    load_called = False
    observables_called = False

    def _load_model(self):
        _DummyBase.load_called = True

    def _setup_observables(self):
        _DummyBase.observables_called = True
        return {}


class TestMakeTactileEnvClass:

    def test_returns_subclass(self):
        Tactile = make_tactile_env_class(_DummyBase)
        assert issubclass(Tactile, _DummyBase)

    def test_renames_class(self):
        Tactile = make_tactile_env_class(_DummyBase)
        assert Tactile.__name__ == "Tactile_DummyBase"
        assert Tactile.__qualname__ == "Tactile_DummyBase"

    def test_generated_defaults(self):
        # Should accept the default rows/cols/spacing without raising.
        Tactile = make_tactile_env_class(_DummyBase)
        assert Tactile is not None
        # Defaults should match the module-level constants.
        assert DEFAULT_ROWS == 24
        assert DEFAULT_COLS == 16
        assert DEFAULT_SPACING == pytest.approx(0.01)
