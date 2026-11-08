"""Tactile sensor utilities — inject touch sensors into any robosuite env.

Provides:
- ``inject_tactile_for_robot``  — add tactile pads + touch sensors to a
  robot's finger gripper in the MuJoCo XML model.
- ``get_tactile_readings``      — read (rows, cols) touch data from sim.
- ``find_finger_bodies``        — discover finger body names for a robot.
- ``make_tactile_env_class``    — dynamically subclass any robosuite env
  class so that tactile injection happens during ``_load_model``.
"""

from __future__ import annotations

import numpy as np
import xml.etree.ElementTree as ET
from robosuite.utils.observables import Observable, sensor

DEFAULT_ROWS = 24
DEFAULT_COLS = 16
DEFAULT_SPACING = 0.01

# Pad placement relative to finger body (same geometry for left & right,
# mirrored on Y axis).
FINGER_PAD_CONFIGS = {
    "right": dict(pos="0 0.004 0.045", euler="1.5708 0 0"),
    "left":  dict(pos="0 -0.004 0.045", euler="-1.5708 0 0"),
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_tactile_sites(
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    spacing: float = DEFAULT_SPACING,
) -> list[tuple[int, int, float, float]]:
    offset_r = (rows - 1) / 2.0 * spacing
    offset_c = (cols - 1) / 2.0 * spacing
    sites = []
    for r in range(rows):
        for c in range(cols):
            x = r * spacing - offset_r
            y = c * spacing - offset_c
            sites.append((r, c, x, y))
    return sites


def find_finger_bodies(model_root: ET.Element, robot_idx: int) -> dict[str, str]:
    """Return ``{"right": body_name, "left": body_name}`` for *robot_idx*.

    Searches for bodies whose name starts with ``gripper{robot_idx}_`` and
    ends with ``_rightfinger`` / ``_leftfinger``.  Returns an empty dict if
    no finger bodies are found (e.g. suction or null gripper).
    """
    prefix = f"gripper{robot_idx}_"
    result = {}
    for body in model_root.iter("body"):
        name = body.get("name", "")
        if name.startswith(prefix):
            if name.endswith("_rightfinger"):
                result["right"] = name
            elif name.endswith("_leftfinger"):
                result["left"] = name
    return result


# ---------------------------------------------------------------------------
#  XML injection
# ---------------------------------------------------------------------------

def inject_tactile_for_robot(
    model: object,
    robot_idx: int,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    spacing: float = DEFAULT_SPACING,
) -> bool:
    """Inject tactile pad bodies and touch sensors for one robot.

    Must be called **before** the MuJoCo sim is compiled (i.e. inside
    ``_load_model``).

    Args:
        model:     robosuite ``MujocoModel`` (has ``.root`` and ``.sensor``).
        robot_idx: index of the robot (0, 1, ...).
        rows:      number of sensor rows (m).
        cols:      number of sensor columns (n).
        spacing:   distance between adjacent sensor sites.

    Returns:
        ``True`` if sensors were injected, ``False`` if no finger bodies
        were found for this robot.
    """
    root = model.root
    finger_bodies = find_finger_bodies(root, robot_idx)

    if not finger_bodies:
        return False

    sensor_prefix = f"robot{robot_idx}_"

    # Taxel sphere radius — under half-spacing so neighbours don't overlap,
    # but at least 1 mm for reliable contact detection (at 1 mm spacing the
    # floor wins and neighbours overlap).
    taxel_radius = max(spacing * 0.45, 0.001)

    # Site radius — slightly larger than the taxel sphere so the contact
    # point (which lies on the sphere surface) falls inside the site.
    site_radius = taxel_radius * 1.2

    # Map finger key → original collision pad suffix so we can disable it.
    _ORIG_PAD_SUFFIXES = {
        "right": "_finger2_pad_collision",
        "left":  "_finger1_pad_collision",
    }

    gripper_prefix = f"gripper{robot_idx}_"

    # Mass per taxel sphere (spread total pad mass across all taxels)
    total_pad_mass = 0.005
    n_taxels = rows * cols
    taxel_mass = total_pad_mass / max(n_taxels, 1)

    for finger in ("right", "left"):
        if finger not in finger_bodies:
            continue

        finger_body = root.find(f".//body[@name='{finger_bodies[finger]}']")
        if finger_body is None:
            continue

        pad_cfg = FINGER_PAD_CONFIGS[finger]

        # ── 0. Disable the original collision pad ────────────────────
        # Set contype=0 so it no longer participates in contacts;
        # the tactile taxel spheres take over as the contact surface.
        suffix = _ORIG_PAD_SUFFIXES[finger]
        for geom_el in root.iter("geom"):
            gname = geom_el.get("name", "")
            if gname.startswith(gripper_prefix) and gname.endswith(suffix):
                geom_el.set("contype", "0")
                break

        # ── 1. Tactile pad body ──────────────────────────────────────
        # A single body that holds all the taxel spheres and sites.
        pad = ET.SubElement(finger_body, "body")
        pad.set("name",  f"{sensor_prefix}tactile_pad_{finger}")
        pad.set("pos",   pad_cfg["pos"])
        pad.set("euler", pad_cfg["euler"])

        # ── 2. Visual backing plate ─────────────────────────────────
        # A visual-only box behind the taxel spheres that fills the gap
        # between the finger mesh surface and the tactile contact layer,
        # preventing objects from appearing to float when grasped.
        grid_half_x = (rows - 1) / 2.0 * spacing + taxel_radius
        grid_half_y = (cols - 1) / 2.0 * spacing + taxel_radius
        plate_half_z = taxel_radius * 2.5  # thick enough to bridge the finger-to-taxel gap
        backing = ET.SubElement(pad, "geom")
        backing.set("name",        f"{sensor_prefix}tactile_backing_{finger}")
        backing.set("type",        "box")
        backing.set("size",        f"{grid_half_x:.5f} {grid_half_y:.5f} {plate_half_z:.5f}")
        backing.set("pos",         "0 0 0")
        backing.set("contype",     "0")
        backing.set("conaffinity", "0")
        backing.set("group",       "1")
        backing.set("mass",        "0")
        backing.set("rgba",        "0.15 0.15 0.15 1")

        # ── 3. One sphere geom + co-located site per taxel ───────────
        # Each sphere acts as its own contact primitive.  MuJoCo generates
        # a contact point on the sphere surface when it touches an object,
        # and the co-located site captures that force as a touch reading.
        for r, c, x, y in _make_tactile_sites(rows, cols, spacing):
            # Sphere geom
            g = ET.SubElement(pad, "geom")
            g.set("name",        f"{sensor_prefix}taxel_{finger}_{r}_{c}")
            g.set("type",        "sphere")
            g.set("size",        f"{taxel_radius:.5f}")
            g.set("pos",         f"{x:.5f} {y:.5f} 0")
            g.set("contype",     "1")
            g.set("conaffinity", "1")
            g.set("condim",      "4")
            g.set("friction",    "1.2 0.005 0.0001")
            g.set("mass",        f"{taxel_mass:.6f}")
            g.set("rgba",        "0.1 0.9 0.3 0.4")

            # Co-located touch site
            site = ET.SubElement(pad, "site")
            site.set("name", f"{sensor_prefix}touch_{finger}_{r}_{c}")
            site.set("pos",  f"{x:.5f} {y:.5f} 0")
            site.set("size", f"{site_radius:.5f}")
            site.set("type", "sphere")
            site.set("rgba", "0 0 0 0")

        # ── 4. Touch sensors ─────────────────────────────────────────
        sensor_elem = model.sensor
        for r, c, _, _ in _make_tactile_sites(rows, cols, spacing):
            s = ET.SubElement(sensor_elem, "touch")
            s.set("name", f"{sensor_prefix}touch_sensor_{finger}_{r}_{c}")
            s.set("site", f"{sensor_prefix}touch_{finger}_{r}_{c}")

    return True


# ---------------------------------------------------------------------------
#  Runtime readout
# ---------------------------------------------------------------------------

def get_tactile_readings(
    sim: object,
    robot_idx: int,
    finger: str,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> np.ndarray:
    """Return a ``(rows, cols)`` array of touch-sensor readings."""
    prefix = f"robot{robot_idx}_"
    readings = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            name = f"{prefix}touch_sensor_{finger}_{r}_{c}"
            sid = sim.model.sensor_name2id(name)
            adr = sim.model.sensor_adr[sid]
            readings[r, c] = sim.data.sensordata[adr]
    return readings


# ---------------------------------------------------------------------------
#  Dynamic env-class factory
# ---------------------------------------------------------------------------

def make_tactile_env_class(
    base_cls: type,
    tactile_rows: int = DEFAULT_ROWS,
    tactile_cols: int = DEFAULT_COLS,
    tactile_spacing: float = DEFAULT_SPACING,
) -> type:
    """Return a subclass of *base_cls* that injects tactile sensors.

    The returned class overrides ``_load_model`` to inject tactile pads and
    sensors for **every** robot whose gripper has finger bodies.  It also
    registers robosuite ``Observable`` objects so that tactile data flows
    through the standard observation pipeline.

    Usage::

        TactileLift = make_tactile_env_class(Lift)
        env = TactileLift(robots="Panda", ...)
    """

    class TactileEnv(base_cls):

        def _load_model(self):
            super()._load_model()
            self._tactile_robot_indices = []
            for idx in range(len(self.robots)):
                injected = inject_tactile_for_robot(
                    self.model, idx, tactile_rows, tactile_cols,
                    tactile_spacing,
                )
                if injected:
                    self._tactile_robot_indices.append(idx)

        def _setup_observables(self):
            observables = super()._setup_observables()

            for robot_idx in self._tactile_robot_indices:
                for finger in ("right", "left"):
                    obs_name = f"robot{robot_idx}_tactile_{finger}"

                    def _make_sensor(ri=robot_idx, f=finger):
                        @sensor(modality="tactile")
                        def _tactile(obs_cache):
                            return get_tactile_readings(
                                self.sim, ri, f,
                                tactile_rows, tactile_cols,
                            )
                        return _tactile

                    observables[obs_name] = Observable(
                        name=obs_name,
                        sensor=_make_sensor(),
                        sampling_rate=self.control_freq,
                    )
            return observables

    TactileEnv.__name__ = f"Tactile{base_cls.__name__}"
    TactileEnv.__qualname__ = f"Tactile{base_cls.__name__}"
    return TactileEnv
