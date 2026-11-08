"""Tests for imbench.grippers.suction_gripper.PandaSuctionGripper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imbench.grippers import suction_gripper as sg
from imbench.grippers.suction_gripper import PandaSuctionGripper


class TestPandaSuctionGripper:

    def test_construct(self):
        g = PandaSuctionGripper(idn=0)
        assert g is not None

    def test_dof_is_one(self):
        g = PandaSuctionGripper(idn=0)
        assert g.dof == 1

    def test_init_qpos_shape(self):
        g = PandaSuctionGripper(idn=0)
        assert g.init_qpos.shape == (1,)
        assert g.init_qpos[0] == 0.0

    def test_speed_positive(self):
        g = PandaSuctionGripper(idn=0)
        assert g.speed > 0

    def test_important_sites_keys(self):
        g = PandaSuctionGripper(idn=0)
        sites = g._important_sites
        for key in ("grip_site", "grip_cylinder", "ee", "ee_x", "ee_y", "ee_z"):
            assert key in sites, f"Missing site: {key}"

    def test_important_geoms_keys(self):
        g = PandaSuctionGripper(idn=0)
        geoms = g._important_geoms
        for key in ("left_finger", "right_finger", "left_fingerpad",
                    "right_fingerpad", "root"):
            assert key in geoms, f"Missing geom group: {key}"
        # Parallel-jaw keys are intentionally empty — this is a suction gripper.
        assert geoms["left_finger"] == []
        assert geoms["right_finger"] == []

    def test_format_action_passthrough(self):
        g = PandaSuctionGripper(idn=0)
        for val in (-1.0, -0.2, 0.0, 0.3, 1.0):
            out = g.format_action(np.array([val]))
            assert out[0] == val

    def test_format_action_preserves_array(self):
        g = PandaSuctionGripper(idn=0)
        a = np.array([0.5])
        out = g.format_action(a)
        # Should return the same array (identity per docstring), not a copy.
        assert np.array_equal(out, a)

    def test_missing_xml_raises(self, monkeypatch):
        """If the gripper XML is missing the constructor should fail clearly."""
        monkeypatch.setattr(sg, "_ASSETS_DIR", Path("/nonexistent/path/xyz"))
        with pytest.raises(FileNotFoundError):
            PandaSuctionGripper(idn=0)

    def test_contact_geoms_references_snout(self):
        g = PandaSuctionGripper(idn=0)
        # correct_naming prefixes the idn, but the suffix should still contain
        # "geo_snout".
        contact = g.contact_geoms
        assert any("geo_snout" in n for n in contact)


class TestSuctionGripperRegistration:
    """The module's import registers PandaSuctionGripper with robosuite."""

    def test_importable_by_factory(self):
        # If registration worked, robosuite's factory can build the gripper
        # by name.
        from robosuite.models.grippers import gripper_factory

        g = gripper_factory("PandaSuctionGripper", idn=0)
        assert type(g).__name__ == "PandaSuctionGripper"


def test_best_contact_skips_excluded_and_world_bodies():
    from types import SimpleNamespace

    from imbench.grippers.suction_weld import SuctionWeld

    # geom 0 -> world, geom 1 -> table (excluded), geom 2 -> payload, geom 3 -> snout
    model = SimpleNamespace(geom_bodyid=np.array([0, 4, 7, 9]))
    weld = SuctionWeld(excluded_body_ids={4})
    g1 = np.array([3, 1, 3, 0])
    g2 = np.array([1, 3, 2, 3])
    dist = np.array([-0.004, -0.003, -0.001, -0.005])
    assert weld._best_contact(model, 3, g1, g2, dist) == (2, 2)

    weld.excluded_body_ids = {4, 7}
    assert weld._best_contact(model, 3, g1, g2, dist) is None


class TestSuctionRelease:
    """Releasing a suction-held payload drops it instead of launching it."""

    def test_release_does_not_launch_payload(self):
        import imbench as ib

        task = ib.make("ib-p1-plate-shelf-hetbi-v1", obs_mode="proprio", seed=0)
        try:
            task.reset(seed=0)
            env = task._env
            sim = env.sim
            body = env._payload_body_id
            joint = env._payload.joints[0]
            top_off = float(env._payload.top_offset[2])
            eef_key = "robot0_eef_pos"
            obs = None

            def move(target, grip, gain=1.0):
                nonlocal obs
                a = np.zeros(14)
                a[0:3] = np.clip((target - obs[eef_key]) / 0.0015 * gain, -1, 1)
                a[6] = grip
                obs, *_ = task.step(a)

            obs, *_ = task.step(np.zeros(14))
            start = np.array(sim.data.body_xpos[body])
            above = start + [0, 0, top_off + 0.06]
            for _ in range(600):
                move(above, -1.0)
                if np.linalg.norm(obs[eef_key] - above) < 0.003:
                    break
            for _ in range(400):
                move(np.array(sim.data.body_xpos[body]) + [0, 0, top_off - 0.01], 1.0, gain=0.3)
                if task._suction_weld.welded_body_ids():
                    break
            assert task._suction_weld.welded_body_ids() == {body}
            lift = obs[eef_key] + [0, 0, 0.10]
            for _ in range(300):
                move(lift, 1.0)
            for _ in range(100):
                move(lift, 1.0)
            speeds = []
            for _ in range(4):
                move(lift, -1.0)
                v = sim.data.get_joint_qvel(joint)
                speeds.append((np.linalg.norm(v[:3]), np.linalg.norm(v[3:])))
            assert not task._suction_weld.welded_body_ids()
            # Free fall alone gives about 0.1 m/s within these steps.
            assert max(s[0] for s in speeds) < 0.3, speeds
            assert max(s[1] for s in speeds) < 0.5, speeds
        finally:
            task.close()
