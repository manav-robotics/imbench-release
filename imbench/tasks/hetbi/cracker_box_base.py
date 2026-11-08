"""Shared base for the cracker-box slot-insertion tasks.

Not a task of its own: ``ib-p1-cracker-box-suction-v1`` and
``ib-p1-cracker-box-jaw-v1`` build on it. It holds the slot fixture's
colour and axes, the success thresholds read from the config, and the
scoring helpers (lift, transport toward the slot, aligned insertion).
"""

import numpy as np
import robosuite.utils.transform_utils as T

from imbench.tasks.hetbi.base_hetbi import HetBiBaseTask


class CrackerBoxBase(HetBiBaseTask):
    """Cracker box from the table into a slot fixture (shared scoring).

    Subclasses set ``TASK_ID`` and build their own staged reward in
    :meth:`_build_staged_reward` once their active arm is known.
    """

    _SLOT_RGBA = np.array([0.8, 0.1, 0.1, 1.0])  # red

    # Cracker-box body axes (YCB mesh frame): x is the 75 mm thickness,
    # z the 216 mm length. Slot fixture axes: x runs along the channel,
    # y is normal to its open face.
    _BOX_THIN_AXIS = 0
    _BOX_LONG_AXIS = 2
    _SLOT_LENGTH_AXIS = 0
    _SLOT_OPENING_AXIS = 1

    def __init__(self, task_id: str = None, **kwargs):
        super().__init__(task_id=task_id or self.TASK_ID, **kwargs)
        success_cfg = self.cfg.get("task_conditions", {}).get("success", {})
        self._center_dist_threshold = success_cfg.get("dist_threshold", 0.05)
        # Max angle between the box's long/thin axes and the slot's
        # length/opening axes for the box to count as lying flat in it.
        self._insert_max_tilt_deg = success_cfg.get("insert_max_tilt_deg", 20.0)
        self._lift_height = success_cfg.get("lift_height") or 0.05
        # Normalising distance for the coarse-transport shaping term — not a
        # pass/fail threshold, purely a shaping scale.
        self._transport_target_dist = success_cfg.get("transport_target_dist", 0.3)

        self._staged_reward = self._build_staged_reward()

    def _build_staged_reward(self):
        """Placeholder until a subclass knows its active arm; subclasses
        rebuild the real staged reward at the end of their ``__init__``."""
        return None

    def _randomize_visuals(self, sim):
        super()._randomize_visuals(sim)
        self._tint_fixture(self._SLOT_RGBA)

    def _tint_fixture(self, rgba):
        """Recolor every geom belonging to the slot fixture body."""
        fixture_body_id = getattr(self._env, "_fixture_body_id", None)
        if fixture_body_id is None:
            return
        sim = self._env.sim
        for i in range(sim.model.ngeom):
            if sim.model.geom_bodyid[i] == fixture_body_id:
                sim.model.geom_matid[i] = -1  # strip any material
                sim.model.geom_rgba[i] = rgba
        sim.forward()

    # ── stage helpers (shared by the suction-only and jaw-only tasks) ──

    def _lift_progress(self, raw_obs):
        return self._payload_lift_progress(raw_obs, self._lift_height)

    def _slot_centre(self):
        fixture_bounds = getattr(self._env, "_fixture_bounds", None)
        if fixture_bounds is None:
            return None
        return np.array([
            0.5 * (fixture_bounds["x"][0] + fixture_bounds["x"][1]),
            0.5 * (fixture_bounds["y"][0] + fixture_bounds["y"][1]),
            0.5 * (fixture_bounds["z"][0] + fixture_bounds["z"][1]),
        ])

    def _coarse_transport_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        centre = self._slot_centre()
        # Gated on a gripper having touched the box, so one spawned near the slot earns nothing.
        if pos_key not in raw_obs or centre is None or not self._payload_was_handled():
            return 0.0
        dist = float(np.linalg.norm(np.array(raw_obs[pos_key]) - centre))
        return float(np.clip(1.0 - dist / self._transport_target_dist, 0.0, 1.0))

    def _aligned_with_slot(self, raw_obs) -> bool:
        """Box lies flat in the slot: long axis along the channel and thin
        axis normal to the opening, each within ``insert_max_tilt_deg``."""
        quat_key = f"{self._payload_prefix or 'payload_'}quat"
        fixture_id = getattr(self._env, "_fixture_body_id", None)
        if quat_key not in raw_obs or fixture_id is None:
            return False
        box_rot = T.quat2mat(np.asarray(raw_obs[quat_key]))
        slot_rot = np.asarray(self._env.sim.data.body_xmat[fixture_id]).reshape(3, 3)
        rel = slot_rot.T @ box_rot  # columns: box axes in the slot frame
        cos_tol = np.cos(np.radians(self._insert_max_tilt_deg))
        return bool(
            abs(rel[self._SLOT_LENGTH_AXIS, self._BOX_LONG_AXIS]) >= cos_tol
            and abs(rel[self._SLOT_OPENING_AXIS, self._BOX_THIN_AXIS]) >= cos_tol
        )

    def _fine_insertion_progress(self, raw_obs):
        """Success: box centred in the slot, aligned with it, released and at rest."""
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        centre = self._slot_centre()
        if pos_key not in raw_obs or centre is None:
            return 0.0
        dist = float(np.linalg.norm(np.array(raw_obs[pos_key]) - centre))
        inserted = (
            dist <= self._center_dist_threshold
            and self._aligned_with_slot(raw_obs)
            and self._payload_released()
            and self._payload_is_stable_multistep()
        )
        return 1.0 if inserted else 0.0
