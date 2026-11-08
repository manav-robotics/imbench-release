"""Bowl Table Transfer — heterogeneous bimanual task.

Task ID: ``ib-p1-bowl-table-v1``
"""

import numpy as np
import robosuite.utils.transform_utils as T

from imbench.tasks.hetbi.base_hetbi import HetBiBaseTask
from imbench.tasks.staged_reward import Stage, StagedReward


class BowlTableV1(HetBiBaseTask):
    """Bowl from rack to table using finger + suction grippers.

    Finger gripper (robot 0) grasps the bowl rim inside a rack compartment
    and extracts it, then hands it over to suction gripper (robot 1) which
    attaches to the flat bottom and places it on the table.

    Parameters
    ----------
    task_id : str, optional
        Defaults to ``"ib-p1-bowl-table-v1"``.
    **kwargs
        Forwarded to ``HetBiBaseTask``.
    """

    TASK_ID = "ib-p1-bowl-table-v1"
    DESCRIPTION = (
        "Bowl starts upside-down inside a rack compartment. Finger gripper "
        "(robot 0) grasps the bowl rim and extracts it from the rack, then "
        "hands it over to suction gripper (robot 1) which attaches to the "
        "flat bottom and places it on the table."
    )

    def __init__(self, task_id: str = None, **kwargs):
        super().__init__(task_id=task_id or self.TASK_ID, **kwargs)
        success_cfg = self.cfg.get("task_conditions", {}).get("success", {})
        # Normalising distance for the "transport" stage — how far (in xy)
        # the bowl must travel away from the rack before transport is
        # considered complete. Not a pass/fail threshold, purely a shaping
        # scale, so a sensible default suffices without a YAML knob.
        self._transport_target_dist = success_cfg.get("transport_target_dist", 0.3)

        # Every stage reads 0 while the bowl is still in the rack. The final
        # "placed" stage is the (non-sticky) success condition, so partial
        # credit without success stays below 1.
        self._staged_reward = StagedReward(
            stages=[
                Stage("grasp_in_rack", 0.15, lambda t, o: t._jaw_grasp_progress()),
                Stage("extract_clear", 0.15, lambda t, o: t._extract_clear_progress(o)),
                Stage("suction_handoff", 0.20, lambda t, o: t._suction_handoff_progress()),
                Stage("transport", 0.15, lambda t, o: t._transport_progress(o)),
                Stage("on_table", 0.15, lambda t, o: t._on_table_progress(o)),
                Stage("upright", 0.10, lambda t, o: t._upright_progress(o)),
                Stage("placed", 0.10, lambda t, o: 1.0 if t._placed(o) else 0.0, sticky=False),
            ],
            success_fn=lambda t, o: t._placed(o),
        )

    # ── stage helpers ────────────────────────────────────────────────

    def _jaw_grasp_progress(self):
        payload = getattr(self._env, "_payload", None)
        geoms = payload.contact_geoms if payload is not None else []
        return 1.0 if self._check_grasp(gripper=self._arm_gripper(0), object_geoms=geoms) else 0.0

    def _extract_clear_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        if pos_key not in raw_obs:
            return 0.0
        fb = getattr(self._env, "_fixture_bounds", None)
        if fb is None:
            return 0.0
        x, y = raw_obs[pos_key][0], raw_obs[pos_key][1]
        inside_rack_xy = fb["x"][0] <= x <= fb["x"][1] and fb["y"][0] <= y <= fb["y"][1]
        return 0.0 if inside_rack_xy else 1.0

    def _suction_handoff_progress(self):
        return 1.0 if self._gripper_contacts_payload_idx(1) else 0.0

    def _transport_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        if pos_key not in raw_obs:
            return 0.0
        fb = getattr(self._env, "_fixture_bounds", None)
        if fb is None or not self._extract_clear_progress(raw_obs):
            return 0.0
        rack_centre_xy = np.array([
            (fb["x"][0] + fb["x"][1]) / 2,
            (fb["y"][0] + fb["y"][1]) / 2,
        ])
        dist = float(np.linalg.norm(np.array(raw_obs[pos_key][:2]) - rack_centre_xy))
        return float(np.clip(dist / self._transport_target_dist, 0.0, 1.0))

    def _on_table_progress(self, raw_obs):
        """1 when the bowl rests at table height outside the rack footprint."""
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        if pos_key not in raw_obs or not self._extract_clear_progress(raw_obs):
            return 0.0
        z = raw_obs[pos_key][2]
        return 1.0 if self._table_height <= z <= self._table_height + 0.05 else 0.0

    def _upright_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        quat_key = f"{pf}quat"
        if quat_key not in raw_obs:
            return 0.0
        mat = T.quat2mat(np.array(raw_obs[quat_key]))
        return 1.0 if mat[2, 2] > 0.9 else 0.0

    def _placed(self, raw_obs) -> bool:
        """Success: upright on the table outside the rack, released and at rest."""
        return (
            self._on_table_progress(raw_obs) >= 1.0
            and self._upright_progress(raw_obs) >= 1.0
            and self._payload_released()
            and self._payload_is_stable_multistep()
        )
