"""Cup Inversion — heterogeneous bimanual task.

Task ID: ``ib-p1-cup-inversion-v1``
"""

import numpy as np
import robosuite.utils.transform_utils as T

from imbench.tasks.base_task import load_task_config
from imbench.tasks.hetbi.base_hetbi import HetBiBaseTask
from imbench.tasks.staged_reward import Stage, StagedReward


class CupInversionV1(HetBiBaseTask):
    """Cup inversion using finger + suction grippers.

    Cup starts inverted (upside-down) on the table with its flat base
    exposed upward.  Suction gripper (robot 1) attaches to the flat base
    and lifts the cup, then hands it over to finger gripper (robot 0)
    which receives it upright and places it inside a green goal region
    on the table.

    Parameters
    ----------
    task_id : str, optional
        Defaults to ``"ib-p1-cup-inversion-v1"``.
    **kwargs
        Forwarded to ``HetBiBaseTask``.
    """

    TASK_ID = "ib-p1-cup-inversion-v1"
    ROBOSUITE_ENV_NAME = "HetBiCupInversionEnv"
    DESCRIPTION = (
        "A handle-less cup (YCB stacking cup) starts inverted (upside-down) "
        "on the table with its flat base exposed upward. Suction gripper "
        "(robot 1) attaches to the flat base and lifts the cup, then hands "
        "it over to finger gripper (robot 0) which receives it upright and "
        "places it inside a green goal region on the table."
    )

    # Robot indices (from the `grippers:` YAML order: PandaGripper, PandaSuctionGripper).
    _JAW_ROBOT_IDX = 0
    _SUCTION_ROBOT_IDX = 1

    def __init__(self, task_id: str = None, **kwargs):
        # Pre-load config so _extra_env_kwargs() has the goal area
        # available when super().__init__ triggers _build_env.
        tid = task_id or self.TASK_ID
        _cfg = load_task_config(tid, config_file=f"hetbi/{tid}",
                                overrides=kwargs.get("config_overrides"))
        goal_cfg = _cfg.get("goal_area", {}) or {}
        self._goal_pos = (
            np.array(goal_cfg["pos"], dtype=np.float64)
            if goal_cfg.get("pos") is not None else None
        )
        size = goal_cfg.get("size", [0.1, 0.1])
        self._goal_half_size = (float(size[0]) / 2.0, float(size[1]) / 2.0)
        self._goal_rgba = tuple(goal_cfg.get("rgba", [0.0, 0.8, 0.0, 0.5]))

        super().__init__(task_id=tid, **kwargs)

        success_cfg = self.cfg.get("task_conditions", {}).get("success", {})
        self._success_lift_height = success_cfg.get("lift_height", 0.10)
        self._was_lifted = False

        # "on_table" only counts after the cup has been lifted (it starts on
        # the table); the final "placed" stage is the (non-sticky) success
        # condition, so partial credit without success stays below 1.
        self._staged_reward = StagedReward(
            stages=[
                Stage("suction_attach", 0.15,
                      lambda t, o: 1.0 if t._gripper_contacts_payload_idx(t._SUCTION_ROBOT_IDX) else 0.0),
                Stage("lift", 0.15, lambda t, o: t._lift_progress(o)),
                Stage("jaw_handoff", 0.15,
                      lambda t, o: 1.0 if t._gripper_contacts_payload_idx(t._JAW_ROBOT_IDX) else 0.0),
                Stage("invert_to_upright", 0.20, lambda t, o: t._upright_progress(o)),
                Stage("on_table", 0.15,
                      lambda t, o: t._on_table_progress(o) if t._was_lifted else 0.0),
                Stage("in_goal_region", 0.10, lambda t, o: t._goal_region_progress(o)),
                Stage("placed", 0.10, lambda t, o: 1.0 if t._placed(o) else 0.0, sticky=False),
            ],
            success_fn=lambda t, o: t._placed(o),
        )

    def reset(self, seed=None, options=None):
        self._was_lifted = False
        return super().reset(seed=seed, options=options)

    def _extra_env_kwargs(self) -> dict:
        if self._goal_pos is None:
            return {}
        return {
            "goal_pos": self._goal_pos.tolist(),
            "goal_half_size": self._goal_half_size,
            "goal_rgba": self._goal_rgba,
        }

    # ── stage helpers ────────────────────────────────────────────────

    def _lift_progress(self, raw_obs):
        progress = self._payload_lift_progress(raw_obs, self._success_lift_height)
        if progress >= 1.0:
            self._was_lifted = True
        return progress

    def _upright_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        quat_key = f"{pf}quat"
        if quat_key not in raw_obs:
            return 0.0
        mat = T.quat2mat(np.array(raw_obs[quat_key]))
        return 1.0 if mat[2, 2] > 0.9 else 0.0

    def _on_table_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        if pos_key not in raw_obs:
            return 0.0
        z = raw_obs[pos_key][2]
        return 1.0 if self._table_height <= z <= self._table_height + 0.05 else 0.0

    def _goal_region_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        if pos_key not in raw_obs:
            return 0.0
        return 1.0 if self._payload_in_goal_region(np.array(raw_obs[pos_key])) else 0.0

    def _payload_in_goal_region(self, payload_pos: np.ndarray) -> bool:
        """Return True if payload xy is in the goal region (or no goal is configured)."""
        return self._env.in_goal_region(payload_pos[:2])

    def _placement_in_goal(self, pos) -> bool:
        return self._goal_pos is not None and self._payload_in_goal_region(np.asarray(pos))

    def _placed(self, raw_obs) -> bool:
        """Success: upright on the table inside the goal region, released and at rest."""
        return (
            self._on_table_progress(raw_obs) >= 1.0
            and self._upright_progress(raw_obs) >= 1.0
            and self._goal_region_progress(raw_obs) >= 1.0
            and self._payload_released()
            and self._payload_is_stable_multistep()
        )
