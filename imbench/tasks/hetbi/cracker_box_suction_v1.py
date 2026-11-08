"""Cracker Box — suction-only variant of the slot-insertion task.

Task ID: ``ib-p1-cracker-box-suction-v1``

Objects, fixture and insertion criterion come from
:class:`~imbench.tasks.hetbi.cracker_box_base.CrackerBoxBase`. The box
starts lying flat, the slot is placed within the suction arm's reach, and
there is no handoff stage: the parallel-jaw arm is frozen and only the
suction arm can move.  The robot
index of each gripper is read from ``grippers:`` in the YAML config, so
swapping the list order correctly re-targets the freeze.
"""


from imbench.tasks.hetbi.cracker_box_base import CrackerBoxBase
from imbench.tasks.staged_reward import Stage, StagedReward


# Each arm contributes 7 action dims (6 pose + 1 gripper) under OSC_POSE.
_ARM_ACTION_DIM = 7


def _find_robot_index(grippers: list[str], keyword: str) -> int:
    """Return index of the first gripper whose name contains *keyword*
    (case-insensitive).  Raises ``ValueError`` if no match is found."""
    kw = keyword.lower()
    for i, g in enumerate(grippers):
        if kw in str(g).lower():
            return i
    raise ValueError(
        f"No gripper containing {keyword!r} found in grippers list: {grippers}"
    )


class CrackerBoxSuctionV1(CrackerBoxBase):

    TASK_ID = "ib-p1-cracker-box-suction-v1"
    DESCRIPTION = (
        "Cracker box starts flat on the table beside a horizontal slot. "
        "Only the suction-gripper arm may move; the parallel-jaw arm is "
        "present but frozen.  Suction must pick and slide the box into "
        "the slot on its own."
    )

    # Which gripper keyword identifies the FROZEN arm.  Overridden by the
    # jaw-only variant.
    _FROZEN_GRIPPER_KEYWORD = "pandagripper"
    # Which gripper keyword identifies the ACTIVE (moving) arm.
    _ACTIVE_GRIPPER_KEYWORD = "suction"

    def __init__(self, task_id: str = None, **kwargs):
        super().__init__(task_id=task_id or self.TASK_ID, **kwargs)
        grippers = self.cfg.get("grippers", [])
        frozen_robot = _find_robot_index(grippers, self._FROZEN_GRIPPER_KEYWORD)
        start = frozen_robot * _ARM_ACTION_DIM
        self._frozen_action_slice = slice(start, start + _ARM_ACTION_DIM)
        self._active_robot_idx = _find_robot_index(grippers, self._ACTIVE_GRIPPER_KEYWORD)
        # Re-derive the staged reward now that _active_robot_idx is known
        # (the base __init__ built it before this subclass's __init__ ran).
        self._staged_reward = self._build_staged_reward()

    def _build_staged_reward(self):
        """No handoff stage — a single arm must pick, transport, and
        insert the box entirely on its own."""
        if not hasattr(self, "_active_robot_idx"):
            # Called from CrackerBoxBase.__init__ before this subclass's
            # __init__ has set _active_robot_idx — build a throwaway
            # placeholder; __init__ below rebuilds the real one immediately.
            return super()._build_staged_reward()
        return StagedReward(
            stages=[
                Stage("suction_attach", 0.20,
                      lambda t, o: 1.0 if t._gripper_contacts_payload_idx(t._active_robot_idx) else 0.0),
                Stage("lift", 0.15, lambda t, o: t._lift_progress(o)),
                Stage("transport", 0.25, lambda t, o: t._coarse_transport_progress(o)),
                Stage("insertion", 0.40, lambda t, o: t._fine_insertion_progress(o),
                      sticky=False),
            ],
            success_fn=lambda t, o: t._fine_insertion_progress(o) >= 1.0,
        )

