"""Cracker Box — parallel-jaw-only variant of the slot-insertion task.

Task ID: ``ib-p1-cracker-box-jaw-v1``

Same objects, fixture and insertion criterion as the suction-only variant
(:class:`~imbench.tasks.hetbi.cracker_box_base.CrackerBoxBase`), with its
own layout (box and slot placed within the parallel-jaw arm's reach) and no
handoff stage: the suction arm
is frozen and only the parallel-jaw arm can move.  The robot index of each
gripper is read from ``grippers:`` in the YAML config, so swapping the list
order correctly re-targets the freeze.
"""

from imbench.tasks.hetbi.cracker_box_suction_v1 import CrackerBoxSuctionV1
from imbench.tasks.staged_reward import Stage, StagedReward


class CrackerBoxJawV1(CrackerBoxSuctionV1):

    TASK_ID = "ib-p1-cracker-box-jaw-v1"
    DESCRIPTION = (
        "Cracker box starts flat on the table beside a horizontal slot. "
        "Only the parallel-jaw arm may move; the suction-gripper arm is "
        "present but frozen.  The parallel jaw must pick and slide the "
        "box into the slot on its own."
    )

    # Freeze the suction arm instead.
    _FROZEN_GRIPPER_KEYWORD = "suction"
    _ACTIVE_GRIPPER_KEYWORD = "pandagripper"

    def __init__(self, task_id: str = None, **kwargs):
        super().__init__(task_id=task_id or self.TASK_ID, **kwargs)

    def _build_staged_reward(self):
        """No handoff stage — the parallel jaw must pick, transport, and
        insert the box entirely on its own. Uses `_check_grasp` (a firm
        two-finger pinch check) rather than mere contact, so the grasp
        stage only counts a real parallel-jaw grasp."""
        if not hasattr(self, "_active_robot_idx"):
            return super()._build_staged_reward()

        def _jaw_grasp(t, o):
            payload = getattr(t._env, "_payload", None)
            geoms = payload.contact_geoms if payload is not None else []
            gripper = t._arm_gripper(t._active_robot_idx)
            return 1.0 if t._check_grasp(gripper=gripper, object_geoms=geoms) else 0.0

        return StagedReward(
            stages=[
                Stage("jaw_grasp", 0.20, _jaw_grasp),
                Stage("lift", 0.15, lambda t, o: t._lift_progress(o)),
                Stage("transport", 0.25, lambda t, o: t._coarse_transport_progress(o)),
                Stage("insertion", 0.40, lambda t, o: t._fine_insertion_progress(o),
                      sticky=False),
            ],
            success_fn=lambda t, o: t._fine_insertion_progress(o) >= 1.0,
        )
