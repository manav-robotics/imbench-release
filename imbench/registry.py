"""Task registry and factory for IMBench.

Provides :data:`TASK_REGISTRY` (a mapping of task IDs to task classes),
:func:`make` to instantiate a task by ID, and :func:`list_tasks` to
enumerate all registered task IDs.

Task IDs follow the naming scheme ``ib-<category>-<name>-v<version>``
where category encodes the physics phenomenon tested:

- ``ib-p1-*`` — P1: Geometry-constrained grasping
- ``ib-p2-*`` — P2: Dynamics and trajectory prediction
- ``ib-p3-*`` — P3: Causal chains (indirect action)
- ``ib-p4-*`` — P4: Tool use (augmented reach)
- ``ib-p5-*`` — P5: Hidden state (active discovery)
- ``ib-p6-*`` — P6: Reactive replanning under perturbation
- ``ib-p7-*`` — P7: Stability and equilibrium construction
- ``ib-m-*``  — M: Miscellaneous / utility tasks
"""

import difflib
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imbench.tasks.base_task import BaseTask

from imbench.tasks.prehensile.edge_slide_v1 import EdgeSlideV1
from imbench.tasks.prehensile.cube_toss_v1 import CubeTossV1
from imbench.tasks.prehensile.tool_retrieve_v1 import ToolRetrieveV1
from imbench.tasks.prehensile.cup_extract_v1 import CupExtractV1
from imbench.tasks.prehensile.occluder_push_v1 import OccluderPushV1
from imbench.tasks.prehensile.sheltered_grasp_v1 import ShelteredGraspV1
from imbench.tasks.prehensile.pendulum_grasp_v1 import PendulumGraspV1
from imbench.tasks.prehensile.domino_single_v1 import DominoSingleV1
from imbench.tasks.prehensile.domino_select_v1 import DominoSelectV1
from imbench.tasks.prehensile.seesaw_balance_v1 import SeesawBalanceV1
from imbench.tasks.prehensile.slip_recovery_v1 import SlipRecoveryV1
from imbench.tasks.prehensile.linear_intercept_v1 import LinearInterceptV1
from imbench.tasks.prehensile.bounce_intercept_v1 import BounceInterceptV1
from imbench.tasks.prehensile.mass_sort_v1 import MassSortV1
from imbench.tasks.prehensile.recover_peg_insert_v1 import RecoverPegInsertV1
from imbench.tasks.prehensile.galileo_ramp_drop_v1 import GalileoRampDropV1
from imbench.tasks.prehensile.slide_catch_v1 import SlideCatchV1
from imbench.tasks.prehensile.mirror_pick_place_v1 import MirrorPickPlaceV1
from imbench.tasks.prehensile.packing_v1 import PackingV1
from imbench.tasks.hard.medium_base_balance_v1 import MediumBaseBalanceV1
from imbench.tasks.hard.hard_base_balance_v1 import HardBaseBalanceV1
from imbench.tasks.puzzle.shape_stack_v1 import ShapeStackV1
from imbench.tasks.puzzle.stack_collapse_recovery_v1 import StackCollapseRecoveryV1
from imbench.tasks.puzzle.ramp_sort_v1 import RampSortV1
from imbench.tasks.puzzle.ball_cage_escape_v1 import BallCageEscapeV1
from imbench.tasks.puzzle.indirect_push_v1 import IndirectPushV1
from imbench.tasks.puzzle.gap_funnel_v1 import GapFunnelV1
from imbench.tasks.puzzle.pick_place_ycb_v1 import PickPlaceYCBV1
from imbench.tasks.custom.freeform_v1 import FreeformV1
from imbench.tasks.hetbi.bowl_table_v1 import BowlTableV1
from imbench.tasks.hetbi.cracker_box_jaw_v1 import CrackerBoxJawV1
from imbench.tasks.hetbi.cracker_box_suction_v1 import CrackerBoxSuctionV1
from imbench.tasks.hetbi.cup_inversion_v1 import CupInversionV1
from imbench.tasks.hetbi.plate_shelf_v1 import PlateShelfV1
from imbench.tasks.hobi.plate_shelf_v1 import PlateShelfV1 as HobiPlateShelfV1
from imbench.tasks.hobi.keyboard_typing_v1 import KeyboardTypingV1
from imbench.tasks.hobi.dining_table_v1 import DiningTableV1

from imbench.tasks.prehensile.edge_slide_v2 import EdgeSlideV2
from imbench.tasks.prehensile.domino_select_v2 import DominoSelectV2
from imbench.tasks.prehensile.galileo_ramp_drop_v2 import GalileoRampDropV2
from imbench.tasks.prehensile.domino_single_v2 import DominoSingleV2
from imbench.tasks.prehensile.occluder_push_v2 import OccluderPushV2
from imbench.tasks.hard.medium_base_balance_v2 import MediumBaseBalanceV2
from imbench.tasks.hard.hard_base_balance_v2 import HardBaseBalanceV2
from imbench.tasks.prehensile.sheltered_grasp_v2 import ShelteredGraspV2
from imbench.tasks.prehensile.recover_peg_insert_v2 import RecoverPegInsertV2
from imbench.tasks.hobi.keyboard_typing_v2 import KeyboardTypingV2
from imbench.tasks.prehensile.slip_recovery_v2 import SlipRecoveryV2
from imbench.tasks.prehensile.pendulum_grasp_v2 import PendulumGraspV2
from imbench.tasks.prehensile.tool_retrieve_v2 import ToolRetrieveV2

TASK_REGISTRY = {
    "ib-p1-edge-slide-v1": EdgeSlideV1,
    # Single-arm prehensile tasks
    "ib-p2-cube-toss-v1": CubeTossV1,
    "ib-p4-tool-retrieve-v1": ToolRetrieveV1,
    "ib-p4-cup-extract-v1": CupExtractV1,
    "ib-p5-occluder-push-v1": OccluderPushV1,
    "ib-p1-sheltered-grasp-v1": ShelteredGraspV1,
    "ib-p2-pendulum-grasp-v1": PendulumGraspV1,
    "ib-p3-domino-single-v1": DominoSingleV1,
    "ib-p3-domino-select-v1": DominoSelectV1,
    "ib-p3-seesaw-balance-v1": SeesawBalanceV1,
    "ib-p6-slip-recovery-v1": SlipRecoveryV1,
    "ib-p2-linear-intercept-v1": LinearInterceptV1,
    "ib-p2-bounce-intercept-v1": BounceInterceptV1,
    "ib-p5-mass-sort-v1": MassSortV1,
    "ib-p1-recover-peg-insert-v1": RecoverPegInsertV1,
    "ib-p2-galileo-ramp-drop-v1": GalileoRampDropV1,
    "ib-p2-slide-catch-v1": SlideCatchV1,
    "ib-m-mirror-pick-place-v1": MirrorPickPlaceV1,
    "ib-p7-packing-v1": partial(PackingV1, task_id="ib-p7-packing-v1"),
    # Hard manipulation tasks
    "ib-p7-balance-medium-v1": MediumBaseBalanceV1,
    "ib-p7-balance-hard-v1": HardBaseBalanceV1,
    # Manipulation puzzles
    "ib-p7-shape-stack-v1": ShapeStackV1,
    "ib-p6-stack-collapse-recovery-v1": StackCollapseRecoveryV1,
    "ib-p2-ramp-sort-v1": RampSortV1,
    "ib-p3-ball-cage-escape-v1": BallCageEscapeV1,
    "ib-p3-indirect-push-v1": IndirectPushV1,
    "ib-p3-gap-funnel-v1": GapFunnelV1,
    "ib-p6-pick-place-ycb-v1": PickPlaceYCBV1,
    # Utility / test tasks
    "ib-m-freeform-v1": FreeformV1,
    # Homogeneous bimanual tasks
    "ib-p1-plate-shelf-hobi-v1": partial(HobiPlateShelfV1, task_id="ib-p1-plate-shelf-hobi-v1"),
    "ib-m-keyboard-typing-v1": partial(KeyboardTypingV1, task_id="ib-m-keyboard-typing-v1"),
    "ib-m-dining-table-v1": partial(DiningTableV1, task_id="ib-m-dining-table-v1"),
    # Heterogeneous bimanual tasks
    "ib-p1-bowl-table-v1": partial(BowlTableV1, task_id="ib-p1-bowl-table-v1"),
    "ib-p1-cracker-box-suction-v1": partial(CrackerBoxSuctionV1, task_id="ib-p1-cracker-box-suction-v1"),
    "ib-p1-cracker-box-jaw-v1": partial(CrackerBoxJawV1, task_id="ib-p1-cracker-box-jaw-v1"),
    "ib-p1-cup-inversion-v1": partial(CupInversionV1, task_id="ib-p1-cup-inversion-v1"),
    "ib-p1-plate-shelf-hetbi-v1": partial(PlateShelfV1, task_id="ib-p1-plate-shelf-hetbi-v1"),

    # Out of dist tasks
    "ib-p1-edge-slide-v2": EdgeSlideV2,
    "ib-p3-domino-select-v2": DominoSelectV2,
    "ib-p2-galileo-ramp-drop-v2": GalileoRampDropV2,
    "ib-p3-domino-single-v2": DominoSingleV2,
    "ib-p5-occluder-push-v2": OccluderPushV2,
    "ib-p7-balance-medium-v2": MediumBaseBalanceV2,
    "ib-p7-balance-hard-v2": HardBaseBalanceV2,
    "ib-p1-sheltered-grasp-v2": ShelteredGraspV2,
    "ib-p1-recover-peg-insert-v2": RecoverPegInsertV2,
    "ib-m-keyboard-typing-v2": KeyboardTypingV2,
    "ib-p6-slip-recovery-v2": SlipRecoveryV2,
    "ib-p2-pendulum-grasp-v2": PendulumGraspV2,
    "ib-p4-tool-retrieve-v2": ToolRetrieveV2,
}


def make(task_id: str, **kwargs) -> "BaseTask":
    """Instantiate an IMBench task by ID.

    Parameters
    ----------
    task_id:
        A registered task identifier, e.g. ``"ib-p2-cube-toss-v1"``.
        Call :func:`list_tasks` to see all available IDs.
    **kwargs:
        Forwarded to the task constructor.  Common options:

        - ``obs_mode`` — ``"combo"`` (proprio + cameras, default), ``"rgb"``
          (alias of ``"combo"``) or ``"proprio"``
        - ``render_mode`` — ``None``, ``"human"``, or ``"rgb_array"``
        - ``seed`` — seed for the first ``reset()`` without an explicit seed
        - ``max_episode_steps`` — episode length cap
        - ``config_overrides`` — dict deep-merged over the task YAML
        - ``privileged_obs`` — ``False`` (default) keeps only what a real
          robot could observe: proprioception, cameras, force/torque and
          tactile. ``True`` also returns simulator ground truth such as
          ``object_state`` and object poses, which the scripted oracles and
          demo collectors read; never use it to evaluate learned policies.

        Unknown keyword arguments raise ``TypeError``.

    Returns
    -------
    BaseTask
        The fully constructed, ready-to-use task instance.

    Raises
    ------
    TypeError
        If *task_id* is not a string.
    ValueError
        If *task_id* is not in :data:`TASK_REGISTRY`.
    """
    if not isinstance(task_id, str):
        raise TypeError(f"task_id must be a str, got {type(task_id).__name__}")
    if task_id not in TASK_REGISTRY:
        close = difflib.get_close_matches(task_id, TASK_REGISTRY, n=3)
        hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"Unknown task {task_id!r}.{hint} See imbench.list_tasks().")
    return TASK_REGISTRY[task_id](**kwargs)


def list_tasks() -> list[str]:
    """Return a sorted list of all registered task IDs."""
    return sorted(TASK_REGISTRY)


def _gymnasium_id(task_id: str) -> str:
    """Gymnasium id of *task_id*: unchanged, except ``imbench-ood/<task_id>`` for ``-v2``.

    Gymnasium reads ``-vN`` as an environment version, so a ``-v2`` id beside
    its ``-v1`` task would make ``gymnasium.make`` tell ``-v1`` users to upgrade
    to the OOD variant, and resolve the unversioned name to it.
    """
    return f"imbench-ood/{task_id}" if task_id.endswith("-v2") else task_id


def _register_gymnasium_envs() -> None:
    """Register every task with the gymnasium registry so ``gymnasium.make()`` works.

    Each task's class (or partial) is used directly as the ``entry_point``.
    When ``gymnasium.make("ib-p1-edge-slide-v1", render_mode="rgb_array")``
    is called, gymnasium invokes ``entry_point(**kwargs)`` — i.e. the task
    class constructor — with whatever kwargs were passed to ``make()``.
    The ``-v2`` OOD variants are registered as ``imbench-ood/<task_id>``
    (see :func:`_gymnasium_id`).
    """
    import gymnasium

    for task_id, cls in TASK_REGISTRY.items():
        gym_id = _gymnasium_id(task_id)
        if gym_id in gymnasium.envs.registry:
            continue
        gymnasium.register(
            id=gym_id,
            entry_point=cls,  # gymnasium calls cls(**kwargs) to create the env
        )


# Register on import so ``gymnasium.make("ib-p1-...")`` works out of the box.
_register_gymnasium_envs()
