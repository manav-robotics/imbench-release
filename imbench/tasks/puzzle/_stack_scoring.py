"""Shared contact-counting / stacking-progress helpers.

Factored out of ``ShapeStackV1`` (``imbench/tasks/puzzle/shape_stack_v1.py``)
and ``StackCollapseRecoveryV1``
(``imbench/tasks/puzzle/stack_collapse_recovery_v1.py``), which both score a
"stack N objects until only one touches the table" arrangement using the
identical MuJoCo-body-subtree contact-counting walk and the identical dense
stacking-progress / hold-duration-ramp formulas. Kept here as plain
functions (not a base class) so either task can import just the pieces it
needs without a shared-inheritance refactor.

Usage from a task's staged-reward stage functions::

    from imbench.tasks.puzzle._stack_scoring import (
        object_body_id, body_subtree_geom_ids,
        count_table_and_gripper_contacts, stacking_progress,
        hold_duration_ramp,
    )

    def _update_stack_contact_state(self):
        n_touching, gripper_contact = count_table_and_gripper_contacts(
            self._env, [a["name"] for a in self._selected_archetypes],
        )
        ...
"""

from __future__ import annotations


def is_gripper_geom(name: str) -> bool:
    return ("gripper" in name) or ("finger" in name) or ("taxel" in name)


def object_body_id(sim, name: str) -> int | None:
    """Resolve a MuJoCo body id for an object name.

    Primitives end in ``_main``, composites in ``_root``; the bare name is
    tried too, and a substring scan catches any non-standard naming.
    """
    for suffix in ("_main", "_root", ""):
        try:
            return sim.model.body_name2id(name + suffix)
        except Exception:
            continue
    for bid in range(sim.model.nbody):
        bname = sim.model.body_id2name(bid) or ""
        if bname == name or bname.startswith(name + "_") or bname.endswith("_" + name):
            return bid
    return None


def body_subtree_geom_ids(sim, root_bid: int) -> set[int]:
    """All colliding geom ids attached to ``root_bid`` or its descendants.

    ``CompositeBodyObject.contact_geoms`` returns ``[]`` in robosuite, so we
    resolve composite geoms by walking the MuJoCo body tree directly.
    """
    subtree: set[int] = {root_bid}
    for bid in range(sim.model.nbody):
        cur = bid
        for _ in range(16):
            if cur in subtree:
                subtree.add(bid)
                break
            parent = int(sim.model.body_parentid[cur])
            if parent == cur:
                break
            cur = parent
    ids: set[int] = set()
    for gid in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[gid]) not in subtree:
            continue
        if (int(sim.model.geom_contype[gid]) == 0
                and int(sim.model.geom_conaffinity[gid]) == 0):
            continue
        ids.add(gid)
    return ids


def count_table_and_gripper_contacts(
    env, object_names: list[str],
) -> tuple[int, bool]:
    """Return ``(num_touching_table, any_gripper_contact)`` for the given
    object names, walking each object's body subtree for its collision
    geoms and checking contact against the table and gripper geoms.
    """
    sim = env.sim
    table_geoms = ["table_collision"]
    gripper_geoms = [
        sim.model.geom_id2name(g) for g in range(sim.model.ngeom)
        if sim.model.geom_id2name(g)
        and is_gripper_geom(sim.model.geom_id2name(g))
    ]

    num_touching_table = 0
    any_gripper_contact = False
    for name in object_names:
        root_bid = object_body_id(sim, name)
        if root_bid is None:
            continue
        ids = body_subtree_geom_ids(sim, root_bid)
        geom_names = [
            sim.model.geom_id2name(gid) for gid in ids
            if sim.model.geom_id2name(gid)
        ]
        if not geom_names:
            continue
        if env.check_contact(geom_names, table_geoms):
            num_touching_table += 1
        if gripper_geoms and env.check_contact(geom_names, gripper_geoms):
            any_gripper_contact = True

    return num_touching_table, any_gripper_contact


def stacking_progress(num_touching_table: int, num_objects: int) -> float:
    """Dense stacking progress: 0 when every object still touches the
    table (fully collapsed/scattered) or none does (nothing is resting on
    the table), 1 when exactly one does (fully stacked/restacked).
    """
    if num_touching_table <= 0:
        return 0.0
    if num_objects <= 1:
        return 1.0
    return (num_objects - num_touching_table) / (num_objects - 1)


def hold_duration_ramp(stable_count: int, stable_steps_required: int) -> float:
    """Linear ramp of a sustained-hold counter toward 1.0."""
    return stable_count / max(1, stable_steps_required)
