"""Contact-driven suction weld for the PandaSuctionGripper.

Activated / deactivated **per-step** by calling
``SuctionWeld.update(env, action)`` from the task's ``step()`` override.
The weld is localised to whichever robot carries a suction gripper —
it never touches parallel-jaw robots.

Gripper action convention:
    Action is continuous in [-1, 1].
    For suction grippers, this is effectively binary:
        positive (> 0)  → activate weld on contact (suck)
        negative (< 0)  → deactivate weld (release)
    For parallel-jaw grippers, only the sign matters: the jaws move toward open/closed at a fixed rate per step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _WeldHandles:
    """Resolved MuJoCo IDs for one suction gripper's weld constraint."""
    valid: bool = False
    eq_id: int = -1
    snout_geom_id: int = -1
    eef_body_id: int = -1
    welded_body_id: int | None = None


def _name2id(sim: object, obj_type: str, name: str) -> int:
    model = sim.model
    if obj_type == "body":
        return model.body_name2id(name)
    if obj_type == "geom":
        return model.geom_name2id(name)
    if obj_type == "equality":
        mj_model = model if isinstance(model, mujoco.MjModel) else model._model
        return int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, name))
    return -1


def _id2name(sim: object, obj_type: str, idx: int) -> str | None:
    model = sim.model
    if obj_type == "body":
        return model.body_id2name(idx)
    if obj_type == "geom":
        return model.geom_id2name(idx)
    return None


def _contact_arrays(
    sim: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = sim.data
    ncon = int(data.ncon)
    if ncon <= 0:
        empty_i = np.array([], dtype=np.int32)
        return empty_i, empty_i, np.array([], dtype=np.float64), np.zeros((0, 3))
    return (
        np.asarray(data.contact.geom1[:ncon], dtype=np.int32),
        np.asarray(data.contact.geom2[:ncon], dtype=np.int32),
        np.asarray(data.contact.dist[:ncon], dtype=np.float64),
        np.asarray(data.contact.pos[:ncon], dtype=np.float64),
    )


def _body_pose(sim: object, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    quat = np.asarray(sim.data.body_xquat[body_id], dtype=np.float64)
    return pos, quat


def _current_relpose(
    p_eef: np.ndarray,
    q_eef_wxyz: np.ndarray,
    p_obj: np.ndarray,
    q_obj_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld relpose that holds body2 exactly where it is, in the EEF frame.

    Pulling the contact point onto the EEF origin would bury the object in the
    snout and pump meshes, and the stored penetration launches it on release.

    Returns (p_rel, q_rel) in MuJoCo wxyz quaternion convention.
    """
    rot1 = R.from_quat(T.convert_quat(q_eef_wxyz, to="xyzw"))
    rot2 = R.from_quat(T.convert_quat(q_obj_wxyz, to="xyzw"))
    p_rel = rot1.inv().apply(p_obj - p_eef)
    q_rel_wxyz = T.convert_quat((rot1.inv() * rot2).as_quat(), to="wxyz")
    return p_rel, q_rel_wxyz


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

class SuctionWeld:
    """Manages per-robot suction weld constraints.

    Create one instance per task.  Call :meth:`update` inside ``step()``
    and :meth:`reset` inside ``reset()``.

    Parameters
    ----------
    contact_threshold : float
        Maximum contact distance (metres) for a suction weld to trigger.
    excluded_body_ids : set[int] | None
        Body IDs that should never be welded to (e.g. static fixtures).
    """

    def __init__(
        self,
        contact_threshold: float = 0.02,
        excluded_body_ids: set[int] | None = None,
    ) -> None:
        self.contact_threshold = contact_threshold
        self.excluded_body_ids: set[int] = excluded_body_ids or set()
        # cache keyed by (robot_idx, arm_name)
        self._handles: dict[tuple[int, str], _WeldHandles] = {}

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear cached handles — call after ``env.reset()``."""
        self._handles.clear()

    def welded_body_ids(self) -> set[int]:
        """Body IDs currently held by an active suction weld."""
        return {
            h.welded_body_id for h in self._handles.values()
            if h.welded_body_id is not None
        }

    # ------------------------------------------------------------------
    #  Per-step update — THE main entry point
    # ------------------------------------------------------------------

    def update(self, env: object, action: np.ndarray) -> None:
        """Process suction logic for every suction gripper in *env*.

        Call this once per ``env.step(action)`` AFTER the physics step so
        that contact information is up-to-date.
        """
        try:
            robots = env.robots
            sim = env.sim
            model = sim.model

            per_robot_dims = [r.action_dim for r in robots]
            total_dim = sum(per_robot_dims)
            action_vec = np.asarray(action, dtype=np.float64).reshape(-1)
            if action_vec.size < total_dim:
                return

            g1, g2, dist, _ = _contact_arrays(sim)
            mutated = False

            for ridx, robot in enumerate(robots):
                for arm in list(robot.arms):
                    gripper = robot.gripper[arm]
                    gtype = type(gripper).__name__.lower()
                    if "suction" not in gtype and "vacuum" not in gtype:
                        continue

                    handles = self._resolve(env, ridx, arm, gripper)
                    if not handles or not handles.valid:
                        continue

                    # Extract this robot's gripper action (last dim)
                    offset = sum(per_robot_dims[:ridx])
                    grip_idx = offset + per_robot_dims[ridx] - 1
                    grip_act = float(action_vec[grip_idx])

                    if grip_act == 0.0:
                        continue

                    # --- RELEASE ---
                    if grip_act < 0:
                        if handles.welded_body_id is not None:
                            sim.data.eq_active[handles.eq_id] = 0
                            handles.welded_body_id = None
                            mutated = True
                        else:
                            sim.data.eq_active[handles.eq_id] = 0
                        continue

                    # --- SUCTION ON ---
                    if handles.welded_body_id is not None:
                        # already welded — keep active
                        sim.data.eq_active[handles.eq_id] = 1
                        continue

                    # Find closest valid contact with the snout
                    contact_result = self._best_contact(
                        model, handles.snout_geom_id, g1, g2, dist
                    )
                    if contact_result is None:
                        sim.data.eq_active[handles.eq_id] = 0
                        continue

                    target_geom, best_contact_idx = contact_result
                    body2_id = int(model.geom_bodyid[target_geom])

                    body1_id = handles.eef_body_id
                    # Set weld bodies
                    model.eq_obj1id[handles.eq_id] = int(body1_id)
                    model.eq_obj2id[handles.eq_id] = int(body2_id)

                    # Weld the object where it is, so release leaves no
                    # stored penetration to eject it.
                    p1, q1 = _body_pose(sim, body1_id)
                    p2, q2 = _body_pose(sim, body2_id)
                    p_rel, q_rel = _current_relpose(p1, q1, p2, q2)
                    model.eq_data[handles.eq_id, 0:3] = np.zeros(3)
                    model.eq_data[handles.eq_id, 3:6] = p_rel
                    model.eq_data[handles.eq_id, 6:10] = q_rel

                    sim.data.eq_active[handles.eq_id] = 1
                    handles.welded_body_id = body2_id
                    mutated = True

                    target_name = _id2name(sim, "body", body2_id) or str(body2_id)
                    logger.debug("Suction weld activated → %s", target_name)

            if mutated:
                sim.forward()

        except Exception:
            logger.exception("SuctionWeld.update error (ignored)")

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _resolve(
        self, env: object, robot_id: int, arm: str, gripper: object
    ) -> _WeldHandles | None:
        key = (int(robot_id), str(arm))
        if key in self._handles:
            return self._handles[key]

        sim = env.sim
        cn = gripper.correct_naming
        try:
            eq_id = _name2id(sim, "equality", cn("suction_constraint"))
            snout_id = _name2id(sim, "geom", cn("geo_snout"))
            eef_id = _name2id(sim, "body", cn("eef"))
        except Exception:
            self._handles[key] = _WeldHandles()
            return self._handles[key]

        h = _WeldHandles(
            valid=(eq_id != -1 and snout_id != -1 and eef_id != -1),
            eq_id=eq_id,
            snout_geom_id=snout_id,
            eef_body_id=eef_id,
        )
        self._handles[key] = h
        return h

    def _best_contact(
        self,
        model,
        snout_id: int,
        g1: np.ndarray,
        g2: np.ndarray,
        dist: np.ndarray,
    ) -> tuple[int, int] | None:
        """Find the closest weldable contact involving the snout geom.

        Contacts with the world body or ``excluded_body_ids`` are skipped
        before choosing, so a nearer static contact never masks a weldable one.

        Returns (other_geom_id, contact_index) or None.
        """
        if g1.size == 0:
            return None
        other = np.where(g1 == snout_id, g2, g1)
        bodies = model.geom_bodyid[other]
        mask = ((g1 == snout_id) | (g2 == snout_id)) & (dist < self.contact_threshold) & (bodies > 0)
        if self.excluded_body_ids:
            mask &= ~np.isin(bodies, list(self.excluded_body_ids))
        if not np.any(mask):
            return None
        idxs = np.where(mask)[0]
        best = idxs[np.argmin(dist[idxs])]
        return int(other[best]), int(best)
