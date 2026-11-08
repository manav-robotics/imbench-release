"""Plate Table-to-Shelf Transfer — heterogeneous bimanual task.

Task ID: ``ib-p1-plate-shelf-hetbi-v1``
"""

import numpy as np
import robosuite.utils.transform_utils as T

from imbench.objects import ExtrasObject
from imbench.tasks.hetbi.base_hetbi import HetBiBaseTask
from imbench.tasks.prehensile.contacts import body_geom_ids, geoms_touch
from imbench.tasks.staged_reward import Stage, StagedReward


class PlateShelfV1(HetBiBaseTask):
    """Plate from table into shelf slot using suction + finger grippers.

    Suction gripper (robot 0) attaches to the flat plate surface and
    lifts it, then hands it over to finger gripper (robot 1) which
    grasps the plate edge and inserts it on-edge into the center shelf
    slot.

    Parameters
    ----------
    task_id : str, optional
        Defaults to ``"ib-p1-plate-shelf-hetbi-v1"``.
    **kwargs
        Forwarded to ``HetBiBaseTask``.
    """

    TASK_ID = "ib-p1-plate-shelf-hetbi-v1"
    DESCRIPTION = (
        "Plate lies flat on the table beside a shelf. Suction gripper "
        "(robot 0) attaches to the flat plate surface and lifts it, then "
        "hands it over to finger gripper (robot 1) which grasps the plate "
        "edge and inserts it on-edge into the center shelf slot."
    )

    # Robot indices (from the `grippers:` YAML order: PandaSuctionGripper, PandaGripper).
    _SUCTION_ROBOT_IDX = 0
    _JAW_ROBOT_IDX = 1

    def __init__(self, task_id: str = None, **kwargs):
        super().__init__(task_id=task_id or self.TASK_ID, **kwargs)
        success_cfg = self.cfg.get("task_conditions", {}).get("success", {})
        self._lift_height = success_cfg.get("lift_height") or 0.05
        self._transport_target_dist = success_cfg.get("transport_target_dist", 0.3)
        # On-edge orientation tolerance: the plate's local-Z (face normal)
        # projected onto world-Z must be within this of horizontal (0) for
        # the plate to count as "on edge" rather than lying flat.
        self._on_edge_z_tol = success_cfg.get("on_edge_z_tol", 0.3)
        # Only numerical contact penetration is tolerated, not visible
        # overhang at the mouth or through a divider.
        self._insertion_tolerance = success_cfg.get("insertion_tolerance", 0.0005)

        self._staged_reward = StagedReward(
            stages=[
                Stage("suction_attach", 0.15,
                      lambda t, o: 1.0 if t._gripper_contacts_payload_idx(t._SUCTION_ROBOT_IDX) else 0.0),
                Stage("lift", 0.10, lambda t, o: t._lift_progress(o)),
                Stage("jaw_handoff", 0.15,
                      lambda t, o: float(t._handoff_complete), sticky=False),
                Stage("transport", 0.15, lambda t, o: t._transport_progress(o)),
                Stage("reorient_on_edge", 0.15, lambda t, o: t._on_edge_progress(o)),
                Stage("inserted_in_bounds", 0.20, lambda t, o: t._inserted_in_bounds_progress(o),
                      sticky=False),
                Stage("settled_in_shelf", 0.10, lambda t, o: 1.0 if t._settled_in_shelf(o) else 0.0,
                      sticky=False),
            ],
            success_fn=lambda t, o: t._settled_in_shelf(o),
        )

    def _extra_env_kwargs(self):
        # Keep the canonical shelf name/pose and use a task-specific asset;
        # HoBi and other shelf users keep the default fixture.
        return {"fixture_object": ExtrasObject(
            name="shelf_target", extras_name="plate_shelf_hetbi", joints=None)}

    def reset(self, *, seed=None, options=None):
        self._suction_lifted = False
        self._handoff_complete = False
        self._handoff_last_step = None
        self._jaw_hold_steps = 0
        self._jaw_anchor_relative = None
        self._jaw_anchor_distance = None
        self._jaw_carry_progress = 0.0
        obs, info = super().reset(seed=seed, options=options)
        self._cache_insertion_geometry()
        self._cache_handoff_geometry()
        return obs, info

    def _cache_handoff_geometry(self):
        sim = self._env.sim
        self._payload_geoms = body_geom_ids(sim, self._env._payload.root_body)
        gripper = self._arm_gripper(self._JAW_ROBOT_IDX)
        self._jaw_geoms = body_geom_ids(sim, gripper.root_body)
        self._jaw_body = sim.model.body_name2id(gripper.root_body)
        self._jaw_fingers = tuple(body_geom_ids(sim, f"gripper1_right_{side}finger")
                                  for side in ("left", "right"))
        # Link and snout contact from arm 0 invalidates independent jaw control,
        # even when suction is switched off and no weld remains.
        self._suction_geoms = {g for g in range(sim.model.ngeom)
                               if (sim.model.body_id2name(sim.model.geom_bodyid[g]) or "")
                               .startswith(("robot0_", "gripper0_"))}
        self._plate_diameter = float(np.ptp(self._plate_collision_vertices, axis=0).max())
        # About 5 cm for this plate: larger than grasp drift, but far below
        # the ~40 cm transport in the intended scripted handover. The hold
        # window reuses the task's existing 0.25 s stability scale.
        self._handoff_min_carry = 0.2 * self._plate_diameter
        self._jaw_relative_tolerance = 0.03 * self._plate_diameter
        self._handoff_hold_steps = self._at_rest_window_steps()

    def _jaw_independently_holds_plate(self):
        """Both fingers contact the plate, with no support outside that jaw.

        A brush on the gripper, a plate resting on the table/fixture, or one
        still supported by the suction arm is not an independent handover.
        Stability relative to the jaw and actual carriage are checked over
        time by _update_handoff rather than inferred from one contact frame.
        """
        sim = self._env.sim
        if not all(geoms_touch(sim, finger, self._payload_geoms) for finger in self._jaw_fingers):
            return False
        for contact in sim.data.contact[:sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self._payload_geoms and not pair <= self._payload_geoms | self._jaw_geoms:
                return False
        return True

    def _update_handoff(self):
        if self._handoff_last_step == self._step_count:
            return
        self._handoff_last_step = self._step_count
        sim = self._env.sim
        body = self._env._payload_body_id
        pos = sim.data.body_xpos[body]
        rotation = sim.data.body_xmat[body].reshape(3, 3)
        welded = body in self._suction_weld.welded_body_ids()
        suction_contact = geoms_touch(sim, self._payload_geoms, self._suction_geoms)
        if welded:
            bottom = (self._plate_collision_vertices @ rotation.T + pos)[:, 2].min()
            self._suction_lifted |= bottom > self._table_height + 0.1 * self._plate_diameter
        if welded or suction_contact:
            # Re-grasping/supporting it with suction after a handover cancels
            # the certificate; arm 0 cannot take back the final placement.
            self._handoff_complete = False
        if self._handoff_complete:
            return
        valid = (self._suction_lifted and not welded and not suction_contact
                 and self._jaw_independently_holds_plate())
        relative = ((pos - sim.data.body_xpos[self._jaw_body])
                    @ sim.data.body_xmat[self._jaw_body].reshape(3, 3))
        distance = float(np.linalg.norm(pos - self._env._target_goal_pos))
        if not valid:
            self._jaw_hold_steps = 0
            self._jaw_anchor_relative = None
            self._jaw_anchor_distance = None
            self._jaw_carry_progress = 0.0
            return
        if (self._jaw_anchor_relative is None
                or np.linalg.norm(relative - self._jaw_anchor_relative) > self._jaw_relative_tolerance):
            self._jaw_anchor_relative = relative.copy()
            self._jaw_anchor_distance = distance
            self._jaw_hold_steps = 0
        self._jaw_hold_steps += 1
        self._jaw_carry_progress = max(0.0, self._jaw_anchor_distance - distance)
        if (self._jaw_hold_steps >= self._handoff_hold_steps
                and self._jaw_carry_progress >= self._handoff_min_carry):
            self._handoff_complete = True

    def _compute_task_score(self, raw_obs):
        self._update_handoff()
        return super()._compute_task_score(raw_obs)

    def _cache_insertion_geometry(self):
        """Cache collision vertices and inner faces from this compiled model."""
        sim = self._env.sim
        model = sim.model
        prefix = self._env._fixture.naming_prefix

        def geom(name):
            g = model.geom_name2id(prefix + name)
            return model.geom_pos[g], model.geom_size[g]

        floor, floor_half = geom("col_0")
        back, back_half = geom("col_1")
        left, left_half = geom("col_4")
        right, right_half = geom("col_5")
        self._slot_min = np.array([left[0] + left_half[0],
                                   floor[1] - floor_half[1], floor[2] + floor_half[2]])
        self._slot_max = np.array([right[0] - right_half[0],
                                   back[1] - back_half[1], right[2] + right_half[2]])
        body = self._env._fixture_body_id
        rotation = sim.data.body_xmat[body].reshape(3, 3)
        origin = sim.data.body_xpos[body]
        corners = np.array(np.meshgrid(*zip(self._slot_min, self._slot_max))).reshape(3, -1).T
        world = corners @ rotation.T + origin
        # Compatibility for transport shaping, policies and external callers;
        # insertion itself uses the exact shelf-local volume below.
        self._env._fixture_bounds = {
            axis: (float(world[:, k].min()), float(world[:, k].max()))
            for k, axis in enumerate("xyz")
        }
        self._env._target_goal_pos = (self._slot_min + self._slot_max) / 2 @ rotation.T + origin
        vertices = []
        for g in body_geom_ids(sim, self._env._payload.root_body):
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            mesh = model.geom_dataid[g]
            start = model.mesh_vertadr[mesh]
            points = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            quat = model.geom_quat[g][[1, 2, 3, 0]]
            vertices.append(points @ T.quat2mat(quat).T + model.geom_pos[g])
        self._plate_collision_vertices = np.concatenate(vertices)

    def _insertion_clearances(self, raw_obs):
        """Six signed clearances of the complete plate to the slot's faces.

        Positive is inside. MuJoCo's centred/rotated mesh vertices are first
        put back into the payload body frame, then transformed by the actual
        payload pose into the fixture frame. Every convex collision piece is
        included, so an oblique rim protrusion cannot pass via its centroid.
        """
        pf = self._payload_prefix or "payload_"
        if f"{pf}pos" not in raw_obs or f"{pf}quat" not in raw_obs:
            return None
        sim = self._env.sim
        fixture = self._env._fixture_body_id
        rotation = T.quat2mat(np.asarray(raw_obs[f"{pf}quat"]))
        world = self._plate_collision_vertices @ rotation.T + raw_obs[f"{pf}pos"]
        fixture_rotation = sim.data.body_xmat[fixture].reshape(3, 3)
        local = (world - sim.data.body_xpos[fixture]) @ fixture_rotation
        return np.r_[local.min(axis=0) - self._slot_min,
                     self._slot_max - local.max(axis=0)]

    # ── stage helpers ────────────────────────────────────────────────

    def _lift_progress(self, raw_obs):
        return self._payload_lift_progress(raw_obs, self._lift_height)

    def _fixture_centre(self):
        fixture_bounds = getattr(self._env, "_fixture_bounds", None)
        if fixture_bounds is None:
            return None
        return np.array([
            0.5 * (fixture_bounds["x"][0] + fixture_bounds["x"][1]),
            0.5 * (fixture_bounds["y"][0] + fixture_bounds["y"][1]),
            0.5 * (fixture_bounds["z"][0] + fixture_bounds["z"][1]),
        ])

    def _transport_progress(self, raw_obs):
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        centre = self._fixture_centre()
        if pos_key not in raw_obs or centre is None:
            return 0.0
        dist = float(np.linalg.norm(np.array(raw_obs[pos_key]) - centre))
        return float(np.clip(1.0 - dist / self._transport_target_dist, 0.0, 1.0))

    def _on_edge_progress(self, raw_obs):
        """Plate is "on edge" when its face-normal (local Z) is close to
        horizontal, i.e. its projection onto world-Z is small — as opposed
        to lying flat, where the normal points straight up."""
        pf = self._payload_prefix or "payload_"
        quat_key = f"{pf}quat"
        if quat_key not in raw_obs:
            return 0.0
        mat = T.quat2mat(np.array(raw_obs[quat_key]))
        z_axis_z = abs(mat[2, 2])
        return 1.0 if z_axis_z < self._on_edge_z_tol else 0.0

    def _inserted_in_bounds_progress(self, raw_obs):
        """Insertion needs the entire on-edge plate inside the centre slot."""
        if self._on_edge_progress(raw_obs) < 1.0:
            return 0.0
        clearances = self._insertion_clearances(raw_obs)
        return float(clearances is not None and np.all(clearances >= -self._insertion_tolerance))

    def _settled_in_shelf(self, raw_obs) -> bool:
        """Success after independent jaw carriage, full insertion and release."""
        return (
            self._handoff_complete
            and self._inserted_in_bounds_progress(raw_obs) >= 1.0
            and self._payload_released()
            and self._payload_is_stable_multistep()
        )
