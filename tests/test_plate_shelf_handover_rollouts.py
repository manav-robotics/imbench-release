"""Actual robot-action regressions; no plate teleporting or synthetic welds.

Each case drives ``ib-p1-plate-shelf-hetbi-v1`` with real controller actions
and automatic suction attachment:

- ``handover``: the scripted two-arm policy, which must succeed.
- ``suction_only``: the suction arm aligns and drops the plate into the shelf
  on its own; the jaw arm never touches it, so the task must not succeed.
- ``jaw_touch``: the jaw arm grips the plate briefly, then lets go and the
  suction arm drops it alone; a touch is not a handover, so it must fail.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import imbench as ib
from data_synthetic.hetbi_scripted.plate_shelf_policy import (
    R1_PARK_XYZ,
    PlateShelfPolicy,
    _arm_action,
)
from imbench.tasks.prehensile.contacts import body_geom_ids, geoms_touch
from robosuite.utils import transform_utils as T

MAX_STEPS = 7000
CONTAINMENT_TOL = 0.0005
SETTLE_STEPS_AFTER_RELEASE = 250


def _payload_verts_and_shelf_bounds(task):
    """Payload collision vertices (body frame) and the shelf cavity box (fixture frame)."""
    sim = task._env.sim
    model = sim.model
    parts = []
    for g in body_geom_ids(sim, task._env._payload.root_body):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        mesh_id = model.geom_dataid[g]
        start = model.mesh_vertadr[mesh_id]
        verts = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
        rot = T.quat2mat(model.geom_quat[g][[1, 2, 3, 0]])
        parts.append(verts @ rot.T + model.geom_pos[g])

    prefix = task._env._fixture.naming_prefix

    def geom(name):
        i = model.geom_name2id(prefix + name)
        return model.geom_pos[i], model.geom_size[i]

    floor, floor_size = geom("col_0")
    back, back_size = geom("col_1")
    left, left_size = geom("col_4")
    right, right_size = geom("col_5")
    lo = np.array([left[0] + left_size[0], floor[1] - floor_size[1], floor[2] + floor_size[2]])
    hi = np.array([right[0] - right_size[0], back[1] - back_size[1], right[2] + right_size[2]])
    return np.concatenate(parts), lo, hi


def _run_case(case, seed=0):
    cams = {n: {"enabled": n == "agent_cam", "width": 160, "height": 120}
            for n in ["gripper0_cam", "gripper1_cam", "exo_bird_cam", "agent_cam"]}
    task = ib.make("ib-p1-plate-shelf-hetbi-v1", obs_mode="proprio", privileged_obs=True,
                   seed=seed, config_overrides={"obs": {"cameras": cams}})
    obs, _ = task.reset(seed=seed)
    sim = task._env.sim
    payload_id = task._env._payload_body_id
    fixture_id = task._env._fixture_body_id
    origin = sim.data.body_xpos[fixture_id].copy()
    fixture_rot = sim.data.body_xmat[fixture_id].reshape(3, 3).copy()
    verts, lo, hi = _payload_verts_and_shelf_bounds(task)

    payload = body_geom_ids(sim, task._env._payload.root_body)
    jaw = body_geom_ids(sim, task._arm_gripper(1).root_body)
    fingers = [body_geom_ids(sim, f"gripper1_right_{side}finger") for side in ("left", "right")]

    def clearances():
        world = verts @ sim.data.body_xmat[payload_id].reshape(3, 3).T + sim.data.body_xpos[payload_id]
        local = (world - origin) @ fixture_rot
        return np.r_[local.min(0) - lo, hi - local.max(0)]

    policy = PlateShelfPolicy()
    touch = case == "jaw_touch"
    phase = "pickup_pitch"
    tick = touch_run = total_touch = align_ok = 0
    released_at = None
    ever_success = False
    rows = []
    try:
        for i in range(MAX_STEPS):
            r0 = np.asarray(obs["robot0_eef_pos"])
            q0 = np.asarray(obs["robot0_eef_quat"])
            r1 = np.asarray(obs["robot1_eef_pos"])
            q1 = np.asarray(obs["robot1_eef_quat"])
            plate = sim.data.body_xpos[payload_id].copy()
            both = all(geoms_touch(sim, payload, f) for f in fingers)
            total_touch += int(both)
            touch_run = touch_run + 1 if both else 0

            if case == "handover":
                action = policy.act(obs, env=task)
            elif phase == "pickup_pitch":
                # Let the scripted policy pick up and pitch the plate, but keep
                # the jaw arm idle unless this case wants it to grip briefly.
                action = policy.act(obs, env=task)
                if not touch:
                    action[7:13] = 0
                    action[13] = -1
                ready = (policy.phase == "r1_grip" and touch_run >= 5) if touch \
                    else policy.phase == "r1_approach_far"
                if ready:
                    relative = T.quat2mat(q0).T @ sim.data.body_xmat[payload_id].reshape(3, 3)
                    desired = (Rotation.from_euler("x", 10, degrees=True).as_matrix()
                               @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
                    target_quat = T.mat2quat(desired @ relative.T)
                    v_local = verts @ desired.T @ fixture_rot
                    centre = (lo + hi) / 2 - (v_local.min(0) + v_local.max(0)) / 2
                    centre[0] += 0.02
                    if np.ptp(v_local[:, 1]) > hi[1] - lo[1]:
                        centre[1] = hi[1] - v_local[:, 1].max() - 0.004
                    target_xy = (centre @ fixture_rot.T + origin)[:2]
                    drop_z = origin[2] + hi[2] + 0.04
                    high_z = max(drop_z + 0.10, plate[2] + 0.10)
                    phase = "clear_jaw" if touch else "align_high"
                    tick = 0
            else:
                tick += 1
                a1 = (_arm_action(r1, q1, R1_PARK_XYZ, policy._home_r1_quat, -1.0) if touch
                      else np.array([0, 0, 0, 0, 0, 0, -1.0]))
                if phase == "clear_jaw":
                    a0 = _arm_action(r0, q0, r0, target_quat, 1.0)
                    if not both and np.linalg.norm(r1 - plate) > 0.20:
                        phase, tick = "align_high", 0
                elif phase in ("align_high", "lower_to_drop"):
                    goal = np.r_[target_xy, high_z if phase == "align_high" else drop_z]
                    a0 = _arm_action(r0, q0, goal - (plate - r0), target_quat, 1.0)
                    rot_err = T.quat2axisangle(T.quat_multiply(target_quat, T.quat_inverse(q0)))
                    good = np.linalg.norm(plate - goal) < 0.015 and np.linalg.norm(rot_err) < 0.12
                    align_ok = align_ok + 1 if good else 0
                    if align_ok >= 10 or tick >= 2000:
                        phase = "lower_to_drop" if phase == "align_high" else "release_retreat"
                        tick = align_ok = 0
                        if phase == "release_retreat":
                            released_at = i + 1
                            retreat = r0 + np.array([0.08, -0.20, 0.15])
                else:
                    a0 = _arm_action(r0, q0, retreat, target_quat, -1.0)
                action = np.r_[a0, a1]

            action = np.clip(action, task.action_space.low, task.action_space.high)
            obs, _, term, trunc, info = task.step(action)
            ever_success |= info["is_success"]
            row = {
                "fully_contained": bool(np.all(clearances() >= -CONTAINMENT_TOL)),
                "jaw_any_contact": bool(geoms_touch(sim, payload, jaw)),
                "jaw_both_contact": bool(both),
                "handoff_complete": bool(getattr(task, "_handoff_complete", False)),
                "suction_lifted": bool(getattr(task, "_suction_lifted", False)),
                "released": bool(task._payload_released()),
                "stable": bool(task._payload_is_stable_multistep()),
            }
            rows.append(row)
            settled = (released_at is not None and i + 1 - released_at >= SETTLE_STEPS_AFTER_RELEASE
                       and row["stable"] and row["released"])
            if term or trunc or settled:
                break
    finally:
        task.close()
    return {"success": bool(ever_success), "total_jaw_contact_ticks": total_touch,
            "final": rows[-1], "trajectory": rows}


@pytest.mark.parametrize("case, expected", [("handover", True), ("suction_only", False), ("jaw_touch", False)])
def test_real_controller_handover_and_bypasses(case, expected):
    result = _run_case(case)
    final = result["final"]
    assert result["success"] == expected
    assert final["fully_contained"] and final["released"] and final["stable"]
    assert final["suction_lifted"]
    assert final["handoff_complete"] == expected
    if case == "suction_only":
        assert not any(row["jaw_any_contact"] for row in result["trajectory"])
    elif case == "jaw_touch":
        assert result["total_jaw_contact_ticks"] > 0
        assert not any(row["handoff_complete"] for row in result["trajectory"])
    else:
        assert any(row["jaw_both_contact"] for row in result["trajectory"])
