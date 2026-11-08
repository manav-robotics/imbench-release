"""Plate Table-to-Shelf Transfer — homogeneous bimanual task.

Both arms carry identical parallel-jaw grippers. A flat plate cannot be
grasped by one arm, and a symmetric two-handed pick cannot hand it into the
slot, so one arm presses the rim down to tilt the plate and the other grasps
the raised rim, lifts it alone and inserts it on edge into the shelf.

Task ID: ``ib-p1-plate-shelf-hobi-v1``
"""

from collections import deque

import numpy as np

import imbench.envs  # noqa: F401  — register envs with robosuite
from imbench.objects import YCBObject, ExtrasObject
from imbench.tasks.base_task import BaseTask, load_task_config
from imbench.tasks.physics import commit_model_edits, set_body_mass
from imbench.tasks.placement import jitter_free_joint, table_overhang
from imbench.tasks.prehensile.contacts import body_geom_ids, geoms_touch
from imbench.tasks.staged_reward import Stage, StagedReward


# Shelf inner bounds (must match deps/extras/shelf.xml dimensions)
# Shelf rotated +90° Z: local x→world y, local y→world -x
# dims: W 0.40, D 0.20, H 0.28; dividers at world y = ±0.065, so the
# y range is the centre slot only (less the plate half-thickness).
_SHELF_BOUNDS_OFFSETS = {
    "x": (-0.085, +0.090),
    "y": (-0.06, +0.06),
    "z": (+0.01, +0.27),
}


class PlateShelfV1(BaseTask):
    """Plate from table into shelf using two parallel-jaw grippers.

    One arm presses the plate's rim to tilt it; the other grasps the
    raised rim and inserts the plate on edge into the center shelf slot.
    The staged score rewards that press-then-grasp sequence and gives no
    credit while both arms hold the plate.

    Parameters
    ----------
    task_id : str, optional
        Defaults to ``"ib-p1-plate-shelf-hobi-v1"``.
    **kwargs
        Forwarded to ``BaseTask``.
    """

    TASK_ID = "ib-p1-plate-shelf-hobi-v1"
    DESCRIPTION = (
        "Plate lies flat on the table beside a shelf. Both arms have "
        "identical parallel-jaw grippers. One arm presses the plate's rim "
        "down so it tilts; the other grasps the raised rim, lifts the plate "
        "alone and inserts it on edge into the center shelf slot."
    )

    def __init__(self, task_id: str = None, **kwargs):
        if task_id:
            self.TASK_ID = task_id
        self.CONFIG_FILE = f"hobi/{self.TASK_ID}"

        _cfg = load_task_config(self.TASK_ID, config_file=self.CONFIG_FILE,
                                overrides=kwargs.get("config_overrides"))

        self.DESCRIPTION = _cfg.get("description", self.DESCRIPTION)
        self.YCB_PAYLOAD = _cfg.get("payload", "plate")
        self.FIXTURE = _cfg.get("fixture", "shelf")
        if "max_episode_steps" in _cfg:
            self.MAX_EPISODE_STEPS = _cfg["max_episode_steps"]

        self._rng = np.random.RandomState(kwargs.get("seed", 0))
        self._payload_fell_off = False
        self._stage_debug = {"_success": False}
        self._default_base_quats = None
        self._fixture_bounds = None
        self._pose_history: deque = deque()
        self._pose_history_step = None
        self._was_stable = False

        # Tunables for the staged scoring (see `_build_staged_reward`).
        self._LIFT_HEIGHT = 0.08          # metres above rest height counted "lifted"
        self._REORIENT_ANGLE_DEG = 75.0   # plate-normal-from-vertical angle counted "on edge"
        self._PRESS_TILT_DEG = 8.0        # tilt a press reaches (scripted press: 8-11 deg)
        self._PRESS_MAX_LIFT = 0.03       # plate centre still this close to rest while pressed

        self._staged_reward = self._build_staged_reward()

        # Persistent per-arm target pose (populated in reset). The task
        # integrates user deltas into these targets, then sends IK the
        # correction delta ``target - current_EEF`` each step. This prevents
        # external pushes from dragging the target along, so the arms hold
        # their commanded pose with zero compliance.
        self._target_pos = [None, None]
        self._target_orn_xyzw = [None, None]

        # Placement overrides from config
        self._variant_payload_pos = _cfg.get("payload_pos")
        self._variant_payload_quat = _cfg.get("payload_quat")
        self._variant_fixture_pos = _cfg.get("fixture_pos")
        self._variant_fixture_quat = _cfg.get("fixture_quat")

        # Scene config
        scene_cfg = _cfg.get("scene", {})
        rand_cfg = _cfg.get("randomization", {})

        # Table / arm layout
        table_cfg = scene_cfg.get("table", {})
        self._table_height = table_cfg.get("height", 0.9)
        self._table_full_size = tuple(table_cfg.get("full_size", [1.0, 0.75, 0.05]))
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)
        self._robot_offset = scene_cfg.get("robot_offset")
        self._robot_rotation_z = scene_cfg.get("robot_rotation_z", 0.0)

        # Scene base visuals
        self._object_color = scene_cfg.get("object_color")
        self._table_color = scene_cfg.get("table_color", [1.0, 1.0, 1.0, 1.0])

        # Physics randomization
        mass_cfg = rand_cfg.get("payload_mass", {})
        self._rand_payload_mass = mass_cfg.get("enabled", False)
        self._payload_mass_range = tuple(mass_cfg["range"]) if mass_cfg.get("range") else None

        pfric_cfg = rand_cfg.get("payload_friction", {})
        self._rand_payload_friction = pfric_cfg.get("enabled", False)
        self._payload_friction_range = tuple(pfric_cfg["range"]) if pfric_cfg.get("range") else None

        tfric_cfg = rand_cfg.get("table_friction", {})
        self._rand_table_friction = tfric_cfg.get("enabled", False)
        self._table_friction_range = tuple(tfric_cfg["range"]) if tfric_cfg.get("range") else None

        # Placement randomization
        place_cfg = rand_cfg.get("payload_placement", {})
        self._rand_placement = place_cfg.get("enabled", False)
        self._place_x_std = place_cfg.get("x_std", 0.02)
        self._place_y_std = place_cfg.get("y_std", 0.02)

        # Visual randomization
        self._rand_object_color = rand_cfg.get("object_color", {}).get("enabled", False)
        self._rand_table_color = rand_cfg.get("table_color", {}).get("enabled", False)
        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position = light_cfg.get("position", False)
        self._rand_light_color = light_cfg.get("color", False)

        # Dynamics randomization
        act_noise_cfg = rand_cfg.get("action_noise", {})
        self._rand_action_noise = act_noise_cfg.get("enabled", False)
        self._action_noise_std = act_noise_cfg.get("std", 0.01)

        obs_noise_cfg = rand_cfg.get("observation_noise", {})
        self._rand_obs_noise = obs_noise_cfg.get("enabled", False)
        self._obs_noise_std = obs_noise_cfg.get("std", 0.005)

        # Task conditions
        conditions_cfg = _cfg.get("task_conditions", {})
        failure_cfg = conditions_cfg.get("failure", {})
        self._failure_floor_threshold = failure_cfg.get("floor_drop_below", 0.05)

        # IK input scaling: per-step delta = action * sensitivity
        ik_scale_cfg = _cfg.get("ik_scale", {})
        self._ik_pos_sensitivity = ik_scale_cfg.get("pos_sensitivity", 0.002)
        self._ik_ori_sensitivity = ik_scale_cfg.get("ori_sensitivity", 0.01)

        super().__init__(**kwargs)

    # ── env construction ────────────────────────────────────────────

    def _build_env(self):
        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        # Plate as dynamic object (free joint)
        self._plate = YCBObject(
            name=self.YCB_PAYLOAD, ycb_name=self.YCB_PAYLOAD,
        )
        self._payload_prefix = self._plate.naming_prefix

        # Shelf as static fixture (no joint)
        self._shelf = ExtrasObject(
            name=f"{self.FIXTURE}_target",
            extras_name=self.FIXTURE, joints=None,
        )

        common_kwargs = dict(
            robots=["Panda", "Panda"],
            gripper_types=["PandaGripper", "PandaGripper"],
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            has_renderer=(self.render_mode == "human"),
            has_offscreen_renderer=use_cameras,
            render_camera=self.viewer_camera,
            renderer_config={"cam_config": self._viewer_cam_config} if self.render_mode == "human" else None,
            use_camera_obs=False,  # BaseTask renders camera frames itself
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            horizon=self.MAX_EPISODE_STEPS,
        )

        env_kwargs = dict(
            dynamic_objects=[self._plate],
            static_objects=[self._shelf],
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            robot_offset=self._robot_offset,
            **common_kwargs,
        )

        env = self._make_robosuite_env("HoBiEnv", **env_kwargs)
        return env

    # ── obs ─────────────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)

        if self.object_state_enabled:
            pf = self._payload_prefix or "payload_"
            pos_key = f"{pf}pos"
            quat_key = f"{pf}quat"
            if pos_key in raw_obs:
                obs["object_pos"] = raw_obs[pos_key]
            if quat_key in raw_obs:
                obs["object_quat"] = raw_obs[quat_key]

        return obs

    # ── fixture bounds ──────────────────────────────────────────────

    def _compute_fixture_bounds(self, fix_pos):
        """Compute axis-aligned inner bounds for the shelf fixture."""
        return {
            axis: (fix_pos[i] + lo, fix_pos[i] + hi)
            for i, (axis, (lo, hi)) in enumerate(_SHELF_BOUNDS_OFFSETS.items())
        }

    # ── grasp / geometry helpers ──────────────────────────────────────

    def _gripper_contacts_plate(self, arm_idx: int) -> bool:
        """Whether arm *arm_idx* grasps the plate between both fingers."""
        return self._check_grasp(
            gripper=self._env.robots[arm_idx].gripper,
            object_geoms=self._plate.contact_geoms,
        )

    def _plate_geom_ids(self) -> set[int]:
        sim = self._env.sim
        plate_id = sim.model.body_name2id(self._plate.root_body)
        return set(np.flatnonzero(sim.model.geom_bodyid == plate_id).tolist())

    def _gripper_touches_plate(self, arm_idx: int) -> bool:
        """Whether any geom of arm *arm_idx*'s gripper (taxels included) touches the plate."""
        sim = self._env.sim
        gripper = self._env.robots[arm_idx].gripper["right"]
        return geoms_touch(sim, body_geom_ids(sim, gripper.root_body), self._plate_geom_ids())

    def _plate_rest_z(self, table_z: float) -> float:
        return table_z + abs(float(self._plate.bottom_offset[2]))

    def _plate_angle_from_vertical_deg(self) -> float:
        """Angle between the plate's flat-face normal (local Z) and world-up.

        0 deg == lying flat (normal vertical); 90 deg == standing on edge
        (normal horizontal) — the orientation the shelf slot requires.
        """
        sim = self._env.sim
        bid = sim.model.body_name2id(self._plate.root_body)
        xmat = sim.data.body_xmat[bid].reshape(3, 3)
        body_z = xmat[:, 2]
        cos_angle = np.clip(abs(body_z[2]), 0.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def _plate_in_bounds(self, payload_pos) -> bool:
        if self._fixture_bounds is None:
            return False
        x_ok = self._fixture_bounds["x"][0] <= payload_pos[0] <= self._fixture_bounds["x"][1]
        y_ok = self._fixture_bounds["y"][0] <= payload_pos[1] <= self._fixture_bounds["y"][1]
        z_ok = self._fixture_bounds["z"][0] <= payload_pos[2] <= self._fixture_bounds["z"][1]
        return bool(x_ok and y_ok and z_ok)

    #: Duration (s) the plate must stay released and still to count as at rest.
    _AT_REST_WINDOW_S = 0.25
    #: Tolerance multiplier once at rest, so contact chatter does not flicker the flag.
    _AT_REST_HYSTERESIS = 2.0

    def _plate_released(self) -> bool:
        """True when no geom of either gripper (taxels included) touches the plate."""
        return not any(self._gripper_touches_plate(i) for i in range(len(self._env.robots)))

    def _plate_is_stable_multistep(self, vel_thresh=0.02, ang_vel_thresh=0.2) -> bool:
        """Whether the plate has been released and still for :attr:`_AT_REST_WINDOW_S`.

        Every step in the window must have the plate released, and every pose
        must lie within ``vel_thresh * T`` metres and ``ang_vel_thresh * T``
        radians of the current pose, ``T`` being the window's duration; once
        stable, twice the tolerance is allowed (as in
        ``HetBiBaseTask._payload_is_stable_multistep``).

        Memoized per env tick (keyed on ``self._step_count``) so the pose
        history advances once per env step however often one scoring pass reads it.
        """
        if self._pose_history_step == self._step_count:
            return self._was_stable
        sim = self._env.sim
        bid = sim.model.body_name2id(self._plate.root_body)
        self._pose_history.append(
            (np.array(sim.data.body_xpos[bid]), np.array(sim.data.body_xquat[bid]), self._plate_released())
        )
        self._pose_history_step = self._step_count
        n_steps = max(1, round(self._AT_REST_WINDOW_S * self.control_freq))
        window = list(self._pose_history)[-(n_steps + 1):]
        stable = len(window) == n_steps + 1 and all(released for *_, released in window)
        if stable:
            slack = self._AT_REST_HYSTERESIS if self._was_stable else 1.0
            duration = n_steps / self.control_freq
            pos_now, quat_now, _ = window[-1]
            for pos, quat, _ in window[:-1]:
                angle = 2.0 * np.arccos(np.clip(abs(np.dot(quat, quat_now)), 0.0, 1.0))
                if (np.linalg.norm(pos - pos_now) > slack * vel_thresh * duration
                        or angle > slack * ang_vel_thresh * duration):
                    stable = False
                    break
        self._was_stable = stable
        return stable

    # ── staged scoring ──────────────────────────────────────────────

    def _build_staged_reward(self) -> StagedReward:
        # The canonical strategy is press-then-grasp: one arm presses the rim
        # down so the far rim tilts up, the other arm grasps that rim and
        # carries the plate alone. A two-handed pick earns no credit.
        return StagedReward(
            stages=[
                Stage("press_tilt", 0.15, lambda t, o: t._press_tilt_progress(o)),
                Stage("edge_grasp", 0.15, lambda t, o: t._edge_grasp_progress(o)),
                Stage("single_arm_lift", 0.15, lambda t, o: t._single_arm_lift_progress(o)),
                Stage("transport", 0.15, lambda t, o: t._transport_progress(o)),
                Stage("reorient_on_edge", 0.10, lambda t, o: t._reorient_progress(o)),
                Stage("inserted_in_bounds", 0.20, lambda t, o: t._inserted_in_bounds_progress(o)),
                Stage("settled_in_shelf", 0.10, lambda t, o: 1.0 if t._task_success(o) else 0.0,
                      sticky=False),
            ],
            success_fn=lambda t, o: t._task_success(o),
        )

    def _plate_lift(self, raw_obs) -> float:
        """Height (m) of the plate's centre above its flat resting height."""
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if pos_key not in raw_obs:
            return 0.0
        return float(raw_obs[pos_key][2] - self._plate_rest_z(float(self._env.table_offset[2])))

    def _two_handed(self, raw_obs) -> bool:
        """Both arms grasp the plate, or both hold it up off the table."""
        if self._gripper_contacts_plate(0) and self._gripper_contacts_plate(1):
            return True
        return (self._plate_lift(raw_obs) > self._PRESS_MAX_LIFT
                and self._gripper_touches_plate(0) and self._gripper_touches_plate(1))

    def _press_tilt_progress(self, raw_obs) -> float:
        """A gripper pushes on the resting plate and tilts it up off one rim."""
        if self._two_handed(raw_obs) or self._plate_lift(raw_obs) > self._PRESS_MAX_LIFT:
            return 0.0
        if not (self._gripper_touches_plate(0) or self._gripper_touches_plate(1)):
            return 0.0
        return float(np.clip(self._plate_angle_from_vertical_deg() / self._PRESS_TILT_DEG, 0.0, 1.0))

    def _edge_grasp_progress(self, raw_obs) -> float:
        """Exactly one arm grasps the tilted plate."""
        if self._gripper_contacts_plate(0) == self._gripper_contacts_plate(1):
            return 0.0
        return 1.0 if self._plate_angle_from_vertical_deg() >= 0.5 * self._PRESS_TILT_DEG else 0.0

    def _single_arm_lift_progress(self, raw_obs) -> float:
        """One arm grasps and lifts the plate while the other arm is off it."""
        for holder, other in ((0, 1), (1, 0)):
            if self._gripper_contacts_plate(holder) and not self._gripper_touches_plate(other):
                return float(np.clip(self._plate_lift(raw_obs) / self._LIFT_HEIGHT, 0.0, 1.0))
        return 0.0

    def _transport_progress(self, raw_obs) -> float:
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        if pos_key not in raw_obs or self._fixture_bounds is None or self._two_handed(raw_obs):
            return 0.0
        target_xy = np.array([
            np.mean(self._fixture_bounds["x"]),
            np.mean(self._fixture_bounds["y"]),
        ])
        xy_dist = float(np.linalg.norm(raw_obs[pos_key][:2] - target_xy))
        return 1.0 - np.tanh(6.0 * xy_dist)

    def _reorient_progress(self, raw_obs) -> float:
        if self._two_handed(raw_obs):
            return 0.0
        angle = self._plate_angle_from_vertical_deg()
        return float(np.clip(angle / self._REORIENT_ANGLE_DEG, 0.0, 1.0))

    def _inserted_in_bounds_progress(self, raw_obs) -> float:
        """Plate inside the shelf bounds and reoriented on-edge."""
        pf = self._payload_prefix or "payload_"
        pos_key = f"{pf}pos"
        if pos_key not in raw_obs:
            return 0.0
        inserted = (
            self._plate_in_bounds(raw_obs[pos_key])
            and self._plate_angle_from_vertical_deg() >= self._REORIENT_ANGLE_DEG
        )
        return 1.0 if inserted else 0.0

    def _task_success(self, raw_obs) -> bool:
        """Plate inserted in the shelf slot, on-edge, released and at rest."""
        return (
            self._inserted_in_bounds_progress(raw_obs) >= 1.0
            and self._plate_is_stable_multistep()
        )

    # ── task score ──────────────────────────────────────────────────

    def _plate_below_table(self, raw_obs) -> bool:
        """Latch and return whether the plate has fallen off the table."""
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if raw_obs[pos_key][2] < self._table_height - self._failure_floor_threshold:
            self._payload_fell_off = True
        return self._payload_fell_off

    def _compute_task_score(self, raw_obs):
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if pos_key not in raw_obs or self._plate_below_table(raw_obs):
            self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
            return 0.0
        # Advance the rest window every step, even when the success
        # predicate short-circuits before reaching its stability check.
        self._plate_is_stable_multistep()
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    # ── termination ─────────────────────────────────────────────────

    def _check_termination(self, raw_obs):
        pos_key = f"{self._payload_prefix or 'payload_'}pos"
        if pos_key not in raw_obs:
            return False
        if self._plate_below_table(raw_obs):
            return True
        self._compute_task_score(raw_obs)
        return bool(self._stage_debug["_success"])

    # ── persistent target hold ──────────────────────────────────────

    def _capture_targets_from_current_eef(self, raw_obs):
        """Set persistent targets to each arm's current EEF pose. Called
        at reset so targets start aligned with the robot's home pose.
        """
        for i in (0, 1):
            pkey = f"robot{i}_eef_pos"
            qkey = f"robot{i}_eef_quat"  # xyzw (robosuite convention)
            if pkey in raw_obs:
                self._target_pos[i] = np.array(raw_obs[pkey], dtype=np.float64)
            if qkey in raw_obs:
                self._target_orn_xyzw[i] = np.array(raw_obs[qkey], dtype=np.float64)

    def _integrate_and_correct(self, scaled_action: np.ndarray) -> np.ndarray:
        """Update the persistent target with the user's scaled delta, then
        return the ``target - current_EEF`` correction delta that IK will use.

        Without this, IK recomputes ``target = current_EEF + delta`` every
        step, so an external push drags the reference along with the arm
        (apparent compliance). Tracking the target ourselves keeps it fixed
        in the world frame, so IK drives the arm back each step.
        """
        import robosuite.utils.transform_utils as T

        if self._target_pos[0] is None:
            return scaled_action  # pre-reset safety

        raw_obs = self._get_sim_observations()
        corrected = scaled_action.copy()

        for i, base in enumerate((0, 7)):
            # ── Position: integrate and correct ─────────────────────
            user_dpos = scaled_action[base:base + 3]
            self._target_pos[i] = self._target_pos[i] + user_dpos

            pkey = f"robot{i}_eef_pos"
            if pkey in raw_obs:
                current_pos = np.array(raw_obs[pkey])
                corrected[base:base + 3] = self._target_pos[i] - current_pos

            # ── Orientation: integrate and correct (axis-angle) ─────
            user_axang = scaled_action[base + 3:base + 6]
            if np.any(user_axang):
                dq = T.axisangle2quat(user_axang)  # xyzw
                self._target_orn_xyzw[i] = T.quat_multiply(
                    dq, self._target_orn_xyzw[i]
                )

            qkey = f"robot{i}_eef_quat"
            if qkey in raw_obs:
                current_q = np.array(raw_obs[qkey])  # xyzw
                # Delta quaternion from current to target: q_delta = q_target * q_current^-1
                q_current_inv = T.quat_inverse(current_q)
                q_delta = T.quat_multiply(
                    self._target_orn_xyzw[i], q_current_inv
                )
                corrected[base + 3:base + 6] = T.quat2axisangle(q_delta)

        return corrected

    # ── action scaling ──────────────────────────────────────────────

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale cartesian / orientation deltas so action ∈ [-1, 1] maps
        linearly to a max per-step delta.

        Robosuite's IK uses ``ik_pos_limit`` as a direction-only clip — any
        action with magnitude ≥ the limit produces the *same* motion
        regardless of size. Pre-scaling at the task layer gives true
        proportional control: ``action * pos_sensitivity`` becomes the
        cartesian delta the IK consumes.
        """
        action = np.asarray(action, dtype=np.float64).copy()
        # Per-arm layout: [pos(3), ori(3), gripper(1)] = 7 dims
        for base in (0, 7):
            action[base:base + 3] *= self._ik_pos_sensitivity
            action[base + 3:base + 6] *= self._ik_ori_sensitivity
        return action

    # ── robot base rotation ─────────────────────────────────────────

    def _capture_default_base_quats(self, sim):
        """Snapshot the default robot base quaternions once (after env build)
        so we can compose the user's ``robot_rotation_z`` on top of them.
        """
        if getattr(self, "_default_base_quats", None) is not None:
            return
        self._default_base_quats = {}
        for i in (0, 1):
            bname = f"robot{i}_base"
            try:
                bid = sim.model.body_name2id(bname)
            except Exception:
                continue
            self._default_base_quats[i] = np.array(sim.model.body_quat[bid]).copy()

    def _apply_robot_base_rotation(self, sim):
        """Compose the user's per-robot Z rotation on top of each robot's
        default base orientation.

        HoBiEnv places the two robots in a parallel layout with default base
        quaternions ``[+90°, -90°]`` (so the arms face each other toward the
        table centre). Treating the YAML value as *additive* keeps the
        convention consistent with HetBiEnv, where the defaults were identity:
        the same YAML value produces the same relative arm orientation change.
        """
        if not self._robot_rotation_z:
            return
        self._capture_default_base_quats(sim)
        rot_list = self._robot_rotation_z
        if not isinstance(rot_list, (list, tuple)):
            rot_list = [rot_list, rot_list]
        for i, deg in enumerate(rot_list):
            if i not in self._default_base_quats:
                continue
            rad = np.radians(float(deg))
            half = rad / 2.0
            q_user = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # wxyz
            q_default = self._default_base_quats[i]
            q_composed = self._quat_mul_wxyz(q_user, q_default)
            bid = sim.model.body_name2id(f"robot{i}_base")
            sim.model.body_quat[bid] = q_composed

    @staticmethod
    def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Hamilton product for [w, x, y, z] quaternions (MuJoCo convention)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    # ── actuator overrides ──────────────────────────────────────────

    def _override_arm_actuator_limits(self, sim, limit: float = 1000.0):
        """Widen arm actuator ctrlrange so the IK PD can deliver sufficient
        torque to resist external disturbances (zero-compliance hold).

        Panda's default ctrlrange caps joint torque at ±80 Nm (proximal) and
        ±12 Nm (distal). A push on the end effector saturates these limits
        long before the PD error alone would. Raising the range lets the
        high-kp controller deliver the commanded torque.
        """
        for i in range(sim.model.nu):
            name = sim.model.actuator_id2name(i)
            if name and "_torq_j" in name:
                sim.model.actuator_ctrlrange[i] = [-limit, limit]

    # ── reset ───────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._payload_fell_off = False
        self._pose_history = deque(maxlen=max(1, round(self._AT_REST_WINDOW_S * self.control_freq)) + 1)
        self._pose_history_step = None
        self._was_stable = False
        self._staged_reward.reset()
        self._stage_debug = {**self._staged_reward.stage_values, "_success": False}
        obs, info = super().reset(seed=seed, options=options)

        if seed is not None:
            self._rng = np.random.RandomState(seed)
        sim = self._env.sim
        table_z = self._env.table_offset[2]

        # Widen arm actuator limits so stiff PD can hold against disturbances.
        self._override_arm_actuator_limits(sim)

        # ── Place plate (dynamic object) ────────────────────────────
        if self._variant_payload_pos is not None:
            pay_pos = np.array(self._variant_payload_pos, dtype=np.float64)
            if len(pay_pos) == 2:
                pay_pos = np.array([
                    pay_pos[0], pay_pos[1],
                    table_z + abs(self._plate.bottom_offset[2]) + 0.002,
                ])
        else:
            pay_pos = np.array([
                0.1, 0.0,
                table_z + abs(self._plate.bottom_offset[2]) + 0.002,
            ])
        pay_quat = (
            np.array(self._variant_payload_quat, dtype=np.float64)
            if self._variant_payload_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0])
        )
        sim.data.set_joint_qpos(
            self._plate.joints[0],
            np.concatenate([pay_pos, pay_quat]),
        )

        # ── Place shelf (static fixture) ────────────────────────────
        shelf_bid = sim.model.body_name2id(self._shelf.root_body)
        if self._variant_fixture_pos is not None:
            fix_pos = np.array(self._variant_fixture_pos, dtype=np.float64)
        else:
            fix_pos = np.array([
                -0.25, 0.0,
                table_z + abs(self._shelf.bottom_offset[2]) + 0.001,
            ])
        fix_quat = (
            np.array(self._variant_fixture_quat, dtype=np.float64)
            if self._variant_fixture_quat is not None
            else np.array([0.7071068, 0.0, 0.0, 0.7071068])  # +90° Z
        )
        sim.model.body_pos[shelf_bid] = fix_pos
        sim.model.body_quat[shelf_bid] = fix_quat

        # Compute shelf bounds for success detection
        self._fixture_bounds = self._compute_fixture_bounds(fix_pos)

        # ── Physics randomization ───────────────────────────────────
        self._randomize_physics(sim)

        # ── Placement randomization ─────────────────────────────────
        if self._rand_placement:
            self._randomize_placement(sim)

        # ── Visual randomization ────────────────────────────────────
        self._randomize_visuals(sim)

        # Apply per-robot base Z rotation LAST (composes with default facing
        # quat set during env build). Placed here so nothing downstream
        # overwrites body_quat before sim.forward recomputes kinematics.
        self._apply_robot_base_rotation(sim)

        sim.forward()

        # Re-observe after modifications.  force_update=True is required
        # because robosuite caches observables per timestep — without it
        # the cameras return stale images from before visual randomization.
        self._camera_cache.clear()
        raw_obs = self._get_sim_observations(force_update=True)
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._observe(raw_obs)

        # Initialise persistent hold targets to the arms' home EEF pose.
        self._capture_targets_from_current_eef(raw_obs)

        info = {"task_score": 0.0, "is_success": False, "step": 0}
        return obs, info

    def step(self, action: np.ndarray):
        # Validate and clip first: a rejected action must not reach the
        # persistent targets, and the IK correction below is not re-clipped.
        action = self._validate_action(action)

        # Action noise
        if self._rand_action_noise:
            action = action + self._rng.normal(0, self._action_noise_std,
                                               size=action.shape)

        # Scale cartesian / orientation deltas for proportional fine-grained
        # control. Gripper channel (idx 6, 13) passes through unchanged.
        action = self._scale_action(action)

        # Integrate the user delta into a persistent target, then send IK the
        # correction delta so external pushes don't drag the reference along.
        action = self._integrate_and_correct(action)

        raw_obs, _rs_reward, _rs_done, info = self._env.step(action)
        obs, reward, terminated, truncated, info = self._finish_step(raw_obs, info)

        # Observation noise (proprio keys only)
        if self._rand_obs_noise:
            for key in obs:
                if isinstance(obs[key], np.ndarray) and key.startswith("robot"):
                    obs[key] = obs[key] + self._rng.normal(
                        0, self._obs_noise_std, size=obs[key].shape
                    )

        return obs, reward, terminated, truncated, info

    # ── physics randomization ───────────────────────────────────────

    def _randomize_physics(self, sim):
        plate_bid = sim.model.body_name2id(self._plate.root_body)

        if self._rand_payload_mass and self._payload_mass_range:
            set_body_mass(sim, plate_bid, self._rng.uniform(*self._payload_mass_range))
            commit_model_edits(sim)

        if self._rand_payload_friction and self._payload_friction_range:
            fric = self._rng.uniform(*self._payload_friction_range)
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == plate_bid:
                    sim.model.geom_friction[i][0] = fric

        if self._rand_table_friction and self._table_friction_range:
            fric = self._rng.uniform(*self._table_friction_range)
            for i in range(sim.model.ngeom):
                name = sim.model.geom_id2name(i)
                if name and "table" in name and "visual" not in name:
                    sim.model.geom_friction[i][0] = fric

    # ── placement randomization ─────────────────────────────────────

    def _randomize_placement(self, sim):
        """Jitter the plate's x/y (see :func:`jitter_free_joint`).

        A start is invalid if it overhangs the table further than the nominal
        position, penetrates the shelf, or starts inside the shelf bounds.
        """
        joint_name = self._plate.joints[0]
        radius = float(self._plate.horizontal_radius)
        nominal_xy = np.array(sim.data.get_joint_qpos(joint_name)[:2])
        max_overhang = table_overhang(self._env, radius, nominal_xy)
        plate_id = sim.model.body_name2id(self._plate.root_body)
        shelf_id = sim.model.body_name2id(self._shelf.root_body)

        def valid(pos) -> bool:
            if table_overhang(self._env, radius, pos[:2]) > max_overhang + 1e-9 or self._plate_in_bounds(pos):
                return False
            for c in sim.data.contact[:sim.data.ncon]:
                bodies = {int(sim.model.geom_bodyid[c.geom1]), int(sim.model.geom_bodyid[c.geom2])}
                if bodies == {plate_id, shelf_id} and c.dist < 0.0:
                    return False
            return True

        jitter_free_joint(sim, joint_name, self._rng, (self._place_x_std, self._place_y_std), valid)

    # ── visual randomization ────────────────────────────────────────

    def _randomize_visuals(self, sim):
        plate_bid = sim.model.body_name2id(self._plate.root_body)

        # Object color
        if self._rand_object_color:
            obj_rgba = self._random_object_color()
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == plate_bid:
                    sim.model.geom_matid[i] = -1
                    sim.model.geom_rgba[i] = obj_rgba
        elif self._object_color is not None:
            obj_rgba = np.array(self._object_color)
            for i in range(sim.model.ngeom):
                if sim.model.geom_bodyid[i] == plate_bid:
                    sim.model.geom_matid[i] = -1
                    sim.model.geom_rgba[i] = obj_rgba

        # Table color
        table_rgba = (
            self._random_table_color() if self._rand_table_color
            else np.array(self._table_color)
        )
        for i in range(sim.model.ngeom):
            name = sim.model.geom_id2name(i)
            if name and name == "table_visual":
                sim.model.geom_rgba[i] = table_rgba
                mat_id = sim.model.geom_matid[i]
                if mat_id >= 0:
                    sim.model.mat_rgba[mat_id] = table_rgba

        # Lighting
        for i in range(sim.model.nlight):
            if self._rand_light_position:
                sim.model.light_pos[i] = [
                    self._rng.uniform(0.0, 2.0),
                    self._rng.uniform(-1.0, 2.0),
                    self._rng.uniform(1.0, 2.5),
                ]
            if self._rand_light_color:
                base = self._rng.uniform(0.6, 1.0)
                jitter = self._rng.uniform(-0.08, 0.08, size=3)
                diffuse = np.clip(base + jitter, 0.4, 1.0)
                sim.model.light_diffuse[i] = diffuse
                amb_base = self._rng.uniform(0.05, 0.25)
                amb_jitter = self._rng.uniform(-0.04, 0.04, size=3)
                sim.model.light_ambient[i] = np.clip(
                    amb_base + amb_jitter, 0.0, 0.4
                )

    # ── color helpers ───────────────────────────────────────────────

    def _random_object_color(self):
        rgb = self._rng.uniform(0.15, 0.85, size=3)
        gray = rgb.mean()
        rgb = 0.5 * rgb + 0.5 * gray
        rgb = np.clip(rgb, 0.1, 0.9)
        return np.array([*rgb, 1.0])

    def _random_table_color(self):
        base = self._rng.uniform(0.45, 0.95)
        jitter = self._rng.uniform(-0.06, 0.06, size=3)
        jitter[0] += self._rng.uniform(0.0, 0.04)
        rgb = np.clip(base + jitter, 0.3, 1.0)
        return np.array([*rgb, 1.0])
