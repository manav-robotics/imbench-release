"""Dining Table Assembly — homogeneous bimanual task.

A dining table is scattered with common YCB objects (mugs, bowls, fruit,
bottles, …).  Coloured circular goal markers are placed at random positions
on the table surface.  Each arm must pick up its assigned objects and place
them on the matching coloured goal zone.

Assignment is config-driven: ``arm0_objects`` lists the YCB names that the
**scripted** arm (arm 0) must place; ``arm1_objects`` lists those assigned to
the **teleoperated** arm (arm 1).  The colour of each goal zone is fixed per
object (red/blue/green for arm 0; yellow/cyan/magenta for arm 1) so the
operator always knows which marker belongs to which object.

Object scatter positions **and** goal zone positions are both randomised on
every episode reset.  The two sets of positions are kept sufficiently
separated so neither arm starts with an object already on its goal zone.

Scoring:
- An object is placed while its centroid sits within ``xy_tolerance`` of its
  goal zone centre and within ``z_tolerance`` of its resting height, it is
  upright within ``max_tilt_deg`` (round objects listed in ``tilt_exempt``
  are exempt), no gripper grasps it, and it has been at rest for
  ``rest_steps`` consecutive steps. Placement is re-evaluated every step.
- Each object has a staged score (approach → grasp → lift → transport →
  placed) that saturates to 1 while it is placed; the task score is their
  mean. Success requires all objects placed at the same time.

Task ID: ``ib-m-dining-table-v1``
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)
from robosuite.utils import transform_utils as T

import imbench.envs  # noqa: F401 — register envs
from imbench.objects import ExtrasObject, YCBObject
from imbench.tasks.base_task import BaseTask, load_task_config
from imbench.tasks.prehensile.contacts import body_geom_ids, robot_contacts
from imbench.tasks.staged_reward import Stage, StagedReward, progress_from_reset


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Per-object grasp affordances (derived from YCB mesh geometry)
# ---------------------------------------------------------------------------
# Gripper geometry (PandaGripper + grip_site):
#   • grip_site is the grasping point — its world position = robot0_eef_pos.
#   • grip_site z_col ≈ [0, 0.13, -0.99]: approach direction is straight DOWN.
#   • grip_site x_col = [-1, 0, 0]: fingers open / close along world ±X.
#   • Gripper max opening span: ≈ 80 mm (PandaGripper hardware spec).
#   • Workspace stall at full arm reach: EEF z ≈ table_z + 0.089 m.
#
# For reliable scripted grasping every object in arm0 must satisfy:
#   (a) narrow_dim_X ≤ 80 mm  — fits between finger pads
#   (b) centroid_z ≈ table_z + 0.079–0.110 m  — reachable by OSC arm
#       (centroid = half object height above table)
#
# Objects are spawned with identity quaternion (upright, mesh default axes).
# In YCB convention the NARROWEST horizontal mesh extent is along world-X,
# so the gripper's X-opening direction aligns with the narrow face.
#
# Affordance fields:
#   grasp_clearance   target EEF z above table = centroid height.
#                     Arm descends to this z; z-band trigger fires there.
#   hover_height      EEF transit height above table.  Must exceed
#                     obj_half_height + tallest_OTHER_obj_height + 0.04 m
#                     so the carried object clears everything on the table.
#   place_height      EEF z at release.  Object is lowered to rest height.
#   grasp_x_width_mm  Gripper span needed (narrow mesh X dimension).
#                     Shown for documentation; must be < 80 mm.
_GRASP_AFFORDANCES: dict[str, dict] = {
    # ── sugar_box ──────────────────────────────────────────────────────────
    # Mesh BBox (YCB): x=47 mm × y=94 mm × z=177 mm  (standing upright)
    # Narrow face X = 47 mm  → gripper opens in X ✓
    # Centroid above table: 177/2 = 88.5 mm  → grasp_z = 0.9 + 0.089 ≈ 0.989
    "sugar_box": {
        "grasp_clearance":  0.089,   # ≈ centroid (88.5 mm)
        "approach_height":  0.24,    # EEF z during initial hover → obj (lower = faster reach)
        "transit_height":   0.40,    # EEF z when carrying over scene (clears 177+192+40 mm)
        "place_height":     0.20,    # EEF z at release
        "grasp_x_width_mm": 47,
    },
    # ── mustard_bottle ────────────────────────────────────────────────────
    # Mesh BBox (YCB): x=58 mm × y=96 mm × z=192 mm  (standing upright)
    # Narrow face X = 58 mm  → gripper opens in X ✓
    # Centroid: 192/2 = 96 mm → grasp_z = 0.996; arm stalls at ≈0.989, z-band fires
    "mustard_bottle": {
        "grasp_clearance":  0.096,
        "approach_height":  0.26,
        "transit_height":   0.44,    # clears mustard top (192) + cracker (216) + margin
        "place_height":     0.20,
        "grasp_x_width_mm": 58,
    },
    # ── cracker_box ───────────────────────────────────────────────────────
    # Mesh BBox (YCB): x=75 mm × y=166 mm × z=216 mm  (standing upright)
    # Narrow face X = 75 mm  → just fits in gripper (< 80 mm) ✓
    # Centroid: 216/2 = 108 mm → ideal grasp_z = 1.008; arm stalls at ~0.989
    "cracker_box": {
        "grasp_clearance":  0.108,
        "approach_height":  0.28,
        "transit_height":   0.40,
        "place_height":     0.20,
        "grasp_x_width_mm": 75,
    },
    # ── tomato_soup_can ───────────────────────────────────────────────────
    # Mesh BBox (YCB): x=68 mm × y=68 mm × z=103 mm  (cylindrical, upright)
    # Symmetric → no preferred X alignment needed.
    # Centroid: 103/2 = 51.5 mm  → grasp_z = 0.952; arm stalls at 0.989
    # (37 mm above centroid but still within upper half of can top at ≈1.003)
    "tomato_soup_can": {
        "grasp_clearance":  0.052,
        "approach_height":  0.20,
        "transit_height":   0.36,
        "place_height":     0.20,
        "grasp_x_width_mm": 68,
    },
    # ── apple ─────────────────────────────────────────────────────────────
    # Mesh BBox (YCB): x=75 mm × y=75 mm × z=72 mm  (near-sphere)
    # arm1 (teleop) only — centroid too low for scripted arm at full reach.
    "apple": {
        "grasp_clearance":  0.036,
        "approach_height":  0.18,
        "transit_height":   0.18,
        "place_height":     0.15,
        "grasp_x_width_mm": 75,
    },
    # ── orange ────────────────────────────────────────────────────────────
    # Mesh BBox (YCB): x=71 mm × y=73 mm × z=71 mm  (near-sphere)
    # arm1 (teleop) only.
    "orange": {
        "grasp_clearance":  0.034,
        "approach_height":  0.18,
        "transit_height":   0.18,
        "place_height":     0.15,
        "grasp_x_width_mm": 71,
    },
}

# Fallback affordance when an object name is not in the table above.
_DEFAULT_AFFORDANCE: dict = {
    "grasp_clearance":  0.090,
    "approach_height":  0.22,
    "transit_height":   0.36,
    "place_height":     0.20,
    "grasp_x_width_mm": 80,
}

# Default dining objects.
# arm0 (scripted): all three are tall boxes with centroids reachable at the
#   OSC workspace stall height (z ≈ table_z + 0.089 m) and narrow X ≤ 80 mm.
# arm1 (teleop): shorter/spherical — human operator can reach any height.
_DEFAULT_ARM0_OBJECTS = ["sugar_box", "mustard_bottle", "cracker_box"]
_DEFAULT_ARM1_OBJECTS = ["tomato_soup_can", "apple", "orange"]

# Colour marker asset name for each position in each arm's list (3 max).
_ARM0_MARKER_COLORS = ["goal_marker_red", "goal_marker_blue", "goal_marker_green"]
_ARM1_MARKER_COLORS = ["goal_marker_yellow", "goal_marker_cyan", "goal_marker_magenta"]

# OSC action scaling: robosuite default output_max = 0.05 m per unit action.
_OSC_POS_SCALE = 0.05  # m per normalised action unit

# Workspace bounds for scattering objects and placing goal zones.
# Table is 1.2 × 0.8 m (x: ±0.60, y: ±0.40).  Keep ≥0.12 m clear of all
# four edges to prevent wide objects (bowl, bottle, ~0.10 m radius) from
# overhanging and falling off.
_ARM0_SCATTER = dict(x=(-0.22, 0.22), y=(-0.27, -0.05))
_ARM0_GOALS   = dict(x=(-0.22, 0.22), y=(-0.27, -0.05))
_ARM1_SCATTER = dict(x=(-0.22, 0.22), y=(0.05, 0.27))
_ARM1_GOALS   = dict(x=(-0.22, 0.22), y=(0.05, 0.27))

# Minimum goal↔goal separation so markers don't overlap (diameter = 0.16 m).
_MIN_GOAL_GOAL_SEP = 0.17  # m
# Minimum object↔object scatter separation so objects don't collide on spawn.
_MIN_OBJ_OBJ_SEP  = 0.13  # m

# Scripted arm OSC action speed caps.
_SPEED_TRANSIT = 0.55   # fraction of max speed for long-range motion
_SPEED_FINE    = 0.30   # fraction for precision approach

# Z-only descent trigger band (metres above target_z) for grasp_hold.
# The arm stalls at different z values depending on reach configuration.
# Using 0.070 m band ensures the trigger fires when the arm stalls 55+ mm
# above the nominal target z (empirically observed for sugar_box at y=-0.178).
_Z_BAND = 0.070
# Fallback: trigger grasp after this many steps in descend_obj even if
# z-band hasn't fired (handles extreme workspace configurations).
_MAX_DESCEND_STEPS = 200
# Steps arm 0 may spend reaching an object before moving on (unreachable target).
_MAX_HOVER_STEPS = 800
# Layout redraws allowed to find spawns clear of both robots' home poses.
_MAX_LAYOUT_DRAWS = 50


# ---------------------------------------------------------------------------
#  Task class
# ---------------------------------------------------------------------------

class DiningTableV1(BaseTask):
    """Assemble a dining table by placing scattered YCB objects onto their
    matching coloured goal zones.

    Arm 0 (scripted) picks and places the objects in ``arm0_objects``.
    Arm 1 (teleoperated) picks and places those in ``arm1_objects``.

    Parameters
    ----------
    task_id : str, optional
        Defaults to ``"ib-m-dining-table-v1"``.
    **kwargs
        Forwarded to ``BaseTask``.
    """

    TASK_ID = "ib-m-dining-table-v1"
    DESCRIPTION = (
        "Dining-table assembly: YCB objects are scattered on the table. "
        "Arm 0 (scripted) and arm 1 (teleoperated) must each pick up their "
        "assigned objects and place them on the matching coloured circular "
        "goal zone. Object spawn positions and goal zone positions are "
        "randomised every episode."
    )

    def __init__(self, task_id: str = None, **kwargs):
        if task_id:
            self.TASK_ID = task_id
        self.CONFIG_FILE = f"hobi/{self.TASK_ID}"

        _cfg = load_task_config(self.TASK_ID, config_file=self.CONFIG_FILE,
                                overrides=kwargs.get("config_overrides"))

        self.DESCRIPTION = _cfg.get("description", self.DESCRIPTION)
        if "max_episode_steps" in _cfg:
            self.MAX_EPISODE_STEPS = _cfg["max_episode_steps"]

        self._rng = np.random.RandomState(kwargs.get("seed", 0))

        # ── Object lists ────────────────────────────────────────────
        self._arm0_objects: list[str] = list(
            _cfg.get("arm0_objects", _DEFAULT_ARM0_OBJECTS)
        )
        self._arm1_objects: list[str] = list(
            _cfg.get("arm1_objects", _DEFAULT_ARM1_OBJECTS)
        )
        assert len(self._arm0_objects) <= 3, "arm0_objects: max 3 objects supported"
        assert len(self._arm1_objects) <= 3, "arm1_objects: max 3 objects supported"

        # Lift threshold (metres above rest height) considered "lifted" for
        # the per-object `lift` stage — modest, since YCB objects vary widely
        # in size and this only needs to detect "airborne", not a target height.
        self._LIFT_HEIGHT = 0.08
        # Rise above rest height (m) that counts as lifted for transport credit;
        # closing the gripper or dragging raises an object by a few mm.
        self._MIN_LIFT = 0.02

        # Per-object StagedReward — see `_build_object_staged_rewards()`.
        # Keyed by ycb_name; built once here (object assignment is fixed for
        # the lifetime of this task instance), progress reset per-episode in
        # `reset()`.
        self._per_object_staged: dict[str, StagedReward] = (
            self._build_object_staged_rewards()
        )

        # ── Placement detection parameters ───────────────────────────
        det_cfg = _cfg.get("placement_detection", {})
        self._xy_tolerance: float = det_cfg.get("xy_tolerance", 0.08)
        self._z_tolerance: float  = det_cfg.get("z_tolerance", 0.06)
        self._max_tilt_deg: float = det_cfg.get("max_tilt_deg", 20.0)
        self._tilt_exempt: set[str] = set(det_cfg.get("tilt_exempt", ["apple", "orange"]))
        self._rest_steps: int = det_cfg.get("rest_steps", 10)
        self._rest_lin_vel: float = det_cfg.get("rest_lin_vel", 0.05)
        self._rest_ang_vel: float = det_cfg.get("rest_ang_vel", 0.5)

        # ── Scripted arm parameters ──────────────────────────────────
        script_cfg = _cfg.get("scripted_arm", {})
        self._hover_height: float     = script_cfg.get("hover_height", 0.40)
        self._grasp_clearance: float  = script_cfg.get("grasp_clearance", 0.090)
        self._place_height: float     = script_cfg.get("place_height", 0.20)
        self._hold_steps_grasp: int   = script_cfg.get("hold_steps_grasp", 40)
        self._hold_steps_release: int = script_cfg.get("hold_steps_release", 8)
        self._yaml_affordances: dict[str, dict] = script_cfg.get("affordances", {})

        # ── Scene config ──────────────────────────────────────────────
        scene_cfg = _cfg.get("scene", {})
        table_cfg  = scene_cfg.get("table", {})
        self._table_height: float = table_cfg.get("height", 0.9)
        _full_size = table_cfg.get("full_size")
        self._table_full_size = tuple(_full_size) if _full_size else (1.2, 0.80, 0.05)
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)
        _robot_offset = scene_cfg.get("robot_offset")
        self._robot_offset = [list(r) for r in _robot_offset] if _robot_offset else None

        # ── Domain randomisation ─────────────────────────────────────
        rand_cfg = _cfg.get("randomization", {})
        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position: bool = light_cfg.get("position", False)
        self._rand_light_color: bool    = light_cfg.get("color", False)

        # Object scatter bounds (table-centred XY, in metres).
        obj_cfg  = rand_cfg.get("object_pose", {})
        arm0_obj = obj_cfg.get("arm0", {})
        arm1_obj = obj_cfg.get("arm1", {})
        self._arm0_scatter = dict(
            x=tuple(arm0_obj.get("scatter_x", [-0.22, 0.22])),
            y=tuple(arm0_obj.get("scatter_y", [-0.27, -0.05])),
        )
        self._arm1_scatter = dict(
            x=tuple(arm1_obj.get("scatter_x", [-0.22, 0.22])),
            y=tuple(arm1_obj.get("scatter_y", [0.05, 0.27])),
        )
        self._min_obj_sep: float = obj_cfg.get("min_object_separation", 0.13)

        # Goal zone placement bounds.
        goal_cfg  = rand_cfg.get("goal_zones", {})
        arm0_goal = goal_cfg.get("arm0", {})
        arm1_goal = goal_cfg.get("arm1", {})
        self._arm0_goals = dict(
            x=tuple(arm0_goal.get("bounds_x", [-0.22, 0.22])),
            y=tuple(arm0_goal.get("bounds_y", [-0.27, -0.05])),
        )
        self._arm1_goals = dict(
            x=tuple(arm1_goal.get("bounds_x", [-0.22, 0.22])),
            y=tuple(arm1_goal.get("bounds_y", [0.05, 0.27])),
        )
        self._min_goal_sep: float = goal_cfg.get("min_goal_separation", 0.17)

        # ── Runtime state ─────────────────────────────────────────────
        # Populated at reset().
        self._placed_count: int = 0
        self._placed: dict[str, bool] = {}
        # Consecutive steps each object has met every placement condition.
        self._placement_streak: dict[str, int] = {}
        # Objects grasped and raised by _MIN_LIFT; gates transport credit.
        self._lifted: dict[str, bool] = {}
        # (approach, descend) shaping at reset; progress is rescaled from it.
        self._reach_at_reset: dict[str, tuple[float, float]] = {}

        # Goal zone world-XY for each object (name → [x, y]).
        self._goal_xy: dict[str, np.ndarray] = {}

        # Scripted FSM state for arm 0.
        self._scripted_phase: str = "idle"
        self._scripted_obj_idx: int = 0   # index into arm0_objects
        self._scripted_hold_counter: int = 0
        self._scripted_target: np.ndarray | None = None
        self._scripted_phase_steps: int = 0  # steps in the current hover_obj/descend_obj/descend_goal phase
        self._scripted_retry_queue: list[int] = []  # objects to retry after the main pass
        self._scripted_hover_timed_out: set[int] = set()  # objects whose reach already timed out
        # Active affordance for the object currently being handled (set in idle→hover_obj).
        self._scripted_obj_affordance: dict = _DEFAULT_AFFORDANCE.copy()
        # Robot 0 base rotation (world → base), cached at reset.
        self._robot0_base_R: np.ndarray = np.eye(3)

        # Cached object YCBObject handles (populated in _build_env).
        self._ycb_objects: dict[str, YCBObject] = {}
        self._goal_markers: dict[str, ExtrasObject] = {}

        super().__init__(**kwargs)

    # ── Environment construction ─────────────────────────────────────

    def _build_env(self):
        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        # ── Dynamic objects: one YCBObject per assigned item ────────
        dynamic_objs = []
        for ycb_name in self._arm0_objects + self._arm1_objects:
            obj = YCBObject(
                name=f"ycb_{ycb_name}",
                ycb_name=ycb_name,
                density=250.0,
                joints="default",
            )
            self._ycb_objects[ycb_name] = obj
            dynamic_objs.append(obj)

        # ── Static objects: one goal marker per item ────────────────
        static_objs = []
        for i, ycb_name in enumerate(self._arm0_objects):
            marker_asset = _ARM0_MARKER_COLORS[i]
            marker = ExtrasObject(
                name=f"marker_{ycb_name}",
                extras_name=marker_asset,
                joints=None,
            )
            self._goal_markers[ycb_name] = marker
            static_objs.append(marker)
        for i, ycb_name in enumerate(self._arm1_objects):
            marker_asset = _ARM1_MARKER_COLORS[i]
            marker = ExtrasObject(
                name=f"marker_{ycb_name}",
                extras_name=marker_asset,
                joints=None,
            )
            self._goal_markers[ycb_name] = marker
            static_objs.append(marker)

        common_kwargs = dict(
            robots=["Panda", "Panda"],
            gripper_types=["PandaGripper", "PandaGripper"],
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            has_renderer=(self.render_mode == "human"),
            has_offscreen_renderer=use_cameras,
            render_camera=self.viewer_camera,
            renderer_config={"cam_config": self._viewer_cam_config}
            if self.render_mode == "human" else None,
            use_camera_obs=False,  # BaseTask renders camera frames itself
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            horizon=self.MAX_EPISODE_STEPS,
        )

        env = self._make_robosuite_env(
            "HoBiEnv",
            dynamic_objects=dynamic_objs,
            static_objects=static_objs,
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            robot_offset=self._robot_offset,
            **common_kwargs,
        )
        return env

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng.seed(seed)

        sim = self._env.sim
        table_z = float(self._env.table_offset[2])

        # ── Reset task state ─────────────────────────────────────────
        self._placed_count = 0
        self._placed = {name: False for name in
                        self._arm0_objects + self._arm1_objects}
        self._placement_streak = {name: 0 for name in self._placed}
        self._lifted = {name: False for name in self._placed}
        self._reach_at_reset = {name: (0.0, 0.0) for name in self._placed}
        self._goal_xy = {}
        for staged in self._per_object_staged.values():
            staged.reset()
        self._scripted_phase = "lift_arm"  # first lift clear of objects
        self._scripted_obj_idx = 0
        self._scripted_hold_counter = 0
        self._scripted_target = None
        self._scripted_retry_queue = []
        self._scripted_hover_timed_out = set()

        # ── Cache robot 0 base rotation ──────────────────────────────
        try:
            r0_bid = sim.model.body_name2id("robot0_base")
            quat_wxyz = sim.model.body_quat[r0_bid].copy()
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2],
                                   quat_wxyz[3], quat_wxyz[0]])
            self._robot0_base_R = T.quat2mat(quat_xyzw)
        except Exception:
            self._robot0_base_R = np.eye(3)

        # ── Sample goal zone positions (one per object) ───────────────
        # Sample all arm0/arm1 positions (goals + spawns) jointly so that
        # every goal-spawn pair satisfies the separation constraint.  A
        # restartable joint sampler is used instead of sequential sampling to
        # avoid adversarial configurations where the first few positions leave
        # no valid space for the last one.
        n0, n1 = len(self._arm0_objects), len(self._arm1_objects)
        for _ in range(_MAX_LAYOUT_DRAWS):
            arm0_goal_pos, arm0_spawn_pos = self._sample_positions_joint(
                n_goals=n0,
                goal_bounds=self._arm0_goals,
                n_spawns=n0,
                spawn_bounds=self._arm0_scatter,
                min_sep=self._min_obj_sep,
            )
            arm1_goal_pos, arm1_spawn_pos = self._sample_positions_joint(
                n_goals=n1,
                goal_bounds=self._arm1_goals,
                n_spawns=n1,
                spawn_bounds=self._arm1_scatter,
                min_sep=self._min_obj_sep,
            )
            spawns = list(zip(self._arm0_objects, arm0_spawn_pos)) + list(zip(self._arm1_objects, arm1_spawn_pos))
            if all(self._spawn_clears_robots(name, xy, table_z) for name, xy in spawns):
                break
        else:
            logger.warning("No dining layout cleared both home grippers in %d draws", _MAX_LAYOUT_DRAWS)
        for i, ycb_name in enumerate(self._arm0_objects):
            self._goal_xy[ycb_name] = arm0_goal_pos[i]
        for i, ycb_name in enumerate(self._arm1_objects):
            self._goal_xy[ycb_name] = arm1_goal_pos[i]

        # ── Place goal zone markers ───────────────────────────────────
        for ycb_name, marker in self._goal_markers.items():
            gxy = self._goal_xy[ycb_name]
            marker_bid = sim.model.body_name2id(marker.root_body)
            # Sit flush on table surface (marker height ≈ 0.002 m total).
            sim.model.body_pos[marker_bid] = [gxy[0], gxy[1], table_z + 0.001]

        # ── Scatter dynamic objects on table ─────────────────────────
        # Spawn positions were already chosen jointly with goal positions above.
        # Raised 2 cm above the table so objects settle cleanly via gravity.
        for ycb_name, sxy in spawns:
            self._set_object_joint(self._ycb_objects[ycb_name], self._spawn_xyz(ycb_name, sxy, table_z))

        # ── Lighting randomisation ────────────────────────────────────
        if self._rand_light_position or self._rand_light_color:
            for li in range(sim.model.nlight):
                if self._rand_light_position:
                    sim.model.light_pos[li] = [
                        self._rng.uniform(0.0, 2.0),
                        self._rng.uniform(-1.0, 2.0),
                        self._rng.uniform(1.0, 2.5),
                    ]
                if self._rand_light_color:
                    base = self._rng.uniform(0.75, 1.0)
                    sim.model.light_diffuse[li] = np.clip(
                        base + self._rng.uniform(-0.05, 0.05, size=3), 0.6, 1.0
                    )

        # ── Stabilise tall objects with angular damping ───────────────
        for ycb_name in self._arm0_objects + self._arm1_objects:
            obj = self._ycb_objects[ycb_name]
            try:
                jid      = sim.model.joint_name2id(obj.joints[0])
                dof_addr = sim.model.jnt_dofadr[jid]
                for k in range(3, 6):          # rotational DOFs
                    sim.model.dof_damping[dof_addr + k] = 20.0  # N⋅m⋅s/rad — prevents toppling
            except Exception:
                pass

        sim.forward()
        # Reach shaping is measured from the home pose, so an untouched scene scores 0.
        for name in self._placed:
            eef_pos = np.array(sim.data.site_xpos[sim.model.site_name2id(
                f"gripper{self._arm_index_for(name)}_right_grip_site")])
            self._reach_at_reset[name] = (
                self._approach_shaping(name, eef_pos),
                self._descend_shaping(name, eef_pos),
            )

        obs = self._rebuild_obs()
        info = {
            "task_score": 0.0,
            "is_success": False,
            "step": 0,
            "placed_count": 0,
            "total_objects": len(self._arm0_objects) + len(self._arm1_objects),
            "placed": dict(self._placed),
        }
        return obs, info

    # ── Position sampling helper ──────────────────────────────────────

    def _sample_position(
        self,
        bounds: dict,
        existing: list[np.ndarray],
        min_sep: float,
        max_attempts: int = 100,
    ) -> np.ndarray:
        """Sample a random [x, y] within ``bounds`` that is at least
        ``min_sep`` away from all ``existing`` positions."""
        xlo, xhi = bounds["x"]
        ylo, yhi = bounds["y"]
        for _ in range(max_attempts):
            candidate = np.array([
                self._rng.uniform(xlo, xhi),
                self._rng.uniform(ylo, yhi),
            ])
            if all(np.linalg.norm(candidate - ep) >= min_sep
                   for ep in existing):
                return candidate
        # Fallback: return the last candidate even if separation is not met.
        return np.array([
            self._rng.uniform(xlo, xhi),
            self._rng.uniform(ylo, yhi),
        ])

    def _sample_positions_joint(
        self,
        n_goals: int,
        goal_bounds: dict,
        n_spawns: int,
        spawn_bounds: dict,
        min_sep: float,
        max_restarts: int = 300,
        inner_attempts: int = 200,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Sample ``n_goals`` + ``n_spawns`` positions jointly with mutual separation.

        Goals are sampled from ``goal_bounds``; spawns from ``spawn_bounds``.
        All positions (goals and spawns combined) are mutually separated by at
        least ``min_sep``.  A restartable joint sampler avoids adversarial
        configurations where early choices leave no valid space for later ones.
        Returns (goal_positions, spawn_positions).
        """
        n_total = n_goals + n_spawns
        best: list[np.ndarray] = []
        best_types: list[str] = []   # "goal" or "spawn"

        gxlo, gxhi = goal_bounds["x"]
        gylo, gyhi = goal_bounds["y"]
        sxlo, sxhi = spawn_bounds["x"]
        sylo, syhi = spawn_bounds["y"]

        for _restart in range(max_restarts):
            positions: list[np.ndarray] = []
            types: list[str] = []
            ok = True
            for i in range(n_total):
                if i < n_goals:
                    xlo, xhi, ylo, yhi, kind = gxlo, gxhi, gylo, gyhi, "goal"
                else:
                    xlo, xhi, ylo, yhi, kind = sxlo, sxhi, sylo, syhi, "spawn"
                placed = False
                for _ in range(inner_attempts):
                    candidate = np.array([
                        self._rng.uniform(xlo, xhi),
                        self._rng.uniform(ylo, yhi),
                    ])
                    if all(np.linalg.norm(candidate - p) >= min_sep
                           for p in positions):
                        positions.append(candidate)
                        types.append(kind)
                        placed = True
                        break
                if not placed:
                    if len(positions) > len(best):
                        best = positions
                        best_types = types
                    ok = False
                    break
            if ok:
                goals  = [positions[i] for i in range(n_goals)]
                spawns = [positions[i] for i in range(n_goals, n_total)]
                return goals, spawns
        # Exhausted restarts — fill remaining with unchecked random points.
        while len(best) < n_goals:
            best.append(np.array([
                self._rng.uniform(gxlo, gxhi),
                self._rng.uniform(gylo, gyhi),
            ]))
            best_types.append("goal")
        while len(best) < n_total:
            best.append(np.array([
                self._rng.uniform(sxlo, sxhi),
                self._rng.uniform(sylo, syhi),
            ]))
            best_types.append("spawn")
        goals  = [best[i] for i in range(n_goals)]
        spawns = [best[i] for i in range(n_goals, n_total)]
        return goals, spawns

    def _spawn_xyz(self, ycb_name: str, xy, table_z: float) -> list[float]:
        """Spawn position of *ycb_name*: 2 cm above its resting height at *xy*."""
        return [xy[0], xy[1], table_z + abs(self._ycb_objects[ycb_name].bottom_offset[2]) + 0.02]

    def _spawn_clears_robots(self, ycb_name: str, xy, table_z: float) -> bool:
        """Whether *ycb_name* spawned at *xy* is clear of both robots at their home pose."""
        sim = self._env.sim
        obj = self._ycb_objects[ycb_name]
        self._set_object_joint(obj, self._spawn_xyz(ycb_name, xy, table_z))
        sim.forward()
        return not robot_contacts(sim, body_geom_ids(sim, obj.root_body))

    # ── Joint setter helper ───────────────────────────────────────────

    def _set_object_joint(
        self,
        obj: YCBObject,
        pos: list[float],
        quat: list[float] | None = None,
    ) -> None:
        """Set the free-joint state of a dynamic object."""
        if quat is None:
            quat = [1.0, 0.0, 0.0, 0.0]
        self._env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([pos, quat]),
        )

    # ── Object world-position helper ──────────────────────────────────

    def _get_obj_pos(self, ycb_name: str) -> np.ndarray:
        """Return current world-frame centroid of a dynamic YCB object."""
        obj = self._ycb_objects[ycb_name]
        bid = self._env.sim.model.body_name2id(obj.root_body)
        return np.array(self._env.sim.data.body_xpos[bid])

    def _get_obj_quat(self, ycb_name: str) -> np.ndarray:
        """Return current world-frame quaternion (w,x,y,z) of a dynamic object."""
        obj = self._ycb_objects[ycb_name]
        bid = self._env.sim.model.body_name2id(obj.root_body)
        return np.array(self._env.sim.data.body_xquat[bid])  # wxyz

    def _affordance_for(self, ycb_name: str) -> dict:
        """Return the grasp affordance dict for *ycb_name*.

        Priority order:
          1. Per-object YAML override (``scripted_arm.affordances.<name>`` in config)
          2. Curated ``_GRASP_AFFORDANCES`` table (hardcoded in this file)
          3. Auto-derived from YCBObject mesh geometry (``bottom_offset`` /
             ``top_offset``) — works for any YCB object without manual tuning
          4. ``_DEFAULT_AFFORDANCE`` fallback if no YCBObject is loaded yet
        """
        # 1. YAML override (partial override: only specified keys replace defaults)
        if ycb_name in self._yaml_affordances:
            base = _GRASP_AFFORDANCES.get(ycb_name, _DEFAULT_AFFORDANCE).copy()
            base.update(self._yaml_affordances[ycb_name])
            return base
        # 2. Curated table
        if ycb_name in _GRASP_AFFORDANCES:
            return _GRASP_AFFORDANCES[ycb_name]
        # 3. Auto-derive from mesh geometry
        obj = self._ycb_objects.get(ycb_name)
        if obj is not None:
            centroid_h = abs(float(obj.bottom_offset[2]))   # height of centroid above table
            top_h      = abs(float(obj.top_offset[2]))      # extent from centroid to object top
            total_h    = centroid_h + top_h
            return {
                "grasp_clearance":  centroid_h,
                "approach_height":  min(centroid_h + 0.15, 0.35),
                "transit_height":   max(total_h + 0.08, 0.30),
                "place_height":     max(centroid_h * 0.5, 0.12),
                "grasp_x_width_mm": 80,   # conservative; object may be wide
            }
        # 4. Hard fallback
        return _DEFAULT_AFFORDANCE.copy()

    def _get_obj_grasp_target(self, ycb_name: str, table_z: float) -> np.ndarray:
        """Compute world-frame EEF grasp-site target for *ycb_name*."""
        aff = self._affordance_for(ycb_name)
        obj_pos = self._get_obj_pos(ycb_name)
        grasp_z = table_z + aff["grasp_clearance"]
        return np.array([obj_pos[0], obj_pos[1], grasp_z])

    # ── Placement detection ───────────────────────────────────────────

    def _object_rest_z(self, ycb_name: str, table_z: float) -> float:
        """World-frame Z of *ycb_name*'s centroid when resting flat on the table.

        Uses the object's own centroid height (``bottom_offset``) so objects
        of different sizes are all judged "on the table" correctly. Falls
        back to bare ``table_z`` if the YCBObject reference is unavailable.
        """
        ycb_obj = self._ycb_objects.get(ycb_name)
        if ycb_obj is not None:
            return table_z + abs(float(ycb_obj.bottom_offset[2]))
        return table_z

    def _object_tilt_deg(self, ycb_name: str) -> float:
        """Angle between the object's body z-axis (up when spawned) and world up."""
        obj = self._ycb_objects[ycb_name]
        bid = self._env.sim.model.body_name2id(obj.root_body)
        up_z = float(self._env.sim.data.body_xmat[bid][8])
        return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))

    def _object_at_rest(self, ycb_name: str) -> bool:
        obj = self._ycb_objects[ycb_name]
        qvel = self._env.sim.data.get_joint_qvel(obj.joints[0])
        return bool(
            np.linalg.norm(qvel[:3]) < self._rest_lin_vel
            and np.linalg.norm(qvel[3:6]) < self._rest_ang_vel
        )

    def _object_grasped(self, ycb_name: str) -> bool:
        geoms = self._ycb_objects[ycb_name].contact_geoms
        return any(
            self._check_grasp(gripper=robot.gripper, object_geoms=geoms)
            for robot in self._env.robots
        )

    def _object_on_goal(self, ycb_name: str, table_z: float) -> bool:
        """Whether *ycb_name* currently meets every placement condition."""
        gxy = self._goal_xy.get(ycb_name)
        if gxy is None:
            return False
        obj_pos = self._get_obj_pos(ycb_name)
        rest_z = self._object_rest_z(ycb_name, table_z)
        return bool(
            np.linalg.norm(obj_pos[:2] - gxy) <= self._xy_tolerance
            and abs(obj_pos[2] - rest_z) <= self._z_tolerance
            and (ycb_name in self._tilt_exempt
                 or self._object_tilt_deg(ycb_name) <= self._max_tilt_deg)
            and self._object_at_rest(ycb_name)
            and not self._object_grasped(ycb_name)
        )

    def _check_placements(self) -> None:
        """Re-evaluate ``_placed`` and ``_placed_count`` from the current state.

        An object counts as placed once it has met every placement condition
        for ``_rest_steps`` consecutive steps, and stops counting as soon as
        it no longer does.
        """
        table_z = float(self._env.table_offset[2])
        for ycb_name in self._arm0_objects + self._arm1_objects:
            if self._object_on_goal(ycb_name, table_z):
                self._placement_streak[ycb_name] += 1
            else:
                self._placement_streak[ycb_name] = 0
            placed = self._placement_streak[ycb_name] >= self._rest_steps
            if placed != self._placed[ycb_name]:
                logger.info(
                    "[PLACED s=%s]  '%s' %s",
                    self._step_count, ycb_name, "placed" if placed else "no longer placed",
                )
            self._placed[ycb_name] = placed
        self._placed_count = sum(self._placed.values())

    # ── Per-object staged scoring ───────────────────────────────────────
    #
    # arm0_objects are driven by a deterministic scripted FSM and will place
    # themselves regardless of the policy — their per-object score still
    # reports faithfully, it's just not policy-attributable the way arm1's is.

    def _arm_index_for(self, ycb_name: str) -> int:
        return 0 if ycb_name in self._arm0_objects else 1

    def _eef_pos_for(self, ycb_name: str, raw_obs: dict) -> np.ndarray:
        key = f"robot{self._arm_index_for(ycb_name)}_eef_pos"
        return np.array(raw_obs.get(key, [0.0, 0.0, 0.0]))

    def _gripper_contacts_ycb(self, ycb_name: str) -> bool:
        idx = self._arm_index_for(ycb_name)
        obj = self._ycb_objects.get(ycb_name)
        if obj is None:
            return False
        return self._check_grasp(
            gripper=self._env.robots[idx].gripper,
            object_geoms=obj.contact_geoms,
        )

    def _approach_shaping(self, ycb_name: str, eef_pos) -> float:
        return float(1.0 - np.tanh(8.0 * np.linalg.norm(eef_pos - self._get_obj_pos(ycb_name))))

    def _descend_shaping(self, ycb_name: str, eef_pos) -> float:
        target = self._get_obj_grasp_target(ycb_name, float(self._env.table_offset[2]))
        return float(1.0 - np.tanh(8.0 * np.linalg.norm(eef_pos - target)))

    def _approach_progress(self, ycb_name: str, raw_obs: dict) -> float:
        eef_pos = self._eef_pos_for(ycb_name, raw_obs)
        at_reset = self._reach_at_reset[ycb_name][0]
        return progress_from_reset(self._approach_shaping(ycb_name, eef_pos), at_reset)

    def _grasp_descend_progress(self, ycb_name: str, raw_obs: dict) -> float:
        eef_pos = self._eef_pos_for(ycb_name, raw_obs)
        at_reset = self._reach_at_reset[ycb_name][1]
        ramp = progress_from_reset(self._descend_shaping(ycb_name, eef_pos), at_reset)
        contact = 1.0 if self._gripper_contacts_ycb(ycb_name) else 0.0
        return max(ramp, contact)

    def _lift_progress(self, ycb_name: str, raw_obs: dict) -> float:
        # Gated on a grasp so the sticky stage never keeps the 2 cm spawn drop.
        if not self._gripper_contacts_ycb(ycb_name):
            return 0.0
        table_z = float(self._env.table_offset[2])
        rest_z = self._object_rest_z(ycb_name, table_z)
        obj_z = self._get_obj_pos(ycb_name)[2]
        if obj_z - rest_z >= self._MIN_LIFT:
            self._lifted[ycb_name] = True
        return float(np.clip((obj_z - rest_z) / self._LIFT_HEIGHT, 0.0, 1.0))

    def _transport_progress(self, ycb_name: str, raw_obs: dict) -> float:
        gxy = self._goal_xy.get(ycb_name)
        # Only an object that has been lifted earns transport credit.
        if gxy is None or not self._lifted.get(ycb_name, False):
            return 0.0
        xy_dist = float(np.linalg.norm(self._get_obj_pos(ycb_name)[:2] - gxy))
        return 1.0 - np.tanh(6.0 * xy_dist)

    def _placed_progress(self, ycb_name: str, raw_obs: dict) -> float:
        return 1.0 if self._placed.get(ycb_name, False) else 0.0

    def _build_object_staged_rewards(self) -> dict[str, StagedReward]:
        """One StagedReward per assigned object: approach/grasp_descend/lift/
        transport_over_goal/placed. The task score is the mean over objects;
        each saturates to 1.0 while that object is placed."""
        rewards: dict[str, StagedReward] = {}
        for name in self._arm0_objects + self._arm1_objects:
            rewards[name] = StagedReward(
                stages=[
                    Stage("approach", 0.15,
                          lambda t, o, n=name: t._approach_progress(n, o)),
                    Stage("grasp_descend", 0.20,
                          lambda t, o, n=name: t._grasp_descend_progress(n, o)),
                    Stage("lift", 0.15,
                          lambda t, o, n=name: t._lift_progress(n, o)),
                    Stage("transport_over_goal", 0.20,
                          lambda t, o, n=name: t._transport_progress(n, o)),
                    Stage("placed", 0.30,
                          lambda t, o, n=name: t._placed_progress(n, o), sticky=False),
                ],
                success_fn=lambda t, o, n=name: t._placed.get(n, False),
            )
        return rewards

    # ── Scripted arm 0 FSM ────────────────────────────────────────────

    def _scripted_arm_action(self, raw_obs: dict) -> np.ndarray:
        """Return a 7-dim action for arm 0 to pick-and-place its objects.

        FSM phases:
          idle           → select next unplaced object; load its affordance
          hover_obj      → move above the target object (per-obj hover height)
          descend_obj    → lower gripper to grasp-site height (from mesh centroid)
          grasp_hold     → close gripper, hold N steps
          lift_obj       → lift to per-obj transit height (clears table objects)
          hover_goal     → move to above goal zone at transit height
          descend_goal   → lower to place_height
          release_hold   → open gripper, hold N steps
          retract        → lift back to hover height

        Per-object affordances (from ``_GRASP_AFFORDANCES``) control:
          grasp_clearance   target EEF z above table == object mesh centroid height
          approach_height   EEF z while hovering above the object before descent
                            (lower than transit_height → arm reaches object faster,
                             reducing time for unstable objects to fall)
          transit_height    EEF z while carrying the object across the scene
                            (must exceed obj_half_height + tallest_other_top + margin)
          place_height      EEF z at release
        """
        action = np.zeros(7, dtype=np.float64)
        table_z = float(self._env.table_offset[2])

        # Shorthand: affordance for the *currently active* object.
        aff = self._scripted_obj_affordance

        # eef_pos is needed early (before idle returns) for lift_arm phase.
        eef_pos = np.array(raw_obs.get("robot0_eef_pos", [0.0, 0.0, 0.0]))

        # ── lift_arm: move EEF straight up to clear all objects ──────
        # The default Panda joint pose on reset puts the forearm at
        # y ≈ -0.15 m.  Moving straight up first avoids knocking over
        # objects spawned in that region before the scripted arm starts.
        if self._scripted_phase == "lift_arm":
            safe_z = table_z + 0.55   # well above all object tops
            if self._scripted_target is None:
                # Capture initial EEF XY; keep it fixed while lifting.
                self._scripted_target = np.array([
                    eef_pos[0], eef_pos[1], safe_z
                ])
            dist = np.linalg.norm(eef_pos - self._scripted_target)
            action[:3] = self._delta_action(eef_pos, self._scripted_target, _SPEED_TRANSIT)
            action[6] = -1.0   # gripper open
            if dist < 0.05:
                self._scripted_target = None
                self._scripted_phase  = "idle"
            return action

        # ── idle: find next object to place ──────────────────────────
        if self._scripted_phase == "idle":
            while (self._scripted_obj_idx < len(self._arm0_objects) and
                   self._placed.get(self._arm0_objects[self._scripted_obj_idx], False)):
                self._scripted_obj_idx += 1

            if self._scripted_obj_idx >= len(self._arm0_objects):
                # Main pass done — work through retry queue
                while self._scripted_retry_queue:
                    retry_idx = self._scripted_retry_queue.pop(0)
                    retry_name = self._arm0_objects[retry_idx]
                    if not self._placed.get(retry_name, False):
                        logger.debug("[FSM s=%s]  idle: RETRY obj='%s' (idx=%s)", self._step_count, retry_name, retry_idx)
                        self._scripted_obj_idx = retry_idx
                        break
                else:
                    action[6] = -1.0
                    return action
                if self._scripted_obj_idx >= len(self._arm0_objects):
                    action[6] = -1.0
                    return action

            target_name = self._arm0_objects[self._scripted_obj_idx]
            # Load per-object affordance (YAML → curated table → mesh auto-derive).
            aff = self._affordance_for(target_name)
            self._scripted_obj_affordance = aff

            obj_pos     = self._get_obj_pos(target_name)
            # Use approach_height (lower) for initial hover: faster arrival
            # means less sim time for tall, unstable boxes to fall over.
            approach_z  = table_z + aff["approach_height"]
            hover_target = np.array([obj_pos[0], obj_pos[1], approach_z])
            self._scripted_target = hover_target
            self._scripted_phase  = "hover_obj"
            self._scripted_phase_steps = 0  # reset on new object
            logger.debug(
                "[FSM s=%s]  idle -> hover_obj  target='%s' "
                "obj=(%+.3f,%+.3f,%+.3f) approach_z=%.3f  "
                "aff=(gc=%.3f app=%.2f trans=%.2f ph=%.2f x_mm=%s)",
                self._step_count, target_name,
                obj_pos[0], obj_pos[1], obj_pos[2], approach_z,
                aff["grasp_clearance"], aff["approach_height"],
                aff["transit_height"], aff["place_height"], aff["grasp_x_width_mm"],
            )
            action[6] = -1.0
            return action

        target_name = self._arm0_objects[
            min(self._scripted_obj_idx, len(self._arm0_objects) - 1)
        ]

        # ── hover_obj: move to approach height above object ──────────
        if self._scripted_phase == "hover_obj":
            obj_pos = self._get_obj_pos(target_name)
            # Track object XY continuously (may have settled / rolled).
            self._scripted_target[0] = obj_pos[0]
            self._scripted_target[1] = obj_pos[1]
            self._scripted_target[2] = table_z + aff["approach_height"]

            action[6] = -1.0
            action[:3] = self._delta_action(eef_pos, self._scripted_target, _SPEED_TRANSIT)
            dist = np.linalg.norm(eef_pos - self._scripted_target)
            self._scripted_phase_steps += 1
            if dist >= 0.025 and self._scripted_phase_steps >= _MAX_HOVER_STEPS:
                # Unreachable (e.g. knocked into arm 1's half): retry it once
                # after the other objects, and lift clear meanwhile.
                idx = self._scripted_obj_idx
                if idx not in self._scripted_hover_timed_out:
                    self._scripted_hover_timed_out.add(idx)
                    self._scripted_retry_queue.append(idx)
                logger.debug("[FSM s=%s]  hover_obj: timeout on '%s' -- lifting clear", self._step_count, target_name)
                self._scripted_obj_idx += 1
                self._scripted_phase_steps = 0
                self._scripted_target = None
                self._scripted_phase = "lift_arm"
                return action
            if dist < 0.025:
                self._scripted_phase_steps = 0
                # Compute grasp target using per-object affordance:
                # grasp_z = table_z + grasp_clearance ≈ object mesh centroid height.
                # The z-band trigger fires when EEF reaches within _Z_BAND of this z,
                # even at the workspace stall, so the arm always transitions.
                grasp_tgt = self._get_obj_grasp_target(target_name, table_z)
                self._scripted_target = grasp_tgt
                self._scripted_phase  = "descend_obj"
                logger.debug(
                    "[FSM s=%s]  hover_obj -> descend_obj  "
                    "obj='%s'  eef_z=%+.3f  "
                    "grasp_tgt=(%+.3f,%+.3f,%+.3f)  [gc=%.3fm  x_width=%smm]",
                    self._step_count, target_name, eef_pos[2],
                    grasp_tgt[0], grasp_tgt[1], grasp_tgt[2],
                    aff["grasp_clearance"], aff["grasp_x_width_mm"],
                )
            return action

        # ── descend_obj: lower to grasp height ───────────────────────
        if self._scripted_phase == "descend_obj":
            # Continuously track object XY (object may drift on table).
            obj_pos = self._get_obj_pos(target_name)
            self._scripted_target[0] = obj_pos[0]
            self._scripted_target[1] = obj_pos[1]
            # Z stays fixed at table_z + grasp_clearance (from affordance).

            action[6] = -1.0
            action[:3] = self._delta_action(eef_pos, self._scripted_target, _SPEED_FINE)

            self._scripted_phase_steps += 1
            # Z-only descent trigger: fires when EEF z is within _Z_BAND of
            # target z, regardless of XY convergence.  The arm may stall at
            # different heights (z≈1.044 for close objects, z≈1.001 for far)
            # due to OSC workspace limits.  _Z_BAND=0.070 covers both cases.
            # The stall-timeout is a safety net for any remaining edge cases.
            stalled = self._scripted_phase_steps >= _MAX_DESCEND_STEPS
            if (eef_pos[2] <= self._scripted_target[2] + _Z_BAND) or stalled:
                obj_now  = self._get_obj_pos(target_name)
                obj_quat = self._get_obj_quat(target_name)
                dist_dbg = np.linalg.norm(eef_pos - self._scripted_target)
                trig_reason = "stall" if stalled else f"z_band({_Z_BAND}m)"
                # If stalled too far above object (likely blocked by another object),
                # skip to retry queue instead of attempting a hopeless grasp.
                if stalled and (eef_pos[2] - obj_now[2]) > 0.10:
                    logger.debug(
                        "[FSM s=%s]  descend_obj: stall TOO HIGH "
                        "eef_z=%.4f obj_z=%.4f eef-obj_z=%+.4f -- queuing retry",
                        self._step_count, eef_pos[2], obj_now[2], eef_pos[2] - obj_now[2],
                    )
                    self._scripted_phase_steps = 0
                    self._scripted_retry_queue.append(self._scripted_obj_idx)
                    self._scripted_obj_idx += 1
                    self._scripted_target = None
                    self._scripted_phase  = "idle"
                    return action
                self._scripted_phase = "grasp_hold"
                self._scripted_hold_counter = 0
                self._scripted_phase_steps = 0
                logger.debug(
                    "[FSM s=%s]  descend_obj -> grasp_hold  [%s]  "
                    "obj='%s'  eef=(%+.3f,%+.3f,%+.3f)  obj=(%+.3f,%+.3f,%+.3f)  "
                    "eef-obj_z=%+.4f  3d_dist=%.4f  obj_quat(wxyz)=[%+.3f,%+.3f,%+.3f,%+.3f]",
                    self._step_count, trig_reason, target_name,
                    eef_pos[0], eef_pos[1], eef_pos[2],
                    obj_now[0], obj_now[1], obj_now[2],
                    eef_pos[2] - obj_now[2], dist_dbg,
                    obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3],
                )
            return action

        # ── grasp_hold: close gripper ────────────────────────────────
        if self._scripted_phase == "grasp_hold":
            action[6] = 1.0
            self._scripted_hold_counter += 1
            if self._scripted_hold_counter >= self._hold_steps_grasp:
                obj_pos  = self._get_obj_pos(target_name)
                # After grasping, lift to TRANSIT height (higher than approach)
                # to safely carry the object over all other table objects.
                lift_tgt = np.array([obj_pos[0], obj_pos[1],
                                     table_z + aff["transit_height"]])
                self._scripted_target = lift_tgt
                self._scripted_phase  = "lift_obj"
                logger.debug(
                    "[FSM s=%s]  grasp_hold -> lift_obj  obj_z_before=%.4f  "
                    "lift_tgt_z=%.3f  [transit height = table+%.3fm]",
                    self._step_count, obj_pos[2], lift_tgt[2], aff["transit_height"],
                )
            return action

        # ── lift_obj: raise to transit height ────────────────────────
        if self._scripted_phase == "lift_obj":
            action[6] = 1.0
            action[:3] = self._delta_action(eef_pos, self._scripted_target, _SPEED_TRANSIT)
            dist    = np.linalg.norm(eef_pos - self._scripted_target)
            obj_now = self._get_obj_pos(target_name)
            if dist < 0.025:
                gxy = self._goal_xy[target_name]
                obj_rest_z = table_z + abs(float(
                    self._ycb_objects[target_name].bottom_offset[2]))
                # Check if object was actually grasped by verifying it lifted
                # above its resting height.  If not, advance to next object.
                grasp_succeeded = obj_now[2] > obj_rest_z + 0.05
                if not grasp_succeeded:
                    logger.debug(
                        "[FSM s=%s]  lift_obj: GRASP MISS obj='%s' obj_z=%.4f rest_z=%.4f -- queuing retry",
                        self._step_count, target_name, obj_now[2], obj_rest_z,
                    )
                    self._scripted_retry_queue.append(self._scripted_obj_idx)
                    self._scripted_obj_idx += 1
                    self._scripted_target = None
                    self._scripted_phase  = "idle"
                    return action
                # Hover above goal at transit height (same as carry height).
                hover_tgt = np.array([gxy[0], gxy[1], table_z + aff["transit_height"]])
                self._scripted_target = hover_tgt
                self._scripted_phase  = "hover_goal"
                logger.debug(
                    "[FSM s=%s]  lift_obj -> hover_goal  obj_z_lifted=%.4f  goal_xy=(%+.3f,%+.3f)  hover_tgt_z=%.3f",
                    self._step_count, obj_now[2], gxy[0], gxy[1], hover_tgt[2],
                )
            return action

        # ── hover_goal: transit to above goal zone ───────────────────
        if self._scripted_phase == "hover_goal":
            action[6] = 1.0
            action[:3] = self._delta_action(eef_pos, self._scripted_target, _SPEED_TRANSIT)
            dist = np.linalg.norm(eef_pos - self._scripted_target)
            if dist < 0.025:
                gxy       = self._goal_xy[target_name]
                place_tgt = np.array([gxy[0], gxy[1],
                                      table_z + aff["place_height"]])
                self._scripted_target = place_tgt
                self._scripted_phase  = "descend_goal"
                self._scripted_phase_steps = 0  # reset for place stall detection
                obj_now = self._get_obj_pos(target_name)
                logger.debug(
                    "[FSM s=%s]  hover_goal -> descend_goal  "
                    "eef=(%+.3f,%+.3f,%+.3f)  place_tgt=(%+.3f,%+.3f,%+.3f)  carried_obj_z=%.4f",
                    self._step_count,
                    eef_pos[0], eef_pos[1], eef_pos[2],
                    place_tgt[0], place_tgt[1], place_tgt[2], obj_now[2],
                )
            return action

        # ── descend_goal: lower to release height ────────────────────
        if self._scripted_phase == "descend_goal":
            action[6] = 1.0
            action[:3] = self._delta_action(eef_pos, self._scripted_target, _SPEED_FINE)
            dist      = np.linalg.norm(eef_pos - self._scripted_target)
            tgt_z     = self._scripted_target[2]
            # Same stall-detection as descend_obj: arm may hit workspace limit
            # before reaching the nominal place_height (varies with XY position).
            z_band_ok = eef_pos[2] <= tgt_z + _Z_BAND
            stalled   = self._scripted_phase_steps >= _MAX_DESCEND_STEPS
            self._scripted_phase_steps += 1
            if dist < 0.025 or z_band_ok or stalled:
                obj_now = self._get_obj_pos(target_name)
                reason  = ("dist" if dist < 0.025
                           else "stall" if stalled else f"z_band({_Z_BAND}m)")
                self._scripted_phase = "release_hold"
                self._scripted_hold_counter = 0
                logger.debug(
                    "[FSM s=%s]  descend_goal -> release_hold  [%s]  "
                    "eef=(%+.3f,%+.3f,%+.3f)  obj=(%+.3f,%+.3f,%+.3f)",
                    self._step_count, reason,
                    eef_pos[0], eef_pos[1], eef_pos[2],
                    obj_now[0], obj_now[1], obj_now[2],
                )
            return action

        # ── release_hold: open gripper, wait for object to settle ────
        if self._scripted_phase == "release_hold":
            action[6] = -1.0
            self._scripted_hold_counter += 1
            if self._scripted_hold_counter >= self._hold_steps_release:
                gxy      = self._goal_xy[target_name]
                retract_tgt = np.array([gxy[0], gxy[1],
                                        table_z + aff["transit_height"]])
                self._scripted_target = retract_tgt
                self._scripted_phase  = "retract"
                obj_final = self._get_obj_pos(target_name)
                logger.debug(
                    "[FSM s=%s]  release_hold -> retract  "
                    "obj_final=(%+.3f,%+.3f,%+.3f)  goal_xy=(%+.3f,%+.3f)  xy_err=%.4f",
                    self._step_count,
                    obj_final[0], obj_final[1], obj_final[2],
                    gxy[0], gxy[1], np.linalg.norm(obj_final[:2] - gxy),
                )
            return action

        # ── retract: lift away ────────────────────────────────────────
        if self._scripted_phase == "retract":
            action[6] = -1.0
            action[:3] = self._delta_action(eef_pos, self._scripted_target, _SPEED_TRANSIT)
            dist = np.linalg.norm(eef_pos - self._scripted_target)
            if dist < 0.025:
                self._scripted_obj_idx    += 1
                self._scripted_phase       = "idle"
                self._scripted_obj_affordance = _DEFAULT_AFFORDANCE.copy()
                logger.debug(
                    "[FSM s=%s]  retract -> idle  (next obj_idx=%s)",
                    self._step_count, self._scripted_obj_idx,
                )
            return action

        action[6] = -1.0
        return action

    # ── Delta-action helper ───────────────────────────────────────────

    def _delta_action(
        self,
        current: np.ndarray,
        target: np.ndarray,
        speed: float,
    ) -> np.ndarray:
        """Compute a 3-dim normalised OSC_POSE position delta action.

        Transforms the world-frame delta into the robot 0 base frame
        (``input_ref_frame=base``) and clips to ``speed``."""
        delta_world = target - current
        delta_base  = self._robot0_base_R.T @ delta_world
        norm_delta  = delta_base / _OSC_POS_SCALE
        return np.clip(norm_delta, -speed, speed)

    # ── Step ─────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Override step to inject scripted arm 0 action.

        ``action`` is a 14-dim bimanual action
        ``[arm0(7), arm1(7)]`` where each arm's slice is
        ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``.
        The first 7 dims are replaced by the scripted policy.
        """
        action = self._validate_action(action)
        raw_obs = self._get_sim_observations(force_update=True)

        # Scripted arm 0 override.
        scripted = self._scripted_arm_action(raw_obs)
        action[:7] = scripted

        raw_obs_after, _rs_reward, _rs_done, info = self._env.step(action)

        # Placement detection.
        self._check_placements()

        obs, reward, terminated, truncated, info = self._finish_step(raw_obs_after, info)
        info.update({
            "placed_count":  self._placed_count,
            "total_objects": len(self._arm0_objects) + len(self._arm1_objects),
            "placed":        dict(self._placed),
        })
        return obs, reward, terminated, truncated, info

    # ── Observations ─────────────────────────────────────────────────

    def _get_obs(self, raw_obs: dict) -> dict:
        obs = self._build_obs_from_config(raw_obs)

        if self.object_state_enabled:
            obs["placed_count"]   = np.array([self._placed_count], dtype=np.float32)
            obs["total_objects"]  = np.array(
                [len(self._arm0_objects) + len(self._arm1_objects)], dtype=np.float32
            )
            # Per-object positions and goal XY.
            for ycb_name in self._arm0_objects + self._arm1_objects:
                obj_pos = self._get_obj_pos(ycb_name)
                obs[f"obj_{ycb_name}_pos"]  = obj_pos.astype(np.float32)
                obs[f"obj_{ycb_name}_placed"] = np.array(
                    [float(self._placed.get(ycb_name, False))], dtype=np.float32
                )
                gxy = self._goal_xy.get(ycb_name, np.zeros(2))
                obs[f"goal_{ycb_name}_xy"] = gxy.astype(np.float32)
        return obs

    # ── Scoring ───────────────────────────────────────────────────────

    def _compute_task_score(self, raw_obs) -> float:
        names = self._arm0_objects + self._arm1_objects
        if not names:
            return 1.0
        debug = {}
        scores = []
        for name in names:
            score, stage_values = self._per_object_staged[name].compute(self, raw_obs)
            debug[name] = stage_values
            scores.append(score)
        debug["_success"] = all(self._placed.get(name, False) for name in names)
        self._stage_debug = debug
        return float(np.mean(scores))

    # ── Termination ───────────────────────────────────────────────────

    def _check_termination(self, raw_obs) -> bool:
        n_total = len(self._arm0_objects) + len(self._arm1_objects)
        return self._placed_count >= n_total
