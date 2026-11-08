from __future__ import annotations

import logging
import numbers
import os
import re
import sys
import warnings
from abc import abstractmethod
from contextlib import contextmanager, suppress
from pathlib import Path
import gymnasium
from gymnasium import spaces
import mujoco
import numpy as np
import yaml
import json
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.models.grippers import GripperModel
from robosuite.utils.binding_utils import MjRenderContext

logger = logging.getLogger(__name__)

from imbench.utils.tactile_setup import (
    make_tactile_env_class,
    get_tactile_readings as _tactile_readings,
)


CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "tasks"

# Observation keys a real robot could also provide: proprioception and tactile
# (robot<i>_*), cameras, and wrist force/torque. Everything else a task returns
# (object_state, object poses, progress counters) is simulator ground truth and
# is dropped unless a task is built with privileged_obs=True.
_POLICY_OBS_KEY = re.compile(r"robot\d+_\w+|rgb_\w+|depth_\w+|force_torque(_world)?")


def load_task_config(task_id: str, config_file: str | None = None, overrides: dict | None = None) -> dict:
    """Load YAML config for a task, with optional dict overrides.

    Parameters
    ----------
    task_id : str
        Task identifier (used as filename when *config_file* is ``None``).
    config_file : str, optional
        Explicit config filename (without ``.yaml``).  When set, the config
        is loaded from ``configs/tasks/{config_file}.yaml`` instead of
        ``{task_id}.yaml``.  Useful for task families that share one config.
    overrides : dict, optional
        Recursive dict merge applied on top of the loaded YAML.
    """
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError(f"config_overrides must be a dict or None, got {type(overrides).__name__}")
    if not CONFIGS_DIR.is_dir():
        raise FileNotFoundError(
            f"Task configs not found at {CONFIGS_DIR}. imbench must be installed in "
            f"editable mode from a full git checkout: `pip install -e .`"
        )
    fname = config_file or task_id
    cfg_path = CONFIGS_DIR / f"{fname}.yaml"
    if not cfg_path.exists():
        # Search subdirectories (e.g. configs/tasks/hetbi/<task>.yaml)
        matches = list(CONFIGS_DIR.rglob(f"{fname}.yaml"))
        if matches:
            cfg_path = matches[0]
        else:
            raise FileNotFoundError(f"No config found at {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, modifying base in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _bind_before_free(del_fn):
    """Wrap ``MjRenderContext.__del__`` so it frees its own GL objects.

    robosuite frees against whatever context is current: a collected context
    then deletes the live context's GL objects and unbinds it, corrupting later
    frames. Every render path here rebinds its own context first.
    """

    def __del__(self):
        gl_ctx = getattr(self, "gl_ctx", None)
        if getattr(gl_ctx, "_context", None):
            with suppress(Exception):
                gl_ctx.make_current()
        del_fn(self)

    return __del__


MjRenderContext.__del__ = _bind_before_free(MjRenderContext.__del__)


def _free_render_context(sim) -> None:
    """Free *sim*'s offscreen render context now rather than at collection."""
    ctx = getattr(sim, "_render_context_offscreen", None)
    if ctx is None:
        return
    sim._render_context_offscreen = None
    ctx.gl_ctx.make_current()
    ctx.con.free()
    ctx.gl_ctx.free()


def _make_render_context_current(sim) -> None:
    """Bind *sim*'s offscreen render context; another env may have bound its own."""
    ctx = getattr(sim, "_render_context_offscreen", None)
    if ctx is not None:
        ctx.gl_ctx.make_current()


class BaseTask(gymnasium.Env):
    """Every task in the benchmark implements exactly this interface."""

    metadata = {"render_modes": ["human", "rgb_array"]}
    OBS_MODES: tuple[str, ...] = ("proprio", "rgb", "combo")

    # --- subclasses declare these as class vars ---
    TASK_ID: str = ""
    DESCRIPTION: str = ""
    MAX_EPISODE_STEPS: int = 500
    CONFIG_FILE: str | None = None  # override to share one config across many tasks
    _closed: bool = False

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int = 0,
        obs_mode: str = "combo",
        config_overrides: dict | None = None,
        max_episode_steps: int | None = None,
        reward_fn=None,
        privileged_obs: bool = False,
    ) -> None:
        if obs_mode not in self.OBS_MODES:
            raise ValueError(f"obs_mode must be one of {self.OBS_MODES}, got {obs_mode!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be None or one of {self.metadata['render_modes']}, "
                f"got {render_mode!r}"
            )
        if render_mode == "human" and sys.platform.startswith("linux") and not (
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            # Without a display the MuJoCo viewer blocks at "Press Enter to exit" on the first step.
            raise RuntimeError(
                'render_mode="human" opens the MuJoCo viewer and needs a display, but neither '
                'DISPLAY nor WAYLAND_DISPLAY is set; use render_mode=None or "rgb_array" '
                "when headless"
            )
        self._reward_fn = reward_fn
        self.cfg = load_task_config(
            self.TASK_ID, config_file=self.CONFIG_FILE, overrides=config_overrides,
        )
        obs_cfg = self.cfg.get("obs", {})

        self.obs_mode = obs_mode
        self.privileged_obs = bool(privileged_obs)
        self.render_mode = render_mode
        self.viewer_camera = self.cfg.get("viewer_camera", None)

        # Free-camera perspective used by the MuJoCo viewer (render_mode="human").
        # Defaults give a natural diagonal view; override per task via YAML
        # viewer_perspective: {azimuth, elevation, distance, lookat}.
        _vp = self.cfg.get("viewer_perspective", {})
        self._viewer_cam_config = {
            "azimuth":   _vp.get("azimuth",   45.0),
            "elevation": _vp.get("elevation", -25.0),
            "distance":  _vp.get("distance",   1.5),
            "lookat":    _vp.get("lookat",    [0.0, 0.0, 0.85]),
        }

        self.reward_type = self.cfg.get("reward_type", "sparse")
        if self.reward_type not in ("sparse", "dense"):
            raise ValueError(
                f"reward_type must be 'sparse' or 'dense', got {self.reward_type!r}"
            )
        self.action_mode = self.cfg.get("action_mode", "cartesian_delta")
        if max_episode_steps is not None:
            self.MAX_EPISODE_STEPS = max_episode_steps
            self.cfg["max_episode_steps"] = max_episode_steps
        else:
            self.MAX_EPISODE_STEPS = self.cfg.get("max_episode_steps", self.MAX_EPISODE_STEPS)
        if not isinstance(self.MAX_EPISODE_STEPS, numbers.Integral) or self.MAX_EPISODE_STEPS <= 0:
            raise ValueError(
                f"max_episode_steps must be a positive int, got {self.MAX_EPISODE_STEPS!r}"
            )
        self.control_freq = self.cfg.get("control_freq", 20)
        # Multi-point convex contacts (``physics.multiccd``): without them a flat
        # cylinder or mesh face resting on a box gets one contact and rocks,
        # creeps or sinks into its support.
        self._multiccd = bool(self.cfg.get("physics", {}).get("multiccd", False))
        self.metadata = {**type(self).metadata, "render_fps": self.control_freq}
        self.seed = seed
        # Private stream for robosuite's global ``np.random`` draws (object
        # placement, robot init noise); see ``_robosuite_rng``.
        self._sim_rng_state = np.random.RandomState(seed).get_state()
        self._episode_started = False
        self._last_task_score = 0.0
        self._score_memo_key: object | None = None
        self._score_memo_val: float | None = None
        self._in_step_context = False
        self._step_count = 0
        self._viewer_maximized = False

        # ── Controller configuration ────────────────────────────────
        # Parsed from the YAML ``controller`` section and turned into the
        # ``controller_configs`` list that robosuite.make() expects.
        self.controller_configs = self._build_controller_configs()

        # Parse which modalities are enabled from config
        self.proprio_enabled = obs_cfg.get("proprio", {}).get("enabled", False)
        self.proprio_keys = obs_cfg.get("proprio", {}).get("keys", [])
        self.object_state_enabled = obs_cfg.get("object_state", {}).get("enabled", False)
        self.ft_enabled = obs_cfg.get("force_torque", {}).get("enabled", False)
        self.ft_keys = obs_cfg.get("force_torque", {}).get("keys", [])

        tactile_cfg = obs_cfg.get("tactile", {})
        self.tactile_enabled = tactile_cfg.get("enabled", False)
        self.tactile_fingers = tactile_cfg.get("fingers", ["right", "left"])
        self.tactile_rows = tactile_cfg.get("rows", 24)
        self.tactile_cols = tactile_cfg.get("cols", 16)
        self.tactile_spacing = tactile_cfg.get("spacing", 0.01)

        # Parse cameras — collect only the enabled ones
        # Cameras can be either:
        #   1. Built-in: just a robosuite_name reference
        #   2. Custom: defined with pos, quat, fovy — written onto the named robosuite camera after env build
        #      (pos/quat stay in that camera's own parent-body frame)
        self.camera_cfgs = {}  # label → {robosuite_name, width, height, depth, ...}
        self._custom_cameras = {}  # label → {pos, quat, fovy}
        cameras_cfg = obs_cfg.get("cameras", {})
        for label, cam in cameras_cfg.items():
            if cam.get("enabled", False):
                cam_name = cam.get("robosuite_name", label)
                self.camera_cfgs[label] = {
                    "robosuite_name": cam_name,
                    "width": cam.get("width", 84),
                    "height": cam.get("height", 84),
                    "depth": cam.get("depth", False),
                }
                # If pos is specified, it's a custom camera we need to inject
                if "pos" in cam:
                    self._custom_cameras[label] = {
                        "name": cam_name,
                        "pos": cam["pos"],
                        "quat": cam.get("quat", [1, 0, 0, 0]),
                        "fovy": cam.get("fovy", 60),
                    }

        self.has_cameras = len(self.camera_cfgs) > 0

        # Camera render throttle: MuJoCo offscreen rendering is the most
        # expensive part of the obs pipeline. Nothing downstream consumes
        # frames faster than ``recording.save_freq.cameras`` (dashboards
        # cap at 25 FPS too), so we only call ``sim.render`` every
        # ``_camera_render_skip`` simulation steps and reuse the cached
        # frame in between. At 100 Hz control × 25 Hz cameras this cuts
        # render calls 4x. Frames are rendered on the steps
        # ``_camera_clock_origin + k * _camera_render_skip`` (see
        # ``align_camera_frames``).
        cam_save_freq = self.cfg.get("recording", {}).get("save_freq", {}).get(
            "cameras", self.control_freq
        )
        cam_save_freq = max(1, int(cam_save_freq))
        self._camera_render_skip = max(1, self.control_freq // cam_save_freq)
        self._camera_cache: dict[str, np.ndarray] = {}
        self._camera_clock_origin = 0
        # Per-model geom-name lookups; cleared when robosuite rebuilds the model.
        self._finger_geom_cache: dict[tuple[str, ...], list[str]] = {}

        # ── Dataset save config ────────────────────────────────────────
        # Parse which modalities / extras should be written to the dataset.
        # Each obs modality has a ``save`` flag; extras live under
        # ``recording.save``.
        self._save_proprio = obs_cfg.get("proprio", {}).get("save", True)
        self._save_object_state = obs_cfg.get("object_state", {}).get("save", True)
        self._save_ft = obs_cfg.get("force_torque", {}).get("save", True)
        self._save_tactile = obs_cfg.get("tactile", {}).get("save", True)
        # Per-camera save flags
        self._save_cameras = {}
        for label, cam in cameras_cfg.items():
            if cam.get("enabled", False):
                self._save_cameras[label] = cam.get("save", True)

        rec_save = self.cfg.get("recording", {}).get("save", {})
        self._save_extras = {
            "action": rec_save.get("action", True),
            "reward": rec_save.get("reward", True),
            "task_score": rec_save.get("task_score", True),
            "joint_positions": rec_save.get("joint_positions", True),
            "ee_poses": rec_save.get("ee_poses", True),
        }

        self._observation_space = None
        self._action_space = None
        self._env = None
        self._rebuild_env()
        # Construction may draw from the task stream (an initial colour, a YCB
        # pick); restart it so the first unseeded reset() matches reset(seed=seed).
        if isinstance(getattr(self, "_rng", None), np.random.RandomState):
            self._rng = np.random.RandomState(seed)
        # Initialise gymnasium.Env internals (np_random, etc.)
        super().__init__()

        # Memoise _compute_task_score per step: _check_termination often
        # re-scores the same raw_obs, which must not advance per-step
        # counters (hold timers, streaks) a second time.
        impl = self._compute_task_score

        def _memoised(raw_obs, _impl=impl):
            if self._in_step_context and self._score_memo_key is raw_obs:
                return self._score_memo_val
            val = _impl(raw_obs)
            if self._in_step_context:
                self._score_memo_key = raw_obs
                self._score_memo_val = val
            return val

        self._compute_task_score = _memoised

    # ---- helpers for subclasses to query camera setup ----

    def _camera_names(self) -> list[str]:
        """Robosuite camera name list for enabled cameras."""
        return [c["robosuite_name"] for c in self.camera_cfgs.values()]

    def _camera_heights(self) -> list[int]:
        return [c["height"] for c in self.camera_cfgs.values()]

    def _camera_widths(self) -> list[int]:
        return [c["width"] for c in self.camera_cfgs.values()]

    # ---- controller config builder ----

    def _build_controller_configs(self) -> list[dict] | None:
        ctrl_cfg = self.cfg.get("controller")
        if ctrl_cfg is None:
            return None

        # Normalise to a list (one entry per robot)
        if isinstance(ctrl_cfg, dict):
            ctrl_cfgs = [ctrl_cfg]
        else:
            ctrl_cfgs = list(ctrl_cfg)

        # Load the canonical per-type part configs from robosuite
        _parts_dir = Path(robosuite.__file__).parent / "controllers" / "config" / "default" / "parts"

        result = []
        for cfg in ctrl_cfgs:
            robot_name = cfg.get("robot", "Panda")
            base = load_composite_controller_config(robot=robot_name)
            arm_key = cfg.get("arm_key", next(iter(base["body_parts"])))

            ctrl_type = cfg.get("type", base["body_parts"][arm_key].get("type", "OSC_POSE"))

            # If the requested type differs from the default, replace the arm
            # section with the canonical part config for that type so we don't
            # carry over incompatible keys (e.g. OSC keys when using IK_POSE).
            default_type = base["body_parts"][arm_key].get("type", "OSC_POSE")
            if ctrl_type != default_type:
                type_to_file = {
                    "OSC_POSE": "osc_pose.json",
                    "OSC_POSITION": "osc_position.json",
                    "IK_POSE": "ik_pose.json",
                    "JOINT_POSITION": "joint_position.json",
                    "JOINT_VELOCITY": "joint_velocity.json",
                    "JOINT_TORQUE": "joint_torque.json",
                }
                if ctrl_type not in type_to_file:
                    raise ValueError(
                        f"Unknown controller type {ctrl_type!r}; "
                        f"expected one of {sorted(type_to_file)}"
                    )
                part_file = _parts_dir / type_to_file[ctrl_type]
                with open(part_file) as f:
                    arm = json.load(f)
                # Preserve the gripper config from the default
                gripper_cfg = base["body_parts"][arm_key].get("gripper")
                if gripper_cfg:
                    arm["gripper"] = gripper_cfg
                base["body_parts"][arm_key] = arm
            else:
                arm = base["body_parts"][arm_key]

            # Apply any additional tuning parameters from YAML
            _tunable_keys = (
                "kp", "damping_ratio", "impedance_mode",
                "input_type", "input_ref_frame", "interpolation",
                "ramp_ratio", "kp_limits", "damping_ratio_limits",
                "output_max", "output_min",
                "ik_pos_limit", "ik_ori_limit", "control_delta",
                "uncouple_pos_ori", "position_limits",
                "orientation_limits",
            )
            for key in _tunable_keys:
                if key in cfg:
                    base["body_parts"][arm_key][key] = cfg[key]

            result.append(base)

        return result

    def _inject_custom_cameras(self) -> None:
        """Override camera pos/quat/fovy for cameras defined with custom placement in the YAML.

        The camera must already exist in the MuJoCo model (i.e. robosuite_name must
        reference a built-in camera). This method overrides its position, orientation,
        and field of view after the env is constructed.
        """
        sim = self._env.sim
        for label, cam_def in self._custom_cameras.items():
            cam_name = cam_def["name"]
            try:
                cam_id = sim.model.camera_name2id(cam_name)
            except (ValueError, KeyError):
                raise ValueError(
                    f"Custom camera '{label}': robosuite_name '{cam_name}' "
                    f"not found in model. Available: "
                    f"{[sim.model.camera_id2name(i) for i in range(sim.model.ncam)]}"
                )
            sim.model.cam_pos[cam_id] = cam_def["pos"]
            sim.model.cam_quat[cam_id] = cam_def["quat"]
            sim.model.cam_fovy[cam_id] = cam_def["fovy"]

    def _fix_suction_eye_in_hand(self) -> None:
        """Adjust eye_in_hand camera position for robots with suction grippers.

        The default eye_in_hand camera sits at pos="0.05 0 0" on the
        right_hand body.  The suction pump extends ~11 cm along the -Y axis
        of right_hand (due to the gripper's -90° X rotation), blocking the
        default camera's downward (+Z) view.  This shifts the camera to
        pos="0.1 -0.05 0" which clears the pump body from the field of view.
        """
        gripper_types = self.cfg.get("grippers", [])
        if isinstance(gripper_types, str):
            gripper_types = [gripper_types]

        suction_indices = [
            idx for idx, gtype in enumerate(gripper_types)
            if "Suction" in gtype
        ]
        if not suction_indices:
            return

        import re

        def _patch_suction_cam_xml(xml_string):
            for idx in suction_indices:
                cam_name = f"robot{idx}_eye_in_hand"
                # Match the camera element by name and replace its pos attribute
                pattern = (
                    r'(<camera[^>]*name\s*=\s*"'
                    + re.escape(cam_name)
                    + r'"[^>]*)\bpos\s*=\s*"[^"]*"'
                )
                replacement = r'\1pos="0.1 -0.05 0"'
                xml_string = re.sub(pattern, replacement, xml_string)
            return xml_string

        self._env.set_xml_processor(_patch_suction_cam_xml)
        # Recompile the model so the patched camera position takes effect.
        # Free the old sim's render context first (see _free_render_context),
        # then rebuild via _reset_internal so a new context binds to the new sim.
        _free_render_context(getattr(self._env, "sim", None))
        old_viewer = getattr(self._env, "viewer", None)
        if old_viewer is not None and self._env.renderer == "mujoco":
            try:
                self._env._destroy_viewer()
            except AttributeError:
                pass
        self._env._initialize_sim()
        self._env._reset_internal()

    def _apply_physics_options(self) -> None:
        """Apply the config's ``physics`` options; every model rebuild drops them."""
        if self._multiccd:
            self._env.sim.model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD

    def _rebuild_env(self) -> None:
        """Close the current env, if any, build a fresh one and apply the post-build patches.

        robosuite draws object placements and init noise from the global
        ``np.random`` while constructing the env; the caller's global state is
        restored afterwards (``reset()`` redraws the episode from the task's
        own stream).
        """
        self._check_open()
        if self._env is not None:
            self._close_env()
        caller_state = np.random.get_state()
        try:
            self._env = self._build_env()
            # Robots that actually received tactile sensors (set in _load_model).
            self._tactile_robot_indices = getattr(self._env, "_tactile_robot_indices", [])
            if self._custom_cameras:
                self._inject_custom_cameras()
            # Suction pump body blocks the default eye_in_hand placement.
            self._fix_suction_eye_in_hand()
            self._apply_physics_options()
        finally:
            np.random.set_state(caller_state)

    # ── Robosuite env factory (auto-wraps with tactile) ─────────────

    def _make_robosuite_env(self, env_name, **kwargs):
        """Create a robosuite env, wrapping with tactile if enabled.

        Subclasses should call this instead of ``suite.make()`` so that
        tactile injection happens automatically during ``_load_model``.

        Applies two render safeguards against intermittent black camera
        frames:

        1. Pins MuJoCo offscreen rendering to a specific GPU via
           ``render_gpu_device_id``. The id is read from
           ``MUJOCO_EGL_DEVICE_ID`` (or ``IB_RENDER_GPU_DEVICE_ID``)
           and defaults to ``0``. Set either env var to ``-1`` to
           restore the robosuite default (auto/CPU).
        2. Guards against running the GLFW onscreen viewer
           (``has_renderer=True``) and the offscreen renderer in the
           same process — that combination shares a context group
           between two GL paths and is the most common cause of the
           "sometimes the camera returns black" symptom. To run both
           anyway, export ``MUJOCO_GL=egl`` (the offscreen path then
           gets its own EGL context, isolated from GLFW) and set
           ``IB_ALLOW_VIEWER_WITH_OFFSCREEN=1`` to silence the warning.
        """
        env_cls = REGISTERED_ENVS[env_name]

        if self.tactile_enabled:
            env_cls = make_tactile_env_class(
                env_cls,
                tactile_rows=self.tactile_rows,
                tactile_cols=self.tactile_cols,
                tactile_spacing=self.tactile_spacing,
            )

        if self.render_mode == "rgb_array":
            kwargs["has_offscreen_renderer"] = True

        if "render_gpu_device_id" not in kwargs:
            gpu_id_str = (
                os.environ.get("IB_RENDER_GPU_DEVICE_ID")
                or os.environ.get("MUJOCO_EGL_DEVICE_ID")
                or "0"
            )
            try:
                kwargs["render_gpu_device_id"] = int(gpu_id_str)
            except ValueError:
                kwargs["render_gpu_device_id"] = 0

        if (
            kwargs.get("has_renderer")
            and kwargs.get("has_offscreen_renderer")
            and os.environ.get("MUJOCO_GL", "").lower() != "egl"
            and os.environ.get("IB_ALLOW_VIEWER_WITH_OFFSCREEN") != "1"
        ):
            warnings.warn(
                "Running has_renderer=True together with has_offscreen_renderer=True "
                "on the GLFW backend is a known cause of intermittent black camera "
                "frames. Either drop the viewer (render_mode=None), export "
                "MUJOCO_GL=egl so the offscreen path uses its own EGL context, or "
                "set IB_ALLOW_VIEWER_WITH_OFFSCREEN=1 to silence this warning.",
                RuntimeWarning,
                stacklevel=2,
            )

        return env_cls(**kwargs)

    # ── Tactile helpers ───────────────────────────────────────────────

    def get_tactile_readings(self, finger: str, robot_idx: int = 0) -> np.ndarray:
        """Return ``(rows, cols)`` touch array for *finger* on *robot_idx*.

        Returns a zero array when tactile is disabled or the robot has no
        finger gripper, so callers never need to branch.
        """
        if not self.tactile_enabled or robot_idx not in self._tactile_robot_indices:
            return np.zeros((self.tactile_rows, self.tactile_cols))
        return _tactile_readings(
            self._env.sim, robot_idx, finger,
            self.tactile_rows, self.tactile_cols,
        )

    def get_all_tactile_readings(self) -> dict[int, dict[str, np.ndarray]]:
        """Return ``{robot_idx: {finger: ndarray}}`` for every tactile-equipped robot."""
        result = {}
        for robot_idx in self._tactile_robot_indices:
            result[robot_idx] = {
                f: self.get_tactile_readings(f, robot_idx)
                for f in self.tactile_fingers
            }
        return result

    def _inject_tactile_obs(self, raw_obs: dict[str, np.ndarray]) -> None:
        """Write ``(rows, cols)`` tactile matrices into *raw_obs* so that
        ``_build_obs_from_config`` picks them up automatically.

        Keys written: ``robot{idx}_tactile_{finger}`` for each
        tactile-equipped robot and enabled finger.  Values are
        ``(rows, cols)`` numpy arrays.
        """
        if not self.tactile_enabled:
            return
        for robot_idx in self._tactile_robot_indices:
            for finger in self.tactile_fingers:
                key = f"robot{robot_idx}_tactile_{finger}"
                if key not in raw_obs:
                    raw_obs[key] = self.get_tactile_readings(finger, robot_idx)

    # ---- 4 methods every subclass must implement ----

    @abstractmethod
    def _build_env(self) -> object:
        """Return the raw robosuite env."""

    @abstractmethod
    def _get_obs(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert raw robosuite obs → standardised obs dict."""

    @abstractmethod
    def _compute_task_score(self, raw_obs: dict[str, np.ndarray]) -> float:
        """Return normalised 0–1 task completion score."""

    @abstractmethod
    def _check_termination(self, raw_obs: dict[str, np.ndarray]) -> bool:
        """Return True when episode should end (success or failure)."""

    # ---- shared obs builder (subclasses call this from _get_obs) ----

    def _build_obs_from_config(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Build the obs dict based on what the YAML config enables."""
        obs = {}

        # Proprio — store each key separately so downstream code can
        # access e.g. obs["robot0_eef_pos"] directly.
        if self.proprio_enabled:
            for key in self.proprio_keys:
                if key in raw_obs:
                    obs[key] = np.atleast_1d(raw_obs[key])

        # Cameras — render directly via sim.render() instead of using
        # raw_obs images.  Robosuite's observable sampling has a float-
        # precision bug that causes it to skip camera re-renders after
        # many steps, returning stale/frozen images.  Direct rendering
        # bypasses this entirely and also handles the OpenGL vertical
        # flip (origin at bottom-left → top-left).
        #
        # Throttling: render at ``recording.save_freq.cameras`` Hz rather
        # than the full control frequency — this is the slowest consumer
        # and anything faster is wasted work. Intermediate steps reuse
        # frames from ``self._camera_cache``. Always render when the
        # cache is empty (first call after reset, ``_rebuild_obs``).
        if self.obs_mode in ("rgb", "combo"):
            sim = getattr(self._env, "sim", None)
            render_now = (
                not self._camera_cache
                or (self._step_count - self._camera_clock_origin) % self._camera_render_skip == 0
            )
            if render_now:
                _make_render_context_current(sim)
            for label, cam_cfg in self.camera_cfgs.items():
                rs_name = cam_cfg["robosuite_name"]
                w, h = cam_cfg["width"], cam_cfg["height"]
                rgb_key = f"rgb_{label}"
                depth_key_out = f"depth_{label}"
                if render_now:
                    if sim is not None:
                        img = sim.render(width=w, height=h, camera_name=rs_name, depth=False)
                        self._camera_cache[rgb_key] = np.ascontiguousarray(img[::-1])
                    else:
                        img_key = f"{rs_name}_image"
                        if img_key in raw_obs:
                            self._camera_cache[rgb_key] = np.ascontiguousarray(raw_obs[img_key][::-1])
                    if cam_cfg["depth"]:
                        if sim is not None:
                            _, depth = sim.render(width=w, height=h, camera_name=rs_name, depth=True)
                            self._camera_cache[depth_key_out] = np.ascontiguousarray(depth[::-1])
                        else:
                            depth_key = f"{rs_name}_depth"
                            if depth_key in raw_obs:
                                self._camera_cache[depth_key_out] = np.ascontiguousarray(
                                    raw_obs[depth_key][::-1]
                                )
                # Copy so frames returned on different steps never alias.
                if rgb_key in self._camera_cache:
                    obs[rgb_key] = self._camera_cache[rgb_key].copy()
                if cam_cfg["depth"] and depth_key_out in self._camera_cache:
                    obs[depth_key_out] = self._camera_cache[depth_key_out].copy()

        # Object state
        if self.object_state_enabled and "object-state" in raw_obs:
            obs["object_state"] = raw_obs["object-state"]

        # Force/torque
        if self.ft_enabled:
            parts = []
            world_parts = []
            for key in self.ft_keys:
                if key in raw_obs:
                    parts.append(raw_obs[key])
                wkey = f"{key}_world_force"
                if wkey in raw_obs:
                    world_parts.append(raw_obs[wkey])
            if parts:
                obs["force_torque"] = np.concatenate(parts)
            if world_parts:
                # Shape (N_robots * 3,) — per-robot world-frame [Fx, Fy, Fz].
                obs["force_torque_world"] = np.concatenate(world_parts)
        
        if self.tactile_enabled:
            for robot_idx in self._tactile_robot_indices:
                for finger in self.tactile_fingers:
                    key = f"robot{robot_idx}_tactile_{finger}"
                    if key in raw_obs:
                        obs[key] = raw_obs[key]

        return obs

    # ---- dataset save filter ----

    def filter_obs_for_saving(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return a copy of *obs* containing only keys whose ``save``
        flag is ``true`` in the YAML config."""
        out = {}
        for key, val in obs.items():
            if key.startswith("rgb_") or key.startswith("depth_"):
                # Camera key — check per-camera save flag
                # key is e.g. "rgb_exo_front_cam" → label "exo_front_cam"
                label = key.split("_", 1)[1] if "_" in key else key
                if self._save_cameras.get(label, True):
                    out[key] = val
            elif key in ("force_torque", "force_torque_world"):
                if self._save_ft:
                    out[key] = val
            elif key in ("object_state", "object_pos", "object_quat",
                         "cube_pos", "cube_quat") or key.startswith("obj_"):
                if self._save_object_state:
                    out[key] = val
            elif "_tactile_" in key:
                if self._save_tactile:
                    out[key] = val
            elif key in ("joint_positions", "ee_poses"):
                if self._save_extras.get(key, True):
                    out[key] = val
            else:
                # Proprio keys (robot0_eef_pos, etc.)
                if self._save_proprio:
                    out[key] = val
        return out

    @property
    def save_extras(self) -> dict[str, bool]:
        """Return the ``recording.save`` config for non-obs fields."""
        return self._save_extras

    # ---- F/T sensor extraction ----

    def _inject_ft_sensor(self, raw_obs: dict[str, np.ndarray]) -> None:
        """Extract force/torque from the robot and inject into raw_obs.

        Robosuite's ``ee_force`` / ``ee_torque`` are read from a MuJoCo
        force sensor at the eef site, so the wrench is expressed in the
        **eef local frame**. For weight-sensing UIs we also rotate the
        force into the **world frame** so gravity always loads the same
        axis — the z-component of the world-frame force is then a direct
        proxy for held object weight regardless of gripper orientation.

        Stores a 6D vector (Fx, Fy, Fz, Tx, Ty, Tz) in the eef frame
        under the configured sensor key, plus a world-frame 3D force
        under ``robot{i}_ft_sensor_world_force`` for the dashboard.
        """
        if not self.ft_enabled:
            return
        for i, robot in enumerate(self._env.robots):
            key = f"robot{i}_ft_sensor"
            try:
                ee_force = robot.ee_force
                ee_torque = robot.ee_torque
                # ee_force/ee_torque are dicts keyed by arm name for multi-arm
                # robots, or plain arrays for single-arm — normalise to array
                if isinstance(ee_force, dict):
                    arm_name = next(iter(ee_force.keys()))
                    ee_force = ee_force[arm_name]
                    ee_torque = ee_torque[arm_name] if isinstance(ee_torque, dict) else ee_torque
                else:
                    arm_name = None
                force = np.array(ee_force, dtype=np.float64).flatten()
                torque = np.array(ee_torque, dtype=np.float64).flatten()
                raw_obs[key] = np.concatenate([force, torque])

                # World-frame force — rotate sensor-frame force by the
                # sensor site's world orientation. The force sensor sits
                # at site ``gripper*_ft_frame`` (dereferenced via the
                # gripper's ``important_sensors['force_ee']`` name).
                try:
                    sim = self._env.sim
                    grip = robot.gripper[arm_name] if arm_name is not None else robot.gripper
                    sens_name = grip.important_sensors["force_ee"]
                    sid = sim.model.sensor_name2id(sens_name)
                    site_id = sim.model.sensor_objid[sid]
                    site_name = sim.model.site_id2name(site_id)
                    R = sim.data.get_site_xmat(site_name)  # world<-sensor
                    raw_obs[f"{key}_world_force"] = R @ force
                except (KeyError, ValueError) as e:
                    logger.debug("World-frame F/T rotation skipped for robot%d: %s", i, e)
            except (AttributeError, RuntimeError) as e:
                logger.warning("Failed to read F/T sensor for robot%d: %s", i, e)

    # ---- base class handles everything else ----

    @contextmanager
    def _robosuite_rng(self):
        """Route robosuite's global ``np.random`` draws through this task's stream.

        robosuite samples placements and init noise from the global RNG. Swapping
        in a per-task state keeps episodes reproducible from ``seed`` without
        clobbering the caller's global RNG or coupling interleaved envs.
        """
        caller_state = np.random.get_state()
        np.random.set_state(self._sim_rng_state)
        try:
            yield
        finally:
            self._sim_rng_state = np.random.get_state()
            np.random.set_state(caller_state)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        """Start a new episode.

        ``seed`` reseeds every task RNG; ``None`` continues the current streams,
        which start from the constructor ``seed``.
        """
        self._check_open()
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            self._sim_rng_state = np.random.RandomState(seed).get_state()
        old_sim = self._env.sim
        with self._robosuite_rng():
            raw_obs = self._env.reset()
        if self._env.sim is not old_sim:
            # A hard reset builds a new sim and render context; free the old one now.
            _free_render_context(old_sim)
            _make_render_context_current(self._env.sim)
        self._apply_physics_options()
        self._finger_geom_cache.clear()
        self._step_count = 0
        self._episode_started = True
        self._last_task_score = 0.0
        # Force a fresh camera render on the first obs of the new episode.
        self._camera_cache.clear()
        # Re-inject custom cameras — robosuite's reset() rebuilds the MuJoCo
        # model, which overwrites any camera pos/quat/fovy we set earlier.
        if self._custom_cameras:
            self._inject_custom_cameras()
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._observe(raw_obs)
        info = {"task_score": 0.0, "is_success": False, "step": 0}
        return obs, info

    def _get_sim_observations(self, force_update: bool = False) -> dict[str, np.ndarray]:
        """Retrieve observations from the underlying robosuite environment.

        Wraps robosuite's internal ``_get_observations`` so all callers
        in this codebase go through a single, documented point of contact
        with the private robosuite API.  The ``force_update`` flag is
        required after scene mutations (object teleport, visual randomisation)
        to bypass robosuite's per-timestep observable cache.

        Parameters
        ----------
        force_update:
            When ``True``, bypasses robosuite's observable cache and
            forces a full re-evaluation of all sensors.
        """
        return self._env._get_observations(force_update=force_update)

    def _check_grasp(
        self,
        gripper,
        object_geoms: list[str] | str,
    ) -> bool:
        """Check whether *gripper* is currently grasping *object_geoms*.

        Wraps robosuite's internal ``_check_grasp`` so task subclasses
        do not reference robosuite private API directly. Tactile taxels count
        as part of the finger they are mounted on: they protrude past the
        fingerpad, so a real grasp often touches only taxels.

        Parameters
        ----------
        gripper:
            Robosuite gripper model (e.g. ``self._env.robots[0].gripper``),
            or a dict of them for multi-arm robots (any arm grasping counts).
        object_geoms:
            A single geom name or list of geom names belonging to the
            target object.

        Returns
        -------
        bool
            ``True`` if both fingers are in contact with the object.
        """
        if isinstance(gripper, dict):
            return any(self._check_grasp(g, object_geoms) for g in gripper.values())
        if self.tactile_enabled and isinstance(gripper, GripperModel):
            gripper = [
                self._finger_contact_geoms(gripper.important_geoms[side])
                for side in ("left_fingerpad", "right_fingerpad")
            ]
        return self._env._check_grasp(gripper=gripper, object_geoms=object_geoms)

    def _finger_contact_geoms(self, pad_geoms: list[str]) -> list[str]:
        """*pad_geoms* plus the tactile taxel geoms mounted on the same finger.

        The finger is the parent body of the pad geom's body; taxels hang off
        a sibling pad body, so any taxel in the finger's subtree belongs to it.
        """
        key = tuple(pad_geoms)
        if key not in self._finger_geom_cache:
            model = self._env.sim.model
            fingers = {
                int(model.body_parentid[model.geom_bodyid[model.geom_name2id(g)]])
                for g in pad_geoms
            }

            def on_finger(body: int) -> bool:
                while body > 0:
                    if body in fingers:
                        return True
                    body = int(model.body_parentid[body])
                return False

            taxels = [
                name for gid in range(model.ngeom)
                if "taxel_" in (name := model.geom_id2name(gid) or "")
                and on_finger(int(model.geom_bodyid[gid]))
            ]
            self._finger_geom_cache[key] = list(pad_geoms) + taxels
        return self._finger_geom_cache[key]

    def align_camera_frames(self, first_recorded_step: int) -> None:
        """Render camera frames on steps ``first_recorded_step + k * period``.

        *period* is ``control_freq / recording.save_freq.cameras``.  A recorder
        that keeps every period-th obs from step *first_recorded_step* on (e.g.
        after warm-up steps) then saves only freshly rendered frames.  Persists
        across resets; the default is 0.
        """
        self._camera_clock_origin = int(first_recorded_step)

    def _rebuild_obs(self) -> dict[str, np.ndarray]:
        """Re-render and rebuild the obs dict from current sim state.

        Subclass ``reset()`` methods call ``super().reset()`` first (which
        renders an obs), then mutate the scene — reposition objects,
        change masses, place static markers, etc. The obs returned by
        the super-call is therefore stale: the first saved demo frame
        shows the pre-mutation scene. Calling this helper at the end of
        a subclass reset returns a fresh obs that reflects all changes.

        Cameras honour the render-throttle cache; this call forces a
        fresh frame by clearing the cache first.
        """
        self._camera_cache.clear()
        raw_obs = self._get_sim_observations(force_update=True)
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        return self._observe(raw_obs)

    def _observe(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return the obs dict the caller sees for *raw_obs*.

        Every observation handed out by ``reset``/``step`` goes through here,
        so ``privileged_obs=False`` filters it in one place. Subclass resets
        that rebuild their obs must call this, not ``_get_obs``.
        """
        obs = self._get_obs(raw_obs)
        if self.privileged_obs:
            return obs
        return {k: v for k, v in obs.items() if _POLICY_OBS_KEY.fullmatch(k)}

    def _is_success(self, task_score: float) -> bool:
        """Whether the success condition holds after the current step.

        Uses the ``_success`` flag of a :class:`StagedReward` ``_stage_debug``
        when present, else ``task_score >= 1.0``. Override when neither fits.
        """
        stage_debug = getattr(self, "_stage_debug", None)
        if isinstance(stage_debug, dict) and "_success" in stage_debug:
            return bool(stage_debug["_success"])
        return task_score >= 1.0

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Advance one control step.

        ``action`` is clipped to ``action_space``. ``reward`` is 1.0 when the
        task's success condition holds and 0.0 otherwise (``reward_type:
        sparse``, the default), or the task score (``reward_type: dense``),
        unless ``reward_fn`` was given; robosuite's own reward is ignored.
        ``info`` carries ``task_score``, ``is_success`` and ``step``.
        """
        action = self._validate_action(action)
        raw_obs, _rs_reward, _rs_done, info = self._env.step(action)
        return self._finish_step(raw_obs, info)

    def _validate_action(self, action) -> np.ndarray:
        """Return *action* as a float64 array clipped to ``action_space``.

        Rejects a closed task, a wrong shape and non-finite values. ``step``
        overrides call this before any task-level noise, scaling or target
        integration, so every task bounds the policy's action alike.
        """
        self._check_open()
        action = np.array(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action shape {action.shape} does not match action_space "
                f"shape {self.action_space.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"action contains non-finite values: {action}")
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _check_open(self) -> None:
        """Raise once :meth:`close` has freed the simulator."""
        if self._closed:
            raise RuntimeError(
                f"{self.TASK_ID} is closed; create a new task with imbench.make()"
            )

    def _finish_step(
        self, raw_obs: dict[str, np.ndarray], info: dict
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Score the post-physics *raw_obs* and assemble the gymnasium step tuple.

        Overrides that call ``self._env.step`` themselves must finish through
        this method so scoring, ``is_success`` and the step counter stay uniform.
        """
        self._step_count += 1
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        obs = self._observe(raw_obs)

        self._score_memo_key = None
        self._score_memo_val = None
        self._in_step_context = True
        try:
            task_score = float(self._compute_task_score(raw_obs))
            terminated = bool(self._check_termination(raw_obs))
            is_success = bool(self._is_success(task_score))
        finally:
            self._in_step_context = False
        truncated = bool(self._step_count >= self.MAX_EPISODE_STEPS)
        self._last_task_score = task_score

        if self._reward_fn is not None:
            reward = float(self._reward_fn(self, raw_obs))
        elif self.reward_type == "sparse":
            reward = float(is_success)
        else:
            reward = task_score
        info.update({"task_score": task_score, "is_success": is_success, "step": self._step_count})
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        self._check_open()
        if self.render_mode == "human":
            self._env.render()
            if not self._viewer_maximized:
                self._try_maximize_mujoco_viewer()
                self._viewer_maximized = True
            return None
        if self.render_mode == "rgb_array":
            sim = getattr(self._env, "sim", None)
            if sim is None:
                return None
            # Use the first configured camera, fall back to agentview
            if self.camera_cfgs:
                first = next(iter(self.camera_cfgs.values()))
                cam_name = first["robosuite_name"]
                w, h = first["width"], first["height"]
            else:
                cam_name, w, h = "agentview", 256, 256
            _make_render_context_current(sim)
            frame = sim.render(width=w, height=h, camera_name=cam_name, depth=False)
            return np.ascontiguousarray(frame[::-1])  # flip vertical (MuJoCo origin is bottom-left)
        return None

    def _try_maximize_mujoco_viewer(self) -> None:
        import subprocess
        import time
        time.sleep(0.2)  # let the window manager register the new window
        target = getattr(self, "_viewer_target_geometry", None)
        try:
            if target:
                x, y, w, h = target
                subprocess.run(
                    ["wmctrl", "-r", "MuJoCo", "-e", f"0,{x},{y},{w},{h}"],
                    check=False, capture_output=True, timeout=2,
                )
            else:
                subprocess.run(
                    ["wmctrl", "-r", "MuJoCo", "-b", "add,maximized_vert,maximized_horz"],
                    check=False, capture_output=True, timeout=2,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def task_score(self, *, recompute: bool = False) -> float:
        """Task score in [0, 1] from the latest ``reset()``/``step()``.

        ``recompute=True`` re-scores the current sim state instead (e.g. after
        teleporting objects in a test); stateful scorers may then advance
        counters such as hold timers.
        """
        if not recompute:
            return self._last_task_score
        raw_obs = self._get_sim_observations(force_update=True)
        self._inject_ft_sensor(raw_obs)
        self._inject_tactile_obs(raw_obs)
        return float(self._compute_task_score(raw_obs))

    def close(self) -> None:
        """Free the simulator; ``step``, ``reset`` and ``render`` then raise."""
        self._closed = True
        self._close_env()

    def _close_env(self) -> None:
        """Close the robosuite env, freeing its render context while it is current."""
        _free_render_context(getattr(self._env, "sim", None))
        self._env.close()

    @property
    def observation_space(self) -> spaces.Dict:
        """Gymnasium observation space, built lazily on first access.

        Each key maps to a :class:`~gymnasium.spaces.Box` with the shape and
        dtype of the corresponding observation. ``uint8`` arrays are bounded
        ``[0, 255]``; floating arrays ``[-inf, inf]``. Building it never
        disturbs a running episode or consumes episode randomness.
        """
        if self._observation_space is None:
            obs = self._rebuild_obs() if self._episode_started else self._probe_obs()
            box_spaces = {}
            for k, v in obs.items():
                if v.dtype == np.uint8:
                    box_spaces[k] = spaces.Box(low=0, high=255, shape=v.shape, dtype=np.uint8)
                else:
                    dtype = v.dtype if np.issubdtype(v.dtype, np.floating) else np.float64
                    box_spaces[k] = spaces.Box(low=-np.inf, high=np.inf, shape=v.shape, dtype=dtype)
            self._observation_space = spaces.Dict(box_spaces)
        return self._observation_space

    def _probe_obs(self) -> dict[str, np.ndarray]:
        """Reset once to sample an observation, then restore all RNG state."""
        rng = getattr(self, "_rng", None)
        rng_state = rng.get_state() if isinstance(rng, np.random.RandomState) else None
        sim_rng_state = self._sim_rng_state
        obs, _ = self.reset()
        if rng_state is not None:
            self._rng.set_state(rng_state)
        self._sim_rng_state = sim_rng_state
        self._episode_started = False
        return obs

    @observation_space.setter
    def observation_space(self, space: spaces.Space) -> None:
        """Override the cached observation space (used in tests)."""
        self._observation_space = space

    @property
    def action_space(self) -> spaces.Box:
        """Gymnasium action space derived from the robosuite action spec.

        Returns a :class:`~gymnasium.spaces.Box` whose ``low`` and ``high``
        match the underlying robosuite environment's action bounds.
        """
        if self._action_space is None:
            low, high = self._env.action_spec
            self._action_space = spaces.Box(
                low=np.asarray(low, dtype=np.float32),
                high=np.asarray(high, dtype=np.float32),
                dtype=np.float32,
            )
        return self._action_space

    @action_space.setter
    def action_space(self, space: spaces.Space) -> None:
        """Override the cached action space (used in tests)."""
        self._action_space = space
