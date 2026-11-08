#!/usr/bin/env python3
"""Collect teleoperation demonstrations in LeRobot v2.1 format.

Usage examples::

    # Keyboard control, 10 episodes, with rendering + obs dashboard
    python scripts/collect_demos.py --task ib-p2-cube-toss-v1 --teleop keyboard --n 10 --render

    # SpaceMouse, 5 episodes, custom output directory
    python scripts/collect_demos.py --task ib-p2-cube-toss-v1 --teleop spacemouse --n 5 --render --out-dir my_demos

    # List available teleop devices
    python scripts/collect_demos.py --list-devices

Episodes are written to ``<out-dir>/M<machine_id>/`` (default out-dir:
``demos/<task_id>``) in LeRobot v2.1 multi-rate format (Parquet + MP4).
Per-stream save frequencies are configured in the task YAML config under
``recording.save_freq``.

An episode is saved when the task terminates; the teleop device's
end-episode input (Escape / gamepad B) discards it and restarts the scene.
Ctrl-C, SIGTERM, SIGHUP and Q in the dashboard stop collection and keep
every saved episode.  Machine M resets its episodes with seeds
``M * SEEDS_PER_MACHINE``, ``+1``, ... (a resumed dataset continues after
its largest seed), so machines never record the same scenes.  Without a
display the dashboard is skipped; ``--render`` needs one.
"""

import argparse
import contextlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

import imbench as ib
from imbench.tasks.base_task import load_task_config
from imbench.teleop import get_teleop_class, list_teleop_devices
from imbench.utils.dashboard_reporter import report_episode
from imbench.utils.lerobot_writer import DatasetLockedError, DatasetSchemaError
from imbench.utils.recording import (
    deferred_stop_signals,
    load_session,
    make_writer,
    require_machine_id,
    require_run_counter,
    stop_signals_raise_interrupt,
    write_episode_metadata,
)
from imbench.utils.robot_state import extract_robot_state
from imbench.utils.run_id import next_run_id
from imbench.utils.task_names import task_display_name

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_TELEOP_CONFIG = _REPO_ROOT / "configs" / "teleop" / "gamepad.yaml"

# Unrecorded zero-action steps taken after reset() so the viewer and
# dashboard show the scene before the operator acts.  Stored per episode
# so replay can reproduce the recorded trajectory.
WARMUP_STEPS = 1

# Zero-action steps recorded after termination so the episode includes
# the scene settling.
COOLDOWN_STEPS = 10

# Seed range per machine, as in data_synthetic/collection.py, so machines
# never record the same scenes.
SEEDS_PER_MACHINE = 1_000_000

# After a gripper command change, idle ticks are still stepped (zero motion,
# gripper held) for this long so the gripper finishes moving.
GRIPPER_SETTLE_S = 0.5


def _has_display() -> bool:
    """Whether windows can open; on Linux the viewer and cv2 windows need X (``DISPLAY``)."""
    return not sys.platform.startswith("linux") or bool(os.environ.get("DISPLAY"))


def _largest_recorded_seed(dataset: str) -> Optional[int]:
    """Largest ``seed`` in the dataset's per-episode metadata, or ``None``."""
    seeds = []
    for path in Path(dataset, "meta").glob("episode_*_metadata.json"):
        try:
            with open(path) as f:
                seed = json.load(f).get("seed")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(seed, int) and not isinstance(seed, bool):
            seeds.append(seed)
    return max(seeds) if seeds else None


def _num_arms(task_cfg: dict) -> int:
    ctrl = task_cfg.get("controller", {})
    return len(ctrl) if isinstance(ctrl, list) else 1


def _load_action_filter_config(teleop_config: Optional[str]) -> dict:
    """Return the ``action_filter`` section of the teleop YAML.

    Defaults to the repository's ``configs/teleop/gamepad.yaml`` regardless
    of the working directory.
    """
    path = Path(teleop_config) if teleop_config else _DEFAULT_TELEOP_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("action_filter") or {}


# ── Inline dashboard (throttled) ───────────────────────────────────
#
# cv2.imshow / cv2.waitKey stop refreshing reliably in child processes
# on virtual displays (:1).  Instead the dashboard runs inline in the
# main loop, throttled to the camera save rate so it doesn't bottleneck
# the 100 Hz control loop.

class _InlineDashboard:
    """Inline dashboard, throttled to max_fps.

    With *integrated* a single full-screen window is shown: MuJoCo viewer on
    top (70 %) and all camera feeds in a strip below (30 %).  Otherwise the
    cameras-only ObsDashboard is used.
    """

    def __init__(self, window_name, integrated=False, max_fps=25, visualize_ft=False):
        self._min_dt = 1.0 / max_fps
        self._last_update = 0.0

        if integrated:
            from imbench.utils.obs_visualizer import IntegratedDashboard
            self._dash = IntegratedDashboard(
                window_name=window_name, max_fps=max_fps, visualize_ft=visualize_ft,
            )
        else:
            from imbench.utils.obs_visualizer import ObsDashboard
            self._dash = ObsDashboard(window_name=window_name, visualize_ft=visualize_ft)

    def post(self, obs, info, reward) -> int:
        """Update the dashboard; return its key code, or -1 when throttled."""
        now = time.monotonic()
        if now - self._last_update < self._min_dt:
            return -1
        self._last_update = now
        return self._dash.update(obs, info, reward=reward) & 0xFF

    def reset(self):
        """Clear the per-episode histories (and the F/T tare)."""
        self._dash.reset()

    def close(self):
        self._dash.close()


# ── Action smoothing ──────────────────────────────────────────────

class ActionFilter:
    """Low-pass filter + delta clamp applied before env.step().

    Smooths the 6 motion channels (dx, dy, dz, droll, dpitch, dyaw)
    independently per arm.  Gripper channels (index 6, 13, 20, …) are
    passed through unfiltered so snap-toggle grasping stays responsive.

    Parameters
    ----------
    action_dim : int
        Total action dimensionality (7 * num_arms).
    alpha : float
        EMA smoothing factor in [0, 1].  Higher = smoother / more lag.
        0.0 = no filtering, 1.0 = output never changes.
    max_delta : float
        Maximum per-step change allowed on any motion channel.
    """

    def __init__(self, action_dim: int, alpha: float = 0.8, max_delta: float = 0.15):
        self._alpha = alpha
        self._max_delta = max_delta
        self._prev = np.zeros(action_dim, dtype=np.float64)
        self._action_dim = action_dim

    def reset(self):
        self._prev[:] = 0.0

    def filter(self, raw: np.ndarray) -> np.ndarray:
        out = np.empty_like(raw)
        for start in range(0, self._action_dim, 7):
            # Motion channels: low-pass then delta clamp
            mot = slice(start, start + 6)
            smoothed = self._alpha * self._prev[mot] + (1.0 - self._alpha) * raw[mot]
            delta = np.clip(smoothed - self._prev[mot],
                            -self._max_delta, self._max_delta)
            out[mot] = self._prev[mot] + delta

            # Gripper channel: pass through unfiltered
            g = start + 6
            out[g] = raw[g]

        self._prev[:] = out
        return out


# ── Solution preview (packing task only) ──────────────────────────


class _SolutionPreview:
    """Opens a small cv2 window showing the goal layout for the packing task.

    For ``ib-p7-packing-v1`` the operator must place 4 procedurally-
    generated L-pieces into the bin. Since the layout changes every seed,
    show the target tiling so the operator knows which slot is which.
    No-op for any other task, or when not *enabled* (no dashboard).
    """

    WINDOW_NAME = "Goal layout (packing)"

    def __init__(self, env, task_id: str, enabled: bool = True) -> None:
        self._enabled = enabled and task_id == "ib-p7-packing-v1" and hasattr(env, "solution_image")
        self._env = env
        self._opened = False
        if not self._enabled:
            return
        try:
            import cv2  # noqa: F401  — verify availability
        except Exception:
            self._enabled = False

    def refresh(self) -> None:
        if not self._enabled:
            return
        import cv2
        rgb = self._env.solution_image()
        bgr = rgb[..., ::-1].copy()
        if not self._opened:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, rgb.shape[1], rgb.shape[0])
            self._opened = True
        # Re-assert always-on-top each refresh — some window managers reset
        # the topmost flag when other windows take focus.
        try:
            cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.imshow(self.WINDOW_NAME, bgr)
        cv2.waitKey(1)

    def pump(self) -> None:
        """Pump cv2 events so the window stays responsive during teleop."""
        if not self._enabled or not self._opened:
            return
        import cv2
        cv2.waitKey(1)

    def close(self) -> None:
        if not self._enabled or not self._opened:
            return
        try:
            import cv2
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass


# ── Main collection loop ───────────────────────────────────────────

def _close_writer(writer) -> None:
    """Discard an unfinished episode and write the final info.json."""
    writer.abort_episode()
    writer.finalize()


def collect(task_id, teleop_name, n_demos, out_dir, render, use_dashboard, teleop_config=None):
    """Run the collection loop with multi-rate LeRobot v2.1 output."""
    if (render or use_dashboard) and not _has_display():
        if render:
            print("[collect] ERROR: --render needs a display, but DISPLAY is not set.")
            sys.exit(1)
        print("[collect] No display (DISPLAY is not set): running without the dashboard.")
        use_dashboard = False

    # Machine identity for global run IDs and per-machine output dirs
    machine_id = require_machine_id("collect")
    require_run_counter("collect")
    out_dir = os.path.join(out_dir, f"M{machine_id}")

    task_cfg = load_task_config(task_id)
    control_freq = task_cfg.get("control_freq", 20)

    filter_cfg = _load_action_filter_config(teleop_config)
    try:
        teleop_cls = get_teleop_class(teleop_name)
    except (ImportError, ValueError) as exc:  # missing device library / unknown name
        print(f"[collect] ERROR: {exc}")
        sys.exit(1)
    teleop_kwargs = {"num_arms": _num_arms(task_cfg)}
    if teleop_config and "config_path" in inspect.signature(teleop_cls).parameters:
        teleop_kwargs["config_path"] = teleop_config

    integrated = render and use_dashboard
    dt = 1.0 / control_freq

    with contextlib.ExitStack() as stack:
        stack.enter_context(stop_signals_raise_interrupt())

        # Lock the dataset before the simulator and windows open, so a second
        # collector on the same directory stops at once.
        try:
            writer = make_writer(out_dir, task_id, task_cfg)
        except (DatasetLockedError, DatasetSchemaError) as exc:
            print(f"[collect] ERROR: {exc}")
            sys.exit(1)
        stack.callback(_close_writer, writer)
        # Read after the writer has locked the dataset, so every seed saved by
        # a collector that held it until now is seen.
        first_seed = int(machine_id) * SEEDS_PER_MACHINE
        recorded = _largest_recorded_seed(out_dir)
        if recorded is not None and recorded >= first_seed:
            first_seed = recorded + 1

        env = ib.make(
            task_id,
            privileged_obs=True,
            obs_mode="combo",
            render_mode="human" if render else None,
            config_overrides={"max_episode_steps": 999_999},
        )
        stack.callback(env.close)

        if integrated:
            # Position the MuJoCo viewer in the top 70 % of the screen so the
            # camera-strip window can occupy the bottom 30 %.
            from imbench.utils.obs_visualizer import _screen_width, _screen_height
            scr_w, scr_h = _screen_width(), _screen_height()
            env._viewer_target_geometry = (0, 0, scr_w, int(scr_h * 0.70))

        try:
            teleop = teleop_cls(**teleop_kwargs)
            teleop.start()
        except (ValueError, RuntimeError) as exc:  # unsupported task, device not found
            print(f"[collect] ERROR: {exc}")
            sys.exit(1)
        stack.callback(teleop.stop)

        dashboard = None
        visualize_ft = bool(task_cfg.get("visualize_FT", False))
        if use_dashboard:
            dashboard = _InlineDashboard(
                window_name=f"Teleop: {task_id}", integrated=integrated, visualize_ft=visualize_ft,
            )
            stack.callback(dashboard.close)

        # Undo this episode's mirror only for the live operator feed.
        # Recorded observations retain the mirror.
        _unflip_for_teleop = task_id == "ib-m-mirror-pick-place-v1"

        # Packing task: show the goal layout in its own window so the operator
        # knows which slot each procedurally-generated L-piece fills.
        _solution_preview = _SolutionPreview(env, task_id, enabled=use_dashboard)
        stack.callback(_solution_preview.close)

        def _post_dashboard(obs_dict, info_dict, reward_val):
            if dashboard is None:
                return
            if _unflip_for_teleop:
                view = dict(obs_dict)
                # Every supported reflection is its own inverse.
                env._flip_cameras(view)
                obs_dict = view
            if dashboard.post(obs_dict, info_dict, reward_val) == ord("q"):
                print("\n[collect] Q pressed in the dashboard.")
                raise KeyboardInterrupt

        action_dim = env.action_space.shape[0]
        action_filter = ActionFilter(
            action_dim,
            alpha=filter_cfg.get("alpha", 0.8),
            max_delta=filter_cfg.get("max_delta", 0.15),
        )

        save_extras = env.save_extras
        start_ep = writer.episode_index  # resume from existing count
        settle_ticks = int(np.ceil(GRIPPER_SETTLE_S * control_freq))

        print(f"[collect] task={task_id}  device={teleop_name}  "
              f"control_freq={control_freq} Hz")
        print(f"[collect] save_freq: {writer.save_freq}")
        print(f"[collect] saving to {out_dir}/")
        if start_ep > 0:
            print(f"[collect] resuming from episode {start_ep}")
        print(f"[collect] seeds from {first_seed}")
        print("[collect] episodes save when the task terminates; "
              "Escape discards and restarts, Ctrl-C or Q stops\n")

        def _record_step(action, obs, step):
            """Step the env with *action* and record the pre-step observation."""
            obs_snapshot = {k: v.copy() for k, v in obs.items()}
            # Capture absolute robot state before stepping
            jpos, epose = extract_robot_state(env)
            obs_snapshot["joint_positions"] = jpos
            obs_snapshot["ee_poses"] = epose

            obs, reward, terminated, _truncated, info = env.step(action)
            # Record what ran: keyboard-typing replaces arm 0 with its scripted arm.
            action = np.asarray(info.get("applied_action", action), dtype=np.float32)

            writer.add_step(
                timestamp=step * dt,
                obs=env.filter_obs_for_saving(obs_snapshot),
                action=action if save_extras["action"] else None,
                reward=reward,
                task_score=info["task_score"],
                save_reward=save_extras["reward"],
                save_task_score=save_extras["task_score"],
            )
            return obs, reward, terminated, info

        try:
            ep_idx = start_ep
            while ep_idx < start_ep + n_demos:
                seed = first_seed + ep_idx - start_ep
                obs, info = env.reset(seed=seed)
                teleop.reset()
                action_filter.reset()
                if dashboard:
                    dashboard.reset()
                _solution_preview.refresh()

                env.align_camera_frames(WARMUP_STEPS)  # saved frames are rendered on their own step
                for _ in range(WARMUP_STEPS):
                    obs, _, _, _, info = env.step(np.zeros(action_dim))
                if render:
                    env.render()
                if dashboard:
                    _post_dashboard(obs, info, 0.0)

                writer.begin_episode()
                done = False
                step = 0
                # Gripper channels of the last stepped action; devices start open.
                gripper_cmd = np.full(action_dim // 7, -1.0, dtype=np.float32)
                settle = 0          # idle ticks still stepped while the gripper moves

                while not done:
                    t_loop = time.perf_counter()

                    teleop_result = teleop.get_action(obs)
                    action = teleop_result["action"]
                    step_no_action = teleop_result.get("step_no_action", False)

                    if teleop_result["paused"]:
                        if not settle:
                            # When the operator is idle, skip the physics step and
                            # recording so the dataset only contains intentional
                            # actions, and drop the filter state so motion after the
                            # pause starts from rest.  Pump the viewer event loop so
                            # the window doesn't freeze.
                            action_filter.reset()
                            if render and env._env.viewer is not None:
                                env._env.viewer.update()
                            if dashboard:
                                _post_dashboard(obs, info, 0.0)
                            done = teleop.should_end_episode()
                            time.sleep(dt)
                            continue
                        # The gripper is still moving to its last command (e.g.
                        # after a snap toggle): step in place until it settles.
                        action = np.zeros(action_dim)
                        action[6::7] = gripper_cmd
                        step_no_action = True

                    # Pad to env action_dim if teleop returns fewer dims (e.g. keyboard on bimanual task)
                    if len(action) < action_dim:
                        action = np.concatenate([action, np.zeros(action_dim - len(action))])
                    if step_no_action:
                        # Gamepad Y held: advance the sim with literal zero
                        # motion (gripper unchanged). Bypass the filter to
                        # avoid EMA leak and reset its state so the next
                        # real input starts from zeros.
                        action_filter.reset()
                        # Seed prev with the zero-motion action so the filter
                        # doesn't ramp up on the next tick.
                        action_filter.filter(action)
                    else:
                        action = action_filter.filter(action)
                    # Step with exactly the float32 values that are recorded.
                    action = np.asarray(action, dtype=np.float32)
                    if not np.array_equal(action[6::7], gripper_cmd):
                        settle = settle_ticks
                    elif settle:
                        settle -= 1
                    gripper_cmd = action[6::7].copy()

                    obs, reward, terminated, info = _record_step(action, obs, step)
                    step += 1

                    if render:
                        env.render()
                    if dashboard:
                        _post_dashboard(obs, info, reward)
                    _solution_preview.pump()

                    done = terminated or teleop.should_end_episode()

                    # Pace loop to control frequency
                    remaining = dt - (time.perf_counter() - t_loop)
                    if remaining > 0:
                        time.sleep(remaining)

                # Escape: discard the episode and restart the same scene
                # without consuming an episode index, seed or run ID.
                if teleop.should_end_episode():
                    writer.abort_episode()
                    print(f"  ep {ep_idx:04d}  discarded — restarting scene")
                    continue

                # Outcome at termination: is_success is evaluated per step and
                # may lapse while the scene settles.
                score = info["task_score"]
                success = bool(info["is_success"])

                # Post-termination cooldown, recorded like the active steps.
                # The operator can cut it short with the end-episode input.
                zero_action = np.zeros(action_dim, dtype=np.float32)
                for _ in range(COOLDOWN_STEPS):
                    t_loop = time.perf_counter()
                    obs, reward, _term, info = _record_step(zero_action, obs, step)
                    step += 1

                    if render:
                        env.render()
                    if dashboard:
                        _post_dashboard(obs, info, reward)

                    # Keep polling so a device that reads its end-episode input in
                    # get_action() (rather than a listener thread) sees it here too.
                    teleop.get_action(obs)
                    if teleop.should_end_episode():
                        break

                    remaining = dt - (time.perf_counter() - t_loop)
                    if remaining > 0:
                        time.sleep(remaining)

                with deferred_stop_signals():
                    run_id = next_run_id(machine_id)
                    session = load_session()
                    writer.end_episode(metadata={
                        "final_score": score,
                        "teleop_device": teleop_name,
                        "run_id": run_id,
                        "seed": seed,
                        "warmup_steps": WARMUP_STEPS,
                    })
                    write_episode_metadata(
                        out_dir, ep_idx,
                        task_id=task_id,
                        task_cfg=task_cfg,
                        display_name=task_display_name(task_id),
                        run_id=run_id,
                        n_steps=step,
                        action_dim=action_dim,
                        final_score=score,
                        success=success,
                        seed=seed,
                        warmup_steps=WARMUP_STEPS,
                        teleop_device=teleop_name,
                        session=session,
                    )
                report_episode(
                    task_name=task_display_name(task_id),
                    outcome="success" if success else "failure",
                    score=score,
                    episode_id=run_id,
                    machine_id=machine_id,
                    operator_id=session.get("teleoperator_id") or None,
                )
                print(f"  ep {ep_idx:04d}  run_id={run_id}  seed={seed}  score={score:.3f}  steps={step}")
                ep_idx += 1

        except KeyboardInterrupt:
            print("\n[collect] interrupted — saved episodes are intact.")
        except DatasetSchemaError as exc:
            print(f"\n[collect] ERROR: {exc}")
            sys.exit(1)

    total = writer.episode_index
    print(f"[collect] done. {total - start_ep} new + {start_ep} existing = "
          f"{total} total episodes in {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Collect teleoperation demonstrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        type=str,
        help=f"Task ID. Available: {ib.list_tasks()}",
    )
    parser.add_argument(
        "--teleop",
        type=str,
        default="keyboard",
        help=f"Teleop device name. Available: {list_teleop_devices()} (default: keyboard)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of episodes to collect (default: 10)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output root; episodes go to <out-dir>/M<machine_id>/ (default: demos/<task_id>)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the robosuite viewer",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable the real-time obs dashboard (and the packing goal-layout window)",
    )
    parser.add_argument(
        "--teleop-config",
        type=str,
        default=None,
        help="Path to teleop YAML config: action filter for every device, plus the "
             "gamepad mapping (default: the repository's configs/teleop/gamepad.yaml)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available teleop devices and exit",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print available tasks and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        print("Available teleop devices:", list_teleop_devices())
        sys.exit(0)

    if args.list_tasks:
        print("Available tasks:", ib.list_tasks())
        sys.exit(0)

    if not args.task:
        parser.error("--task is required (use --list-tasks to see options)")

    out_dir = args.out_dir or f"demos/{args.task}"

    collect(
        task_id=args.task,
        teleop_name=args.teleop,
        n_demos=args.n,
        out_dir=out_dir,
        render=args.render,
        use_dashboard=not args.no_dashboard,
        teleop_config=args.teleop_config,
    )


if __name__ == "__main__":
    main()
