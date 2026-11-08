# Data Collection, Replay & Annotation

## Collecting Demonstrations

Collect teleoperation demonstrations and save them in LeRobot v2.1 format (Parquet + MP4).

Run `python scripts/setup_machine.py` once per machine first: it stores the 2-digit machine ID and creates the global run counter used for run IDs (see the Post-Processing Pipeline section of the README).

```bash
# Collect 10 episodes with gamepad control → demos/ib-p1-edge-slide-v1/M<id>/
python scripts/collect_demos.py \
    --task ib-p1-edge-slide-v1 \
    --teleop gamepad \
    --n 10 \
    --render

# Use keyboard with a custom output root → my_demos/ib-p1-bowl-table-v1/M<id>/
python scripts/collect_demos.py \
    --task ib-p1-bowl-table-v1 \
    --teleop keyboard \
    --n 5 \
    --render \
    --out-dir my_demos/ib-p1-bowl-table-v1

# Use a custom teleop config
python scripts/collect_demos.py \
    --task ib-p2-cube-toss-v1 \
    --teleop gamepad \
    --teleop-config my_gamepad.yaml \
    --n 5 --render

# Discovery helpers
python scripts/collect_demos.py --list-tasks
python scripts/collect_demos.py --list-devices
```

Episodes are written to `<out-dir>/M<machine_id>/`. Keep the default `--out-dir` (`demos/<task>`) or pass `<root>/<task>` so that `merge_annotated_runs.py --input <root>` finds the `<task>/M<id>/` folders.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | (required) | Task ID |
| `--teleop` | `keyboard` | Teleop device name (`keyboard`, `gamepad`, `spacemouse`, `quest`) |
| `--n` | `10` | Number of episodes to save |
| `--out-dir` | `demos/<task>` | Output root; episodes go to `<out-dir>/M<machine_id>/` |
| `--render` | off | Show MuJoCo viewer |
| `--no-dashboard` | off | Disable the obs dashboard (and the packing goal-layout window). Without a display the dashboard is skipped automatically; `--render` needs a display |
| `--teleop-config` | `configs/teleop/gamepad.yaml` in the repository | Teleop YAML: the action filter for every device, plus the gamepad mapping |
| `--list-tasks` | | Print available tasks and exit |
| `--list-devices` | | Print available devices and exit |

### Episode Lifecycle and Controls

- Each episode starts from `reset(seed=<seed>)`, with seeds from `machine_id × 1,000,000` (a resumed dataset continues after its largest recorded seed; a discarded attempt restarts with the same seed), followed by one unrecorded zero-action warm-up step. Both values are stored per episode (`seed`, `warmup_steps`) so the episode can be replayed exactly.
- An episode is **saved when the task terminates**; its `outcome` and `final_score` are taken at the terminating step. Ten zero-action cooldown steps are then recorded so the scene can settle.
- The operator, supervisor, robot and version fields of the episode metadata come from the session configured in the `scripts/dashboard.py` web dashboard (stored in `~/.imbench/session.json`) at the time the episode is saved.
- **Escape** (keyboard / Quest) or **B** (gamepad) **discards** the current episode and restarts the same scene; nothing is written and no run ID is consumed. Pressed during the cooldown, it only cuts the cooldown short and the episode is still saved. The SpaceMouse has no discard input.
- **Ctrl-C** (or SIGTERM / SIGHUP, e.g. closing the terminal, or **Q** in the dashboard window) stops collection. The episode in progress is discarded; every saved episode and `meta/info.json` are kept.

### Pause Detection

When the operator is idle (no buttons, sticks, or triggers active), the collection script skips `env.step()` and does not record the frame. This ensures the dataset only contains intentional actions -- no stretches of zero-action frames that would teach a policy to freeze. The exception is the gripper: after a gripper command changes, idle ticks keep stepping in place (zero motion, gripper held) for 0.5 s so the jaws finish moving, and these steps are recorded.

The viewer and dashboard continue updating during pauses so the operator sees a live scene. The loop is paced to `control_freq` in both active and idle states to keep the viewer responsive.

**Known limitation:** because MuJoCo is not stepped while paused (apart from the gripper settle), no physics runs during a pause. A held object stays put with a parallel-jaw gripper, but an object that gravity or external forces would move during the pause (e.g. a loosely balanced object) resumes from the state it had when the pause began.

### Action Smoothing

Before actions reach the simulator, a low-pass filter + delta clamp is applied to the 6 motion channels. The gripper channel passes through unfiltered for responsive grasping. The stepped action and the recorded action are the same filtered float32 values.

Configured under `action_filter` in the teleop YAML (`--teleop-config`, default `configs/teleop/gamepad.yaml` resolved from the repository, whatever the working directory):

```yaml
action_filter:
  alpha: 0.8        # EMA factor [0-1]. Higher = smoother. 0 = off.
  max_delta: 0.40   # Max per-step change per motion channel.
```

### Resume / Append

Re-running the script on the same output directory detects existing episodes and appends new ones. Episode indices, global frame counts and the `info.json` feature schema continue from where the previous run left off. Each episode's parquet is renamed into place before `meta/episodes.jsonl` and `meta/info.json` are updated, so even a hard-killed session leaves a dataset that reopens with every saved episode. Resuming with a different observation schema (obs mode or cameras) is refused, and only one collector can record into a dataset directory at a time.

### Dashboard Telemetry (opt-in)

Nothing is sent over the network by default. To report each saved episode (task, outcome, score, run ID, machine and operator ID) to your own collection dashboard, set `DASHBOARD_API_URL=https://...` or write `{"api_url": "https://..."}` to `~/.imbench/dashboard.json`; the collectors then POST to `<api_url>/report` in a background thread and ignore network errors.

### LeRobot v2.1 Dataset Layout

Each dataset is stored as a directory with this structure:

```
<out-dir>/M<machine_id>/
    meta/
        info.json                       # schema, per-feature fps, robot type
        episodes.jsonl                  # one JSON line per episode (incl. run_id, seed, warmup_steps)
        tasks.jsonl                     # task descriptions
        episode_000000_metadata.json    # per-episode metadata (run_id, outcome, operator, seed, ...)
    data/
        chunk-000/
            episode_000000.parquet      # scalar / vector streams
            episode_000001.parquet
            ...
    videos/
        chunk-000/
            rgb_<cam>/
                episode_000000.mp4
                ...
```

Each parquet row is one recorded control step. Per-stream save frequencies are configured in the task YAML config under `recording.save_freq`; streams saved below the control rate are null on the steps where they are not sampled. Camera streams are saved as MP4 videos at their own fps and referenced by `frame_index.rgb_<cam>` columns.

### Observation Streams

| Stream | Parquet columns | Rate category |
|--------|----------------|---------------|
| `robot<i>_eef_pos` | `observation.robot0_eef_pos.{0,1,2}` | `proprio` |
| `robot<i>_eef_quat` | `observation.robot0_eef_quat.{0..3}` | `proprio` |
| `robot<i>_gripper_qpos` | `observation.robot0_gripper_qpos.{0,1}` | `proprio` |
| `robot<i>_joint_pos` | `observation.robot0_joint_pos.{0..6}` | `proprio` |
| `robot<i>_joint_vel` | `observation.robot0_joint_vel.{0..6}` | `proprio` |
| `joint_positions` | `observation.joint_positions.{0..D}` | `proprio` |
| `ee_poses` | `observation.ee_poses.{0..D}` | `proprio` |
| `force_torque` | `observation.force_torque.{0..5}` | `force_torque` |
| `force_torque_world` | `observation.force_torque_world.{0..2}` | `force_torque` |
| `object_pos` | `observation.object_pos.{0..2}` | `object_state` |
| `object_quat` | `observation.object_quat.{0..3}` | `object_state` |
| `robot<i>_tactile_<finger>` | `observation.robot0_tactile_right.{0..63}` | `tactile` |
| `rgb_<cam>` | MP4 video + `frame_index.rgb_<cam>` | `cameras` |

## Episode Replay

Replay saved demonstrations in the simulator using the recorded actions (full physics with grasping) or joint positions (kinematics only, no grasping). The scene is reset with the episode's recorded `seed` and `warmup_steps`, so action replay reproduces the recording; episodes recorded without them fall back to `seed=0` with a warning.

```bash
# Replay all episodes of a collection folder
python scripts/replay_episode.py demos/ib-p1-edge-slide-v1/M01/

# Replay a specific episode (episode_index from meta/episodes.jsonl)
python scripts/replay_episode.py demos/ib-p1-edge-slide-v1/M01/ --episode 3

# Merged datasets use 8-digit MMCCCCCC indices
python scripts/replay_episode.py ~/processed_runs/ib-p1-edge-slide-v1/ --episode 1000042

# Slow-motion playback with the obs dashboard
python scripts/replay_episode.py demos/ib-p1-edge-slide-v1/M01/ --speed 0.5 --dashboard

# Headless: re-simulate and compare final scores with the recording
python scripts/replay_episode.py demos/ib-p1-edge-slide-v1/M01/ --no-render --speed 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `path` | (required) | LeRobot v2.1 dataset directory |
| `--episode` | all | `episode_index` to replay, as listed in `meta/episodes.jsonl` |
| `--speed` | `1.0` | Playback speed multiplier (`0` = as fast as possible) |
| `--dashboard` | off | Show obs dashboard alongside viewer |
| `--no-render` | off | Run without the MuJoCo viewer |

## Data Annotation

Review collected demonstrations and rate each episode from 1 (worst) to 5 (best) using an interactive dashboard (see [data_verification/rubric.md](../../data_verification/rubric.md)).

```bash
# Launch (auto-resumes from first un-rated episode)
python -m data_verification demos/ib-p1-edge-slide-v1/M01/

# Start at a specific episode with annotator tracking
python -m data_verification demos/ib-p1-edge-slide-v1/M01/ --episode 3 --annotator alice
```

### Dashboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play / Pause video |
| `+` / `-` | Increase / decrease playback speed (default 2x) |
| `LEFT` / `RIGHT` | Step one frame backward / forward |
| `UP` / `DOWN` | Seek +/- 10% of episode |
| `1`-`5` | Rate the episode (auto-advances to the next un-rated episode) |
| `U` | Undo / clear the rating for the current episode |
| `N` / `P` | Jump to next / previous episode |
| `R` | Jump to first un-rated episode |
| `S` | Print summary to terminal |
| `Q` / `ESC` | Quit |

### Annotation Storage

Ratings are saved to `annotations.json` alongside the dataset:

```json
{
  "dataset": "demos/ib-p1-edge-slide-v1/M01",
  "task_id": "ib-p1-edge-slide-v1",
  "total_episodes": 9,
  "annotations": {
    "0": {"rating": 5, "annotator": "alice", "timestamp": "2026-04-14T..."},
    "1": {"rating": 2, "annotator": "alice", "timestamp": "2026-04-14T..."},
    "2": null
  }
}
```

Key properties:
- **Merge-on-write** -- every rating re-reads the file under a file lock and changes only the rated episode, so several annotators can work on the same dataset without losing each other's ratings; a rating for an episode another annotator changed since it was loaded is not saved until the key is pressed again.
- **Atomic writes** -- uses `os.replace()` via a temp file to avoid corruption. An unreadable file is moved aside to `annotations.json.corrupt-<time>` with a warning.
- **Auto-resume** -- on launch, the dashboard jumps to the first un-rated episode.
- **Legacy entries** -- `{"status": "good"}` / `{"status": "bad"}` from older annotation files count as ratings 5 / 1.

### Dataset Format Conversion

Merged datasets are converted to the LeRobot v3.0 directory layout with `scripts/convert_v21_to_v30.py`. The parquet keeps IMBench's flattened per-dimension columns and multi-rate streams, so upstream `LeRobotDataset` cannot load it directly.

```bash
# Every dataset under ~/processed_runs → ~/processed_runs_v30/<task>/
python scripts/convert_v21_to_v30.py --demos-dir ~/processed_runs --out-dir ~/processed_runs_v30

# A single dataset → ~/processed_runs/ib-p1-edge-slide-v1_v30/
python scripts/convert_v21_to_v30.py --dataset ~/processed_runs/ib-p1-edge-slide-v1
```

See the Post-Processing Pipeline section of the README for the full merge → convert workflow.
