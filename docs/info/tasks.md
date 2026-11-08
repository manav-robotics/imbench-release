# Tasks

## Overview

All tasks share a standardised interface defined in `BaseTask` ([imbench/tasks/base_task.py](../../imbench/tasks/base_task.py)). Each task is registered by ID and instantiated via `ib.make()`.

Task IDs follow `ib-<category>-<name>-v<version>`. The category prefix encodes the **physics phenomenon** being tested (documented in [imbench/registry.py](../../imbench/registry.py)), not the source directory a task's Python file happens to live in — several categories span multiple directories (e.g. P1 tasks live under `prehensile/`, `hetbi/`, and `hobi/`; P6 spans `prehensile/` and `puzzle/`; P7 spans `hard/`, `prehensile/` and `puzzle/`).

| Prefix | Phenomenon | Paper count | Registered (incl. `v2` OOD + non-paper) |
|--------|------------|:-----------:|:-----------------------------------------:|
| `ib-p1-*` | P1 — Geometry-constrained grasping | 9 | 12 |
| `ib-p2-*` | P2 — Dynamics and trajectory prediction | 7 | 9 |
| `ib-p3-*` | P3 — Causal chains (indirect action) | 6 | 8 |
| `ib-p4-*` | P4 — Tool use (augmented reach) | 2 | 3 |
| `ib-p5-*` | P5 — Hidden state (active discovery) | 2 | 3 |
| `ib-p6-*` | P6 — Reactive replanning under perturbation | 3 | 4 |
| `ib-p7-*` | P7 — Stability and equilibrium construction | 4 | 6 |
| `ib-m-*` | M — Miscellaneous / utility | 2 | 5 |
| **Total** | | **35** | **50** |

A task's arm configuration (single arm / homogeneous bimanual "HoBi" / heterogeneous bimanual "HetBi") is independent of its phenomenon category and is implied by which directory it lives in (`prehensile/`, `puzzle/`, `hard/` = single arm; `hobi/` = two identical arms; `hetbi/` = two arms with different grippers, finger + suction).

## Available Tasks

Each table below lists every registered task in that phenomenon category. `v2` rows are out-of-distribution (OOD) variants of the `v1` task directly above them, sharing the same scoring logic with a different physical/visual condition. Two registered tasks are **not** part of the paper's 35-task benchmark (kept as extra examples/pilots, not officially counted) — marked `†`.

### P1 — Geometry-constrained grasping

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-p1-edge-slide-v1` | single | Slide a flat plate to the table edge to create a graspable overhang, then grasp and lift it |
| `ib-p1-edge-slide-v2` | single | OOD: same task with a cylinder instead of a plate |
| `ib-p1-recover-peg-insert-v1` | single | A rod stands upright beside a through-slot fixture on the robot's left; tip it onto its side, align, and push it through. Once grasped and aligned, the rod is yawed ±10° once and must be realigned |
| `ib-p1-recover-peg-insert-v2` | single | OOD: the rod starts already lying on its side (skips the tip-over step) |
| `ib-p1-sheltered-grasp-v1` | single | Reach into the correct compartment (top/bottom, randomised) of a covered two-story rack, extract a cube, place it in a tray. Each episode samples a left/right reflection of the complete rack/cube/tray layout. |
| `ib-p1-sheltered-grasp-v2` | single | OOD: the rack stays on the robot's right at a fixed pose (v1 samples both sides and rack position) |
| `ib-p1-cup-inversion-v1` | hetbi | Suction grasps an inverted cup by its base, lifts it, hands off to the jaw arm, which sets it upright inside a green goal region on the table |
| `ib-p1-plate-shelf-hobi-v1` | hobi | Two identical jaw grippers cooperate to press-tilt a plate onto its edge and insert it into a shelf slot |
| `ib-p1-plate-shelf-hetbi-v1` | hetbi | Suction lifts a flat plate off the table and hands it to the jaw arm, which inserts it on edge into the shelf slot |
| `ib-p1-bowl-table-v1` | hetbi | Suction extracts an inverted bowl from a rack compartment; jaw arm receives the handoff and sets it upright on the table |
| `ib-p1-cracker-box-suction-v1` | hetbi | Cracker box into a slot fixture with the suction arm alone — the jaw arm is frozen |
| `ib-p1-cracker-box-jaw-v1` | hetbi | Cracker box into a slot fixture with the parallel jaw alone — the suction arm is frozen |

### P2 — Dynamics and trajectory prediction

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-p2-cube-toss-v1` | single | Pick up a cube and toss it so it comes to rest inside a goal circle; the cube must leave the gripper at least 15 cm outside the circle |
| `ib-p2-pendulum-grasp-v1` | single | A sphere swings on an undamped pendulum; time the grasp to catch it within 1 cm of the swing apex |
| `ib-p2-pendulum-grasp-v2` | single | OOD: the horizontal hinge axis is turned 45° about the vertical, so the sphere swings in a diagonal plane |
| `ib-p2-linear-intercept-v1` | single | Intercept and grasp a ball moving in a straight line across the table before it exits the workspace, then hold it 10 cm above the table for 30 steps |
| `ib-p2-bounce-intercept-v1` | single | Predict a ball's post-bounce trajectory off table-top walls, then grasp it and hold it 10 cm above the table for 30 steps; touching it before its first wall bounce fails the episode |
| `ib-p2-galileo-ramp-drop-v1` | single | Release a ball at the correct point on a symmetric ramp so its first goal-side turnaround lands on a 1 cm goal strip |
| `ib-p2-galileo-ramp-drop-v2` | single | OOD: the ramp is turned 90° so the ball rolls toward and away from the robot instead of left-right |
| `ib-p2-slide-catch-v1` | single | An incline tilts up (random start, rate and one pause or speed-up); move the bin held in the gripper to its base to catch 2-3 cubes of random friction and size as they slide off. The incline's pose and colour and the arm's start vary per seed; every seed is checked to be catchable at gamepad teleop speed |
| `ib-p2-ramp-sort-v1` | single | Move 2–5 blue and red balls from a spawn box onto a ramp; each rolls down the lane it is released in, and is sorted once it is at rest inside the bin of its colour, below the rim (on the floor or on other balls). Ball order and spacing, bin sides and the ramp's pose (±15°, ±3 cm) vary per seed |

### P3 — Causal chains (indirect action)

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-p3-domino-single-v1` | single | Topple the first domino in a chain to trigger a cascade that displaces the red target at its end |
| `ib-p3-domino-single-v2` | single | OOD: three decoy dominoes are scattered on the table away from the chain |
| `ib-p3-domino-select-v1` | single | Each seed draws one of 27 layouts (zig-zags, loops, spirals, junctions, crossings, ladders, shadow routes, near-misses) with up to 48 dominoes, every piece within 0.72 m of the robot base. Exactly one piece topples red: look-alike chains stop at hidden breaks (gaps, missing, low or twisted pieces, sharp corners, sideways jogs), meet other chains side-on, or end beside or short of red, and no decoy piece pushed either way can reach red or drive the route. |
| `ib-p3-domino-select-v2` | single | OOD: the fixed cross-chain merge layout (an indirect feeder crosses a bent connection into another chain after its gap), rotated 90° |
| `ib-p3-seesaw-balance-v1` | single | Pick the counterweight and place it at the right spot on the seesaw's far side so the plank balances within the tip-angle threshold, then hold it there |
| `ib-p3-gap-funnel-v1` | single | Push a sphere through the single gap in a barrier strip into a goal zone on the far side |
| `ib-p3-indirect-push-v1` | single | Arrange scattered blocks into a contiguous chain, then push the chain so force propagates to drive a distant target into the goal zone |
| `ib-p3-ball-cage-escape-v1` | single | A ball is trapped inside a cuboid whose floor has a hole; tilt or lift the cuboid to drop the ball out |

### P4 — Tool use (augmented reach)

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-p4-tool-retrieve-v1` | single | Grasp a hook tool, drag an out-of-reach cube into the workspace, release the tool, then grasp and lift the cube |
| `ib-p4-tool-retrieve-v2` | single | OOD: hook tip mirror-flipped onto the opposite side of the shaft, requiring a different approach/drag strategy |
| `ib-p4-cup-extract-v1` | single | Use a handled cup's handle to extract a trapped cuboid, then place it on a goal region |

### P5 — Hidden state (active discovery)

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-p5-mass-sort-v1` | single | Three same-size cubes have hidden masses that their colours do not reveal; discover them via lift force and sort into mass-ordered zones |
| `ib-p5-occluder-push-v1` | single | A quarter of seeds: the original three flat plates in a row, each on a cube. The rest: 3-4 covers of at least two kinds (flat plates, upside-down cups lifted by a knob, hinged flaps, spring-loaded flaps that look the same, boxes with a lid that slides one way) hide the red cube, coloured distractors or nothing; uncover the red cube and place it on the green patch. Layout, cover kinds and headings vary per seed; knocking anything off the table fails, a distractor left on the patch blocks success |
| `ib-p5-occluder-push-v2` | single | OOD: 2-3 stacked lid layers instead of one |

### P6 — Reactive replanning under perturbation

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-p6-slip-recovery-v1` | single | Pick a YCB object; the gripper is forced open once mid-transport, forcing a re-grasp before placing it in the goal region |
| `ib-p6-slip-recovery-v2` | single | OOD: exactly two scripted slips per episode instead of one |
| `ib-p6-pick-place-ycb-v1` | single | Place a YCB object on a goal patch that relocates 1-3 times mid-episode as the object approaches, forcing replans |
| `ib-p6-stack-collapse-recovery-v1` | single | A pre-stacked tower of three cubes always collapses early in the episode; rebuild and hold it stable |

### P7 — Stability and equilibrium construction

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-p7-balance-medium-v1` | single | Balance an elongated rod vertically on a narrow perpendicular ridge for a sustained duration; tactile feedback drives micro-corrections |
| `ib-p7-balance-medium-v2` | single | OOD: platform ridge and rod are oriented at a freshly randomised angle each episode |
| `ib-p7-balance-hard-v1` | single | Same as balance-medium on a thinner, shorter, less forgiving ridge |
| `ib-p7-balance-hard-v2` | single | OOD variant of balance-hard (randomised orientation, as with balance-medium-v2) |
| `ib-p7-shape-stack-v1` | single | Stack 3 or 4 randomly chosen mixed-shape objects so only one remains touching the table, held stable |
| `ib-p7-packing-v1` | single (suction) | Pack 3-4 procedurally-generated L-shaped pieces (from a guillotine-split decomposition) flat into a snug bin (about 4 mm of slack around each piece), in a single layer, each within 5 mm and 5° of its own target slot |

### M — Miscellaneous / utility

| Task ID | Arm | Description |
|---------|-----|-------------|
| `ib-m-mirror-pick-place-v1` | single | Pick up a cube and lift it to a target height, with a seeded horizontal, vertical or combined flip of every camera, fixed per episode |
| `ib-m-keyboard-typing-v1` | hobi | Two arms type a word, drawn per seed from a 25-word list, on a virtual QWERTY keyboard in word order, each pressing its own letters on its turn (scripted arm 0, teleoperated arm 1); a wrong key from arm 1 fails the episode, and the keyboard position is jittered per seed |
| `ib-m-keyboard-typing-v2` | hobi | OOD: word is drawn per seed from a held-out 5-word list (none in v1's list) |
| `ib-m-dining-table-v1` † | hobi | Two arms (one scripted, one teleoperated) place YCB dining objects onto colour-coded goal zones |
| `ib-m-freeform-v1` † | configurable | Configurable sandbox task — robots placed by base position, box/cylinder/sphere objects, and weighted multi-goal distance shaping |

† Not part of the paper's 35-task benchmark suite — registered as extra examples, not officially counted. See [Paper Task Mapping](#paper-task-mapping) below for the 35 that are.

## Paper Task Mapping

The paper numbers each of its 35 tasks `T01`-`T33` (P1-P7) and `M01`-`M02` (misc). This is the canonical mapping to registry task IDs:

| Paper # | Name | Task ID |
|---|---|---|
| T01 | Edge Slide | `ib-p1-edge-slide-v1` |
| T02 | Recover Peg Insert | `ib-p1-recover-peg-insert-v1` |
| T03 | Sheltered Grasp | `ib-p1-sheltered-grasp-v1` |
| T04 | Cup Inversion | `ib-p1-cup-inversion-v1` |
| T05 | Plate-Shelf (HoBi) | `ib-p1-plate-shelf-hobi-v1` |
| T06 | Plate-Shelf (HetBi) | `ib-p1-plate-shelf-hetbi-v1` |
| T07 | Bowl to Table | `ib-p1-bowl-table-v1` |
| T08 | Cracker Box (Suction) | `ib-p1-cracker-box-suction-v1` |
| T09 | Cracker Box (Jaw) | `ib-p1-cracker-box-jaw-v1` |
| T10 | Cube Toss | `ib-p2-cube-toss-v1` |
| T11 | Pendulum Grasp | `ib-p2-pendulum-grasp-v1` |
| T12 | Linear Intercept | `ib-p2-linear-intercept-v1` |
| T13 | Bounce Intercept | `ib-p2-bounce-intercept-v1` |
| T14 | Galileo Ramp Drop | `ib-p2-galileo-ramp-drop-v1` |
| T15 | Slide Catch | `ib-p2-slide-catch-v1` |
| T16 | Ramp Sort | `ib-p2-ramp-sort-v1` |
| T17 | Domino Single | `ib-p3-domino-single-v1` |
| T18 | Domino Select | `ib-p3-domino-select-v1` |
| T19 | Seesaw Balance | `ib-p3-seesaw-balance-v1` |
| T20 | Indirect Push | `ib-p3-indirect-push-v1` |
| T21 | Gap Funnel | `ib-p3-gap-funnel-v1` |
| T22 | Ball Cage Escape | `ib-p3-ball-cage-escape-v1` |
| T23 | Tool Retrieve | `ib-p4-tool-retrieve-v1` |
| T24 | Cup Extract | `ib-p4-cup-extract-v1` |
| T25 | Mass Sort | `ib-p5-mass-sort-v1` |
| T26 | Occluder Push | `ib-p5-occluder-push-v1` |
| T27 | Slip Recovery | `ib-p6-slip-recovery-v1` |
| T28 | Stack Collapse Recovery | `ib-p6-stack-collapse-recovery-v1` |
| T29 | Pick-Place YCB | `ib-p6-pick-place-ycb-v1` |
| T30 | Shape Stack | `ib-p7-shape-stack-v1` |
| T31 | Balance (Medium) | `ib-p7-balance-medium-v1` |
| T32 | Balance (Hard) | `ib-p7-balance-hard-v1` |
| T33 | Packing | `ib-p7-packing-v1` |
| M01 | Mirror Pick-Place | `ib-m-mirror-pick-place-v1` |
| M02 | Keyboard Typing | `ib-m-keyboard-typing-v1` |

The 13 `-v2` tasks are each an OOD variant of one paper task above (same paper number's task, not separately numbered), and `ib-m-dining-table-v1` / `ib-m-freeform-v1` are the 2 non-paper extras noted above.

## Creating a Task

```python
env = ib.make(
    "ib-p1-bowl-table-v1",
    render_mode="human",          # "human" for viewer, None for headless
    obs_mode="combo",             # proprio + cameras (project default)
    privileged_obs=False,         # default; True adds sim ground truth (object poses) for scripted oracles
    config_overrides={            # override any YAML config key
        "max_episode_steps": 2000,
    },
)
```

`max_episode_steps` must be a positive int and `config_overrides` a dict. After `env.close()` the task cannot be reset, stepped or rendered again.

Every task is also registered with Gymnasium on `import imbench`, e.g. `gymnasium.make("ib-p1-bowl-table-v1", obs_mode="proprio")`. The 13 `-v2` OOD variants are registered as `imbench-ood/<task-id>`, e.g. `gymnasium.make("imbench-ood/ib-p1-edge-slide-v2")`: Gymnasium reads `-vN` as a version and would otherwise tell `-v1` users to upgrade to the OOD variant. `ib.make()` takes the plain ids.

## Action Space

All tasks use a Cartesian delta action per arm:

```
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

- Actions are clipped to `action_space` (`[-1, 1]` per dimension) before any task-level scaling, so a value beyond the bound acts as the bound
- Gripper: `[-1, 1]`, sign-based. Parallel-jaw grippers move toward open (`< 0`) or closed (`> 0`) at a fixed rate per step regardless of magnitude; `0` holds the current width. Suction grippers engage on any positive value, release on any negative value, and keep their state at exactly `0`
- Two-arm tasks concatenate both arms (7 dims per arm): `[arm0 (7), arm1 (7)]`
- Actions with the wrong shape or non-finite values raise `ValueError`

## Observation Space

Observations are returned as a dict. Which keys are present depends on the task YAML config:

| Key | Shape | Description |
|-----|-------|-------------|
| `robot<i>_eef_pos` | `(3,)` | End-effector position |
| `robot<i>_eef_quat` | `(4,)` | End-effector orientation (quaternion) |
| `robot<i>_gripper_qpos` | `(2,)`, `(1,)` for suction | Gripper joint positions |
| `robot<i>_joint_pos` | `(7,)` | Arm joint positions |
| `robot<i>_joint_vel` | `(7,)` | Arm joint velocities |
| `rgb_<cam>` | `(H, W, 3)` | RGB image from a named camera |
| `depth_<cam>` | `(H, W)` | Depth image (if enabled) |
| `object_pos` | `(3,)` | Ground-truth object position (requires `obs.object_state.enabled: true`) |
| `object_quat` | `(4,)` | Ground-truth object orientation (requires `obs.object_state.enabled: true`) |
| `object_state` | `(D,)` | Raw robosuite object state vector (requires `obs.object_state.enabled: true`) |
| `force_torque` | `(6,)` per arm | End-effector `[Fx, Fy, Fz, Tx, Ty, Tz]` in the end-effector frame, arms concatenated (requires `obs.force_torque.enabled: true`) |
| `force_torque_world` | `(3,)` per arm | End-effector `[Fx, Fy, Fz]` in the world frame, arms concatenated (requires `obs.force_torque.enabled: true`) |
| `robot<i>_tactile_<finger>` | `(rows, cols)` | Tactile sensor grid (finger grippers only) |

`object_pos`, `object_quat` and `object_state` are simulator ground truth, and some tasks add more of their own (e.g. `platform_pos` and `rod_tilt_deg` in the balance tasks, `lpiece_<i>_pos` in packing, `typed_count` in keyboard-typing). They reveal what several tasks hide on purpose, such as the red cube under a lid in occluder-push. By default (`privileged_obs=False`) a task returns only `robot<i>_*` proprioception and tactile, `rgb_*`/`depth_*` cameras, and `force_torque`/`force_torque_world`, both from `ib.make()` and from `gymnasium.make()`. Pass `privileged_obs=True` to keep the ground truth, which the scripted oracles and the demo collectors (and so the recorded datasets) use. `imbench.eval.run_benchmark` likewise defaults to `privileged_obs=False` and `env_access=False`, so the policy never sees the live task.

Camera frames are rendered at `recording.save_freq.cameras` (25 Hz for most tasks at 100 Hz control): on reset and then every `control_freq / cameras`-th step, counted from step 0 or from the step passed to `env.align_camera_frames(n)`. Steps in between return the last rendered frame. The collectors align rendering to the first recorded step, so every saved video frame shows the state of its parquet row.

## Task Configuration

Each task loads `<task_id>.yaml` from `configs/tasks/` or one of its subdirectories. Abridged from [`configs/tasks/hetbi/ib-p1-bowl-table-v1.yaml`](../../configs/tasks/hetbi/ib-p1-bowl-table-v1.yaml):

```yaml
task_id: ib-p1-bowl-table-v1
description: 'Bowl starts upside-down inside a rack compartment. Suction gripper grasps the bowl bottom and extracts it from the rack, then hands it over to parallel jaw gripper which grasps the rim and places it on the table.'
max_episode_steps: 5000
action_mode: cartesian_delta
control_freq: 100

recording:
  save_freq:
    proprio: 100
    force_torque: 50
    object_state: 50
    cameras: 25
    tactile: 50

payload: bowl
fixture: rack
grippers:
- PandaGripper
- PandaSuctionGripper

controller:
- type: IK_POSE
  robot: Panda
  ik_pos_limit: 1.0
  ik_ori_limit: 1.0
  input_type: absolute
  impedance_mode: fixed
  kp: 10000
  damping_ratio: 1
  kp_limits: [0, 100000]
- type: IK_POSE
  robot: Panda
  ik_pos_limit: 1.0
  ik_ori_limit: 1.0
  input_type: absolute
  impedance_mode: fixed
  kp: 10000
  damping_ratio: 1
  kp_limits: [0, 100000]
ik_scale:
  pos_sensitivity: 0.005
  ori_sensitivity: 0.01

obs:
  proprio:
    enabled: true
    keys:
    - robot0_eef_pos
    - robot0_eef_quat
    - robot0_gripper_qpos
    - robot0_joint_pos
    - robot0_joint_vel
    - robot1_eef_pos
    - robot1_eef_quat
    - robot1_gripper_qpos
    - robot1_joint_pos
    - robot1_joint_vel
  object_state:
    enabled: true
  cameras:
    exo_front_cam:
      enabled: true
      robosuite_name: frontview
      width: 640
      height: 480
      depth: false
  force_torque:
    enabled: true
    keys:
    - robot0_ft_sensor
    - robot1_ft_sensor
  tactile:
    enabled: true
    fingers:
    - right
    - left
    rows: 8
    cols: 8
    spacing: 0.001
```

Optional `physics` keys:

- `multiccd: true` enables MuJoCo's multi-point convex collisions, for objects that rest on flat cylinder or mesh faces; with single-point contacts they rock, creep or sink into their support. It is per task, not global, because it changes how on-edge objects settle (it breaks `ib-p1-plate-shelf-hobi-v1`).

## Reward

The YAML `reward_type` key sets the `reward` returned by `step()`. Every task ships with `reward_type: sparse` (also the default when the key is absent): `reward` is 1.0 on steps where `info["is_success"]` is true, else 0.0. With `reward_type: dense` (e.g. `config_overrides={"reward_type": "dense"}`), `reward` equals `info["task_score"]`, a progress score in `[0, 1]`: dense for every task except `ib-p2-galileo-ramp-drop-v1` / `-v2`, where it is binary. Pass `reward_fn=f` to `ib.make()` to return `f(task, raw_obs)` instead; it overrides `reward_type`. `info["task_score"]` and `info["is_success"]` are the same in every mode.

## Contributing a New Task

Subclass `BaseTask` (or an existing family base such as `SingleArmBase` / `HetBiBaseTask`), implement the 4 required methods (`_build_env`, `_get_obs`, `_compute_task_score`, `_check_termination`), add a YAML config under `configs/tasks/`, and add the task to `TASK_REGISTRY` in `imbench/registry.py`. Tasks are not auto-registered from YAML. Report success via `StagedReward`'s `success_fn` or override `_is_success`; it surfaces as `info["is_success"]`.
