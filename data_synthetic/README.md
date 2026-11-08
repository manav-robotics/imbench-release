# Synthetic Data Collection

Scripted policies that drive tasks to completion and record the successful
episodes as **LeRobot v2.1** datasets (Parquet + MP4), in the same layout as
teleoperated data.

## Prerequisites

Run `python scripts/setup_machine.py` once per machine. Every collector reads
the machine ID from `~/.imbench/machine.json` and takes run IDs from the shared
counter in `~/.imbench/run_counter.json`.

## Collectors and tasks

| Collector | Tasks |
|-----------|-------|
| [`singlearm_collect_demos.py`](singlearm_collect_demos.py) | The 25 single-arm `-v1` tasks whose row below names it |
| [`bimanual_collect_demos.py`](bimanual_collect_demos.py) | `ib-m-keyboard-typing-v1` |
| [`hetbi_scripted/collect.py`](hetbi_scripted/collect.py) | All 5 heterogeneous-bimanual tasks |
| [`mp_hobi_plate_shelf.py`](mp_hobi_plate_shelf.py) | `ib-p1-plate-shelf-hobi-v1` |
| [`mp_balance_hard.py`](mp_balance_hard.py) | `ib-p7-balance-hard-v1`: stands the rod upright on the ridge; succeeds on about 1 seed in 6 |

There is no scripted policy for `ib-m-dining-table-v1`, `ib-m-freeform-v1`,
`ib-p3-ball-cage-escape-v1` or `ib-p7-balance-medium-v1`.
The `-v2` tasks are out-of-distribution evaluation variants: no scripted
demonstrations are provided for them and the collectors reject them.

| Task ID | How to collect |
|---------|----------------|
| `ib-m-mirror-pick-place-v1` | `singlearm_collect_demos.py --task-id ib-m-mirror-pick-place-v1` |
| `ib-p1-edge-slide-v1` | `singlearm_collect_demos.py --task-id ib-p1-edge-slide-v1` |
| `ib-p1-recover-peg-insert-v1` | `singlearm_collect_demos.py --task-id ib-p1-recover-peg-insert-v1` |
| `ib-p1-sheltered-grasp-v1` | `singlearm_collect_demos.py --task-id ib-p1-sheltered-grasp-v1` |
| `ib-p2-bounce-intercept-v1` | `singlearm_collect_demos.py --task-id ib-p2-bounce-intercept-v1` |
| `ib-p2-cube-toss-v1` | `singlearm_collect_demos.py --task-id ib-p2-cube-toss-v1` |
| `ib-p2-galileo-ramp-drop-v1` | `singlearm_collect_demos.py --task-id ib-p2-galileo-ramp-drop-v1` |
| `ib-p2-linear-intercept-v1` | `singlearm_collect_demos.py --task-id ib-p2-linear-intercept-v1` |
| `ib-p2-pendulum-grasp-v1` | `singlearm_collect_demos.py --task-id ib-p2-pendulum-grasp-v1` |
| `ib-p2-ramp-sort-v1` | `singlearm_collect_demos.py --task-id ib-p2-ramp-sort-v1` |
| `ib-p2-slide-catch-v1` | `singlearm_collect_demos.py --task-id ib-p2-slide-catch-v1` |
| `ib-p3-domino-select-v1` | `singlearm_collect_demos.py --task-id ib-p3-domino-select-v1` |
| `ib-p3-domino-single-v1` | `singlearm_collect_demos.py --task-id ib-p3-domino-single-v1` |
| `ib-p3-gap-funnel-v1` | `singlearm_collect_demos.py --task-id ib-p3-gap-funnel-v1` |
| `ib-p3-indirect-push-v1` | `singlearm_collect_demos.py --task-id ib-p3-indirect-push-v1` |
| `ib-p3-seesaw-balance-v1` | `singlearm_collect_demos.py --task-id ib-p3-seesaw-balance-v1` |
| `ib-p4-cup-extract-v1` | `singlearm_collect_demos.py --task-id ib-p4-cup-extract-v1` |
| `ib-p4-tool-retrieve-v1` | `singlearm_collect_demos.py --task-id ib-p4-tool-retrieve-v1` |
| `ib-p5-mass-sort-v1` | `singlearm_collect_demos.py --task-id ib-p5-mass-sort-v1` |
| `ib-p5-occluder-push-v1` | `singlearm_collect_demos.py --task-id ib-p5-occluder-push-v1` |
| `ib-p6-pick-place-ycb-v1` | `singlearm_collect_demos.py --task-id ib-p6-pick-place-ycb-v1` |
| `ib-p6-slip-recovery-v1` | `singlearm_collect_demos.py --task-id ib-p6-slip-recovery-v1` |
| `ib-p6-stack-collapse-recovery-v1` | `singlearm_collect_demos.py --task-id ib-p6-stack-collapse-recovery-v1` |
| `ib-p7-balance-hard-v1` | `python -m data_synthetic.mp_balance_hard` (low yield, see above) |
| `ib-p7-packing-v1` | `singlearm_collect_demos.py --task-id ib-p7-packing-v1` |
| `ib-p7-shape-stack-v1` | `singlearm_collect_demos.py --task-id ib-p7-shape-stack-v1` |
| `ib-m-keyboard-typing-v1` | `bimanual_collect_demos.py --task-id ib-m-keyboard-typing-v1` |
| `ib-p1-plate-shelf-hobi-v1` | `python -m data_synthetic.mp_hobi_plate_shelf` |
| `ib-p1-plate-shelf-hetbi-v1`, `ib-p1-bowl-table-v1`, `ib-p1-cup-inversion-v1`, `ib-p1-cracker-box-suction-v1`, `ib-p1-cracker-box-jaw-v1` | `python -m data_synthetic.hetbi_scripted.collect --task <task-id>` (or `--all`) |

`singlearm_collect_demos.py` and `bimanual_collect_demos.py` are run as
`python data_synthetic/<script> ...`; the others as `python -m data_synthetic.<module> ...`.
A scripted policy does not succeed on every seed, and some currently succeed
rarely under the tasks' success conditions; the collector keeps retrying (see
`--n-episodes`) and exits non-zero if it cannot reach the requested count.

## Quick start

```bash
# Single-arm: 10 successful mirror-pick-place episodes in the viewer + dashboard
python data_synthetic/singlearm_collect_demos.py \
    --task-id ib-m-mirror-pick-place-v1 --n-episodes 10 --render

# Bimanual: 5 keyboard-typing episodes, headless, custom output root
python data_synthetic/bimanual_collect_demos.py \
    --task-id ib-m-keyboard-typing-v1 --n-episodes 5 --out-dir demos_synthetic

# Het-bi: every task, 5 episodes each
python -m data_synthetic.hetbi_scripted.collect --all --n-episodes 5
```

## Keyboard word list

Keyboard collection randomly samples a word for each attempt from
[`keyboard_words.txt`](keyboard_words.txt), a list of 25 words. The task's
own config draws from the same 25 words, so demos and evaluation match. Pass
`--words-file path/to/words.txt` to use another vocabulary
(one A-Z word per line). The attempt seed reproduces both the word and arm
assignment. Only successful episodes are saved, so word counts need not be equal.
The selected word and effective task config are saved with each episode;
`scripts/replay_episode.py` restores that config when replaying.

```bash
python data_synthetic/bimanual_collect_demos.py \
    --task-id ib-m-keyboard-typing-v1 \
    --words-file data_synthetic/keyboard_words.txt \
    --n-episodes 100 --out-dir demos_keyboard --seed 0
```

## What gets written

```
<out-dir>/<task-id>/M<machine-id>/          (default <out-dir>: demos/)
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/rgb_<camera>/episode_000000.mp4
├── meta/info.json, episodes.jsonl, tasks.jsonl
├── meta/episode_000000_metadata.json
├── npy/traj_000000.npy                      (--save-npy)
└── episode_length_stats.json   (step-count stats of every episode in the dataset)
```

- Only episodes that end with the task's success condition met
  (`info["is_success"]` on the final step) are saved; failed attempts and an
  episode interrupted with Ctrl-C, SIGTERM or SIGHUP are discarded. An episode
  ends when the task terminates or reaches its horizon (the het-bi and `mp_*`
  collectors also end it when the policy reports it is done).
- Parquet, MP4, metadata and `.npy` files of an episode share the same
  episode index. The metadata records the task ID, run ID, `seed`,
  `warmup_steps`, policy name and final score; operator fields are `null`.
- Attempt *k* resets the task with seed `first_seed + k` and seeds the
  policy's random generator from the same value, so an episode is reproducible
  from its recorded seed. `first_seed` is `--seed`, by default
  `machine_id × 1,000,000` so machines do not duplicate each other's episodes.
- Re-running into an existing dataset appends to it and continues after the
  largest seed already recorded.
- Actions are clipped to the task's action space before they are stepped, and
  the recorded action is the clipped policy output. Tasks that override part of
  the action inside `step()` (arm 0 of keyboard-typing-v1, the frozen arm of
  cracker-box-suction/jaw) execute something else in those columns.
- The single-arm and bimanual collectors step one zero action after each reset
  before recording (`warmup_steps: 1` in the metadata).

## Flags

All collectors accept:

| Flag | Default | Notes |
|------|---------|-------|
| `--n-episodes N` | 40 single-arm, 10 otherwise | Number of **successful** episodes to save. Up to `max(10·N, N+20)` attempts are made; the exit status is non-zero if fewer than N succeed. |
| `--out-dir PATH` | `demos` | Output root; the dataset goes to `PATH/<task-id>/M<machine-id>/`. |
| `--max-steps N` | YAML `max_episode_steps` | Per-episode step limit override. |
| `--seed N` | `machine_id × 1,000,000` | Seed of the first attempt (see above). |
| `--obs-mode` | `combo` | `combo` (proprio + cameras), `proprio` or `rgb`. |
| `--render` | off | Open the MuJoCo viewer (paced to real time) and the camera/observation dashboard. Needs a display; ignored with a warning on a headless machine. |
| `--no-dashboard` | off | With `--render`, open only the viewer. |
| `--save-npy` | off | Also write each saved episode's `(T, action_dim)` actions to `npy/traj_NNNNNN.npy`. |

`singlearm_collect_demos.py` and `bimanual_collect_demos.py` also take
`--task-id` and `--task-display-name` (metadata name; default derived from
the task ID). `hetbi_scripted/collect.py` takes `--task ID` or `--all` (and
`--verbose` for per-phase logging); it and
the `mp_*` collectors take `--no-lerobot` (dry run: nothing is written unless
`--save-npy` is given, in which case actions go to `npy/seed_<seed>.npy`).

## Adding a policy

A policy implements `reset()` and `act(obs)` or `act(obs, env=None)` and draws
all randomness from `self.rng`; the collector constructs it once per episode
(passing `env=` and `rng=` if the constructor accepts them), assigns
`policy.rng = numpy.random.default_rng(seed)` and calls `reset()`. Register
single-arm policies (modules in [`single_arm_pipeline/`](single_arm_pipeline/))
in `_POLICY_REGISTRY` of
[`singlearm_collect_demos.py`](singlearm_collect_demos.py) under the exact task
ID (bimanual ones, in [`bimanual_pipeline/`](bimanual_pipeline/), in
[`bimanual_collect_demos.py`](bimanual_collect_demos.py), het-bi ones in
`TASK_POLICY_MAP` of [`hetbi_scripted/__init__.py`](hetbi_scripted/__init__.py)).
Shared collection logic lives in [`collection.py`](collection.py).

Some policy modules also contain an ad-hoc `__main__` debugging runner; most of
those open the MuJoCo viewer and need a display. Policies that print per-phase
diagnostics take a `verbose` constructor argument, off by default. To try a policy headless, run
its collector with `--n-episodes 1 --out-dir /tmp/try`.

## Troubleshooting

- **`machine not configured`** — run `python scripts/setup_machine.py`.
- **`... is an out-of-distribution evaluation variant`** — `-v2` tasks have no
  scripted demos; collect the `-v1` task instead.
- **`No scripted policy for ...`** — the task is not registered in that
  collector; see the table above for which collector handles it.
- **Exit status 1, `attempt budget exhausted`** — the policy succeeded on fewer
  seeds than requested; the episodes that did succeed are saved.
