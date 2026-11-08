"""Episode-length statistics for scripted-policy collection runs.

After each collect_* run finishes, the caller passes the step counts of
every episode in the dataset to ``record_run``. The function:

  - Prints a summary block (n, min, mean ± std, max).
  - Updates a JSON log file at ``<out_dir>/episode_length_stats.json``,
    keyed by ``task_id``. Existing entries for the same task are
    replaced; new tasks are appended.

The JSON layout is human-readable so it can be diffed across runs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterable

import numpy as np

from imbench.utils.fileio import atomic_write_json


_LOG_FILENAME = "episode_length_stats.json"


def sample_target_length(rng: np.random.RandomState, mean: float, std: float, lo: float, hi: float) -> int:
    """Draw a target episode length from N(*mean*, *std*) truncated to [*lo*, *hi*].

    Resampling, unlike clipping, keeps out-of-range draws from piling up at a bound.
    """
    for _ in range(1000):
        x = rng.normal(mean, std)
        if lo <= x <= hi:
            return int(round(x))
    return int(round(np.clip(mean, lo, hi)))


def _format_summary(name: str, lengths: np.ndarray) -> str:
    if lengths.size == 0:
        return f"[{name}] no successful episodes — nothing to log"
    return (
        f"[{name}]  n={lengths.size}  "
        f"min={int(lengths.min())}  "
        f"mean={lengths.mean():.1f} ± {lengths.std():.1f}  "
        f"max={int(lengths.max())}"
    )


def record_run(
    out_dir: str,
    task_id: str,
    task_display_name: str,
    lengths: Iterable[int],
) -> None:
    """Print stats for *lengths* and persist them to the log file.

    ``lengths`` should contain the step count of every saved episode.
    Failed/discarded attempts must not be included.
    """
    arr = np.asarray(list(lengths), dtype=np.int64)
    print()
    print(_format_summary(task_display_name or task_id, arr))

    if arr.size == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _LOG_FILENAME)

    log: dict = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                log = json.load(f)
            if not isinstance(log, dict):
                log = {}
        except (json.JSONDecodeError, OSError):
            log = {}

    log[task_id] = {
        "task_display_name": task_display_name or task_id,
        "n_episodes":        int(arr.size),
        "min":               int(arr.min()),
        "mean":              float(arr.mean()),
        "std":               float(arr.std()),
        "max":               int(arr.max()),
        "last_updated":      datetime.now().isoformat(timespec="seconds"),
    }

    atomic_write_json(path, log, sort_keys=True)
    print(f"[episode_length_stats] updated {path}")
