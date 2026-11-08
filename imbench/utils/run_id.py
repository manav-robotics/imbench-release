"""Machine identity and the global run-ID counter.

Every recorded episode gets a run ID ``YYYYMMCCCCCC``: 4-digit year, 2-digit
machine ID, 6-digit counter.  The counter lives in
``~/.imbench/run_counter.json`` and is shared by every tool on the machine
(``collect_demos.py``, ``generate_dataset.py``, ``merge_annotated_runs.py``),
so increments are serialised with an exclusive
``flock`` on ``run_counter.json.lock`` and written atomically.  A counter
file that cannot be parsed raises :class:`RunCounterError` instead of being
silently restarted from zero, which would re-issue existing IDs.

Post-processing names episode files after the last eight digits of the run
ID (``MMCCCCCC``, see :func:`episode_index_from_run_id`).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
from datetime import datetime
from pathlib import Path

from imbench.utils.fileio import atomic_write_json

RUN_ID_RE = re.compile(r"^\d{12}$")
MACHINE_ID_RE = re.compile(r"^\d{2}$")
MAX_COUNTER = 999_999


class RunCounterError(RuntimeError):
    """The run counter is unreadable, invalid or exhausted."""


def imbench_dir() -> Path:
    """Return ``~/.imbench`` (resolved against the current ``HOME``)."""
    return Path(os.path.expanduser("~/.imbench"))


def machine_config_path() -> Path:
    return imbench_dir() / "machine.json"


def counter_path() -> Path:
    return imbench_dir() / "run_counter.json"


def validate_machine_id(machine_id: object) -> str:
    """Return *machine_id* if it is exactly two digits, else raise ``ValueError``."""
    if not isinstance(machine_id, str) or not MACHINE_ID_RE.match(machine_id):
        raise ValueError(f"machine_id must be exactly two digits (00-99), got {machine_id!r}")
    return machine_id


def load_machine_id(path: str | os.PathLike | None = None) -> str:
    """Read and validate ``machine_id`` from ``~/.imbench/machine.json``.

    Raises ``FileNotFoundError`` when the machine has not been configured and
    ``ValueError`` when the stored ID is malformed.
    """
    path = Path(path) if path is not None else machine_config_path()
    with open(path) as f:
        cfg = json.load(f)
    try:
        return validate_machine_id(cfg.get("machine_id"))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}; rerun scripts/setup_machine.py") from exc


def episode_index_from_run_id(run_id: str) -> int:
    """Return the ``MMCCCCCC`` part of a run ID, used as the merged episode index."""
    if not RUN_ID_RE.match(run_id):
        raise ValueError(f"not a valid run ID (YYYYMMCCCCCC): {run_id!r}")
    return int(run_id[-8:])


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{path}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _read(path: Path) -> int | None:
    """Return the stored counter, ``None`` if the file is absent."""
    try:
        with open(path) as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    try:
        counter = json.loads(raw)["counter"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RunCounterError(
            f"{path} is unreadable ({exc.__class__.__name__}: {exc}). Refusing to "
            "restart from 0, which would re-issue existing run IDs. Restore the file, "
            "or run `python scripts/setup_machine.py --set-counter N` with N at least "
            "the highest counter already issued on this machine."
        ) from exc
    if not isinstance(counter, int) or isinstance(counter, bool) or not 0 <= counter <= MAX_COUNTER:
        raise RunCounterError(f"{path}: invalid counter value {counter!r}")
    return counter


def _write(path: Path, counter: int) -> None:
    atomic_write_json(path, {"counter": counter})


def read_counter(path: str | os.PathLike | None = None) -> int | None:
    """Return the current counter value, or ``None`` if no counter file exists."""
    path = Path(path) if path is not None else counter_path()
    with _locked(path):
        return _read(path)


def init_counter(
    value: int = 0,
    *,
    overwrite: bool = False,
    path: str | os.PathLike | None = None,
) -> int:
    """Create the counter file with *value* and return the stored value.

    Without *overwrite* an existing counter is kept and its value returned,
    and an unreadable one raises :class:`RunCounterError`.
    """
    if not 0 <= value <= MAX_COUNTER:
        raise ValueError(f"counter must be in [0, {MAX_COUNTER}], got {value}")
    path = Path(path) if path is not None else counter_path()
    with _locked(path):
        if not overwrite:
            current = _read(path)
            if current is not None:
                return current
        _write(path, value)
        return value


def next_run_id(
    machine_id: str,
    *,
    path: str | os.PathLike | None = None,
    year: int | None = None,
) -> str:
    """Atomically increment the counter and return the new run ID."""
    validate_machine_id(machine_id)
    path = Path(path) if path is not None else counter_path()
    with _locked(path):
        counter = (_read(path) or 0) + 1
        if counter > MAX_COUNTER:
            raise RunCounterError(f"{path}: counter exhausted ({MAX_COUNTER})")
        _write(path, counter)
    year = datetime.now().year if year is None else year
    return f"{year:04d}{machine_id}{counter:06d}"
