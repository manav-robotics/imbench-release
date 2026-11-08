"""Persistent, non-destructive annotation store.

Saves per-episode ratings to ``annotations.json`` alongside the dataset.
Designed so that:
  - Every save is merge-on-write: under an exclusive ``flock`` on
    ``annotations.json.lock`` the file is re-read, only the episode being
    rated or cleared is changed, and the result is written atomically.
    Concurrent sessions on the same dataset therefore keep each other's
    ratings; a save to an episode that another session changed since this
    one last read the file is refused instead of overwriting it.
  - New episodes discovered in the dataset are added as pending entries;
    entries for episodes no longer present are kept.
  - The annotator can resume from the first un-annotated episode.
  - An unreadable file is moved aside to ``annotations.json.corrupt-<time>``
    with a warning instead of being overwritten or crashing the dashboard.

File format (annotations.json)::

    {
        "dataset": "demos/ib-p1-edge-slide-v1/M01",
        "task_id": "ib-p1-edge-slide-v1",
        "total_episodes": 9,
        "annotations": {
            "0": {"rating": 5, "annotator": "alice", "timestamp": "..."},
            "1": {"rating": 2, "annotator": "alice", "timestamp": "..."},
            "2": null,
            ...
        }
    }

Legacy entries of the form ``{"status": "good" | "bad"}`` read as ratings
5 and 1 (see :func:`rating_from_annotation`); a rating that is not an
integer from 1 to 5 reads as pending, with a warning.
"""

import contextlib
import fcntl
import json
import os
import warnings
from datetime import datetime, timezone

from imbench.utils.fileio import atomic_write_json

_LEGACY_STATUS_RATINGS = {"good": 5, "bad": 1}


def rating_from_annotation(ann: dict | None) -> int | None:
    """Return the 1-5 rating stored in one annotation entry, or None if pending."""
    if not isinstance(ann, dict):
        return None
    rating = ann.get("rating")
    if rating is None:
        return _LEGACY_STATUS_RATINGS.get(ann.get("status"))
    if isinstance(rating, int) and not isinstance(rating, bool) and 1 <= rating <= 5:
        return rating
    warnings.warn(f"ignoring annotation with invalid rating {rating!r}", RuntimeWarning, stacklevel=2)
    return None


class AnnotationStore:
    """Read / write annotation decisions for a LeRobot dataset."""

    FILENAME = "annotations.json"

    def __init__(self, dataset_root: str, task_id: str, total_episodes: int):
        self.dataset_root = dataset_root
        self.task_id = task_id
        self.total_episodes = total_episodes
        self.path = os.path.join(dataset_root, self.FILENAME)
        with self._locked():
            self._data = self._with_episode_keys(self._read_disk())

    # ── persistence ──────────────────────────────────────────────────

    @contextlib.contextmanager
    def _locked(self):
        with open(self.path + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _empty(self) -> dict:
        return {
            "dataset": self.dataset_root,
            "task_id": self.task_id,
            "total_episodes": self.total_episodes,
            "annotations": {},
        }

    def _read_disk(self) -> dict:
        """Return the on-disk document; a corrupt file is moved aside with a warning."""
        try:
            with open(self.path) as f:
                data = json.load(f)
            if not isinstance(data, dict) or not isinstance(data.get("annotations", {}), dict):
                raise ValueError("unexpected document structure")
            return data
        except FileNotFoundError:
            return self._empty()
        except (ValueError, UnicodeDecodeError) as exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = f"{self.path}.corrupt-{stamp}"
            os.replace(self.path, backup)
            warnings.warn(
                f"{self.path} is unreadable ({exc}); moved it to {backup} and "
                "started from an empty annotation set",
                RuntimeWarning,
                stacklevel=3,
            )
            return self._empty()

    def _with_episode_keys(self, data: dict) -> dict:
        """Ensure every current episode index has an entry, preserving existing ones."""
        annotations = data.setdefault("annotations", {})
        for i in range(self.total_episodes):
            annotations.setdefault(str(i), None)
        data["total_episodes"] = self.total_episodes
        return data

    def _store(self, episode_index: int, entry: dict | None) -> bool:
        """Merge one episode's entry into the file on disk (locked, atomic).

        Returns False without writing when another session changed that
        episode's entry since this one last read the file.  Either way the
        in-memory copy is refreshed from disk.
        """
        key = str(episode_index)
        with self._locked():
            disk = self._with_episode_keys(self._read_disk())
            unchanged = disk["annotations"].get(key) == self._data["annotations"].get(key)
            if unchanged:
                disk["annotations"][key] = entry
                atomic_write_json(self.path, disk)
            self._data = disk
        return unchanged

    # ── read helpers ─────────────────────────────────────────────────

    def get(self, episode_index: int) -> dict | None:
        """Return annotation dict for an episode, or None if not yet annotated."""
        return self._data["annotations"].get(str(episode_index))

    def is_annotated(self, episode_index: int) -> bool:
        return self.rating_of(episode_index) is not None

    def rating_of(self, episode_index: int) -> int | None:
        """Return the rating (1-5) for an episode, or None if pending."""
        return rating_from_annotation(self.get(episode_index))

    def first_unannotated(self) -> int | None:
        """Return the index of the first episode without an annotation, or None."""
        for i in range(self.total_episodes):
            if not self.is_annotated(i):
                return i
        return None

    def next_unannotated(self, after: int) -> int | None:
        """Return the first episode without an annotation after *after*, wrapping around, or None."""
        for step in range(1, self.total_episodes + 1):
            i = (after + step) % self.total_episodes
            if not self.is_annotated(i):
                return i
        return None

    def summary(self) -> dict:
        """Return {rated, pending, average, counts}."""
        counts: dict[int, int] = {}
        pending = 0
        total_rating = 0
        rated = 0
        for i in range(self.total_episodes):
            r = self.rating_of(i)
            if r is not None:
                counts[r] = counts.get(r, 0) + 1
                total_rating += r
                rated += 1
            else:
                pending += 1
        return {
            "rated": rated,
            "pending": pending,
            "average": total_rating / rated if rated > 0 else None,
            "counts": counts,
        }

    # ── write helpers ────────────────────────────────────────────────

    def annotate(self, episode_index: int, rating: int, annotator: str = "") -> bool:
        """Record a 1-5 rating and immediately persist; False if refused (see :meth:`_store`)."""
        return self._store(episode_index, {
            "rating": int(rating),
            "annotator": annotator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def clear(self, episode_index: int) -> bool:
        """Reset an episode to pending and persist; False if refused (see :meth:`_store`)."""
        return self._store(episode_index, None)
