#!/usr/bin/env python3
"""Merge annotated runs from per-machine M{id} folders into a unified dataset.

Walks ``<input>/<task>/M<id>/`` folders (as written by ``collect_demos.py``),
optionally filters by annotation rating, assigns globally unique run_ids,
renames episode files to the last eight digits of the run_id (``MMCCCCCC``)
and copies everything into a flat output directory per task.  The parquet
``episode_index`` column is set to that merged index; the global ``index``
column keeps the source values until ``convert_v21_to_v30.py`` assigns the
dataset-wide index.

If an episode's metadata already has a valid run_id (``YYYYMMCCCCCC``), it is
reused and the counter is not bumped.  Otherwise a new run_id is issued from
the shared counter in ``~/.imbench/run_counter.json`` under the source
folder's machine ID.

Rating filter: ``--min-rating N`` keeps only episodes whose rating in the
source ``annotations.json`` is at least N (legacy ``status: good|bad``
entries count as 5 and 1); unrated episodes and sources without
``annotations.json`` are dropped.  Without ``--min-rating`` every episode is
taken, rated or not.  A source whose ``annotations.json`` cannot be read is
skipped and the script exits non-zero.

Before anything is copied, the run aborts if the sources of a task or its
existing output differ in ``features`` / ``fps`` / ``save_freq``, if an output
task folder is not a v2.1 dataset (e.g. after ``convert_v21_to_v30.py
--in-place``), or on a name collision: an output episode name claimed by two
planned episodes, or by an existing output episode copied from a different
source (e.g. two machines configured with the same ID, or a reset counter
re-issuing a run_id).  Source episodes are told apart by run_id plus a
fingerprint of their parquet (row count, SHA-256), kept as
``source_fingerprint`` in the output metadata.

A manifest (processed_manifest.json) in the output directory records each
processed source episode (``<task>/M<id>/<episode>``, run_id, fingerprint) so
re-runs skip it.  Planned run_ids are saved there before copying, and each
episode is written under temporary names, renamed into place (metadata last)
and recorded as soon as it is complete, so an interrupted run resumes with
the same run_ids and without duplicates.

Usage:
    # Dry run
    python scripts/merge_annotated_runs.py --input demos/ --dry-run

    # Keep episodes rated 4 or 5
    python scripts/merge_annotated_runs.py --input demos/ --output ~/processed_runs --min-rating 4

    # No rating filter (every episode, rated or not)
    python scripts/merge_annotated_runs.py --input demos/ --output ~/processed_runs
"""

import argparse
import contextlib
import glob
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_verification.annotation_store import rating_from_annotation  # noqa: E402
from imbench.utils.fileio import atomic_write_json  # noqa: E402
from imbench.utils.run_id import (  # noqa: E402
    RUN_ID_RE,
    RunCounterError,
    episode_index_from_run_id,
    load_machine_id,
    next_run_id,
    read_counter,
)

# Per-episode fields carried into the rebuilt episodes.jsonl
_EPISODE_FIELDS = ("final_score", "teleop_device", "run_id", "seed", "warmup_steps",
                   "annotation_rating")


def _load_manifest(output_dir: str) -> dict:
    path = os.path.join(output_dir, "processed_manifest.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"processed": {}}


def _save_manifest(output_dir: str, manifest: dict):
    os.makedirs(output_dir, exist_ok=True)
    atomic_write_json(os.path.join(output_dir, "processed_manifest.json"), manifest)


def _load_annotations(task_dir: str) -> dict:
    """Return {episode_index: rating} for rated episodes in annotations.json, or {}.

    Raises ``ValueError`` naming the file when it is not valid JSON.
    """
    ann_path = os.path.join(task_dir, "annotations.json")
    if not os.path.exists(ann_path):
        return {}
    try:
        with open(ann_path) as f:
            data = json.load(f)
    except ValueError as exc:
        raise ValueError(f"cannot read {ann_path}: {exc}") from exc
    ratings = {}
    for ep_str, ann in data.get("annotations", {}).items():
        rating = rating_from_annotation(ann)
        if rating is not None:
            ratings[int(ep_str)] = rating
    return ratings


def _load_episode_entries(task_dir: str) -> dict:
    """Return {episode_index: entry} from the source meta/episodes.jsonl."""
    path = os.path.join(task_dir, "meta", "episodes.jsonl")
    entries = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entries[int(entry["episode_index"])] = entry
    return entries


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_episode_metadata(task_dir: str, tag: str) -> dict:
    path = os.path.join(task_dir, "meta", f"{tag}_metadata.json")
    if not os.path.exists(path):
        return {}
    return _read_json(path)


def _episode_record(meta: dict) -> dict:
    """Pick the per-episode fields for episodes.jsonl from a metadata JSON."""
    summary = meta.get("dataset_summary") or {}
    values = {
        "final_score": meta.get("final_score", summary.get("final_score")),
        "teleop_device": meta.get("teleop_device", summary.get("teleop_device")),
    }
    for key in _EPISODE_FIELDS:
        values.setdefault(key, meta.get(key))
    return {k: values[k] for k in _EPISODE_FIELDS if values[k] is not None}


def _output_parquets(out_task_dir: str) -> list:
    return sorted(glob.glob(os.path.join(out_task_dir, "data", "*", "episode_*.parquet")))


def _meta_is_stale(out_task_dir: str) -> bool:
    """True if a v2.1 info.json is missing or does not count every episode parquet (interrupted run)."""
    info_path = os.path.join(out_task_dir, "meta", "info.json")
    if not os.path.exists(info_path):
        return True
    info = _read_json(info_path)
    if info.get("codebase_version", "v2.1") != "v2.1":
        return False  # converted with --in-place; never rebuilt
    return info.get("total_episodes") != len(_output_parquets(out_task_dir))


def _rebuild_dataset_meta(out_task_dir: str, source_info_path: str, source_tasks_path: str = None):
    """Rebuild info.json and episodes.jsonl in the output task folder."""
    meta_dir = os.path.join(out_task_dir, "meta")

    info = _read_json(source_info_path)

    total_frames = 0
    episode_entries = []
    for pq_path in _output_parquets(out_task_dir):
        stem = os.path.basename(pq_path)[: -len(".parquet")]
        ep_idx = int(stem.split("_")[1])
        pq_file = pq.ParquetFile(pq_path)
        length = pq_file.metadata.num_rows
        total_frames += length
        ts = []
        if "timestamp" in pq_file.schema_arrow.names:
            ts = pq_file.read(columns=["timestamp"]).column("timestamp").to_pylist()

        entry = {
            "episode_index": ep_idx,
            "task_index": 0,
            "task": info.get("task_id", ""),
            "length": length,
            "timestamp_start": ts[0] if ts else 0.0,
            "timestamp_end": ts[-1] if ts else 0.0,
        }
        entry.update(_episode_record(_read_episode_metadata(out_task_dir, stem)))
        episode_entries.append(entry)

    info["total_episodes"] = len(episode_entries)
    info["total_frames"] = total_frames

    episode_entries.sort(key=lambda e: e["episode_index"])
    tmp = os.path.join(meta_dir, ".episodes.jsonl.tmp")
    with open(tmp, "w") as f:
        for entry in episode_entries:
            f.write(json.dumps(entry) + "\n")
    os.replace(tmp, os.path.join(meta_dir, "episodes.jsonl"))

    if source_tasks_path and os.path.exists(source_tasks_path):
        shutil.copy2(source_tasks_path, os.path.join(meta_dir, "tasks.jsonl"))

    # info.json last: _meta_is_stale() checks its total_episodes
    atomic_write_json(os.path.join(meta_dir, "info.json"), info)

    print(f"  [meta] {info.get('task_id', '?')}: info.json ({len(episode_entries)} episodes, "
          f"{total_frames} frames), episodes.jsonl, tasks.jsonl")


def _discover_tasks(input_dir: str) -> list:
    """Find (task_dir, source_label, task_name, machine_id) tuples.

    Handles two layouts:
      - New: {input}/{task}/M{id}/  → machine_id from folder name
      - Old: {input}/{task}/        → machine_id from ~/.imbench/machine.json
    """
    results = []
    machine_id = None

    for task_name in sorted(os.listdir(input_dir)):
        task_path = os.path.join(input_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        has_machine_dirs = False
        for entry in sorted(os.listdir(task_path)):
            entry_path = os.path.join(task_path, entry)
            if os.path.isdir(entry_path) and re.match(r"^M\d{2}$", entry):
                if os.path.isdir(os.path.join(entry_path, "meta")):
                    results.append((entry_path, f"{task_name}/{entry}", task_name, entry[1:]))
                    has_machine_dirs = True

        if not has_machine_dirs and os.path.isdir(os.path.join(task_path, "meta")):
            if machine_id is None:
                try:
                    machine_id = load_machine_id()
                except FileNotFoundError:
                    sys.exit("ERROR: Machine not configured. Run:  python scripts/setup_machine.py")
            results.append((task_path, f"{task_name} (legacy)", task_name, machine_id))

    return results


def _find_episodes(task_dir: str) -> list:
    """Find all episode indices by scanning meta/*_metadata.json."""
    meta_dir = os.path.join(task_dir, "meta")
    episodes = []
    for name in os.listdir(meta_dir):
        m = re.match(r"^episode_(\d+)_metadata\.json$", name)
        if m:
            episodes.append(int(m.group(1)))
    return sorted(episodes)


def _parquets_without_metadata(task_dir: str, episodes: list) -> list:
    """Return the episode parquets (relative to *task_dir*) whose index is not in *episodes*."""
    known = set(episodes)
    orphans = []
    for path in sorted(glob.glob(os.path.join(task_dir, "data", "*", "episode_*.parquet"))):
        m = re.match(r"^episode_(\d+)\.parquet$", os.path.basename(path))
        if m and int(m.group(1)) not in known:
            orphans.append(os.path.relpath(path, task_dir))
    return orphans


def _source_task_config(task_dir: str) -> dict:
    """The task config in a source folder's info.json, or {}."""
    path = os.path.join(task_dir, "meta", "info.json")
    return (_read_json(path).get("task_config") or {}) if os.path.exists(path) else {}


def _read_episode_run_id(task_dir: str, ep_idx: int) -> str:
    """Read the run_id from an episode's metadata, or return empty string."""
    return _read_episode_metadata(task_dir, f"episode_{ep_idx:06d}").get("run_id") or ""


def _source_parquet(task_dir: str, ep_idx: int) -> str | None:
    matches = glob.glob(os.path.join(task_dir, "data", "*", f"episode_{ep_idx:06d}.parquet"))
    return matches[0] if matches else None


def _fingerprint(parquet_path: str) -> dict:
    """Identify an episode by its parquet: row count and SHA-256 of the file."""
    rows = pq.read_metadata(parquet_path).num_rows
    with open(parquet_path, "rb") as f:
        return {"rows": rows, "sha256": hashlib.file_digest(f, "sha256").hexdigest()}


def _source_fingerprint(label: str, parquet_path: str) -> dict | None:
    """Return the fingerprint of a source parquet, or warn and return None if it is unreadable."""
    try:
        return _fingerprint(parquet_path)
    except (OSError, pa.ArrowException) as exc:
        print(f"  WARNING: {label}: cannot read {parquet_path} ({exc}) — skipping")
        return None


def _output_episode(out_task_dir: str, out_idx: int) -> dict | None:
    """Describe an existing output episode, or return None if none of its files exist.

    Returns ``{"complete": False}`` for files without the metadata JSON (an
    interrupted copy, see :func:`_copy_episode_files`), else ``complete``,
    ``run_id`` and the source ``fingerprint``.  Episodes merged before
    fingerprints were recorded are byte-for-byte copies of their source, so
    theirs is computed from the output parquet.
    """
    tag = f"episode_{out_idx:08d}"
    data = glob.glob(os.path.join(out_task_dir, "data", "*", f"{tag}.parquet"))
    if not os.path.exists(os.path.join(out_task_dir, "meta", f"{tag}_metadata.json")):
        videos = glob.glob(os.path.join(out_task_dir, "videos", "*", "*", f"{tag}.mp4"))
        return {"complete": False} if data or videos else None
    meta = _read_episode_metadata(out_task_dir, tag)
    fingerprint = meta.get("source_fingerprint")
    if fingerprint is None and data:
        fingerprint = _fingerprint(data[0])
    return {"complete": True, "run_id": meta.get("run_id") or "", "fingerprint": fingerprint}


def _already_merged(done: dict | None, run_id: str | None, fingerprint: dict, out_task_dir: str) -> bool:
    """True if manifest entry *done* records this source episode (same run_id and fingerprint).

    *run_id* is None for a source episode without one.  An entry from before
    fingerprints were recorded is checked against its output episode, and
    trusted if that is gone.
    """
    if done is None or (run_id is not None and done.get("run_id") != run_id):
        return False
    recorded = done.get("fingerprint")
    if recorded is None:
        existing = _output_episode(out_task_dir, done["out_episode"])
        if existing is None or not existing["complete"]:
            return True
        recorded = existing["fingerprint"]
    return recorded == fingerprint


def _pending_run_id(pending: dict, item: dict) -> str | None:
    """Return the run_id an interrupted run issued for this source episode, if any."""
    entry = pending.get(item["key"]) or {}
    if (entry.get("machine_id"), entry.get("fingerprint")) == (item["machine_id"], item["fingerprint"]):
        return entry["run_id"]
    return None


def find_conflicts(plan: list, output_dir: str, pending: dict | None = None) -> list:
    """Return descriptions of planned episodes whose output name is already taken.

    *plan* items are dicts with ``key``, ``task_name``, ``source``, ``out_idx``,
    ``run_id`` and ``fingerprint``.  A complete output episode may only be
    rewritten by the episode it was copied from (same run_id and fingerprint),
    an incomplete one only by the episode that the manifest's *pending*
    entries say an interrupted run was copying into it.
    """
    conflicts = []
    claimed = {}
    pending = pending or {}
    for item in plan:
        key = (item["task_name"], item["out_idx"])
        name = f"{item['task_name']}/episode_{item['out_idx']:08d}"
        source = f"{item['source']} ({item['run_id']})"
        if key in claimed:
            conflicts.append(f"{name}: {source} and "
                             f"{claimed[key]['source']} ({claimed[key]['run_id']})")
            continue
        claimed[key] = item
        existing = _output_episode(os.path.join(output_dir, item["task_name"]), item["out_idx"])
        if existing is None:
            continue
        if not existing["complete"]:
            interrupted = pending.get(item["key"]) or {}
            if (interrupted.get("out_episode"), interrupted.get("fingerprint")) != (
                    item["out_idx"], item["fingerprint"]):
                conflicts.append(f"{name}: {source} would overwrite an incomplete output episode "
                                 f"(files without metadata JSON)")
        elif existing["run_id"] != item["run_id"]:
            conflicts.append(f"{name}: {source} would overwrite the existing output episode "
                             f"(run_id {existing['run_id'] or 'missing'})")
        elif existing["fingerprint"] != item["fingerprint"]:
            conflicts.append(f"{name}: {source} would overwrite the existing output episode, "
                             f"a different recording with the same run_id")
    return conflicts


def abort_on_conflicts(conflicts: list) -> None:
    if not conflicts:
        return
    print(f"\nERROR: {len(conflicts)} output episode name collision(s); nothing was copied:")
    for c in conflicts:
        print(f"  {c}")
    print("Each machine needs a distinct ID and a counter that is never reset "
          "(see scripts/setup_machine.py).")
    sys.exit(1)


def abort_on_converted_outputs(output_dir: str, task_names) -> None:
    """Exit if an output task folder is not a v2.1 dataset, e.g. after ``convert --in-place``."""
    problems = []
    for task_name in sorted(set(task_names)):
        out_task_dir = os.path.join(output_dir, task_name)
        info_path = os.path.join(out_task_dir, "meta", "info.json")
        version = _read_json(info_path).get("codebase_version") if os.path.exists(info_path) else "v2.1"
        if version != "v2.1":
            problems.append(f"{out_task_dir}: meta/info.json has codebase_version {version!r}")
        elif glob.glob(os.path.join(out_task_dir, "data", "*", "file-*.parquet")):
            problems.append(f"{out_task_dir}: holds v3.0 shards (data/*/file-*.parquet)")
    if not problems:
        return
    print(f"\nERROR: {len(problems)} output task folder(s) are not LeRobot v2.1 datasets; "
          "nothing was copied:")
    for p in problems:
        print(f"  {p}")
    print("A dataset converted with convert_v21_to_v30.py cannot take more episodes: merge into a "
          "new --output directory, or first put the v2.1 dataset (<task>_v21_backup) back in its place.")
    sys.exit(1)


def _schema_diff(ref: dict, info: dict) -> list:
    """Describe how the ``fps``, ``save_freq`` and ``features`` of *info* differ from *ref*."""
    diffs = [f"{key} {info.get(key)} (not {ref.get(key)})"
             for key in ("fps", "save_freq") if info.get(key) != ref.get(key)]
    ref_features, features = ref.get("features", {}), info.get("features", {})
    for what, names in (
        ("missing", set(ref_features) - set(features)),
        ("extra", set(features) - set(ref_features)),
        ("different", {k for k in set(ref_features) & set(features) if ref_features[k] != features[k]}),
    ):
        if names:
            diffs.append(f"{what} features {sorted(names)}")
    return diffs


def abort_on_schema_mismatch(task_infos: dict) -> None:
    """Exit if the info.json files of a task differ in ``fps``, ``save_freq`` or ``features``.

    *task_infos* maps a task name to ``[(label, info_path), ...]``; each entry
    is compared with the first.
    """
    problems = []
    for (ref_label, ref_path), *others in task_infos.values():
        ref = _read_json(ref_path)
        for label, path in others:
            problems += [f"{label}: {d} vs {ref_label}" for d in _schema_diff(ref, _read_json(path))]
    if not problems:
        return
    print("\nERROR: sources of the same task were recorded with different settings; nothing was copied:")
    for p in problems:
        print(f"  {p}")
    print("Merge them into separate --output directories.")
    sys.exit(1)


def _copy_episode_files(task_dir: str, out_task_dir: str, ep_idx: int, out_idx: int,
                        run_id: str, rating: int = None, episode_entry: dict = None,
                        fingerprint: dict = None):
    """Copy one episode, renaming files and patching metadata.

    The parquet's ``episode_index`` column is set to *out_idx*; the global
    ``index`` column is left for ``convert_v21_to_v30.py`` to assign.  The
    metadata gets the run_id, the source *fingerprint*, the annotation
    rating, and any of ``seed`` / ``warmup_steps`` / ``final_score`` it lacks
    from the source episodes.jsonl entry, so replay and the rebuilt
    episodes.jsonl keep them.  Files are written under temporary names and
    renamed into place once all are written, metadata last, so an output
    episode with a metadata JSON is complete.
    """
    src_tag = f"episode_{ep_idx:06d}"
    dst_tag = f"episode_{out_idx:08d}"
    staged = []  # (temporary path, final path) in rename order

    def stage(dst_dir: str, name: str) -> str:
        os.makedirs(dst_dir, exist_ok=True)
        staged.append((os.path.join(dst_dir, f".{name}.tmp"), os.path.join(dst_dir, name)))
        return staged[-1][0]

    try:
        # data/chunk-*/episode_XXXXXX.parquet
        data_dir = os.path.join(task_dir, "data")
        if os.path.isdir(data_dir):
            for chunk in os.listdir(data_dir):
                src = os.path.join(data_dir, chunk, f"{src_tag}.parquet")
                if os.path.exists(src):
                    table = pq.read_table(src)
                    col = table.schema.get_field_index("episode_index")
                    if col >= 0:
                        field = table.schema.field(col)
                        table = table.set_column(
                            col, field, pa.array([out_idx] * table.num_rows, type=field.type))
                    pq.write_table(table, stage(os.path.join(out_task_dir, "data", chunk),
                                                f"{dst_tag}.parquet"))

        # videos/chunk-*/{cam_name}/episode_XXXXXX.mp4
        videos_dir = os.path.join(task_dir, "videos")
        if os.path.isdir(videos_dir):
            for chunk in os.listdir(videos_dir):
                chunk_path = os.path.join(videos_dir, chunk)
                if not os.path.isdir(chunk_path):
                    continue
                for cam in os.listdir(chunk_path):
                    src = os.path.join(chunk_path, cam, f"{src_tag}.mp4")
                    if os.path.exists(src):
                        shutil.copy2(src, stage(os.path.join(out_task_dir, "videos", chunk, cam),
                                                f"{dst_tag}.mp4"))

        # meta/episode_XXXXXX_metadata.json
        meta = _read_episode_metadata(task_dir, src_tag)
        meta["run_id"] = run_id
        if fingerprint is not None:
            meta["source_fingerprint"] = fingerprint
        if rating is not None:
            meta["annotation_rating"] = rating
        for key in ("seed", "warmup_steps", "final_score"):
            if meta.get(key) is None and (episode_entry or {}).get(key) is not None:
                meta[key] = episode_entry[key]
        # Replay needs the task config this episode was recorded with; the
        # merged info.json carries only the first source's.
        if "task_config" not in meta:
            config = (episode_entry or {}).get("task_config") or _source_task_config(task_dir)
            if config:
                meta["task_config"] = config
        with open(stage(os.path.join(out_task_dir, "meta"), f"{dst_tag}_metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

        for tmp, final in staged:
            os.replace(tmp, final)
    except BaseException:
        for tmp, _ in staged:
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmp)
        raise


def _manifest_entry(item: dict) -> dict:
    return {
        "run_id": item["run_id"],
        "out_episode": item["out_idx"],
        "machine_id": item["machine_id"],
        "fingerprint": item["fingerprint"],
    }


def _copy_planned_episodes(plan: list, output_dir: str, manifest: dict) -> None:
    """Copy every *plan* item, recording progress in the manifest.

    The planned run_ids go under ``pending`` before anything is copied, so a
    re-run after an interruption reuses them, and each episode moves to
    ``processed`` as soon as its files are in place.
    """
    pending = manifest.setdefault("pending", {})
    processed = manifest.setdefault("processed", {})
    for item in plan:
        pending[item["key"]] = _manifest_entry(item)
    _save_manifest(output_dir, manifest)
    for item in plan:
        _copy_episode_files(item["task_dir"], os.path.join(output_dir, item["task_name"]),
                            item["ep_idx"], item["out_idx"], item["run_id"],
                            rating=item["rating"], episode_entry=item["entry"],
                            fingerprint=item["fingerprint"])
        processed[item["key"]] = pending.pop(item["key"])
        _save_manifest(output_dir, manifest)


def _rebuild_outputs(output_dir: str, task_sources: dict, plan: list) -> None:
    """Rebuild the dataset metadata of task folders that got episodes or are stale."""
    copied = {item["task_name"] for item in plan}
    rebuild = [task_name for task_name in task_sources
               if os.path.isdir(os.path.join(output_dir, task_name))
               and (task_name in copied or _meta_is_stale(os.path.join(output_dir, task_name)))]
    if rebuild:
        print()
    for task_name in rebuild:
        sources = task_sources[task_name]
        _rebuild_dataset_meta(os.path.join(output_dir, task_name), sources["info"], sources.get("tasks"))


def main():
    parser = argparse.ArgumentParser(
        description="Post-process collected episodes with global run IDs and optional rating filter.",
    )
    parser.add_argument("--input", required=True, help="Path to demos directory (e.g. demos/)")
    parser.add_argument("--output", default=None, help="Path for output (e.g. ~/processed_runs)")
    parser.add_argument("--min-rating", type=int, default=None,
                        help="Keep only episodes rated at least this (1-5); unrated episodes are "
                             "dropped. Omit to take every episode, rated or not.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = parser.parse_args()

    dry_run = args.dry_run
    if not dry_run and not args.output:
        parser.error("--output is required (unless using --dry-run)")

    input_dir = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output) if args.output else None
    min_rating = args.min_rating

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        sys.exit(1)

    tasks = _discover_tasks(input_dir)
    if not tasks:
        print(f"No task folders found in {input_dir}")
        sys.exit(1)

    if dry_run:
        label = "DRY RUN"
        if min_rating is not None:
            label += f" — rating >= {min_rating}"
        print(f"{label}\n")

    manifest = _load_manifest(output_dir) if output_dir else {"processed": {}}
    processed = manifest.setdefault("processed", {})
    pending = manifest.get("pending", {})

    total_new = 0
    total_reuse = 0
    total_already = 0
    total_filtered = 0
    errors = 0
    task_sources = {}
    task_infos = {}
    plan = []

    for task_dir, label, task_name, machine_id in tasks:
        episodes = _find_episodes(task_dir)
        orphans = _parquets_without_metadata(task_dir, episodes)
        if orphans:
            print(f"  WARNING: {label}: no metadata JSON for {', '.join(orphans)} — not merged")
        try:
            annotations = _load_annotations(task_dir)
        except ValueError as exc:
            print(f"  ERROR: {label}: {exc} — skipping")
            errors += 1
            continue
        entries = _load_episode_entries(task_dir)
        out_task_dir = os.path.join(output_dir, task_name) if output_dir else None

        filtered = 0
        if min_rating is not None:
            if not os.path.exists(os.path.join(task_dir, "annotations.json")):
                print(f"  {label} [M{machine_id}]: {len(episodes)} episodes, no annotations.json — skipping")
                total_filtered += len(episodes)
                continue
            before = len(episodes)
            episodes = [e for e in episodes
                        if annotations.get(e) is not None and annotations[e] >= min_rating]
            filtered = before - len(episodes)
            total_filtered += filtered

        new_eps = []
        already_done = 0
        has_valid_rid = 0
        needs_rid = 0

        for ep_idx in episodes:
            key = f"{label}/{ep_idx}"
            parquet = _source_parquet(task_dir, ep_idx)
            if parquet is None:
                print(f"  WARNING: {label} episode {ep_idx} has metadata but no parquet — skipping")
                continue
            fingerprint = _source_fingerprint(f"{label} episode {ep_idx}", parquet)
            if fingerprint is None:
                continue
            rid = _read_episode_run_id(task_dir, ep_idx)
            valid = bool(RUN_ID_RE.match(rid))
            # A manifest hit only counts if it is the same episode (same run_id and parquet)
            if _already_merged(processed.get(key), rid if valid else None, fingerprint, out_task_dir):
                already_done += 1
                continue

            has_valid_rid += valid
            needs_rid += not valid
            new_eps.append(ep_idx)
            plan.append({
                "key": key,
                "task_dir": task_dir,
                "task_name": task_name,
                "machine_id": machine_id,
                "source": f"{label} episode {ep_idx}",
                "ep_idx": ep_idx,
                "run_id": rid if valid else None,
                "out_idx": episode_index_from_run_id(rid) if valid else None,
                "fingerprint": fingerprint,
                "rating": annotations.get(ep_idx),
                "entry": entries.get(ep_idx),
            })

        total_already += already_done
        total_new += len(new_eps)

        # Every source with episodes must match; the first one's info.json is used
        info_path = os.path.join(task_dir, "meta", "info.json")
        if episodes and os.path.exists(info_path):
            task_infos.setdefault(task_name, []).append((label, info_path))
            if task_name not in task_sources:
                tasks_path = os.path.join(task_dir, "meta", "tasks.jsonl")
                task_sources[task_name] = {
                    "info": info_path,
                    "tasks": tasks_path if os.path.exists(tasks_path) else None,
                }

        if dry_run:
            parts = [f"{len(new_eps)} episodes"]
            if has_valid_rid:
                parts.append(f"{has_valid_rid} have run_id")
            if needs_rid:
                parts.append(f"{needs_rid} need run_id")
            if already_done:
                parts.append(f"{already_done} already processed")
            if filtered:
                parts.append(f"{filtered} filtered by rating")
            print(f"  {label} [M{machine_id}]: {', '.join(parts)}")
        else:
            msg = f"  {label} [M{machine_id}]: {len(new_eps)} new, {already_done} already done"
            if filtered:
                msg += f", {filtered} filtered"
            print(msg)

    if output_dir:
        # Only folders that will receive episodes must still be v2.1.
        planned = {item["task_name"] for item in plan}
        abort_on_converted_outputs(output_dir, planned)
        for task_name, infos in task_infos.items():
            merged_info = os.path.join(output_dir, task_name, "meta", "info.json")
            if task_name in planned and os.path.exists(merged_info):
                infos.insert(0, (os.path.join(output_dir, task_name), merged_info))
    abort_on_schema_mismatch(task_infos)

    # Collisions among reused run_ids are checked before any ID is issued
    reused = [item for item in plan if item["run_id"] is not None]
    if output_dir:
        abort_on_conflicts(find_conflicts(reused, output_dir, pending))

    if not dry_run and plan:
        try:
            for item in plan:
                if item["run_id"] is None:
                    item["run_id"] = _pending_run_id(pending, item) or next_run_id(item["machine_id"])
                    item["out_idx"] = episode_index_from_run_id(item["run_id"])
                else:
                    total_reuse += 1
        except RunCounterError as exc:
            sys.exit(f"ERROR: {exc}")
        abort_on_conflicts(find_conflicts(plan, output_dir, pending))
        _copy_planned_episodes(plan, output_dir, manifest)

    # Rebuild info.json + episodes.jsonl
    if not dry_run:
        _rebuild_outputs(output_dir, task_sources, plan)

    print()
    if dry_run:
        msg = f"Would process: {total_new} episodes ({total_already} already done)"
        if total_filtered:
            msg += f", {total_filtered} filtered by rating"
        print(msg)
    else:
        print(f"Processed: {total_new} episodes ({total_reuse} reused existing run_id)")
        print(f"Skipped (already processed): {total_already}")
        if total_filtered:
            print(f"Filtered out by rating: {total_filtered}")
        if total_new > 0:
            print(f"Output: {output_dir}")
            counter = read_counter()
            if counter is not None:
                print(f"Counter now at: {counter}")
    if errors:
        sys.exit(f"\n{errors} source(s) skipped because of errors (see above)")


if __name__ == "__main__":
    main()
