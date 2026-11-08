#!/usr/bin/env python3
"""Convert IMBench LeRobot v2.1 datasets to v3.0 format.

Reads every dataset under a root directory (default: ``demos/``) and writes
a v3.0 copy next to it (``<dataset>_v30/``), then optionally replaces the
original in-place.  Directories named ``*_v30`` or ``*_v21_backup`` (outputs
of earlier runs) are skipped.  Intended for merged datasets
(``merge_annotated_runs.py`` output), whose immediate subdirectories are the
per-task datasets; per-machine collector output (``<task>/M<id>/``) must be
merged first.

The v3.0 data shards get contiguous ``episode_index`` (0..N-1, in source
episode order) and dataset-global ``index`` columns regardless of the
values stored in the source files; ``frame_index.<camera>`` columns stay
nullable integers and count frames within the episode's own video file.

Each dataset is built in a temporary sibling directory and renamed into
place.  An existing output is replaced only if it is empty or an earlier
v3.0 output, and an output overlapping the source is refused.  ``--in-place``
refuses to overwrite an existing ``<dataset>_v21_backup`` and is a final
step: merging more episodes into the converted dataset is not supported.

Usage examples::

    # Convert every merged dataset under ~/processed_runs (-> <dataset>_v30)
    python scripts/convert_v21_to_v30.py --demos-dir ~/processed_runs

    # ... into ~/processed_runs_v30/<dataset>
    python scripts/convert_v21_to_v30.py --demos-dir ~/processed_runs --out-dir ~/processed_runs_v30

    # Convert a single dataset (with --dataset, --out-dir is the dataset directory itself)
    python scripts/convert_v21_to_v30.py --dataset ~/processed_runs/ib-p1-edge-slide-v1

    # Convert and replace the original directories in-place
    python scripts/convert_v21_to_v30.py --demos-dir ~/processed_runs --in-place

The output uses the upstream LeRobot v3.0 directory layout.  The parquet keeps
IMBench's flattened per-dimension columns and multi-rate streams, so upstream
``LeRobotDataset`` cannot load it directly.

    <root>/
        meta/
            info.json
            stats.json
            tasks.parquet
            episodes/
                chunk-000/
                    file-000.parquet
        data/
            chunk-000/
                file-000.parquet
        videos/
            <camera_key>/
                chunk-000/
                    file-000.mp4
"""

import argparse
import concurrent.futures
import json
import os
import re
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

import av
from fractions import Fraction
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants (matching upstream lerobot v3.0)
# ---------------------------------------------------------------------------

V21 = "v2.1"
V30 = "v3.0"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DATA_PATH_TEMPLATE = "data/" + CHUNK_FILE_PATTERN + ".parquet"
VIDEO_PATH_TEMPLATE = "videos/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
EPISODES_PATH_TEMPLATE = "meta/episodes/" + CHUNK_FILE_PATTERN + ".parquet"
EPISODES_EXTRA_FILENAME = "meta/episodes_extra.json"

EP_METADATA_FILENAME_RE = re.compile(r"episode_(\d+)_metadata\.json$")
VIDEO_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")  # video keys become path components


class ConvertError(Exception):
    """A dataset the conversion refuses to process; the message says why."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def update_chunk_file_indices(chunk_idx, file_idx):
    """Advance to the next file slot, rolling over to next chunk if needed."""
    if file_idx == DEFAULT_CHUNK_SIZE - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


def get_parquet_size_mb(path):
    """Uncompressed size of a parquet file in MB."""
    meta = pq.read_metadata(str(path))
    total = 0
    for rg in range(meta.num_row_groups):
        rg_meta = meta.row_group(rg)
        for col in range(rg_meta.num_columns):
            total += rg_meta.column(col).total_uncompressed_size
    return total / (1024 ** 2)


def get_file_size_mb(path):
    return Path(path).stat().st_size / (1024 ** 2)


def get_video_duration_s(video_path):
    """Return duration of a video file in seconds using PyAV."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration / av.time_base)


def concatenate_video_files(input_paths, output_path):
    """Concatenate multiple MP4 files, re-encoding to H.264 for browser compatibility.

    Source videos may use codecs (e.g. MPEG-4 Part 2 / mp4v) that browsers and
    HuggingFace's viewer cannot play.  This function decodes every input frame
    and re-encodes the concatenated result as H.264 in an MP4 container with
    the ``faststart`` flag so the file is streamable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    # Probe first file for resolution and frame rate
    with av.open(str(input_paths[0])) as probe:
        in_stream = probe.streams.video[0]
        width = in_stream.codec_context.width
        height = in_stream.codec_context.height
        fps = in_stream.average_rate  # Fraction

    tmp_out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        out_container = av.open(tmp_out_path, mode="w", options={"movflags": "faststart"})
        out_stream = out_container.add_stream("libx264", rate=fps)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "18", "preset": "fast"}

        frame_idx = 0
        time_base = Fraction(1, int(fps))
        for video_path in input_paths:
            in_container = av.open(str(video_path))
            for frame in in_container.decode(video=0):
                # Re-stamp with sequential PTS so concatenation is seamless
                frame.pts = frame_idx
                frame.dts = frame_idx
                frame.time_base = time_base
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)
                frame_idx += 1
            in_container.close()

        # Flush encoder
        for packet in out_stream.encode():
            out_container.mux(packet)
        out_container.close()

        shutil.move(tmp_out_path, str(output_path))
        tmp_out_path = None
    finally:
        if tmp_out_path and os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ---------------------------------------------------------------------------
# Per-episode statistics
# ---------------------------------------------------------------------------

def compute_episode_stats(df, feature_keys):
    """Compute per-feature min/max/mean/std/count from a DataFrame of one episode.

    Returns a flat dict like ``{"stats/<feature>/min": [...], ...}`` ready for
    inclusion in the episodes parquet.
    """
    stats = {}
    for key in feature_keys:
        # Collect all columns belonging to this feature (e.g. action.0 .. action.6)
        cols = [c for c in df.columns if c == key or c.startswith(key + ".")]
        if not cols:
            continue

        # Build a (N, D) array, dropping rows that are all-NaN (sparse multi-rate)
        arr = df[cols].to_numpy(dtype=np.float64, na_value=np.nan)
        mask = ~np.isnan(arr).all(axis=1)
        arr = arr[mask]
        if len(arr) == 0:
            continue

        # Replace remaining NaNs with 0 for stats (shouldn't happen for valid rows)
        arr = np.nan_to_num(arr, nan=0.0)

        stats[f"stats/{key}/min"] = arr.min(axis=0).tolist()
        stats[f"stats/{key}/max"] = arr.max(axis=0).tolist()
        stats[f"stats/{key}/mean"] = arr.mean(axis=0).tolist()
        stats[f"stats/{key}/std"] = arr.std(axis=0).tolist()
        stats[f"stats/{key}/count"] = [int(len(arr))]

    return stats


def aggregate_episode_stats(all_ep_stats, feature_keys):
    """Aggregate per-episode stats into global stats (weighted by count)."""
    global_stats = {}
    for key in feature_keys:
        mins, maxs, means, stds, counts = [], [], [], [], []
        for ep_stats in all_ep_stats:
            prefix = f"stats/{key}"
            if f"{prefix}/min" not in ep_stats:
                continue
            mins.append(np.array(ep_stats[f"{prefix}/min"]))
            maxs.append(np.array(ep_stats[f"{prefix}/max"]))
            means.append(np.array(ep_stats[f"{prefix}/mean"]))
            stds.append(np.array(ep_stats[f"{prefix}/std"]))
            counts.append(np.array(ep_stats[f"{prefix}/count"]))

        if not mins:
            continue

        # Skip features whose shape varies across episodes (e.g. variable-length
        # object_state when the number of objects differs per episode).
        shapes = {np.array(m).shape for m in mins}
        if len(shapes) > 1:
            print(f"  [stats] skipping {key}: inconsistent shapes {shapes}")
            continue

        mins = np.stack(mins)
        maxs = np.stack(maxs)
        means = np.stack(means)
        stds = np.stack(stds)
        counts = np.stack(counts)

        total_count = counts.sum(axis=0)
        global_mean = (means * counts).sum(axis=0) / np.maximum(total_count, 1)
        # Parallel variance formula
        global_var = ((stds ** 2 + (means - global_mean) ** 2) * counts).sum(axis=0) / np.maximum(total_count, 1)

        global_stats[key] = {
            "min": mins.min(axis=0).tolist(),
            "max": maxs.max(axis=0).tolist(),
            "mean": global_mean.tolist(),
            "std": np.sqrt(global_var).tolist(),
            "count": total_count.tolist(),
        }

    return global_stats


# ---------------------------------------------------------------------------
# Load v2.1 metadata
# ---------------------------------------------------------------------------

def load_v21_info(root):
    """Load ``meta/info.json``; raise ConvertError on a video key that is not a plain name."""
    with open(root / "meta" / "info.json") as f:
        info = json.load(f)
    for key, ft in info.get("features", {}).items():
        if ft.get("dtype") == "video" and (not VIDEO_KEY_RE.match(key) or ".." in key):
            raise ConvertError(f"{root}: invalid video feature key {key!r} in meta/info.json")
    return info


def load_v21_episodes(root):
    episodes = []
    with open(root / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return sorted(episodes, key=lambda x: x["episode_index"])


def load_v21_tasks(root):
    tasks = []
    with open(root / "meta" / "tasks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return sorted(tasks, key=lambda x: x["task_index"])


# ---------------------------------------------------------------------------
# Conversion steps
# ---------------------------------------------------------------------------

def convert_tasks(root, new_root):
    """Convert tasks.jsonl -> tasks.parquet."""
    tasks = load_v21_tasks(root)
    task_strings = [t["task"] for t in tasks]
    task_indices = [t["task_index"] for t in tasks]
    df = pd.DataFrame({"task_index": task_indices}, index=pd.Index(task_strings, name="task"))
    path = new_root / "meta" / "tasks.parquet"
    _ensure_dir(path.parent)
    df.to_parquet(path)
    print(f"  [tasks] wrote {len(tasks)} tasks to {path.relative_to(new_root)}")


def convert_data(root, new_root, data_file_size_mb, video_frame_offsets,
                 valid_episodes=None):
    """Concatenate per-episode parquet files into multi-episode shards.

    Parameters
    ----------
    video_frame_offsets : dict[str, list[int]]
        For each ``frame_index.<cam>`` column, a list with one entry per
        episode giving the index of its first frame within its own video
        file.  E.g. ``{"frame_index.rgb_ego_cam": [0, 2273, 0]}`` means
        episode 1's frame indices must be shifted by 2273 and episode 2
        starts a new file.
    valid_episodes : list[int], optional
        If provided, only episodes with these indices are converted.
        Incomplete episodes are silently skipped.

    Returns a list of episode metadata dicts with data chunk/file indices and
    dataset_from/to_index offsets.
    """
    data_dir = root / "data"
    ep_paths = _episode_files(data_dir.glob("*/*.parquet"), valid_episodes)
    if not ep_paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    chunk_idx = 0
    file_idx = 0
    size_mb = 0
    num_frames = 0
    paths_to_cat = []
    episodes_metadata = []

    def _flush(paths, ep_start_idx, ci, fi):
        dfs = []
        for i, p in enumerate(paths):
            df = pd.read_parquet(str(p))
            ep_i = ep_start_idx + i
            # Source files carry their v2.1 (or pre-merge) indices; renumber
            # to the v3.0 episode and the dataset-global frame index.
            ep_meta = episodes_metadata[ep_i]
            df["episode_index"] = np.full(len(df), ep_meta["episode_index"], dtype=np.int64)
            df["index"] = np.arange(
                ep_meta["dataset_from_index"], ep_meta["dataset_to_index"], dtype=np.int64,
            )
            # Nullable integers: frames are null on steps without a video frame
            for fi_col in (c for c in df.columns if c.startswith("frame_index.")):
                df[fi_col] = df[fi_col].astype("Int64")
            # Offset frame_index columns so they point into the episode's video file
            for fi_col, offsets in video_frame_offsets.items():
                if fi_col in df.columns and ep_i < len(offsets):
                    offset = offsets[ep_i]
                    if offset != 0:
                        df[fi_col] = df[fi_col] + offset
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        out = new_root / DATA_PATH_TEMPLATE.format(chunk_index=ci, file_index=fi)
        _ensure_dir(out.parent)
        combined.to_parquet(str(out), index=False)

    flush_ep_start = 0
    for ep_path in tqdm(ep_paths, desc="  [data] processing", unit="ep"):
        ep_size = get_parquet_size_mb(ep_path)
        ep_nrows = pq.read_metadata(str(ep_path)).num_rows

        # Flush if adding this episode would exceed the file size limit
        if size_mb + ep_size >= data_file_size_mb and len(paths_to_cat) > 0:
            _flush(paths_to_cat, flush_ep_start, chunk_idx, file_idx)
            flush_ep_start += len(paths_to_cat)
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx)
            size_mb = 0
            paths_to_cat = []

        ep_idx = len(episodes_metadata)
        episodes_metadata.append({
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_nrows,
        })

        paths_to_cat.append(ep_path)
        size_mb += ep_size
        num_frames += ep_nrows

    # Flush remaining
    if paths_to_cat:
        _flush(paths_to_cat, flush_ep_start, chunk_idx, file_idx)

    print(f"  [data] aggregated {len(ep_paths)} episodes into "
          f"{chunk_idx * DEFAULT_CHUNK_SIZE + file_idx + 1} file(s), "
          f"{num_frames} total frames")
    return episodes_metadata


def convert_videos(root, new_root, video_file_size_mb, valid_episodes=None,
                   max_workers=1):
    """Concatenate per-episode MP4 files per camera into multi-episode shards.

    Returns
    -------
    episodes_metadata : list[dict] | None
        Per-episode dicts with video chunk/file indices and from/to timestamps.
    frame_offsets : dict[str, list[int]]
        For each ``frame_index.<cam>`` column, the index of each episode's first
        frame within its own video file.  Used by ``convert_data`` to shift
        frame indices.
    """
    info = load_v21_info(root)
    features = info.get("features", {})
    video_keys = sorted(k for k, ft in features.items() if ft.get("dtype") == "video")

    declared = {k.replace("observation.", "") for k in video_keys}
    on_disk = {p.name for p in (root / "videos").glob("*/*") if p.is_dir()}
    if on_disk - declared:
        print(f"  [warn] videos/ has camera(s) not declared in meta/info.json, "
              f"not converted: {', '.join(sorted(on_disk - declared))}")
    if not video_keys:
        return None, {}

    all_cam_metadata = []
    frame_offsets = {}  # "frame_index.<cam_label>" -> per-episode offset in its file
    for video_key in video_keys:
        cam_meta, ep_frame_offsets = _convert_videos_of_camera(
            root, new_root, video_key, video_file_size_mb,
            valid_episodes=valid_episodes,
            max_workers=max_workers,
        )
        all_cam_metadata.append(cam_meta)
        cam_label = video_key.replace("observation.", "")
        frame_offsets[f"frame_index.{cam_label}"] = ep_frame_offsets

    # Merge per-camera metadata into unified per-episode dicts
    num_episodes = len(all_cam_metadata[0])
    merged = []
    for ep_idx in range(num_episodes):
        ep_dict = {"episode_index": ep_idx}
        for cam_meta in all_cam_metadata:
            for k, v in cam_meta[ep_idx].items():
                if k != "episode_index":
                    ep_dict[k] = v
        merged.append(ep_dict)

    return merged, frame_offsets


def _convert_videos_of_camera(root, new_root, video_key, video_file_size_mb,
                              valid_episodes=None, max_workers=1):
    """Convert videos for a single camera key.

    Returns
    -------
    episodes_metadata : list[dict]
    ep_frame_offsets : list[int]
        Index of each episode's first frame within its output video file.
    """
    # v2.1 layout: videos/chunk-XXX/<camera_label>/episode_NNNNNN.mp4
    # The feature key is e.g. "observation.rgb_ego_cam" but the directory
    # uses just the camera label "rgb_ego_cam".
    cam_label = video_key.replace("observation.", "")
    videos_dir = root / "videos"
    ep_paths = _episode_files(videos_dir.glob(f"*/{cam_label}/*.mp4"), valid_episodes)
    if not ep_paths:
        raise FileNotFoundError(f"No video files for {video_key} in {videos_dir}")

    chunk_idx = 0
    file_idx = 0
    size_mb = 0
    duration_s = 0.0
    num_frames = 0  # frames queued for the current output file
    paths_to_cat = []
    episodes_metadata = []
    ep_frame_offsets = []

    # Collect all flush jobs: (input_paths, output_path) tuples
    flush_jobs = []
    # Track which metadata entries belong to each flush job
    flush_meta_ranges = []  # (start_idx, end_idx) into episodes_metadata

    for ep_path in tqdm(ep_paths, desc=f"  [video:{video_key}] scanning", unit="ep"):
        ep_size = get_file_size_mb(ep_path)
        ep_dur = get_video_duration_s(ep_path)
        # Count frames in this episode's video
        with av.open(str(ep_path)) as c:
            ep_nframes = c.streams.video[0].frames

        # Queue flush if adding this episode would exceed the file size limit
        if size_mb + ep_size >= video_file_size_mb and len(paths_to_cat) > 0:
            out = new_root / VIDEO_PATH_TEMPLATE.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            flush_jobs.append((list(paths_to_cat), out))
            start = len(episodes_metadata) - len(paths_to_cat)
            flush_meta_ranges.append((start, len(episodes_metadata), chunk_idx, file_idx))
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx)
            size_mb = 0
            duration_s = 0.0
            num_frames = 0
            paths_to_cat = []

        ep_idx = len(episodes_metadata)
        episodes_metadata.append({
            "episode_index": ep_idx,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": duration_s,
            f"videos/{video_key}/to_timestamp": duration_s + ep_dur,
        })
        ep_frame_offsets.append(num_frames)

        paths_to_cat.append(ep_path)
        size_mb += ep_size
        duration_s += ep_dur
        num_frames += ep_nframes

    # Queue remaining
    if paths_to_cat:
        out = new_root / VIDEO_PATH_TEMPLATE.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        flush_jobs.append((list(paths_to_cat), out))
        start = len(episodes_metadata) - len(paths_to_cat)
        flush_meta_ranges.append((start, len(episodes_metadata), chunk_idx, file_idx))

    # Update metadata with final chunk/file indices for each flush group
    for start, end, ci, fi in flush_meta_ranges:
        for meta in episodes_metadata[start:end]:
            meta[f"videos/{video_key}/chunk_index"] = ci
            meta[f"videos/{video_key}/file_index"] = fi

    # Execute video concatenation jobs in parallel
    if max_workers > 1 and len(flush_jobs) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(concatenate_video_files, paths, out): i
                for i, (paths, out) in enumerate(flush_jobs)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"  [video:{video_key}] encoding",
                unit="shard",
            ):
                future.result()  # raise any exceptions
    else:
        for paths, out in tqdm(flush_jobs, desc=f"  [video:{video_key}] encoding", unit="shard"):
            concatenate_video_files(paths, out)

    print(f"  [video:{video_key}] aggregated {len(ep_paths)} episodes into "
          f"{chunk_idx * DEFAULT_CHUNK_SIZE + file_idx + 1} file(s)")
    return episodes_metadata, ep_frame_offsets


def convert_episodes_metadata(root, new_root, data_metadata, video_metadata,
                              valid_episodes=None):
    """Build the meta/episodes/ chunked parquet with offsets, tasks, and per-episode stats.

    Also writes meta/stats.json with global aggregated statistics.
    """
    v21_episodes = load_v21_episodes(root)
    # A repeated episodes.jsonl line would misalign the per-episode lists below
    by_index = {ep["episode_index"]: ep for ep in v21_episodes}
    if len(by_index) < len(v21_episodes):
        print(f"  [warn] episodes.jsonl repeats {len(v21_episodes) - len(by_index)} "
              f"line(s); using the last entry per episode")
        v21_episodes = [by_index[i] for i in sorted(by_index)]
    if valid_episodes is not None:
        valid_set = set(valid_episodes)
        v21_episodes = [ep for ep in v21_episodes if ep["episode_index"] in valid_set]
    info = load_v21_info(root)
    features = info.get("features", {})

    # Determine which feature keys to compute stats for (non-video, non-metadata)
    stat_keys = []
    for key, ft in features.items():
        if ft.get("dtype") == "video":
            continue
        if key.startswith("frame_index."):
            continue
        stat_keys.append(key)

    # Read episode parquet files for per-episode stats
    ep_paths = _episode_files((root / "data").glob("*/*.parquet"), valid_episodes)
    all_ep_stats = []
    for ep_path in tqdm(ep_paths, desc="  [episodes] computing stats", unit="ep"):
        df = pd.read_parquet(str(ep_path))
        ep_stats = compute_episode_stats(df, stat_keys)
        all_ep_stats.append(ep_stats)

    # v21_episodes, data_metadata, video_metadata and all_ep_stats hold one
    # entry per valid episode, each in source episode order.
    records = []
    for i, ep_legacy in enumerate(v21_episodes):
        rec = {}
        # Data offsets
        rec.update(data_metadata[i])
        # Video offsets
        if video_metadata is not None:
            for k, v in video_metadata[i].items():
                if k != "episode_index":
                    rec[k] = v
        # Legacy metadata
        rec["length"] = ep_legacy["length"]
        tasks_list = ep_legacy.get("tasks") or [ep_legacy.get("task", "")]
        rec["tasks"] = tasks_list
        # Episode stats
        rec.update(all_ep_stats[i])
        # Episodes chunk (all in one file for our scale)
        rec["meta/episodes/chunk_index"] = 0
        rec["meta/episodes/file_index"] = 0
        records.append(rec)

    df_episodes = pd.DataFrame(records)
    ep_out = new_root / EPISODES_PATH_TEMPLATE.format(chunk_index=0, file_index=0)
    _ensure_dir(ep_out.parent)
    df_episodes.to_parquet(str(ep_out), index=False)
    print(f"  [episodes] wrote {len(records)} episode records to "
          f"{ep_out.relative_to(new_root)}")

    # Global stats
    global_stats = aggregate_episode_stats(all_ep_stats, stat_keys)
    stats_path = new_root / "meta" / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(global_stats, f, indent=2)
    print(f"  [stats] wrote global stats for {len(global_stats)} features")


def convert_episode_extras(root, new_root, valid_episodes):
    """Preserve IMBench-specific v2.1 per-episode metadata as a sidecar JSON.

    The v3.0 standard ``meta/episodes/`` parquet only stores LeRobot-defined
    fields (slice locators, stats, length, tasks).  Project-specific fields
    from ``meta/episode_NNNNNN_metadata.json`` (run_id, teleoperator_id,
    outcome, annotation_rating, sensors map, etc.) have no home in that
    schema, so we consolidate every original JSON into a single file at
    ``meta/episodes_extra.json``, preserving the original nested structure
    and adding two keys per record:

    - ``episode_index``         — v3.0 index (join key for the canonical parquet)
    - ``original_episode_index`` — original v2.1 collection id

    Records are emitted in v3.0 episode_index order, with orphans (metadata
    whose original id was filtered out by ``_find_complete_episodes``)
    appended at the end with ``episode_index: null``.
    """
    src_meta = Path(root) / "meta"
    json_paths = sorted(src_meta.glob("episode_*_metadata.json"))
    if not json_paths:
        print("  [extras] no episode_*_metadata.json files found, skipping sidecar")
        return

    # Map original v2.1 id -> v3.0 episode_index using the same ordering the
    # rest of the conversion uses (sorted ascending by original id).
    orig_to_v30 = {orig: v30 for v30, orig in enumerate(valid_episodes)}

    mapped, orphans = [], []
    for path in json_paths:
        m = EP_METADATA_FILENAME_RE.search(path.name)
        if not m:
            continue
        orig_id = int(m.group(1))
        with path.open() as fh:
            raw = json.load(fh)

        v30_idx = orig_to_v30.get(orig_id)
        # Build record with identifiers first, then original fields verbatim
        record = {"episode_index": v30_idx, "original_episode_index": orig_id}
        record.update(raw)

        if v30_idx is not None:
            mapped.append((v30_idx, record))
        else:
            orphans.append(record)

    mapped.sort(key=lambda x: x[0])
    records = [r for _, r in mapped] + orphans

    # Verify mapping by cross-checking length: v2.1 dataset_summary.action[0]
    # should equal the v3.0 episodes parquet's length column.
    canon_path = Path(new_root) / EPISODES_PATH_TEMPLATE.format(chunk_index=0, file_index=0)
    mismatches = 0
    if canon_path.exists():
        canon = pd.read_parquet(canon_path, columns=["episode_index", "length"])
        canon_len = dict(zip(canon["episode_index"].tolist(), canon["length"].tolist()))
        for rec in records:
            v30_idx = rec["episode_index"]
            if v30_idx is None:
                continue
            action_shape = (rec.get("dataset_summary") or {}).get("action")
            if not action_shape:
                continue
            v21_len = int(action_shape[0])
            v30_len = canon_len.get(int(v30_idx))
            if v30_len is not None and v21_len != v30_len:
                if mismatches < 5:
                    print(f"  [extras] WARN length mismatch ep {v30_idx} "
                          f"(orig {rec['original_episode_index']}): "
                          f"v21={v21_len} v30={v30_len}")
                mismatches += 1
        if mismatches:
            print(f"  [extras] WARN: {mismatches} length mismatches — order assumption may be wrong")

    out = Path(new_root) / EPISODES_EXTRA_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(records, f, indent=2)
    print(f"  [extras] wrote {len(records)} v2.1 metadata records "
          f"({len(mapped)} mapped, {len(orphans)} orphan) to {out.relative_to(new_root)}")


def convert_info(root, new_root, total_episodes, total_frames):
    """Write the v3.0 info.json; the totals count the converted episodes and frames."""
    info = load_v21_info(root)

    info["codebase_version"] = V30
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames

    # Remove v2.1-only fields if present
    info.pop("total_chunks", None)
    info.pop("total_videos", None)

    # Add v3.0 fields
    info["data_files_size_in_mb"] = DEFAULT_DATA_FILE_SIZE_IN_MB
    info["video_files_size_in_mb"] = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    info["data_path"] = DATA_PATH_TEMPLATE
    info["fps"] = int(info.get("fps", info.get("control_freq", 30)))

    # Set video_path template (or None if no videos)
    features = info.get("features", {})
    has_video = any(ft.get("dtype") == "video" for ft in features.values())
    info["video_path"] = VIDEO_PATH_TEMPLATE if has_video else None

    # Update feature fps and video paths
    for key, ft in features.items():
        if ft.get("dtype") == "video":
            # Update video_path template to v3.0 style
            ft["video_path"] = VIDEO_PATH_TEMPLATE.replace("{video_key}", key)
        # Ensure every feature has an fps field
        if "fps" not in ft:
            ft["fps"] = info["fps"]

    out = new_root / "meta" / "info.json"
    _ensure_dir(out.parent)
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  [info] wrote {out.relative_to(new_root)}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def _find_complete_episodes(src_root):
    """Return the sorted set of episode indices that have both parquet and all video files.

    Incomplete episodes (e.g. aborted mid-recording) are detected and skipped
    so downstream conversion stages always see consistent counts.
    """
    info = load_v21_info(src_root)
    features = info.get("features", {})
    video_keys = sorted(k for k, ft in features.items() if ft.get("dtype") == "video")

    # Episodes with readable parquet data
    data_dir = src_root / "data"
    parquet_eps = set()
    corrupt_eps = set()
    for p in data_dir.glob("*/*.parquet"):
        stem = p.stem  # "episode_000003"
        try:
            ep_idx = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            pq.read_metadata(str(p))
            parquet_eps.add(ep_idx)
        except Exception:
            print(f"  [warn] unreadable parquet, skipping episode {ep_idx}: {p}")
            corrupt_eps.add(ep_idx)

    # Episodes with video files (must have all cameras and be readable)
    video_eps_per_cam = []
    videos_dir = src_root / "videos"
    for video_key in video_keys:
        cam_label = video_key.replace("observation.", "")
        cam_eps = set()
        for p in videos_dir.glob(f"*/{cam_label}/*.mp4"):
            try:
                ep_idx = int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            # Verify the video file is readable
            try:
                with av.open(str(p)) as c:
                    c.streams.video[0]
                cam_eps.add(ep_idx)
            except Exception:
                print(f"  [warn] corrupt video, skipping episode {ep_idx}: {p}")
                corrupt_eps.add(ep_idx)
        video_eps_per_cam.append(cam_eps)

    # Episodes listed in episodes.jsonl
    jsonl_eps = set()
    for ep in load_v21_episodes(src_root):
        jsonl_eps.add(ep["episode_index"])

    # An episode is complete only if it appears in all sources and has no corrupt files
    complete = parquet_eps & jsonl_eps
    for cam_eps in video_eps_per_cam:
        complete &= cam_eps
    complete -= corrupt_eps

    skipped = (parquet_eps | jsonl_eps) - complete
    if skipped:
        print(f"  [warn] skipping {len(skipped)} incomplete episode(s): "
              f"{sorted(skipped)}")

    return sorted(complete)


def _episode_files(paths, valid_indices=None):
    """Return the ``episode_<index>`` files among *paths* in source episode order.

    With *valid_indices*, only those episodes are kept.  Ordering by the parsed
    index rather than by path keeps the per-episode lists aligned with
    ``episodes.jsonl`` when a dataset spans several ``chunk-NNN`` directories.
    """
    valid = None if valid_indices is None else set(valid_indices)
    indexed = []
    for p in paths:
        try:
            ep_idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if valid is None or ep_idx in valid:
            indexed.append((ep_idx, p))
    return [p for _, p in sorted(indexed)]


def _is_v30_output(path):
    """True if *path* holds an earlier conversion output: v3.0 meta/, data/ and videos/ only."""
    try:
        info = json.loads((path / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return False
    return (isinstance(info, dict) and info.get("codebase_version") == V30
            and all(p.name in ("meta", "data", "videos") for p in path.iterdir()))


def _check_destination(src_root, dst_root):
    """Raise ConvertError unless the v3.0 output may be written to *dst_root*.

    *dst_root* must not be, contain or lie inside *src_root*.  If it exists it
    must be an empty directory or an earlier v3.0 output, which is replaced.
    """
    src, dst = src_root.resolve(), dst_root.resolve()
    if dst == src or dst in src.parents or src in dst.parents:
        raise ConvertError(f"output {dst_root} overlaps the source dataset {src_root}")
    if not os.path.lexists(dst_root):
        return
    if dst_root.is_dir() and not dst_root.is_symlink() and (
            not any(dst_root.iterdir()) or _is_v30_output(dst_root)):
        return
    raise ConvertError(f"output {dst_root} exists and is not an earlier v3.0 output; "
                       f"remove it or choose another output directory")


def _check_not_converted(src_root):
    """Raise ConvertError if the v2.1 dataset *src_root* also holds v3.0 data shards.

    Merging episodes into a dataset converted with ``--in-place`` leaves this
    mix; converting it would keep only the merged episodes.
    """
    shard = next((src_root / "data").glob("*/file-*.parquet"), None)
    if shard is not None:
        raise ConvertError(
            f"{src_root} has a v2.1 meta/info.json but also v3.0 data shards "
            f"({shard.relative_to(src_root)}): episodes were merged into a converted "
            f"dataset. Rebuild a v2.1 dataset (e.g. from {src_root.name}_v21_backup "
            f"plus the new episode files) and convert that instead")


def convert_dataset(src_root, dst_root=None, in_place=False, max_workers=1):
    """Convert a single v2.1 dataset to v3.0.

    The output is built in a temporary sibling directory and renamed into
    place, so a failed conversion leaves no partial output.

    Parameters
    ----------
    src_root : Path
        Path to the existing v2.1 dataset.
    dst_root : Path, optional
        Where to write the v3.0 dataset. Defaults to ``<src_root>_v30``.  It
        may not overlap *src_root*; an existing directory is replaced only if
        it is empty or an earlier v3.0 output.  Not used with *in_place*.
    in_place : bool
        If True, move the original to ``<src_root>_v21_backup`` and put the
        v3.0 output at *src_root*.  Refused if that backup already exists.
    max_workers : int
        Number of parallel workers for video encoding.

    Returns False if the dataset is skipped (not v2.1, or no complete
    episodes); raises ConvertError if the conversion is refused.
    """
    src_root = Path(src_root)
    if in_place and dst_root is not None:
        raise ValueError("dst_root cannot be combined with in_place")
    if dst_root is None:
        dst_root = src_root.parent / f"{src_root.name}_v30"
    dst_root = Path(dst_root)
    backup = src_root.parent / f"{src_root.name}_v21_backup"

    # Validate source is v2.1
    info = load_v21_info(src_root)
    version = info.get("codebase_version", "unknown")
    if version != V21:
        print(f"  [skip] {src_root.name}: codebase_version is '{version}', expected '{V21}'")
        return False

    _check_not_converted(src_root)
    if not in_place:
        _check_destination(src_root, dst_root)
    elif os.path.lexists(backup):
        raise ConvertError(f"{backup} already exists and may hold the only v2.1 copy of "
                           f"earlier episodes; move it away before converting {src_root} in place")

    print(f"\nConverting {src_root} " + ("in place" if in_place else f"-> {dst_root}"))

    # Determine which episodes are complete before converting anything
    valid_episodes = _find_complete_episodes(src_root)
    if not valid_episodes:
        print(f"  [skip] no complete episodes found in {src_root}")
        return False

    final = src_root if in_place else dst_root
    final.parent.mkdir(parents=True, exist_ok=True)
    work = final.parent / f".{final.name}.tmp-{secrets.token_hex(4)}"
    work.mkdir()
    try:
        convert_tasks(src_root, work)
        # Convert videos first so we know per-episode frame counts for offset correction
        video_metadata, frame_offsets = convert_videos(
            src_root, work, DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            valid_episodes=valid_episodes,
            max_workers=max_workers,
        )
        data_metadata = convert_data(
            src_root, work, DEFAULT_DATA_FILE_SIZE_IN_MB, frame_offsets,
            valid_episodes=valid_episodes,
        )
        convert_info(src_root, work, total_episodes=len(data_metadata),
                     total_frames=data_metadata[-1]["dataset_to_index"])
        convert_episodes_metadata(
            src_root, work, data_metadata, video_metadata,
            valid_episodes=valid_episodes,
        )
        convert_episode_extras(src_root, work, valid_episodes)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    if in_place:
        os.rename(src_root, backup)
        os.rename(work, src_root)
        print(f"  [in-place] original backed up to {backup.name}, v3.0 is now at {src_root}")
    else:
        if os.path.lexists(dst_root):
            # Empty or an earlier v3.0 output (checked above): swap, then delete it
            stale = work.with_name(f"{work.name}-old")
            os.rename(dst_root, stale)
            os.rename(work, dst_root)
            shutil.rmtree(stale)
        else:
            os.rename(work, dst_root)
        print(f"  [done] v3.0 dataset written to {dst_root}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert IMBench LeRobot v2.1 datasets to v3.0 format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--demos-dir",
        type=str,
        default="demos",
        help="Root directory containing merged dataset subdirectories (default: demos/)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a single dataset to convert (overrides --demos-dir)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="With --dataset: the v3.0 dataset directory. With --demos-dir: the "
             "parent directory, each dataset goes to <out-dir>/<name>. "
             "Defaults to <dataset>_v30 next to each source. An existing output "
             "is replaced only if it is empty or an earlier v3.0 output.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the original v2.1 directory with the v3.0 output "
             "(backs up original to <dataset>_v21_backup; refused if that exists).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for video encoding. "
             "Defaults to the number of CPU cores.",
    )

    args = parser.parse_args()
    if args.in_place and args.out_dir:
        parser.error("--in-place and --out-dir cannot be combined")
    max_workers = args.workers or os.cpu_count() or 1

    if args.dataset:
        # Convert a single dataset
        dst = Path(args.out_dir) if args.out_dir else None
        try:
            ok = convert_dataset(args.dataset, dst, args.in_place,
                                 max_workers=max_workers)
        except ConvertError as exc:
            sys.exit(f"Error: {exc}")
        if not ok:
            sys.exit(1)
    else:
        # Convert all datasets under demos_dir
        demos_dir = Path(args.demos_dir)
        if not demos_dir.is_dir():
            print(f"Error: demos directory not found: {demos_dir}")
            sys.exit(1)

        # Skip outputs of earlier runs: <ds>_v30, the --in-place backups and
        # the hidden work directories of interrupted runs
        datasets = sorted(
            d for d in demos_dir.iterdir()
            if d.is_dir() and (d / "meta" / "info.json").exists()
            and not d.name.startswith(".")
            and not d.name.endswith(("_v30", "_v21_backup"))
        )
        # Per-machine collector output (<task>/M<id>/) has to be merged first
        unmerged = sorted(d.name for d in demos_dir.iterdir()
                          if d.is_dir() and any(d.glob("M[0-9]*/meta/info.json")))
        if unmerged:
            print(f"Note: {', '.join(unmerged)} in {demos_dir}/ hold per-machine collection "
                  f"runs (<task>/M<id>/); merge them with scripts/merge_annotated_runs.py "
                  f"and convert the merged output.")
        if not datasets:
            print(f"No v2.1 datasets found in {demos_dir}/")
            sys.exit(1)

        print(f"Found {len(datasets)} dataset(s) in {demos_dir}/ "
              f"(using {max_workers} worker(s))")
        converted = 0
        failed = []
        for ds_path in tqdm(datasets, desc="Datasets", unit="ds"):
            dst = None
            if args.out_dir:
                dst = Path(args.out_dir) / ds_path.name
            try:
                ok = convert_dataset(ds_path, dst, args.in_place,
                                     max_workers=max_workers)
            except Exception as exc:
                print(f"  [error] {ds_path.name} not converted: {type(exc).__name__}: {exc}")
                failed.append(ds_path.name)
                continue
            if ok:
                converted += 1

        print(f"\n{'=' * 50}")
        print(f"Converted {converted}/{len(datasets)} dataset(s) to v3.0")
        if failed:
            print(f"Failed: {', '.join(failed)}")
            sys.exit(1)


if __name__ == "__main__":
    main()
