# Data Verification

Interactive dashboard for reviewing and rating collected robot demonstration episodes. Ratings follow the 1–5 scoring rubric in [rubric.md](rubric.md).

## Usage

Point it at one collection folder (`<out-dir>/M<machine_id>/` as written by `scripts/collect_demos.py`). Merged datasets (`scripts/merge_annotated_runs.py` output) are not supported; rate the collection folders before merging.

```bash
python -m data_verification demos/ib-p1-bowl-table-v1/M01/

# Start at a specific episode
python -m data_verification demos/ib-p1-bowl-table-v1/M01/ --episode 5

# Tag annotations with your name
python -m data_verification demos/ib-p1-bowl-table-v1/M01/ --annotator alice
```

## Controls

| Key | Action |
|---|---|
| `SPACE` | Play / Pause |
| `+` / `-` | Increase / decrease playback speed (default 2x) |
| `RIGHT` | Next frame |
| `LEFT` | Previous frame |
| `UP` / `DOWN` | Seek +/- 10% of episode |
| `1`-`5` | Rate episode (1=worst, 5=best), advances to the next un-rated episode after it |
| `U` | Undo / clear rating for current episode |
| `N` / `P` | Next / previous episode |
| `R` | Jump to first un-rated episode (resume) |
| `S` | Print summary to terminal |
| `Q` / `ESC` | Quit |

## Annotation Storage

Ratings are saved to `annotations.json` in the dataset root and support resuming across sessions. Every rating is merged into the file under a file lock (re-read, change only the rated episode, atomic replace), so concurrent sessions on the same dataset keep each other's ratings. A rating or undo for an episode that another session changed since this one last read the file is not saved: the terminal names the other rating, and pressing the key again overwrites it. An unreadable file is moved aside to `annotations.json.corrupt-<time>` with a warning rather than overwritten.

```json
{
  "dataset": "demos/ib-p1-bowl-table-v1/M01",
  "task_id": "ib-p1-bowl-table-v1",
  "total_episodes": 9,
  "annotations": {
    "0": {"rating": 5, "annotator": "alice", "timestamp": "2026-04-15T..."},
    "1": {"rating": 2, "annotator": "alice", "timestamp": "2026-04-15T..."},
    "2": null
  }
}
```

A rating that is not an integer from 1 to 5 reads as un-rated, with a warning.

## Module Structure

| File | Purpose |
|---|---|
| `annotation_dashboard.py` | OpenCV-based interactive UI |
| `annotation_store.py` | JSON persistence layer |
| `rubric.md` | Rating rubric |
