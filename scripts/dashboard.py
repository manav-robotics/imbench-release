#!/usr/bin/env python3
"""IMBench Data Collection Dashboard — Web GUI.

Lightweight Flask app for episode labeling and session configuration.
Reads the meta/episode_*_metadata.json files written by collect_demos.py and
updates them when the operator applies a label.

Session config is written to ~/.imbench/session.json (private to the user).
collect_demos.py reads it for every saved episode, so teleoperator_id,
task_type, supervisor_id, etc. are pre-filled in the episode's metadata.
The master config (label codes and crew names) is edited in
~/.imbench/master_config.json, created from the tracked defaults in
scripts/master_config.json on first start.

The server binds to 127.0.0.1 by default and has no authentication; pass
``--host 0.0.0.0`` only on a trusted network.

Usage:
    python scripts/dashboard.py                       # default port 5050
    python scripts/dashboard.py --port 8080
    python scripts/dashboard.py --data-dir demos/ib-p1-edge-slide-v1
"""

import argparse
import copy
import glob
import json
import math
import os
import statistics
import sys
import time

from flask import Flask, jsonify, render_template, request

from imbench.utils.fileio import atomic_write_json
from imbench.utils.recording import session_config_path
from imbench.utils.run_id import imbench_dir

app = Flask(__name__, template_folder="templates")

DATA_DIR = ""
SESSION_CONFIG_PATH = str(session_config_path())

# Host names a loopback-bound server answers to (see _reject_foreign_requests).
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
BIND_HOST = "127.0.0.1"

# ── Episode cache (avoid re-scanning on every poll) ──────────────────
_episodes_cache: list = []
_episodes_cache_time: float = 0.0
_EPISODES_CACHE_TTL: float = 3.0  # seconds


def _find_episodes_cached() -> list:
    global _episodes_cache, _episodes_cache_time
    now = time.monotonic()
    if now - _episodes_cache_time < _EPISODES_CACHE_TTL:
        return _episodes_cache
    _episodes_cache = _find_episodes(DATA_DIR)
    _episodes_cache_time = now
    return _episodes_cache


def _invalidate_cache():
    global _episodes_cache_time
    _episodes_cache_time = 0.0


MASTER_CONFIG_PATH = ""  # ~/.imbench/master_config.json, set in main()
# Tracked defaults the private master config is created from; the dashboard never writes it.
_MASTER_DEFAULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_config.json")

# ── Default master config ───────────────────────────────────────────

_DEFAULT_MASTER = {
    "tasks": {},
    "crew": {
        "operators": [
            {"id": "OP-001", "name": ""},
            {"id": "OP-002", "name": ""},
            {"id": "OP-003", "name": ""},
            {"id": "OP-004", "name": ""},
            {"id": "OP-005", "name": ""},
            {"id": "OP-006", "name": ""},
            {"id": "OP-007", "name": ""},
            {"id": "OP-008", "name": ""},
        ],
        "safety_watchers": [
            {"id": "SF-001", "name": ""},
            {"id": "SF-002", "name": ""},
            {"id": "SF-003", "name": ""},
            {"id": "SF-004", "name": ""},
            {"id": "SF-005", "name": ""},
            {"id": "SF-006", "name": ""},
            {"id": "SF-007", "name": ""},
            {"id": "SF-008", "name": ""},
        ],
        "supervisors": [
            {"id": "SV-001", "name": ""},
            {"id": "SV-002", "name": ""},
            {"id": "SV-003", "name": ""},
            {"id": "SV-004", "name": ""},
        ],
    },
}


# Label codes for tasks the master config does not list.
_DEFAULT_LABEL_CODES = {"1": "success", "2": "partial - partial_completion", "3": "failure - task_failure"}


def _read_master(path: str) -> dict | None:
    try:
        with open(path) as f:
            config = json.load(f)
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _load_master() -> dict:
    return _read_master(MASTER_CONFIG_PATH) or copy.deepcopy(_DEFAULT_MASTER)


def _save_master(config: dict):
    """Write the master config; it holds crew names, so it is private to the user."""
    os.makedirs(os.path.dirname(MASTER_CONFIG_PATH), mode=0o700, exist_ok=True)
    atomic_write_json(MASTER_CONFIG_PATH, config, indent=4, mode=0o600)


def _master_shape_error(config: dict):
    """Describe why *config* is not a usable master config, or return None."""
    tasks, crew = config.get("tasks", {}), config.get("crew", {})
    if not isinstance(tasks, dict) or not all(isinstance(t, dict) for t in tasks.values()):
        return "tasks must map task ids to objects"
    if not isinstance(crew, dict) or not all(
            isinstance(people, list) and all(isinstance(p, dict) for p in people)
            for people in crew.values()):
        return "crew must map roles to lists of objects"
    return None


def _get_label_codes_for_task(task_id: str) -> dict:
    tasks = _load_master().get("tasks")
    task = tasks.get(task_id) if isinstance(tasks, dict) else None
    if isinstance(task, dict) and isinstance(task.get("label_codes"), dict) and task["label_codes"]:
        return task["label_codes"]
    return dict(_DEFAULT_LABEL_CODES)


# ── Episode discovery ───────────────────────────────────────────────

_skipped_meta: set = set()


def _read_meta(meta_path: str) -> dict | None:
    """Return a metadata file's JSON object, or None (warning once per file) if unreadable."""
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        if meta_path not in _skipped_meta:
            _skipped_meta.add(meta_path)
            print(f"[dashboard] skipping {meta_path}: {exc}", file=sys.stderr)
        return None
    return meta


def _text(value) -> str:
    return "" if value is None else str(value)


def _number(value):
    """*value* if it is an int or a finite float (not a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if isinstance(value, int) or math.isfinite(value) else None


def _rating(value) -> int | None:
    """*value* as an int if it is a 1-5 annotation rating, else None."""
    return int(value) if _number(value) is not None and value in range(1, 6) else None


def _n_steps(meta: dict) -> int:
    """Episode length from the action shape in ``dataset_summary``, or 0."""
    summary = meta.get("dataset_summary")
    if not isinstance(summary, dict):
        return 0
    shapes = [summary.get("action")] + [v for k, v in summary.items() if "action" in k]
    for shape in shapes:
        if isinstance(shape, list) and shape and _number(shape[0]) is not None and shape[0] > 0:
            return int(shape[0])
    return 0


def _episode_size_mb(meta_path: str, ep_name: str) -> float:
    """Size of the episode's HDF5 sibling, or of its LeRobot parquet and video files."""
    hdf5_path = meta_path.replace("_metadata.json", ".hdf5")
    if os.path.exists(hdf5_path):
        paths = [hdf5_path]
    else:
        root = glob.escape(os.path.dirname(os.path.dirname(meta_path)))
        paths = (glob.glob(os.path.join(root, "data", "*", f"{ep_name}.parquet"))
                 + glob.glob(os.path.join(root, "videos", "*", "*", f"{ep_name}.mp4")))
    size = 0
    for path in paths:
        try:
            size += os.path.getsize(path)
        except OSError:
            pass
    return round(size / 1e6, 1)


def _parse_episode_meta(meta_path: str, data_dir: str) -> dict | None:
    """Load one metadata JSON and return a normalised episode dict, or None on error.

    ``id`` is the metadata path relative to *data_dir*: episode names repeat
    across tasks and machines, so it is what identifies an episode.
    """
    meta = _read_meta(meta_path)
    if meta is None:
        return None

    ep_name = os.path.basename(meta_path).replace("_metadata.json", "")
    hdf5_path = meta_path.replace("_metadata.json", ".hdf5")

    # final_score: nested in dataset_summary for lerobot format, or top-level
    final_score = _number(meta.get("final_score"))
    summary = meta.get("dataset_summary")
    if final_score is None and isinstance(summary, dict):
        final_score = _number(summary.get("final_score"))

    outcome = meta.get("outcome")
    return {
        "id":                 os.path.relpath(meta_path, data_dir),
        "name":               ep_name,
        "meta_path":          meta_path,
        "has_hdf5":           os.path.exists(hdf5_path),
        "size_mb":            _episode_size_mb(meta_path, ep_name),
        "outcome":            None if outcome is None else str(outcome),
        "failure_modes":      meta.get("failure_modes"),
        "recovery_actions":   meta.get("recovery_actions"),
        "task_id":            _text(meta.get("task_id", meta.get("task_name", ""))),
        "subtask_id":         meta.get("subtask_id", ""),
        "run_id":             meta.get("run_id", ""),
        "date":               meta.get("date", ""),
        "task_name":          _text(meta.get("task_name", "")),
        "teleoperator_id":    _text(meta.get("teleoperator_id", "")),
        "supervisor_id":      meta.get("supervisor_id", ""),
        "hardware_version":   meta.get("hardware_version", ""),
        "software_version":   meta.get("software_version", ""),
        "final_score":        final_score,
        "n_steps":            _n_steps(meta),
        "annotation_rating":  _rating(meta.get("annotation_rating")),
        "mtime":              os.path.getmtime(meta_path),
    }


def _find_episodes(data_dir: str) -> list:
    """Find all metadata JSON files recursively under data_dir.

    Handles two patterns:
      - HDF5 format:    ep_XXXX_metadata.json  (anywhere under data_dir)
      - LeRobot format: meta/episode_XXXXXX_metadata.json
    """
    seen = set()
    episodes = []

    # HDF5-format: ep_*_metadata.json
    for meta_path in sorted(glob.glob(
            os.path.join(data_dir, "**", "ep_*_metadata.json"), recursive=True), reverse=True):
        if meta_path in seen:
            continue
        seen.add(meta_path)
        ep = _parse_episode_meta(meta_path, data_dir)
        if ep:
            episodes.append(ep)

    # LeRobot-format: meta/episode_XXXXXX_metadata.json
    for meta_path in sorted(glob.glob(
            os.path.join(data_dir, "**", "meta", "episode_*_metadata.json"), recursive=True), reverse=True):
        if meta_path in seen:
            continue
        seen.add(meta_path)
        ep = _parse_episode_meta(meta_path, data_dir)
        if ep:
            episodes.append(ep)

    # Sort newest first by mtime
    episodes.sort(key=lambda e: e["mtime"], reverse=True)
    return episodes


# ── Routes ───────────────────────────────────────────────────────────

@app.before_request
def _reject_foreign_requests():
    """Serve only this machine, and accept writes only as same-origin JSON.

    The API is unauthenticated: a page on another origin could otherwise POST
    a ``text/plain`` body without a CORS preflight, or reach it through DNS
    rebinding.
    """
    if BIND_HOST in _LOOPBACK_HOSTS:
        host = request.host.rsplit(":", 1)[0].strip("[]")
        if host not in _LOOPBACK_HOSTS:
            return jsonify({"error": "unexpected Host header"}), 403
    if request.method != "POST":
        return None
    origin = request.headers.get("Origin")
    if origin is not None and origin.rstrip("/") != request.host_url.rstrip("/"):
        return jsonify({"error": "cross-origin request rejected"}), 403
    if not request.is_json:
        return jsonify({"error": "expected application/json"}), 415
    return None


def _json_object() -> dict | None:
    """The request's JSON body if it is an object, else None."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/episodes")
def api_episodes():
    return jsonify(_find_episodes_cached())


@app.route("/api/label", methods=["POST"])
def api_label():
    data = _json_object()
    if data is None:
        return jsonify({"error": "expected a JSON object"}), 400
    ep_id = data.get("episode", "")
    label = data.get("label")

    if label is None or not ep_id:
        return jsonify({"error": "missing episode or label"}), 400
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return jsonify({"error": "label must be an integer code"}), 400

    # Only episodes the listing returned can be labelled, so no path is built from input.
    _invalidate_cache()
    episode = next((e for e in _find_episodes_cached() if e["id"] == ep_id), None)
    if episode is None:
        return jsonify({"error": f"episode not found: {ep_id}"}), 404
    meta_path = episode["meta_path"]

    # Never replace a metadata file that cannot be read with a label-only skeleton.
    meta = _read_meta(meta_path)
    if meta is None:
        return jsonify({"error": f"cannot read the metadata of {ep_id}"}), 409

    task_id = str(meta.get("task_id", meta.get("task_name", "0")))
    codes = _get_label_codes_for_task(task_id)
    if str(label_int) not in codes:
        return jsonify({"error": f"label {label_int} is not defined for {task_id}"}), 400
    label_name = str(codes[str(label_int)])

    ln = label_name.lower()

    def _extract_detail(name):
        for sep in (" - ", " — ", " – "):
            if sep in name:
                return name.split(sep, 1)[-1].strip()
        return name.strip()

    if ln.startswith("success"):
        meta["outcome"] = "success"
        meta["failure_modes"] = None
        meta["recovery_actions"] = None
    elif "partial" in ln:
        meta["outcome"] = "partial_success"
        meta["failure_modes"] = None
        meta["recovery_actions"] = _extract_detail(ln)
    else:
        meta["outcome"] = "failure"
        meta["failure_modes"] = _extract_detail(ln)
        meta["recovery_actions"] = None

    # Fill provenance the episode lacks from the current session; never replace
    # what was recorded with it.
    session = {}
    if os.path.exists(SESSION_CONFIG_PATH):
        try:
            with open(SESSION_CONFIG_PATH) as f:
                session = json.load(f)
        except Exception:
            pass
    for key in ("task_type", "teleoperator_id", "safety_watcher_id", "supervisor_id", "robot_model",
                "robot_id", "hardware_version", "software_version", "sim_id", "teleoperation_mode"):
        if not meta.get(key) and session.get(key):
            meta[key] = session[key]

    try:
        atomic_write_json(meta_path, meta, indent=4)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    _invalidate_cache()
    return jsonify({"ok": True, "episode": ep_id, "outcome": meta["outcome"]})


# ── Master config CRUD ──────────────────────────────────────────────

@app.route("/api/master", methods=["GET"])
def api_master_get():
    return jsonify(_load_master())


@app.route("/api/master", methods=["POST"])
def api_master_set():
    data = _json_object()
    if data is None:
        return jsonify({"error": "expected a JSON object"}), 400
    error = _master_shape_error(data)
    if error:
        return jsonify({"error": error}), 400
    try:
        _save_master(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


@app.route("/api/master/task", methods=["POST"])
def api_master_task_upsert():
    data = _json_object()
    if data is None:
        return jsonify({"error": "expected a JSON object"}), 400
    tid = str(data.get("task_id", ""))
    if not tid:
        return jsonify({"error": "missing task_id"}), 400
    master = _load_master()
    master.setdefault("tasks", {})[tid] = {
        "name":        data.get("name", ""),
        "subtasks":    data.get("subtasks", {}),
        "label_codes": data.get("label_codes", {}),
    }
    _save_master(master)
    return jsonify({"ok": True, "task_id": tid})


@app.route("/api/master/crew", methods=["POST"])
def api_master_crew_set():
    data = _json_object()
    if data is None:
        return jsonify({"error": "expected a JSON object"}), 400
    master = _load_master()
    master["crew"] = {
        "operators":   data.get("operators", []),
        "supervisors": data.get("supervisors", []),
    }
    error = _master_shape_error(master)
    if error:
        return jsonify({"error": error}), 400
    _save_master(master)
    return jsonify({"ok": True})


@app.route("/api/label_codes")
def api_label_codes():
    explicit = request.args.get("task_id")
    if explicit is not None:
        task_id = str(explicit)
    else:
        session = {}
        if os.path.exists(SESSION_CONFIG_PATH):
            try:
                with open(SESSION_CONFIG_PATH) as f:
                    session = json.load(f)
            except Exception:
                pass
        task_id = str(session.get("task_id", "0"))
    codes = _get_label_codes_for_task(task_id)
    return jsonify({"task_id": task_id, "codes": codes})


# ── Session config ──────────────────────────────────────────────────

@app.route("/api/session", methods=["GET"])
def api_session_get():
    if os.path.exists(SESSION_CONFIG_PATH):
        try:
            with open(SESSION_CONFIG_PATH) as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({
        "task_id": "", "subtask_id": "0", "task_type": "single_stage",
        "teleoperator_id": "", "safety_watcher_id": "", "supervisor_id": "",
        "robot_model": "", "robot_id": "", "hardware_version": "",
        "software_version": "", "sim_id": "", "teleoperation_mode": "",
    })


@app.route("/api/session", methods=["POST"])
def api_session_set():
    data = _json_object()
    if data is None:
        return jsonify({"error": "expected a JSON object"}), 400
    config = {
        "task_id":           data.get("task_id", ""),
        "subtask_id":        data.get("subtask_id", "0"),
        "task_type":         data.get("task_type", "single_stage"),
        "teleoperator_id":   data.get("teleoperator_id", ""),
        "safety_watcher_id": data.get("safety_watcher_id", ""),
        "supervisor_id":     data.get("supervisor_id", ""),
        "robot_model":       data.get("robot_model", ""),
        "robot_id":          data.get("robot_id", ""),
        "hardware_version":    data.get("hardware_version", ""),
        "software_version":    data.get("software_version", ""),
        "sim_id":              data.get("sim_id", ""),
        "teleoperation_mode":  data.get("teleoperation_mode", ""),
        "updated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        os.makedirs(os.path.dirname(SESSION_CONFIG_PATH), mode=0o700, exist_ok=True)
        atomic_write_json(SESSION_CONFIG_PATH, config, indent=4, mode=0o600)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, **config})


@app.route("/api/status")
def api_status():
    episodes = _find_episodes_cached()
    total = len(episodes)
    labeled = sum(1 for e in episodes if e["outcome"] is not None)
    total_mb = sum(e["size_mb"] for e in episodes)

    # Build per-operator summary, cross-referenced with master config names
    master = _load_master()
    crew = master.get("crew")
    operators = crew.get("operators") if isinstance(crew, dict) else None
    op_names = {
        op["id"]: op.get("name") or op["id"]
        for op in (operators if isinstance(operators, list) else [])
        if isinstance(op, dict) and "id" in op
    }

    per_operator: dict = {}
    for ep in episodes:
        op_id = ep.get("teleoperator_id") or "unknown"
        task_id = ep.get("task_id") or "unknown"
        if op_id not in per_operator:
            per_operator[op_id] = {
                "name":    op_names.get(op_id, op_id),
                "total":   0,
                "by_task": {},
            }
        per_operator[op_id]["total"] += 1
        per_operator[op_id]["by_task"][task_id] = \
            per_operator[op_id]["by_task"].get(task_id, 0) + 1

    return jsonify({
        "total_episodes": total,
        "labeled":        labeled,
        "unlabeled":      total - labeled,
        "total_size_mb":  round(total_mb, 1),
        "per_operator":   per_operator,
    })


# ── Annotation Ratings ─────────────────────────────────────────────

@app.route("/api/annotation_ratings")
def api_annotation_ratings():
    agg = {}
    for ep in _find_episodes_cached():
        rating = ep.get("annotation_rating")
        task = ep.get("task_name") or ep.get("task_id") or "unknown"
        if rating is None or not task:
            continue
        if task not in agg:
            agg[task] = {"task_name": task, "count": 0, "total": 0, "dist": {}}
        agg[task]["count"] += 1
        agg[task]["total"] += rating
        key = str(int(rating))
        agg[task]["dist"][key] = agg[task]["dist"].get(key, 0) + 1

    rows = []
    for task, d in sorted(agg.items()):
        rows.append({
            "task_name": d["task_name"],
            "count":     d["count"],
            "avg":       round(d["total"] / d["count"], 2) if d["count"] else 0,
            "dist":      d["dist"],
        })
    return jsonify({"rows": rows})


# ── Episode Durations ──────────────────────────────────────────────

@app.route("/api/episode_durations")
def api_episode_durations():
    by_task_raw = {}
    for ep in _find_episodes_cached():
        if ep.get("outcome") != "success" or ep.get("n_steps", 0) <= 0:
            continue
        task = ep.get("task_name") or ep.get("task_id") or "unknown"
        by_task_raw.setdefault(task, []).append(ep["n_steps"])

    rows = []
    for task in sorted(by_task_raw):
        steps = by_task_raw[task]
        mean  = statistics.mean(steps)
        stdev = statistics.stdev(steps) if len(steps) > 1 else 0.0
        stride = max(1, len(steps) // 2000)
        rows.append({
            "task_name": task,
            "count":     len(steps),
            "min":       min(steps),
            "max":       max(steps),
            "mean":      round(mean, 1),
            "stdev":     round(stdev, 1),
            "steps":     steps[::stride],
        })
    return jsonify({"rows": rows})


# ── Main ─────────────────────────────────────────────────────────────

def main():
    global DATA_DIR, BIND_HOST, MASTER_CONFIG_PATH

    parser = argparse.ArgumentParser(description="IMBench Data Collection Dashboard")
    parser.add_argument("--data-dir", default="demos",
                        help="Directory containing demos (default: demos/)")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1; the API is unauthenticated)")
    args = parser.parse_args()

    DATA_DIR = os.path.abspath(args.data_dir)
    BIND_HOST = args.host
    os.makedirs(DATA_DIR, exist_ok=True)

    MASTER_CONFIG_PATH = str(imbench_dir() / "master_config.json")
    if not os.path.exists(MASTER_CONFIG_PATH):
        _save_master(_read_master(_MASTER_DEFAULTS_PATH) or _DEFAULT_MASTER)
        print(f"Created master config from {_MASTER_DEFAULTS_PATH}: {MASTER_CONFIG_PATH}")

    print(f"IMBench Dashboard: http://{args.host}:{args.port}")
    print(f"Data directory:         {DATA_DIR}")
    print(f"Session config:         {SESSION_CONFIG_PATH}")
    print(f"Master config:          {MASTER_CONFIG_PATH}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
