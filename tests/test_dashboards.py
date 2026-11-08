"""Regression tests for the labeling dashboard (``scripts/dashboard.py``) and the
annotation dashboard (``data_verification``).  No simulator or display required.
"""

import importlib
import json
import re
import stat
import sys
from pathlib import Path

import pytest

from data_verification.annotation_store import AnnotationStore

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TASK = "ib-p1-edge-slide-v1"


# ── labeling dashboard ───────────────────────────────────────────────


@pytest.fixture()
def dash(tmp_path, monkeypatch):
    """``scripts/dashboard.py`` reading ``tmp_path/data`` with a private master and session."""
    pytest.importorskip("flask")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    dashboard = importlib.import_module("dashboard")
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(dashboard, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(dashboard, "SESSION_CONFIG_PATH", str(tmp_path / "session.json"))
    monkeypatch.setattr(dashboard, "MASTER_CONFIG_PATH", str(tmp_path / "master.json"))
    dashboard._invalidate_cache()
    return dashboard


@pytest.fixture()
def client(dash):
    return dash.app.test_client()


def _episode(data, meta, index=1, task=TASK, machine="M01"):
    """Write one episode metadata file; returns (path, dashboard episode id)."""
    rel = f"{task}/{machine}/meta/episode_{index:06d}_metadata.json"
    path = data / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta))
    return path, rel


def _master(tmp_path, tasks):
    (tmp_path / "master.json").write_text(json.dumps({"tasks": tasks}))


class TestLabeling:

    def test_unlisted_task_gets_generic_codes(self, client, tmp_path):
        """Not the codes of whichever task the master config lists first."""
        _master(tmp_path, {"ib-p2-cube-toss-v1": {"label_codes": {"1": "success", "2": "failure - timing_miss"}}})
        codes = client.get("/api/label_codes?task_id=ib-m-mirror-pick-place-v1").get_json()["codes"]
        assert codes == {"1": "success", "2": "partial - partial_completion", "3": "failure - task_failure"}

    def test_outcome_follows_the_configured_label(self, client, tmp_path):
        _master(tmp_path, {TASK: {"label_codes": {"1": "failure - grasp_miss", "2": "success"}}})
        meta, ep = _episode(tmp_path / "data", {"task_id": TASK})
        assert client.post("/api/label", json={"episode": ep, "label": 1}).status_code == 200
        saved = json.loads(meta.read_text())
        assert (saved["outcome"], saved["failure_modes"]) == ("failure", "grasp_miss")

    def test_undefined_label_code_is_rejected(self, client, tmp_path):
        _master(tmp_path, {TASK: {"label_codes": {"1": "success", "2": "failure - grasp_miss"}}})
        meta, ep = _episode(tmp_path / "data", {"task_id": TASK})
        before = meta.read_text()
        assert client.post("/api/label", json={"episode": ep, "label": 9}).status_code == 400
        assert meta.read_text() == before

    def test_unreadable_metadata_is_not_replaced(self, dash, client, tmp_path, monkeypatch):
        """A file that breaks after the listing read it is reported, not overwritten with a skeleton."""
        meta, ep = _episode(tmp_path / "data", {"task_id": TASK, "run_id": "202601000001", "seed": 3})
        truncated = meta.read_text()[:25]
        find_episodes = dash._find_episodes

        def find_then_truncate(data_dir):
            episodes = find_episodes(data_dir)
            meta.write_text(truncated)
            return episodes

        monkeypatch.setattr(dash, "_find_episodes", find_then_truncate)
        assert client.post("/api/label", json={"episode": ep, "label": 1}).status_code == 409
        assert meta.read_text() == truncated


class TestEpisodeListing:

    def test_malformed_metadata_does_not_break_the_api(self, dash, client, tmp_path):
        data = tmp_path / "data"
        _episode(data, [1, 2], index=1)
        _episode(data, {"task_id": 5, "teleoperator_id": 7, "final_score": "0.9"}, index=2)
        _episode(data, {"task_id": TASK, "outcome": "success", "annotation_rating": "four",
                        "dataset_summary": {"action": ["x", 7]}}, index=3)
        for url in ("/api/episodes", "/api/status", "/api/annotation_ratings", "/api/episode_durations"):
            assert client.get(url).status_code == 200, url
        episodes = {e["name"]: e for e in client.get("/api/episodes").get_json()}
        assert sorted(episodes) == ["episode_000002", "episode_000003"]
        second, third = episodes["episode_000002"], episodes["episode_000003"]
        assert (second["task_id"], second["teleoperator_id"], second["final_score"]) == ("5", "7", None)
        assert (third["annotation_rating"], third["n_steps"]) == (None, 0)

    def test_lerobot_episode_size_counts_parquet_and_videos(self, client, tmp_path):
        data = tmp_path / "data"
        _episode(data, {"task_id": TASK})
        root = data / TASK / "M01"
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "data" / "chunk-000" / "episode_000001.parquet").write_bytes(b"0" * 300_000)
        for cam in ("rgb_ego_cam", "rgb_gripper_cam"):
            (root / "videos" / "chunk-000" / cam).mkdir(parents=True)
            (root / "videos" / "chunk-000" / cam / "episode_000001.mp4").write_bytes(b"0" * 450_000)
        (root / "videos" / "chunk-000" / "rgb_ego_cam" / "episode_000002.mp4").write_bytes(b"0" * 900_000)
        [episode] = client.get("/api/episodes").get_json()
        assert episode["size_mb"] == 1.2
        assert client.get("/api/status").get_json()["total_size_mb"] == 1.2


@pytest.mark.parametrize("path,body", [
    ("/api/master", {"crew": 5}),
    ("/api/master", {"tasks": {"t": 3}}),
    ("/api/master/crew", {"operators": "OP-1"}),
])
def test_malformed_master_config_is_rejected(dash, client, path, body):
    assert client.post(path, json=body).status_code == 400
    assert client.get("/api/status").status_code == 200


def test_master_config_is_private_and_untracked(dash, tmp_path, monkeypatch):
    """Crew names are saved under ~/.imbench, never into the git-tracked scripts/master_config.json."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in ("DATA_DIR", "BIND_HOST", "MASTER_CONFIG_PATH"):  # restored after main() sets them
        monkeypatch.setattr(dash, name, getattr(dash, name))
    monkeypatch.setattr(dash.app, "run", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["dashboard.py", "--data-dir", str(tmp_path / "data")])
    tracked = SCRIPTS / "master_config.json"
    defaults = tracked.read_text()
    dash.main()
    path = home / ".imbench" / "master_config.json"
    assert dash.MASTER_CONFIG_PATH == str(path)
    assert json.loads(path.read_text()) == json.loads(defaults)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    resp = dash.app.test_client().post(
        "/api/master/crew", json={"operators": [{"id": "OP-001", "name": "Jane Doe"}], "supervisors": []})
    assert resp.status_code == 200
    assert "Jane Doe" in path.read_text()
    assert tracked.read_text() == defaults


def test_external_scripts_are_pinned_with_sri():
    html = (SCRIPTS / "templates" / "index.html").read_text()
    tags = re.findall(r"<script\b[^>]*\bsrc=[^>]*>", html)
    assert tags
    for tag in tags:
        assert re.search(r"@\d+\.\d+\.\d+/", tag), tag
        assert 'integrity="sha384-' in tag and 'crossorigin="anonymous"' in tag, tag


# ── annotation dashboard ─────────────────────────────────────────────


def _collection(root, indices, videos=True):
    """Write a collection folder with one camera and the given episode indices."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({
        "task_id": TASK, "chunk_size": 1000,
        "features": {"observation.rgb_cam": {"dtype": "video"}},
    }))
    (meta / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": i}) + "\n" for i in indices))
    if videos:
        cam = root / "videos" / "chunk-000" / "rgb_cam"
        cam.mkdir(parents=True)
        for i in indices:
            (cam / f"episode_{i:06d}.mp4").write_bytes(b"")
    return root


@pytest.fixture()
def annotation_dashboard():
    return pytest.importorskip("data_verification.annotation_dashboard")


@pytest.fixture()
def open_dashboard(annotation_dashboard, monkeypatch):
    """Factory for dashboards that track the current episode without opening its videos."""
    def load_episode(self, ep_idx):
        self._ep_idx = ep_idx

    monkeypatch.setattr(annotation_dashboard.AnnotationDashboard, "_load_episode", load_episode)

    def make(root, annotator="", start=0):
        dashboard = annotation_dashboard.AnnotationDashboard(str(root), annotator=annotator)
        dashboard._load_episode(start)
        return dashboard

    return make


def _saved(root, episode):
    ann = json.loads((root / "annotations.json").read_text())["annotations"][str(episode)]
    return ann and (ann["rating"], ann["annotator"])


class TestAnnotationDashboard:

    def test_rating_moves_to_the_next_unrated_episode_after_the_current_one(self, tmp_path, open_dashboard):
        dashboard = open_dashboard(_collection(tmp_path, range(4)))
        dashboard._handle_key(ord("n"))  # skip episode 0
        dashboard._handle_key(ord("4"))
        assert dashboard._ep_idx == 2
        dashboard._handle_key(ord("5"))
        dashboard._handle_key(ord("5"))  # episode 3: wraps around to the skipped episode
        assert dashboard._ep_idx == 0

    def test_concurrent_annotators_do_not_overwrite_each_other(self, tmp_path, open_dashboard, capsys):
        root = _collection(tmp_path, range(3))
        alice, bob = open_dashboard(root, "alice"), open_dashboard(root, "bob")
        alice._handle_key(ord("5"))
        bob._handle_key(ord("1"))  # bob's view of episode 0 predates alice's rating
        assert _saved(root, 0) == (5, "alice")
        assert (bob._ep_idx, bob.store.rating_of(0)) == (0, 5)
        assert "changed by another session (now rated 5/5 by alice)" in capsys.readouterr().out
        bob._handle_key(ord("1"))  # pressed again: a deliberate overwrite
        assert _saved(root, 0) == (1, "bob")
        assert bob._ep_idx == 1

    @pytest.mark.parametrize("case", ["no_episodes", "no_videos", "merged"])
    def test_unusable_dataset_is_a_one_line_error(self, annotation_dashboard, tmp_path, monkeypatch, capsys,
                                                  case):
        root = tmp_path / "ds"
        if case == "no_episodes":
            _collection(root, [])
        elif case == "no_videos":
            _collection(root, range(2), videos=False)
        else:  # scripts/merge_annotated_runs.py output
            _collection(root, [7000001, 7000002], videos=False)
            cam = root / "videos" / "chunk-000" / "rgb_cam"
            cam.mkdir(parents=True)
            (cam / "episode_07000001.mp4").write_bytes(b"")
        monkeypatch.setattr(sys, "argv", ["annotation_dashboard", str(root)])
        with pytest.raises(SystemExit) as exit_info:
            annotation_dashboard.main()
        assert exit_info.value.code == 1
        assert capsys.readouterr().out.startswith("Error: ")
        assert not (root / "annotations.json.lock").exists()


class TestAnnotationStore:

    def test_stale_write_is_refused(self, tmp_path):
        alice = AnnotationStore(str(tmp_path), TASK, 3)
        bob = AnnotationStore(str(tmp_path), TASK, 3)
        assert alice.annotate(0, 5, "alice") is True
        assert bob.annotate(0, 1, "bob") is False
        assert _saved(tmp_path, 0) == (5, "alice")
        assert bob.rating_of(0) == 5  # refreshed: the next change is deliberate
        assert bob.clear(0) is True
        assert _saved(tmp_path, 0) is None

    def test_invalid_rating_reads_as_pending(self, tmp_path):
        (tmp_path / "annotations.json").write_text(json.dumps({"annotations": {
            "0": {"rating": "four"}, "1": {"rating": 9}, "2": {"rating": True}, "3": {"rating": 4},
        }}))
        with pytest.warns(RuntimeWarning, match="invalid rating"):
            store = AnnotationStore(str(tmp_path), TASK, 4)
            assert [store.rating_of(i) for i in range(4)] == [None, None, None, 4]
            assert store.first_unannotated() == 0
            assert store.summary()["pending"] == 3
