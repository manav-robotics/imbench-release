"""Regression tests for the data-collection and post-processing tooling.

Covers the shared run-ID counter, atomic JSON writes, setup_machine,
merge_annotated_runs, convert_v21_to_v30, replay_episode
file resolution, the annotation store and the labeling dashboard.  No
simulator required.
"""

import importlib
import json
import multiprocessing as mp
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from data_verification.annotation_store import AnnotationStore, rating_from_annotation
from imbench.utils import recording
from imbench.utils import run_id as rid
from imbench.utils.fileio import atomic_write_json
from imbench.utils.lerobot_writer import LeRobotWriter

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TASK = "ib-p2-cube-toss-v1"
SAVE_FREQ = {"proprio": 100, "force_torque": 50, "object_state": 50, "cameras": 25, "tactile": 50}


def _script(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


def _run_main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__file__, *argv])
    module.main()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Isolated HOME with an empty ~/.imbench."""
    path = tmp_path / "home"
    (path / ".imbench").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(path))
    return path


def _build_source(root, run_ids, annotations=None, n=12, camera=True, chunk_size=1000):
    """Write a collect_demos-style raw dataset with one episode per run id.

    Episode ``k`` has ``n + k`` steps.
    """
    root = Path(root)
    writer = LeRobotWriter(root, TASK, 100, SAVE_FREQ, {}, chunk_size=chunk_size)
    for k, run in enumerate(run_ids):
        writer.begin_episode()
        for t in range(n + k):
            obs = {"robot0_eef_pos": np.full(3, t, np.float32),
                   "object_state": np.ones(4, np.float32)}
            if camera:
                obs["rgb_front"] = np.full((32, 32, 3), (t * 9) % 255, np.uint8)
            writer.add_step(t * 0.01, obs, action=np.full(7, 0.01 * t, np.float32),
                            task_score=1.0)
        ep = writer.episode_index
        writer.end_episode({"final_score": 1.0, "run_id": run, "seed": ep, "warmup_steps": 1})
        meta = {
            "run_id": run,
            "seed": ep,
            "warmup_steps": 1,
            "outcome": "success",
            "dataset_summary": {"action": [n + k, 7], "final_score": 1.0,
                                "teleop_device": "keyboard"},
        }
        (root / "meta" / f"episode_{ep:06d}_metadata.json").write_text(json.dumps(meta))
    writer.finalize()
    if annotations is not None:
        (root / "annotations.json").write_text(json.dumps({"annotations": annotations}))
    return root


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ── run IDs ──────────────────────────────────────────────────────────


def _issue_ids(counter_file, n, start, queue):
    from imbench.utils.run_id import next_run_id

    start.wait()
    queue.put([next_run_id("03", path=counter_file) for _ in range(n)])


class TestRunId:

    def test_ids_unique_across_processes(self, tmp_path):
        counter_file = tmp_path / "run_counter.json"
        ctx = mp.get_context("spawn")
        queue, start = ctx.Queue(), ctx.Event()
        procs = [ctx.Process(target=_issue_ids, args=(str(counter_file), 100, start, queue))
                 for _ in range(3)]
        for p in procs:
            p.start()
        start.set()
        ids = [x for _ in procs for x in queue.get(timeout=120)]
        for p in procs:
            p.join(timeout=30)
        assert len(ids) == 300
        assert len(set(ids)) == 300
        assert rid.read_counter(counter_file) == 300

    def test_format_and_output_name(self, tmp_path):
        run = rid.next_run_id("07", path=tmp_path / "c.json", year=2026)
        assert run == "202607000001"
        assert rid.episode_index_from_run_id(run) == 7000001

    def test_corrupt_counter_raises_and_is_kept(self, tmp_path):
        counter_file = tmp_path / "run_counter.json"
        counter_file.write_text('{"counter": ')
        with pytest.raises(rid.RunCounterError, match="Refusing to restart"):
            rid.next_run_id("01", path=counter_file)
        assert counter_file.read_text() == '{"counter": '

    @pytest.mark.parametrize("bad", ["7", "007", "ab", 7, None])
    def test_machine_id_must_be_two_digits(self, bad):
        with pytest.raises(ValueError):
            rid.validate_machine_id(bad)

    def test_setup_machine_keeps_existing_counter(self, home, monkeypatch):
        setup = _script("setup_machine")
        rid.init_counter(41, overwrite=True)
        _run_main(setup, ["--machine-id", "03"], monkeypatch)
        assert rid.load_machine_id() == "03"
        assert rid.read_counter() == 41
        _run_main(setup, ["--set-counter", "5"], monkeypatch)
        assert rid.read_counter() == 5
        assert rid.load_machine_id() == "03"


# ── atomic JSON writes ───────────────────────────────────────────────


class TestAtomicWriteJson:

    def test_failed_write_keeps_previous_file_and_no_temp_file(self, tmp_path):
        merge = _script("merge_annotated_runs")
        path = tmp_path / "data.json"
        atomic_write_json(path, {"a": 1})
        with pytest.raises(TypeError):
            atomic_write_json(path, {"a": object()})
        with pytest.raises(TypeError):
            merge._save_manifest(str(tmp_path), {"processed": {"x": object()}})
        assert json.loads(path.read_text()) == {"a": 1}
        assert sorted(p.name for p in tmp_path.iterdir()) == ["data.json"]

    def test_new_file_mode_follows_umask(self, tmp_path):
        old_umask = os.umask(0o022)
        try:
            atomic_write_json(tmp_path / "shared.json", {})
            atomic_write_json(tmp_path / "private.json", {}, mode=0o600)
        finally:
            os.umask(old_umask)
        assert stat.S_IMODE((tmp_path / "shared.json").stat().st_mode) == 0o644
        assert stat.S_IMODE((tmp_path / "private.json").stat().st_mode) == 0o600


# ── merge_annotated_runs ─────────────────────────────────────────────


class TestMerge:

    def test_merge_names_and_episode_fields(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002"])
        _build_source(inp / TASK / "M02", ["202602000001"])
        _run_main(merge, ["--input", str(inp), "--output", str(out)], monkeypatch)

        names = sorted(p.name for p in (out / TASK / "data" / "chunk-000").glob("*.parquet"))
        assert names == ["episode_01000001.parquet", "episode_01000002.parquet",
                         "episode_02000001.parquet"]
        entries = _jsonl(out / TASK / "meta" / "episodes.jsonl")
        assert [e["episode_index"] for e in entries] == [1000001, 1000002, 2000001]
        for e in entries:
            assert e["final_score"] == 1.0
            assert e["seed"] in (0, 1) and e["warmup_steps"] == 1
            assert rid.episode_index_from_run_id(e["run_id"]) == e["episode_index"]
        info = json.loads((out / TASK / "meta" / "info.json").read_text())
        assert info["total_episodes"] == 3

        # A second run finds everything in the manifest and copies nothing.
        _run_main(merge, ["--input", str(inp), "--output", str(out)], monkeypatch)
        assert len(list((out / TASK / "data" / "chunk-000").glob("*.parquet"))) == 3

    def test_collision_within_run_aborts_before_copying(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001"])
        _build_source(inp / TASK / "M02", ["202501000001"])  # same MMCCCCCC, other year
        with pytest.raises(SystemExit) as exc:
            _run_main(merge, ["--input", str(inp), "--output", str(out)], monkeypatch)
        assert exc.value.code == 1
        assert not (out / TASK).exists()

    def test_collision_with_existing_output_is_not_overwritten(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        out = tmp_path / "out"
        _build_source(tmp_path / "a" / TASK / "M01", ["202601000001"])
        _run_main(merge, ["--input", str(tmp_path / "a"), "--output", str(out)], monkeypatch)
        meta_path = out / TASK / "meta" / "episode_01000001_metadata.json"
        before = meta_path.read_text()

        _build_source(tmp_path / "b" / TASK / "M01", ["202501000001"], n=30)
        with pytest.raises(SystemExit):
            _run_main(merge, ["--input", str(tmp_path / "b"), "--output", str(out)], monkeypatch)
        assert meta_path.read_text() == before
        table = pq.read_table(out / TASK / "data" / "chunk-000" / "episode_01000001.parquet")
        assert table.num_rows == 12

    def test_issues_ids_for_episodes_without_run_id(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M04", [""])
        rid.init_counter(9, overwrite=True)
        _run_main(merge, ["--input", str(inp), "--output", str(out)], monkeypatch)
        (entry,) = _jsonl(out / TASK / "meta" / "episodes.jsonl")
        assert entry["episode_index"] == 4000010
        assert entry["run_id"].endswith("04000010")
        assert rid.read_counter() == 10

    def test_rating_filter_reads_legacy_status_and_drops_unrated(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        inp, out = tmp_path / "demos", tmp_path / "out"
        annotations = {"0": {"status": "good"}, "1": {"status": "bad"}, "2": None}
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002", "202601000003"],
                      annotations=annotations)
        assert merge._load_annotations(str(inp / TASK / "M01")) == {0: 5, 1: 1}

        _run_main(merge, ["--input", str(inp), "--output", str(out), "--min-rating", "0"],
                  monkeypatch)
        entries = _jsonl(out / TASK / "meta" / "episodes.jsonl")
        assert [e["episode_index"] for e in entries] == [1000001, 1000002]
        assert [e["annotation_rating"] for e in entries] == [5, 1]


# ── convert_v21_to_v30 ───────────────────────────────────────────────


class TestConvert:

    @pytest.fixture(autouse=True)
    def _requires_data_extra(self):
        pytest.importorskip("pandas")
        pytest.importorskip("av")
        pytest.importorskip("tqdm")

    def test_shards_get_contiguous_episode_and_frame_indices(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        convert = _script("convert_v21_to_v30")
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002"])
        _build_source(inp / TASK / "M02", ["202602000001"])
        _run_main(merge, ["--input", str(inp), "--output", str(out)], monkeypatch)

        dst = tmp_path / "v30"
        assert convert.convert_dataset(out / TASK, dst, max_workers=1)
        df = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet").to_pandas()
        lengths = [12, 13, 12]
        assert df["episode_index"].tolist() == [i for i, n in enumerate(lengths) for _ in range(n)]
        assert df["index"].tolist() == list(range(sum(lengths)))
        fi = df["frame_index.rgb_front"]
        assert str(fi.dtype) == "Int64"
        assert fi.isna().any()
        # Camera at 25 Hz of 100 Hz: 3 / 4 / 3 frames per episode, offset into one video
        assert fi.dropna().tolist() == list(range(10))

    def test_episodes_in_several_chunks_keep_their_metadata(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        convert = _script("convert_v21_to_v30")
        inp, out = tmp_path / "demos", tmp_path / "out"
        m01 = ["202601000001", "202601000002", "202601000003"]
        m02 = ["202602000001", "202602000002"]
        _build_source(inp / TASK / "M01", m01, n=10, chunk_size=2)
        _build_source(inp / TASK / "M02", m02, n=20, chunk_size=2)
        _run_main(merge, ["--input", str(inp), "--output", str(out)], monkeypatch)
        assert (out / TASK / "data" / "chunk-001" / "episode_01000003.parquet").exists()

        dst = tmp_path / "v30"
        assert convert.convert_dataset(out / TASK, dst, max_workers=1)
        lengths = [10, 11, 12, 20, 21]
        episodes = pq.read_table(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        assert episodes.column("length").to_pylist() == lengths
        spans = zip(episodes.column("dataset_from_index").to_pylist(),
                    episodes.column("dataset_to_index").to_pylist())
        assert [end - start for start, end in spans] == lengths
        data = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
        steps = data.column("observation.robot0_eef_pos.0").to_pylist()
        assert steps == [float(t) for n in lengths for t in range(n)]
        extras = json.loads((dst / "meta" / "episodes_extra.json").read_text())
        assert [r["run_id"] for r in extras] == m01 + m02
        assert [r["dataset_summary"]["action"][0] for r in extras] == lengths

    def test_info_totals_count_only_converted_episodes(self, home, tmp_path, monkeypatch):
        merge = _script("merge_annotated_runs")
        convert = _script("convert_v21_to_v30")
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002"])
        _run_main(merge, ["--input", str(inp), "--output", str(out)], monkeypatch)
        (out / TASK / "videos" / "chunk-000" / "rgb_front" / "episode_01000002.mp4").unlink()

        dst = tmp_path / "v30"
        assert convert.convert_dataset(out / TASK, dst, max_workers=1)
        info = json.loads((dst / "meta" / "info.json").read_text())
        assert (info["total_episodes"], info["total_frames"]) == (1, 12)
        assert pq.read_metadata(dst / "data" / "chunk-000" / "file-000.parquet").num_rows == 12

    def test_discovery_skips_previous_outputs(self, tmp_path, monkeypatch):
        convert = _script("convert_v21_to_v30")
        root = tmp_path / "runs"
        _build_source(root / "ds", ["202601000001"])
        argv = ["--demos-dir", str(root), "--in-place", "--workers", "1"]
        _run_main(convert, argv, monkeypatch)
        assert json.loads((root / "ds" / "meta" / "info.json").read_text())["codebase_version"] == "v3.0"
        assert (root / "ds_v21_backup").is_dir()

        _run_main(convert, argv, monkeypatch)  # ds is now v3.0; the backup must be ignored
        assert sorted(p.name for p in root.iterdir()) == ["ds", "ds_v21_backup"]
        backup_info = json.loads((root / "ds_v21_backup" / "meta" / "info.json").read_text())
        assert backup_info["codebase_version"] == "v2.1"


# ── replay_episode ───────────────────────────────────────────────────


class TestReplayResolution:

    @pytest.mark.parametrize("width", [6, 8])
    def test_find_parquet_by_episode_index(self, tmp_path, width):
        replay = _script("replay_episode")
        index = 1000002 if width == 8 else 3
        chunk = tmp_path / "data" / "chunk-000"
        chunk.mkdir(parents=True)
        target = chunk / f"episode_{index:0{width}d}.parquet"
        target.touch()
        (chunk / "episode_00000000.parquet").touch()
        assert replay._find_episode_parquet(str(tmp_path), index) == str(target)
        with pytest.raises(FileNotFoundError):
            replay._find_episode_parquet(str(tmp_path), 42)

    def test_reset_protocol_sources(self, tmp_path):
        replay = _script("replay_episode")
        (tmp_path / "meta").mkdir()
        assert replay._reset_protocol(str(tmp_path), {"seed": 4, "warmup_steps": 1}, 4) == (4, 1)
        assert replay._reset_protocol(str(tmp_path), {}, 7) == (None, None)
        (tmp_path / "meta" / "episode_01000007_metadata.json").write_text(
            json.dumps({"seed": 7, "warmup_steps": 1}))
        assert replay._reset_protocol(str(tmp_path), {}, 1000007) == (7, 1)

    def test_replay_uses_episode_vocabulary_even_after_dataset_changes(self, tmp_path):
        replay = _script("replay_episode")
        original = {"word": None, "words": ["JOINT", "MATRIX"], "max_episode_steps": 8000}
        info = {"task_config": {"word": "NEURIPS"}}
        entry = {"episode_index": 1000007, "task_config": original}
        expected = {**original, "max_episode_steps": 999_999}
        assert replay._replay_config(str(tmp_path), info, entry) == expected
        # Merging can rebuild episodes.jsonl without the extra fields; the
        # original per-episode metadata still restores the correct vocabulary.
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "episode_01000007_metadata.json").write_text(
            json.dumps({"dataset_summary": {"task_config": original}}))
        assert replay._replay_config(str(tmp_path), info, {"episode_index": 1000007}) == expected
        assert original["max_episode_steps"] == 8000
        assert replay._replay_config(str(tmp_path), info, {"episode_index": 0}) == {
            "word": "NEURIPS", "max_episode_steps": 999_999,
        }


    def test_replayed_score_is_taken_at_termination(self, monkeypatch):
        """Teleop episodes record cooldown steps after termination; the score can change there."""
        replay = _script("replay_episode")

        class Env:
            action_space = type("Space", (), {"shape": (7,)})()

            def reset(self, seed=None):
                self.t = 0
                return {}, {"task_score": 0.0}

            def step(self, action):
                self.t += 1
                done = self.t >= 3
                score = 0.6 if self.t == 3 else (0.0 if done else 0.3)
                return {}, 0.0, done, False, {"task_score": score}

            def close(self):
                pass

        monkeypatch.setattr(replay.ib, "make", lambda *a, **kw: Env())
        monkeypatch.setattr(replay, "_read_episode", lambda root, idx: (np.zeros((6, 7)), None))
        ep_meta = {"episode_index": 0, "length": 6, "final_score": 0.6, "seed": 0, "warmup_steps": 0}
        info = {"task_id": TASK, "control_freq": 100, "save_freq": {}}
        assert replay.replay_one("unused", info, ep_meta, speed=0, use_dashboard=False,
                                 render=False) == 0.6


# ── annotation store ─────────────────────────────────────────────────


class TestAnnotationStore:

    def test_concurrent_sessions_keep_each_others_ratings(self, tmp_path):
        a = AnnotationStore(str(tmp_path), TASK, 4)
        b = AnnotationStore(str(tmp_path), TASK, 4)
        a.annotate(0, 5, "alice")
        b.annotate(1, 2, "bob")
        a.clear(0)
        a.annotate(2, 4, "alice")
        data = json.loads((tmp_path / "annotations.json").read_text())["annotations"]
        assert data["0"] is None
        assert data["1"]["rating"] == 2
        assert data["2"]["rating"] == 4
        assert data["3"] is None
        assert a.rating_of(1) == 2  # a sees b's rating after its own save

    def test_corrupt_file_is_backed_up_with_warning(self, tmp_path):
        (tmp_path / "annotations.json").write_text('{"annotations": {"0": {"rati')
        with pytest.warns(RuntimeWarning, match="unreadable"):
            store = AnnotationStore(str(tmp_path), TASK, 2)
        backups = list(tmp_path.glob("annotations.json.corrupt-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == '{"annotations": {"0": {"rati'
        store.annotate(1, 3)
        assert json.loads((tmp_path / "annotations.json").read_text())["annotations"]["1"]["rating"] == 3

    def test_legacy_status_ratings(self, tmp_path):
        assert rating_from_annotation({"status": "good"}) == 5
        assert rating_from_annotation({"status": "bad"}) == 1
        assert rating_from_annotation({"rating": 3}) == 3
        assert rating_from_annotation(None) is None


# ── labeling dashboard ───────────────────────────────────────────────


class TestDashboard:

    @pytest.fixture(autouse=True)
    def _requires_flask(self):
        pytest.importorskip("flask")

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        dashboard = _script("dashboard")
        monkeypatch.setattr(dashboard, "DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(dashboard, "SESSION_CONFIG_PATH", str(tmp_path / "session.json"))
        monkeypatch.setattr(dashboard, "MASTER_CONFIG_PATH", str(tmp_path / "master.json"))
        dashboard._invalidate_cache()
        return dashboard.app.test_client()

    @staticmethod
    def _episode(root, task, machine, meta=None):
        path = root / task / machine / "meta" / "episode_000001_metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"task_id": task, **(meta or {})}))
        return path

    def test_label_rejects_paths_outside_the_listing(self, client, tmp_path):
        (tmp_path / "data").mkdir()
        outside = tmp_path / "secret_metadata.json"
        outside.write_text("{}")
        for name in ("../secret_metadata.json", "../../etc/passwd", "episode_*", "episode_000001"):
            resp = client.post("/api/label", json={"episode": name, "label": 3})
            assert resp.status_code == 404, name
        assert outside.read_text() == "{}"

    def test_label_updates_episode(self, client, tmp_path):
        meta = self._episode(tmp_path / "data", TASK, "M01")
        ep_id = f"{TASK}/M01/meta/episode_000001_metadata.json"
        assert [e["id"] for e in client.get("/api/episodes").get_json()] == [ep_id]
        resp = client.post("/api/label", json={"episode": ep_id, "label": 1})
        assert resp.status_code == 200
        assert json.loads(meta.read_text())["outcome"] == "success"
        assert client.post("/api/label", json={"episode": ep_id, "label": "x"}).status_code == 400

    def test_label_targets_one_of_several_same_named_episodes(self, client, tmp_path):
        """Episode names repeat across tasks and machines; only the chosen one is labelled."""
        data = tmp_path / "data"
        metas = [self._episode(data, task, machine) for task, machine in
                 ((TASK, "M01"), (TASK, "M02"), ("ib-p3-gap-funnel-v1", "M01"))]
        target = f"{TASK}/M02/meta/episode_000001_metadata.json"
        assert client.post("/api/label", json={"episode": target, "label": 1}).status_code == 200
        outcomes = [json.loads(m.read_text()).get("outcome") for m in metas]
        assert outcomes == [None, "success", None]

    def test_label_follows_symlinked_dataset_folders(self, client, tmp_path):
        meta = self._episode(tmp_path / "bigdisk", TASK, "M01")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / TASK).symlink_to(tmp_path / "bigdisk" / TASK)
        ep_id = f"{TASK}/M01/meta/episode_000001_metadata.json"
        assert client.post("/api/label", json={"episode": ep_id, "label": 1}).status_code == 200
        assert json.loads(meta.read_text())["outcome"] == "success"

    def test_label_keeps_recorded_provenance(self, client, tmp_path):
        meta = self._episode(tmp_path / "data", TASK, "M01",
                             {"teleoperator_id": "OP-001", "supervisor_id": "SV-001"})
        (tmp_path / "session.json").write_text(json.dumps(
            {"teleoperator_id": "OP-009", "supervisor_id": "SV-009", "robot_id": "R-2"}))
        ep_id = f"{TASK}/M01/meta/episode_000001_metadata.json"
        assert client.post("/api/label", json={"episode": ep_id, "label": 1}).status_code == 200
        saved = json.loads(meta.read_text())
        assert (saved["teleoperator_id"], saved["supervisor_id"], saved["robot_id"]) == \
            ("OP-001", "SV-001", "R-2")

    @pytest.mark.parametrize("path", ["/api/session", "/api/master/crew", "/api/master/task",
                                      "/api/master", "/api/label"])
    def test_cross_site_writes_are_rejected(self, client, tmp_path, path):
        """A page on another origin can POST text/plain without a CORS preflight."""
        session = tmp_path / "session.json"
        session.write_text(json.dumps({"teleoperator_id": "OP-002"}))
        assert client.post(path, data="x", content_type="text/plain").status_code == 415
        assert client.post(path, json={}, headers={"Origin": "http://evil.example"}).status_code == 403
        assert json.loads(session.read_text()) == {"teleoperator_id": "OP-002"}

    def test_foreign_host_header_is_rejected(self, client):
        """DNS rebinding reaches a loopback server under the attacker's host name."""
        assert client.get("/api/episodes", headers={"Host": "evil.example:5050"}).status_code == 403
        assert client.get("/api/episodes", headers={"Host": "127.0.0.1:5050"}).status_code == 200

    def test_session_is_private_to_the_user(self, client, home, monkeypatch):
        dashboard = _script("dashboard")
        path = recording.session_config_path()
        assert path == home / ".imbench" / "session.json"
        (home / ".imbench").rmdir()
        monkeypatch.setattr(dashboard, "SESSION_CONFIG_PATH", str(path))
        resp = client.post("/api/session", json={"task_id": TASK, "teleoperator_id": "OP-002"})
        assert resp.status_code == 200
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert recording.load_session()["teleoperator_id"] == "OP-002"

    def test_default_master_has_no_names(self):
        dashboard = _script("dashboard")
        crew = dashboard._DEFAULT_MASTER["crew"]
        people = [p for role in crew.values() for p in role]
        assert people and all(p["name"] == "" and p["id"] for p in people)
        assert dashboard._DEFAULT_MASTER["tasks"] == {}
