"""Regression tests for convert_v21_to_v30.

Covers video-shard frame offsets, output-directory safety, the --in-place
backup and damaged inputs.  No simulator required.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from imbench.utils.lerobot_writer import LeRobotWriter
from tests.test_data_pipeline import SAVE_FREQ, TASK, _build_source, _jsonl, _run_main, _script

pd = pytest.importorskip("pandas")
av = pytest.importorskip("av")
pytest.importorskip("tqdm")

FRAME_STEP = 10  # intensity step between consecutive camera frames of an episode


def _build_coded(root, n_eps, n_steps=16):
    """v2.1 dataset whose camera frame k of episode e has intensity 30 + 40 e + 10 k in its left half.

    rgb_front is flat; rgb_wrist has noise in its right half, so its videos are
    much larger and roll over to a new file at other episodes.
    """
    rng = np.random.default_rng(0)
    writer = LeRobotWriter(root, TASK, 100, SAVE_FREQ, {})
    for e in range(n_eps):
        writer.begin_episode()
        for t in range(n_steps):
            k = t // (100 // SAVE_FREQ["cameras"])
            front = np.full((32, 32, 3), 30 + 40 * e + FRAME_STEP * k, np.uint8)
            wrist = front.copy()
            wrist[:, 16:] = rng.integers(0, 256, (32, 16, 3), dtype=np.uint8)
            obs = {"robot0_eef_pos": np.full(3, t, np.float32), "rgb_front": front, "rgb_wrist": wrist}
            writer.add_step(t * 0.01, obs, action=np.zeros(7, np.float32))
        writer.end_episode()
    writer.finalize()
    return Path(root)


def _decoded_codes(path):
    with av.open(str(path)) as c:
        return [float(f.to_ndarray(format="rgb24")[8:24, 2:10].mean()) for f in c.decode(video=0)]


def _check_frame_index(dst):
    """Assert every frame_index row decodes, from its episode's own file, to the frame it was recorded with.

    Returns the video file index of each episode, per camera.
    """
    info = json.loads((dst / "meta" / "info.json").read_text())
    eps = pq.read_table(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    data = pd.concat([pd.read_parquet(p) for p in sorted((dst / "data").glob("chunk-*/file-*.parquet"))],
                     ignore_index=True)
    files = {}
    for key in sorted(k for k, ft in info["features"].items() if ft.get("dtype") == "video"):
        cam = key.replace("observation.", "")
        files[cam] = []
        for e, ep in eps.iterrows():
            path = dst / info["video_path"].format(video_key=key,
                                                   chunk_index=ep[f"videos/{key}/chunk_index"],
                                                   file_index=ep[f"videos/{key}/file_index"])
            frames = _decoded_codes(path)
            rows = data.iloc[ep["dataset_from_index"]:ep["dataset_to_index"]]
            fi = rows[f"frame_index.{cam}"].dropna().astype(int).tolist()
            assert fi and all(0 <= i < len(frames) for i in fi), (cam, e, path.name, fi, len(frames))
            expected = [30 + 40 * e + FRAME_STEP * k for k in range(len(fi))]
            # Lossy re-encoding shifts the flat region by a few levels, under half a step
            assert np.allclose([frames[i] for i in fi], expected, rtol=0, atol=FRAME_STEP / 2 - 0.1), \
                (cam, e, path.name, fi)
            assert round(ep[f"videos/{key}/from_timestamp"] * info["features"][key]["fps"]) == fi[0]
            files[cam].append(int(ep[f"videos/{key}/file_index"]))
    return files


def _tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _set_jsonl(path, entries):
    Path(path).write_text("".join(json.dumps(e) + "\n" for e in entries))


# ── video shards ─────────────────────────────────────────────────────


class TestVideoShards:

    def test_frame_index_points_into_each_episodes_own_file(self, tmp_path, monkeypatch):
        convert = _script("convert_v21_to_v30")
        src = _build_coded(tmp_path / "ds", 4)
        mb = [p.stat().st_size / 1024 ** 2
              for p in sorted((src / "videos" / "chunk-000" / "rgb_front").glob("*.mp4"))]
        # Two rgb_front episodes per file: episode 2 overflows file-000 and starts
        # file-001.  Every larger rgb_wrist episode gets a file of its own.
        monkeypatch.setattr(convert, "DEFAULT_VIDEO_FILE_SIZE_IN_MB",
                            max(mb[0] + mb[1], mb[2] + mb[3]) * (1 + 1e-6))
        dst = tmp_path / "v30"
        assert convert.convert_dataset(src, dst, max_workers=1)
        assert _check_frame_index(dst) == {"rgb_front": [0, 0, 1, 1], "rgb_wrist": [0, 1, 2, 3]}

    def test_parallel_encoding_one_file_per_episode(self, tmp_path, monkeypatch):
        convert = _script("convert_v21_to_v30")
        src = _build_coded(tmp_path / "ds", 3)
        monkeypatch.setattr(convert, "DEFAULT_VIDEO_FILE_SIZE_IN_MB", 1e-6)
        dst = tmp_path / "v30"
        assert convert.convert_dataset(src, dst, max_workers=2)
        assert _check_frame_index(dst) == {"rgb_front": [0, 1, 2], "rgb_wrist": [0, 1, 2]}

    def test_single_file_offsets_unchanged(self, tmp_path):
        convert = _script("convert_v21_to_v30")
        dst = tmp_path / "v30"
        assert convert.convert_dataset(_build_coded(tmp_path / "ds", 3), dst, max_workers=1)
        assert _check_frame_index(dst) == {"rgb_front": [0, 0, 0], "rgb_wrist": [0, 0, 0]}

    def test_warns_about_cameras_missing_from_info(self, tmp_path, capsys):
        convert = _script("convert_v21_to_v30")
        src = _build_coded(tmp_path / "ds", 1)
        info_path = src / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        del info["features"]["observation.rgb_front"]
        info_path.write_text(json.dumps(info))
        dst = tmp_path / "v30"
        assert convert.convert_dataset(src, dst, max_workers=1)
        assert "not declared in meta/info.json, not converted: rgb_front\n" in capsys.readouterr().out
        assert sorted(p.name for p in (dst / "videos").iterdir()) == ["observation.rgb_wrist"]


# ── output directory ─────────────────────────────────────────────────


class TestOutputDir:

    @pytest.mark.parametrize("argv", [
        ["--demos-dir", "{runs}", "--out-dir", "{runs}"],
        ["--dataset", "{runs}/ds", "--out-dir", "{runs}"],
        ["--dataset", "{runs}/ds", "--out-dir", "{runs}/ds"],
        ["--dataset", "{runs}/ds", "--out-dir", "{runs}/ds/v30"],
    ])
    def test_refuses_output_overlapping_the_source(self, tmp_path, monkeypatch, argv):
        convert = _script("convert_v21_to_v30")
        runs = tmp_path / "runs"
        _build_source(runs / "ds", ["202601000001"])
        (runs / "processed_manifest.json").write_text("{}")
        before = _tree(runs)
        with pytest.raises(SystemExit) as exc:
            _run_main(convert, [a.format(runs=runs) for a in argv] + ["--workers", "1"], monkeypatch)
        assert exc.value.code not in (0, None)
        assert _tree(runs) == before

    def test_keeps_unrelated_output_and_replaces_earlier_conversion(self, tmp_path, monkeypatch):
        convert = _script("convert_v21_to_v30")
        runs, out = tmp_path / "runs", tmp_path / "runs_v30"
        _build_source(runs / "a", ["202601000001"])
        _build_source(runs / "b", ["202601000002", "202601000003"])
        _run_main(convert, ["--demos-dir", str(runs), "--out-dir", str(out), "--workers", "1"],
                  monkeypatch)
        assert sorted(p.name for p in out.iterdir()) == ["a", "b"]
        before = _tree(out)

        # With --dataset, --out-dir is the dataset directory: the batch root is refused
        with pytest.raises(SystemExit) as exc:
            _run_main(convert, ["--dataset", str(runs / "a"), "--out-dir", str(out), "--workers", "1"],
                      monkeypatch)
        assert "not an earlier v3.0 output" in str(exc.value.code)
        assert _tree(out) == before

        # A directory with other content is refused too
        (out / "c").mkdir()
        (out / "c" / "notes.txt").write_text("keep")
        with pytest.raises(SystemExit):
            _run_main(convert, ["--dataset", str(runs / "a"), "--out-dir", str(out / "c"),
                                "--workers", "1"], monkeypatch)
        assert (out / "c" / "notes.txt").read_text() == "keep"

        # An earlier v3.0 output of the dataset is replaced, in both modes
        _run_main(convert, ["--dataset", str(runs / "b"), "--out-dir", str(out / "b"), "--workers", "1"],
                  monkeypatch)
        _run_main(convert, ["--demos-dir", str(runs), "--out-dir", str(out), "--workers", "1"],
                  monkeypatch)
        assert _tree(out) == {**before, "c/notes.txt": b"keep"}
        assert sorted(p.name for p in out.iterdir()) == ["a", "b", "c"]


# ── --in-place ───────────────────────────────────────────────────────


class TestInPlace:

    def test_existing_backup_is_never_deleted(self, tmp_path, monkeypatch, capsys):
        convert = _script("convert_v21_to_v30")
        root = tmp_path / "runs"
        _build_source(root / "ds", ["202601000001"])
        (root / "ds_v21_backup").mkdir()
        (root / "ds_v21_backup" / "episode_01000001.parquet").write_text("earlier batch")
        before = _tree(root)
        with pytest.raises(SystemExit) as exc:
            _run_main(convert, ["--demos-dir", str(root), "--in-place", "--workers", "1"], monkeypatch)
        assert exc.value.code not in (0, None)
        assert "ds_v21_backup" in capsys.readouterr().out
        assert _tree(root) == before
        assert sorted(p.name for p in root.iterdir()) == ["ds", "ds_v21_backup"]

    def test_refuses_dataset_merged_into_after_in_place_conversion(self, tmp_path, monkeypatch):
        convert = _script("convert_v21_to_v30")
        root = tmp_path / "runs"
        ds = _build_source(root / "ds", ["202601000001", "202601000002"])
        _run_main(convert, ["--demos-dir", str(root), "--in-place", "--workers", "1"], monkeypatch)
        assert (ds / "data" / "chunk-000" / "file-000.parquet").exists()

        # What a later merge does: copy new v2.1 episode files in and rewrite the v2.1 meta
        new = _build_source(tmp_path / "new", ["202602000001"])
        for p in new.rglob("*"):
            if p.is_file():
                (ds / p.relative_to(new)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, ds / p.relative_to(new))
        before = _tree(root)

        with pytest.raises(SystemExit) as exc:
            _run_main(convert, ["--dataset", str(ds), "--workers", "1"], monkeypatch)
        assert "v3.0 data shards" in str(exc.value.code)
        with pytest.raises(SystemExit):
            _run_main(convert, ["--demos-dir", str(root), "--in-place", "--workers", "1"], monkeypatch)
        assert _tree(root) == before
        assert sorted(p.name for p in root.iterdir()) == ["ds", "ds_v21_backup"]


# ── damaged inputs and batch mode ────────────────────────────────────


class TestDamagedInputs:

    def test_truncated_parquet_is_skipped_with_warning(self, tmp_path, capsys):
        convert = _script("convert_v21_to_v30")
        src = _build_source(tmp_path / "ds", ["202601000001", "202601000002", "202601000003"])
        bad = src / "data" / "chunk-000" / "episode_000001.parquet"
        bad.write_bytes(bad.read_bytes()[:100])
        dst = tmp_path / "v30"
        assert convert.convert_dataset(src, dst, max_workers=1)
        assert str(bad) in capsys.readouterr().out
        info = json.loads((dst / "meta" / "info.json").read_text())
        assert (info["total_episodes"], info["total_frames"]) == (2, 12 + 14)

    def test_repeated_episodes_jsonl_line(self, tmp_path):
        convert = _script("convert_v21_to_v30")
        src = _build_source(tmp_path / "ds", ["202601000001", "202601000002"])
        entries = _jsonl(src / "meta" / "episodes.jsonl")
        _set_jsonl(src / "meta" / "episodes.jsonl", entries + entries[:1])
        dst = tmp_path / "v30"
        assert convert.convert_dataset(src, dst, max_workers=1)
        episodes = pq.read_table(dst / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        assert episodes.column("length").to_pylist() == [12, 13]

    def test_video_key_cannot_escape_the_output(self, tmp_path):
        convert = _script("convert_v21_to_v30")
        src = _build_source(tmp_path / "bundle" / "ds", ["202601000001"])
        info_path = src / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        key = next(k for k, ft in info["features"].items() if ft.get("dtype") == "video")
        info["features"][f"{key}/../../../../escaped"] = info["features"].pop(key)
        info_path.write_text(json.dumps(info))
        with pytest.raises(convert.ConvertError, match="invalid video feature key"):
            convert.convert_dataset(src, tmp_path / "work" / "out", max_workers=1)
        assert not list(tmp_path.rglob("escaped*"))

    def test_batch_reports_failed_dataset_and_converts_the_rest(self, tmp_path, monkeypatch, capsys):
        convert = _script("convert_v21_to_v30")
        runs = tmp_path / "runs"
        for name in ("a_task", "b_task", "c_task"):
            _build_source(runs / name, ["202601000001"])
        # Fails in the episodes stage, after data and video shards were written
        jsonl = runs / "b_task" / "meta" / "episodes.jsonl"
        _set_jsonl(jsonl, [{k: v for k, v in e.items() if k != "length"} for e in _jsonl(jsonl)])
        with pytest.raises(SystemExit) as exc:
            _run_main(convert, ["--demos-dir", str(runs), "--workers", "1"], monkeypatch)
        assert exc.value.code not in (0, None)
        assert "b_task" in capsys.readouterr().out
        # No partial (or hidden temporary) output is left for b_task
        assert sorted(p.name for p in runs.iterdir()) == ["a_task", "a_task_v30", "b_task",
                                                          "c_task", "c_task_v30"]
        for name in ("a_task", "c_task"):
            info = json.loads((runs / f"{name}_v30" / "meta" / "info.json").read_text())
            assert info["codebase_version"] == "v3.0"

    def test_collector_layout_hint(self, tmp_path, monkeypatch, capsys):
        convert = _script("convert_v21_to_v30")
        demos = tmp_path / "demos"
        _build_source(demos / TASK / "M01", ["202601000001"])
        with pytest.raises(SystemExit) as exc:
            _run_main(convert, ["--demos-dir", str(demos), "--workers", "1"], monkeypatch)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert TASK in out and "merge_annotated_runs.py" in out
