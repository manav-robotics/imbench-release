"""merge_annotated_runs: source identity, interrupted runs, schema and
output-format checks, parquet episode_index.  No simulator required."""

import functools
import json
import os
import shutil

import pyarrow.parquet as pq
import pytest

from imbench.utils import run_id as rid
from tests.test_data_pipeline import TASK, _build_source, _jsonl, _run_main, _script, home  # noqa: F401


@pytest.fixture()
def merge():
    return _script("merge_annotated_runs")


def _merge(merge, monkeypatch, inp, out, *extra):
    _run_main(merge, ["--input", str(inp), "--output", str(out), *extra], monkeypatch)


def _rated(n):
    return {str(i): {"rating": 5} for i in range(n)}


def _snapshot(root):
    """{relative path: (size, mtime_ns)} of every file under *root*."""
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def _rows(out, idx):
    return pq.read_metadata(out / TASK / "data" / "chunk-000" / f"episode_{idx:08d}.parquet").num_rows


def _assert_complete(out, lengths):
    """Every output episode has parquet, metadata and video, and *lengths* appear once each."""
    task = out / TASK
    stems = sorted(p.stem for p in (task / "data" / "chunk-000").glob("episode_*.parquet"))
    assert sorted(_rows(out, int(s[-8:])) for s in stems) == sorted(lengths)
    for stem in stems:
        assert (task / "meta" / f"{stem}_metadata.json").exists()
        assert (task / "videos" / "chunk-000" / "rgb_front" / f"{stem}.mp4").exists()
    assert not list(out.rglob("*.tmp"))
    entries = _jsonl(task / "meta" / "episodes.jsonl")
    assert [e["episode_index"] for e in entries] == [int(s[-8:]) for s in stems]
    assert json.loads((task / "meta" / "info.json").read_text())["total_episodes"] == len(lengths)


def _fail_once(monkeypatch, module, name, when):
    """Make ``module.name`` raise KeyboardInterrupt the first time ``when(call_number, *args)`` is true."""
    real = getattr(module, name)
    state = {"calls": 0, "fired": False}

    def flaky(*args, **kwargs):
        state["calls"] += 1
        if not state["fired"] and when(state["calls"], *args):
            state["fired"] = True
            raise KeyboardInterrupt("simulated Ctrl-C")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, flaky)


# ── re-issued run_ids (same year) ────────────────────────────────────


@pytest.mark.usefixtures("home")
class TestSourceIdentity:

    def test_reset_counter_does_not_overwrite_merged_episodes(self, merge, tmp_path, monkeypatch):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002", "202601000003"], n=10)
        _merge(merge, monkeypatch, inp, out)
        before = _snapshot(out)

        # Counter reset: the resumed M01 writer records episodes 3 and 4 with re-issued run_ids
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002"], n=30)
        with pytest.raises(SystemExit) as exc:
            _merge(merge, monkeypatch, inp, out)
        assert exc.value.code == 1
        assert _snapshot(out) == before
        assert [_rows(out, i) for i in (1000001, 1000002, 1000003)] == [10, 11, 12]

    def test_new_recording_under_a_merged_name_is_not_skipped(self, merge, tmp_path, monkeypatch,
                                                              capsys):
        out = tmp_path / "out"
        _build_source(tmp_path / "a" / TASK / "M01", ["202601000001", "202601000002"], n=10)
        _merge(merge, monkeypatch, tmp_path / "a", out)
        before = _snapshot(out)

        # Fresh folder, same M01/<episode> keys and run_ids, different data
        _build_source(tmp_path / "b" / TASK / "M01", ["202601000001", "202601000002"], n=30)
        with pytest.raises(SystemExit) as exc:
            _merge(merge, monkeypatch, tmp_path / "b", out)
        assert exc.value.code == 1
        assert "a different recording with the same run_id" in capsys.readouterr().out
        assert _snapshot(out) == before

    def test_rerun_on_unchanged_input_is_a_no_op(self, merge, tmp_path, monkeypatch, capsys):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002"])
        _build_source(inp / TASK / "M02", ["", ""])
        _merge(merge, monkeypatch, inp, out)
        before = _snapshot(out)
        capsys.readouterr()

        _merge(merge, monkeypatch, inp, out)
        assert "Processed: 0 episodes" in capsys.readouterr().out
        assert _snapshot(out) == before
        assert rid.read_counter() == 2

    def test_outputs_merged_before_fingerprints_are_checked(self, merge, tmp_path, monkeypatch):
        out = tmp_path / "out"
        src = _build_source(tmp_path / "a" / TASK / "M01", ["202601000001", "202601000002"], n=10)
        _merge(merge, monkeypatch, tmp_path / "a", out)
        # Rewrite the output as the previous version left it: byte copies, no fingerprints
        manifest_path = out / "processed_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["processed"].values():
            entry.pop("fingerprint", None)
        manifest_path.write_text(json.dumps(manifest))
        for ep in (0, 1):
            tag = f"episode_{1000001 + ep:08d}"
            shutil.copy2(src / "data" / "chunk-000" / f"episode_{ep:06d}.parquet",
                         out / TASK / "data" / "chunk-000" / f"{tag}.parquet")
            meta_path = out / TASK / "meta" / f"{tag}_metadata.json"
            meta = json.loads(meta_path.read_text())
            meta.pop("source_fingerprint", None)
            meta_path.write_text(json.dumps(meta))
        before = _snapshot(out)

        _merge(merge, monkeypatch, tmp_path / "a", out)
        assert _snapshot(out) == before

        _build_source(tmp_path / "b" / TASK / "M01", ["202601000001", "202601000002"], n=30)
        with pytest.raises(SystemExit):
            _merge(merge, monkeypatch, tmp_path / "b", out)
        assert _snapshot(out) == before


# ── interrupted runs ─────────────────────────────────────────────────


@pytest.mark.usefixtures("home")
class TestInterrupted:

    def test_rerun_after_interrupted_copy_has_no_duplicates(self, merge, tmp_path, monkeypatch):
        out = tmp_path / "out"
        inp = tmp_path / "demos"
        _build_source(inp / TASK / "M04", ["", "", "", ""])
        run = functools.partial(_merge, merge, monkeypatch, inp, out)

        # Ctrl-C while copying the third episode's video
        _fail_once(monkeypatch, shutil, "copy2", lambda calls, *args: calls == 3)
        with pytest.raises(KeyboardInterrupt):
            run()
        run()
        _assert_complete(out, [12, 13, 14, 15])
        assert rid.read_counter() == 4

    def test_rerun_completes_episode_interrupted_between_renames(self, merge, tmp_path, monkeypatch):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M04", ["", "", "", ""])
        _fail_once(monkeypatch, os, "replace",
                   lambda calls, src, dst: str(dst).endswith("episode_04000002_metadata.json"))
        with pytest.raises(KeyboardInterrupt):
            _merge(merge, monkeypatch, inp, out)
        assert (out / TASK / "data" / "chunk-000" / "episode_04000002.parquet").exists()
        assert not (out / TASK / "meta" / "episode_04000002_metadata.json").exists()

        _merge(merge, monkeypatch, inp, out)
        _assert_complete(out, [12, 13, 14, 15])
        assert rid.read_counter() == 4

    def test_rerun_rebuilds_metadata_after_interrupted_rebuild(self, merge, tmp_path, monkeypatch):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002"])
        _fail_once(monkeypatch, merge, "_rebuild_dataset_meta", lambda calls, *args: True)
        with pytest.raises(KeyboardInterrupt):
            _merge(merge, monkeypatch, inp, out)
        _merge(merge, monkeypatch, inp, out)
        _assert_complete(out, [12, 13])


# ── schema and output format ─────────────────────────────────────────


@pytest.mark.usefixtures("home")
class TestPreconditions:

    def test_sources_with_different_features_abort(self, merge, tmp_path, monkeypatch, capsys):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001"], camera=False)
        _build_source(inp / TASK / "M02", ["202602000001"])
        info_path = inp / TASK / "M02" / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["save_freq"] = {**info["save_freq"], "cameras": 50}
        info_path.write_text(json.dumps(info))
        with pytest.raises(SystemExit) as exc:
            _merge(merge, monkeypatch, inp, out)
        assert exc.value.code == 1
        assert not (out / TASK).exists()
        text = capsys.readouterr().out
        assert (f"{TASK}/M02: extra features ['frame_index.rgb_front', 'observation.rgb_front'] "
                f"vs {TASK}/M01") in text
        assert f"{TASK}/M02: save_freq" in text

    def test_source_differing_from_merged_output_aborts(self, merge, tmp_path, monkeypatch):
        out = tmp_path / "out"
        _build_source(tmp_path / "a" / TASK / "M01", ["202601000001"])
        _merge(merge, monkeypatch, tmp_path / "a", out)
        before = _snapshot(out)
        _build_source(tmp_path / "b" / TASK / "M02", ["202602000001"], camera=False)
        with pytest.raises(SystemExit):
            _merge(merge, monkeypatch, tmp_path / "b", out)
        assert _snapshot(out) == before

    @pytest.mark.parametrize("converted", ["info", "shard"])
    def test_converted_output_is_refused(self, converted, merge, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out"
        _build_source(tmp_path / "a" / TASK / "M01", ["202601000001"])
        _build_source(tmp_path / "b" / TASK / "M02", ["202602000001"])
        run = functools.partial(_merge, merge, monkeypatch)
        run(tmp_path / "a", out)
        task = out / TASK
        if converted == "info":
            info = json.loads((task / "meta" / "info.json").read_text())
            (task / "meta" / "info.json").write_text(json.dumps({**info, "codebase_version": "v3.0"}))
        else:
            first = next((task / "data" / "chunk-000").glob("episode_*.parquet"))
            shutil.copy2(first, task / "data" / "chunk-000" / "file-000.parquet")
        before = _snapshot(out)
        capsys.readouterr()

        with pytest.raises(SystemExit) as exc:
            run(tmp_path / "b", out)
        assert exc.value.code == 1
        assert "not LeRobot v2.1 datasets" in capsys.readouterr().out
        assert _snapshot(out) == before

    def test_rerun_with_nothing_new_leaves_a_converted_output_alone(self, merge, tmp_path,
                                                                    monkeypatch):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001"])
        _merge(merge, monkeypatch, inp, out)
        info_path = out / TASK / "meta" / "info.json"
        info_path.write_text(json.dumps({**json.loads(info_path.read_text()), "codebase_version": "v3.0"}))
        before = _snapshot(out)

        _merge(merge, monkeypatch, inp, out)  # nothing new: exits 0 and rebuilds nothing
        assert _snapshot(out) == before


# ── copied episodes ──────────────────────────────────────────────────


@pytest.mark.usefixtures("home")
class TestCopiedEpisodes:

    def test_parquet_episode_index_is_the_merged_index(self, merge, tmp_path, monkeypatch):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M01", ["202601000001", "202601000002"])
        _build_source(inp / TASK / "M02", ["202602000001"])
        _merge(merge, monkeypatch, inp, out)

        sources = {1000001: ("M01", 0), 1000002: ("M01", 1), 2000001: ("M02", 0)}
        for out_idx, (machine, ep) in sources.items():
            src = inp / TASK / machine / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
            dst = out / TASK / "data" / "chunk-000" / f"episode_{out_idx:08d}.parquet"
            assert pq.read_schema(dst).equals(pq.read_schema(src), check_metadata=True)
            table, src_table = pq.read_table(dst), pq.read_table(src)
            assert set(table.column("episode_index").to_pylist()) == {out_idx}
            assert table.drop_columns(["episode_index"]).equals(src_table.drop_columns(["episode_index"]))

    def test_episode_keeps_the_task_config_it_was_recorded_with(self, merge, tmp_path, monkeypatch):
        inp, out = tmp_path / "demos", tmp_path / "out"
        for machine, words in (("M01", ["ALPHA"]), ("M02", ["BRAVO"])):
            src = _build_source(inp / TASK / machine, [f"20260{machine[-1]}000001"])
            info = json.loads((src / "meta" / "info.json").read_text())
            (src / "meta" / "info.json").write_text(json.dumps({**info, "task_config": {"words": words}}))
        _merge(merge, monkeypatch, inp, out)

        replay = _script("replay_episode")
        info = json.loads((out / TASK / "meta" / "info.json").read_text())
        for idx, words in ((1000001, ["ALPHA"]), (2000001, ["BRAVO"])):
            entry = {"episode_index": idx}
            assert replay._replay_config(str(out / TASK), info, entry)["words"] == words

    def test_warns_about_parquets_without_metadata(self, merge, tmp_path, monkeypatch, capsys):
        inp, out = tmp_path / "demos", tmp_path / "out"
        _build_source(inp / TASK / "M06", ["202606000001", "202606000002"])
        (inp / TASK / "M06" / "meta" / "episode_000001_metadata.json").unlink()
        _merge(merge, monkeypatch, inp, out)
        text = capsys.readouterr().out
        assert "WARNING" in text and "data/chunk-000/episode_000001.parquet" in text
        assert [e["episode_index"] for e in _jsonl(out / TASK / "meta" / "episodes.jsonl")] == [6000001]

    def test_corrupt_annotations_skip_the_source(self, merge, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out"
        inp = tmp_path / "demos"
        _build_source(inp / TASK / "M01", ["202601000001"], annotations=_rated(1))
        bad = _build_source(inp / TASK / "M02", ["202602000001"], annotations=_rated(1))
        run = functools.partial(_merge, merge, monkeypatch, inp, out, "--min-rating", "4")
        (bad / "annotations.json").write_text("{not json")

        with pytest.raises(SystemExit) as exc:
            run()
        assert exc.value.code not in (None, 0)
        assert str(bad / "annotations.json") in capsys.readouterr().out
        assert len(_jsonl(out / TASK / "meta" / "episodes.jsonl")) == 1
