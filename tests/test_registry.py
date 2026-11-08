"""Tests for the task registry and config loading."""

import pytest
import yaml

from imbench.registry import TASK_REGISTRY, make, list_tasks
from imbench.tasks.base_task import load_task_config, _deep_merge, CONFIGS_DIR


# ── Registry tests ──────────────────────────────────────────────────


class TestRegistry:
    """Tests for task registration and listing."""

    def test_list_tasks_returns_list(self):
        tasks = list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0

    def test_list_tasks_contains_known_ids(self):
        tasks = list_tasks()
        expected = [
            "ib-p1-edge-slide-v1",
            "ib-m-freeform-v1",
            "ib-p7-packing-v1",
            # Single-arm prehensile tasks
            "ib-p2-cube-toss-v1",
            "ib-p4-tool-retrieve-v1",
            "ib-p4-cup-extract-v1",
            "ib-p5-occluder-push-v1",
            "ib-p1-sheltered-grasp-v1",
            "ib-p2-pendulum-grasp-v1",
            "ib-p3-domino-single-v1",
            "ib-p3-domino-select-v1",
            "ib-p3-seesaw-balance-v1",
            "ib-p6-slip-recovery-v1",
            "ib-p2-linear-intercept-v1",
            "ib-p2-bounce-intercept-v1",
            "ib-p5-mass-sort-v1",
            # Puzzle tasks
            "ib-p7-shape-stack-v1",
            "ib-p6-stack-collapse-recovery-v1",
            "ib-p2-ramp-sort-v1",
            "ib-p3-ball-cage-escape-v1",
            "ib-p3-indirect-push-v1",
        ]
        for task_id in expected:
            assert task_id in tasks, f"{task_id} missing from registry"

    def test_hetbi_tasks_auto_registered(self):
        tasks = list_tasks()
        hetbi_ids = [
            t for t in tasks
            if "hetbi" in t or t in {
                "ib-p1-bowl-table-v1", "ib-p1-cup-inversion-v1",
                "ib-p1-cracker-box-jaw-v1", "ib-p1-cracker-box-suction-v1",
            }
        ]
        assert len(hetbi_ids) >= 5, f"Expected ≥5 hetbi tasks, got {len(hetbi_ids)}"

    def test_prehensile_tasks_registered(self):
        import functools, inspect
        from imbench.tasks.prehensile.single_arm_base import SingleArmBase
        pr_ids = [
            tid for tid, factory in TASK_REGISTRY.items()
            if inspect.isclass(
                factory.func if hasattr(factory, "func") else factory
            ) and issubclass(
                factory.func if hasattr(factory, "func") else factory, SingleArmBase
            )
        ]
        assert len(pr_ids) >= 13, f"Expected ≥13 prehensile tasks, got {len(pr_ids)}"

    def test_puzzle_tasks_registered(self):
        # Puzzle tasks span the ib-p2-, ib-p3-, ib-p6- and ib-p7- categories
        puzzle_task_ids = {
            "ib-p2-ramp-sort-v1", "ib-p3-ball-cage-escape-v1",
            "ib-p3-gap-funnel-v1", "ib-p3-indirect-push-v1",
            "ib-p6-pick-place-ycb-v1", "ib-p7-shape-stack-v1",
            "ib-p6-stack-collapse-recovery-v1",
        }
        tasks = set(list_tasks())
        pz_ids = [t for t in puzzle_task_ids if t in tasks]
        assert len(pz_ids) >= 6, f"Expected ≥6 former-puzzle tasks, got {len(pz_ids)}"

    def test_hetbi_task_ids_match_yaml_filenames(self):
        """Each hetbi YAML should produce a matching registry entry."""
        hetbi_dir = CONFIGS_DIR / "hetbi"
        yaml_ids = sorted(p.stem for p in hetbi_dir.glob("*.yaml"))
        tasks = list_tasks()
        for yid in yaml_ids:
            assert yid in tasks, f"Hetbi config {yid}.yaml not registered"

    def test_make_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            make("ib-nonexistent-task-v999")

    def test_make_returns_base_task_subclass(self):
        from imbench.tasks.base_task import BaseTask
        env = make("ib-p2-cube-toss-v1", obs_mode="combo", render_mode=None, seed=0)
        try:
            assert isinstance(env, BaseTask)
        finally:
            env.close()

    def test_registry_keys_are_strings(self):
        for key in TASK_REGISTRY:
            assert isinstance(key, str), f"Registry key {key!r} is not a string"

    def test_registry_values_are_callable(self):
        for key, val in TASK_REGISTRY.items():
            assert callable(val), f"Registry entry '{key}' is not callable"


# ── Config loading tests ────────────────────────────────────────────


class TestConfigLoading:
    """Tests for YAML config loading."""

    def test_load_existing_config(self):
        cfg = load_task_config("ib-p1-edge-slide-v1")
        assert isinstance(cfg, dict)
        assert cfg["task_id"] == "ib-p1-edge-slide-v1"

    def test_load_config_with_explicit_file(self):
        cfg = load_task_config("ib-p1-bowl-table-v1", config_file="hetbi/ib-p1-bowl-table-v1")
        assert cfg["task_id"] == "ib-p1-bowl-table-v1"

    def test_load_config_subdirectory_search(self):
        """Config files in subdirectories should be found by task_id."""
        cfg = load_task_config("ib-p1-bowl-table-v1")
        assert cfg["task_id"] == "ib-p1-bowl-table-v1"

    def test_load_nonexistent_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_task_config("ib-totally-fake-task")

    def test_config_overrides(self):
        cfg = load_task_config(
            "ib-p1-edge-slide-v1",
            overrides={"max_episode_steps": 42},
        )
        assert cfg["max_episode_steps"] == 42

    def test_config_deep_merge_nested(self):
        cfg = load_task_config(
            "ib-p1-edge-slide-v1",
            overrides={"scene": {"table": {"height": 1.5}}},
        )
        assert cfg["scene"]["table"]["height"] == 1.5
        # Other scene keys should survive
        assert "object" in cfg["scene"]

    def test_deep_merge_basic(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_deep_merge_replaces_non_dict(self):
        base = {"a": {"b": 1}}
        override = {"a": 42}
        result = _deep_merge(base, override)
        assert result["a"] == 42

    def test_deep_merge_empty_override(self):
        base = {"a": 1, "b": {"c": 2}}
        result = _deep_merge(base, {})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_deep_merge_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_deep_merge_nested_three_levels(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        result = _deep_merge(base, {"a": {"b": {"c": 99}}})
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2

    def test_deep_merge_new_key_added(self):
        base = {"a": 1}
        result = _deep_merge(base, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_deep_merge_list_replaced_not_merged(self):
        base = {"a": [1, 2, 3]}
        result = _deep_merge(base, {"a": [4, 5]})
        assert result["a"] == [4, 5]

    def test_deep_merge_none_override_value(self):
        base = {"a": 1}
        result = _deep_merge(base, {"a": None})
        assert result["a"] is None


# ── YAML integrity tests ───────────────────────────────────────────


class TestYAMLIntegrity:
    """Validate all YAML config files parse and have required fields."""

    @staticmethod
    def _all_yaml_paths():
        return sorted(CONFIGS_DIR.rglob("*.yaml"))

    def test_all_yamls_parse(self):
        for path in self._all_yaml_paths():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert isinstance(cfg, dict), f"{path} did not parse to a dict"

    def test_all_yamls_have_task_id(self):
        for path in self._all_yaml_paths():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "task_id" in cfg, f"{path.name} missing task_id"

    def test_all_yamls_have_max_episode_steps(self):
        for path in self._all_yaml_paths():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "max_episode_steps" in cfg, f"{path.name} missing max_episode_steps"
            assert isinstance(cfg["max_episode_steps"], int)
            assert cfg["max_episode_steps"] > 0

    def test_all_yamls_have_obs_section(self):
        for path in self._all_yaml_paths():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "obs" in cfg, f"{path.name} missing obs section"

    def test_hetbi_yamls_have_task_conditions(self):
        hetbi_dir = CONFIGS_DIR / "hetbi"
        for path in sorted(hetbi_dir.glob("*.yaml")):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "task_conditions" in cfg, f"{path.name} missing task_conditions"
            tc = cfg["task_conditions"]
            assert "success" in tc, f"{path.name} missing task_conditions.success"
            assert "failure" in tc, f"{path.name} missing task_conditions.failure"

    def test_hetbi_yamls_have_scene_table(self):
        hetbi_dir = CONFIGS_DIR / "hetbi"
        for path in sorted(hetbi_dir.glob("*.yaml")):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            scene = cfg.get("scene", {})
            assert "table" in scene, f"{path.name} missing scene.table"
            table = scene["table"]
            assert "height" in table

    def test_hobi_yamls_have_task_conditions(self):
        hobi_dir = CONFIGS_DIR / "hobi"
        for path in sorted(hobi_dir.glob("*.yaml")):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "task_conditions" in cfg, f"{path.name} missing task_conditions"
            tc = cfg["task_conditions"]
            assert "success" in tc
            assert "failure" in tc
            # Failure may use floor_z_threshold or floor_drop_below
            assert ("floor_z_threshold" in tc["failure"]
                    or "floor_drop_below" in tc["failure"]), (
                f"{path.name}: failure needs floor_z_threshold or floor_drop_below"
            )

    def test_hobi_yamls_have_scene_table(self):
        hobi_dir = CONFIGS_DIR / "hobi"
        for path in sorted(hobi_dir.glob("*.yaml")):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            scene = cfg.get("scene", {})
            assert "table" in scene, f"{path.name} missing scene.table"
            table = scene["table"]
            assert "height" in table
            assert "full_size" in table
            assert "offset" in table

    def test_controller_section_is_valid(self):
        for path in self._all_yaml_paths():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            ctrl = cfg.get("controller")
            if ctrl is not None:
                # Must be a dict or list of dicts
                if isinstance(ctrl, list):
                    for c in ctrl:
                        assert isinstance(c, dict)
                        assert "type" in c
                else:
                    assert isinstance(ctrl, dict)
                    assert "type" in ctrl

    def test_prehensile_yamls_have_required_sections(self):
        """All prehensile YAMLs (except edge-slide) should have task section."""
        for path in self._all_yaml_paths():
            cfg_name = path.stem
            if "prehensile" not in str(path):
                continue
            if "edge-slide" in cfg_name:
                continue
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert "task" in cfg, f"{cfg_name}: missing 'task' section"
            assert "task_conditions" in cfg, f"{cfg_name}: missing 'task_conditions'"
            tc = cfg["task_conditions"]
            assert "success" in tc, f"{cfg_name}: missing task_conditions.success"
            assert "failure" in tc, f"{cfg_name}: missing task_conditions.failure"
            scene = cfg.get("scene", {})
            assert "table" in scene, f"{cfg_name}: missing scene.table"

    def test_puzzle_yamls_have_required_sections(self):
        puzzle_dir = CONFIGS_DIR / "puzzle"
        if not puzzle_dir.exists():
            pytest.skip("No puzzle config directory")
        for path in sorted(puzzle_dir.glob("*.yaml")):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            name = path.stem
            assert "task" in cfg, f"{name}: missing 'task' section"
            assert "task_conditions" in cfg, f"{name}: missing 'task_conditions'"
            tc = cfg["task_conditions"]
            assert "success" in tc, f"{name}: missing task_conditions.success"
            assert "failure" in tc, f"{name}: missing task_conditions.failure"
            scene = cfg.get("scene", {})
            assert "table" in scene, f"{name}: missing scene.table"

    def test_prehensile_yamls_task_id_matches_filename(self):
        for path in self._all_yaml_paths():
            cfg_name = path.stem
            if "prehensile" not in str(path):
                continue
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert cfg["task_id"] == cfg_name, (
                f"{cfg_name}: task_id '{cfg['task_id']}' != filename"
            )

    def test_puzzle_yamls_task_id_matches_filename(self):
        puzzle_dir = CONFIGS_DIR / "puzzle"
        if not puzzle_dir.exists():
            pytest.skip("No puzzle config directory")
        for path in sorted(puzzle_dir.glob("*.yaml")):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            assert cfg["task_id"] == path.stem, (
                f"{path.stem}: task_id '{cfg['task_id']}' != filename"
            )
