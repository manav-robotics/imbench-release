"""Episode word sampling keeps the display, arm split and scoring in sync."""

from pathlib import Path

import numpy as np
import pytest

import imbench as ib
from imbench.tasks.base_task import load_task_config
from data_synthetic.bimanual_collect_demos import load_words


TASK = "ib-m-keyboard-typing-v1"


@pytest.fixture
def task():
    env = ib.make(TASK, obs_mode="proprio", seed=7, privileged_obs=True,
                  config_overrides={"word": None, "words": load_words()})
    try:
        yield env
    finally:
        env.close()


def _snapshot(task, seed=None):
    obs, info = task.reset(seed=seed)
    return (info["word"], tuple(task._arm0_indices), tuple(task._arm1_indices),
            task._env.sim.data.qpos.copy(),
            obs)


def test_seed_reproduces_word_partition_and_initial_state(task):
    first = _snapshot(task)  # constructor seed controls the first episode
    _snapshot(task, 19)
    replay = _snapshot(task, 7)
    assert first[:3] == replay[:3]
    np.testing.assert_array_equal(first[3], replay[3])
    for key in first[4]:
        np.testing.assert_array_equal(first[4][key], replay[4][key])
    assert len({_snapshot(task)[0] for _ in range(8)}) > 1


def test_all_training_words_reset_display_scoring_and_arm_assignments(task):
    words = load_words()
    assert len(words) == len(set(words)) == 25
    assert all(len(set(word)) == len(word) for word in words)
    assert set(words).isdisjoint({"ROBOT", "TYPER", "BENCH", "HELLO", "WORLD"})
    # Select a seed for each word so every configured length/letter is exercised.
    seeds = {}
    for seed in range(1000):
        seeds.setdefault(str(np.random.RandomState(seed).choice(words)), seed)
    assert set(seeds) == set(words)

    for word, seed in seeds.items():
        obs, info = task.reset(seed=seed)
        assert info["word"] == word
        assert info["typed"] == "" and not info["is_success"]
        assert task._typed_mask == [False] * len(word)
        assert obs["word_length"].item() == len(word)
        assert task.observation_space.contains(obs)
        assert sorted(task._arm0_indices + task._arm1_indices) == list(range(len(word)))
        assert task._arm0_indices and task._arm1_indices
        assert len(task._staged_reward._stages) == 3 * len(word)
        assert len(task._letter_geom_ids) == len(word)
        for bold, faint in task._letter_geom_ids:
            assert bold >= 0 and faint >= 0
            assert task._env.sim.model.geom_rgba[bold, 3] == 0
            assert task._env.sim.model.geom_rgba[faint, 3] == 1
        # Glyph files are cached per word under one instance dir (MuJoCo caches
        # textures by path), so disk use is bounded by the vocabulary.
        assets_dir = Path(task._word_display_assets_dir)
        assert (assets_dir / word.lower()).is_dir()
        assert len(list(assets_dir.iterdir())) <= len(words)
        _, _, _, _, step_info = task.step(np.zeros(task.action_space.shape))
        assert step_info["word"] == word and not step_info["is_success"]
        # Progress in one episode must not leak into the next word's episode.
        task._typed_mask[:] = [True] * len(word)
        task._typed_count = len(word)
    task.close()
    assert not assets_dir.exists()


def test_fixed_word_override_preserves_explicit_arm_partition():
    task = ib.make(TASK, obs_mode="proprio", config_overrides={
        "word": "neurips", "arm0_indices": [0, 1, 2], "arm1_indices": [3, 4, 5, 6],
    })
    try:
        for seed in (0, 7):
            _, info = task.reset(seed=seed)
            assert info["word"] == "NEURIPS"
            assert task._arm0_indices == [0, 1, 2]
            assert task._arm1_indices == [3, 4, 5, 6]
    finally:
        task.close()


def test_default_task_draws_its_word_from_the_training_list_by_seed():
    """The task config itself randomizes the word: same list as the collector, picked by the reset seed."""
    cfg = load_task_config(TASK)
    assert cfg.get("word") is None
    assert [w.upper() for w in cfg["words"]] == load_words()
    task = ib.make(TASK, obs_mode="proprio")
    try:
        drawn = {seed: task.reset(seed=seed)[1]["word"] for seed in range(12)}
        assert set(drawn.values()) <= set(load_words())
        assert len(set(drawn.values())) > 3
        for seed in (3, 0, 3):
            assert task.reset(seed=seed)[1]["word"] == drawn[seed]
    finally:
        task.close()
