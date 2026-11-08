"""Tests for imbench.envs.edge_slide_env.EdgeSlideEnv.

The single-arm variant is already extensively tested in test_edge_slide.py
via ``ib-p1-edge-slide-v1``. These tests fill the gap for env-level
attributes (book_size, book_density, book_friction, book_color).
"""
from __future__ import annotations

import pytest

from imbench.envs.edge_slide_env import EdgeSlideEnv
from imbench.registry import make


# ── Single-arm EdgeSlideEnv attribute round-trips ──────────────────


class TestEdgeSlideEnvAttributes:
    """Env-level attributes set in __init__ should be accessible from the
    task's ``env._env`` handle and reflect the configured values."""

    @pytest.fixture(scope="class")
    @classmethod
    def env(cls):
        task = make(
            "ib-p1-edge-slide-v1",
            obs_mode="proprio",
            render_mode=None,
            seed=0,
            max_episode_steps=3,
        )
        task.reset()
        yield task._env
        task.close()

    def test_is_edge_slide_env(self, env):
        assert isinstance(env, EdgeSlideEnv)

    def test_book_size_is_tuple(self, env):
        assert isinstance(env.book_size, tuple)
        assert len(env.book_size) == 3
        assert all(x > 0 for x in env.book_size)

    def test_book_friction_is_tuple(self, env):
        assert isinstance(env.book_friction, tuple)
        assert len(env.book_friction) == 3

    def test_book_color_is_list_rgba(self, env):
        assert isinstance(env.book_color, list)
        assert len(env.book_color) == 4

    def test_book_density_positive(self, env):
        assert env.book_density > 0

    def test_cube_object_registered(self, env):
        # Object is named "cube" per docstring contract.
        assert env.cube is not None
        assert env.cube.name == "cube"
