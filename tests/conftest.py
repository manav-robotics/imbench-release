"""Shared pytest fixtures for IMBench.

Session-scoped task cache avoids rebuilding the same MuJoCo + robosuite
environment once per test class. Class-scoped fixtures elsewhere can
route through :func:`cached_task` so that, e.g., ``ib-p1-bowl-table-v1``
is only constructed once per worker across an entire test session.

All cached tasks are closed at session teardown.
"""
from __future__ import annotations

import pytest

from imbench.registry import make


_CACHE: dict[tuple, object] = {}


def cached_task(task_id: str, **kwargs):
    """Return a shared task for ``(task_id, kwargs)`` within this session.

    The returned task has already had ``reset()`` called at least once,
    but callers should still ``.reset()`` before relying on state.

    Not thread-safe — xdist uses one process per worker, so each worker
    owns its own cache. Do not call ``task.close()`` on the returned
    value; session teardown handles cleanup.
    """
    key = (task_id, tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        task = make(task_id, **kwargs)
        task.reset()
        _CACHE[key] = task
    return _CACHE[key]


@pytest.fixture(scope="session", autouse=True)
def _close_cached_tasks():
    """Close every cached task at session end."""
    yield
    for task in _CACHE.values():
        try:
            task.close()
        except Exception:
            pass
    _CACHE.clear()
