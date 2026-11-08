"""Opt-in, fire-and-forget reporter for a data-collection dashboard.

Nothing is sent unless an ingest endpoint is configured.  When one is, a
lightweight POST to ``<api_url>/report`` is issued in a daemon thread
whenever an episode completes; errors are swallowed so reporting never
interrupts data collection.

Configuration (checked in order; the first one found is used):
  1. ``DASHBOARD_API_URL`` environment variable
  2. ``~/.imbench/dashboard.json``  ->  ``{"api_url": "https://..."}``

Example::

    from imbench.utils.dashboard_reporter import report_episode

    report_episode(
        task_name="ib-p1-plate-shelf-hetbi-v1",
        outcome="success",
        score=0.95,
        episode_id="202601000042",
        machine_id="01",
    )
"""

import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

_CONFIG_PATH = Path.home() / ".imbench" / "dashboard.json"
_TIMEOUT = 5  # seconds


def _get_api_url() -> Optional[str]:
    """Resolve the API URL (env var > config file), or ``None`` if unconfigured."""
    url = os.environ.get("DASHBOARD_API_URL")
    if url:
        return url.rstrip("/")
    try:
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        url = cfg.get("api_url") if isinstance(cfg, dict) else None
        if url:
            return url.rstrip("/")
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _post(url: str, payload: dict) -> None:
    """Blocking POST — meant to be called from a daemon thread."""
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=_TIMEOUT):
            pass
    except (URLError, OSError, socket.timeout):
        pass


def report_episode(
    task_name: str,
    outcome: str,
    *,
    score: Optional[float] = None,
    episode_id: Optional[str] = None,
    machine_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Report an episode completion to the configured dashboard.

    No-op unless an endpoint is configured (see module docstring).  When
    one is, the call returns immediately and the HTTP request runs in a
    background daemon thread.

    Parameters
    ----------
    task_name : str
        Task name or ID (e.g. ``"ib-p1-plate-shelf-hetbi-v1"``).
    outcome : str
        ``"success"`` or ``"failure"``.
    score : float, optional
        Final task score (0-1).
    episode_id : str, optional
        Episode identifier (e.g. the run ID).
    machine_id : str, optional
        Identifier for the machine running the collection.
    operator_id : str, optional
        Teleoperator identifier.
    extra : dict, optional
        Any additional key-value pairs to include in the report.
    """
    api_url = _get_api_url()
    if api_url is None:
        return

    payload: dict = {
        "task_name": task_name,
        "outcome": outcome,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if score is not None:
        payload["score"] = round(float(score), 4)
    if episode_id is not None:
        payload["episode_id"] = str(episode_id)
    if machine_id is not None:
        payload["machine_id"] = machine_id
    if operator_id is not None:
        payload["operator_id"] = operator_id
    if extra:
        payload.update(extra)

    url = f"{api_url}/report"
    t = threading.Thread(target=_post, args=(url, payload), daemon=True)
    t.start()
