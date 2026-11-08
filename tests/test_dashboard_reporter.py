"""Tests for imbench.utils.dashboard_reporter.

Covers:
- _get_api_url (env-var > config-file > unconfigured)
- report_episode (opt-in gating, payload construction, thread dispatch)
- _post (network error silencing)

No network access required — HTTP calls are intercepted by monkeypatching.
"""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imbench.utils.dashboard_reporter import (
    _get_api_url,
    _post,
    report_episode,
)

_MISSING_CONFIG = Path("/nonexistent/imbench_dashboard_config.json")


@pytest.fixture()
def unconfigured(monkeypatch):
    """No env var and no config file: reporting must be disabled."""
    monkeypatch.delenv("DASHBOARD_API_URL", raising=False)
    monkeypatch.setattr("imbench.utils.dashboard_reporter._CONFIG_PATH", _MISSING_CONFIG)


# ── _get_api_url ────────────────────────────────────────────────────


class TestGetApiUrl:

    def test_returns_none_when_nothing_set(self, unconfigured):
        assert _get_api_url() is None

    def test_env_var_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_API_URL", "https://my-custom-api.example.com/")
        url = _get_api_url()
        assert url == "https://my-custom-api.example.com"  # trailing slash stripped

    def test_env_var_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_API_URL", "https://example.com/api/")
        url = _get_api_url()
        assert url == "https://example.com/api"

    def test_config_file_used_when_no_env_var(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DASHBOARD_API_URL", raising=False)
        cfg = tmp_path / "dashboard.json"
        cfg.write_text(json.dumps({"api_url": "https://config-file-url.com"}))
        with patch("imbench.utils.dashboard_reporter._CONFIG_PATH", cfg):
            url = _get_api_url()
        assert url == "https://config-file-url.com"

    def test_config_file_strips_trailing_slash(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DASHBOARD_API_URL", raising=False)
        cfg = tmp_path / "dashboard.json"
        cfg.write_text(json.dumps({"api_url": "https://config-file-url.com/"}))
        with patch("imbench.utils.dashboard_reporter._CONFIG_PATH", cfg):
            url = _get_api_url()
        assert url == "https://config-file-url.com"

    def test_malformed_config_means_unconfigured(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DASHBOARD_API_URL", raising=False)
        cfg = tmp_path / "dashboard.json"
        cfg.write_text("not valid json {{{")
        with patch("imbench.utils.dashboard_reporter._CONFIG_PATH", cfg):
            url = _get_api_url()
        assert url is None

    def test_config_file_missing_api_url_key_means_unconfigured(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DASHBOARD_API_URL", raising=False)
        cfg = tmp_path / "dashboard.json"
        cfg.write_text(json.dumps({"other_key": "value"}))
        with patch("imbench.utils.dashboard_reporter._CONFIG_PATH", cfg):
            url = _get_api_url()
        assert url is None

    def test_env_var_overrides_config_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DASHBOARD_API_URL", "https://env-url.com")
        cfg = tmp_path / "dashboard.json"
        cfg.write_text(json.dumps({"api_url": "https://config-url.com"}))
        with patch("imbench.utils.dashboard_reporter._CONFIG_PATH", cfg):
            url = _get_api_url()
        assert url == "https://env-url.com"


# ── _post ───────────────────────────────────────────────────────────


class TestPost:

    def test_swallows_url_error(self):
        """_post must not raise even if the URL is unreachable."""
        from urllib.error import URLError

        with patch(
            "imbench.utils.dashboard_reporter.urlopen",
            side_effect=URLError("connection refused"),
        ):
            _post("https://unreachable.example.com/report", {"key": "val"})

    def test_swallows_os_error(self):
        with patch(
            "imbench.utils.dashboard_reporter.urlopen",
            side_effect=OSError("network error"),
        ):
            _post("https://example.com/report", {"key": "val"})

    def test_swallows_socket_timeout(self):
        import socket

        with patch(
            "imbench.utils.dashboard_reporter.urlopen",
            side_effect=socket.timeout("timed out"),
        ):
            _post("https://example.com/report", {"key": "val"})

    def test_sends_json_content_type(self):
        """Verifies the Request is built with the correct content-type."""
        captured = {}

        def fake_urlopen(req, timeout):
            captured["content_type"] = req.get_header("Content-type")
            return MagicMock().__enter__.return_value

        with patch("imbench.utils.dashboard_reporter.urlopen", fake_urlopen):
            _post("https://example.com/report", {"task": "test"})

        assert captured.get("content_type") == "application/json"

    def test_payload_serialised_as_json(self):
        """The POST body must be valid JSON matching the payload."""
        captured = {}

        def fake_urlopen(req, timeout):
            captured["data"] = req.data
            return MagicMock().__enter__.return_value

        payload = {"task_name": "ib-test", "outcome": "success", "score": 0.9}
        with patch("imbench.utils.dashboard_reporter.urlopen", fake_urlopen):
            _post("https://example.com/report", payload)

        parsed = json.loads(captured["data"].decode())
        assert parsed == payload


# ── report_episode ──────────────────────────────────────────────────


class TestReportEpisode:

    def _capture_post_call(self, monkeypatch):
        """Configure an endpoint and return (calls, event); event is set when _post runs."""
        monkeypatch.setenv("DASHBOARD_API_URL", "https://dashboard.example.invalid")
        calls = []
        event = threading.Event()

        def fake_post(url, payload):
            calls.append((url, payload))
            event.set()

        monkeypatch.setattr("imbench.utils.dashboard_reporter._post", fake_post)
        return calls, event

    def test_nothing_sent_when_unconfigured(self, unconfigured, monkeypatch):
        posted = threading.Event()
        monkeypatch.setattr(
            "imbench.utils.dashboard_reporter._post",
            lambda url, payload: posted.set(),
        )
        with patch("imbench.utils.dashboard_reporter.urlopen") as urlopen:
            assert report_episode("ib-test-task", "success", score=1.0) is None
            assert not posted.wait(timeout=0.2)
        urlopen.assert_not_called()

    def test_dispatches_in_background_thread(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success")
        fired = event.wait(timeout=2.0)
        assert fired, "_post was never called"
        assert len(calls) == 1

    def test_required_fields_in_payload(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "failure")
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        assert payload["task_name"] == "ib-test-task"
        assert payload["outcome"] == "failure"
        assert "timestamp" in payload

    def test_optional_score_included(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success", score=0.87)
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        assert "score" in payload
        assert abs(payload["score"] - 0.87) < 1e-4

    def test_optional_score_omitted_when_none(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success")
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        assert "score" not in payload

    def test_optional_episode_id(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success", episode_id="000042")
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        assert payload["episode_id"] == "000042"

    def test_optional_machine_id(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success", machine_id="gpu-box-1")
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        assert payload["machine_id"] == "gpu-box-1"

    def test_extra_dict_merged_into_payload(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success", extra={"custom_key": "custom_val"})
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        assert payload["custom_key"] == "custom_val"

    def test_score_rounded_to_4_decimal_places(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success", score=0.123456789)
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        # Should be rounded to 4 dp
        assert payload["score"] == round(0.123456789, 4)

    def test_url_includes_report_endpoint(self, monkeypatch):
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success")
        assert event.wait(timeout=2.0)
        url, _ = calls[0]
        assert url == "https://dashboard.example.invalid/report"

    def test_timestamp_is_utc_iso_format(self, monkeypatch):
        """Timestamp must follow ISO 8601 UTC format (ends with 'Z')."""
        calls, event = self._capture_post_call(monkeypatch)
        report_episode("ib-test-task", "success")
        assert event.wait(timeout=2.0)
        _, payload = calls[0]
        ts = payload["timestamp"]
        assert ts.endswith("Z"), f"Expected UTC timestamp ending with 'Z', got {ts!r}"
        # Must also be parseable
        from datetime import datetime
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_returns_none_immediately(self, monkeypatch):
        """report_episode is fire-and-forget and must return None right away."""
        self._capture_post_call(monkeypatch)
        assert report_episode("ib-test-task", "success") is None

    def test_does_not_raise_on_network_failure(self, monkeypatch):
        """A failing request in the background thread must not propagate."""
        from urllib.error import URLError

        monkeypatch.setenv("DASHBOARD_API_URL", "https://dashboard.example.invalid")
        done = threading.Event()

        def failing_urlopen(req, timeout):
            done.set()
            raise URLError("connection refused")

        monkeypatch.setattr("imbench.utils.dashboard_reporter.urlopen", failing_urlopen)
        errors = []
        monkeypatch.setattr(threading, "excepthook", errors.append)
        before = set(threading.enumerate())
        report_episode("ib-test-task", "success", score=0.5)
        assert done.wait(timeout=2.0)
        for thread in set(threading.enumerate()) - before:
            thread.join(timeout=2.0)
        assert errors == []
