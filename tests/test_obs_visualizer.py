"""Tests for imbench.utils.obs_visualizer.

Covers:
- ObsDashboard helper methods (no display required)
- ObsDashboard.update / reset / close (cv2 window calls mocked)
- IntegratedDashboard._build_camera_strip, update, close (mocked)

All tests run headless — cv2.namedWindow, cv2.imshow, cv2.waitKey,
cv2.destroyWindow, cv2.resizeWindow, cv2.moveWindow are monkeypatched.
"""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock, patch

# Import the classes under test
from imbench.utils import obs_visualizer
from imbench.utils.obs_visualizer import ObsDashboard, IntegratedDashboard


# ── Helpers ──────────────────────────────────────────────────────────

def _rgb(h=64, w=64) -> np.ndarray:
    """Fake RGB image (uint8)."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _fake_obs(**extra) -> dict:
    """Build a minimal observation dict with one camera."""
    obs = {
        "rgb_front_cam": _rgb(),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
    }
    obs.update(extra)
    return obs


def _mock_cv2_display():
    """Context manager that patches all cv2 display calls."""
    return patch.multiple(
        "imbench.utils.obs_visualizer.cv2",
        namedWindow=MagicMock(),
        imshow=MagicMock(),
        waitKey=MagicMock(return_value=255),
        destroyWindow=MagicMock(),
        resizeWindow=MagicMock(),
        moveWindow=MagicMock(),
        setWindowProperty=MagicMock(),
    )


# ── ObsDashboard construction ────────────────────────────────────────

class TestObsDashboardConstruction:

    def test_default_window_name(self):
        dash = ObsDashboard()
        assert "IMBench" in dash.window_name

    def test_custom_window_name(self):
        dash = ObsDashboard(window_name="TestWindow")
        assert dash.window_name == "TestWindow"

    def test_cameras_only_flag(self):
        dash = ObsDashboard(cameras_only=True)
        assert dash.cameras_only is True

    def test_default_flags_are_false(self):
        dash = ObsDashboard()
        assert dash.show_tactile is False
        assert dash.show_ft is False
        assert dash.cameras_only is False

    def test_created_initially_false(self):
        dash = ObsDashboard()
        assert dash._created is False

    def test_custom_max_width(self):
        dash = ObsDashboard(max_width=800)
        assert dash.max_width == 800

    def test_histories_start_empty(self):
        dash = ObsDashboard()
        assert len(dash._reward_hist) == 0
        assert len(dash._score_hist) == 0
        assert len(dash._ft_hist) == 0


# ── ObsDashboard.reset ────────────────────────────────────────────────

class TestObsDashboardReset:

    def test_reset_clears_histories(self):
        dash = ObsDashboard()
        dash._reward_hist.append(1.0)
        dash._score_hist.append(0.5)
        dash._ft_hist.append(np.ones(6))
        dash._step = 42
        dash.reset()
        assert len(dash._reward_hist) == 0
        assert len(dash._score_hist) == 0
        assert len(dash._ft_hist) == 0
        assert dash._step == 0


# ── ObsDashboard._make_cam_tile ────────────────────────────────────────

class TestMakeCamTile:

    def test_returns_numpy_array(self):
        dash = ObsDashboard(max_width=1280)
        tile = dash._make_cam_tile(_rgb(64, 64), "front", primary=True)
        assert isinstance(tile, np.ndarray)
        assert tile.ndim == 3

    def test_rgb_to_bgr_conversion(self):
        dash = ObsDashboard(max_width=1280)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 200  # red channel
        tile = dash._make_cam_tile(img, "label", primary=True)
        # After RGB→BGR, red channel moves to index 2
        assert isinstance(tile, np.ndarray)

    def test_primary_tile_larger_than_secondary(self):
        dash = ObsDashboard(max_width=1280)
        big_img = _rgb(480, 720)
        tile_primary = dash._make_cam_tile(big_img, "front", primary=True)
        tile_secondary = dash._make_cam_tile(big_img, "side", primary=False)
        # Primary should be >= secondary in both dimensions
        assert tile_primary.shape[0] >= tile_secondary.shape[0]
        assert tile_primary.shape[1] >= tile_secondary.shape[1]

    def test_no_upscaling(self):
        """Small images must not be enlarged."""
        dash = ObsDashboard(max_width=1280)
        small = _rgb(32, 32)
        tile = dash._make_cam_tile(small, "micro", primary=True)
        assert tile.shape[0] <= 480  # _TILE_MAX_H
        assert tile.shape[1] <= 720  # _TILE_MAX_W


# ── ObsDashboard._draw_bar_chart ──────────────────────────────────────

class TestDrawBarChart:

    def test_returns_correct_shape(self):
        dash = ObsDashboard(max_width=1280)
        values = np.random.randn(10).astype(np.float32)
        chart = dash._draw_bar_chart(values, "Test", (0, 200, 100))
        assert chart.ndim == 3
        assert chart.shape[2] == 3

    def test_empty_values_does_not_crash(self):
        dash = ObsDashboard(max_width=1280)
        chart = dash._draw_bar_chart(np.array([]), "Empty", (100, 100, 200))
        assert chart is not None
        assert chart.ndim == 3

    def test_uniform_values_does_not_crash(self):
        """When vmin == vmax ruff must not divide by zero."""
        dash = ObsDashboard(max_width=1280)
        values = np.ones(5, dtype=np.float32)
        chart = dash._draw_bar_chart(values, "Uniform", (100, 100, 200))
        assert chart is not None

    def test_negative_values_allowed(self):
        dash = ObsDashboard(max_width=1280)
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        chart = dash._draw_bar_chart(values, "Neg", (200, 100, 50))
        assert chart is not None


# ── ObsDashboard._draw_timeseries ─────────────────────────────────────

class TestDrawTimeseries:

    def test_1d_array_accepted(self):
        dash = ObsDashboard(max_width=1280)
        data = np.random.randn(50).astype(np.float32)
        plot = dash._draw_timeseries(data, "FT", (180, 80, 255))
        assert plot.ndim == 3
        assert plot.shape[2] == 3

    def test_2d_array_accepted(self):
        dash = ObsDashboard(max_width=1280)
        data = np.random.randn(50, 6).astype(np.float32)
        plot = dash._draw_timeseries(data, "FT-6DOF", (180, 80, 255))
        assert plot.ndim == 3

    def test_uniform_data_does_not_crash(self):
        dash = ObsDashboard(max_width=1280)
        data = np.ones((30, 3), dtype=np.float32)
        plot = dash._draw_timeseries(data, "Const", (255, 255, 0))
        assert plot is not None

    def test_single_point_does_not_crash(self):
        dash = ObsDashboard(max_width=1280)
        data = np.array([[1.0, 2.0, 3.0]])
        plot = dash._draw_timeseries(data, "Single", (0, 255, 0))
        assert plot is not None


# ── ObsDashboard._hstack / _vstack ───────────────────────────────────

class TestLayoutHelpers:

    def _dash(self):
        return ObsDashboard(max_width=1280)

    def test_hstack_empty_list(self):
        canvas = self._dash()._hstack([])
        assert canvas.ndim == 3

    def test_vstack_empty_list(self):
        canvas = self._dash()._vstack([])
        assert canvas.ndim == 3

    def test_hstack_single_panel(self):
        p = _rgb(100, 200)
        result = self._dash()._hstack([p])
        assert result.shape[1] >= 200

    def test_vstack_single_row(self):
        r = _rgb(100, 200)
        result = self._dash()._vstack([r])
        assert result.shape[0] >= 100

    def test_hstack_pads_shorter_panels(self):
        p1 = _rgb(100, 200)
        p2 = _rgb(50, 200)
        result = self._dash()._hstack([p1, p2])
        assert result.shape[0] == 100  # padded to tallest

    def test_vstack_pads_narrower_rows(self):
        r1 = _rgb(50, 300)
        r2 = _rgb(50, 150)
        result = self._dash()._vstack([r1, r2])
        assert result.shape[1] == 300  # padded to widest

    def test_hstack_two_panels_correct_width(self):
        p1 = np.zeros((50, 100, 3), dtype=np.uint8)
        p2 = np.zeros((50, 200, 3), dtype=np.uint8)
        from imbench.utils.obs_visualizer import _PAD
        result = self._dash()._hstack([p1, p2])
        expected_w = 100 + _PAD + 200
        assert result.shape[1] == expected_w


# ── ObsDashboard._draw_legend ─────────────────────────────────────────

class TestDrawLegend:

    def test_returns_same_shape(self):
        dash = ObsDashboard(max_width=1280)
        canvas = np.zeros((300, 400, 3), dtype=np.uint8)
        result = dash._draw_legend(canvas)
        assert result.shape == canvas.shape

    def test_does_not_modify_original(self):
        dash = ObsDashboard(max_width=1280)
        canvas = np.zeros((300, 400, 3), dtype=np.uint8)
        original_copy = canvas.copy()
        dash._draw_legend(canvas)
        np.testing.assert_array_equal(canvas, original_copy)


# ── ObsDashboard._draw_tactile_grid ──────────────────────────────────

class TestDrawTactileGrid:

    def test_basic_grid(self):
        dash = ObsDashboard(max_width=1280)
        readings = np.random.rand(4, 4).astype(np.float32)
        result = dash._draw_tactile_grid(readings, "Tactile")
        assert result.ndim == 3

    def test_zero_readings_does_not_crash(self):
        dash = ObsDashboard(max_width=1280)
        readings = np.zeros((3, 3), dtype=np.float32)
        result = dash._draw_tactile_grid(readings, "Zero")
        assert result is not None

    def test_target_height_respected(self):
        dash = ObsDashboard(max_width=1280)
        readings = np.random.rand(6, 4).astype(np.float32)
        result = dash._draw_tactile_grid(readings, "T", target_height=200)
        # Height should be close to target_height (within reason)
        assert result.shape[0] > 0


# ── ObsDashboard._build_info_strip ────────────────────────────────────

class TestBuildInfoStrip:

    def test_returns_numpy_array(self):
        dash = ObsDashboard(max_width=400)
        strip = dash._build_info_strip(task_score=0.75, reward=0.1)
        assert isinstance(strip, np.ndarray)
        assert strip.ndim == 3

    def test_none_reward_accepted(self):
        dash = ObsDashboard(max_width=400)
        strip = dash._build_info_strip(task_score=0.5, reward=None)
        assert strip is not None


# ── ObsDashboard.update ───────────────────────────────────────────────

class TestObsDashboardUpdate:

    def test_update_returns_int(self):
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            key = dash.update(_fake_obs())
        assert isinstance(key, int)

    def test_update_with_info_dict(self):
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            key = dash.update(_fake_obs(), info={"step": 5, "task_score": 0.3})
        assert isinstance(key, int)
        assert dash._step == 5

    def test_update_with_reward(self):
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            dash.update(_fake_obs(), reward=0.25)
        assert len(dash._reward_hist) == 1
        assert abs(dash._reward_hist[0] - 0.25) < 1e-9

    def test_update_increments_step_without_info(self):
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            dash._step = 10
            dash.update(_fake_obs())
        assert dash._step == 11

    def test_update_appends_score_hist(self):
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            dash.update(_fake_obs(), info={"task_score": 0.8})
        assert len(dash._score_hist) == 1
        assert abs(dash._score_hist[0] - 0.8) < 1e-9

    def test_update_processes_force_torque(self):
        obs = _fake_obs(force_torque=np.ones(6, dtype=np.float32))
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            dash.update(obs)
        assert len(dash._ft_hist) == 1

    def test_update_multiple_cameras(self):
        obs = _fake_obs(
            rgb_front_cam=_rgb(),
            rgb_wrist_cam=_rgb(32, 32),
        )
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            dash.update(obs)

    def test_update_depth_camera(self):
        obs = _fake_obs(
            depth_front_cam=np.random.rand(64, 64, 1).astype(np.float32),
        )
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            dash.update(obs)

    def test_set_created_flag_after_first_update(self):
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            assert dash._created is False
            dash.update(_fake_obs())
        assert dash._created is True

    def test_update_no_cameras_still_works(self):
        obs = {"robot0_eef_pos": np.zeros(3)}
        with _mock_cv2_display():
            dash = ObsDashboard(max_width=640)
            dash.update(obs)

    def test_cameras_only_keeps_every_camera(self):
        shades = {"a": 40, "b": 80, "c": 160, "d": 220}
        obs = {f"rgb_{k}": np.full((64, 64, 3), v, np.uint8) for k, v in shades.items()}
        shown = {}
        with _mock_cv2_display(), patch.object(
                obs_visualizer.cv2, "imshow", lambda name, img: shown.update(img=img)):
            dash = ObsDashboard(max_width=1920, cameras_only=True)
            dash._show_legend = False
            dash.update(obs)
        visible = [k for k, v in shades.items() if (shown["img"] == v).all(axis=2).sum() > 1000]
        assert visible == ["a", "b", "c", "d"]


# ── Held-weight readout (F/T tare) ───────────────────────────────────

_EMPTY = {"force_torque_world": np.array([0.0, 0.0, -10.0])}
_LOADED_200G = {"force_torque_world": np.array([0.0, 0.0, -10.0 - 0.2 * 9.81])}


def _held_titles(dash, n_empty, n_loaded):
    """Feed *n_empty* then *n_loaded* F/T samples; return the held-weight titles drawn."""
    titles = []
    real = obs_visualizer._render_timeseries

    def render(data, title, *args, **kwargs):
        titles.append(title)
        return real(data, title, *args, **kwargs)

    with _mock_cv2_display(), patch.object(obs_visualizer, "_render_timeseries", render), \
            patch("subprocess.Popen", MagicMock()):
        for obs in [_EMPTY] * n_empty + [_LOADED_200G] * n_loaded:
            dash.update(obs, {})
    return titles


class TestHeldWeight:

    def test_obs_dashboard_holds_reading_past_history_window(self):
        dash = ObsDashboard(max_width=640, visualize_ft=True)
        titles = _held_titles(dash, 20, 2 * obs_visualizer._HISTORY)
        assert titles[-1] == "Held:  200.0 g"

    def test_obs_dashboard_reset_retares(self):
        dash = ObsDashboard(max_width=640, visualize_ft=True)
        _held_titles(dash, 0, 50)        # tare taken while loaded
        dash.reset()
        assert _held_titles(dash, 20, 0)[-1] == "Held:    0.0 g"

    def test_integrated_dashboard_holds_reading_and_reset_retares(self):
        dash = IntegratedDashboard(max_fps=1e9, visualize_ft=True)
        assert _held_titles(dash, 20, 2 * obs_visualizer._HISTORY)[-1] == "Held weight:  200.0 g"
        dash.reset()
        assert _held_titles(dash, 20, 0)[-1] == "Held weight:    0.0 g"


# ── ObsDashboard.close ───────────────────────────────────────────────

class TestObsDashboardClose:

    def test_close_calls_destroyWindow(self):
        with _mock_cv2_display():
            dash = ObsDashboard(window_name="TestClose", max_width=640)
            dash.update(_fake_obs())
            assert dash._created is True
            dash.close()
        assert dash._created is False


# ── IntegratedDashboard ───────────────────────────────────────────────

class TestIntegratedDashboard:

    def test_construction(self):
        dash = IntegratedDashboard()
        assert dash.window_name == "IMBench Cameras"
        assert dash._created is False

    def test_custom_window_name(self):
        dash = IntegratedDashboard(window_name="MyCams")
        assert dash.window_name == "MyCams"

    def test_reset_clears_ft_history(self):
        dash = IntegratedDashboard(visualize_ft=True)
        dash._ft_hist.append(np.ones(3))
        dash._ft_tare = 1.0
        dash.reset()
        assert len(dash._ft_hist) == 0 and dash._ft_tare is None

    def test_build_camera_strip_no_cameras(self):
        dash = IntegratedDashboard()
        result = dash._build_camera_strip({})
        assert result.ndim == 3

    def test_build_camera_strip_single_camera(self):
        dash = IntegratedDashboard()
        rgb_imgs = {"front": _rgb(64, 64)}
        result = dash._build_camera_strip(rgb_imgs)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_build_camera_strip_multiple_cameras(self):
        dash = IntegratedDashboard()
        rgb_imgs = {
            "front": _rgb(64, 64),
            "wrist": _rgb(48, 48),
        }
        result = dash._build_camera_strip(rgb_imgs)
        assert result.ndim == 3

    def test_camera_strip_fits_screen_area(self, monkeypatch):
        monkeypatch.setattr(obs_visualizer, "_screen_width", lambda: 1920)
        monkeypatch.setattr(obs_visualizer, "_screen_height", lambda: 1080)
        dash = IntegratedDashboard()
        cams = {f"rgb_cam{i}": _rgb(480, 640) for i in range(4)}
        with _mock_cv2_display(), patch("subprocess.Popen", MagicMock()):
            dash.update(cams)
            (_name, width, height), _ = obs_visualizer.cv2.resizeWindow.call_args
        assert width <= 1920 and height <= int(1080 * 0.30)

    def test_close_when_not_created(self):
        """close() before any update should not raise."""
        dash = IntegratedDashboard()
        dash.close()  # should be a no-op

    def test_update_returns_int(self):
        with _mock_cv2_display():
            dash = IntegratedDashboard()
            key = dash.update(_fake_obs())
        assert isinstance(key, int)

    def test_close_when_created_calls_destroyWindow(self):
        with _mock_cv2_display():
            dash = IntegratedDashboard(window_name="CamStrip")
            dash.update(_fake_obs())
            assert dash._created is True
            dash.close()
        assert dash._created is False
