"""Real-time observation dashboard.

:class:`ObsDashboard` renders the env's observations into a single CV2 window:
  - the first two RGB cameras across the top row
  - further RGB cameras and colormapped depth images in a second row
  - held-weight trace from the force/torque sensor (tasks with
    ``visualize_FT``; toggle with F)
  - tactile grids (toggle with T)

:class:`IntegratedDashboard` shows all RGB cameras in a strip along the
bottom of the screen, below the MuJoCo viewer.

Usage:
    from imbench.utils.obs_visualizer import ObsDashboard

    dash = ObsDashboard()
    obs, info = env.reset()
    dash.update(obs, info)       # call each step
    dash.close()
"""

from __future__ import annotations

from collections import deque
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Layout constants ─────────────────────────────────────────────────
_TILE_MAX_H = 480      # max height of each camera tile (original res if smaller)
_TILE_MAX_W = 720      # max width of each camera tile (original res if smaller)
_TILE_SEC_H = 240   
_TILE_SEC_W = 360    
_N_PRIMARY  = 2
_BAR_H = 140           # height of bar chart panels
_PLOT_H = 120          # height of time-series panel
_PAD = 6               # padding between panels
_BG = (30, 30, 30)     # dashboard background
_TEXT_COLOR = (220, 220, 220)
_ACCENT = (0, 200, 120)
_WARN = (0, 100, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_S = 0.4
_HISTORY = 200         # time-series buffer length
_N_TARE = 10           # F/T samples after reset averaged for the held-weight tare

# Key bindings  {key_code: (attribute_name, display_label)}
_TOGGLES = {
    ord('t'): ('show_tactile',      'Tactile'),
    ord('f'): ('show_ft',           'Force/Torque'),
    # ord('i'): ('show_info_strip',   'Info Strip'),
}
_KEY_LEGEND = ord('h')
_KEY_QUIT   = ord('q')


def _render_timeseries(
    data: np.ndarray,
    title: str,
    color: tuple[int, int, int],
    width: int = 400,
    height: int = _PLOT_H,
    y_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a (T, C) time series as a CV2 image tile. Pure function.

    ``y_range`` locks the y-axis to ``(vmin, vmax)``; values are clipped
    to stay within the plot area. Defaults to auto-scale from data.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    T, C = data.shape
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)
    cv2.putText(canvas, title, (4, 14), _FONT, _FONT_S, _TEXT_COLOR, 1, cv2.LINE_AA)

    margin_l, margin_r, margin_t, margin_b = 40, 10, 22, 12
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    if y_range is not None:
        vmin, vmax = float(y_range[0]), float(y_range[1])
    else:
        vmin, vmax = float(data.min()), float(data.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    colors = [color, (100, 200, 255), (100, 255, 100),
              (255, 150, 100), (255, 100, 255), (255, 255, 100)]

    for c in range(C):
        ch_color = colors[c % len(colors)]
        pts = []
        for t in range(T):
            x = margin_l + int(t / max(T - 1, 1) * plot_w)
            frac = (data[t, c] - vmin) / (vmax - vmin)
            frac = min(1.0, max(0.0, frac))  # clip to plot area
            y = margin_t + plot_h - int(frac * plot_h)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], ch_color, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{vmax:.2f}", (2, margin_t + 10), _FONT, 0.3, (150, 150, 150), 1)
    cv2.putText(canvas, f"{vmin:.2f}", (2, margin_t + plot_h), _FONT, 0.3, (150, 150, 150), 1)
    return canvas


def _vertical_load(ft_hist) -> np.ndarray:
    """Fz of each sample in an F/T history."""
    ft_arr = np.array(ft_hist)
    return ft_arr[:, 2] if ft_arr.ndim == 2 and ft_arr.shape[1] >= 3 else ft_arr.reshape(-1)


def _held_mass_g(ft_hist, tare: float | None) -> np.ndarray:
    """Held mass in grams per F/T sample, ``|Fz - tare| / g``.

    Until the tare is set the first sample stands in for it.
    """
    fz = _vertical_load(ft_hist)
    baseline = tare if tare is not None else float(fz[0])
    return np.abs(fz - baseline) / 9.81 * 1000.0


def _tare(ft_hist, tare: float | None) -> float | None:
    """*tare*, or the mean Fz of the first ``_N_TARE`` samples once *ft_hist* holds them.

    The tare is taken once per episode (empty gripper) and then stays fixed
    while the history window slides.
    """
    if tare is None and len(ft_hist) >= _N_TARE:
        return float(np.mean(_vertical_load(ft_hist)[:_N_TARE]))
    return tare


def _screen_width() -> int:
    """Return the primary screen width, falling back to 1920 if undetectable."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        root.destroy()
        return w
    except Exception:
        return 1920


def _screen_height() -> int:
    """Return the primary screen height, falling back to 1080 if undetectable."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        h = root.winfo_screenheight()
        root.destroy()
        return h
    except Exception:
        return 1080


class ObsDashboard:
    """Displays a live observation dashboard in a CV2 window."""

    def __init__(
        self,
        window_name: str = "IMBench Obs Dashboard",
        max_width: int | None = None,
        cameras_only: bool = False,
        fullscreen: bool = False,
        visualize_ft: bool = False,
    ) -> None:
        self.window_name = window_name
        self.max_width = max_width if max_width is not None else _screen_width()
        self.cameras_only = cameras_only
        self.fullscreen = fullscreen
        self.visualize_ft = visualize_ft
        self._reward_hist = deque(maxlen=_HISTORY)
        self._score_hist = deque(maxlen=_HISTORY)
        self._ft_hist = deque(maxlen=_HISTORY)
        self._ft_tare: float | None = None
        self._step = 0
        self._created = False

        # Runtime-toggle flags. show_ft defaults on when the task config
        # opts in via visualize_FT so the user sees wrench data immediately.
        self.show_tactile      = False
        self.show_ft           = visualize_ft
        self.show_proprio      = False
        self.show_object_state = False
        self.show_info_strip   = False
        self._show_legend      = True   # visible on first launch so users discover it

    # ── Public API ───────────────────────────────────────────────────

    def update(self, obs: dict[str, np.ndarray], info: dict | None = None, reward: float | None = None) -> int:
        """Render one frame of the dashboard. Returns the cv2 waitKey code."""
        info = info or {}
        self._step = info.get("step", self._step + 1)
        task_score = info.get("task_score", 0.0)
        if reward is not None:
            self._reward_hist.append(reward)
        self._score_hist.append(task_score)

        # Separate modalities
        rgb_imgs = {}
        depth_imgs = {}
        tactile_grids = {}  # "robot0_right" → (grid, grid) array

        for k, v in obs.items():
            if k.startswith("rgb_"):
                label = k[4:]
                if label == "ego_cam":
                    v = np.ascontiguousarray(v[:, ::-1])
                rgb_imgs[label] = v
            elif k.startswith("depth_"):
                depth_imgs[k[6:]] = v
            elif "_tactile_" in k:
                tactile_grids[k] = v if v.ndim == 2 else v.reshape(-1, 1)

        # Prefer world-frame force for the Fz plot (proxy for held weight).
        if "force_torque_world" in obs:
            self._ft_hist.append(np.asarray(obs["force_torque_world"]).copy())
        elif "force_torque" in obs:
            self._ft_hist.append(np.asarray(obs["force_torque"])[:3].copy())
        self._ft_tare = _tare(self._ft_hist, self._ft_tare)

        # ── Build rows ───────────────────────────────────────────────
        rows = []

        # Row 1: camera tiles
        cam_row = self._build_camera_row(rgb_imgs, depth_imgs)
        if cam_row is not None:
            rows.append(cam_row)

        # Row 2: secondary cameras and depth, then the force/torque
        # time-series and tactile grids (skipped in cameras_only mode)
        ts_panels = []
        for label, img in self._secondary_items:
            ts_panels.append(self._make_cam_tile(img, label, primary=False))
        for label, img in self._depth_items:
            d      = img.squeeze()
            d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d_col  = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
            ts_panels.append(self._make_cam_tile(d_col, f"depth_{label}", primary=False))
        if not self.cameras_only:
            if self.visualize_ft and self.show_ft and len(self._ft_hist) >= 1:
                mass_g = _held_mass_g(self._ft_hist, self._ft_tare)
                title = f"Held: {mass_g[-1]:6.1f} g"
                ts_panels.append(_render_timeseries(
                    mass_g.reshape(-1, 1), title, (180, 80, 255),
                    width=400, height=_PLOT_H, y_range=(0.0, 500.0),
                ))
            if self.show_tactile:
                for label, tgrid in sorted(tactile_grids.items()):
                    ts_panels.append(self._draw_tactile_grid(tgrid, label, cell_size=32, gap=3, target_height=_TILE_SEC_H))
        if ts_panels:
            rows.append(self._hstack(ts_panels))

        # ── Compose final canvas ─────────────────────────────────────
        canvas = self._vstack(rows) if rows else self._blank(200, 400)

        if self._show_legend:
            canvas = self._draw_legend(canvas)

        if not self._created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            if self.fullscreen:
                cv2.setWindowProperty(
                    self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
            cv2.moveWindow(self.window_name, 0, 0)
            self._created = True
            self._last_canvas_shape = (0, 0)

        # Re-size the window whenever the canvas dimensions change (e.g. the
        # Fz panel appears a frame after the first render, growing row 2).
        if not self.fullscreen and canvas.shape[:2] != self._last_canvas_shape:
            cv2.resizeWindow(
                self.window_name,
                min(canvas.shape[1], self.max_width),
                min(canvas.shape[0], 900),
            )
            self._last_canvas_shape = canvas.shape[:2]

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
 
        # ── Handle key presses ───────────────────────────────────────────
        if key in _TOGGLES:
            attr, _ = _TOGGLES[key]
            if attr == "show_ft" and not self.visualize_ft:
                pass  # F/T viz disabled by task config — ignore the toggle
            else:
                new_val = not getattr(self, attr)
                setattr(self, attr, new_val)
                # Clear stale F/T data when the panel is hidden
                if attr == "show_ft" and not new_val:
                    self._ft_hist.clear()
        elif key == _KEY_LEGEND:
            self._show_legend = not self._show_legend
 
        return key

    def reset(self) -> None:
        """Clear histories and the F/T tare for a new episode."""
        self._reward_hist.clear()
        self._score_hist.clear()
        self._ft_hist.clear()
        self._ft_tare = None
        self._step = 0

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
        self._created = False

    # ── Camera row ───────────────────────────────────────────────────

    def _build_camera_row(
        self,
        rgb_imgs: dict[str, np.ndarray],
        depth_imgs: dict[str, np.ndarray],
    ) -> np.ndarray:
        rgb_items   = list(rgb_imgs.items())   # preserves insertion order
        self._depth_items = list(depth_imgs.items())

        primary_items   = rgb_items[:_N_PRIMARY]
        self._secondary_items = rgb_items[_N_PRIMARY:] 

        # ── Primary row ──────────────────────────────────────────────────
        primary_tiles = [
            self._make_cam_tile(img, label, primary=True)
            for label, img in primary_items
        ]

        return self._hstack(primary_tiles) if primary_tiles else None

    def _make_cam_tile(self, img, label, *, primary: bool = True):
        # Robosuite images are RGB, CV2 expects BGR
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Only downscale if the image exceeds the max tile size; never upscale.
        max_h = _TILE_MAX_H if primary else _TILE_SEC_H
        max_w = _TILE_MAX_W if primary else _TILE_SEC_W

        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = min(max_h / h, max_w / w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
        tile = img.copy()
        tw = tile.shape[1]
        # label overlay
        cv2.rectangle(tile, (0, 0), (tw, 20), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 14), _FONT, _FONT_S, _TEXT_COLOR, 1, cv2.LINE_AA)
        return tile
    
    # ── Legend overlay ────────────────────────────────────────────────────
 
    def _draw_legend(self, canvas: np.ndarray) -> np.ndarray:
        """Render a semi-transparent toggle cheat-sheet in the top-right corner."""
        canvas = canvas.copy()
 
        # Build text lines
        lines = [(" Panels", None)]
        for key_code, (attr, label) in _TOGGLES.items():
            active = getattr(self, attr)
            marker = "[x]" if active else "[ ]"
            state  = " ON" if active else "off"
            lines.append((f"  {marker} [{chr(key_code).upper()}] {label:<16}{state}", active))
        lines.append(("  [H] Hide legend", None))
        lines.append(("  [Q] Quit", None))
 
        line_h  = 17
        padding = 8
        box_w   = 210
        box_h   = len(lines) * line_h + padding * 2
 
        # Position: top-right, with a small margin
        x0 = canvas.shape[1] - box_w - 10
        y0 = 10
        x1 = min(x0 + box_w, canvas.shape[1])
        y1 = min(y0 + box_h, canvas.shape[0])
        x0 = max(x0, 0)
        y0 = max(y0, 0)
 
        # Semi-transparent dark background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (65, 65, 65), 1)
 
        for i, (line, active) in enumerate(lines):
            tx = x0 + padding
            ty = y0 + padding + i * line_h + 11
            if active is True:
                color = _ACCENT          # green  → panel is on
            elif active is False:
                color = (120, 120, 120)  # dim    → panel is off
            else:
                color = _TEXT_COLOR      # white  → header / footer
            cv2.putText(canvas, line, (tx, ty), _FONT, 0.34, color, 1, cv2.LINE_AA)
 
        return canvas

    # ── Bar chart ────────────────────────────────────────────────────

    def _draw_bar_chart(
        self,
        values: np.ndarray,
        title: str,
        color: tuple[int, int, int],
        width: int | None = None,
    ) -> np.ndarray:
        n = len(values)
        w = width or max(n * 8 + 60, 300)
        canvas = np.full((_BAR_H, w, 3), _BG, dtype=np.uint8)

        # title
        cv2.putText(canvas, title, (4, 14), _FONT, _FONT_S, _TEXT_COLOR, 1, cv2.LINE_AA)

        if n == 0:
            return canvas

        margin_l, margin_r, margin_t, margin_b = 40, 10, 24, 16
        plot_w = w - margin_l - margin_r
        plot_h = _BAR_H - margin_t - margin_b
        bar_w = max(plot_w // n - 1, 1)

        vmin, vmax = float(values.min()), float(values.max())
        if vmax == vmin:
            vmax = vmin + 1.0

        # zero line
        zero_y = margin_t + int(plot_h * vmax / (vmax - vmin))
        zero_y = np.clip(zero_y, margin_t, margin_t + plot_h)
        cv2.line(canvas, (margin_l, zero_y), (w - margin_r, zero_y), (80, 80, 80), 1)

        for i, v in enumerate(values):
            x = margin_l + i * (bar_w + 1)
            frac = (v - vmin) / (vmax - vmin)
            bar_h_px = max(int(frac * plot_h), 1)
            y_top = margin_t + plot_h - bar_h_px
            cv2.rectangle(canvas, (x, y_top), (x + bar_w, margin_t + plot_h), color, -1)

        # axis labels
        cv2.putText(canvas, f"{vmax:.2f}", (2, margin_t + 10), _FONT, 0.3, (150, 150, 150), 1)
        cv2.putText(canvas, f"{vmin:.2f}", (2, margin_t + plot_h), _FONT, 0.3, (150, 150, 150), 1)
        cv2.putText(canvas, f"[{n}]", (w - 40, _BAR_H - 4), _FONT, 0.3, (120, 120, 120), 1)

        return canvas

    # ── Time-series plot ─────────────────────────────────────────────

    def _draw_timeseries(
        self,
        data: np.ndarray,
        title: str,
        color: tuple[int, int, int],
        width: int = 400,
    ) -> np.ndarray:
        """data: (T, C) array. Plots each channel as a separate line."""
        canvas = _render_timeseries(data, title, color, width=width, height=_PLOT_H)

        return canvas

    # ── Info strip ───────────────────────────────────────────────────

    def _build_info_strip(self, task_score: float, reward: float | None) -> np.ndarray:
        h = 28
        w = self.max_width
        strip = np.full((h, w, 3), _BG, dtype=np.uint8)
        parts = [f"Step: {self._step}"]
        parts.append(f"Score: {task_score:.3f}")
        if reward is not None:
            parts.append(f"Reward: {reward:.4f}")
        score_color = _ACCENT if task_score > 0.5 else _WARN if task_score > 0.1 else _TEXT_COLOR
        text = "   |   ".join(parts)
        cv2.putText(strip, text, (8, 18), _FONT, 0.5, score_color, 1, cv2.LINE_AA)
        return strip

    def _draw_tactile_grid(
        self,
        readings: np.ndarray,
        title: str = "Tactile",
        cell_size: int = 32,
        gap: int = 3,
        target_height: int | None = None,
    ) -> np.ndarray:
        rows, cols = readings.shape

        if target_height is not None:
            margin_t, margin_b = 22, 28
            available = target_height - margin_t - margin_b
            cell_size = max(8, (available - gap * (rows - 1)) // rows)

        margin_t, margin_b, margin_l, margin_r = 22, 28, 8, 8
        w = margin_l + cols * (cell_size + gap) - gap + margin_r
        h = margin_t + rows * (cell_size + gap) - gap + margin_b
        canvas = np.full((h, w, 3), _BG, dtype=np.uint8)

        cv2.putText(canvas, title, (4, 14), _FONT, _FONT_S, _TEXT_COLOR, 1, cv2.LINE_AA)

        vmax = float(readings.max())
        if vmax < 1e-6:
            vmax = 1.0

        # Build a small image of the grid values, then resize with interpolation
        # so the gradient flows continuously across cell boundaries
        norm = (readings / vmax * 255).astype(np.uint8)
        small = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)  # dark→purple→red→yellow

        grid_w = cols * (cell_size + gap) - gap
        grid_h = rows * (cell_size + gap) - gap
        smooth = cv2.resize(small, (grid_w, grid_h), interpolation=cv2.INTER_CUBIC)

        canvas[margin_t:margin_t + grid_h, margin_l:margin_l + grid_w] = smooth

        # colour-scale bar
        bar_y = h - margin_b + 4
        bar_w = w - margin_l - margin_r
        bar_row = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, bar_w)
        bar_colored = cv2.applyColorMap(bar_row, cv2.COLORMAP_INFERNO)  # shape (1, bar_w, 3)
        bar_colored = np.repeat(bar_colored, 6, axis=0)                 # stretch to 6px tall
        canvas[bar_y:bar_y + 6, margin_l:margin_l + bar_w] = bar_colored

        cv2.putText(canvas, "0", (margin_l, h - 1), _FONT, 0.28, (120, 120, 120), 1)
        cv2.putText(canvas, f"{vmax:.1f}", (w - 28, h - 1), _FONT, 0.28, (120, 120, 120), 1)

        return canvas

    # ── Layout helpers ───────────────────────────────────────────────

    def _blank(self, h: int, w: int) -> np.ndarray:
        """Return a blank background-coloured canvas."""
        return np.full((h, w, 3), _BG, dtype=np.uint8)

    def _hstack(self, panels: list[np.ndarray]) -> np.ndarray:
        """Horizontally stack panels, padding shorter ones."""
        if not panels:
            return np.full((10, 10, 3), _BG, dtype=np.uint8)
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            if p.shape[0] < max_h:
                pad = np.full((max_h - p.shape[0], p.shape[1], 3), _BG, dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
            # separator
            padded.append(np.full((max_h, _PAD, 3), _BG, dtype=np.uint8))
        return np.hstack(padded[:-1])  # drop trailing separator

    def _vstack(self, rows: list[np.ndarray]) -> np.ndarray:
        """Vertically stack rows, padding narrower ones."""
        if not rows:
            return np.full((10, 10, 3), _BG, dtype=np.uint8)
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.full((r.shape[0], max_w - r.shape[1], 3), _BG, dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
            # separator
            padded.append(np.full((_PAD, max_w, 3), _BG, dtype=np.uint8))
        return np.vstack(padded[:-1])


class IntegratedDashboard:
    """Camera strip window: all obs cameras in a single horizontal line.

    Occupies the bottom 30 % of the screen so the MuJoCo robosuite viewer can
    be placed in the top 70 %.

    Parameters
    ----------
    window_name : str
        CV2 window title.
    max_fps : int
        Maximum display refresh rate (default 25 Hz).
    """

    def __init__(
        self,
        window_name: str = "IMBench Cameras",
        max_fps: int = 25,
        visualize_ft: bool = False,
    ) -> None:
        self.window_name = window_name
        self._min_dt = 1.0 / max_fps
        self._last_t = 0.0
        self._created = False
        self._scr_w = _screen_width()
        self._scr_h = _screen_height()
        self._win_h = int(self._scr_h * 0.30)
        self._win_y = int(self._scr_h * 0.70)   # top of camera strip = bottom of MuJoCo viewer
        self.visualize_ft = visualize_ft
        self._ft_hist: deque = deque(maxlen=_HISTORY)
        self._ft_tare: float | None = None
        self._ft_window_name = f"{window_name} · F/T" if visualize_ft else None
        self._ft_window_created = False
        self._ft_win_w = 720
        self._ft_win_h = 320

    # ── Public API ───────────────────────────────────────────────────

    def update(self, obs: dict[str, np.ndarray], info: dict | None = None, reward: float | None = None) -> int:
        """Render one frame of the camera strip. Returns cv2.waitKey code."""
        import time
        # Keep sampling the F/T history on every call so throttled display
        # frames still show a continuous trace. Prefer the world-frame
        # force when available — its Z component is a direct proxy for
        # held-object weight (in kg × 9.81) regardless of gripper pose.
        if self.visualize_ft:
            if "force_torque_world" in obs:
                self._ft_hist.append(np.asarray(obs["force_torque_world"]).copy())
            elif "force_torque" in obs:
                self._ft_hist.append(np.asarray(obs["force_torque"])[:3].copy())
            self._ft_tare = _tare(self._ft_hist, self._ft_tare)

        now = time.monotonic()
        if now - self._last_t < self._min_dt:
            return cv2.waitKey(1)
        self._last_t = now

        rgb_imgs = {k[4:]: v for k, v in obs.items() if k.startswith("rgb_")}
        canvas = self._build_camera_strip(rgb_imgs)

        if not self._created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, canvas.shape[1], canvas.shape[0])
            cv2.moveWindow(self.window_name, 0, self._win_y)
            self._created = True
            import subprocess
            try:
                subprocess.Popen(
                    ["wmctrl", "-r", self.window_name, "-b", "add,above"],
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass  # wmctrl not installed

        cv2.imshow(self.window_name, canvas)

        # Dedicated F/T window — rendered standalone so it doesn't compete
        # with the camera strip for horizontal space.
        if self.visualize_ft and self._ft_hist:
            self._render_ft_window()

        return cv2.waitKey(1)

    def _render_ft_window(self) -> None:
        """Draw the tared vertical-load time-series in its own CV2 window.

        The plot shows ``|ΔFz_world|`` against the tare taken after reset
        (empty gripper reads ~0) as a held-mass estimate in grams, assuming
        the extra vertical load is ``m × g``.
        """
        mass_g = _held_mass_g(self._ft_hist, self._ft_tare)

        title = f"Held weight: {mass_g[-1]:6.1f} g"
        ft_canvas = _render_timeseries(
            mass_g.reshape(-1, 1), title, (180, 80, 255),
            width=self._ft_win_w, height=self._ft_win_h,
            y_range=(0.0, 500.0),
        )
        if not self._ft_window_created:
            cv2.namedWindow(self._ft_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._ft_window_name, self._ft_win_w, self._ft_win_h)
            # Top-right of the screen, tucked above the camera strip.
            x = max(0, self._scr_w - self._ft_win_w)
            y = max(0, self._win_y - self._ft_win_h - 40)
            cv2.moveWindow(self._ft_window_name, x, y)
            self._ft_window_created = True
            logger.info(
                "[ObsDashboard] Created F/T window '%s' at (%s,%s) size %sx%s",
                self._ft_window_name, x, y, self._ft_win_w, self._ft_win_h,
            )
        cv2.imshow(self._ft_window_name, ft_canvas)

    def reset(self) -> None:
        """Clear the F/T history and tare for a new episode."""
        self._ft_hist.clear()
        self._ft_tare = None

    def close(self) -> None:
        if self._created:
            cv2.destroyWindow(self.window_name)
            self._created = False
        if self._ft_window_created:
            cv2.destroyWindow(self._ft_window_name)
            self._ft_window_created = False

    # ── Layout ───────────────────────────────────────────────────────

    def _build_camera_strip(self, rgb_imgs: dict[str, np.ndarray]) -> np.ndarray:
        """Scale all cameras to one height (correct aspect ratios) and hstack them.

        The height is the tallest source's, reduced so the strip fits its
        screen area (full width, bottom 30 %).
        """
        if not rgb_imgs:
            return np.full((self._win_h, self._scr_w, 3), _BG, dtype=np.uint8)

        bgr_imgs = []
        for label, img in rgb_imgs.items():
            if img.ndim == 3 and img.shape[2] == 3:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img.copy()
            bgr_imgs.append((label, bgr))

        target_h = max(img.shape[0] for _, img in bgr_imgs)
        strip_w = sum(img.shape[1] * target_h / img.shape[0] for _, img in bgr_imgs)
        scale = min(1.0, self._win_h / target_h, self._scr_w / strip_w)
        target_h = max(1, int(target_h * scale))

        tiles = []
        for label, img in bgr_imgs:
            h, w = img.shape[:2]
            new_w = max(1, int(w * target_h / h))
            tile = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(tile, (0, 0), (new_w, 22), (0, 0, 0), -1)
            cv2.putText(tile, label, (4, 15), _FONT, 0.45, _TEXT_COLOR, 1, cv2.LINE_AA)
            tiles.append(tile)

        return np.hstack(tiles)
