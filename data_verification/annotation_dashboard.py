"""Data annotation dashboard for reviewing collected demonstrations.

Displays all camera feeds for each episode side-by-side, lets the annotator
scrub through frames, play/pause, and rate each episode from 1 to 5.
Annotations are persisted to a JSON file that supports incremental updates
and resume-from-last-unannotated.  It opens one collection folder
(``<out-dir>/M<machine_id>/``); merged datasets are not supported.

Controls (keyboard):
    SPACE       Play / Pause
    + / -       Increase / decrease playback speed (default 2x)
    RIGHT       Next frame  (hold for fast-forward)
    LEFT        Previous frame
    UP / DOWN   Seek +/- 10 % of episode
    1-5         Rate episode (1=worst, 5=best) -> next un-annotated episode
    U           Undo / clear annotation for current episode
    N           Jump to next episode
    P           Jump to previous episode
    R           Jump to first un-annotated episode (resume)
    S           Print summary to terminal
    Q / ESC     Quit

Usage::

    python -m data_verification demos/ib-p1-edge-slide-v1/M01/
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from data_verification.annotation_store import AnnotationStore

# ── Layout & style constants ────────────────────────────────────────
_TILE_H = 360
_TILE_W = 480
_PAD = 4
_BG = (30, 30, 30)
_TEXT_COLOR = (220, 220, 220)
_PENDING_COLOR = (180, 180, 0) # cyan-ish
# Rating 1-5 color ramp (BGR): red -> orange -> yellow -> lime -> green
_RATING_COLORS = {
    1: (0, 60, 220),     # red
    2: (0, 120, 255),    # orange
    3: (0, 200, 230),    # yellow
    4: (80, 200, 80),    # lime
    5: (0, 200, 120),    # green
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_S = 0.5
_FONT_S_SMALL = 0.38
_BAR_H = 36                    # height of the status / controls bar
_PROGRESS_H = 8                # height of playback progress bar

# Key codes — use full codes from cv2.waitKeyEx() to avoid
# arrow keys aliasing ASCII letters after & 0xFF
# (e.g. Left Arrow 65361 & 0xFF == 81 == ord('Q'), causing quit on Left press).
_K_SPACE = 32
_K_ESC = 27
# GTK / X11 full key codes
_K_LEFT  = 65361
_K_UP    = 65362
_K_RIGHT = 65363
_K_DOWN  = 65364


def _read_dataset_meta(dataset_root: str):
    """Return (info, episodes) from meta/."""
    with open(os.path.join(dataset_root, "meta", "info.json")) as f:
        info = json.load(f)
    episodes = []
    with open(os.path.join(dataset_root, "meta", "episodes.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return info, episodes


def _discover_cameras(dataset_root: str, info: dict) -> list[str]:
    """Return sorted list of camera label strings present in the dataset."""
    cameras = []
    for key, feat in info.get("features", {}).items():
        if key.startswith("observation.rgb_") and feat.get("dtype") == "video":
            label = key.replace("observation.", "")
            if os.path.basename(label) != label or label in (".", ".."):
                raise ValueError(f"invalid video feature key {key!r} in meta/info.json")
            cameras.append(label)
    cameras.sort()
    return cameras


class _VideoReader:
    """Lazy, frame-accurate reader for a single MP4 file."""

    def __init__(self, path: str):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._last_idx = -1

    def read(self, frame_idx: int) -> np.ndarray | None:
        """Read a specific frame (BGR). Returns None on failure."""
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        # Sequential read is faster than seeking every time
        if frame_idx != self._last_idx + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        self._last_idx = frame_idx
        return frame if ret else None

    def release(self):
        self._cap.release()


class AnnotationDashboard:
    """Interactive CV2 dashboard for annotating robot demonstration episodes."""

    def __init__(self, dataset_root: str, annotator: str = ""):
        self.dataset_root = os.path.abspath(dataset_root)
        self.annotator = annotator

        # Load dataset metadata
        self.info, self.episodes = _read_dataset_meta(self.dataset_root)
        self.task_id = self.info.get("task_id", "unknown")
        self.total_episodes = len(self.episodes)
        self.cameras = _discover_cameras(self.dataset_root, self.info)

        if self.total_episodes == 0:
            raise ValueError(f"No episodes found in {self.dataset_root}")
        if not self.cameras:
            raise ValueError(f"No camera streams found in {self.dataset_root}")
        # Ratings and video paths are keyed by position in episodes.jsonl.
        indices = [ep.get("episode_index", i) for i, ep in enumerate(self.episodes)]
        if indices != list(range(self.total_episodes)):
            raise ValueError(
                f"Episode indices in {self.dataset_root}/meta/episodes.jsonl are not "
                f"0..{self.total_episodes - 1}; merged datasets are not supported, "
                "annotate the M<machine_id> collection folders instead"
            )
        if not any(self._has_videos(i) for i in range(self.total_episodes)):
            raise ValueError(f"No video files found for any episode in {self.dataset_root}")

        # Annotation store (creates / loads annotations.json)
        self.store = AnnotationStore(
            self.dataset_root, self.task_id, self.total_episodes,
        )

        # Playback state
        self._ep_idx = 0
        self._frame_idx = 0
        self._playing = False
        self._speed = 2  # playback speed multiplier (frames per tick)
        self._readers: dict[str, _VideoReader] = {}
        self._total_frames = 0

        self._window_name = "Annotation Dashboard"
        self._created = False

    # ── Episode lifecycle ────────────────────────────────────────────

    def _has_videos(self, ep_idx: int) -> bool:
        """Return True if at least one camera video file exists for this episode."""
        chunk_idx = ep_idx // self.info.get("chunk_size", 1000)
        chunk_dir = f"chunk-{chunk_idx:03d}"
        for cam in self.cameras:
            video_path = os.path.join(
                self.dataset_root, "videos", chunk_dir, cam,
                f"episode_{ep_idx:06d}.mp4",
            )
            if os.path.exists(video_path):
                return True
        return False

    def _load_episode(self, ep_idx: int):
        """Open video readers for the requested episode."""
        self._release_readers()
        self._ep_idx = ep_idx
        self._frame_idx = 0
        self._playing = False

        chunk_idx = ep_idx // self.info.get("chunk_size", 1000)
        chunk_dir = f"chunk-{chunk_idx:03d}"

        max_frames = 0
        for cam in self.cameras:
            video_path = os.path.join(
                self.dataset_root, "videos", chunk_dir, cam,
                f"episode_{ep_idx:06d}.mp4",
            )
            if os.path.exists(video_path):
                reader = _VideoReader(video_path)
                self._readers[cam] = reader
                max_frames = max(max_frames, reader.frame_count)

        if not self._readers:
            print(f"[annotator] Warning: no video files found for episode {ep_idx}")
        self._total_frames = max_frames

    def _release_readers(self):
        for r in self._readers.values():
            r.release()
        self._readers.clear()

    # ── Drawing ──────────────────────────────────────────────────────

    def _render_frame(self) -> np.ndarray:
        """Build the full dashboard canvas for the current state."""
        tiles = []
        for cam in self.cameras:
            reader = self._readers.get(cam)
            if reader:
                frame = reader.read(self._frame_idx)
            else:
                frame = None

            if frame is None:
                tile = np.full((_TILE_H, _TILE_W, 3), 50, dtype=np.uint8)
                cv2.putText(tile, "no video", (_TILE_W // 2 - 40, _TILE_H // 2),
                            _FONT, 0.6, (100, 100, 100), 1, cv2.LINE_AA)
            else:
                tile = cv2.resize(frame, (_TILE_W, _TILE_H), interpolation=cv2.INTER_AREA)

            # Camera label overlay
            cv2.rectangle(tile, (0, 0), (_TILE_W, 22), (0, 0, 0), -1)
            cv2.putText(tile, cam, (6, 16), _FONT, _FONT_S, _TEXT_COLOR, 1, cv2.LINE_AA)
            tiles.append(tile)

        # Arrange tiles in a 2-column grid
        n_cols = min(len(tiles), 2)
        rows = []
        for i in range(0, len(tiles), n_cols):
            row_tiles = tiles[i : i + n_cols]
            while len(row_tiles) < n_cols:
                row_tiles.append(np.full((_TILE_H, _TILE_W, 3), _BG[0], dtype=np.uint8))
            sep = np.full((_TILE_H, _PAD, 3), _BG[0], dtype=np.uint8)
            row_parts = []
            for j, t in enumerate(row_tiles):
                if j > 0:
                    row_parts.append(sep)
                row_parts.append(t)
            rows.append(np.hstack(row_parts))

        # Stack rows vertically
        canvas_w = rows[0].shape[1] if rows else _TILE_W * 2 + _PAD
        full_rows = []
        for i, r in enumerate(rows):
            if i > 0:
                full_rows.append(np.full((_PAD, canvas_w, 3), _BG[0], dtype=np.uint8))
            full_rows.append(r)

        video_canvas = np.vstack(full_rows) if full_rows else np.full((100, canvas_w, 3), _BG[0], dtype=np.uint8)

        # Progress bar
        progress_bar = self._draw_progress_bar(canvas_w)

        # Status bar
        status_bar = self._draw_status_bar(canvas_w)

        # Controls help bar
        help_bar = self._draw_help_bar(canvas_w)

        canvas = np.vstack([video_canvas, progress_bar, status_bar, help_bar])
        return canvas

    def _draw_progress_bar(self, width: int) -> np.ndarray:
        """Thin playback progress indicator."""
        bar = np.full((_PROGRESS_H, width, 3), 50, dtype=np.uint8)
        if self._total_frames > 0:
            denom = max(1, self._total_frames)
            frac = min(max((self._frame_idx + 1) / denom, 0.0), 1.0)
            fill_w = int(frac * width)
            bar[:, :fill_w] = (0, 200, 120)
        return bar

    def _draw_status_bar(self, width: int) -> np.ndarray:
        """Episode info + annotation status."""
        bar = np.full((_BAR_H, width, 3), 40, dtype=np.uint8)

        rating = self.store.rating_of(self._ep_idx)
        if rating is not None:
            status_label = f"RATING:{rating}/5"
            status_color = _RATING_COLORS.get(rating, _TEXT_COLOR)
        else:
            status_label = "PENDING"
            status_color = _PENDING_COLOR

        ep_meta = self.episodes[self._ep_idx]
        score = ep_meta.get("final_score", None)
        score_str = f"  score: {score:.2f}" if score is not None else ""

        frame_str = f"{self._frame_idx + 1}/{self._total_frames}" if self._total_frames else "0/0"
        play_str = f"PLAYING x{self._speed}" if self._playing else "PAUSED"

        summary = self.store.summary()

        left_text = (
            f"Episode {self._ep_idx}/{self.total_episodes - 1}  |  "
            f"Frame {frame_str}  |  {play_str}{score_str}"
        )
        rated = summary["rated"]
        pending = summary["pending"]
        avg = summary["average"]
        avg_str = f"{avg:.1f}" if avg is not None else "-"
        right_text = (
            f"[{status_label}]  "
            f"Rated:{rated}  Pending:{pending}  Avg:{avg_str}"
        )

        cv2.putText(bar, left_text, (8, 24), _FONT, _FONT_S, _TEXT_COLOR, 1, cv2.LINE_AA)

        # Right-align the status text
        (tw, _), _ = cv2.getTextSize(right_text, _FONT, _FONT_S, 1)
        cv2.putText(bar, right_text, (width - tw - 10, 24), _FONT, _FONT_S, status_color, 1, cv2.LINE_AA)

        return bar

    def _draw_help_bar(self, width: int) -> np.ndarray:
        """Keyboard shortcut reference."""
        bar = np.full((22, width, 3), 30, dtype=np.uint8)
        help_text = (
            "SPACE:play/pause  +/-:speed  LEFT/RIGHT:frame  UP/DOWN:seek  "
            "1-5:rate  U:undo  N/P:next/prev  R:resume  S:summary  Q:quit"
        )
        cv2.putText(bar, help_text, (8, 15), _FONT, _FONT_S_SMALL, (140, 140, 140), 1, cv2.LINE_AA)
        return bar

    # ── Main loop ────────────────────────────────────────────────────

    def run(self, start_episode: int | None = None):
        """Launch the interactive annotation session.

        Parameters
        ----------
        start_episode : int or None
            Episode to start at.  ``None`` → resume from first un-annotated.
        """
        if start_episode is not None:
            ep = max(0, min(start_episode, self.total_episodes - 1))
        else:
            ep = self.store.first_unannotated()
            if ep is None:
                ep = 0
                print("[annotator] All episodes already annotated. Starting at episode 0.")
            else:
                print(f"[annotator] Resuming at first un-annotated episode: {ep}")

        # Advance past episodes that have no video files on disk
        if not self._has_videos(ep):
            original_ep = ep
            ep = next(
                i % self.total_episodes
                for i in range(ep, ep + self.total_episodes)
                if self._has_videos(i % self.total_episodes)
            )
            print(f"[annotator] Episode {original_ep} has no videos; advancing to episode {ep}")

        self._load_episode(ep)
        self._print_summary()

        camera_fps = self.info.get("save_freq", {}).get("cameras", 25)
        frame_delay = 1.0 / camera_fps

        while True:
            t0 = time.perf_counter()

            canvas = self._render_frame()

            if not self._created:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._window_name,
                                 min(canvas.shape[1], 1400),
                                 min(canvas.shape[0], 900))
                self._created = True

            cv2.imshow(self._window_name, canvas)

            # Shorter wait when playing for smoother playback
            wait_ms = 1 if self._playing else 30
            key = cv2.waitKeyEx(wait_ms)

            action = self._handle_key(key)
            if action == "quit":
                break

            # Auto-advance frame when playing
            if self._playing and self._total_frames > 0:
                self._frame_idx += self._speed
                if self._frame_idx >= self._total_frames:
                    self._frame_idx = self._total_frames - 1
                    self._playing = False  # stop at end

            # Pace playback to camera fps
            if self._playing:
                elapsed = time.perf_counter() - t0
                remaining = frame_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        self._release_readers()
        cv2.destroyAllWindows()
        self._print_summary()

    def _handle_key(self, key: int) -> str | None:
        """Process a keypress. Returns 'quit' to exit, else None."""
        if key == -1:
            return None  # no key (waitKeyEx returns -1 on timeout)

        # Quit
        if key in (ord('q'), ord('Q'), _K_ESC):
            return "quit"

        # Play / Pause
        if key == _K_SPACE:
            self._playing = not self._playing

        # Frame step
        elif key in (_K_RIGHT, ord('d')) and self._total_frames > 0:
            self._playing = False
            self._frame_idx = min(self._frame_idx + 1, self._total_frames - 1)
        elif key in (_K_LEFT, ord('a')) and self._total_frames > 0:
            self._playing = False
            self._frame_idx = max(self._frame_idx - 1, 0)

        # Seek 10%
        elif key == _K_UP and self._total_frames > 0:
            jump = max(1, self._total_frames // 10)
            self._frame_idx = min(self._frame_idx + jump, self._total_frames - 1)
        elif key == _K_DOWN and self._total_frames > 0:
            jump = max(1, self._total_frames // 10)
            self._frame_idx = max(self._frame_idx - jump, 0)

        # Annotate: Rating 1-5
        elif ord('1') <= key <= ord('5'):
            rating = key - ord('0')
            if self.store.annotate(self._ep_idx, rating, self.annotator):
                print(f"  Episode {self._ep_idx} -> {rating}/5")
                self._advance_to_next()
            else:
                self._report_conflict()

        # Playback speed
        elif key in (ord('+'), ord('='), ord(']')):
            self._speed = min(self._speed + 1, 10)
            print(f"  Speed: x{self._speed}")
        elif key in (ord('-'), ord('_'), ord('[')):
            self._speed = max(self._speed - 1, 1)
            print(f"  Speed: x{self._speed}")

        # Undo annotation
        elif key in (ord('u'), ord('U')):
            if self.store.clear(self._ep_idx):
                print(f"  Episode {self._ep_idx} -> CLEARED (pending)")
            else:
                self._report_conflict()

        # Next / Previous episode
        elif key in (ord('n'), ord('N')):
            if self._ep_idx < self.total_episodes - 1:
                self._load_episode(self._ep_idx + 1)
        elif key in (ord('p'), ord('P')):
            if self._ep_idx > 0:
                self._load_episode(self._ep_idx - 1)

        # Resume (jump to first un-annotated)
        elif key in (ord('r'), ord('R')):
            nxt = self.store.first_unannotated()
            if nxt is not None:
                print(f"  Jumping to first un-annotated: episode {nxt}")
                self._load_episode(nxt)
            else:
                print("  All episodes annotated!")

        # Summary
        elif key in (ord('s'), ord('S')):
            self._print_summary()

        return None

    def _report_conflict(self):
        """Say that another session changed the current episode's annotation; nothing was saved."""
        ann = self.store.get(self._ep_idx)
        rating = self.store.rating_of(self._ep_idx)
        now = f"rated {rating}/5 by {ann.get('annotator') or 'unknown'}" if rating is not None else "pending"
        print(f"  Episode {self._ep_idx} was changed by another session (now {now}); "
              "not saved. Press the key again to overwrite.")

    def _advance_to_next(self):
        """After annotating, jump to the next un-annotated episode after this one (or next in sequence)."""
        nxt = self.store.next_unannotated(self._ep_idx)
        if nxt is not None:
            self._load_episode(nxt)
        elif self._ep_idx < self.total_episodes - 1:
            self._load_episode(self._ep_idx + 1)
        else:
            print("  All episodes annotated!")

    def _print_summary(self):
        summary = self.store.summary()
        total = self.total_episodes
        avg = summary["average"]
        avg_str = f"{avg:.2f}" if avg is not None else "-"
        print(f"\n[annotation summary] {self.task_id}  ({self.dataset_root})")
        print(f"  Rated:   {summary['rated']:>3} / {total}  (avg: {avg_str})")
        for r in range(1, 6):
            print(f"    {r}/5:   {summary['counts'].get(r, 0):>3}")
        print(f"  Pending: {summary['pending']:>3} / {total}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive data annotation dashboard for robot demonstrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Path to a collection folder (e.g. demos/ib-p1-edge-slide-v1/M01/)",
    )
    parser.add_argument(
        "--episode", type=int, default=None,
        help="Episode index to start at (default: first un-annotated)",
    )
    parser.add_argument(
        "--annotator", type=str, default="",
        help="Name of the annotator (saved in the JSON for tracking)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a directory")
        sys.exit(1)
    if not os.path.exists(os.path.join(args.path, "meta", "info.json")):
        print(f"Error: {args.path} is not a LeRobot dataset (missing meta/info.json)")
        sys.exit(1)

    try:
        dashboard = AnnotationDashboard(args.path, annotator=args.annotator)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    dashboard.run(start_episode=args.episode)


if __name__ == "__main__":
    main()
