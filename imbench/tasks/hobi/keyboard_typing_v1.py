"""Keyboard Typing — homogeneous bimanual task.

Both arms use identical parallel-jaw (finger) grippers to type a word on a
large QWERTY virtual keyboard.  One arm presses a scripted subset of the
letters (``arm0_indices`` in the config) while the other arm is teleoperated
to press the remaining letters (``arm1_indices``).

Index arrays may interleave arbitrarily (sequential split, alternating, or
any custom partition). The word is typed left to right: only the next
letter of the word can be typed, and only by the arm that owns it, so the
arms take turns whenever ownership changes. The scripted arm 0 waits at
its home pose while it is arm 1's turn.

Scoring:
- A letter is typed once a finger pad of the responsible arm has pressed
  its key from above for ``hold_steps`` consecutive steps.
- A typo — arm 1 pressing any other key for ``hold_steps`` steps, including
  its own letters out of turn — ends the episode as a failure.
- Each letter contributes ``1 / len(word)``, staged hover → descend → hold,
  credited only once every earlier letter is typed; success when every
  letter is typed.

Task ID: ``ib-m-keyboard-typing-v1``
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from robosuite.models.objects.objects import MujocoXMLObject
from robosuite.utils import transform_utils as T

import imbench.envs  # noqa: F401 — register envs
from imbench.objects import ExtrasObject
from imbench.tasks.base_task import BaseTask, load_task_config
from imbench.tasks.staged_reward import Stage, StagedReward


# ── QWERTY key centre positions relative to keyboard board origin ──────────
# Derived from keyboard.xml geometry (board origin at bottom-centre of board).
# Key top surface z = 0.024 m.
_KEY_POSITIONS: dict[str, tuple[float, float]] = {
    # Row 0
    "Q": (-0.1800, 0.0450), "W": (-0.1400, 0.0450), "E": (-0.1000, 0.0450),
    "R": (-0.0600, 0.0450), "T": (-0.0200, 0.0450), "Y": (0.0200, 0.0450),
    "U": (0.0600, 0.0450),  "I": (0.1000, 0.0450),  "O": (0.1400, 0.0450),
    "P": (0.1800, 0.0450),
    # Row 1
    "A": (-0.1400, 0.0000), "S": (-0.1000, 0.0000), "D": (-0.0600, 0.0000),
    "F": (-0.0200, 0.0000), "G": (0.0200, 0.0000),  "H": (0.0600, 0.0000),
    "J": (0.1000, 0.0000),  "K": (0.1400, 0.0000),  "L": (0.1800, 0.0000),
    # Row 2
    "Z": (-0.0800, -0.0450), "X": (-0.0400, -0.0450), "C": (0.0000, -0.0450),
    "V": (0.0400, -0.0450),  "B": (0.0800, -0.0450),  "N": (0.1200, -0.0450),
    "M": (0.1600, -0.0450),
}
_KEY_TOP_Z = 0.024  # key top surface z in board-local frame

# OSC_POSE action → physical motion: action * output_max = metres/step.
# Default robosuite output_max for position dims = 0.05 m/step at action=1.
_OSC_POS_SCALE = 0.05  # m / normalised-action-unit


# ─────────────────────────────────────────────────────────────────────────────
# Word-progress display: a strip of small textured planes on the table beside
# the keyboard that shows the task word, with each letter rendered in either
# bold dark colour (already typed) or faint light colour (not yet typed). We
# pre-bake two glyph PNGs per character in the word (one bold, one faint),
# stack them at the same world position, and toggle the alpha of each pair
# per letter to switch between the two states based on ``_typed_mask``.
# This keeps the texture set independent of which arm typed which letter
# and uses only 2 N textures total.
# ─────────────────────────────────────────────────────────────────────────────

_WORD_DISPLAY_HALF_W = 0.34   # strip half-extent along strip-local x (68 cm)
_WORD_DISPLAY_HALF_H = 0.06   # strip half-extent along strip-local y (12 cm)
_WORD_DISPLAY_GLYPH_PX = 256  # glyph texture height in pixels
_WORD_DISPLAY_TYPED_RGB = (15, 15, 25)        # bold colour for already-typed
_WORD_DISPLAY_UNTYPED_RGB = (200, 200, 200)   # faint colour for not-yet-typed
_WORD_DISPLAY_BG_RGB = (245, 245, 240)        # background card colour
_WORD_DISPLAY_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _glyph_font(size: int):
    """DejaVu Sans Bold at *size* px, else Pillow's built-in font.

    Pillow < 10.1 has no scalable built-in font, so glyphs fall back to its
    small bitmap font there.
    """
    from PIL import ImageFont

    for font_file in (_WORD_DISPLAY_FONT, "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_file, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_glyph_png(
    ch: str, fill_rgb: tuple[int, int, int], out_path: str, *,
    size_px: int = _WORD_DISPLAY_GLYPH_PX,
) -> None:
    """Render a single bold character centred on a card-coloured square."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size_px, size_px), _WORD_DISPLAY_BG_RGB)
    draw = ImageDraw.Draw(img)
    font = _glyph_font(int(0.78 * size_px))
    bbox = draw.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (size_px - w) / 2 - bbox[0]
    y = (size_px - h) / 2 - bbox[1]
    draw.text((x, y), ch, font=font, fill=fill_rgb)
    img.save(out_path)


def _build_word_display_xml(word: str, asset_dir: Path) -> str:
    """Generate textures + XML for the per-letter word-progress display.

    Returns the absolute path to a MuJoCo XML file containing:
      * a card backing geom
      * for each letter position i in 0..N-1, two stacked textured plane
        geoms named ``letter_{i}_bold`` and ``letter_{i}_faint`` at the
        same x position. The task toggles alpha so exactly one of each
        pair is visible based on ``_typed_mask[i]``.
    """
    n = len(word)
    asset_dir.mkdir(parents=True, exist_ok=True)

    # Per-(position, state) PNG so identical letters in the word get
    # independent bold/faint alpha control without any name collisions.
    for i, ch in enumerate(word):
        _render_glyph_png(ch, _WORD_DISPLAY_TYPED_RGB,
                          str(asset_dir / f"glyph_{i}_bold.png"))
        _render_glyph_png(ch, _WORD_DISPLAY_UNTYPED_RGB,
                          str(asset_dir / f"glyph_{i}_faint.png"))

    sx = _WORD_DISPLAY_HALF_W
    sy = _WORD_DISPLAY_HALF_H

    # Layout: spread letters evenly across the strip with small gaps.
    # Each glyph plane is square with half-side = sy (height of strip).
    glyph_half = sy * 0.95
    # Total width occupied by N glyphs, evenly spaced edge-to-edge:
    spacing = (2 * sx) / n  # centre-to-centre distance
    if spacing < 2 * glyph_half:
        glyph_half = spacing * 0.49  # shrink to fit when word is long

    tex_lines, mat_lines, geom_lines = [], [], []
    for i in range(n):
        for state in ("bold", "faint"):
            tex_lines.append(
                f'    <texture name="tex_{i}_{state}" type="2d" '
                f'file="glyph_{i}_{state}.png"/>'
            )
            mat_lines.append(
                f'    <material name="mat_{i}_{state}" texture="tex_{i}_{state}" '
                f'specular="0" shininess="0"/>'
            )
        # Centre x of letter i (strip spans -sx..+sx in local frame).
        cx = -sx + spacing * (i + 0.5)
        # Stack bold/faint planes at the same x with a sub-mm z offset
        # so MuJoCo doesn't fight over depth during alpha-1 rendering.
        geom_lines.append(
            f'        <geom name="letter_{i}_faint" type="plane" '
            f'size="{glyph_half} {glyph_half} 0.001" '
            f'pos="{cx} 0 0.00110" '
            f'material="mat_{i}_faint" rgba="1 1 1 1" '
            f'group="1" contype="0" conaffinity="0"/>'
        )
        geom_lines.append(
            f'        <geom name="letter_{i}_bold" type="plane" '
            f'size="{glyph_half} {glyph_half} 0.001" '
            f'pos="{cx} 0 0.00115" '
            f'material="mat_{i}_bold" rgba="1 1 1 0" '
            f'group="1" contype="0" conaffinity="0"/>'
        )

    xml = f"""<mujoco model="word_display">
  <compiler angle="radian"/>
  <asset>
{chr(10).join(tex_lines)}
{chr(10).join(mat_lines)}
  </asset>
  <worldbody>
    <body>
      <body name="object" pos="0 0 0">
        <!-- Card backing so the strip reads even off-camera-axis. -->
        <geom name="word_card" type="box" size="{sx + 0.005} {sy + 0.005} 0.0005"
              pos="0 0 0.0005" rgba="0.96 0.96 0.94 1"
              group="1" contype="0" conaffinity="0"/>
{chr(10).join(geom_lines)}
        <inertial pos="0 0 0.001" mass="0.1" diaginertia="0.001 0.001 0.001"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0.002" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{sx} {sy} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = asset_dir / "word_display.xml"
    xml_path.write_text(xml)
    return str(xml_path)


class _WordDisplayObject(MujocoXMLObject):
    """Robosuite wrapper around the dynamically generated word-display XML."""

    def __init__(self, name: str, xml_path: str):
        super().__init__(
            fname=xml_path,
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class KeyboardTypingV1(BaseTask):
    """Type a word on a virtual QWERTY keyboard with two arms.

    The YAML ``word`` is fixed by default. Collectors can override it with
    ``word: null`` and a ``words`` list to sample a word each reset.
    Arm 0 presses the letters at
    ``arm0_indices`` in a scripted fashion; arm 1 is teleoperated to
    press the letters at ``arm1_indices``.

    Letters are typed in word order, each by the arm that owns it; a key
    held by arm 1 out of turn is a typo and fails the episode.

    Parameters
    ----------
    task_id : str, optional
        Defaults to ``"ib-m-keyboard-typing-v1"``.
    **kwargs
        Forwarded to ``BaseTask``.
    """

    TASK_ID = "ib-m-keyboard-typing-v1"
    DESCRIPTION = (
        "Type a word on a large QWERTY virtual keyboard. "
        "One arm presses a scripted subset of letters while the other "
        "arm is teleoperated to press the remaining letters. Letters "
        "must be typed in word order; a wrong key fails the episode."
    )

    def __init__(self, task_id: str = None, **kwargs):
        if task_id:
            self.TASK_ID = task_id
        self.CONFIG_FILE = f"hobi/{self.TASK_ID}"

        _cfg = load_task_config(self.TASK_ID, config_file=self.CONFIG_FILE,
                                overrides=kwargs.get("config_overrides"))

        self.DESCRIPTION = _cfg.get("description", self.DESCRIPTION)
        if "max_episode_steps" in _cfg:
            self.MAX_EPISODE_STEPS = _cfg["max_episode_steps"]

        self._rng = np.random.RandomState(kwargs.get("seed", 0))

        # ── Task word & arm assignment ───────────────────────────────
        fixed_word = _cfg.get("word")
        words = [fixed_word] if fixed_word is not None else _cfg.get("words", ["IMBENCH"])
        if not isinstance(words, (list, tuple)) or not words:
            raise ValueError("words must be a non-empty list of QWERTY words")
        if any(not isinstance(w, str) or len(w) < 2
               or any(ch not in _KEY_POSITIONS for ch in w.upper()) for w in words):
            raise ValueError("each word must contain at least two letters from A-Z")
        self._words = tuple(w.upper() for w in words)
        # Build a placeholder display without consuming the first episode's RNG.
        self._word = self._words[0]
        # arm0_indices: list of word-positions (0-based) that arm 0 types.
        # arm1_indices: list of word-positions that arm 1 (teleop) types.
        # Together they must cover 0..len(word)-1 exactly once.
        # If both are empty in the config, the partition is randomized
        # every reset (see ``_resolve_indices``); otherwise the configured
        # split is used verbatim and validated.
        self._arm0_indices_cfg: list[int] = list(_cfg.get("arm0_indices") or [])
        self._arm1_indices_cfg: list[int] = list(_cfg.get("arm1_indices") or [])
        # Minimum number of letters arm 0 must get when the partition is
        # drawn randomly each reset. Clamped against the word length in
        # ``_resolve_indices``; ignored when arm0/arm1 are set in YAML.
        self._arm0_min_letters: int = int(_cfg.get("arm0_min_letters", 3))
        self._arm0_indices: list[int] = []
        self._arm1_indices: list[int] = []

        # ── Key press detection parameters ───────────────────────────
        press_cfg = _cfg.get("press_detection", {})
        self._key_xy_tol: float = press_cfg.get("xy_tolerance", 0.025)
        # EEF z below this (in world frame) counts as a key press
        self._press_z_threshold: float = press_cfg.get("z_threshold_above_key", 0.012)
        # Minimum number of steps EEF must stay in contact to register
        self._press_hold_steps: int = press_cfg.get("hold_steps", 3)

        # ── Scripted arm motion parameters ───────────────────────────
        script_cfg = _cfg.get("scripted_arm", {})
        self._hover_height: float = script_cfg.get("hover_height", 0.08)
        self._approach_speed: float = script_cfg.get("approach_speed", 0.003)
        self._lift_speed: float = script_cfg.get("lift_speed", 0.004)

        # ── Keyboard placement ───────────────────────────────────────
        kbd_cfg = _cfg.get("keyboard", {})
        self._keyboard_pos: list[float] = kbd_cfg.get("pos", [0.0, 0.0])

        # ── Scene config ──────────────────────────────────────────────
        scene_cfg = _cfg.get("scene", {})
        table_cfg = scene_cfg.get("table", {})
        self._table_height: float = table_cfg.get("height", 0.9)
        _full_size = table_cfg.get("full_size")
        self._table_full_size = tuple(_full_size) if _full_size else (1.2, 0.80, 0.05)
        _offset = table_cfg.get("offset")
        self._table_offset = tuple(_offset) if _offset else (0, 0, self._table_height)
        _robot_offset = scene_cfg.get("robot_offset")
        self._robot_offset = [list(r) for r in _robot_offset] if _robot_offset else None

        # Domain randomization
        rand_cfg = _cfg.get("randomization", {})
        light_cfg = rand_cfg.get("lighting", {})
        self._rand_light_position: bool = light_cfg.get("position", False)
        self._rand_light_color: bool = light_cfg.get("color", False)
        kbd_rand_cfg = rand_cfg.get("keyboard_position", {})
        self._rand_kbd_pos: bool = kbd_rand_cfg.get("enabled", False)
        self._kbd_pos_x_std: float = kbd_rand_cfg.get("x_std", 0.02)
        self._kbd_pos_y_std: float = kbd_rand_cfg.get("y_std", 0.02)

        # Steps to keep the episode alive after the final letter is typed,
        # so the word-display shows all glyphs bold for a moment before
        # termination (and any trailing observations / video frames capture
        # the completed state).
        self._post_complete_hold_steps: int = int(
            _cfg.get("post_complete_hold_steps", 30)
        )

        # ── Runtime state (populated at reset) ───────────────────────
        self._typed_count: int = 0          # how many letters correctly typed
        self._typed_mask: list[bool] = []   # per word-position: has it been typed?
        self._typo_key: str | None = None   # key arm 1 pressed out of turn (fails the episode)
        self._wrong_hold: dict[str, int] = {}  # per-key hold counter for arm 1's out-of-turn presses
        self._arm1_release_key: str | None = None  # key arm 1 just typed, still held down
        self._post_complete_counter: int = 0  # steps since word was completed
        # Two separate hold counters to avoid FSM / detection interference:
        self._scripted_hold_counter: int = 0  # used by the scripted FSM only
        self._press_hold_per_letter: dict[int, int] = {}  # per word-pos hold counter
        self._letters_to_release: set[str] = set()  # typed letters whose key must be released before a repeat counts
        self._keyboard_world_pos: np.ndarray | None = None  # xyz in world frame
        self._scripted_phase: str = "idle"  # idle | hover | press | lift | return | done
        self._scripted_target: np.ndarray | None = None
        self._scripted_word_idx: int = 0  # word position arm 0 is pressing (or last pressed)
        self._robot0_base_R: np.ndarray = np.eye(3)  # rotation matrix base→world
        self._arm0_home_pos: np.ndarray | None = None  # EEF pos at reset (world frame)

        # Word-display glyph textures + XML, one subdir per word, generated on
        # first use and kept until close(). MuJoCo caches decoded textures by
        # file path for the whole process, so a word's files must keep their
        # path across resets.
        self._word_display_assets_dir = Path(tempfile.mkdtemp(prefix="kbword_"))
        self._word_display_xml_path = self._word_display_xml(self._word)
        # Per-letter (bold, faint) plane geom IDs, populated at reset.
        self._letter_geom_ids: list[tuple[int, int]] = []
        # Key geom ID → letter and finger-pad geom ID → arm, populated at reset.
        self._key_geom_letters: dict[int, str] = {}
        self._pad_geom_arm: dict[int, int] = {}
        # Last-applied per-letter typed mask (for skipping no-op updates).
        self._last_display_mask: list[bool] = []

        # Per-letter continuous staging — see `_build_staged_reward()`.
        self._staged_reward = self._build_staged_reward()

        super().__init__(**kwargs)

    # ── Staged scoring ────────────────────────────────────────────────

    def _build_staged_reward(self) -> StagedReward | None:
        """Flatten typed_count/len(word) into one continuous score.

        Each letter gets an equal 1/N budget, split into three substages —
        hover_over_key (0.25), descend_contact (0.35), hold_to_completion
        (0.40) — made continuous via the `_press_hold_per_letter`
        counter/`_press_hold_steps` ratio rather than an all-or-nothing
        per-letter gate. Once a letter is marked typed, all
        three of its substages read 1.0 permanently (guards against the
        hold counter being reset to 0 by `_tick_letter_hold` in the very
        step the letter completes, which runs before scoring). Letters
        after the next one to type read 0, so credit follows the word
        order the same way typing does.
        """
        n = len(self._word)
        if n == 0:
            return None
        per_letter = 1.0 / n
        stages = []
        for i in range(n):
            stages.append(Stage(f"letter{i}_hover", per_letter * 0.25,
                                 lambda t, o, i=i: t._letter_hover_progress(i, o)))
            stages.append(Stage(f"letter{i}_descend", per_letter * 0.35,
                                 lambda t, o, i=i: t._letter_descend_progress(i, o)))
            stages.append(Stage(f"letter{i}_hold", per_letter * 0.40,
                                 lambda t, o, i=i: t._letter_hold_progress(i, o)))
        return StagedReward(
            stages=stages,
            success_fn=lambda t, o: t._typed_count >= len(t._word),
        )

    def _arm_for_letter(self, word_idx: int) -> int:
        return 0 if word_idx in self._arm0_indices else 1

    def _next_word_idx(self) -> int | None:
        """Word position to type next, or None once the word is complete.

        Letters are typed strictly left to right, so the typed letters
        are always a prefix of the word.
        """
        return self._typed_count if self._typed_count < len(self._word) else None

    def _letter_hover_progress(self, word_idx: int, raw_obs: dict) -> float:
        if self._typed_mask[word_idx]:
            return 1.0
        if word_idx != self._next_word_idx():
            return 0.0
        arm_idx = self._arm_for_letter(word_idx)
        eef_pos = np.array(raw_obs.get(f"robot{arm_idx}_eef_pos", [0.0, 0.0, 0.0]))
        key_xyz = self._key_world_xyz(self._word[word_idx])
        xy_dist = float(np.linalg.norm(eef_pos[:2] - key_xyz[:2]))
        return 1.0 - np.tanh(10.0 * xy_dist)

    def _letter_descend_progress(self, word_idx: int, raw_obs: dict) -> float:
        if self._typed_mask[word_idx]:
            return 1.0
        if word_idx != self._next_word_idx():
            return 0.0
        letter = self._word[word_idx]
        arm_idx = self._arm_for_letter(word_idx)
        if self._gripper_contacts_key(arm_idx, letter):
            return 1.0
        eef_pos = np.array(raw_obs.get(f"robot{arm_idx}_eef_pos", [0.0, 0.0, 0.0]))
        key_xyz = self._key_world_xyz(letter)
        dist = float(np.linalg.norm(eef_pos - key_xyz))
        return 1.0 - np.tanh(10.0 * dist)

    def _letter_hold_progress(self, word_idx: int, raw_obs: dict) -> float:
        if self._typed_mask[word_idx]:
            return 1.0
        c = self._press_hold_per_letter.get(word_idx, 0)
        return float(np.clip(c / self._press_hold_steps, 0.0, 1.0))

    # ── Index resolution ─────────────────────────────────────────────

    @staticmethod
    def _resolve_indices(
        word: str,
        arm0: list[int],
        arm1: list[int],
        rng: np.random.RandomState | None = None,
        arm0_min_letters: int = 3,
    ) -> tuple[list[int], list[int]]:
        """Validate and auto-fill arm index lists.

        If both are empty → randomly partition all positions, with arm 0
        guaranteed at least ``arm0_min_letters`` letters and arm 1 at
        least one (clamped when the word is too short). If one is empty
        → the other gets the complement.
        """
        n = len(word)
        all_idx = set(range(n))

        if not arm0 and not arm1:
            r = rng if rng is not None else np.random
            # Choose how many of the n positions arm 0 gets, then sample
            # which ones uniformly without replacement. Both arms need at
            # least one letter, and arm 0 needs at least arm0_min_letters
            # (clamped so the constraint is feasible for short words).
            min_arm0 = max(1, min(arm0_min_letters, n - 1))
            max_arm0 = n - 1  # leave at least one letter for arm 1
            k = int(r.randint(min_arm0, max_arm0 + 1))
            arm0 = sorted(int(i) for i in r.choice(n, size=k, replace=False))
            arm1 = sorted(all_idx - set(arm0))
        elif not arm0:
            arm0 = sorted(all_idx - set(arm1))
        elif not arm1:
            arm1 = sorted(all_idx - set(arm0))

        # Ensure disjoint and covering
        arm0_set, arm1_set = set(arm0), set(arm1)
        assert arm0_set | arm1_set == all_idx, \
            f"arm0_indices ∪ arm1_indices must cover all {n} positions of the word"
        assert arm0_set & arm1_set == set(), \
            "arm0_indices and arm1_indices must be disjoint"
        return arm0, arm1

    # ── Environment construction ─────────────────────────────────────

    def _build_env(self):
        use_cameras = self.obs_mode in ("rgb", "combo") and self.has_cameras

        self._keyboard = ExtrasObject(
            name="keyboard",
            extras_name="keyboard",
            joints=None,
        )

        self._word_display = _WordDisplayObject(
            name="word_display",
            xml_path=self._word_display_xml_path,
        )

        common_kwargs = dict(
            robots=["Panda", "Panda"],
            gripper_types=["PandaGripper", "PandaGripper"],
            control_freq=self.control_freq,
            controller_configs=self.controller_configs,
            has_renderer=(self.render_mode == "human"),
            has_offscreen_renderer=use_cameras,
            render_camera=self.viewer_camera,
            renderer_config={"cam_config": self._viewer_cam_config}
            if self.render_mode == "human" else None,
            use_camera_obs=False,  # BaseTask renders camera frames itself
            camera_names=self._camera_names() if use_cameras else None,
            camera_heights=self._camera_heights() if use_cameras else 84,
            camera_widths=self._camera_widths() if use_cameras else 84,
            horizon=self.MAX_EPISODE_STEPS,
        )

        env = self._make_robosuite_env(
            "HoBiEnv",
            dynamic_objects=[],
            static_objects=[self._keyboard, self._word_display],
            table_full_size=self._table_full_size,
            table_offset=self._table_offset,
            robot_offset=self._robot_offset,
            **common_kwargs,
        )
        return env

    # ── Reset ────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # Seed before sampling so the word, keyboard jitter and arm partition
        # replay together. An unseeded reset continues the current stream.
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        word = str(self._rng.choice(self._words)) if len(self._words) > 1 else self._words[0]
        if word != self._word:
            self._word = word
            # Glyphs are compiled into the model: changing the target requires
            # rebuilding the display and its per-letter scoring stages together.
            self._word_display_xml_path = self._word_display_xml(self._word)
            self._staged_reward = self._build_staged_reward()
            self._rebuild_env()
        super().reset(seed=seed, options=options)

        # Re-resolve arm0/arm1 partition every reset. When both index lists
        # are empty in the YAML, this draws a fresh random partition each
        # episode using ``self._rng``; otherwise the configured split is
        # used and only validated.
        self._arm0_indices, self._arm1_indices = self._resolve_indices(
            self._word,
            list(self._arm0_indices_cfg),
            list(self._arm1_indices_cfg),
            rng=self._rng,
            arm0_min_letters=self._arm0_min_letters,
        )

        sim = self._env.sim
        table_z = float(self._env.table_offset[2])

        # Place keyboard
        kbd_x = self._keyboard_pos[0]
        kbd_y = self._keyboard_pos[1]
        if self._rand_kbd_pos:
            kbd_x += self._rng.normal(0, self._kbd_pos_x_std)
            kbd_y += self._rng.normal(0, self._kbd_pos_y_std)

        kbd_bid = sim.model.body_name2id(self._keyboard.root_body)
        # Raise keyboard 10 mm above table so key tops (at +0.024 local) sit at
        # z ≈ 0.934 — within gripper fingertip reach — enabling physical contact.
        kbd_z = table_z + 0.010
        sim.model.body_pos[kbd_bid] = [kbd_x, kbd_y, kbd_z]

        self._keyboard_world_pos = np.array([kbd_x, kbd_y, kbd_z])

        # Place word-progress display flat on the table, perpendicular to
        # the keyboard's long axis: the strip runs along world Y instead
        # of world X (rotated 90° around Z), and sits beside the keyboard
        # on the +X end of the table so it doesn't crowd the typing area.
        try:
            wd_bid = sim.model.body_name2id(self._word_display.root_body)
            # Offset the strip along +X (away from the keyboard's right
            # edge, which is at kbd_x + ~0.21). Add ~14 cm clearance plus
            # the strip's half-extent so it sits comfortably to the side.
            keyboard_half_w = 0.21
            gap = 0.05
            # After 90°-around-Z rotation, the strip's local Y (half-height)
            # extends along world X — that's what determines the X gap.
            strip_offset_x = keyboard_half_w + gap + _WORD_DISPLAY_HALF_H
            sim.model.body_pos[wd_bid] = [
                kbd_x + strip_offset_x,
                kbd_y,
                table_z + 0.001,
            ]
            # MuJoCo quat is [w, x, y, z]; rotation by +90° around +Z so
            # local +X (strip's long axis) maps to world +Y.
            sim.model.body_quat[wd_bid] = [
                np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4),
            ]
        except Exception:
            pass

        # Cache the (bold, faint) geom IDs for each letter so step() can
        # toggle alpha without name lookups every frame. Names are prefixed
        # by the robosuite object name.
        self._letter_geom_ids = []
        n = len(self._word)
        for i in range(n):
            ids = []
            for state in ("bold", "faint"):
                gname = f"word_display_letter_{i}_{state}"
                try:
                    ids.append(sim.model.geom_name2id(gname))
                except Exception:
                    ids.append(-1)
            self._letter_geom_ids.append((ids[0], ids[1]))
        # Reset bookkeeping that drives the alpha update.
        self._last_display_mask = []

        self._setup_contact_layers()

        # Cache robot 0 base rotation for world→base frame transform in scripted arm
        try:
            r0_bid = sim.model.body_name2id("robot0_base")
            quat_wxyz = sim.model.body_quat[r0_bid].copy()  # MuJoCo: [w, x, y, z]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            self._robot0_base_R = T.quat2mat(quat_xyzw)  # 3×3, columns = base axes in world
        except Exception:
            self._robot0_base_R = np.eye(3)

        # Lighting
        if self._rand_light_position or self._rand_light_color:
            for i in range(sim.model.nlight):
                if self._rand_light_position:
                    sim.model.light_pos[i] = [
                        self._rng.uniform(0.0, 2.0),
                        self._rng.uniform(-1.0, 2.0),
                        self._rng.uniform(1.0, 2.5),
                    ]
                if self._rand_light_color:
                    base = self._rng.uniform(0.75, 1.0)
                    sim.model.light_diffuse[i] = np.clip(
                        base + self._rng.uniform(-0.05, 0.05, size=3), 0.6, 1.0
                    )

        # Reset task state
        self._typed_count = 0
        self._typed_mask = [False] * len(self._word)
        self._typo_key = None
        self._wrong_hold = {}
        self._arm1_release_key = None
        self._press_hold_per_letter = {}
        self._letters_to_release = set()
        if self._staged_reward is not None:
            self._staged_reward.reset()
        self._scripted_hold_counter = 0
        self._scripted_phase = "idle"
        self._scripted_target = None
        self._scripted_word_idx = 0
        self._post_complete_counter = 0

        sim.forward()

        # Capture arm 0's starting EEF position (world frame) so the scripted
        # FSM can drive it back home after the last scripted letter is typed.
        raw_obs_init = self._get_sim_observations(force_update=True)
        eef0 = raw_obs_init.get("robot0_eef_pos")
        if eef0 is not None:
            self._arm0_home_pos = np.array(eef0, dtype=np.float64)

        # Show the all-faded ("0 letters typed") word progress state.
        self._update_word_display()

        obs = self._rebuild_obs()
        info = {"task_score": 0.0, "is_success": False, "step": 0,
                "word": self._word, "typed": ""}
        return obs, info

    # ── Word-display tempdir lifecycle ──────────────────────────────────

    def _word_display_xml(self, word: str) -> str:
        """Path of *word*'s word-display XML, generated on first use."""
        word_dir = self._word_display_assets_dir / word.lower()
        xml_path = word_dir / "word_display.xml"
        if not xml_path.exists():  # written after the glyphs
            _build_word_display_xml(word, word_dir)
        return str(xml_path)

    def _cleanup_word_display_assets_dir(self):
        d = getattr(self, "_word_display_assets_dir", None)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._cleanup_word_display_assets_dir()

    def __del__(self):
        # Best-effort backstop -- callers should prefer close(). Swallow
        # everything: __del__ running during interpreter teardown can't
        # rely on module globals (e.g. shutil) still being bound.
        try:
            self._cleanup_word_display_assets_dir()
        except Exception:
            pass

    # ── Observations ─────────────────────────────────────────────────

    def _get_obs(self, raw_obs):
        obs = self._build_obs_from_config(raw_obs)

        if self.object_state_enabled:
            if self._keyboard_world_pos is not None:
                obs["keyboard_pos"] = self._keyboard_world_pos.copy()
            obs["typed_count"] = np.array([self._typed_count], dtype=np.float32)
            obs["word_length"] = np.array([len(self._word)], dtype=np.float32)
        return obs

    # ── Word-progress display ─────────────────────────────────────────

    def _update_word_display(self) -> None:
        """Set each letter plane's alpha to match ``_typed_mask``.

        Idempotent and cheap: skips work when no letter's typed-state
        has changed since the last call.
        """
        if not self._letter_geom_ids:
            return
        if list(self._typed_mask) == self._last_display_mask:
            return
        sim = self._env.sim
        for i, (bold_gid, faint_gid) in enumerate(self._letter_geom_ids):
            typed = bool(self._typed_mask[i])
            if bold_gid >= 0:
                sim.model.geom_rgba[bold_gid, 3] = 1.0 if typed else 0.0
            if faint_gid >= 0:
                sim.model.geom_rgba[faint_gid, 3] = 0.0 if typed else 1.0
        self._last_display_mask = list(self._typed_mask)

    # ── Collision-layer setup ─────────────────────────────────────────

    def _setup_contact_layers(self) -> None:
        """Route collisions so only finger *pad* geoms can touch keyboard keys.
        """
        sim = self._env.sim
        self._key_geom_letters = {}
        self._pad_geom_arm = {}
        for letter in "QWERTYUIOPASDFGHJKLZXCVBNM":
            gname = f"keyboard_key_{letter.lower()}"
            try:
                gid = sim.model.geom_name2id(gname)
                sim.model.geom_contype[gid] = 2
                sim.model.geom_conaffinity[gid] = 2
                self._key_geom_letters[gid] = letter
            except Exception:
                pass
        for arm_idx in range(2):
            for finger in (1, 2):
                gname = f"gripper{arm_idx}_right_finger{finger}_pad_collision"
                try:
                    gid = sim.model.geom_name2id(gname)
                    sim.model.geom_contype[gid] = 3   # layers 1 + 2
                    sim.model.geom_conaffinity[gid] = 3
                    self._pad_geom_arm[gid] = arm_idx
                except Exception:
                    pass

    # ── Key world position helper ─────────────────────────────────────

    def _keys_pressed_by(self, arm_idx: int) -> set[str]:
        """Letters whose key a finger PAD of ``arm_idx`` is pressing from
        above (downward force on the key top surface).

        Two conditions beyond a plain geom-pair match:
        1. Only ``finger{N}_pad_collision`` counts — not the finger mesh.
        2. Contact normal must be mostly vertical: ``|frame[2]| > 0.5``.
           In MuJoCo the contact frame is stored row-major with the **first
           row** (indices [0:3]) being the normal from geom1 → geom2.
           A pad above a key produces a near-vertical normal.
        """
        sim = self._env.sim
        pressed: set[str] = set()
        for ci in range(sim.data.ncon):
            c = sim.data.contact[ci]
            # Must be a key ↔ pad pair
            if self._pad_geom_arm.get(c.geom1) == arm_idx:
                key_gid = c.geom2
            elif self._pad_geom_arm.get(c.geom2) == arm_idx:
                key_gid = c.geom1
            else:
                continue
            letter = self._key_geom_letters.get(key_gid)
            if letter is None:
                continue
            key_top_z = (
                sim.data.geom_xpos[key_gid][2]
                + sim.model.geom_size[key_gid][2]
            )
            # Contact point must be on/near the key top surface (not the sides)
            if c.pos[2] < key_top_z - 0.006:
                continue

            if abs(c.frame[2]) < 0.5:
                continue
            pressed.add(letter)
        return pressed

    def _gripper_contacts_key(self, arm_idx: int, letter: str) -> bool:
        """Return True if a finger PAD of ``arm_idx`` is pressing ``letter``'s key."""
        return letter.upper() in self._keys_pressed_by(arm_idx)

    def _key_world_xyz(self, letter: str) -> np.ndarray:
        """Return world-frame XYZ of the top surface of a key."""
        lx, ly = _KEY_POSITIONS[letter.upper()]
        kp = self._keyboard_world_pos
        # Keyboard board bottom = kp[2], key top surface = kp[2] + _KEY_TOP_Z
        return np.array([
            kp[0] + lx,
            kp[1] + ly,
            kp[2] + _KEY_TOP_Z,
        ])

    # ── Scripted arm (arm 0) step ─────────────────────────────────────

    def _drive_toward_home(self, raw_obs: dict, action: np.ndarray) -> bool:
        """Drive arm 0's EEF toward its captured home pose.

        Writes the OSC translation into ``action[:3]`` (in robot-0 base
        frame). Returns True iff the EEF is already within 1.5 cm of
        home — useful for FSM transitions. No-op when no home pose
        has been captured yet.
        """
        if self._arm0_home_pos is None:
            return False
        eef_pos = np.array(raw_obs.get("robot0_eef_pos", [0, 0, 0]))
        delta = self._arm0_home_pos - eef_pos
        if np.linalg.norm(delta) < 0.015:
            return True
        delta_base = self._robot0_base_R.T @ delta
        norm_delta = delta_base / _OSC_POS_SCALE
        action[:3] = np.clip(norm_delta, -0.6, 0.6)
        return False


    def _scripted_arm_action(self, raw_obs: dict) -> np.ndarray:
        """Return a 7-dim action for arm 0 to press the next scripted key.

        Finite-state machine: idle → hover → press → lift → (next key)
        While it is arm 1's turn, and after the last letter: → return →
        done (hold at home).

        Arm 0 only moves on its turn: when the next letter of the word is
        one of ``_arm0_indices``. Otherwise it drives back to the EEF
        position captured at reset, out of arm 1's way, and waits there.

        Actions are normalised to OSC_POSE's ``[-1, 1]`` space by dividing
        the metre-delta by ``_OSC_POS_SCALE`` (0.05 m per unit).
        """
        action = np.zeros(7, dtype=np.float64)
        action[6] = 1.0  # gripper closed (fingertip press)
        eef_pos = np.array(raw_obs.get("robot0_eef_pos", [0, 0, 0]))

        # Once the current key's press has registered, lift straight off it
        # before planning anything else.
        if (self._scripted_phase in ("press", "hover")
                and self._typed_mask[self._scripted_word_idx]):
            self._scripted_phase = "lift"
        if self._scripted_phase == "lift":
            hover_target = self._key_world_xyz(self._word[self._scripted_word_idx])
            hover_target[2] += self._hover_height
            delta = hover_target - eef_pos
            if np.linalg.norm(delta) < 0.012:
                self._scripted_phase = "idle"
            else:
                delta_base = self._robot0_base_R.T @ delta
                action[:3] = np.clip(delta_base / _OSC_POS_SCALE, -0.6, 0.6)
                return action

        # Not arm 0's turn (or the word is done) → return home, wait there.
        next_idx = self._next_word_idx()
        if next_idx is None or self._arm_for_letter(next_idx) != 0:
            if self._scripted_phase not in ("return", "done"):
                self._scripted_phase = (
                    "return" if self._arm0_home_pos is not None else "done"
                )
            if self._scripted_phase == "return":
                if self._drive_toward_home(raw_obs, action):
                    self._scripted_phase = "done"
            return action

        # Arm 0's turn → start hovering over the next letter's key.
        if self._scripted_phase in ("idle", "return", "done"):
            self._scripted_word_idx = next_idx
            hover_target = self._key_world_xyz(self._word[next_idx])
            hover_target[2] += self._hover_height
            self._scripted_target = hover_target
            self._scripted_phase = "hover"

        target_word_idx = self._scripted_word_idx
        key_xyz = self._key_world_xyz(self._word[target_word_idx])

        if self._scripted_phase == "hover":
            delta = self._scripted_target - eef_pos
            dist = np.linalg.norm(delta)
            # Transform world-frame delta to robot 0 base frame (input_ref_frame=base)
            delta_base = self._robot0_base_R.T @ delta
            norm_delta = delta_base / _OSC_POS_SCALE
            step = np.clip(norm_delta, -0.6, 0.6)  # cap speed at 60 % of max
            if dist < 0.008:
                self._scripted_phase = "press"
                self._scripted_hold_counter = 0
            else:
                action[:3] = step
            return action

        if self._scripted_phase == "press":
            press_target = key_xyz.copy()
            press_target[2] -= 0.05  # well below key surface — physics blocks arm, creating a real press
            delta = press_target - eef_pos
            delta_base = self._robot0_base_R.T @ delta
            norm_delta = delta_base / _OSC_POS_SCALE
            step = np.clip(norm_delta, -0.3, 0.3)  # slower approach
            # Advance to LIFT once detection has registered this key press
            if self._typed_mask[target_word_idx]:
                self._scripted_phase = "lift"
                self._scripted_hold_counter = 0
            else:
                action[:3] = step  # keep pressing
            return action

        return action

    # ── Step ─────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        """Override step to inject scripted arm 0 action.

        ``action`` should be a 14-dim bimanual action
        ``[arm0(6), gripper0(1), arm1(6), gripper1(1)]``.
        The first 7 dims (arm 0) are **replaced** by the scripted policy;
        the last 7 dims (arm 1) come from the teleoperator.
        ``info["applied_action"]`` is the 14-dim action actually executed,
        which is what the demo collectors record.
        """
        action = self._validate_action(action)

        raw_obs = self._get_sim_observations(force_update=True)

        # Scripted arm 0 override (first 7 dims)
        scripted = self._scripted_arm_action(raw_obs)
        action[:7] = scripted

        raw_obs_after, _rs_reward, _rs_done, info = self._env.step(action)

        # Key presses and the word display must update before the obs render.
        self._detect_key_presses(raw_obs_after)
        self._update_word_display()

        obs, reward, terminated, truncated, info = self._finish_step(raw_obs_after, info)
        # Letters are typed in word order, so the typed text is a prefix.
        next_idx = self._next_word_idx()
        info.update({
            "word": self._word,
            "typed": self._word[:self._typed_count],
            "next_letter": self._word[next_idx] if next_idx is not None else None,
            "typo": self._typo_key,
            "applied_action": action.copy(),
        })
        return obs, reward, terminated, truncated, info

    # ── Key-press detection logic ─────────────────────────────────────

    def _detect_key_presses(self, raw_obs: dict) -> None:
        """Type the next letter when its arm presses it; flag arm 1's typos.

        Only the next letter of the word can be typed, and only by the arm
        that owns it. Arm 1 holding any other key for ``hold_steps``
        steps — a wrong letter, or one of its own letters out of turn — is
        a typo: it is recorded in ``_typo_key`` and ends the episode. The
        key arm 1 has just typed is exempt until it is released.

        Per-letter hold counters (``_press_hold_per_letter``) prevent
        switching keys mid-press from cancelling progress.
        """
        next_idx = self._next_word_idx()
        if next_idx is None or self._typo_key is not None:
            return
        owner = self._arm_for_letter(next_idx)
        letter = self._word[next_idx]
        arm1_keys = self._keys_pressed_by(1)
        owner_keys = arm1_keys if owner == 1 else self._keys_pressed_by(0)
        self._tick_letter_hold(next_idx, letter in owner_keys)
        if owner == 1 and self._typed_mask[next_idx]:
            self._arm1_release_key = letter

        if self._arm1_release_key not in arm1_keys:
            self._arm1_release_key = None
        allowed = {self._arm1_release_key, letter if owner == 1 else None}
        self._wrong_hold = {k: c for k, c in self._wrong_hold.items() if k in arm1_keys}
        for key in arm1_keys - allowed:
            self._wrong_hold[key] = self._wrong_hold.get(key, 0) + 1
            if self._wrong_hold[key] >= self._press_hold_steps:
                self._typo_key = key

    def _tick_letter_hold(self, word_idx: int, in_contact: bool) -> None:
        """Advance the per-letter hold counter and mark typed once held.

        After a letter types, its key must be released before another
        occurrence of the same letter can count, so one press types one letter.
        """
        letter = self._word[word_idx]
        if letter in self._letters_to_release:
            if in_contact:
                self._press_hold_per_letter[word_idx] = 0
                return
            self._letters_to_release.discard(letter)
        c = self._press_hold_per_letter.get(word_idx, 0)
        if in_contact:
            c += 1
            if c >= self._press_hold_steps:
                self._typed_mask[word_idx] = True
                self._typed_count = sum(self._typed_mask)
                self._letters_to_release.add(letter)
                c = 0
        else:
            c = 0
        self._press_hold_per_letter[word_idx] = c

    # ── Task score ────────────────────────────────────────────────────

    def _compute_task_score(self, raw_obs) -> float:
        if self._staged_reward is None:
            return 1.0
        score, self._stage_debug = self._staged_reward.compute(self, raw_obs)
        return score

    # ── Termination ───────────────────────────────────────────────────

    def _check_termination(self, raw_obs) -> bool:
        # A typo fails the episode at once.
        if self._typo_key is not None:
            return True
        # Hold for ``_post_complete_hold_steps`` after the last letter so
        # the bold-glyph state is visible in the final observations.
        if self._typed_count < len(self._word):
            return False
        self._post_complete_counter += 1
        return self._post_complete_counter > self._post_complete_hold_steps
