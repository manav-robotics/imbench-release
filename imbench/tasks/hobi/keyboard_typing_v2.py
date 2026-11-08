"""Keyboard Typing (OOD word list) — homogeneous bimanual task.

Same task as ``ib-m-keyboard-typing-v1`` (see that module for the rules and
scoring), but the word is drawn per episode from the ``words`` list in the
YAML config — five words held out from the training vocabulary — instead of
being fixed.

Task ID: ``ib-m-keyboard-typing-v2``
"""

from __future__ import annotations

from imbench.tasks.hobi.keyboard_typing_v1 import KeyboardTypingV1


class KeyboardTypingV2(KeyboardTypingV1):
    """Keyboard typing with the word resampled each episode from a fixed list.

    Parameters
    ----------
    task_id : str, optional
        Defaults to ``"ib-m-keyboard-typing-v2"``.
    **kwargs
        Forwarded to ``KeyboardTypingV1``.
    """

    TASK_ID = "ib-m-keyboard-typing-v2"
