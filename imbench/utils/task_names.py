"""Human-readable task names written by every collector (episode ``task_name``, telemetry)."""

import re

TASK_DISPLAY_NAMES: dict[str, str] = {
    # Prehensile
    "ib-p1-edge-slide-v1":          "Edge Slide and Grasp",
    "ib-p2-cube-toss-v1":           "Cube Toss",
    "ib-p4-tool-retrieve-v1":       "Tool-Use Retrieval",
    "ib-p4-cup-extract-v1":         "Cup Extract (Single-Arm)",
    "ib-p5-occluder-push-v1":       "Occluder Push",
    "ib-p2-pendulum-grasp-v1":      "Pendulum Grasp",
    "ib-p1-sheltered-grasp-v1":     "Sheltered Grasp",
    "ib-p3-domino-single-v1":       "Domino Chain Trigger",
    "ib-p3-domino-select-v1":       "Domino Chain Selection",
    "ib-p3-seesaw-balance-v1":      "Seesaw Counterweight",
    "ib-p6-slip-recovery-v1":       "Slip Recovery",
    "ib-p2-linear-intercept-v1":    "Linear Intercept",
    "ib-p2-bounce-intercept-v1":    "Post-Collision Intercept",
    "ib-p5-mass-sort-v1":           "Adversarial Mass Sorting",
    "ib-p1-recover-peg-insert-v1":  "Recover Peg Insert",
    "ib-p2-galileo-ramp-drop-v1":   "Galileo Ramp Drop",
    "ib-p2-slide-catch-v1":         "Slide Catch",
    "ib-m-mirror-pick-place-v1":   "Mirror Pick and Place",
    # Puzzle
    "ib-p7-shape-stack-v1":     "Shape Stack",
    "ib-p6-stack-collapse-recovery-v1": "Stack Collapse Recovery",
    "ib-p2-ramp-sort-v1":       "Ramp Sort",
    "ib-p3-ball-cage-escape-v1": "Ball Cage Escape",
    "ib-p3-indirect-push-v1":   "Indirect Push",
    "ib-p3-gap-funnel-v1":      "Gap Funnel",
    "ib-p6-pick-place-ycb-v1":  "Pick and Place YCB (Moving Goal)",
    # Heterogeneous bimanual
    "ib-p1-bowl-table-v1":       "Bowl Table Transfer",
    "ib-p1-cracker-box-suction-v1": "Cracker Box to Slot (Suction Only)",
    "ib-p1-cracker-box-jaw-v1":  "Cracker Box to Slot (Parallel-Jaw Only)",
    "ib-p1-cup-inversion-v1":    "Cup Inversion",
    "ib-p1-plate-shelf-hetbi-v1":      "Plate Table-to-Shelf Transfer",
    # Single-arm prehensile (suction)
    "ib-p7-packing-v1":             "Packing (Single-arm Suction)",
    # Homogeneous bimanual
    "ib-p1-plate-shelf-hobi-v1":       "Plate Table-to-Shelf (Bimanual)",
    "ib-m-keyboard-typing-v1":   "Keyboard Typing (Bimanual)",
    "ib-m-dining-table-v1":      "Dining Table (Bimanual)",
    # Hard
    "ib-p7-balance-medium-v1":    "Balance Rod (Medium)",
    "ib-p7-balance-hard-v1":      "Balance Rod (Hard)",
    # Test / debug
    "ib-m-freeform-v1":        "Freeform (Custom)",
}


def task_display_name(task_id: str) -> str:
    """Return a human-readable name for *task_id*.

    Unlisted ``-vN`` variants reuse the ``-v1`` name with a version suffix;
    anything else is derived from the task id slug.
    """
    if task_id in TASK_DISPLAY_NAMES:
        return TASK_DISPLAY_NAMES[task_id]
    m = re.match(r"^(?P<base>.+)-v(?P<ver>\d+)$", task_id)
    if m and f"{m['base']}-v1" in TASK_DISPLAY_NAMES:
        return f"{TASK_DISPLAY_NAMES[m['base'] + '-v1']} (v{m['ver']})"
    slug = re.sub(r"-v\d+$", "", re.sub(r"^ib-(?:p\d+|m)-", "", task_id))
    return slug.replace("-", " ").title()
