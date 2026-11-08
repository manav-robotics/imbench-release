"""Tool-Use Retrieval v2 — OOD: hook mirror-flipped, tip on the opposite side.

Task ID: ``ib-p4-tool-retrieve-v2``

Same task as v1 (drag a distant cube into reach with an L-shaped hook,
then release the hook and grasp/lift the cube), but the hook's tip is
mirrored onto the opposite side of the shaft. Per the paper's OOD-shift
table for P4 (tool use for indirect object interaction): "standard hook
orientation used for retrieval -> hook mirror-flipped with the tip on
the opposite side, requiring strategy adaptation."
"""

from imbench.tasks.prehensile.tool_retrieve_v1 import ToolRetrieveV1


class ToolRetrieveV2(ToolRetrieveV1):
    TASK_ID = "ib-p4-tool-retrieve-v2"
    DESCRIPTION = (
        "Single arm: OOD variant of tool-retrieve. The L-shaped hook's tip "
        "is mirror-flipped onto the opposite side of the shaft (same shaft "
        "position/length), so the approach and drag direction that works "
        "for v1 no longer works directly -- the policy must adapt its "
        "strategy to the flipped geometry."
    )

    def _make_objects(self):
        specs = super()._make_objects()
        for spec in specs:
            if spec.get("name") == "hook_tool":
                shaft_loc, tip_loc = spec["sub_locations"]
                # Mirror the tip across the shaft's long axis (flip its
                # y-offset sign) -- everything else about the hook
                # (shaft position/length, tip length, total_size) is
                # unchanged.
                spec["sub_locations"] = [
                    shaft_loc,
                    [tip_loc[0], -tip_loc[1], tip_loc[2]],
                ]
        return specs
