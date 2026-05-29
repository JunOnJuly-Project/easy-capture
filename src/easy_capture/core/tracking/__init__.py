"""추적·occlusion·샷경계 재매칭 로직."""
from easy_capture.core.tracking.cut_selection import (
    CutSelection,
    ShotChoice,
    box_center,
    build_selections_from_choices,
    index_selections_by_shot,
    pick_box_at,
    validate_negative_points,
    validate_selections,
)
from easy_capture.core.tracking.gap_policy import GapPolicy, build_output_indices
from easy_capture.core.tracking.rematch import iou, rematch_score

__all__ = [
    "CutSelection",
    "GapPolicy",
    "ShotChoice",
    "box_center",
    "build_output_indices",
    "build_selections_from_choices",
    "index_selections_by_shot",
    "iou",
    "pick_box_at",
    "rematch_score",
    "validate_negative_points",
    "validate_selections",
]
