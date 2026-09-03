"""
app/ui/widgets package
Reusable Apple HIG styled UI components.
"""
from app.ui.widgets.card import CardWidget
from app.ui.widgets.segmented_control import SegmentedControl
from app.ui.widgets.progress_pill import ProgressPill, StatusDot
from app.ui.widgets.thumbnail_loader import AsyncThumbnailManager

__all__ = [
    "CardWidget",
    "SegmentedControl",
    "ProgressPill",
    "StatusDot",
    "AsyncThumbnailManager",
]
