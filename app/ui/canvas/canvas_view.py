"""
app/ui/canvas/canvas_view.py
Extended MangaCanvasView with live bubble item update synchronization.
"""
from typing import Dict, Any, Optional
from app.ui.canvas.view import MangaCanvasView as _BaseMangaCanvasView, cvimg_to_qpixmap


class MangaCanvasView(_BaseMangaCanvasView):
    """
    High-DPI Manga & Webtoon Canvas Viewport.
    Extended with dynamic bubble item update capabilities.
    """

    def update_bubble_item(self, block_data: Dict[str, Any]):
        """
        Finds the corresponding BubbleItem on the canvas scene and updates its
        geometry and rotation in real-time.
        """
        if not block_data or not hasattr(self, "bubble_items"):
            return
        target_id = str(block_data.get("id", ""))
        for item in self.bubble_items:
            item_block = getattr(item, "block_data", None)
            if item_block and str(item_block.get("id", "")) == target_id:
                item.update_block_data(block_data)
                break


# Also attach to _BaseMangaCanvasView so that instances created via app.ui.canvas.view
# immediately have update_bubble_item available.
_BaseMangaCanvasView.update_bubble_item = MangaCanvasView.update_bubble_item

__all__ = ["MangaCanvasView", "cvimg_to_qpixmap"]
