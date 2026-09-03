"""
app/ui/canvas/scene.py
QGraphicsScene coordinator managing background layers, split comparison items, and spatial indexing.
"""
from typing import Optional
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsItem
from PyQt6.QtCore import QRectF, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QColor

from app.ui.canvas.items.background_item import BackgroundItem
from app.ui.canvas.items.split_slider_item import SplitSliderItem


class MangaCanvasScene(QGraphicsScene):
    """
    Hardware-accelerated scene hosting manga artwork layers and interactive comparison widgets.
    Uses BSP tree spatial partitioning for sub-millisecond hit-testing.
    """
    sig_split_changed = pyqtSignal(float)
    sig_scene_rect_changed = pyqtSignal(QRectF)

    def __init__(self, parent: Optional[QGraphicsScene] = None):
        super().__init__(parent)
        self.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        self.setBackgroundBrush(QColor("#141416"))

        # Background artwork layer
        self.background_item = BackgroundItem()
        self.addItem(self.background_item)

        # Split slider comparison layer
        self.split_slider_item = SplitSliderItem()
        self.split_slider_item.sig_split_changed.connect(self.sig_split_changed)
        self.addItem(self.split_slider_item)
        self.split_slider_item.hide()

        self._view_mode = "translated"
        self._current_rect = QRectF(0, 0, 0, 0)

    def set_images(
        self,
        original_pixmap: Optional[QPixmap],
        translated_pixmap: Optional[QPixmap] = None,
        erased_pixmap: Optional[QPixmap] = None
    ):
        """Loads artwork pixmaps into scene items and updates coordinate boundaries."""
        orig = original_pixmap if (original_pixmap and not original_pixmap.isNull()) else QPixmap()
        trans = translated_pixmap if (translated_pixmap and not translated_pixmap.isNull()) else QPixmap()
        eras = erased_pixmap if (erased_pixmap and not erased_pixmap.isNull()) else QPixmap()

        self.background_item.set_pixmaps(orig, trans, eras)
        self.split_slider_item.set_pixmaps(orig, trans)

        self._update_layout()

    def set_view_mode(self, mode: str):
        """
        Updates active display mode:
        'original' | 'translated' | 'inpainted' | 'side_by_side' | 'split_slider'
        """
        self._view_mode = mode
        if mode == "split_slider":
            self.background_item.hide()
            self.split_slider_item.show()
        else:
            self.split_slider_item.hide()
            self.background_item.show()
            self.background_item.set_mode(mode)

        self._update_layout()

    def set_split_position(self, ratio: float):
        """Sets split-slider divider position ratio [0.0, 1.0]."""
        self.split_slider_item.set_split_ratio(ratio)

    def _update_layout(self):
        """Recalculates scene rectangle and notifies subscribers."""
        if self._view_mode == "split_slider":
            rect = self.split_slider_item.boundingRect()
        else:
            rect = self.background_item.boundingRect()

        if rect.width() > 0 and rect.height() > 0:
            self._current_rect = rect
            self.setSceneRect(rect)
            self.sig_scene_rect_changed.emit(rect)
        else:
            self._current_rect = QRectF(0, 0, 0, 0)
            self.setSceneRect(self._current_rect)

    def current_rect(self) -> QRectF:
        return self._current_rect
