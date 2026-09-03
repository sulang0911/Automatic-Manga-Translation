"""
app/ui/canvas/items/background_item.py
High-performance QGraphicsItem for rendering manga scans in single and side-by-side comparison modes.
"""
from typing import Optional
from PyQt6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen


class BackgroundItem(QGraphicsItem):
    """
    Renders the background manga page artwork.
    Supports single image display (original/translated/erased) and side-by-side dual display.
    """

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsToShape, False)
        self._original_pixmap = QPixmap()
        self._translated_pixmap = QPixmap()
        self._erased_pixmap = QPixmap()
        self._mode = "translated"  # "original" | "translated" | "inpainted" | "side_by_side"

    def set_pixmaps(
        self,
        original: Optional[QPixmap] = None,
        translated: Optional[QPixmap] = None,
        erased: Optional[QPixmap] = None,
    ):
        """Sets artwork pixmaps and schedules geometry update."""
        self.prepareGeometryChange()
        self._original_pixmap = original if (original and not original.isNull()) else QPixmap()
        self._translated_pixmap = translated if (translated and not translated.isNull()) else QPixmap()
        self._erased_pixmap = erased if (erased and not erased.isNull()) else QPixmap()
        self.update()

    def set_mode(self, mode: str):
        """Switches display mode: 'original', 'translated', 'inpainted', 'side_by_side'."""
        if self._mode != mode:
            self.prepareGeometryChange()
            self._mode = mode
            self.update()

    def active_pixmap(self) -> QPixmap:
        """Returns the primary active pixmap for single view mode."""
        if self._mode == "original":
            return self._original_pixmap
        elif self._mode == "inpainted":
            return self._erased_pixmap if not self._erased_pixmap.isNull() else self._original_pixmap
        else:
            # Translated mode falls back to original if translated not ready
            return self._translated_pixmap if not self._translated_pixmap.isNull() else self._original_pixmap

    def pixmap(self) -> QPixmap:
        """Compatibility alias returning the primary active pixmap."""
        return self.active_pixmap()

    def boundingRect(self) -> QRectF:
        pix = self.active_pixmap()
        if pix.isNull():
            return QRectF(0, 0, 0, 0)

        w = float(pix.width())
        h = float(pix.height())

        if self._mode == "side_by_side":
            # Side by side stacks original on left and translated on right (2x width)
            return QRectF(0, 0, w * 2.0, h)
        return QRectF(0, 0, w, h)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None):
        pix = self.active_pixmap()
        if pix.isNull():
            return

        w = float(pix.width())
        h = float(pix.height())

        if self._mode == "side_by_side":
            # Left side: Original manga scan
            left_pix = self._original_pixmap if not self._original_pixmap.isNull() else pix
            painter.drawPixmap(0, 0, left_pix)

            # Right side: Translated (or inpainted fallback)
            right_pix = self._translated_pixmap if not self._translated_pixmap.isNull() else (
                self._erased_pixmap if not self._erased_pixmap.isNull() else self._original_pixmap
            )
            painter.drawPixmap(int(w), 0, right_pix)

            # Subtle 1px vertical divider between pages
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
            painter.drawLine(int(w), 0, int(w), int(h))
        else:
            painter.drawPixmap(0, 0, pix)
