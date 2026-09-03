"""
app/ui/canvas/items/split_slider_item.py
Interactive split-screen comparison item with draggable vertical divider and circular grip handle.
"""
from typing import Optional
from PyQt6.QtWidgets import QGraphicsObject, QGraphicsItem, QStyleOptionGraphicsItem, QWidget, QGraphicsSceneMouseEvent
from PyQt6.QtCore import QRectF, QPointF, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QPainterPath


class SplitSliderItem(QGraphicsObject):
    """
    Apple HIG Split-Slider interactive comparison item.
    Renders the original artwork on the left and seamlessly reveals translated artwork on the right.
    Features a 60 FPS draggable divider line with an Apple-style circular grip handle.
    """
    sig_split_changed = pyqtSignal(float)

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsToShape, False)

        self._original_pixmap = QPixmap()
        self._translated_pixmap = QPixmap()
        self._split_ratio = 0.5  # 0.0 to 1.0
        self._is_dragging = False
        self._handle_radius = 16.0
        self._hit_tolerance = 16.0

    def set_pixmaps(self, original: Optional[QPixmap], translated: Optional[QPixmap]):
        """Sets both comparison pixmaps and updates geometry."""
        self.prepareGeometryChange()
        self._original_pixmap = original if (original and not original.isNull()) else QPixmap()
        self._translated_pixmap = translated if (translated and not translated.isNull()) else QPixmap()
        self.update()

    def set_split_ratio(self, ratio: float):
        """Sets split divider position ratio [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(ratio)))
        if abs(self._split_ratio - clamped) > 1e-4:
            self._split_ratio = clamped
            self.update()
            self.sig_split_changed.emit(self._split_ratio)

    def split_ratio(self) -> float:
        return self._split_ratio

    def boundingRect(self) -> QRectF:
        pix = self._original_pixmap if not self._original_pixmap.isNull() else self._translated_pixmap
        if pix.isNull():
            return QRectF(0, 0, 0, 0)
        return QRectF(0, 0, float(pix.width()), float(pix.height()))

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None):
        if self._original_pixmap.isNull() and self._translated_pixmap.isNull():
            return

        w = float(self.boundingRect().width())
        h = float(self.boundingRect().height())
        split_x = self._split_ratio * w

        # 1. Base Layer: Original artwork
        if not self._original_pixmap.isNull():
            painter.drawPixmap(0, 0, self._original_pixmap)

        # 2. Reveal Layer: Translated artwork clipped to right of split line
        if not self._translated_pixmap.isNull():
            painter.save()
            painter.setClipRect(QRectF(split_x, 0, w - split_x, h))
            painter.drawPixmap(0, 0, self._translated_pixmap)
            painter.restore()

        # 3. Vertical Divider Line
        # Soft shadow behind line
        painter.setPen(QPen(QColor(0, 0, 0, 80), 3))
        painter.drawLine(int(split_x), 0, int(split_x), int(h))

        # Main crisp white line
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        painter.drawLine(int(split_x), 0, int(split_x), int(h))

        # 4. Apple HIG Circular Grip Handle
        handle_y = h / 2.0
        handle_rect = QRectF(
            split_x - self._handle_radius,
            handle_y - self._handle_radius,
            self._handle_radius * 2.0,
            self._handle_radius * 2.0,
        )

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Handle outer shadow
        painter.setBrush(QBrush(QColor(0, 0, 0, 60)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(handle_rect.adjusted(-1, 1, 1, 3))

        # Handle white circular body
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(QPen(QColor(0, 0, 0, 40), 1))
        painter.drawEllipse(handle_rect)

        # Chevrons inside handle (< | >)
        painter.setPen(QPen(QColor("#3B82F6"), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        # Left chevron <
        painter.drawLine(int(split_x - 5), int(handle_y), int(split_x - 2), int(handle_y - 4))
        painter.drawLine(int(split_x - 5), int(handle_y), int(split_x - 2), int(handle_y + 4))

        # Right chevron >
        painter.drawLine(int(split_x + 5), int(handle_y), int(split_x + 2), int(handle_y - 4))
        painter.drawLine(int(split_x + 5), int(handle_y), int(split_x + 2), int(handle_y + 4))

        painter.restore()

    def _is_near_split(self, pos: QPointF) -> bool:
        w = float(self.boundingRect().width())
        split_x = self._split_ratio * w
        return abs(pos.x() - split_x) <= self._hit_tolerance

    def hoverMoveEvent(self, event):
        if self._is_near_split(event.pos()):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._is_near_split(event.pos()):
            self._is_dragging = True
            self.setCursor(Qt.CursorShape.SplitHCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self._is_dragging:
            w = float(self.boundingRect().width())
            if w > 0:
                new_ratio = event.pos().x() / w
                self.set_split_ratio(new_ratio)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging:
            self._is_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)
