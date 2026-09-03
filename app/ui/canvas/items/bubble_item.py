"""
app/ui/canvas/items/bubble_item.py
Interactive speech bubble overlay item for QGraphicsView canvas.
Provides selection, dragging, hover states, coordinate normalization, and Apple HIG visual styling.
"""
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsItem, QWidget, QStyleOptionGraphicsItem
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QFont


class BubbleItemSignals(QObject):
    clicked = pyqtSignal(dict)
    double_clicked = pyqtSignal(dict)
    changed = pyqtSignal(dict)
    swap_prev_requested = pyqtSignal(str)
    swap_next_requested = pyqtSignal(str)


class BubbleItem(QGraphicsRectItem):
    """
    Interactive speech bubble overlay item.
    Renders bounding box with Apple-inspired accent colors, hover state, and ID tag.
    Maintains bidirectional synchronization with normalized [0.0, 100.0] coordinates.
    """

    def __init__(self, block_data: Dict[str, Any], img_w: int, img_h: int, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self.block_data = block_data
        self.img_w = max(1, int(img_w))
        self.img_h = max(1, int(img_h))
        self.signals = BubbleItemSignals()

        self.setAcceptHoverEvents(True)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        self.is_hovered = False
        self._update_geometry_from_data()

    def _update_geometry_from_data(self):
        xmin = (float(self.block_data.get("xmin", 0)) / 100.0) * self.img_w
        ymin = (float(self.block_data.get("ymin", 0)) / 100.0) * self.img_h
        xmax = (float(self.block_data.get("xmax", 0)) / 100.0) * self.img_w
        ymax = (float(self.block_data.get("ymax", 0)) / 100.0) * self.img_h

        w = max(10.0, xmax - xmin)
        h = max(10.0, ymax - ymin)
        self.setRect(0, 0, w, h)
        self.setPos(xmin, ymin)

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.signals.clicked.emit(self.block_data)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self.signals.double_clicked.emit(self.block_data)

    def contextMenuEvent(self, event):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu()
        b_id = str(self.block_data.get("id", ""))[:4]
        title_act = menu.addAction(f"气泡 #{b_id}")
        title_act.setEnabled(False)
        menu.addSeparator()
        act_swap_prev = menu.addAction("⬆️ 与上一气泡互换翻译")
        act_swap_next = menu.addAction("⬇️ 与下一气泡互换翻译")
        action = menu.exec(event.screenPos())
        full_id = str(self.block_data.get("id", ""))
        if action == act_swap_prev:
            self.signals.swap_prev_requested.emit(full_id)
        elif action == act_swap_next:
            self.signals.swap_next_requested.emit(full_id)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            pos = self.pos()
            rect = self.rect()
            self.block_data["xmin"] = round((pos.x() / self.img_w) * 100.0, 2)
            self.block_data["ymin"] = round((pos.y() / self.img_h) * 100.0, 2)
            self.block_data["xmax"] = round(((pos.x() + rect.width()) / self.img_w) * 100.0, 2)
            self.block_data["ymax"] = round(((pos.y() + rect.height()) / self.img_h) * 100.0, 2)
            self.signals.changed.emit(self.block_data)
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        if self.isSelected():
            # Selected: Apple HIG Vibrant Blue with solid highlight
            pen = QPen(QColor("#2563EB"), 2.0, Qt.PenStyle.SolidLine)
            brush = QBrush(QColor(37, 99, 235, 50))
        elif self.is_hovered:
            # Hover: Electric Sky Blue
            pen = QPen(QColor("#0284C7"), 1.8, Qt.PenStyle.SolidLine)
            brush = QBrush(QColor(14, 165, 233, 35))
        else:
            # Default: High-contrast distinct outline
            is_onoma = self.block_data.get("type") == "onomatopoeia"
            if is_onoma:
                # Vivid Amber for Onomatopoeia (contrast against white paper & art)
                pen = QPen(QColor("#D97706"), 1.6, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(245, 158, 11, 28))
            else:
                # High-contrast Royal Blue outline (clearly visible on pure white speech bubbles & black art)
                pen = QPen(QColor("#2563EB"), 1.6, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(37, 99, 235, 22))

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRoundedRect(rect, 4, 4)

        # Draw corner anchor ticks when selected for intuitive drag cues
        if self.isSelected():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2563EB"))
            handle_size = 6.0
            # 4 corners
            corners = [
                (rect.left(), rect.top()),
                (rect.right() - handle_size, rect.top()),
                (rect.left(), rect.bottom() - handle_size),
                (rect.right() - handle_size, rect.bottom() - handle_size),
            ]
            for cx, cy in corners:
                painter.drawRect(QRectF(cx, cy, handle_size, handle_size))

        # Draw mini index / ID pill with high contrast dark capsule and white text
        b_id = str(self.block_data.get("id", ""))[:4]
        if b_id:
            tag_rect = QRectF(rect.x() + 2, rect.y() + 2, min(42, rect.width() - 4), 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 42, 220))  # Slate-900 high contrast pill
            painter.drawRoundedRect(tag_rect, 3, 3)

            painter.setPen(QColor("#FFFFFF"))
            font = QFont("-apple-system, Segoe UI, sans-serif", 8, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, f"#{b_id}")

