"""
app/ui/canvas/items/bubble_item.py
Interactive speech bubble overlay item for QGraphicsView canvas.
Provides high-performance selection, 8-directional drag resizing, body moving,
hover states, coordinate normalization, and Apple HIG visual styling.
"""
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsItem, QWidget, QStyleOptionGraphicsItem
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QFont

HANDLE_NONE = 0
HANDLE_TOP_LEFT = 1
HANDLE_TOP_MID = 2
HANDLE_TOP_RIGHT = 3
HANDLE_RIGHT_MID = 4
HANDLE_BOTTOM_RIGHT = 5
HANDLE_BOTTOM_MID = 6
HANDLE_BOTTOM_LEFT = 7
HANDLE_LEFT_MID = 8

HANDLE_SIZE = 7.0


class BubbleItemSignals(QObject):
    clicked = pyqtSignal(dict)
    double_clicked = pyqtSignal(dict)
    changed = pyqtSignal(dict)
    geometry_commit = pyqtSignal(dict)
    swap_prev_requested = pyqtSignal(str)
    swap_next_requested = pyqtSignal(str)


class BubbleItem(QGraphicsRectItem):
    """
    Interactive speech bubble overlay item.
    Renders bounding box with Apple-inspired accent colors, hover state, 8-directional resize handles, and ID tag.
    Maintains bidirectional synchronization with normalized [0.0, 100.0] coordinates.
    """

    def __init__(self, block_data: Dict[str, Any], img_w: int, img_h: int, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self.block_data = block_data
        self.img_w = max(1, int(img_w))
        self.img_h = max(1, int(img_h))
        self.signals = BubbleItemSignals()

        # Always stay on top of background images
        self.setZValue(100.0)
        self.setAcceptHoverEvents(True)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        self.is_hovered = False
        self._hover_handle = HANDLE_NONE
        self._active_handle = HANDLE_NONE
        self._is_moving_body = False
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_start_rect: Optional[QRectF] = None
        self._drag_start_scene_pos: Optional[QPointF] = None

        self._update_geometry_from_data()

    def _update_geometry_from_data(self):
        xmin = (float(self.block_data.get("xmin", 0)) / 100.0) * self.img_w
        ymin = (float(self.block_data.get("ymin", 0)) / 100.0) * self.img_h
        xmax = (float(self.block_data.get("xmax", 0)) / 100.0) * self.img_w
        ymax = (float(self.block_data.get("ymax", 0)) / 100.0) * self.img_h

        w = max(12.0, xmax - xmin)
        h = max(12.0, ymax - ymin)
        self.setRect(0, 0, w, h)
        self.setPos(xmin, ymin)

    def boundingRect(self) -> QRectF:
        """Expands bounding rect so mouse hit-testing catches handles outside the inner rect."""
        pad = HANDLE_SIZE + 6.0
        return self.rect().adjusted(-pad, -pad, pad, pad)

    def _get_handle_at(self, pos: QPointF) -> int:
        if not self.isSelected():
            return HANDLE_NONE
        rect = self.rect()
        hs = HANDLE_SIZE + 4.0

        # Check 4 corners first
        if abs(pos.x() - rect.left()) <= hs and abs(pos.y() - rect.top()) <= hs:
            return HANDLE_TOP_LEFT
        if abs(pos.x() - rect.right()) <= hs and abs(pos.y() - rect.top()) <= hs:
            return HANDLE_TOP_RIGHT
        if abs(pos.x() - rect.left()) <= hs and abs(pos.y() - rect.bottom()) <= hs:
            return HANDLE_BOTTOM_LEFT
        if abs(pos.x() - rect.right()) <= hs and abs(pos.y() - rect.bottom()) <= hs:
            return HANDLE_BOTTOM_RIGHT

        # Check 4 edge midpoints
        if abs(pos.y() - rect.top()) <= hs and rect.left() <= pos.x() <= rect.right():
            return HANDLE_TOP_MID
        if abs(pos.y() - rect.bottom()) <= hs and rect.left() <= pos.x() <= rect.right():
            return HANDLE_BOTTOM_MID
        if abs(pos.x() - rect.left()) <= hs and rect.top() <= pos.y() <= rect.bottom():
            return HANDLE_LEFT_MID
        if abs(pos.x() - rect.right()) <= hs and rect.top() <= pos.y() <= rect.bottom():
            return HANDLE_RIGHT_MID

        return HANDLE_NONE

    def hoverEnterEvent(self, event):
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.is_hovered = False
        self._hover_handle = HANDLE_NONE
        self.unsetCursor()
        self.update()
        super().hoverLeaveEvent(event)

    def hoverMoveEvent(self, event):
        pos = event.pos() if hasattr(event, "pos") else QPointF()
        handle = self._get_handle_at(pos)
        self._hover_handle = handle

        if handle in (HANDLE_TOP_LEFT, HANDLE_BOTTOM_RIGHT):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in (HANDLE_TOP_RIGHT, HANDLE_BOTTOM_LEFT):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in (HANDLE_TOP_MID, HANDLE_BOTTOM_MID):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle in (HANDLE_LEFT_MID, HANDLE_RIGHT_MID):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self.isSelected():
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        super().hoverMoveEvent(event)

    def _extract_scene_pos(self, event) -> QPointF:
        if hasattr(event, "scenePos"):
            val = event.scenePos
            return val() if callable(val) else val
        elif hasattr(event, "position"):
            val = event.position
            return val() if callable(val) else val
        elif hasattr(event, "pos"):
            val = event.pos
            return val() if callable(val) else val
        return QPointF()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Explicitly select this item
            if not self.isSelected():
                if self.scene():
                    self.scene().clearSelection()
                self.setSelected(True)

            pos = event.pos() if hasattr(event, "pos") else QPointF()
            handle = self._get_handle_at(pos)

            if handle != HANDLE_NONE:
                self._active_handle = handle
                self._is_moving_body = False
            else:
                self._active_handle = HANDLE_NONE
                self._is_moving_body = True

            self._drag_start_pos = self._extract_scene_pos(event)
            self._drag_start_rect = QRectF(self.rect())
            self._drag_start_scene_pos = QPointF(self.pos())
            self.signals.clicked.emit(self.block_data)
            self.update()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        min_dim = 12.0
        sp_event = self._extract_scene_pos(event)

        # 1. Resizing via handle
        if self._active_handle != HANDLE_NONE and self._drag_start_pos is not None:
            delta = sp_event - self._drag_start_pos
            r = self._drag_start_rect
            sp = self._drag_start_scene_pos

            new_x = sp.x()
            new_y = sp.y()
            new_w = r.width()
            new_h = r.height()

            if self._active_handle == HANDLE_BOTTOM_RIGHT:
                new_w = max(min_dim, r.width() + delta.x())
                new_h = max(min_dim, r.height() + delta.y())
            elif self._active_handle == HANDLE_BOTTOM_LEFT:
                new_w = max(min_dim, r.width() - delta.x())
                new_h = max(min_dim, r.height() + delta.y())
                new_x = sp.x() + (r.width() - new_w)
            elif self._active_handle == HANDLE_TOP_RIGHT:
                new_w = max(min_dim, r.width() + delta.x())
                new_h = max(min_dim, r.height() - delta.y())
                new_y = sp.y() + (r.height() - new_h)
            elif self._active_handle == HANDLE_TOP_LEFT:
                new_w = max(min_dim, r.width() - delta.x())
                new_h = max(min_dim, r.height() - delta.y())
                new_x = sp.x() + (r.width() - new_w)
                new_y = sp.y() + (r.height() - new_h)
            elif self._active_handle == HANDLE_RIGHT_MID:
                new_w = max(min_dim, r.width() + delta.x())
            elif self._active_handle == HANDLE_LEFT_MID:
                new_w = max(min_dim, r.width() - delta.x())
                new_x = sp.x() + (r.width() - new_w)
            elif self._active_handle == HANDLE_BOTTOM_MID:
                new_h = max(min_dim, r.height() + delta.y())
            elif self._active_handle == HANDLE_TOP_MID:
                new_h = max(min_dim, r.height() - delta.y())
                new_y = sp.y() + (r.height() - new_h)

            self.setPos(new_x, new_y)
            self.setRect(0, 0, new_w, new_h)
            self._sync_block_coords()
            self.signals.changed.emit(self.block_data)
            event.accept()
            return

        # 2. Moving whole bubble body
        if self._is_moving_body and self._drag_start_pos is not None:
            delta = sp_event - self._drag_start_pos
            sp = self._drag_start_scene_pos
            new_x = max(0.0, min(float(self.img_w - self.rect().width()), sp.x() + delta.x()))
            new_y = max(0.0, min(float(self.img_h - self.rect().height()), sp.y() + delta.y()))
            self.setPos(new_x, new_y)
            self._sync_block_coords()
            self.signals.changed.emit(self.block_data)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        was_dragging = (self._active_handle != HANDLE_NONE) or self._is_moving_body
        self._active_handle = HANDLE_NONE
        self._is_moving_body = False

        if was_dragging:
            self._sync_block_coords()
            self.signals.changed.emit(self.block_data)
            self.signals.geometry_commit.emit(self.block_data)
            self.update()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def _sync_block_coords(self):
        p = self.pos()
        r = self.rect()
        self.block_data["xmin"] = round(max(0.0, min(100.0, (p.x() / self.img_w) * 100.0)), 2)
        self.block_data["ymin"] = round(max(0.0, min(100.0, (p.y() / self.img_h) * 100.0)), 2)
        self.block_data["xmax"] = round(max(0.0, min(100.0, ((p.x() + r.width()) / self.img_w) * 100.0)), 2)
        self.block_data["ymax"] = round(max(0.0, min(100.0, ((p.y() + r.height()) / self.img_h) * 100.0)), 2)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._sync_block_coords()
            self.signals.changed.emit(self.block_data)
        return super().itemChange(change, value)

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

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        if self.isSelected():
            # Selected: Apple HIG Vibrant Blue with solid highlight and fill
            pen = QPen(QColor("#2563EB"), 2.0, Qt.PenStyle.SolidLine)
            brush = QBrush(QColor(37, 99, 235, 45))
        elif self.is_hovered:
            # Hover: Electric Sky Blue
            pen = QPen(QColor("#0284C7"), 1.8, Qt.PenStyle.SolidLine)
            brush = QBrush(QColor(14, 165, 233, 30))
        else:
            # Default: Distinct dashed outline
            is_onoma = self.block_data.get("type") == "onomatopoeia"
            if is_onoma:
                pen = QPen(QColor("#D97706"), 1.6, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(245, 158, 11, 25))
            else:
                pen = QPen(QColor("#2563EB"), 1.6, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(37, 99, 235, 20))

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRoundedRect(rect, 4, 4)

        # Draw 8 interactive handles when selected
        if self.isSelected():
            painter.setPen(QPen(QColor("#2563EB"), 1.5))
            painter.setBrush(QColor("#FFFFFF"))
            hs = HANDLE_SIZE
            half_hs = hs / 2.0

            handle_pts = [
                # 4 Corners
                (rect.left(), rect.top()),
                (rect.right(), rect.top()),
                (rect.left(), rect.bottom()),
                (rect.right(), rect.bottom()),
                # 4 Midpoints
                (rect.center().x(), rect.top()),
                (rect.center().x(), rect.bottom()),
                (rect.left(), rect.center().y()),
                (rect.right(), rect.center().y()),
            ]
            for cx, cy in handle_pts:
                painter.drawRect(QRectF(cx - half_hs, cy - half_hs, hs, hs))

        # Draw mini index / ID pill
        b_id = str(self.block_data.get("id", ""))[:4]
        if b_id:
            tag_rect = QRectF(rect.x() + 3, rect.y() + 3, min(44, rect.width() - 6), 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(15, 23, 42, 225))  # Slate-900 pill
            painter.drawRoundedRect(tag_rect, 3, 3)

            painter.setPen(QColor("#FFFFFF"))
            font = QFont("-apple-system, Segoe UI, sans-serif", 8, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, f"#{b_id}")

        # Draw live text preview while moving or resizing
        trans_text = self.block_data.get("translated_text", "")
        if trans_text and (self._is_moving_body or self._active_handle != HANDLE_NONE):
            painter.save()
            painter.setPen(QColor("#1D4ED8"))
            font = QFont("-apple-system, Segoe UI, sans-serif", 9, QFont.Weight.DemiBold)
            painter.setFont(font)
            text_rect = rect.adjusted(6, 22, -6, -6)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                trans_text
            )
            painter.restore()
