from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsItem
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QFont

class BubbleItemSignals(QObject):
    clicked = pyqtSignal(dict)
    double_clicked = pyqtSignal(dict)
    changed = pyqtSignal(dict)

class BubbleItem(QGraphicsRectItem):
    def __init__(self, block_data: dict, img_w: int, img_h: int, parent=None):
        super().__init__(parent)
        self.block_data = block_data
        self.img_w = img_w
        self.img_h = img_h
        self.signals = BubbleItemSignals()

        self.setAcceptHoverEvents(True)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

        self._update_geometry_from_data()
        self.is_hovered = False

    def _update_geometry_from_data(self):
        xmin = (self.block_data.get("xmin", 0) / 100.0) * self.img_w
        ymin = (self.block_data.get("ymin", 0) / 100.0) * self.img_h
        xmax = (self.block_data.get("xmax", 0) / 100.0) * self.img_w
        ymax = (self.block_data.get("ymax", 0) / 100.0) * self.img_h

        w = max(10, xmax - xmin)
        h = max(10, ymax - ymin)
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

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            pos = self.pos()
            rect = self.rect()
            self.block_data["xmin"] = round((pos.x() / self.img_w) * 100.0, 2)
            self.block_data["ymin"] = round((pos.y() / self.img_h) * 100.0, 2)
            self.block_data["xmax"] = round(((pos.x() + rect.width()) / self.img_w) * 100.0, 2)
            self.block_data["ymax"] = round(((pos.y() + rect.height()) / self.img_h) * 100.0, 2)
            self.signals.changed.emit(self.block_data)
        return super().itemChange(change, value)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        if self.isSelected():
            # Selected: Apple Accent Blue
            pen = QPen(QColor("#0A84FF"), 2.0, Qt.PenStyle.SolidLine)
            brush = QBrush(QColor(10, 132, 255, 38))
        elif self.is_hovered:
            # Hover: Soft Cyan
            pen = QPen(QColor("#64D2FF"), 1.8, Qt.PenStyle.DashLine)
            brush = QBrush(QColor(100, 210, 255, 25))
        else:
            # Default: Translucent White / Orange for Onomatopoeia
            is_onoma = self.block_data.get("type") == "onomatopoeia"
            if is_onoma:
                pen = QPen(QColor(255, 159, 10, 180), 1.2, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(255, 159, 10, 15))
            else:
                pen = QPen(QColor(255, 255, 255, 160), 1.2, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(255, 255, 255, 12))

        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRoundedRect(rect, 4, 4)

        # Draw mini index / ID pill
        b_id = str(self.block_data.get("id", ""))[:4]
        if b_id:
            tag_rect = QRectF(rect.x() + 2, rect.y() + 2, min(36, rect.width() - 4), 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(tag_rect, 3, 3)

            painter.setPen(QColor("#FFFFFF"))
            font = QFont("-apple-system, Segoe UI, sans-serif", 8, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, f"#{b_id}")
