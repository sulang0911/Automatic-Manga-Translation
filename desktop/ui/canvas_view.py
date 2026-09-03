import cv2
import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem, QWidget
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QWheelEvent, QMouseEvent

from .bubble_item import BubbleItem

def cvimg_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    if cv_img is None or cv_img.size == 0:
        return QPixmap()
    h, w = cv_img.shape[:2]
    if len(cv_img.shape) == 2:
        qimg = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        # BGR to RGB
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

class CanvasView(QGraphicsView):
    sig_bubble_selected = pyqtSignal(dict)
    sig_bubble_double_clicked = pyqtSignal(dict)
    sig_bubble_changed = pyqtSignal(dict)
    sig_zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # High-DPI & Antialiasing setup
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.TextAntialiasing
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # Canvas State
        self.original_cv = None
        self.translated_cv = None
        self.erased_cv = None
        self.blocks = []
        self.view_mode = "translated"  # "translated" | "original" | "inpainted" | "side_by_side" | "split_slider"
        self.show_bubbles = True
        self.split_position = 0.5  # 0.0 to 1.0 for split slider

        # Background grid styling
        self.setBackgroundBrush(QColor("#141416"))

        # Panning State
        self.is_panning = False
        self.pan_start_pos = QPointF()
        self.zoom_factor = 1.0

        # Graphics Items
        self.base_pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.base_pixmap_item)
        self.bubble_items = []

    def set_data(self, original_cv: np.ndarray, translated_cv: np.ndarray = None, 
                 erased_cv: np.ndarray = None, blocks: list = None):
        self.original_cv = original_cv
        self.translated_cv = translated_cv
        self.erased_cv = erased_cv
        self.blocks = blocks or []
        self.refresh_display()
        self.fit_in_view()

    def set_view_mode(self, mode: str):
        self.view_mode = mode
        self.refresh_display()

    def set_show_bubbles(self, show: bool):
        self.show_bubbles = show
        for item in self.bubble_items:
            item.setVisible(show)

    def set_split_position(self, pos: float):
        self.split_position = max(0.0, min(1.0, pos))
        if self.view_mode == "split_slider":
            self.refresh_display()

    def refresh_display(self):
        if self.original_cv is None:
            self.base_pixmap_item.setPixmap(QPixmap())
            self.scene.setSceneRect(QRectF())
            self._clear_bubbles()
            return

        h, w = self.original_cv.shape[:2]
        composed_cv = None

        if self.view_mode == "original" or (self.translated_cv is None and self.erased_cv is None):
            composed_cv = self.original_cv
        elif self.view_mode == "inpainted" and self.erased_cv is not None:
            composed_cv = self.erased_cv
        elif self.view_mode == "translated" and self.translated_cv is not None:
            composed_cv = self.translated_cv
        elif self.view_mode == "side_by_side":
            right_img = self.translated_cv if self.translated_cv is not None else self.original_cv
            composed_cv = np.hstack([self.original_cv, right_img])
        elif self.view_mode == "split_slider" and self.translated_cv is not None:
            split_x = int(w * self.split_position)
            composed_cv = np.zeros_like(self.original_cv)
            composed_cv[:, :split_x] = self.original_cv[:, :split_x]
            composed_cv[:, split_x:] = self.translated_cv[:, split_x:]
            # Draw sleek vertical divider line
            cv2.line(composed_cv, (split_x, 0), (split_x, h), (255, 255, 255), 2)
        else:
            composed_cv = self.translated_cv if self.translated_cv is not None else self.original_cv

        pixmap = cvimg_to_qpixmap(composed_cv)
        self.base_pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))

        # Recreate bubble items
        self._recreate_bubbles(w, h)

    def _clear_bubbles(self):
        for item in self.bubble_items:
            self.scene.removeItem(item)
        self.bubble_items.clear()

    def _recreate_bubbles(self, w: int, h: int):
        self._clear_bubbles()
        if not self.blocks or self.view_mode == "side_by_side":
            return

        for block in self.blocks:
            b_item = BubbleItem(block, w, h)
            b_item.signals.clicked.connect(self.sig_bubble_selected.emit)
            b_item.signals.double_clicked.connect(self.sig_bubble_double_clicked.emit)
            b_item.signals.changed.connect(self.sig_bubble_changed.emit)
            b_item.setVisible(self.show_bubbles)
            self.scene.addItem(b_item)
            self.bubble_items.append(b_item)

    def fit_in_view(self):
        if not self.scene.sceneRect().isEmpty():
            self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_factor = self.transform().m11()
            self.sig_zoom_changed.emit(self.zoom_factor)

    def wheelEvent(self, event: QWheelEvent):
        # Zoom with mouse wheel
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_factor = self.zoom_factor * factor
        if 0.1 <= new_factor <= 10.0:
            self.scale(factor, factor)
            self.zoom_factor = new_factor
            self.sig_zoom_changed.emit(self.zoom_factor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in [Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton] or \
           (event.button() == Qt.MouseButton.LeftButton and event.modifiers() == Qt.KeyboardModifier.AltModifier):
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
