"""
app/ui/canvas/view.py
60 FPS High-DPI QGraphicsView Canvas with AnchorUnderMouse, smooth pan/zoom,
QOpenGLWidget acceleration with software QWidget fallback, and comparison modes.
"""
from typing import Optional, List, Any
import numpy as np
import cv2

from PyQt6.QtWidgets import QGraphicsView, QWidget
from PyQt6.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QWheelEvent, QMouseEvent, QKeyEvent

from app.ui.canvas.scene import MangaCanvasScene
from app.ui.canvas.items.bubble_item import BubbleItem


def cvimg_to_qpixmap(cv_img: Optional[np.ndarray]) -> QPixmap:
    """
    High-performance, safe conversion from OpenCV BGR/grayscale numpy ndarray to QPixmap.
    Handles None and empty images gracefully.
    """
    if cv_img is None or not isinstance(cv_img, np.ndarray) or cv_img.size == 0:
        return QPixmap()

    h, w = cv_img.shape[:2]
    if len(cv_img.shape) == 2:
        # Grayscale image
        qimg = QImage(cv_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())
    elif len(cv_img.shape) == 3:
        # BGR image
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())
    return QPixmap()


class MangaCanvasView(QGraphicsView):
    """
    60 FPS High-DPI Manga & Webtoon Canvas Viewport.
    Supports cursor-centered zoom (25% to 500%+), smooth pan, split-slider,
    and side-by-side comparison with hardware OpenGL acceleration.
    """
    sig_zoom_changed = pyqtSignal(float)
    sig_view_mode_changed = pyqtSignal(str)
    sig_split_changed = pyqtSignal(float)
    sig_bubble_selected = pyqtSignal(dict)
    sig_bubble_changed = pyqtSignal(dict)
    sig_bubble_swap_prev = pyqtSignal(str)
    sig_bubble_swap_next = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Hardware Acceleration: QOpenGLWidget with software QWidget fallback
        self._using_opengl = False
        self._init_viewport()

        # Custom Scene
        self._scene = MangaCanvasScene(self)
        self.setScene(self._scene)
        self._scene.sig_split_changed.connect(self._on_split_changed_from_scene)

        # High-DPI Rendering & Antialiasing setup
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

        # Background canvas styling
        self.setBackgroundBrush(QColor("#141416"))

        # State tracking
        self.original_cv: Optional[np.ndarray] = None
        self.translated_cv: Optional[np.ndarray] = None
        self.erased_cv: Optional[np.ndarray] = None
        self.blocks: List[Any] = []
        self.bubble_items: List[BubbleItem] = []
        self.view_mode = "translated"
        self.show_bubbles = True
        self.split_position = 0.5

        # Pan & Zoom state
        self.zoom_factor = 1.0
        self.is_panning = False
        self.pan_start_pos = QPointF()
        self._space_held = False

    def _init_viewport(self):
        """Initializes QOpenGLWidget viewport with graceful software fallback."""
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
            gl_widget = QOpenGLWidget()
            self.setViewport(gl_widget)
            self._using_opengl = True
        except Exception:
            self.setViewport(QWidget())
            self._using_opengl = False

    @property
    def scene(self) -> MangaCanvasScene:
        return self._scene

    @property
    def base_pixmap_item(self):
        """Compatibility property pointing to active background item."""
        if self.view_mode == "split_slider":
            return self._scene.split_slider_item
        return self._scene.background_item

    def _clear_bubbles(self):
        """Removes all bubble overlay items from the scene."""
        for item in self.bubble_items:
            if item.scene() == self._scene:
                self._scene.removeItem(item)
        self.bubble_items.clear()

    def set_show_bubbles(self, visible: bool):
        """Toggles visibility of all bubble overlays."""
        self.show_bubbles = visible
        for item in self.bubble_items:
            item.setVisible(visible)

    def _rebuild_bubbles(self):
        """Constructs interactive BubbleItem overlays matching the active image dimensions."""
        self._clear_bubbles()
        if self.view_mode == "side_by_side" or not self.blocks:
            return

        ref_img = self.translated_cv if self.translated_cv is not None else self.original_cv
        if ref_img is None:
            return

        h, w = ref_img.shape[:2]
        for b in self.blocks:
            b_dict = b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else vars(b))
            item = BubbleItem(b_dict, img_w=w, img_h=h)
            item.setVisible(self.show_bubbles)
            item.signals.clicked.connect(self.sig_bubble_selected.emit)
            item.signals.changed.connect(self.sig_bubble_changed.emit)
            item.signals.swap_prev_requested.connect(self.sig_bubble_swap_prev.emit)
            item.signals.swap_next_requested.connect(self.sig_bubble_swap_next.emit)
            self._scene.addItem(item)
            self.bubble_items.append(item)

    def set_data(
        self,
        original_cv: Optional[np.ndarray],
        translated_cv: Optional[np.ndarray] = None,
        erased_cv: Optional[np.ndarray] = None,
        blocks: Optional[List[Any]] = None,
    ):
        """Sets raw image data, generates pixmaps, and resets layout."""
        self.original_cv = original_cv
        self.translated_cv = translated_cv
        self.erased_cv = erased_cv
        self.blocks = blocks if blocks is not None else []

        orig_pix = cvimg_to_qpixmap(original_cv)
        trans_pix = cvimg_to_qpixmap(translated_cv)
        eras_pix = cvimg_to_qpixmap(erased_cv)

        self._scene.set_images(orig_pix, trans_pix, eras_pix)
        self.set_view_mode(self.view_mode)
        self._rebuild_bubbles()

    def set_view_mode(self, mode: str):
        """
        Updates viewport mode:
        'original' | 'translated' | 'inpainted' | 'side_by_side' | 'split_slider'
        """
        self.view_mode = mode
        self._scene.set_view_mode(mode)
        if mode == "side_by_side":
            self._clear_bubbles()
        elif self.blocks and not self.bubble_items:
            self._rebuild_bubbles()
        self.sig_view_mode_changed.emit(mode)

    def set_split_position(self, ratio: float):
        """Sets split slider ratio [0.0, 1.0]."""
        clamped = max(0.0, min(1.0, float(ratio)))
        self.split_position = clamped
        self._scene.set_split_position(clamped)

    def _on_split_changed_from_scene(self, ratio: float):
        self.split_position = ratio
        self.sig_split_changed.emit(ratio)

    # -------------------------------------------------------------------------
    # Zoom Engine: Mouse Wheel & Controls
    # -------------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent):
        """Smooth, cursor-centered zoom engine clamped from 0.05 to 10.0."""
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = self.zoom_factor * factor

        if new_zoom < 0.05:
            factor = 0.05 / self.zoom_factor
            self.zoom_factor = 0.05
        elif new_zoom > 10.0:
            factor = 10.0 / self.zoom_factor
            self.zoom_factor = 10.0
        else:
            self.zoom_factor = new_zoom

        self.scale(factor, factor)
        self.sig_zoom_changed.emit(self.zoom_factor)
        event.accept()

    def zoom_in(self):
        """Steps zoom in by 1.25x."""
        factor = 1.25
        if self.zoom_factor * factor <= 10.0:
            self.zoom_factor *= factor
            self.scale(factor, factor)
            self.sig_zoom_changed.emit(self.zoom_factor)

    def zoom_out(self):
        """Steps zoom out by 0.8x."""
        factor = 0.8
        if self.zoom_factor * factor >= 0.05:
            self.zoom_factor *= factor
            self.scale(factor, factor)
            self.sig_zoom_changed.emit(self.zoom_factor)

    def reset_zoom(self):
        """Resets zoom level to 1:1 (100%)."""
        self.resetTransform()
        self.zoom_factor = 1.0
        self.sig_zoom_changed.emit(1.0)

    def fit_in_view(self):
        """Fits active scene contents perfectly within the viewport window."""
        rect = self._scene.current_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

        # Update zoom factor based on current transform matrix
        transform = self.transform()
        self.zoom_factor = float(transform.m11())
        self.sig_zoom_changed.emit(self.zoom_factor)

    # -------------------------------------------------------------------------
    # Pan Engine: Middle-Click or Space+Left-Drag
    # -------------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space and not self._space_held:
            self._space_held = True
            if not self.is_panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            self._space_held = False
            if not self.is_panning:
                self.unsetCursor()
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        # Middle click OR Space+Left click triggers panning
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_held
        ):
            self.is_panning = True
            self.pan_start_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_panning:
            delta = event.position() - self.pan_start_pos
            self.pan_start_pos = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.is_panning and (
            event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton
        ):
            self.is_panning = False
            if self._space_held:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)
