"""
app/ui/canvas/view.py
60 FPS High-DPI QGraphicsView Canvas with AnchorUnderMouse, smooth pan/zoom,
QOpenGLWidget acceleration with software QWidget fallback, comparison modes,
interactive manual bubble creation, and floating Apple HIG zoom HUD.
"""
from typing import Optional, List, Any
import uuid
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QGraphicsView, QWidget, QFrame, QHBoxLayout, QToolButton,
    QLabel, QGraphicsRectItem
)
from PyQt6.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QWheelEvent, QMouseEvent,
    QKeyEvent, QPen, QBrush
)

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


class CanvasZoomHud(QFrame):
    """
    Floating HUD capsule widget anchored in the canvas bottom-right corner.
    Integrates zoom in/out, 100%, fit view, and Draw Bubble tool.
    """
    sig_zoom_in = pyqtSignal()
    sig_zoom_out = pyqtSignal()
    sig_zoom_reset = pyqtSignal()
    sig_zoom_fit = pyqtSignal()
    sig_tool_draw_toggled = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("canvasZoomHud")
        self.setStyleSheet("""
            #canvasZoomHud {
                background-color: rgba(24, 24, 27, 0.88);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 14px;
            }
            QToolButton {
                color: #D4D4D8;
                background: transparent;
                border: none;
                padding: 4px 6px;
                font-size: 11px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.15);
                color: #FFFFFF;
            }
            QToolButton:checked {
                background: #2563EB;
                color: #FFFFFF;
            }
            QLabel {
                color: #E4E4E7;
                font-size: 11px;
                font-weight: 600;
                padding: 0 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(4)

        # Draw tool button
        self.btn_draw = QToolButton(self)
        self.btn_draw.setText("➕ 框选 (R)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("手动在画布上拖拽框选新建气泡 (快捷键: R)")
        self.btn_draw.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_draw.toggled.connect(self.sig_tool_draw_toggled.emit)
        layout.addWidget(self.btn_draw)

        # Separator line
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: rgba(255, 255, 255, 0.2);")
        layout.addWidget(sep)

        # Zoom controls
        self.btn_zoom_out = QToolButton(self)
        self.btn_zoom_out.setText("−")
        self.btn_zoom_out.setToolTip("缩小")
        self.btn_zoom_out.clicked.connect(self.sig_zoom_out.emit)
        layout.addWidget(self.btn_zoom_out)

        self.zoom_lbl = QLabel("100%", self)
        layout.addWidget(self.zoom_lbl)

        self.btn_zoom_in = QToolButton(self)
        self.btn_zoom_in.setText("+")
        self.btn_zoom_in.setToolTip("放大")
        self.btn_zoom_in.clicked.connect(self.sig_zoom_in.emit)
        layout.addWidget(self.btn_zoom_in)

        self.btn_fit = QToolButton(self)
        self.btn_fit.setText("适应")
        self.btn_fit.setToolTip("适应窗口")
        self.btn_fit.clicked.connect(self.sig_zoom_fit.emit)
        layout.addWidget(self.btn_fit)

        self.btn_reset = QToolButton(self)
        self.btn_reset.setText("1:1")
        self.btn_reset.setToolTip("恢复实际大小 100%")
        self.btn_reset.clicked.connect(self.sig_zoom_reset.emit)
        layout.addWidget(self.btn_reset)


class MangaCanvasView(QGraphicsView):
    """
    60 FPS High-DPI Manga & Webtoon Canvas Viewport.
    Supports cursor-centered zoom (25% to 500%+), smooth pan, split-slider,
    side-by-side comparison, manual bubble creation, and floating Apple HIG zoom HUD.
    """
    sig_zoom_changed = pyqtSignal(float)
    sig_view_mode_changed = pyqtSignal(str)
    sig_split_changed = pyqtSignal(float)
    sig_bubble_selected = pyqtSignal(dict)
    sig_bubble_changed = pyqtSignal(dict)
    sig_bubble_swap_prev = pyqtSignal(str)
    sig_bubble_swap_next = pyqtSignal(str)
    sig_bubble_created = pyqtSignal(dict)
    sig_bubble_commit = pyqtSignal(dict)
    sig_tool_mode_changed = pyqtSignal(str)

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

        # Tool Mode & Manual Rect Drawing
        self.tool_mode = "select"  # "select" | "draw"
        self._is_drawing_rect = False
        self._draw_start_scene_pt: Optional[QPointF] = None
        self._rubber_band_item: Optional[QGraphicsRectItem] = None

        # Floating HUD widget
        self.hud = CanvasZoomHud(self)
        self.hud.sig_zoom_in.connect(self.zoom_in)
        self.hud.sig_zoom_out.connect(self.zoom_out)
        self.hud.sig_zoom_reset.connect(self.reset_zoom)
        self.hud.sig_zoom_fit.connect(self.fit_in_view)
        self.hud.sig_tool_draw_toggled.connect(self._on_hud_draw_toggled)

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_hud_position()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_hud_position()

    def _update_hud_position(self):
        if hasattr(self, "hud") and self.hud:
            margin = 16
            x = self.viewport().width() - self.hud.width() - margin
            y = self.viewport().height() - self.hud.height() - margin
            self.hud.move(max(10, x), max(10, y))

    def _on_hud_draw_toggled(self, checked: bool):
        self.set_tool_mode("draw" if checked else "select")

    def toggle_draw_tool(self):
        """Toggles draw tool mode (called by shortcut R)."""
        new_state = (self.tool_mode != "draw")
        self.hud.btn_draw.setChecked(new_state)

    def set_tool_mode(self, mode: str):
        self.tool_mode = mode
        if hasattr(self, "hud") and self.hud:
            self.hud.btn_draw.blockSignals(True)
            self.hud.btn_draw.setChecked(mode == "draw")
            self.hud.btn_draw.blockSignals(False)
        if mode == "draw":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.unsetCursor()
        self.sig_tool_mode_changed.emit(mode)

    def _clear_bubbles(self):
        """Removes all bubble overlay items from the scene."""
        for item in self.bubble_items:
            try:
                self._scene.removeItem(item)
            except RuntimeError:
                pass
        self.bubble_items.clear()

    def _rebuild_bubbles(self):
        """Recreates interactive BubbleItems based on current blocks and visibility."""
        self._clear_bubbles()
        if not self.show_bubbles or not self.blocks:
            return

        img_h, img_w = 1, 1
        if self.original_cv is not None and self.original_cv.size > 0:
            img_h, img_w = self.original_cv.shape[:2]
        elif self.translated_cv is not None and self.translated_cv.size > 0:
            img_h, img_w = self.translated_cv.shape[:2]

        for b in self.blocks:
            b_dict = b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else vars(b))
            item = BubbleItem(b_dict, img_w, img_h)
            item.signals.clicked.connect(self.sig_bubble_selected.emit)
            item.signals.changed.connect(self.sig_bubble_changed.emit)
            item.signals.geometry_commit.connect(self.sig_bubble_commit.emit)
            item.signals.swap_prev_requested.connect(self.sig_bubble_swap_prev.emit)
            item.signals.swap_next_requested.connect(self.sig_bubble_swap_next.emit)
            self._scene.addItem(item)
            self.bubble_items.append(item)

    def set_show_bubbles(self, show: bool):
        """Toggles speech bubble overlay bounding box visibility."""
        self.show_bubbles = show
        for item in self.bubble_items:
            item.setVisible(show)

    def set_data(
        self,
        original_cv: Optional[np.ndarray],
        translated_cv: Optional[np.ndarray] = None,
        erased_cv: Optional[np.ndarray] = None,
        blocks: Optional[List[Any]] = None
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

    def update_translated_image(self, translated_cv: Optional[np.ndarray], erased_cv: Optional[np.ndarray] = None):
        """Updates translated pixmap in scene background without destroying or recreating interactive bubble items."""
        self.translated_cv = translated_cv
        if erased_cv is not None:
            self.erased_cv = erased_cv
        orig_pix = cvimg_to_qpixmap(self.original_cv)
        trans_pix = cvimg_to_qpixmap(self.translated_cv)
        eras_pix = cvimg_to_qpixmap(self.erased_cv)
        self._scene.set_images(orig_pix, trans_pix, eras_pix)

    def select_bubble_by_id(self, block_id: str):
        """Selects and focuses the BubbleItem corresponding to block_id on canvas."""
        for item in self.bubble_items:
            bid = str(item.block_data.get("id", ""))
            if bid == str(block_id):
                item.setSelected(True)
                item.update()
            else:
                item.setSelected(False)
                item.update()

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
    def _update_zoom_state(self, factor: float):
        self.zoom_factor = factor
        self.hud.zoom_lbl.setText(f"{int(factor * 100)}%")
        self.sig_zoom_changed.emit(self.zoom_factor)

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
            new_zoom = 0.05
        elif new_zoom > 10.0:
            factor = 10.0 / self.zoom_factor
            new_zoom = 10.0

        self.scale(factor, factor)
        self._update_zoom_state(new_zoom)
        event.accept()

    def zoom_in(self):
        """Steps zoom in by 1.25x."""
        factor = 1.25
        if self.zoom_factor * factor <= 10.0:
            self.scale(factor, factor)
            self._update_zoom_state(self.zoom_factor * factor)

    def zoom_out(self):
        """Steps zoom out by 0.8x."""
        factor = 0.8
        if self.zoom_factor * factor >= 0.05:
            self.scale(factor, factor)
            self._update_zoom_state(self.zoom_factor * factor)

    def reset_zoom(self):
        """Resets zoom level to 1:1 (100%)."""
        self.resetTransform()
        self._update_zoom_state(1.0)

    def fit_in_view(self):
        """Fits active scene contents perfectly within the viewport window."""
        rect = self._scene.current_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

        # Update zoom factor based on current transform matrix
        transform = self.transform()
        self._update_zoom_state(float(transform.m11()))

    # -------------------------------------------------------------------------
    # Pan & Tool Engine
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
                if self.tool_mode == "draw":
                    self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.unsetCursor()
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        # Draw tool active: start drawing rubber band rectangle
        if self.tool_mode == "draw" and event.button() == Qt.MouseButton.LeftButton:
            self._is_drawing_rect = True
            self._draw_start_scene_pt = self.mapToScene(event.pos())
            if self._rubber_band_item is None:
                self._rubber_band_item = QGraphicsRectItem()
                self._rubber_band_item.setPen(QPen(QColor("#2563EB"), 2, Qt.PenStyle.DashLine))
                self._rubber_band_item.setBrush(QBrush(QColor(37, 99, 235, 45)))
                self._scene.addItem(self._rubber_band_item)
            self._rubber_band_item.setRect(QRectF(self._draw_start_scene_pt, self._draw_start_scene_pt))
            self._rubber_band_item.show()
            event.accept()
            return

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
        if self._is_drawing_rect and self._rubber_band_item and self._draw_start_scene_pt:
            curr_pt = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start_scene_pt, curr_pt).normalized()
            self._rubber_band_item.setRect(rect)
            event.accept()
            return

        if self.is_panning:
            delta = event.position() - self.pan_start_pos
            self.pan_start_pos = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._is_drawing_rect:
            self._is_drawing_rect = False
            if self._rubber_band_item and self.original_cv is not None:
                rect = self._rubber_band_item.rect().normalized()
                self._scene.removeItem(self._rubber_band_item)
                self._rubber_band_item = None

                if rect.width() >= 12 and rect.height() >= 12:
                    img_h, img_w = self.original_cv.shape[:2]
                    xmin = round(max(0.0, min(100.0, (rect.left() / img_w) * 100.0)), 2)
                    ymin = round(max(0.0, min(100.0, (rect.top() / img_h) * 100.0)), 2)
                    xmax = round(max(0.0, min(100.0, (rect.right() / img_w) * 100.0)), 2)
                    ymax = round(max(0.0, min(100.0, (rect.bottom() / img_h) * 100.0)), 2)
                    new_block = {
                        "id": f"m_{uuid.uuid4().hex[:4]}",
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "original_text": "",
                        "translated_text": "",
                        "type": "bubble",
                        "confidence": 1.0,
                    }
                    self.sig_bubble_created.emit(new_block)
            elif self._rubber_band_item:
                self._scene.removeItem(self._rubber_band_item)
                self._rubber_band_item = None

            self.set_tool_mode("select")
            event.accept()
            return

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
