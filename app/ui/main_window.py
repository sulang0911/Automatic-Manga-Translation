"""
app/ui/main_window.py
Apple HIG Desktop Application Shell.
Integrates Sidebar Navigation Rail, Chapter Queue, Canvas Viewport, and Action Toolbar.
"""
import os
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QProgressBar,
    QButtonGroup, QFrame, QSlider, QCheckBox, QToolButton, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut, QColor, QDragEnterEvent, QDropEvent

from app.core.config import AppConfig
from app.core.models import TranslationBlock
from app.core.typography.engine import TypographyEngine
from app.core.pipeline.pipeline_worker import PipelineWorker
from app.core.pipeline.batch_worker import BatchWorker
from app.core.pipeline.exporter import MangaExporter
from app.core.cache.cache_manager import get_cache_manager, safe_cv2_imread, safe_cv2_imwrite
from app.ui.theme.tokens import get_tokens, build_stylesheet
from app.ui.theme.icons import get_icon
from app.ui.widgets.card import CardWidget
from app.ui.widgets.segmented_control import SegmentedControl
from app.ui.widgets.progress_pill import ProgressPill
from app.ui.widgets.toast import Toast
from app.ui.canvas.view import MangaCanvasView
from app.ui.sidebar.nav_rail import NavRail
from app.ui.sidebar.page_list import PageListWidget
from app.ui.sidebar.drop_zone import DropZoneWidget
from app.ui.inspector.inspector_panel import InspectorPanel
from app.ui.settings.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """
    Apple Human Interface Guidelines Desktop Application Window.
    Provides responsive 60 FPS workspace for manga OCR, inpainting, AI translation,
    and side-by-side or split-slider visual inspection.
    """

    def __init__(self, theme: str = "dark", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("🌌 AetherLens — 漫画智能翻译工作台 (PyQt6)")
        self.resize(1380, 880)
        self.setMinimumSize(1000, 650)
        self.setAcceptDrops(True)

        self._current_theme = theme
        self.setStyleSheet(build_stylesheet(get_tokens(theme)))

        self.config = AppConfig.load("desktop_config.json")
        self.typo_engine = TypographyEngine()
        self.current_image_data: Optional[Dict[str, Any]] = None
        self.active_worker: Optional[PipelineWorker] = None
        self.active_batch_worker: Optional[BatchWorker] = None

        # 300ms Debounce Timer for live editing re-renders
        self._rerender_timer = QTimer(self)
        self._rerender_timer.setSingleShot(True)
        self._rerender_timer.setInterval(300)
        self._rerender_timer.timeout.connect(self._re_render_current_page)

        self._init_ui()
        self._init_shortcuts()
        self.toast = Toast(self)

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Canvas Viewport (pre-instantiated for toolbar signal binding)
        self.canvas_view = MangaCanvasView(self)
        self.canvas_view.sig_zoom_changed.connect(self._on_zoom_changed)
        self.canvas_view.sig_split_changed.connect(self._on_split_slider_moved_from_canvas)
        self.canvas_view.sig_bubble_selected.connect(self._on_bubble_selected_from_canvas)
        self.canvas_view.sig_bubble_changed.connect(self._on_bubble_geometry_changed)
        self.canvas_view.sig_bubble_swap_prev.connect(self._on_canvas_bubble_swap_prev)
        self.canvas_view.sig_bubble_swap_next.connect(self._on_canvas_bubble_swap_next)

        # 2. Action Toolbar
        self.toolbar_widget = self._create_toolbar()
        main_layout.addWidget(self.toolbar_widget)

        # 3. Workspace Splitter [Sidebar | Canvas | Inspector]
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setHandleWidth(1)

        # 3A. Left: Sidebar (NavRail + PageList + DropZone)
        self.queue_panel = self._create_sidebar()
        self.splitter.addWidget(self.queue_panel)

        # 3B. Center: Canvas + Split Slider Bar
        center_container = QWidget(self)
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.canvas_view)

        # Bottom Split Slider (visible when in split_slider mode)
        self.slider_bar = QFrame(self)
        self.slider_bar.setObjectName("sliderBar")
        self.slider_bar.setFixedHeight(36)
        slider_layout = QHBoxLayout(self.slider_bar)
        slider_layout.setContentsMargins(16, 4, 16, 4)

        orig_label = QLabel("原图", self.slider_bar)
        orig_label.setStyleSheet("font-size: 11px; font-weight: 500;")
        slider_layout.addWidget(orig_label)

        self.split_slider = QSlider(Qt.Orientation.Horizontal, self.slider_bar)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(self._on_split_slider_changed)
        slider_layout.addWidget(self.split_slider)

        trans_label = QLabel("译图", self.slider_bar)
        trans_label.setStyleSheet("font-size: 11px; font-weight: 600;")
        slider_layout.addWidget(trans_label)

        self.slider_bar.hide()
        center_layout.addWidget(self.slider_bar)

        self.splitter.addWidget(center_container)

        # 3C. Right: Inspector Panel Card
        self.inspector_panel = InspectorPanel(config=self.config, parent=self)
        self.inspector_panel.sig_re_render_requested.connect(self._re_render_current_page)
        self.inspector_panel.sig_block_updated.connect(self._on_block_updated_from_inspector)
        self.inspector_panel.sig_block_deleted.connect(self._on_block_deleted)
        self.inspector_panel.sig_translate_page_requested.connect(lambda: self._start_pipeline_for_page(mode="translate_only"))
        self.inspector_panel.sig_erase_page_requested.connect(lambda: self._start_pipeline_for_page(mode="inpaint_only"))
        self.inspector_panel.sig_export_page_requested.connect(self._export_current_page)
        self.inspector_panel.sig_open_export_dir_requested.connect(self._open_export_directory)
        self.splitter.addWidget(self.inspector_panel)

        # Set balanced initial proportions [sidebar, canvas, inspector]
        self.splitter.setSizes([260, 840, 280])
        main_layout.addWidget(self.splitter, 1)

        # 4. Status Bar
        self.status_bar_widget = self._create_status_bar()
        main_layout.addWidget(self.status_bar_widget)

    def _create_sidebar(self) -> QWidget:
        """Constructs the sidebar combining NavRail, Chapter PageList, and DropZone."""
        container = QWidget(self)
        container.setProperty("class", "sidebar")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Navigation Rail
        self.nav_rail = NavRail(container)
        layout.addWidget(self.nav_rail)

        # Drawer container
        self.sidebar_drawer = QWidget(container)
        drawer_layout = QVBoxLayout(self.sidebar_drawer)
        drawer_layout.setContentsMargins(8, 8, 8, 8)
        drawer_layout.setSpacing(8)

        # Drop Zone
        self.drop_zone = DropZoneWidget(self.sidebar_drawer)
        self.drop_zone.sig_paths_dropped.connect(self._on_paths_dropped)
        drawer_layout.addWidget(self.drop_zone)

        # Page List
        self.page_list = PageListWidget(self.sidebar_drawer)
        self.page_list.sig_page_selected.connect(self._on_page_selected)
        self.page_list.sig_start_batch.connect(self._start_batch)
        self.page_list.sig_clear_requested.connect(self._on_page_list_cleared)
        self.page_list.sig_translate_page.connect(self._on_translate_page_from_list)
        self.page_list.sig_export_page.connect(self._on_export_page_from_list)
        drawer_layout.addWidget(self.page_list, 1)

        layout.addWidget(self.sidebar_drawer, 1)

        # Nav rail bindings
        self.nav_rail.sig_toggle_sidebar.connect(self.toggle_sidebar)
        self.nav_rail.sig_nav_changed.connect(self._on_nav_rail_changed)

        # Mirror compatibility properties onto queue_panel
        container.list_widget = self.page_list.list_widget
        container.count_badge = self.page_list.count_badge
        container.items_data = self.page_list.items_data
        container.add_paths = self.page_list.add_paths
        container.clear_all = self.page_list.clear_all
        container.update_item_status = self.page_list.update_item_status
        container.sig_image_selected = self.page_list.sig_page_selected

        return container

    def _create_toolbar(self) -> QWidget:
        """Constructs the top action toolbar conforming to Apple HIG."""
        toolbar = QFrame(self)
        toolbar.setProperty("class", "toolbar")
        toolbar.setFixedHeight(46)

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(8)

        # View Mode Segmented Control
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        mode_specs = [
            ("translated", "译图", "sparkles"),
            ("split_slider", "对比", "split"),
            ("side_by_side", "双联", "columns"),
            ("original", "原图", "eye"),
            ("inpainted", "抹字", "eraser"),
        ]

        self._mode_buttons: Dict[str, QToolButton] = {}
        for idx, (mode_key, label, icon_name) in enumerate(mode_specs):
            btn = QToolButton(toolbar)
            btn.setProperty("class", "segment-btn")
            btn.setText(label)
            btn.setIcon(get_icon(icon_name, color="#A1A1AA", active_color="#FFFFFF", size=16))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

            if idx == 0:
                btn.setChecked(True)

            self.btn_group.addButton(btn, idx)
            self._mode_buttons[mode_key] = btn
            layout.addWidget(btn)

        self.btn_group.idClicked.connect(self._on_segment_button_clicked)

        layout.addSpacing(16)

        # Bubble overlay toggle checkbox
        self.bubble_cb = QCheckBox("显示对话框", toolbar)
        self.bubble_cb.setChecked(True)
        self.bubble_cb.toggled.connect(self._on_bubble_cb_toggled)
        layout.addWidget(self.bubble_cb)

        layout.addSpacing(16)

        # Source Language Selector (Default: 自动识别)
        lang_lbl = QLabel("源语言:", toolbar)
        lang_lbl.setStyleSheet("font-size: 11px; font-weight: 500;")
        layout.addWidget(lang_lbl)

        self.source_lang_combo = QComboBox(toolbar)
        self.source_lang_combo.setFixedWidth(100)
        self.source_lang_combo.addItems(["自动识别", "日语", "韩语", "英语", "繁体中文", "简体中文"])
        self.source_lang_combo.setCurrentText(getattr(self.config, "source_lang", "自动识别"))
        self.source_lang_combo.currentTextChanged.connect(self._on_source_lang_changed)
        layout.addWidget(self.source_lang_combo)

        layout.addStretch()

        # Canvas Zoom Controls
        self.fit_btn = QToolButton(toolbar)
        self.fit_btn.setIcon(get_icon("fit_window", color="#A1A1AA", size=16))
        self.fit_btn.setToolTip("适应窗口 (Fit in View)")
        self.fit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fit_btn.clicked.connect(self.canvas_view.fit_in_view)
        layout.addWidget(self.fit_btn)

        self.actual_size_btn = QToolButton(toolbar)
        self.actual_size_btn.setIcon(get_icon("actual_size", color="#A1A1AA", size=16))
        self.actual_size_btn.setToolTip("实际大小 1:1")
        self.actual_size_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.actual_size_btn.clicked.connect(self.canvas_view.reset_zoom)
        layout.addWidget(self.actual_size_btn)

        self.zoom_out_btn = QToolButton(toolbar)
        self.zoom_out_btn.setIcon(get_icon("zoom_out", color="#A1A1AA", size=16))
        self.zoom_out_btn.setToolTip("缩小 (Ctrl -)")
        self.zoom_out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zoom_out_btn.clicked.connect(self.canvas_view.zoom_out)
        layout.addWidget(self.zoom_out_btn)

        self.zoom_in_btn = QToolButton(toolbar)
        self.zoom_in_btn.setIcon(get_icon("zoom_in", color="#A1A1AA", size=16))
        self.zoom_in_btn.setToolTip("放大 (Ctrl +)")
        self.zoom_in_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.zoom_in_btn.clicked.connect(self.canvas_view.zoom_in)
        layout.addWidget(self.zoom_in_btn)

        layout.addSpacing(12)

        # Theme Switcher Button
        self.theme_btn = QToolButton(toolbar)
        self.theme_btn.setIcon(get_icon("sun" if self._current_theme == "dark" else "moon", color="#A1A1AA", size=16))
        self.theme_btn.setToolTip("切换明暗主题")
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_btn)

        # Settings Button
        self.settings_btn = QPushButton("设置", toolbar)
        self.settings_btn.setIcon(get_icon("settings", color="#A1A1AA", size=16))
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        layout.addWidget(self.settings_btn)

        # Export Button
        self.export_btn = QPushButton("导出", toolbar)
        self.export_btn.setIcon(get_icon("download", color="#A1A1AA", size=16))
        self.export_btn.setToolTip("导出当前已翻译页面 (PNG / JPG / WebP / PDF)")
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.clicked.connect(self._export_current_page)
        layout.addWidget(self.export_btn)

        # Batch Translate Button
        self.batch_toolbar_btn = QPushButton("批量翻译", toolbar)
        self.batch_toolbar_btn.setIcon(get_icon("play_all", color="#3B82F6", size=16))
        self.batch_toolbar_btn.setToolTip("批量翻译页面列表中的全部漫画图片")
        self.batch_toolbar_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.batch_toolbar_btn.clicked.connect(self._start_batch)
        layout.addWidget(self.batch_toolbar_btn)

        # Primary Action Button: Run Translation
        self.run_btn = QPushButton("开始翻译", toolbar)
        self.run_btn.setProperty("class", "primaryBtn")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setIcon(get_icon("play", color="#FFFFFF", size=16))
        self.run_btn.setToolTip("翻译当前选中的漫画单页")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)

        return toolbar

    def _create_status_bar(self) -> QWidget:
        """Constructs the bottom status bar widget."""
        status_bar = QFrame(self)
        status_bar.setObjectName("statusBar")
        status_bar.setFixedHeight(28)

        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(12, 2, 12, 2)
        layout.setSpacing(12)

        self.status_label = QLabel("就绪", status_bar)
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.progress_bar = QProgressBar(status_bar)
        self.progress_bar.setFixedSize(140, 8)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        return status_bar

    def _init_shortcuts(self):
        """Keyboard accelerators."""
        QShortcut(QKeySequence("Ctrl+="), self, self.canvas_view.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self.canvas_view.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.canvas_view.reset_zoom)
        QShortcut(QKeySequence("Ctrl+F"), self, self.canvas_view.fit_in_view)
        QShortcut(QKeySequence("Ctrl+B"), self, self.toggle_sidebar)

    # -------------------------------------------------------------------------
    # Event Handlers & View Synchronization
    # -------------------------------------------------------------------------
    def _on_segment_button_clicked(self, btn_id: int):
        id_to_mode = {
            0: "translated",
            1: "split_slider",
            2: "side_by_side",
            3: "original",
            4: "inpainted",
        }
        mode = id_to_mode.get(btn_id, "translated")
        self._on_view_mode_changed(mode)

    def _on_view_mode_changed(self, mode: str):
        """Switches view mode and manages split-slider bar visibility."""
        self.canvas_view.set_view_mode(mode)
        if mode == "split_slider":
            self.slider_bar.show()
        else:
            self.slider_bar.hide()

    def _on_split_slider_changed(self, value: int):
        ratio = value / 100.0
        self.canvas_view.set_split_position(ratio)

    def _on_split_slider_moved_from_canvas(self, ratio: float):
        self.split_slider.blockSignals(True)
        self.split_slider.setValue(int(ratio * 100))
        self.split_slider.blockSignals(False)

    def _on_bubble_cb_toggled(self, checked: bool):
        self.canvas_view.set_show_bubbles(checked)

    def _on_zoom_changed(self, zoom_factor: float):
        pct = int(zoom_factor * 100)
        self.status_label.setText(f"就绪 | 缩放 {pct}%")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        paths: List[str] = []
        for url in urls:
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        if paths:
            self._on_paths_dropped(paths)
            event.acceptProposedAction()

    def _on_paths_dropped(self, paths: List[str]):
        initial_count = len(self.page_list.items_data)
        self.page_list.add_paths(paths)
        new_count = len(self.page_list.items_data) - initial_count
        if len(self.page_list.items_data) > 0:
            self.drop_zone.set_compact(True)
        if new_count > 0:
            self.toast.show_message(f"已成功载入 {new_count} 个漫画页面！", "success")
        elif paths:
            self.toast.show_message("所选路径已全部在列表中或未检测到支持的漫画图片。", "info")

    def _on_page_selected(self, item_data: Dict[str, Any]):
        path = item_data.get("path")
        if path and os.path.exists(path):
            self.current_image_data = item_data
            cv_img = safe_cv2_imread(path)
            if cv_img is not None:
                cache_mgr = get_cache_manager()
                cached = cache_mgr.load_page_cache(path, load_images=True)

                cached_blocks = cached.get("blocks")
                blocks = cached_blocks if cached_blocks else item_data.get("blocks", [])

                cached_erased = cached.get("erased_img")
                erased = cached_erased if cached_erased is not None else item_data.get("erased_img")

                cached_rendered = cached.get("rendered_img")
                translated = cached_rendered if cached_rendered is not None else item_data.get("translated_img")

                # If blocks exist and erased exists but rendered image not on disk yet, render on demand
                if translated is None and blocks and (erased is not None or cv_img is not None):
                    base_bg = erased if erased is not None else cv_img
                    model_blocks = [
                        b if isinstance(b, TranslationBlock) else TranslationBlock.from_dict(b)
                        for b in blocks
                    ]
                    try:
                        translated = self.typo_engine.render_page(base_bg, model_blocks, self.config.style)
                        cache_mgr.save_page_cache(path, rendered_img=translated)
                    except Exception as e:
                        print(f"[-] Auto render on select error: {e}")

                self.current_image_data["blocks"] = blocks
                self.current_image_data["erased_img"] = erased
                self.current_image_data["translated_img"] = translated

                self.canvas_view.set_data(cv_img, translated_cv=translated, erased_cv=erased, blocks=blocks)
                self.canvas_view.fit_in_view()
                self.inspector_panel.set_blocks(blocks)

    def _on_bubble_selected_from_canvas(self, block_data: Dict[str, Any]):
        b_id = block_data.get("id")
        if b_id:
            self.inspector_panel.select_block_by_id(b_id)

    def _on_bubble_geometry_changed(self, block_data: Dict[str, Any]):
        self._schedule_rerender()

    def _on_canvas_bubble_swap_prev(self, block_id: str):
        self.inspector_panel.select_block_by_id(block_id)
        self.inspector_panel._on_swap_prev_clicked()

    def _on_canvas_bubble_swap_next(self, block_id: str):
        self.inspector_panel.select_block_by_id(block_id)
        self.inspector_panel._on_swap_next_clicked()

    def _on_block_updated_from_inspector(self, block_data: Dict[str, Any]):
        if not self.current_image_data:
            return
        target_id = str(block_data.get("id"))
        blocks = self.current_image_data.get("blocks", [])
        for idx, b in enumerate(blocks):
            bid = str(b.get("id") if isinstance(b, dict) else getattr(b, "id", None))
            if bid == target_id:
                if isinstance(b, dict):
                    b.update(block_data)
                else:
                    for k, v in block_data.items():
                        if hasattr(b, k):
                            setattr(b, k, v)
                break
        self._schedule_rerender()

    def _on_block_deleted(self, block_id: str):
        if self.current_image_data and "blocks" in self.current_image_data:
            blocks = self.current_image_data["blocks"]
            self.current_image_data["blocks"] = [
                b for b in blocks
                if (b.get("id") if isinstance(b, dict) else getattr(b, "id", None)) != block_id
            ]
            self.inspector_panel.set_blocks(self.current_image_data["blocks"])
            self.canvas_view.blocks = self.current_image_data["blocks"]
            self.canvas_view._rebuild_bubbles()
            self._re_render_current_page()

    def _schedule_rerender(self):
        """Starts 300ms debounce timer to prevent lag during rapid slider/text input."""
        self._rerender_timer.start(300)

    def _re_render_current_page(self):
        """Performs live typography re-render onto canvas using current style configuration."""
        if not self.current_image_data:
            return
        blocks = self.current_image_data.get("blocks", [])
        if not blocks:
            return

        base_img = self.current_image_data.get("erased_img")
        if base_img is None:
            path = self.current_image_data.get("path")
            if path:
                cached = get_cache_manager().load_page_cache(path, load_images=True)
                base_img = cached.get("erased_img")
                if base_img is not None:
                    self.current_image_data["erased_img"] = base_img

        if base_img is None:
            base_img = self.canvas_view.original_cv
        if base_img is None:
            return

        # Convert dict blocks to TranslationBlock objects if needed
        model_blocks = [
            b if isinstance(b, TranslationBlock) else TranslationBlock.from_dict(b)
            for b in blocks
        ]

        try:
            rendered = self.typo_engine.render_page(base_img, model_blocks, self.config.style)
            self.current_image_data["translated_img"] = rendered
            self.canvas_view.translated_cv = rendered
            self.canvas_view.set_data(
                original_cv=self.canvas_view.original_cv,
                translated_cv=rendered,
                erased_cv=base_img,
                blocks=blocks
            )
            # Ensure view mode displays translated artwork
            if self.canvas_view.view_mode in ("original", "inpainted"):
                self.canvas_view.set_view_mode("translated")
                if "translated" in self._mode_buttons:
                    self._mode_buttons["translated"].setChecked(True)

            path = self.current_image_data.get("path")
            if path:
                get_cache_manager().save_page_cache(path, blocks=model_blocks, rendered_img=rendered)
            self.status_label.setText("排版重绘完成并已自动保存")
        except Exception as e:
            self.status_label.setText(f"重排版失败: {e}")

    def toggle_sidebar(self):
        """Collapses or expands the sidebar drawer."""
        is_visible = self.sidebar_drawer.isVisible()
        self.sidebar_drawer.setVisible(not is_visible)
        self.nav_rail.set_collapsed_icon(is_visible)

    def _on_nav_rail_changed(self, key: str):
        """Handles navigation rail section switches."""
        if key == "pages":
            if not self.sidebar_drawer.isVisible():
                self.sidebar_drawer.show()
                self.nav_rail.set_collapsed_icon(False)
            self.page_list.setFocus()
        elif key == "inspector":
            is_vis = self.inspector_panel.isVisible()
            self.inspector_panel.setVisible(not is_vis)
            self.status_label.setText("已展开对话框检查器" if not is_vis else "已隐藏对话框检查器（全屏画布模式）")
        elif key == "settings":
            self._open_settings_dialog()

    def _on_translate_page_from_list(self, item_data: Dict[str, Any]):
        """Translates the specific page requested from the context menu."""
        self._on_page_selected(item_data)
        self._start_pipeline_for_page(mode="full")

    def _on_export_page_from_list(self, item_data: Dict[str, Any]):
        """Exports the specific page requested from the context menu."""
        self._on_page_selected(item_data)
        self._export_current_page()

    def _open_export_directory(self):
        """Opens the exported chapter directory in OS file manager."""
        export_dir = os.path.join(os.getcwd(), "exported_chapter")
        os.makedirs(export_dir, exist_ok=True)
        import subprocess
        subprocess.Popen(f'explorer "{os.path.normpath(export_dir)}"')

    def _on_page_list_cleared(self):
        """Resets drop zone and canvas view when all items are cleared."""
        self.drop_zone.set_compact(False)
        self.current_image_data = None
        self.canvas_view.set_data(None)
        self.inspector_panel.set_blocks([])
        self.status_label.setText("就绪 | 页面队列已清空")

    def _on_source_lang_changed(self, lang: str):
        """Updates active source language in application configuration."""
        self.config.source_lang = lang
        self.status_label.setText(f"翻译源语言已切换为: {lang}")

    def toggle_theme(self):
        """Toggles between dark and light themes."""
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self.set_theme(new_theme)

    def set_theme(self, theme_name: str):
        """Applies theme stylesheet and updates theme button icon."""
        self._current_theme = theme_name
        tokens = get_tokens(theme_name)
        self.setStyleSheet(build_stylesheet(tokens))
        self.theme_btn.setIcon(get_icon("sun" if theme_name == "dark" else "moon", color=tokens.text_secondary, size=16))
        self.settings_btn.setIcon(get_icon("settings", color=tokens.text_secondary, size=16))
        self.export_btn.setIcon(get_icon("download", color=tokens.text_secondary, size=16))
        self.batch_toolbar_btn.setIcon(get_icon("play_all", color=tokens.accent_primary, size=16))
        self.canvas_view.setBackgroundBrush(QColor(tokens.canvas_bg))
        self.canvas_view.scene.setBackgroundBrush(QColor(tokens.canvas_bg))

    # -------------------------------------------------------------------------
    # M4: Pipeline & Batch Asynchronous Execution & Export
    # -------------------------------------------------------------------------
    def _open_settings_dialog(self):
        """Opens Apple HIG system settings preference modal."""
        dialog = SettingsDialog(config=self.config, parent=self)
        if dialog.exec():
            self.toast.show_message("设置已成功保存并生效！", "success")

    def _on_run_clicked(self):
        """Starts single page translation or cancels active worker."""
        if self.active_worker and self.active_worker.isRunning():
            self.active_worker.cancel()
            self.status_label.setText("正在取消任务...")
            self.run_btn.setText("开始翻译")
            self.run_btn.setIcon(get_icon("play", color="#FFFFFF", size=16))
            return

        if self.current_image_data and "path" in self.current_image_data:
            self._start_pipeline_for_page(mode="full")
        elif self.page_list.items_data:
            self._start_batch()
        else:
            self.toast.show_message("请先拖入漫画图片或选择页面！", "warning")

    def _start_pipeline_for_page(self, mode: str = "full"):
        """Launches PipelineWorker QThread for active page."""
        if not self.current_image_data:
            self.toast.show_message("请先选择或拖入一张漫画图片！", "warning")
            return

        path = self.current_image_data.get("path")
        if not path or not os.path.exists(path):
            self.toast.show_message(f"图片路径不存在: {path}", "error")
            return

        self.run_btn.setText("取消任务")
        self.run_btn.setIcon(get_icon("trash", color="#FFFFFF", size=16))
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        blocks = self.current_image_data.get("blocks")
        erased = self.current_image_data.get("erased_img")

        self.active_worker = PipelineWorker(
            image_path=path,
            config=self.config.to_dict(),
            existing_blocks=blocks,
            existing_erased=erased,
            mode=mode,
            parent=self
        )
        self.active_worker.sig_progress.connect(self._on_pipeline_progress)
        self.active_worker.sig_step_done.connect(self._on_pipeline_step_done)
        self.active_worker.sig_finished.connect(self._on_pipeline_finished)
        self.active_worker.sig_error.connect(self._on_pipeline_error)
        self.active_worker.start()

    def _on_pipeline_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"{msg} ({pct}%)")

    def _on_pipeline_step_done(self, step_name: str, result_data: Any):
        if step_name == "ocr":
            if self.current_image_data is not None:
                self.current_image_data["blocks"] = result_data
            self.inspector_panel.set_blocks(result_data)
        elif step_name == "inpaint":
            if self.current_image_data is not None:
                self.current_image_data["erased_img"] = result_data
            self.canvas_view.erased_cv = result_data
        elif step_name == "translate":
            if self.current_image_data is not None:
                self.current_image_data["blocks"] = result_data
            self.inspector_panel.set_blocks(result_data)
        elif step_name == "render":
            if self.current_image_data is not None:
                self.current_image_data["translated_img"] = result_data
            self.canvas_view.translated_cv = result_data

    def _on_pipeline_finished(self, result: Dict[str, Any]):
        self.run_btn.setText("开始翻译")
        self.run_btn.setIcon(get_icon("play", color="#FFFFFF", size=16))
        self.progress_bar.hide()
        self.status_label.setText("就绪 | 翻译处理完成")

        if self.current_image_data is not None:
            self.current_image_data.update(result)
            self.canvas_view.set_data(
                original_cv=result.get("original_img"),
                translated_cv=result.get("translated_img"),
                erased_cv=result.get("erased_img"),
                blocks=result.get("blocks", [])
            )
            self.canvas_view.set_view_mode("translated")
            self.inspector_panel.set_blocks(result.get("blocks", []))

        self.toast.show_message("漫画翻译已成功完成！", "success")

    def _on_pipeline_error(self, err_msg: str):
        self.run_btn.setText("开始翻译")
        self.run_btn.setIcon(get_icon("play", color="#FFFFFF", size=16))
        self.progress_bar.hide()
        self.status_label.setText(f"错误: {err_msg}")
        self.toast.show_message(err_msg, "error")

    def _start_batch(self, queue_items: Optional[List[Dict[str, Any]]] = None):
        """Launches BatchWorker QThread for chapter queue."""
        if self.active_batch_worker and self.active_batch_worker.isRunning():
            self.active_batch_worker.cancel()
            self.status_label.setText("正在取消批处理任务...")
            self.batch_toolbar_btn.setText("批量翻译")
            if hasattr(self.page_list, "batch_btn"):
                self.page_list.batch_btn.setText("🚀 批量翻译本章全部页面")
            return

        items = queue_items or self.page_list.items_data
        if not items:
            self.toast.show_message("处理队列为空，请先添加漫画页面！", "warning")
            return

        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.batch_toolbar_btn.setText("取消批处理")
        if hasattr(self.page_list, "batch_btn"):
            self.page_list.batch_btn.setText("⏹ 取消批处理")

        export_dir = os.path.join(os.getcwd(), "exported_chapter")
        os.makedirs(export_dir, exist_ok=True)

        self.active_batch_worker = BatchWorker(
            queue_items=items,
            config=self.config.to_dict(),
            export_dir=export_dir,
            parent=self
        )
        self.active_batch_worker.sig_batch_progress.connect(self._on_batch_progress)
        self.active_batch_worker.sig_item_completed.connect(self._on_batch_item_completed)
        self.active_batch_worker.sig_item_failed.connect(self._on_batch_item_failed)
        self.active_batch_worker.sig_batch_finished.connect(self._on_batch_finished)
        self.active_batch_worker.start()

    def _on_batch_progress(self, cur: int, total: int, filename: str, pct: int, msg: str):
        overall = int(((cur - 1) / total) * 100 + (pct / total))
        self.progress_bar.setValue(overall)
        self.status_label.setText(f"批处理 ({cur}/{total}): {filename} - {msg}")

    def _on_batch_item_completed(self, image_id: str, result: Dict[str, Any]):
        self.page_list.update_item_status(image_id, "completed", "已完成")
        # If this is the currently active page on canvas, reload from disk cache and update view
        if self.current_image_data and self.current_image_data.get("id") == image_id:
            self._on_page_selected(self.current_image_data)

    def _on_batch_item_failed(self, image_id: str, error_msg: str):
        short_err = error_msg.strip().split("\n")[-1][:20]
        self.page_list.update_item_status(image_id, "failed", f"失败: {short_err}")

    def _on_batch_finished(self, success_count: int, fail_count: int):
        self.progress_bar.hide()
        self.batch_toolbar_btn.setText("批量翻译")
        if hasattr(self.page_list, "batch_btn"):
            self.page_list.batch_btn.setText("🚀 批量翻译本章全部页面")
        self.status_label.setText(f"批处理完成: {success_count} 成功, {fail_count} 失败")
        self.toast.show_message(f"批处理完成: {success_count} 成功, {fail_count} 失败", "success" if fail_count == 0 else "warning")

    def _export_current_page(self):
        """Exports currently active translated manga page to high-res PNG/JPG/WebP/PDF."""
        if not self.current_image_data or self.canvas_view.translated_cv is None:
            self.toast.show_message("当前页面尚未翻译，无法导出！", "warning")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出当前已翻译页面",
            "manga_page_translated.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;WebP Image (*.webp);;PDF Document (*.pdf)"
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            # Save temporary image and compile PDF
            import tempfile
            tmp_img = os.path.join(tempfile.gettempdir(), "temp_export_page.png")
            cv2.imwrite(tmp_img, self.canvas_view.translated_cv)
            success = MangaExporter.compile_chapter_pdf([tmp_img], file_path)
        else:
            fmt = "JPEG" if ext in [".jpg", ".jpeg"] else ("WEBP" if ext == ".webp" else "PNG")
            success = MangaExporter.export_single_image(
                self.canvas_view.translated_cv,
                file_path,
                fmt=fmt,
                compressed=self.config.style.export_compressed
            )

        if success:
            self.toast.show_message(f"成功导出页面至: {os.path.basename(file_path)}", "success")
        else:
            self.toast.show_message("页面导出失败，请检查写入权限。", "error")

