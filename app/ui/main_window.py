"""
app/ui/main_window.py
Apple HIG Desktop Application Shell.
Integrates Sidebar Navigation Rail, Chapter Queue, Canvas Viewport, and Action Toolbar.
"""
import os
import copy
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QSplitter, QProgressBar,
    QButtonGroup, QFrame, QSlider, QCheckBox, QToolButton, QFileDialog, QComboBox,
    QApplication, QTextEdit, QPlainTextEdit, QLineEdit, QMenu
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut, QColor, QDragEnterEvent, QDropEvent

from app.core.config import AppConfig
from app.core.models import TranslationBlock, StyleConfig
from app.core.typography.engine import TypographyEngine
from app.core.undo_manager import UndoManager, PageSnapshot, are_blocks_equal
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
from app.ui.canvas.canvas_view import MangaCanvasView
from app.ui.sidebar.nav_rail import NavRail
from app.ui.sidebar.page_list import PageListWidget
from app.ui.sidebar.drop_zone import DropZoneWidget
from app.ui.inspector.inspector_panel import InspectorPanel
from app.ui.settings.settings_dialog import SettingsDialog
from app.ui.settings.page_style_dialog import PageStyleDialog


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
        tokens = get_tokens(theme)
        app_inst = QApplication.instance()
        if app_inst:
            app_inst.setStyleSheet(build_stylesheet(tokens))
        self.setStyleSheet(build_stylesheet(tokens))

        self.config = AppConfig.load("desktop_config.json")
        self.typo_engine = TypographyEngine()
        self.undo_manager = UndoManager(max_depth=50)
        self._pending_drag_snapshot: Optional[PageSnapshot] = None
        self._pending_inspector_snapshot: Optional[PageSnapshot] = None
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
        self.canvas_view.sig_bubble_changed.connect(self._on_bubble_moving)
        self.canvas_view.sig_bubble_geometry_start.connect(self._on_bubble_geometry_start)
        self.canvas_view.sig_bubble_commit.connect(self._on_bubble_geometry_changed)
        self.canvas_view.sig_bubble_swap_prev.connect(self._on_canvas_bubble_swap_prev)
        self.canvas_view.sig_bubble_swap_next.connect(self._on_canvas_bubble_swap_next)
        self.canvas_view.sig_bubble_merge_prev.connect(self._on_canvas_bubble_merge_prev)
        self.canvas_view.sig_bubble_merge_next.connect(self._on_canvas_bubble_merge_next)
        self.canvas_view.sig_bubble_delete.connect(self._on_block_deleted)
        self.canvas_view.sig_bubble_created.connect(self._on_bubble_created)
        self.canvas_view.sig_bubble_ocr_requested.connect(self._on_bubble_ocr_requested)
        self.canvas_view.sig_clear_cache_requested.connect(self._on_canvas_clear_cache_requested)
        self.canvas_view.sig_retranslate_requested.connect(lambda: self._start_pipeline_for_page(mode="full"))
        self.canvas_view.sig_open_style_requested.connect(self._open_current_page_style_dialog)
        self.canvas_view.sig_undo_requested.connect(self._undo)
        self.canvas_view.sig_redo_requested.connect(self._redo)

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
        self.inspector_panel.sig_add_bubble_requested.connect(self.canvas_view.toggle_draw_tool)
        self.inspector_panel.sig_block_selected.connect(self.canvas_view.select_bubble_by_id)
        self.inspector_panel.sig_blocks_reordered.connect(self._on_blocks_reordered)
        self.inspector_panel.sig_ocr_translate_block_requested.connect(self._on_inspector_ocr_translate_block)
        self.splitter.addWidget(self.inspector_panel)

        # Set balanced initial proportions [sidebar, canvas, inspector]
        self.splitter.setSizes([260, 840, 280])
        self._saved_sidebar_width = 260
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
        self.page_list.sig_start_batch.connect(lambda: self._start_batch(force_retranslate=False))
        self.page_list.sig_start_retranslate_all.connect(lambda: self._start_batch(force_retranslate=True))
        self.page_list.sig_clear_requested.connect(self._on_page_list_cleared)
        self.page_list.sig_translate_page.connect(self._on_translate_page_from_list)
        self.page_list.sig_export_page.connect(self._on_export_page_from_list)
        self.page_list.sig_export_all.connect(self._export_all_pages)
        self.page_list.sig_edit_page_style.connect(self._open_page_style_dialog)
        self.page_list.sig_cache_cleared.connect(self._on_page_cache_cleared)
        self.page_list.sig_count_changed.connect(lambda count: self.drop_zone.set_compact(count > 0))
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
        """Constructs the top action command bar conforming to Apple HIG & Pro Dev layout."""
        toolbar = QFrame(self)
        toolbar.setProperty("class", "toolbar")
        toolbar.setFixedHeight(42)

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(10, 3, 10, 3)
        layout.setSpacing(6)

        # 1. Left Cluster: Sidebar Toggle & Breadcrumb
        self.sidebar_toggle_btn = QToolButton(toolbar)
        self.sidebar_toggle_btn.setProperty("class", "icon-action-btn")
        self.sidebar_toggle_btn.setIcon(get_icon("menu", color="#A1A1AA", size=15))
        self.sidebar_toggle_btn.setToolTip("切换侧边栏展开/折叠 (Ctrl+B)")
        self.sidebar_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sidebar_toggle_btn.clicked.connect(lambda: self.toggle_sidebar())
        layout.addWidget(self.sidebar_toggle_btn)

        self.breadcrumb_label = QLabel("📂 工作区", toolbar)
        self.breadcrumb_label.setStyleSheet(
            "font-family: 'JetBrains Mono', 'SF Mono', monospace;"
            "font-size: 11px; font-weight: 600; color: #ECECEF; padding: 3px 8px;"
            "background: rgba(255, 255, 255, 0.05); border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.08);"
        )
        self.breadcrumb_label.setMaximumWidth(200)
        layout.addWidget(self.breadcrumb_label)

        layout.addSpacing(6)

        # 2. Center Cluster: Segmented View Modes Control
        mode_container = QFrame(toolbar)
        mode_container.setObjectName("segmentedControl")
        mode_layout = QHBoxLayout(mode_container)
        mode_layout.setContentsMargins(2, 2, 2, 2)
        mode_layout.setSpacing(2)

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
            btn = QToolButton(mode_container)
            btn.setProperty("class", "segment-btn")
            btn.setText(label)
            btn.setIcon(get_icon(icon_name, color="#A1A1AA", active_color="#FFFFFF", size=13))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

            if idx == 0:
                btn.setChecked(True)

            self.btn_group.addButton(btn, idx)
            self._mode_buttons[mode_key] = btn
            mode_layout.addWidget(btn)

        self.btn_group.idClicked.connect(self._on_segment_button_clicked)
        layout.addWidget(mode_container)

        # Bubble overlay toggle checkbox
        self.bubble_cb = QCheckBox("气泡框", toolbar)
        self.bubble_cb.setChecked(True)
        self.bubble_cb.setToolTip("显示/隐藏画布上的气泡框覆盖层")
        self.bubble_cb.toggled.connect(self._on_bubble_cb_toggled)
        layout.addWidget(self.bubble_cb)

        layout.addStretch()

        # 3. Right Cluster: Language, History, Style, Export, Settings, Theme, Primary Action
        lang_lbl = QLabel("源语言:", toolbar)
        lang_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #71717A;")
        layout.addWidget(lang_lbl)

        self.source_lang_combo = QComboBox(toolbar)
        self.source_lang_combo.setFixedWidth(84)
        self.source_lang_combo.addItems(["自动识别", "日语", "韩语", "英语", "繁体中文", "简体中文"])
        self.source_lang_combo.setCurrentText(getattr(self.config, "source_lang", "自动识别"))
        self.source_lang_combo.currentTextChanged.connect(self._on_source_lang_changed)
        layout.addWidget(self.source_lang_combo)

        layout.addSpacing(2)

        # Undo & Redo (Icon-only tool buttons conforming to Pro IDE standards)
        self.undo_btn = QToolButton(toolbar)
        self.undo_btn.setProperty("class", "icon-action-btn")
        self.undo_btn.setIcon(get_icon("undo", color="#A1A1AA", size=14))
        self.undo_btn.setToolTip("撤销 (Ctrl+Z)")
        self.undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._undo)
        layout.addWidget(self.undo_btn)

        self.redo_btn = QToolButton(toolbar)
        self.redo_btn.setProperty("class", "icon-action-btn")
        self.redo_btn.setIcon(get_icon("redo", color="#A1A1AA", size=14))
        self.redo_btn.setToolTip("重做 (Ctrl+Y / Ctrl+Shift+Z)")
        self.redo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.redo_btn.setEnabled(False)
        self.redo_btn.clicked.connect(self._redo)
        layout.addWidget(self.redo_btn)

        layout.addSpacing(4)

        # Single Page Typography Style Modal Button
        self.page_style_btn = QPushButton("排版样式", toolbar)
        self.page_style_btn.setIcon(get_icon("sparkles", color="#0A84FF", size=13))
        self.page_style_btn.setToolTip("当前单页字体与文字排版独立配置")
        self.page_style_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.page_style_btn.clicked.connect(self._open_current_page_style_dialog)
        layout.addWidget(self.page_style_btn)

        # Export Button (Dropdown with Single Page & Batch Export)
        self.export_btn = QPushButton("导出", toolbar)
        self.export_btn.setIcon(get_icon("download", color="#A1A1AA", size=13))
        self.export_btn.setToolTip("导出已翻译漫画 (点击展开单页或全本导出)")
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        export_menu = QMenu(self)
        act_export_curr = export_menu.addAction(get_icon("download", color="#0A84FF", size=14), "导出当前页面 (PNG / JPG / WebP / PDF)...")
        act_export_curr.triggered.connect(self._export_current_page)
        act_export_all = export_menu.addAction(get_icon("folder_open", color="#22C55E", size=14), "批量导出全本已翻译页面...")
        act_export_all.triggered.connect(self._export_all_pages)
        self.export_btn.setMenu(export_menu)
        layout.addWidget(self.export_btn)

        layout.addSpacing(2)

        # Theme Switcher
        self.theme_btn = QToolButton(toolbar)
        self.theme_btn.setProperty("class", "icon-action-btn")
        self.theme_btn.setIcon(get_icon("sun" if self._current_theme == "dark" else "moon", color="#A1A1AA", size=14))
        self.theme_btn.setToolTip("切换明暗模式")
        self.theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_btn)

        # Global Settings
        self.settings_btn = QToolButton(toolbar)
        self.settings_btn.setProperty("class", "icon-action-btn")
        self.settings_btn.setIcon(get_icon("settings", color="#A1A1AA", size=14))
        self.settings_btn.setToolTip("全局系统偏好与默认排版设置")
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.clicked.connect(self._open_settings_dialog)
        layout.addWidget(self.settings_btn)

        layout.addSpacing(4)

        # Primary Action Button: Translate Active Page
        self.run_btn = QPushButton("翻译单页", toolbar)
        self.run_btn.setProperty("class", "primaryBtn")
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.setMinimumWidth(88)
        self.run_btn.setIcon(get_icon("play", color="#FFFFFF", size=13))
        self.run_btn.setToolTip("翻译当前选中的漫画单页 (快捷键: Enter)")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)

        # Keep compatibility instances
        self.batch_toolbar_btn = QPushButton("批量", self)
        self.batch_toolbar_btn.hide()
        self.batch_toolbar_btn.clicked.connect(lambda: self._start_batch(force_retranslate=False))
        self.retranslate_toolbar_btn = QPushButton("全部重翻", self)
        self.retranslate_toolbar_btn.hide()
        self.retranslate_toolbar_btn.clicked.connect(lambda: self._start_batch(force_retranslate=True))

        return toolbar

    def _create_status_bar(self) -> QWidget:
        """Constructs the bottom IDE status bar widget."""
        status_bar = QFrame(self)
        status_bar.setObjectName("statusBar")
        status_bar.setFixedHeight(26)

        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(10, 1, 10, 1)
        layout.setSpacing(10)

        self.status_label = QLabel("就绪", status_bar)
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.progress_bar = QProgressBar(status_bar)
        self.progress_bar.setFixedSize(120, 6)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_page_label = QLabel("PAGE: --", status_bar)
        self.status_page_label.setStyleSheet("font-family: 'JetBrains Mono', 'Cascadia Code', monospace; font-size: 11px; color: #A1A1AA;")
        layout.addWidget(self.status_page_label)

        self.status_bubble_label = QLabel("0 BUBBLES", status_bar)
        self.status_bubble_label.setStyleSheet("font-family: 'JetBrains Mono', 'Cascadia Code', monospace; font-size: 11px; color: #A1A1AA;")
        layout.addWidget(self.status_bubble_label)

        self.status_zoom_label = QLabel("100%", status_bar)
        self.status_zoom_label.setStyleSheet("font-family: 'JetBrains Mono', 'Cascadia Code', monospace; font-size: 11px; color: #A1A1AA;")
        layout.addWidget(self.status_zoom_label)

        encoding_label = QLabel("UTF-8", status_bar)
        encoding_label.setStyleSheet("font-family: 'JetBrains Mono', 'Cascadia Code', monospace; font-size: 11px; color: #71717A;")
        layout.addWidget(encoding_label)

        return status_bar

    def _init_shortcuts(self):
        """Keyboard accelerators."""
        QShortcut(QKeySequence("Ctrl+="), self, self.canvas_view.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self.canvas_view.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.canvas_view.reset_zoom)
        QShortcut(QKeySequence("Ctrl+F"), self, self.canvas_view.fit_in_view)
        QShortcut(QKeySequence("Ctrl+B"), self, self.toggle_sidebar)
        QShortcut(QKeySequence("R"), self, self.canvas_view.toggle_draw_tool)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._handle_undo_shortcut)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._handle_redo_shortcut)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self._handle_redo_shortcut)

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
                        page_style = item_data.get("style") or self.config.style
                        translated = self.typo_engine.render_page(base_bg, model_blocks, page_style)
                        cache_mgr.save_page_cache(path, rendered_img=translated)
                    except Exception as e:
                        print(f"[-] Auto render on select error: {e}")

                self.current_image_data["blocks"] = blocks
                self.current_image_data["erased_img"] = erased
                self.current_image_data["translated_img"] = translated

                self.canvas_view.set_data(cv_img, translated_cv=translated, erased_cv=erased, blocks=blocks)
                self.canvas_view.fit_in_view()
                self.inspector_panel.set_blocks(blocks)

                # Update breadcrumb and status chips
                fname = os.path.basename(path)
                if hasattr(self, "breadcrumb_label"):
                    metrics = self.breadcrumb_label.fontMetrics()
                    elided = metrics.elidedText(f"📂 {fname}", Qt.TextElideMode.ElideMiddle, 190)
                    self.breadcrumb_label.setText(elided)
                    self.breadcrumb_label.setToolTip(path)
                if hasattr(self, "status_page_label"):
                    self.status_page_label.setText(f"PAGE: {fname}")
                if hasattr(self, "status_bubble_label"):
                    self.status_bubble_label.setText(f"{len(blocks)} BUBBLES")

    def _on_zoom_changed(self, zoom: float):
        """Updates zoom percentage chip on status bar and status label."""
        zoom_text = f"{int(round(zoom * 100))}%"
        if hasattr(self, "status_zoom_label"):
            self.status_zoom_label.setText(zoom_text)
        if hasattr(self, "status_label"):
            curr = self.status_label.text()
            if "(" in curr:
                base = curr.split("(")[0].strip()
            else:
                base = curr
            self.status_label.setText(f"{base} ({zoom_text})")

    def _on_bubble_selected_from_canvas(self, block_data: Dict[str, Any]):
        b_id = block_data.get("id")
        if b_id:
            self.inspector_panel.select_block_by_id(b_id)

    def _on_bubble_moving(self, block_data: Dict[str, Any]):
        """Live syncs block coordinates in current_image_data while dragging without heavy re-render."""
        if not self.current_image_data:
            return
        target_id = str(block_data.get("id"))
        blocks = self.current_image_data.get("blocks", [])
        for b in blocks:
            bid = str(b.get("id") if isinstance(b, dict) else getattr(b, "id", None))
            if bid == target_id:
                if isinstance(b, dict):
                    b.update(block_data)
                else:
                    for k, v in block_data.items():
                        if hasattr(b, k):
                            setattr(b, k, v)
                break
        if self.inspector_panel.selected_block and str(self.inspector_panel.selected_block.get("id")) == target_id:
            if isinstance(self.inspector_panel.selected_block, dict):
                self.inspector_panel.selected_block.update(block_data)
            self.inspector_panel.block_title.setText(
                f"气泡 #{str(target_id)[:6]} (位置: {block_data.get('xmin', 0):.1f}%, {block_data.get('ymin', 0):.1f}%)"
            )

    def _on_bubble_geometry_start(self, block_data: Dict[str, Any]):
        """Captured when user presses mouse on bubble before dragging or resizing."""
        if self.current_image_data:
            self._pending_drag_snapshot = self._take_current_snapshot("调整气泡位置/大小")

    def _on_bubble_geometry_changed(self, block_data: Dict[str, Any]):
        """Committed when user releases mouse after dragging or resizing bubble on canvas."""
        self._on_bubble_moving(block_data)
        if self._pending_drag_snapshot and self.current_image_data:
            current_blocks = self.current_image_data.get("blocks", [])
            serialized_curr = [b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else vars(b)) for b in current_blocks]
            if not are_blocks_equal(self._pending_drag_snapshot.blocks, serialized_curr):
                self.undo_manager.push(self._pending_drag_snapshot)
                self._update_undo_redo_ui()
        self._pending_drag_snapshot = None
        self._re_render_current_page()

    def _on_bubble_created(self, new_block: Dict[str, Any]):
        """Handles manual bubble creation from canvas drag selection."""
        if not self.current_image_data:
            self.toast.show_message("请先载入并选择一张漫画页面！", "warning")
            return

        self._push_undo_snapshot("新建气泡")

        blocks = self.current_image_data.get("blocks", [])
        blocks.append(new_block)
        self.current_image_data["blocks"] = blocks

        # Re-set data to update overlays & inspector
        self.canvas_view.set_data(
            original_cv=self.canvas_view.original_cv,
            translated_cv=self.canvas_view.translated_cv,
            erased_cv=self.canvas_view.erased_cv,
            blocks=blocks
        )
        self.inspector_panel.set_blocks(blocks)
        self.inspector_panel.select_block_by_id(new_block["id"])
        self.inspector_panel.tab_widget.setCurrentIndex(0)
        self.inspector_panel.trans_text_edit.setFocus()

        self.toast.show_message(f"已新建气泡 #{str(new_block['id'])[:4]}，可直接在右侧输入译文！", "success")

    def _on_bubble_ocr_requested(self, new_block: Dict[str, Any]):
        """Handles manual box selection for immediate OCR and translation."""
        if not self.current_image_data:
            self.toast.show_message("请先载入并选择一张漫画页面！", "warning")
            return

        self._push_undo_snapshot("框选识别翻译")

        blocks = self.current_image_data.get("blocks", [])
        blocks.append(new_block)
        self.current_image_data["blocks"] = blocks

        # Re-set canvas data to show the new bubble overlay immediately
        self.canvas_view.set_data(
            original_cv=self.canvas_view.original_cv,
            translated_cv=self.canvas_view.translated_cv,
            erased_cv=self.canvas_view.erased_cv,
            blocks=blocks
        )
        self.inspector_panel.set_blocks(blocks)
        self.inspector_panel.select_block_by_id(new_block["id"])

        self._start_block_ocr_translate(new_block)

    def _on_inspector_ocr_translate_block(self, block_data: Dict[str, Any]):
        """Handles Inspector button request to OCR and translate the selected block."""
        self._start_block_ocr_translate(block_data)

    def _start_block_ocr_translate(self, target_block: Dict[str, Any]):
        """Launches BlockOcrTranslateWorker for the target block."""
        if not self.current_image_data:
            return

        original_cv = self.canvas_view.original_cv
        if original_cv is None or original_cv.size == 0:
            self.toast.show_message("原图未就绪，无法执行框选识别！", "warning")
            return

        image_path = self.current_image_data.get("path", "")
        all_blocks = self.current_image_data.get("blocks", [])
        existing_erased = self.current_image_data.get("erased_img", self.canvas_view.erased_cv)

        self.progress_bar.show()
        self.progress_bar.setValue(10)
        self.status_label.setText("正在对框选区域执行 OCR 识别与翻译...")

        from app.core.pipeline.block_worker import BlockOcrTranslateWorker
        self.block_worker = BlockOcrTranslateWorker(
            image_path=image_path,
            original_cv=original_cv,
            target_block=target_block,
            all_blocks=all_blocks,
            existing_erased=existing_erased,
            config=self.config.to_dict(),
            parent=self
        )
        self.block_worker.sig_progress.connect(self._on_pipeline_progress)
        self.block_worker.sig_completed.connect(self._on_block_ocr_completed)
        self.block_worker.sig_error.connect(self._on_block_ocr_error)
        self.block_worker.start()

    def _on_block_ocr_completed(self, result: Dict[str, Any]):
        target_block = result["target_block"]
        all_blocks = result["blocks"]
        erased_img = result["erased_img"]
        translated_img = result["translated_img"]

        if self.current_image_data:
            self.current_image_data["blocks"] = all_blocks
            self.current_image_data["erased_img"] = erased_img
            self.current_image_data["translated_img"] = translated_img

        self.canvas_view.set_data(
            original_cv=self.canvas_view.original_cv,
            translated_cv=translated_img,
            erased_cv=erased_img,
            blocks=all_blocks
        )
        self.inspector_panel.set_blocks(all_blocks)
        self.inspector_panel.select_block_by_id(target_block.get("id"))

        # If user was in original view, switch to translated view so they see the rendered text
        if self.canvas_view.view_mode == "original":
            self.canvas_view.set_view_mode("translated")
            if "translated" in self._mode_buttons:
                self._mode_buttons["translated"].setChecked(True)

        self.progress_bar.hide()
        orig_text = str(target_block.get("original_text", "")).strip().replace("\n", " ")
        trans_text = str(target_block.get("translated_text", "")).strip().replace("\n", " ")
        if orig_text:
            self.toast.show_message(f"已识别并翻译: 【{orig_text[:12]}】→【{trans_text[:12]}】", "success")
            self.status_label.setText(f"框选识别翻译完成: {orig_text[:20]} -> {trans_text[:20]}")
        else:
            self.toast.show_message("框选区域未检测到文字，已创建气泡框供输入！", "info")
            self.status_label.setText("框选区域未检测到文字")

    def _on_block_ocr_error(self, err_msg: str):
        self.progress_bar.hide()
        self.toast.show_message(err_msg, "error")
        self.status_label.setText(f"框选识别失败: {err_msg}")

    def _on_canvas_bubble_swap_prev(self, block_id: str):
        self._push_undo_snapshot("对调气泡翻译")
        self.inspector_panel.select_block_by_id(block_id)
        self.inspector_panel._on_swap_prev_clicked()

    def _on_canvas_bubble_swap_next(self, block_id: str):
        self._push_undo_snapshot("对调气泡翻译")
        self.inspector_panel.select_block_by_id(block_id)
        self.inspector_panel._on_swap_next_clicked()

    def _on_canvas_bubble_merge_prev(self, block_id: str):
        self._push_undo_snapshot("合并气泡")
        self.inspector_panel.select_block_by_id(block_id)
        self.inspector_panel._on_merge_prev_clicked()

    def _on_canvas_bubble_merge_next(self, block_id: str):
        self._push_undo_snapshot("合并气泡")
        self.inspector_panel.select_block_by_id(block_id)
        self.inspector_panel._on_merge_next_clicked()

    def _on_blocks_reordered(self, blocks: list):
        if not self.current_image_data:
            return
        self._push_undo_snapshot("重排气泡顺序")
        self.current_image_data["blocks"] = blocks
        self.canvas_view.blocks = blocks
        self.canvas_view._rebuild_bubbles()
        path = self.current_image_data.get("path")
        if path:
            get_cache_manager().save_page_cache(path, blocks=blocks)
        self._re_render_current_page()

    def _on_block_updated_from_inspector(self, block_data: Dict[str, Any]):
        if not self.current_image_data:
            return
        if self._pending_inspector_snapshot is None:
            self._pending_inspector_snapshot = self._take_current_snapshot("修改气泡内容/样式")
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
        self.canvas_view.update_bubble_item(block_data)
        self._schedule_rerender()

    def _on_block_deleted(self, block_id: str):
        if self.current_image_data and "blocks" in self.current_image_data:
            blocks = self.current_image_data["blocks"]
            target_block = next(
                (b for b in blocks if str(b.get("id") if isinstance(b, dict) else getattr(b, "id", None)) == str(block_id)),
                None
            )
            if target_block is None:
                return

            self._push_undo_snapshot("删除气泡")

            remaining_blocks = [
                b for b in blocks
                if str(b.get("id") if isinstance(b, dict) else getattr(b, "id", None)) != str(block_id)
            ]

            # Pixel restoration: restore original comic artwork pixels at deleted block location
            path = self.current_image_data.get("path")
            original_cv = getattr(self.canvas_view, "original_cv", None)
            if original_cv is None:
                original_cv = self.current_image_data.get("original_cv") or self.current_image_data.get("img")
            if original_cv is None and path and os.path.exists(path):
                original_cv = safe_cv2_imread(path)
                if original_cv is not None:
                    self.current_image_data["original_cv"] = original_cv

            erased_cv = self.current_image_data.get("erased_img")
            if erased_cv is None:
                erased_cv = getattr(self.canvas_view, "erased_cv", None)
            if erased_cv is None and path:
                cached = get_cache_manager().load_page_cache(path, load_images=True)
                erased_cv = cached.get("erased_img")

            if original_cv is not None and erased_cv is not None:
                from app.core.inpaint.restore_helper import restore_block_pixels
                restored_erased = restore_block_pixels(
                    original_img=original_cv,
                    erased_img=erased_cv,
                    deleted_block=target_block,
                    remaining_blocks=remaining_blocks,
                    padding=4
                )
                self.current_image_data["erased_img"] = restored_erased
                self.canvas_view.erased_cv = restored_erased
            elif len(remaining_blocks) == 0 and original_cv is not None:
                self.current_image_data["erased_img"] = original_cv.copy()
                self.canvas_view.erased_cv = original_cv.copy()

            self.current_image_data["blocks"] = remaining_blocks
            self.inspector_panel.set_blocks(remaining_blocks)
            self.canvas_view.blocks = remaining_blocks
            self.canvas_view._rebuild_bubbles()

            # Immediately persist updated blocks and restored erased_img to local disk cache (.amt_cache)
            if path and self.current_image_data.get("erased_img") is not None:
                get_cache_manager().save_page_cache(
                    path,
                    blocks=remaining_blocks,
                    erased_img=self.current_image_data["erased_img"]
                )

            self._re_render_current_page()

    def _schedule_rerender(self):
        """Starts 300ms debounce timer to prevent lag during rapid slider/text input."""
        self._rerender_timer.start(300)

    def _re_render_current_page(self):
        """Performs live typography re-render onto canvas using current style configuration."""
        if not self.current_image_data:
            return

        if self._pending_inspector_snapshot and self.current_image_data:
            current_blocks = self.current_image_data.get("blocks", [])
            serialized_curr = [b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else vars(b)) for b in current_blocks]
            if not are_blocks_equal(self._pending_inspector_snapshot.blocks, serialized_curr):
                self.undo_manager.push(self._pending_inspector_snapshot)
                self._update_undo_redo_ui()
            self._pending_inspector_snapshot = None

        blocks = self.current_image_data.get("blocks", [])

        base_img = self.current_image_data.get("erased_img")
        if base_img is None:
            path = self.current_image_data.get("path")
            if path:
                cached = get_cache_manager().load_page_cache(path, load_images=True)
                base_img = cached.get("erased_img")
                if base_img is not None:
                    self.current_image_data["erased_img"] = base_img

        if base_img is None:
            base_img = getattr(self.canvas_view, "original_cv", None)
        if base_img is None:
            return

        path = self.current_image_data.get("path")

        # If all blocks were deleted, display the clean base image and persist
        if not blocks:
            rendered = base_img.copy()
            self.current_image_data["translated_img"] = rendered
            self.canvas_view.translated_cv = rendered
            self.canvas_view.update_translated_image(rendered, erased_cv=base_img)
            if path:
                get_cache_manager().save_page_cache(
                    path,
                    blocks=[],
                    erased_img=base_img,
                    rendered_img=rendered
                )
            self.status_label.setText("气泡已删除，已还原原图底图")
            return

        # Convert dict blocks to TranslationBlock objects if needed
        model_blocks = [
            b if isinstance(b, TranslationBlock) else TranslationBlock.from_dict(b)
            for b in blocks
        ]

        try:
            page_style = self.current_image_data.get("style") or self.config.style
            rendered = self.typo_engine.render_page(base_img, model_blocks, page_style)
            self.current_image_data["translated_img"] = rendered
            self.canvas_view.translated_cv = rendered
            self.canvas_view.update_translated_image(rendered, erased_cv=base_img)
            # Ensure view mode displays translated artwork
            if self.canvas_view.view_mode in ("original", "inpainted"):
                self.canvas_view.set_view_mode("translated")
                if "translated" in self._mode_buttons:
                    self._mode_buttons["translated"].setChecked(True)

            if path:
                get_cache_manager().save_page_cache(
                    path,
                    blocks=model_blocks,
                    erased_img=base_img,
                    rendered_img=rendered
                )
            self.status_label.setText("排版重绘完成并已自动保存")
        except Exception as e:
            self.status_label.setText(f"重排版失败: {e}")

    def toggle_sidebar(self, *args, force_state: Optional[bool] = None, **kwargs):
        """Collapses or expands the sidebar drawer and dynamically redistributes splitter width."""
        is_visible = self.sidebar_drawer.isVisible()
        new_state = (not is_visible) if force_state is None else force_state
        if new_state == is_visible:
            return

        sizes = self.splitter.sizes()
        nav_w = self.nav_rail.width() if hasattr(self, "nav_rail") and self.nav_rail.isVisible() else 46
        if nav_w <= 0:
            nav_w = 46

        if not new_state:
            # Collapsing: save current sidebar width and shrink queue_panel to nav_rail width
            current_total_w = sizes[0] if sizes else 260
            if current_total_w > nav_w + 50:
                self._saved_sidebar_width = current_total_w
            else:
                self._saved_sidebar_width = getattr(self, "_saved_sidebar_width", 260)

            self.sidebar_drawer.setVisible(False)
            self.queue_panel.setFixedWidth(nav_w)
            reclaimed = max(0, current_total_w - nav_w)
            canvas_w = sizes[1] if len(sizes) > 1 else 800
            insp_w = sizes[2] if len(sizes) > 2 else 280
            self.splitter.setSizes([nav_w, canvas_w + reclaimed, insp_w])
            self.nav_rail.set_collapsed_icon(True)
            self.status_label.setText("已折叠页面列表（沉浸画布模式）")
        else:
            # Expanding: restore full drawer
            target_w = getattr(self, "_saved_sidebar_width", 260)
            target_w = max(target_w, nav_w + 180)
            self.queue_panel.setMinimumWidth(0)
            self.queue_panel.setMaximumWidth(16777215)
            self.sidebar_drawer.setVisible(True)
            current_w = sizes[0] if sizes else nav_w
            canvas_w = sizes[1] if len(sizes) > 1 else 800
            insp_w = sizes[2] if len(sizes) > 2 else 280
            diff = max(0, target_w - current_w)
            self.splitter.setSizes([target_w, max(100, canvas_w - diff), insp_w])
            self.nav_rail.set_collapsed_icon(False)
            self.status_label.setText("已展开页面列表")

    def _on_nav_rail_changed(self, key: str):
        """Handles navigation rail section switches."""
        if key == "pages":
            if not self.sidebar_drawer.isVisible():
                self.toggle_sidebar(force_state=True)
            else:
                self.toggle_sidebar(force_state=False)
            if self.sidebar_drawer.isVisible():
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
        export_dir = getattr(self.config, "export_dir", "") or os.path.join(os.getcwd(), "exported_chapter")
        export_dir = os.path.abspath(export_dir)
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

    def _on_page_cache_cleared(self, item_data: Dict[str, Any]):
        """Called when user clears cache for a page from the sidebar right-click menu."""
        path = item_data.get("path", "")
        filename = os.path.basename(path)
        if not self.current_image_data or self.current_image_data.get("path") != path:
            self.toast.show_message(f"已清除【{filename}】缓存", "success")
            return

        self._push_undo_snapshot("清除页面缓存")

        # Reset in-memory data
        self.current_image_data["blocks"] = []
        self.current_image_data["erased_img"] = None
        self.current_image_data["translated_img"] = None

        # Reset canvas to original image
        original_cv = self.current_image_data.get("original_cv") or self.current_image_data.get("img")
        if original_cv is None:
            # Try to reload from disk
            import cv2
            if path and os.path.exists(path):
                original_cv = cv2.imread(path)
                if original_cv is not None:
                    self.current_image_data["original_cv"] = original_cv
                    self.current_image_data["img"] = original_cv

        self.canvas_view.set_data(
            original_cv=original_cv,
            translated_cv=None,
            erased_cv=None,
            blocks=[]
        )
        self.canvas_view.set_view_mode("original")
        if "original" in self._mode_buttons:
            self._mode_buttons["original"].setChecked(True)
        self.inspector_panel.set_blocks([])
        self.status_label.setText(f"已清除【{filename}】本地缓存，已重置为原图！")
        self.toast.show_message(f"🧹 已清除【{filename}】缓存，重置为原图", "success")

    def _on_canvas_clear_cache_requested(self):
        """Called when user clicks 'Clear Page Cache' from the canvas right-click menu."""
        if not self.current_image_data:
            return
        self._push_undo_snapshot("清除页面缓存")
        path = self.current_image_data.get("path", "")
        page_id = self.current_image_data.get("id", "")
        if path:
            from app.core.cache.cache_manager import get_cache_manager
            get_cache_manager().clear_cache(path)
        self.current_image_data["blocks"] = []
        self.current_image_data["erased_img"] = None
        self.current_image_data["translated_img"] = None
        if page_id:
            self.page_list.update_item_status(page_id, "queued", "等待中")
            item_widget = self.page_list._item_widgets.get(page_id)
            if item_widget:
                item_widget.reload_thumbnail()

        original_cv = self.current_image_data.get("original_cv") or self.current_image_data.get("img")
        if original_cv is None and path and os.path.exists(path):
            import cv2
            original_cv = cv2.imread(path)
            if original_cv is not None:
                self.current_image_data["original_cv"] = original_cv
                self.current_image_data["img"] = original_cv

        self.canvas_view.set_data(
            original_cv=original_cv,
            translated_cv=None,
            erased_cv=None,
            blocks=[]
        )
        self.canvas_view.set_view_mode("original")
        if "original" in self._mode_buttons:
            self._mode_buttons["original"].setChecked(True)
        self.inspector_panel.set_blocks([])
        filename = os.path.basename(path)
        self.status_label.setText(f"已清除【{filename}】本地缓存，已重置为原图！")
        self.toast.show_message(f"🧹 已清除【{filename}】缓存，重置为原图", "success")

    def toggle_theme(self):
        """Toggles between dark and light themes."""
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self.set_theme(new_theme)

    def set_theme(self, theme_name: str):
        """Applies theme stylesheet and updates theme button icon."""
        self._current_theme = theme_name
        tokens = get_tokens(theme_name)
        css = build_stylesheet(tokens)
        app_inst = QApplication.instance()
        if app_inst:
            app_inst.setStyleSheet(css)
        self.setStyleSheet(css)
        self.theme_btn.setIcon(get_icon("sun" if theme_name == "dark" else "moon", color=tokens.text_secondary, size=16))
        self.settings_btn.setIcon(get_icon("settings", color=tokens.text_secondary, size=16))
        self.page_style_btn.setIcon(get_icon("sparkles", color=tokens.accent_primary, size=16))
        self.export_btn.setIcon(get_icon("download", color=tokens.text_secondary, size=16))
        self.batch_toolbar_btn.setIcon(get_icon("play_all", color=tokens.accent_primary, size=16))
        if hasattr(self, "retranslate_toolbar_btn"):
            self.retranslate_toolbar_btn.setIcon(get_icon("refresh", color=tokens.status_warning, size=16))
        if hasattr(self, "undo_btn"):
            self.undo_btn.setIcon(get_icon("undo", color=tokens.text_secondary, size=16))
        if hasattr(self, "redo_btn"):
            self.redo_btn.setIcon(get_icon("redo", color=tokens.text_secondary, size=16))
        self.canvas_view.setBackgroundBrush(QColor(tokens.canvas_bg))
        self.canvas_view.scene.setBackgroundBrush(QColor(tokens.canvas_bg))

    # -------------------------------------------------------------------------
    # M4: Pipeline & Batch Asynchronous Execution & Export
    # -------------------------------------------------------------------------
    def _open_settings_dialog(self):
        """Opens Apple HIG system settings preference modal."""
        dialog = SettingsDialog(config=self.config, parent=self)
        if dialog.exec():
            if getattr(dialog, "re_render_all_requested", False):
                self._re_render_all_pages()
            else:
                self.toast.show_message("设置已成功保存并生效！", "success")

    def _open_page_style_dialog(self, item_data: Dict[str, Any]):
        """Opens page-specific typography modal for the given page."""
        cur_style = item_data.get("style")
        dialog = PageStyleDialog(
            item_data=item_data,
            global_style=self.config.style,
            current_page_style=cur_style,
            parent=self
        )
        if dialog.exec():
            new_style = dialog.applied_style
            item_data["style"] = new_style
            self._re_render_single_page(item_data, new_style)

    def _open_current_page_style_dialog(self):
        """Opens page-specific typography modal for current canvas page."""
        if self.current_image_data and "path" in self.current_image_data:
            self._open_page_style_dialog(self.current_image_data)
        elif self.page_list.items_data:
            self._open_page_style_dialog(self.page_list.items_data[0])
        else:
            self.toast.show_message("请先载入漫画图片或选择页面！", "warning")

    def _re_render_all_pages(self):
        """
        Applies updated global typography settings with zero GUI freezing:
        1. When a page is currently active on canvas (self.current_image_data is not None):
           - Immediately re-renders the active page (< 30ms).
           - Background pages are deferred: in-memory and disk rendered caches are invalidated (< 5ms)
             and will be lazily rendered on-demand when selected or exported.
        2. When no page is active on canvas (e.g. batch headless mode):
           - Re-renders all pages in the chapter queue that have blocks.
        """
        items = self.page_list.items_data
        if not items:
            self.toast.show_message("当前列表无页面，全局文字设置已保存！", "info")
            return

        cache_mgr = get_cache_manager()

        if self.current_image_data is not None:
            # Interactive deferred mode:
            # 1. Immediately re-render current canvas page
            if self.current_image_data.get("blocks"):
                if not self.current_image_data.get("style"):
                    self.current_image_data["style"] = self.config.style
                self._re_render_current_page()

            # 2. Invalidate stale rendered caches for background pages (extremely fast, < 5ms)
            current_path = self.current_image_data.get("path")
            for item in items:
                path = item.get("path")
                if not path or path == current_path:
                    continue

                if not item.get("style"):
                    item["translated_img"] = None
                    paths = cache_mgr.get_cache_paths(path, create_dir=False)
                    for r_key in ("rendered_webp", "rendered_png"):
                        p = paths.get(r_key)
                        if p and os.path.exists(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass

            self.toast.show_message("全局文字设置已生效！当前页面已即时刷新，其余页面将在查看或导出时按需渲染。", "success")
            self.status_label.setText("全局文字设置已生效（按需排版模式，零等待）")
            return

        # Headless / full batch re-render mode (no active canvas item)
        re_rendered_count = 0
        for item in items:
            path = item.get("path")
            if not path or not os.path.exists(path):
                continue

            blocks = item.get("blocks")
            if not blocks:
                cached = cache_mgr.load_page_cache(path, load_images=False)
                blocks = cached.get("blocks")
            if not blocks:
                continue

            erased_img = item.get("erased_img")
            if erased_img is None:
                cached_full = cache_mgr.load_page_cache(path, load_images=True)
                erased_img = cached_full.get("erased_img")
            if erased_img is None:
                erased_img = safe_cv2_imread(path)
            if erased_img is None:
                continue

            model_blocks = [
                b if isinstance(b, TranslationBlock) else TranslationBlock.from_dict(b)
                for b in blocks
            ]
            effective_style = item.get("style") or self.config.style
            try:
                rendered = self.typo_engine.render_page(erased_img, model_blocks, effective_style)
                item["translated_img"] = rendered
                cache_mgr.save_page_cache(path, erased_img=erased_img, blocks=model_blocks, rendered_img=rendered)
                re_rendered_count += 1
            except Exception as e:
                print(f"[-] Re-render page error for {path}: {e}")

        if re_rendered_count > 0:
            self.toast.show_message(f"全局文字设置已生效，已重新渲染全部 {re_rendered_count} 个页面！", "success")
            self.status_label.setText(f"排版重绘完成: 全部 {re_rendered_count} 个页面已重新渲染并保存")
        else:
            self.toast.show_message("全局设置已保存（尚未识别翻译的页面将在翻译时自动套用新排版）", "info")

    def _re_render_single_page(self, item_data: Dict[str, Any], page_style: Optional[StyleConfig]):
        """
        Re-renders only the specified page using its dedicated page style or global style.
        """
        path = item_data.get("path")
        if not path or not os.path.exists(path):
            return

        cache_mgr = get_cache_manager()
        is_current = (self.current_image_data and self.current_image_data.get("path") == path)

        # Get blocks
        blocks = None
        if is_current and self.current_image_data and self.current_image_data.get("blocks"):
            blocks = self.current_image_data["blocks"]
        elif item_data.get("blocks"):
            blocks = item_data["blocks"]
        else:
            cached = cache_mgr.load_page_cache(path, load_images=False)
            blocks = cached.get("blocks")

        # Get base image
        erased_img = None
        if is_current and self.current_image_data and self.current_image_data.get("erased_img") is not None:
            erased_img = self.current_image_data["erased_img"]
        elif item_data.get("erased_img") is not None:
            erased_img = item_data["erased_img"]
        else:
            cached_full = cache_mgr.load_page_cache(path, load_images=True)
            erased_img = cached_full.get("erased_img")

        if erased_img is None:
            erased_img = safe_cv2_imread(path)

        if not blocks:
            filename = os.path.basename(path)
            self.toast.show_message(f"页面【{filename}】排版设置已保存，将在执行翻译时生效！", "info")
            return

        if erased_img is None:
            self.toast.show_message(f"无法读取底图文件: {path}", "error")
            return

        model_blocks = [
            b if isinstance(b, TranslationBlock) else TranslationBlock.from_dict(b)
            for b in blocks
        ]

        style_to_use = page_style if page_style is not None else self.config.style

        try:
            rendered = self.typo_engine.render_page(erased_img, model_blocks, style_to_use)
            item_data["translated_img"] = rendered
            cache_mgr.save_page_cache(path, erased_img=erased_img, blocks=model_blocks, rendered_img=rendered)

            if is_current:
                self.current_image_data["translated_img"] = rendered
                self.canvas_view.translated_cv = rendered
                self.canvas_view.set_data(
                    original_cv=self.canvas_view.original_cv,
                    translated_cv=rendered,
                    erased_cv=erased_img,
                    blocks=blocks
                )
                if self.canvas_view.view_mode in ("original", "inpainted"):
                    self.canvas_view.set_view_mode("translated")
                    if "translated" in self._mode_buttons:
                        self._mode_buttons["translated"].setChecked(True)

            filename = os.path.basename(path)
            self.toast.show_message(f"页面【{filename}】文字设置已应用并重新渲染！", "success")
            self.status_label.setText(f"单页重绘完成: 页面【{filename}】已重新渲染并保存")
        except Exception as e:
            self.toast.show_message(f"单页重排版失败: {e}", "error")

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

        self._push_undo_snapshot("整页翻译")

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

    def _start_batch(self, queue_items: Optional[List[Dict[str, Any]]] = None, force_retranslate: bool = False, export_dir: Optional[str] = None):
        """Launches BatchWorker QThread for chapter queue."""
        if self.active_batch_worker and self.active_batch_worker.isRunning():
            self.active_batch_worker.cancel()
            self.status_label.setText("正在取消批处理任务...")
            self.batch_toolbar_btn.setText("批量翻译")
            if hasattr(self, "retranslate_toolbar_btn"):
                self.retranslate_toolbar_btn.setText("全部重新翻译")
            if hasattr(self.page_list, "batch_btn"):
                self.page_list.batch_btn.setText("🚀 批量翻译 (跳过已完成)")
            if hasattr(self.page_list, "retranslate_all_btn"):
                self.page_list.retranslate_all_btn.setText("🔄 全部重新翻译 (强制覆盖)")
            return

        items = queue_items or self.page_list.items_data
        if not items:
            self.toast.show_message("处理队列为空，请先添加漫画页面！", "warning")
            return

        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.batch_toolbar_btn.setText("取消批处理")
        if hasattr(self, "retranslate_toolbar_btn"):
            self.retranslate_toolbar_btn.setText("取消批处理")
        if hasattr(self.page_list, "batch_btn"):
            self.page_list.batch_btn.setText("⏹ 取消批处理")
        if hasattr(self.page_list, "retranslate_all_btn"):
            self.page_list.retranslate_all_btn.setText("⏹ 取消批处理")

        # When force_retranslate is True (全部重新翻译), clear all software-generated cache files & folders first
        if force_retranslate:
            cache_mgr = get_cache_manager()
            cache_mgr.clear_caches_for_items(items)

            # Reset in-memory state for all queue items
            for it in items:
                it["blocks"] = []
                it["erased_img"] = None
                it["translated_img"] = None
                if hasattr(self.page_list, "update_item_status"):
                    self.page_list.update_item_status(it.get("id"), "queued", "等待中")

            # Reset canvas view to original mode if current page is in list
            if self.current_image_data:
                self.current_image_data["blocks"] = []
                self.current_image_data["erased_img"] = None
                self.current_image_data["translated_img"] = None
                if hasattr(self, "canvas_view") and hasattr(self.canvas_view, "original_cv"):
                    self.canvas_view.set_data(
                        original_cv=self.canvas_view.original_cv,
                        translated_cv=None,
                        erased_cv=None,
                        blocks=[]
                    )
                    self.canvas_view.set_view_mode("original")
                    if hasattr(self, "_mode_buttons") and "original" in self._mode_buttons:
                        self._mode_buttons["original"].setChecked(True)
                if hasattr(self, "inspector_panel"):
                    self.inspector_panel.set_blocks([])

            self.status_label.setText("已清理旧缓存文件夹，正在启动全量重新翻译...")
            self.toast.show_message("已清理原有缓存文件夹，开始全量重新翻译...", "info")

        # When export_dir is provided (e.g. Batch Export), prepare directory.
        # When export_dir is None (Batch Translate), raster baking is deferred to eliminate GUI freeze & save disk space!
        if export_dir:
            export_dir = os.path.abspath(export_dir)
            os.makedirs(export_dir, exist_ok=True)

        self.active_batch_worker = BatchWorker(
            queue_items=items,
            config=self.config.to_dict(),
            export_dir=export_dir,
            force_retranslate=force_retranslate,
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
        if hasattr(self, "retranslate_toolbar_btn"):
            self.retranslate_toolbar_btn.setText("全部重新翻译")
        if hasattr(self.page_list, "batch_btn"):
            self.page_list.batch_btn.setText("🚀 批量翻译 (跳过已完成)")
        if hasattr(self.page_list, "retranslate_all_btn"):
            self.page_list.retranslate_all_btn.setText("🔄 全部重新翻译 (强制覆盖)")
        self.status_label.setText(f"批处理完成: {success_count} 成功, {fail_count} 失败")
        self.toast.show_message(f"批处理完成: {success_count} 成功, {fail_count} 失败", "success" if fail_count == 0 else "warning")

    def _export_current_page(self):
        """Exports currently active translated manga page to high-res PNG/JPG/WebP/PDF."""
        if not self.current_image_data:
            self.toast.show_message("当前页面尚未翻译，无法导出！", "warning")
            return

        # Lazy on-demand rendering if translated image not yet rendered
        if self.canvas_view.translated_cv is None:
            if self.current_image_data.get("blocks"):
                self._re_render_current_page()

        if self.canvas_view.translated_cv is None:
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

    def _export_all_pages(self):
        """Batch exports all pages to an output folder with on-the-fly rendering."""
        items = self.page_list.items_data
        if not items:
            self.toast.show_message("当前列表无页面，无法执行批量导出！", "warning")
            return

        default_dir = getattr(self.config, "export_dir", "") or os.path.join(os.getcwd(), "exported_chapter")
        chosen_dir = QFileDialog.getExistingDirectory(
            self,
            "选择全本漫画导出目录",
            default_dir
        )
        if not chosen_dir:
            return

        self.toast.show_message(f"开始批量导出至: {os.path.basename(chosen_dir)}", "info")
        self._start_batch(force_retranslate=False, export_dir=chosen_dir)

    # -------------------------------------------------------------------------
    # Undo / Redo Architecture (Ctrl+Z / Ctrl+Y)
    # -------------------------------------------------------------------------
    def _take_current_snapshot(self, description: str = "操作") -> Optional[PageSnapshot]:
        if not self.current_image_data:
            return None
        path = self.current_image_data.get("path", "")
        blocks = self.current_image_data.get("blocks", [])
        erased_img = self.current_image_data.get("erased_img")
        style = self.current_image_data.get("style")
        return PageSnapshot.create(
            page_path=path,
            blocks=blocks,
            erased_img=erased_img,
            style=style,
            description=description
        )

    def _push_undo_snapshot(self, description: str = "操作"):
        snap = self._take_current_snapshot(description)
        if snap:
            self.undo_manager.push(snap)
            self._update_undo_redo_ui()

    def _handle_undo_shortcut(self):
        """Intelligent undo: delegates to focused text input if it has local edits, otherwise undoes page operation."""
        focus_w = QApplication.focusWidget()
        if isinstance(focus_w, (QTextEdit, QPlainTextEdit, QLineEdit)):
            doc = getattr(focus_w, "document", lambda: None)()
            if doc and hasattr(doc, "isUndoAvailable") and doc.isUndoAvailable():
                focus_w.undo()
                return
            elif hasattr(focus_w, "isUndoAvailable") and focus_w.isUndoAvailable():
                focus_w.undo()
                return
        self._undo()

    def _handle_redo_shortcut(self):
        """Intelligent redo: delegates to focused text input if it has local edits, otherwise redoes page operation."""
        focus_w = QApplication.focusWidget()
        if isinstance(focus_w, (QTextEdit, QPlainTextEdit, QLineEdit)):
            doc = getattr(focus_w, "document", lambda: None)()
            if doc and hasattr(doc, "isRedoAvailable") and doc.isRedoAvailable():
                focus_w.redo()
                return
            elif hasattr(focus_w, "isRedoAvailable") and focus_w.isRedoAvailable():
                focus_w.redo()
                return
        self._redo()

    def _undo(self):
        """Reverts the last page operation."""
        if not self.undo_manager.can_undo():
            self.toast.show_message("已是最初状态，无法继续撤销", "info")
            return

        current_snap = self._take_current_snapshot("当前状态")
        if not current_snap:
            return

        target_snap = self.undo_manager.undo(current_snap)
        if not target_snap:
            return

        self._restore_snapshot(target_snap)
        self._update_undo_redo_ui()
        self.toast.show_message(f"↩️ 已撤销: {target_snap.description}", "info")
        self.status_label.setText(f"就绪 | 已撤销: {target_snap.description}")

    def _redo(self):
        """Reapplies the previously undone page operation."""
        if not self.undo_manager.can_redo():
            self.toast.show_message("已是最新状态，无法继续重做", "info")
            return

        current_snap = self._take_current_snapshot("当前状态")
        if not current_snap:
            return

        target_snap = self.undo_manager.redo(current_snap)
        if not target_snap:
            return

        self._restore_snapshot(target_snap)
        self._update_undo_redo_ui()
        self.toast.show_message(f"↪️ 已重做: {target_snap.description}", "info")
        self.status_label.setText(f"就绪 | 已重做: {target_snap.description}")

    def _restore_snapshot(self, snapshot: PageSnapshot):
        """Restores application canvas and inspector state from a PageSnapshot."""
        # 1. Switch to the target page if different from currently displayed page
        if snapshot.page_path:
            if not self.current_image_data or self.current_image_data.get("path") != snapshot.page_path:
                for item in self.page_list.items_data:
                    if item.get("path") == snapshot.page_path:
                        self._on_page_selected(item)
                        break

        if not self.current_image_data:
            return

        # 2. Restore blocks, erased background, and style
        restored_blocks = [copy.deepcopy(b) for b in snapshot.blocks]
        self.current_image_data["blocks"] = restored_blocks
        if snapshot.erased_img is not None:
            self.current_image_data["erased_img"] = snapshot.erased_img
            self.canvas_view.erased_cv = snapshot.erased_img
        if snapshot.style is not None:
            self.current_image_data["style"] = snapshot.style

        # 3. Synchronize canvas viewport
        self.canvas_view.set_data(
            original_cv=self.canvas_view.original_cv,
            translated_cv=self.canvas_view.translated_cv,
            erased_cv=self.current_image_data.get("erased_img"),
            blocks=restored_blocks
        )

        # 4. Synchronize inspector panel
        self.inspector_panel.set_blocks(restored_blocks)

        # 5. Persist to disk cache
        if snapshot.page_path:
            get_cache_manager().save_page_cache(
                snapshot.page_path,
                blocks=restored_blocks,
                erased_img=self.current_image_data.get("erased_img")
            )

        # 6. Re-render typography onto canvas
        self._re_render_current_page()

    def _update_undo_redo_ui(self):
        """Synchronizes toolbar buttons state and tooltips with undo/redo stack availability."""
        can_u = self.undo_manager.can_undo()
        can_r = self.undo_manager.can_redo()
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(can_u)
            desc_u = self.undo_manager.get_undo_description()
            self.undo_btn.setToolTip(f"撤销: {desc_u} (Ctrl+Z)" if desc_u else "撤销 (Ctrl+Z)")
        if hasattr(self, "redo_btn"):
            self.redo_btn.setEnabled(can_r)
            desc_r = self.undo_manager.get_redo_description()
            self.redo_btn.setToolTip(f"重做: {desc_r} (Ctrl+Y / Ctrl+Shift+Z)" if desc_r else "重做 (Ctrl+Y)")


