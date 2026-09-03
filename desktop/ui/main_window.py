import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QToolBar,
    QPushButton, QLabel, QSplitter, QProgressBar, QFileDialog,
    QButtonGroup, QFrame, QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut

from ..core.config_manager import ConfigManager
from ..core.pipeline_worker import PipelineWorker
from ..core.batch_worker import BatchWorker
from ..core.typography_engine import TypographyEngine
from .canvas_view import CanvasView
from .queue_panel import QueuePanel
from .inspector_panel import InspectorPanel
from .settings_dialog import SettingsDialog
from .toast import Toast
from .styles import get_stylesheet

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.setWindowTitle("🌌 AetherLens - 智能漫画翻译桌面客户端 (PyQt6)")
        self.resize(1380, 880)
        self.setMinimumSize(1000, 650)

        self.current_image_data = None
        self.active_worker = None
        self.active_batch_worker = None

        self.setStyleSheet(get_stylesheet("dark"))

        self._init_ui()
        self._init_shortcuts()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Pre-instantiate canvas view for toolbar signal bindings
        self.canvas_view = CanvasView(self)
        self.canvas_view.sig_bubble_selected.connect(self._on_bubble_selected)
        self.canvas_view.sig_bubble_changed.connect(self._on_bubble_geometry_changed)
        self.canvas_view.sig_zoom_changed.connect(self._on_zoom_changed)

        # 1. Top Apple-style Toolbar
        self.toolbar_widget = self._create_toolbar()
        main_layout.addWidget(self.toolbar_widget)

        # 2. Middle Tri-panel Splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: rgba(255, 255, 255, 0.08); }")

        # Left: Queue
        self.queue_panel = QueuePanel(self)
        self.queue_panel.sig_image_selected.connect(self._on_image_selected)
        self.queue_panel.sig_start_batch.connect(lambda items: self._on_start_batch(items, force_retranslate=False))
        self.queue_panel.sig_start_retranslate_all.connect(lambda items: self._on_start_batch(items, force_retranslate=True))
        self.splitter.addWidget(self.queue_panel)

        # Center: Canvas Container
        center_container = QWidget(self)
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.canvas_view)

        # Bottom Split Slider (visible in split slider mode)
        self.slider_bar = QFrame(self)
        self.slider_bar.setObjectName("glassHeader")
        self.slider_bar.setFixedHeight(36)
        slider_layout = QHBoxLayout(self.slider_bar)
        slider_layout.setContentsMargins(16, 4, 16, 4)
        slider_layout.addWidget(QLabel("原图", self))
        self.split_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.split_slider.setRange(0, 100)
        self.split_slider.setValue(50)
        self.split_slider.valueChanged.connect(lambda v: self.canvas_view.set_split_position(v / 100.0))
        slider_layout.addWidget(self.split_slider)
        slider_layout.addWidget(QLabel("译图", self))
        self.slider_bar.hide()
        center_layout.addWidget(self.slider_bar)

        self.splitter.addWidget(center_container)

        # Right: Inspector & Controls
        self.inspector_panel = InspectorPanel(self.config_manager, self)
        self.inspector_panel.sig_re_render_requested.connect(self._re_render_current_page)
        self.inspector_panel.sig_translate_page_requested.connect(self._run_full_pipeline)
        self.inspector_panel.sig_erase_page_requested.connect(self._run_erase_only)
        self.inspector_panel.sig_export_page_requested.connect(self._export_current_page)
        self.inspector_panel.sig_block_updated.connect(self._on_block_text_updated)
        self.inspector_panel.sig_block_deleted.connect(self._on_block_deleted)
        self.splitter.addWidget(self.inspector_panel)

        # Set Splitter Initial Proportions (260px, Stretch, 320px)
        self.splitter.setSizes([260, 800, 320])
        main_layout.addWidget(self.splitter)

        # 3. Toast Notifications
        self.toast = Toast(self)

    def _create_toolbar(self) -> QWidget:
        header = QFrame(self)
        header.setObjectName("glassHeader")
        header.setFixedHeight(54)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        # Brand / Title
        title = QLabel("🌌 AetherLens", self)
        title.setStyleSheet("font-size: 15px; font-weight: 800; color: #0A84FF; letter-spacing: 0.5px;")
        layout.addWidget(title)

        # Segmented View Controls
        seg_frame = QFrame(self)
        seg_frame.setObjectName("segmentedControl")
        seg_layout = QHBoxLayout(seg_frame)
        seg_layout.setContentsMargins(2, 2, 2, 2)
        seg_layout.setSpacing(2)

        self.btn_group = QButtonGroup(self)
        modes = [
            ("translated", "译图预览"),
            ("side_by_side", "双联对比"),
            ("split_slider", "卷帘滑块"),
            ("inpainted", "擦除背景"),
            ("original", "原图")
        ]
        for mode_id, label in modes:
            btn = QPushButton(label, self)
            btn.setCheckable(True)
            btn.setProperty("class", "segmentedItem")
            if mode_id == "translated":
                btn.setChecked(True)
            self.btn_group.addButton(btn)
            seg_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, m=mode_id: self._on_view_mode_changed(m))

        layout.addWidget(seg_frame)

        # Toggle Bubbles Checkbox
        self.bubble_cb = QCheckBox("显示气泡框", self)
        self.bubble_cb.setChecked(True)
        self.bubble_cb.toggled.connect(self.canvas_view.set_show_bubbles)
        layout.addWidget(self.bubble_cb)

        # Zoom Reset
        self.zoom_btn = QPushButton("适应窗口", self)
        self.zoom_btn.clicked.connect(self.canvas_view.fit_in_view)
        layout.addWidget(self.zoom_btn)

        layout.addStretch()

        # Progress Indicator
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪", self)
        self.status_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        layout.addWidget(self.status_label)

        # Action Buttons
        self.run_btn = QPushButton("🚀 一键翻译", self)
        self.run_btn.setObjectName("primaryBtn")
        self.run_btn.clicked.connect(self._run_full_pipeline)
        layout.addWidget(self.run_btn)

        self.settings_btn = QPushButton("⚙️ 设置", self)
        self.settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(self.settings_btn)

        return header

    def _init_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, self.queue_panel._on_add_files)
        QShortcut(QKeySequence("Ctrl+R"), self, self._run_full_pipeline)
        QShortcut(QKeySequence("Ctrl+S"), self, self._export_current_page)
        QShortcut(QKeySequence("Ctrl+0"), self, self.canvas_view.fit_in_view)

    def _on_view_mode_changed(self, mode: str):
        self.canvas_view.set_view_mode(mode)
        if mode == "split_slider":
            self.slider_bar.show()
        else:
            self.slider_bar.hide()

    def _on_zoom_changed(self, zoom: float):
        self.status_label.setText(f"缩放: {int(zoom * 100)}%")

    def _on_image_selected(self, item_data: dict):
        self.current_image_data = item_data
        path = item_data["path"]
        if not os.path.exists(path):
            return

        stream = open(path, "rb")
        bytes_data = bytearray(stream.read())
        stream.close()
        nparr = np.asarray(bytes_data, dtype=np.uint8)
        original_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        blocks = item_data.get("blocks")
        erased_cv = item_data.get("erased_img")
        translated_cv = item_data.get("translated_img")

        self.canvas_view.set_data(original_cv, translated_cv, erased_cv, blocks)
        self.inspector_panel.set_blocks(blocks)
        self.status_label.setText(f"已载入: {os.path.basename(path)}")

    def _on_bubble_selected(self, block_data: dict):
        self.inspector_panel.select_block_by_id(block_data.get("id"))

    def _on_bubble_geometry_changed(self, block_data: dict):
        self._re_render_current_page()

    def _on_block_text_updated(self, block_data: dict):
        # Trigger real-time re-render
        self._re_render_current_page()

    def _on_block_deleted(self, block_id: str):
        if self.current_image_data and self.current_image_data.get("blocks"):
            self.current_image_data["blocks"] = [
                b for b in self.current_image_data["blocks"] if b.get("id") != block_id
            ]
            self.canvas_view.blocks = self.current_image_data["blocks"]
            self._re_render_current_page()

    def _run_full_pipeline(self):
        if not self.current_image_data:
            self.toast.show_message("请先在左侧选择或添加漫画图片", "warning")
            return

        self._start_pipeline(mode="full")

    def _run_erase_only(self):
        if not self.current_image_data:
            self.toast.show_message("请先在左侧选择漫画图片", "warning")
            return
        self._start_pipeline(mode="inpaint_only")

    def _start_pipeline(self, mode: str = "full"):
        if self.active_worker and self.active_worker.isRunning():
            self.toast.show_message("已有任务正在运行中...", "info")
            return

        img_id = self.current_image_data["id"]
        path = self.current_image_data["path"]

        self.queue_panel.update_item_status(img_id, "processing", "处理中")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)

        self.active_worker = PipelineWorker(
            image_path=path,
            config=self.config_manager.data,
            existing_blocks=self.current_image_data.get("blocks") if mode != "full" else None,
            existing_erased=self.current_image_data.get("erased_img") if mode == "render_only" else None,
            mode=mode,
            parent=self
        )
        self.active_worker.sig_progress.connect(self._on_worker_progress)
        self.active_worker.sig_step_done.connect(self._on_worker_step_done)
        self.active_worker.sig_finished.connect(self._on_worker_finished)
        self.active_worker.sig_error.connect(self._on_worker_error)
        self.active_worker.start()

    def _on_worker_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_worker_step_done(self, step_name: str, data):
        if step_name == "ocr":
            self.current_image_data["blocks"] = data
            self.inspector_panel.set_blocks(data)
            self.canvas_view.blocks = data
        elif step_name == "inpaint":
            self.current_image_data["erased_img"] = data
            self.canvas_view.erased_cv = data
        elif step_name == "render":
            self.current_image_data["translated_img"] = data
            self.canvas_view.translated_cv = data
            self.canvas_view.refresh_display()

    def _on_worker_finished(self, results: dict):
        self.progress_bar.hide()
        self.run_btn.setEnabled(True)
        img_id = self.current_image_data["id"]

        self.current_image_data["blocks"] = results.get("blocks")
        self.current_image_data["erased_img"] = results.get("erased_img")
        self.current_image_data["translated_img"] = results.get("translated_img")

        self.queue_panel.update_item_status(img_id, "completed", "已完成")
        self.queue_panel.update_item_results(img_id, results)

        self.canvas_view.set_data(
            results.get("original_img"),
            results.get("translated_img"),
            results.get("erased_img"),
            results.get("blocks")
        )
        self.inspector_panel.set_blocks(results.get("blocks"))
        self.status_label.setText("页面翻译完成！")
        self.toast.show_message("漫画页面翻译完成！", "success")

    def _on_worker_error(self, err_msg: str):
        self.progress_bar.hide()
        self.run_btn.setEnabled(True)
        if self.current_image_data:
            self.queue_panel.update_item_status(self.current_image_data["id"], "failed", "失败")
        self.status_label.setText("处理出错")
        self.toast.show_message(err_msg, "error")

    def _re_render_current_page(self):
        if not self.current_image_data:
            return
        blocks = self.current_image_data.get("blocks")
        if not blocks:
            return

        base_img = self.current_image_data.get("erased_img")
        if base_img is None:
            base_img = self.canvas_view.original_cv

        if base_img is not None:
            typo_eng = TypographyEngine()
            translated_cv = typo_eng.render_translations(base_img, blocks, self.config_manager.data)
            self.current_image_data["translated_img"] = translated_cv
            self.canvas_view.translated_cv = translated_cv
            self.canvas_view.refresh_display()

    def _export_current_page(self):
        if not self.current_image_data or self.canvas_view.translated_cv is None:
            self.toast.show_message("当前页面尚未翻译，无法导出", "warning")
            return

        raw_path = self.current_image_data["path"]
        default_dir = self.config_manager.get("recent_export_dir") or os.path.dirname(raw_path)
        base_name = os.path.splitext(os.path.basename(raw_path))[0]
        default_save_path = os.path.join(default_dir, f"{base_name}_translated.png")

        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出翻译漫画图像", default_save_path, "PNG 图片 (*.png);;JPG 图片 (*.jpg)"
        )
        if save_path:
            _, buf = cv2.imencode(".png", self.canvas_view.translated_cv)
            with open(save_path, "wb") as f:
                f.write(buf.tobytes())
            self.toast.show_message(f"已导出至: {os.path.basename(save_path)}", "success")

    def _on_start_batch(self, items: list, force_retranslate: bool = False):
        if not items:
            self.toast.show_message("队列为空，请先添加漫画图片", "warning")
            return

        export_dir = QFileDialog.getExistingDirectory(self, "选择批量导出目录 (取消则仅在内存中处理)")
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.toast.show_message(f"开始批量翻译 {len(items)} 个页面...", "info")

        self.active_batch_worker = BatchWorker(
            queue_items=items,
            config=self.config_manager.data,
            export_dir=export_dir,
            force_retranslate=force_retranslate,
            parent=self
        )
        self.active_batch_worker.sig_batch_progress.connect(self._on_batch_progress)
        self.active_batch_worker.sig_item_completed.connect(self._on_batch_item_completed)
        self.active_batch_worker.sig_item_failed.connect(self._on_batch_item_failed)
        self.active_batch_worker.sig_batch_finished.connect(self._on_batch_finished)
        self.active_batch_worker.start()

    def _on_batch_progress(self, current: int, total: int, filename: str, pct: int, msg: str):
        overall_pct = int(((current - 1) / total) * 100 + (pct / total))
        self.progress_bar.setValue(overall_pct)
        self.status_label.setText(f"[{current}/{total}] {filename}: {msg}")

    def _on_batch_item_completed(self, img_id: str, results: dict):
        self.queue_panel.update_item_status(img_id, "completed", "已完成")
        self.queue_panel.update_item_results(img_id, results)

    def _on_batch_item_failed(self, img_id: str, err: str):
        self.queue_panel.update_item_status(img_id, "failed", "失败")

    def _on_batch_finished(self, success_cnt: int, fail_cnt: int):
        self.progress_bar.hide()
        self.status_label.setText("批量任务完成")
        self.toast.show_message(f"批量翻译完成！成功: {success_cnt} 页，失败: {fail_cnt} 页", "success")

    def _open_settings(self):
        dialog = SettingsDialog(self.config_manager, self)
        if dialog.exec():
            self.toast.show_message("设置已更新并保存", "success")
            self._re_render_current_page()
