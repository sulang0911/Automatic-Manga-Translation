"""
app/ui/inspector/inspector_panel.py
Apple HIG Inspector Panel for speech bubble metadata, typography styling, and live re-rendering.
"""
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSlider, QCheckBox,
    QFrame, QTabWidget, QListWidget, QListWidgetItem, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from app.core.config import AppConfig


class InspectorPanel(QFrame):
    """
    Inspector Panel Card conforming to Apple HIG.
    Features 3-tab layout: Bubble Editor, Typography Styles, and Quick Actions.
    """
    sig_re_render_requested = pyqtSignal()
    sig_translate_page_requested = pyqtSignal()
    sig_erase_page_requested = pyqtSignal()
    sig_export_page_requested = pyqtSignal()
    sig_open_export_dir_requested = pyqtSignal()
    sig_block_updated = pyqtSignal(dict)
    sig_block_deleted = pyqtSignal(str)

    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.setObjectName("inspectorCard")
        self.current_blocks: List[Dict[str, Any]] = []
        self.selected_block: Optional[Dict[str, Any]] = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Tabs
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)

        # 1. Bubble Editor Tab
        self.bubble_tab = self._create_bubble_tab()
        self.tab_widget.addTab(self.bubble_tab, "💬 气泡编辑")

        # 2. Typography Tab
        self.style_tab = self._create_style_tab()
        self.tab_widget.addTab(self.style_tab, "🎨 排版样式")

        # 3. Actions Tab
        self.action_tab = self._create_action_tab()
        self.tab_widget.addTab(self.action_tab, "⚡ 快速操作")

    def _create_bubble_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(6)

        # Bubble selector list
        layout.addWidget(QLabel("已识别对话气泡列表:"))
        self.bubble_list = QListWidget(self)
        self.bubble_list.setMaximumHeight(110)
        self.bubble_list.itemClicked.connect(self._on_bubble_list_clicked)
        layout.addWidget(self.bubble_list)

        # Selected Bubble Details
        self.detail_frame = QFrame(self)
        self.detail_frame.setObjectName("detailFrame")
        detail_layout = QVBoxLayout(self.detail_frame)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(4)

        self.block_title = QLabel("未选中任何气泡", self.detail_frame)
        self.block_title.setObjectName("blockTitle")
        detail_layout.addWidget(self.block_title)

        detail_layout.addWidget(QLabel("原文 (OCR):"))
        self.orig_text_edit = QTextEdit(self.detail_frame)
        self.orig_text_edit.setFixedHeight(45)
        self.orig_text_edit.textChanged.connect(self._on_orig_text_changed)
        detail_layout.addWidget(self.orig_text_edit)

        detail_layout.addWidget(QLabel("译文 (可实时修改编辑):"))
        self.trans_text_edit = QTextEdit(self.detail_frame)
        self.trans_text_edit.setFixedHeight(55)
        self.trans_text_edit.textChanged.connect(self._on_trans_text_changed)
        detail_layout.addWidget(self.trans_text_edit)

        # Block Type & Delete
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("类型:"))
        self.type_combo = QComboBox(self.detail_frame)
        self.type_combo.addItems(["bubble", "onomatopoeia", "other"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        row_type.addWidget(self.type_combo)

        self.delete_block_btn = QPushButton("🗑️ 删除气泡", self.detail_frame)
        self.delete_block_btn.clicked.connect(self._on_delete_block)
        row_type.addWidget(self.delete_block_btn)
        detail_layout.addLayout(row_type)

        # Translation Swap Group (Quick Fix for Bubble Misplacement)
        swap_group = QGroupBox("🔄 翻译对调与纠偏 (Swap)", self.detail_frame)
        swap_layout = QVBoxLayout(swap_group)
        swap_layout.setContentsMargins(6, 6, 6, 6)
        swap_layout.setSpacing(4)

        row_swap_btns = QHBoxLayout()
        self.swap_prev_btn = QPushButton("⬆️ 与上一气泡互换", swap_group)
        self.swap_prev_btn.setToolTip("将当前选中气泡的译文与上一气泡互相对调")
        self.swap_prev_btn.clicked.connect(self._on_swap_prev_clicked)
        row_swap_btns.addWidget(self.swap_prev_btn)

        self.swap_next_btn = QPushButton("⬇️ 与下一气泡互换", swap_group)
        self.swap_next_btn.setToolTip("将当前选中气泡的译文与下一气泡互相对调")
        self.swap_next_btn.clicked.connect(self._on_swap_next_clicked)
        row_swap_btns.addWidget(self.swap_next_btn)
        swap_layout.addLayout(row_swap_btns)

        row_swap_target = QHBoxLayout()
        row_swap_target.addWidget(QLabel("目标:"))
        self.swap_target_combo = QComboBox(swap_group)
        row_swap_target.addWidget(self.swap_target_combo, 1)
        self.swap_target_btn = QPushButton("执行互换", swap_group)
        self.swap_target_btn.clicked.connect(self._on_swap_target_clicked)
        row_swap_target.addWidget(self.swap_target_btn)
        swap_layout.addLayout(row_swap_target)

        detail_layout.addWidget(swap_group)

        # Apply Re-render Button
        self.apply_block_btn = QPushButton("✨ 应用修改并重绘", self.detail_frame)
        self.apply_block_btn.setProperty("class", "primaryBtn")
        self.apply_block_btn.clicked.connect(self.sig_re_render_requested.emit)
        detail_layout.addWidget(self.apply_block_btn)

        layout.addWidget(self.detail_frame)
        layout.addStretch()
        return widget

    def _create_style_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(8)

        # Font Family
        layout.addWidget(QLabel("字体族 (Font Family):"))
        self.font_combo = QComboBox(widget)
        self.font_options = [
            "霞鹜文楷 (日漫萌系)",
            "幼圆 (圆润可爱)",
            "得意黑 (潮流漫画)",
            "Comic Sans MS (卡通英文)",
            "Ink Free (随性手绘)",
            "Segoe Print (手写涂鸦)",
            "楷体 (清秀书法)",
            "Microsoft YaHei",
            "SimHei",
            "Arial",
        ]
        self.font_combo.addItems(self.font_options)
        current_font = getattr(self.config.style, "font_family", "霞鹜文楷")
        found_idx = -1
        for i, opt in enumerate(self.font_options):
            if current_font.lower() in opt.lower() or opt.lower().startswith(current_font.lower()):
                found_idx = i
                break
        if found_idx >= 0:
            self.font_combo.setCurrentIndex(found_idx)
        self.font_combo.currentTextChanged.connect(self._on_font_family_changed)
        layout.addWidget(self.font_combo)

        # Font Size Scale
        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("字号缩放比例:"))
        self.size_val_label = QLabel("1.0x")
        self.size_val_label.setObjectName("sizeValLabel")
        row_scale.addWidget(self.size_val_label)
        layout.addLayout(row_scale)

        self.size_slider = QSlider(Qt.Orientation.Horizontal, widget)
        self.size_slider.setRange(5, 30)  # 0.5x to 3.0x
        scale_val = int(getattr(self.config.style, "font_size_scale", 1.0) * 10)
        self.size_slider.setValue(scale_val)
        self.size_slider.valueChanged.connect(self._on_font_scale_changed)
        layout.addWidget(self.size_slider)

        # Auto-fit Font Size
        self.auto_fit_cb = QCheckBox("自动适应气泡大小 (二分法寻优)", widget)
        self.auto_fit_cb.setChecked(getattr(self.config.style, "auto_fit_font_size", True))
        self.auto_fit_cb.toggled.connect(self._on_auto_fit_toggled)
        layout.addWidget(self.auto_fit_cb)

        # Bold & Italic
        row_style = QHBoxLayout()
        self.bold_cb = QCheckBox("粗体 (Bold)", widget)
        self.bold_cb.setChecked(getattr(self.config.style, "font_bold", False))
        self.bold_cb.toggled.connect(self._on_bold_toggled)
        row_style.addWidget(self.bold_cb)

        self.italic_cb = QCheckBox("斜体 (Italic)", widget)
        self.italic_cb.setChecked(getattr(self.config.style, "font_italic", False))
        self.italic_cb.toggled.connect(self._on_italic_toggled)
        row_style.addWidget(self.italic_cb)
        layout.addLayout(row_style)

        # Stroke Mode
        layout.addWidget(QLabel("文字边缘描边 (Text Stroke):"))
        self.stroke_mode_combo = QComboBox(widget)
        self.stroke_mode_combo.addItems([
            "auto (ITU-R BT.709 智能对比度)",
            "manual (自定义白色描边)",
            "off (关闭描边)"
        ])
        self.stroke_mode_combo.currentIndexChanged.connect(self._on_stroke_mode_changed)
        layout.addWidget(self.stroke_mode_combo)

        # Background Fill Mode
        layout.addWidget(QLabel("背景填充覆盖 (Background):"))
        self.bg_mode_combo = QComboBox(widget)
        self.bg_mode_combo.addItems([
            "original (自适应周边环境背景色)",
            "custom (纯色白色覆盖)",
            "none (透明背景仅文字)"
        ])
        self.bg_mode_combo.currentIndexChanged.connect(self._on_bg_mode_changed)
        layout.addWidget(self.bg_mode_combo)

        layout.addStretch()
        return widget

    def _create_action_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(10)

        self.btn_export = QPushButton("💾 导出当前已翻译页面", widget)
        self.btn_export.setProperty("class", "primaryBtn")
        self.btn_export.clicked.connect(self.sig_export_page_requested.emit)
        layout.addWidget(self.btn_export)

        self.btn_open_folder = QPushButton("📂 打开导出成果目录", widget)
        self.btn_open_folder.clicked.connect(self.sig_open_export_dir_requested.emit)
        layout.addWidget(self.btn_open_folder)

        layout.addSpacing(6)
        lbl_pipeline = QLabel("分步处理工具:")
        lbl_pipeline.setStyleSheet("font-size: 11px; opacity: 0.7;")
        layout.addWidget(lbl_pipeline)

        self.btn_translate = QPushButton("🤖 仅重译当前页面 (LLM)", widget)
        self.btn_translate.clicked.connect(self.sig_translate_page_requested.emit)
        layout.addWidget(self.btn_translate)

        self.btn_erase = QPushButton("🧹 仅重新擦除背景", widget)
        self.btn_erase.clicked.connect(self.sig_erase_page_requested.emit)
        layout.addWidget(self.btn_erase)

        layout.addStretch()
        return widget

    # -------------------------------------------------------------------------
    # Public Data Synchronization
    # -------------------------------------------------------------------------
    def set_blocks(self, blocks: List[Any]):
        """Populates the bubble list from TranslationBlock objects or dicts."""
        self.current_blocks = [
            b if isinstance(b, dict) else (b.to_dict() if hasattr(b, "to_dict") else vars(b))
            for b in (blocks or [])
        ]
        self.bubble_list.clear()
        for idx, b in enumerate(self.current_blocks):
            b_id = b.get("id", f"b{idx}")
            orig = b.get("original_text", "").replace("\n", " ")[:16]
            item = QListWidgetItem(f"#{b_id} [{b.get('type', 'bubble')}]: {orig}")
            item.setData(Qt.ItemDataRole.UserRole, b)
            self.bubble_list.addItem(item)

        if self.current_blocks:
            self.bubble_list.setCurrentRow(0)
            self._populate_detail(self.current_blocks[0])
        else:
            self._clear_detail()

    def select_block_by_id(self, block_id: str):
        """Selects a bubble item in the list by its block ID."""
        for i in range(self.bubble_list.count()):
            item = self.bubble_list.item(i)
            b = item.data(Qt.ItemDataRole.UserRole)
            if b and b.get("id") == block_id:
                self.bubble_list.setCurrentItem(item)
                self._populate_detail(b)
                break

    # -------------------------------------------------------------------------
    # Internal Handlers
    # -------------------------------------------------------------------------
    def _populate_detail(self, block: Dict[str, Any]):
        self.selected_block = block
        b_id = block.get("id", "Unknown")
        xmin = block.get("xmin", 0)
        ymin = block.get("ymin", 0)
        self.block_title.setText(f"气泡 #{b_id} (位置: {xmin:.1f}%, {ymin:.1f}%)")

        self.orig_text_edit.blockSignals(True)
        self.orig_text_edit.setText(block.get("original_text", ""))
        self.orig_text_edit.blockSignals(False)

        self.trans_text_edit.blockSignals(True)
        self.trans_text_edit.setText(block.get("translated_text", ""))
        self.trans_text_edit.blockSignals(False)

        btype = block.get("type", "bubble")
        idx = self.type_combo.findText(btype)
        if idx >= 0:
            self.type_combo.blockSignals(True)
            self.type_combo.setCurrentIndex(idx)
            self.type_combo.blockSignals(False)

        # Populate swap target combo
        self.swap_target_combo.blockSignals(True)
        self.swap_target_combo.clear()
        for other in self.current_blocks:
            other_id = other.get("id")
            if other_id and str(other_id) != str(b_id):
                preview = other.get("original_text", "").replace("\n", " ")[:14]
                self.swap_target_combo.addItem(f"#{str(other_id)[:4]}: {preview}", other_id)
        self.swap_target_combo.blockSignals(False)

    def _clear_detail(self):
        self.selected_block = None
        self.block_title.setText("未选中任何气泡")
        self.orig_text_edit.clear()
        self.trans_text_edit.clear()

    def _on_bubble_list_clicked(self, item: QListWidgetItem):
        block = item.data(Qt.ItemDataRole.UserRole)
        if block:
            self._populate_detail(block)

    def _on_orig_text_changed(self):
        if self.selected_block:
            self.selected_block["original_text"] = self.orig_text_edit.toPlainText()
            self.sig_block_updated.emit(self.selected_block)

    def _on_trans_text_changed(self):
        if self.selected_block:
            self.selected_block["translated_text"] = self.trans_text_edit.toPlainText()
            self.sig_block_updated.emit(self.selected_block)

    def _on_type_changed(self, new_type: str):
        if self.selected_block:
            self.selected_block["type"] = new_type
            self.sig_block_updated.emit(self.selected_block)

    def _on_delete_block(self):
        if self.selected_block:
            b_id = self.selected_block.get("id", "")
            self.sig_block_deleted.emit(b_id)
            self._clear_detail()

    def _find_block_index_by_id(self, block_id: str) -> int:
        for idx, b in enumerate(self.current_blocks):
            bid = b.get("id") if isinstance(b, dict) else getattr(b, "id", None)
            if str(bid) == str(block_id):
                return idx
        return -1

    def _on_swap_prev_clicked(self):
        if not self.selected_block:
            return
        curr_id = self.selected_block.get("id")
        idx = self._find_block_index_by_id(curr_id)
        if idx > 0:
            self._swap_blocks_translation(idx, idx - 1)

    def _on_swap_next_clicked(self):
        if not self.selected_block:
            return
        curr_id = self.selected_block.get("id")
        idx = self._find_block_index_by_id(curr_id)
        if 0 <= idx < len(self.current_blocks) - 1:
            self._swap_blocks_translation(idx, idx + 1)

    def _on_swap_target_clicked(self):
        if not self.selected_block:
            return
        curr_id = self.selected_block.get("id")
        target_id = self.swap_target_combo.currentData()
        if not target_id or target_id == curr_id:
            return
        idx1 = self._find_block_index_by_id(curr_id)
        idx2 = self._find_block_index_by_id(target_id)
        if idx1 != -1 and idx2 != -1:
            self._swap_blocks_translation(idx1, idx2)

    def _swap_blocks_translation(self, idx1: int, idx2: int):
        b1 = self.current_blocks[idx1]
        b2 = self.current_blocks[idx2]

        t1 = b1.get("translated_text", "")
        t2 = b2.get("translated_text", "")

        b1["translated_text"] = t2
        b2["translated_text"] = t1

        selected_id = self.selected_block.get("id") if self.selected_block else None

        self.set_blocks(self.current_blocks)
        if selected_id:
            self.select_block_by_id(selected_id)

        self.sig_block_updated.emit(b1)
        self.sig_block_updated.emit(b2)
        self.sig_re_render_requested.emit()

    def _on_font_family_changed(self, font_name: str):
        clean_name = font_name.split("(")[0].split("（")[0].strip()
        if hasattr(self.config, "style"):
            self.config.style.font_family = clean_name
        self.sig_re_render_requested.emit()

    def _on_font_scale_changed(self, val: int):
        scale = val / 10.0
        self.size_val_label.setText(f"{scale:.1f}x")
        if hasattr(self.config, "style"):
            self.config.style.font_size_scale = scale
        self.sig_re_render_requested.emit()

    def _on_auto_fit_toggled(self, checked: bool):
        if hasattr(self.config, "style"):
            self.config.style.auto_fit_font_size = checked
        self.sig_re_render_requested.emit()

    def _on_bold_toggled(self, checked: bool):
        if hasattr(self.config, "style"):
            self.config.style.font_bold = checked
        self.sig_re_render_requested.emit()

    def _on_italic_toggled(self, checked: bool):
        if hasattr(self.config, "style"):
            self.config.style.font_italic = checked
        self.sig_re_render_requested.emit()

    def _on_stroke_mode_changed(self, idx: int):
        modes = ["auto", "manual", "off"]
        if 0 <= idx < len(modes) and hasattr(self.config, "style"):
            self.config.style.stroke_mode = modes[idx]
        self.sig_re_render_requested.emit()

    def _on_bg_mode_changed(self, idx: int):
        modes = ["original", "custom", "none"]
        if 0 <= idx < len(modes) and hasattr(self.config, "style"):
            self.config.style.bg_color_mode = modes[idx]
        self.sig_re_render_requested.emit()
