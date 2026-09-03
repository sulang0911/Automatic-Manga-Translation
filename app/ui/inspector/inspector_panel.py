"""
app/ui/inspector/inspector_panel.py
Apple HIG Inspector Panel for speech bubble metadata, typography styling, and live re-rendering.
"""
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSlider, QCheckBox,
    QFrame, QTabWidget, QListWidget, QListWidgetItem, QGroupBox,
    QSplitter, QScrollArea, QColorDialog, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut

from app.core.config import AppConfig
from app.core.models import StrokeMode
from app.core.ocr.base import _is_cjk_char
from app.ui.settings.page_style_dialog import FONT_CHOICES


class BubbleTextEdit(QTextEdit):
    """Custom text edit supporting Ctrl+Enter to apply & advance to next bubble."""
    sig_ctrl_enter = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.sig_ctrl_enter.emit()
            event.accept()
            return
        super().keyPressEvent(event)


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
    sig_block_selected = pyqtSignal(str)
    sig_blocks_reordered = pyqtSignal(list)
    sig_add_bubble_requested = pyqtSignal()

    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.setObjectName("inspectorCard")
        self.current_blocks: List[Dict[str, Any]] = []
        self.selected_block: Optional[Dict[str, Any]] = None
        self._block_custom_color = "#000000"

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Tabs
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)

        # 1. Bubble Editor Tab
        self.bubble_tab = self._create_bubble_tab()
        self.tab_widget.addTab(self.bubble_tab, "💬 气泡编辑")

        # 2. Typography Tab (Single block overrides)
        self.style_tab = self._create_style_tab()
        self.tab_widget.addTab(self.style_tab, "🎨 单气泡样式")

        # 3. Actions Tab
        self.action_tab = self._create_action_tab()
        self.tab_widget.addTab(self.action_tab, "⚡ 快速操作")

    def _create_bubble_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 4, 2, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Vertical, widget)
        splitter.setChildrenCollapsible(False)

        # Top section: Bubble selector list
        top_container = QWidget(splitter)
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(2, 2, 2, 2)
        top_layout.setSpacing(4)

        top_header = QHBoxLayout()
        self.bubble_count_lbl = QLabel("对话气泡列表 (共 0 个):", top_container)
        self.bubble_count_lbl.setStyleSheet("font-weight: 600; font-size: 11px;")
        top_header.addWidget(self.bubble_count_lbl)
        top_header.addStretch()

        self.btn_add_bubble = QPushButton("➕ 新建", top_container)
        self.btn_add_bubble.setToolTip("手动添加新气泡 (也可在画布按快捷键 R 框选)")
        self.btn_add_bubble.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_bubble.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        self.btn_add_bubble.clicked.connect(self.sig_add_bubble_requested.emit)
        top_header.addWidget(self.btn_add_bubble)
        top_layout.addLayout(top_header)

        self.bubble_list = QListWidget(top_container)
        self.bubble_list.setMinimumHeight(80)
        self.bubble_list.itemClicked.connect(self._on_bubble_list_clicked)
        self.bubble_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.bubble_list.customContextMenuRequested.connect(self._on_bubble_list_context_menu)

        # Delete shortcut on list
        del_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), self.bubble_list)
        del_shortcut.activated.connect(self._on_delete_block)

        top_layout.addWidget(self.bubble_list)
        splitter.addWidget(top_container)

        # Bottom section: Selected Bubble Details
        self.detail_frame = QFrame(splitter)
        self.detail_frame.setObjectName("detailFrame")
        detail_layout = QVBoxLayout(self.detail_frame)
        detail_layout.setContentsMargins(4, 4, 4, 4)
        detail_layout.setSpacing(4)

        self.block_title = QLabel("未选中任何气泡", self.detail_frame)
        self.block_title.setObjectName("blockTitle")
        self.block_title.setStyleSheet("font-weight: 600; font-size: 11px; color: #3B82F6;")
        detail_layout.addWidget(self.block_title)

        detail_layout.addWidget(QLabel("原文 (OCR):", self.detail_frame))
        self.orig_text_edit = QTextEdit(self.detail_frame)
        self.orig_text_edit.setMinimumHeight(40)
        self.orig_text_edit.setMaximumHeight(85)
        self.orig_text_edit.textChanged.connect(self._on_orig_text_changed)
        detail_layout.addWidget(self.orig_text_edit)

        detail_layout.addWidget(QLabel("译文 (按 Ctrl+Enter 提交并跳下一条):", self.detail_frame))
        self.trans_text_edit = BubbleTextEdit(self.detail_frame)
        self.trans_text_edit.setMinimumHeight(55)
        self.trans_text_edit.textChanged.connect(self._on_trans_text_changed)
        self.trans_text_edit.sig_ctrl_enter.connect(self._on_apply_and_next_clicked)
        detail_layout.addWidget(self.trans_text_edit, 1)

        # Block Type & Delete
        row_type = QHBoxLayout()
        row_type.addWidget(QLabel("类型:", self.detail_frame))
        self.type_combo = QComboBox(self.detail_frame)
        self.type_combo.addItems(["bubble", "onomatopoeia", "other"])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        row_type.addWidget(self.type_combo, 1)

        self.delete_block_btn = QPushButton("🗑️ 删除", self.detail_frame)
        self.delete_block_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_block_btn.clicked.connect(self._on_delete_block)
        row_type.addWidget(self.delete_block_btn)
        detail_layout.addLayout(row_type)

        # Translation Swap Group
        swap_group = QGroupBox("🔄 翻译对调与纠偏", self.detail_frame)
        swap_layout = QVBoxLayout(swap_group)
        swap_layout.setContentsMargins(4, 4, 4, 4)
        swap_layout.setSpacing(4)

        row_swap_btns = QHBoxLayout()
        self.swap_prev_btn = QPushButton("⬆️ 上移对调", swap_group)
        self.swap_prev_btn.setToolTip("与上一气泡互换翻译")
        self.swap_prev_btn.clicked.connect(self._on_swap_prev_clicked)
        row_swap_btns.addWidget(self.swap_prev_btn)

        self.swap_next_btn = QPushButton("⬇️ 下移对调", swap_group)
        self.swap_next_btn.setToolTip("与下一气泡互换翻译")
        self.swap_next_btn.clicked.connect(self._on_swap_next_clicked)
        row_swap_btns.addWidget(self.swap_next_btn)
        swap_layout.addLayout(row_swap_btns)

        row_swap_target = QHBoxLayout()
        self.swap_target_combo = QComboBox(swap_group)
        row_swap_target.addWidget(self.swap_target_combo, 1)
        self.swap_target_btn = QPushButton("互换", swap_group)
        self.swap_target_btn.clicked.connect(self._on_swap_target_clicked)
        row_swap_target.addWidget(self.swap_target_btn)
        swap_layout.addLayout(row_swap_target)

        detail_layout.addWidget(swap_group)

        # Bubble Merge Group
        merge_group = QGroupBox("🔗 气泡合并 (修复切分)", self.detail_frame)
        merge_layout = QVBoxLayout(merge_group)
        merge_layout.setContentsMargins(4, 4, 4, 4)
        merge_layout.setSpacing(4)

        row_merge_btns = QHBoxLayout()
        self.merge_prev_btn = QPushButton("⬆️ 与上一气泡合并", merge_group)
        self.merge_prev_btn.setToolTip("将当前气泡与上一气泡合并为一个整体（合并坐标与文字）")
        self.merge_prev_btn.clicked.connect(self._on_merge_prev_clicked)
        row_merge_btns.addWidget(self.merge_prev_btn)

        self.merge_next_btn = QPushButton("⬇️ 与下一气泡合并", merge_group)
        self.merge_next_btn.setToolTip("将当前气泡与下一气泡合并为一个整体（合并坐标与文字）")
        self.merge_next_btn.clicked.connect(self._on_merge_next_clicked)
        row_merge_btns.addWidget(self.merge_next_btn)
        merge_layout.addLayout(row_merge_btns)

        detail_layout.addWidget(merge_group)

        # Apply Re-render Button
        self.apply_block_btn = QPushButton("✨ 应用修改 (Ctrl+Enter)", self.detail_frame)
        self.apply_block_btn.setProperty("class", "primaryBtn")
        self.apply_block_btn.setToolTip("快捷键: Ctrl+Enter 提交当前修改并自动跳转至下一个气泡")
        self.apply_block_btn.clicked.connect(self._on_apply_and_next_clicked)
        detail_layout.addWidget(self.apply_block_btn)

        splitter.addWidget(self.detail_frame)
        splitter.setSizes([130, 250])

        layout.addWidget(splitter)
        return widget

    def _create_style_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Target block header
        self.style_block_title = QLabel("当前气泡: 未选中任何气泡", widget)
        self.style_block_title.setStyleSheet("font-weight: 600; font-size: 11px; color: #3B82F6;")
        layout.addWidget(self.style_block_title)

        # Override toggle
        self.block_style_override_cb = QCheckBox("为此气泡启用独立样式 (覆盖全局)", widget)
        self.block_style_override_cb.setStyleSheet("font-weight: 600; font-size: 11px;")
        self.block_style_override_cb.toggled.connect(self._on_block_style_override_toggled)
        layout.addWidget(self.block_style_override_cb)

        # Sub-container for style controls
        self.style_controls_box = QGroupBox("单气泡排版属性", widget)
        sc_layout = QVBoxLayout(self.style_controls_box)
        sc_layout.setContentsMargins(6, 6, 6, 6)
        sc_layout.setSpacing(6)

        # Font Family
        sc_layout.addWidget(QLabel("字体族 (Font Family):"))
        self.block_font_combo = QComboBox(self.style_controls_box)
        for label, _ in FONT_CHOICES:
            self.block_font_combo.addItem(label)
        self.block_font_combo.currentIndexChanged.connect(self._on_block_font_family_changed)
        sc_layout.addWidget(self.block_font_combo)

        # Font Size Scale
        row_scale = QHBoxLayout()
        row_scale.addWidget(QLabel("字号缩放:"))
        self.block_size_val_label = QLabel("1.0x", self.style_controls_box)
        self.block_size_val_label.setStyleSheet("font-weight: 600; color: #3B82F6;")
        row_scale.addWidget(self.block_size_val_label)
        row_scale.addStretch()
        sc_layout.addLayout(row_scale)

        self.block_size_slider = QSlider(Qt.Orientation.Horizontal, self.style_controls_box)
        self.block_size_slider.setRange(5, 30)  # 0.5x to 3.0x
        self.block_size_slider.setValue(10)
        self.block_size_slider.valueChanged.connect(self._on_block_font_scale_changed)
        sc_layout.addWidget(self.block_size_slider)

        # Bold & Italic
        row_style = QHBoxLayout()
        self.block_bold_cb = QCheckBox("粗体 (Bold)", self.style_controls_box)
        self.block_bold_cb.toggled.connect(self._on_block_bold_toggled)
        row_style.addWidget(self.block_bold_cb)

        self.block_italic_cb = QCheckBox("斜体 (Italic)", self.style_controls_box)
        self.block_italic_cb.toggled.connect(self._on_block_italic_toggled)
        row_style.addWidget(self.block_italic_cb)
        sc_layout.addLayout(row_style)

        # Stroke Mode
        sc_layout.addWidget(QLabel("文字描边模式:"))
        self.block_stroke_mode_combo = QComboBox(self.style_controls_box)
        self.block_stroke_mode_combo.addItem("智能对比度 (auto)", StrokeMode.AUTO.value)
        self.block_stroke_mode_combo.addItem("自定义描边 (manual)", StrokeMode.MANUAL.value)
        self.block_stroke_mode_combo.addItem("关闭描边 (off)", StrokeMode.OFF.value)
        self.block_stroke_mode_combo.currentIndexChanged.connect(self._on_block_stroke_mode_changed)
        sc_layout.addWidget(self.block_stroke_mode_combo)

        # Stroke Width
        row_sw = QHBoxLayout()
        row_sw.addWidget(QLabel("描边粗细:"))
        self.block_stroke_w_lbl = QLabel("2.0px", self.style_controls_box)
        row_sw.addWidget(self.block_stroke_w_lbl)
        row_sw.addStretch()
        sc_layout.addLayout(row_sw)

        self.block_stroke_w_slider = QSlider(Qt.Orientation.Horizontal, self.style_controls_box)
        self.block_stroke_w_slider.setRange(5, 50)
        self.block_stroke_w_slider.setValue(20)
        self.block_stroke_w_slider.valueChanged.connect(self._on_block_stroke_w_changed)
        sc_layout.addWidget(self.block_stroke_w_slider)

        # Text Color Presets + Custom
        row_color = QHBoxLayout()
        row_color.addWidget(QLabel("文字颜色:"))

        self.btn_color_black = QPushButton("⚫ 黑字", self.style_controls_box)
        self.btn_color_black.setToolTip("设为纯黑文字 (#000000) 适用于白底")
        self.btn_color_black.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_color_black.clicked.connect(lambda: self._set_quick_text_color("#000000"))
        row_color.addWidget(self.btn_color_black)

        self.btn_color_white = QPushButton("⚪ 白字", self.style_controls_box)
        self.btn_color_white.setToolTip("设为纯白文字 (#FFFFFF) 适用于黑底/暗色背景")
        self.btn_color_white.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_color_white.clicked.connect(lambda: self._set_quick_text_color("#FFFFFF"))
        row_color.addWidget(self.btn_color_white)

        self.block_color_btn = QPushButton("🎨 自定义...", self.style_controls_box)
        self.block_color_btn.setToolTip("在调色板中选择任意文字颜色")
        self.block_color_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.block_color_btn.clicked.connect(self._pick_block_color)
        row_color.addWidget(self.block_color_btn)
        sc_layout.addLayout(row_color)

        # Rotation Angle
        row_angle = QHBoxLayout()
        row_angle.addWidget(QLabel("文字倾斜旋转:"))
        self.block_angle_lbl = QLabel("0.0°", self.style_controls_box)
        self.block_angle_lbl.setStyleSheet("font-weight: 600; color: #8B5CF6;")
        row_angle.addWidget(self.block_angle_lbl)
        row_angle.addStretch()
        sc_layout.addLayout(row_angle)

        self.block_angle_slider = QSlider(Qt.Orientation.Horizontal, self.style_controls_box)
        self.block_angle_slider.setRange(-90, 90)
        self.block_angle_slider.setValue(0)
        self.block_angle_slider.valueChanged.connect(self._on_block_angle_changed)
        sc_layout.addWidget(self.block_angle_slider)

        # Quick angle buttons
        row_quick_angle = QHBoxLayout()
        self.btn_angle_0 = QPushButton("0° 水平", self.style_controls_box)
        self.btn_angle_0.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_angle_0.clicked.connect(lambda: self._set_quick_angle(0.0))
        row_quick_angle.addWidget(self.btn_angle_0)

        self.btn_angle_m15 = QPushButton("↺ -15°", self.style_controls_box)
        self.btn_angle_m15.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_angle_m15.clicked.connect(lambda: self._set_quick_angle(-15.0))
        row_quick_angle.addWidget(self.btn_angle_m15)

        self.btn_angle_p15 = QPushButton("↻ +15°", self.style_controls_box)
        self.btn_angle_p15.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_angle_p15.clicked.connect(lambda: self._set_quick_angle(15.0))
        row_quick_angle.addWidget(self.btn_angle_p15)

        self.btn_angle_p30 = QPushButton("↻ +30°", self.style_controls_box)
        self.btn_angle_p30.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_angle_p30.clicked.connect(lambda: self._set_quick_angle(30.0))
        row_quick_angle.addWidget(self.btn_angle_p30)
        sc_layout.addLayout(row_quick_angle)

        layout.addWidget(self.style_controls_box)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset_block_style = QPushButton("🔄 恢复继承", widget)
        self.btn_reset_block_style.setToolTip("清除本气泡独立样式，恢复继承全局/页面排版")
        self.btn_reset_block_style.clicked.connect(self._on_reset_block_style_clicked)
        btn_layout.addWidget(self.btn_reset_block_style)

        self.btn_apply_block_style = QPushButton("✨ 重绘气泡", widget)
        self.btn_apply_block_style.setProperty("class", "primaryBtn")
        self.btn_apply_block_style.clicked.connect(self.sig_re_render_requested.emit)
        btn_layout.addWidget(self.btn_apply_block_style)
        layout.addLayout(btn_layout)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

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
        count = len(self.current_blocks)
        self.bubble_count_lbl.setText(f"对话气泡列表 (共 {count} 个):")

        for idx, b in enumerate(self.current_blocks):
            b_id = b.get("id", f"b{idx}")
            orig = b.get("original_text", "").replace("\n", " ")[:14]
            trans = b.get("translated_text", "").replace("\n", " ")[:14]
            display_txt = trans if trans else orig
            item = QListWidgetItem(f"#{str(b_id)[:6]} [{b.get('type', 'bubble')}]: {display_txt}")
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
            if b and str(b.get("id")) == str(block_id):
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
        self.block_title.setText(f"气泡 #{str(b_id)[:6]} (位置: {xmin:.1f}%, {ymin:.1f}%)")

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
                preview = other.get("original_text", "").replace("\n", " ")[:12]
                self.swap_target_combo.addItem(f"#{str(other_id)[:4]}: {preview}", other_id)
        self.swap_target_combo.blockSignals(False)

        # Also populate style tab
        self._populate_style_tab(block)

    def _clear_detail(self):
        self.selected_block = None
        self.block_title.setText("未选中任何气泡")
        self.orig_text_edit.clear()
        self.trans_text_edit.clear()
        self._populate_style_tab(None)

    def _populate_style_tab(self, block: Optional[Dict[str, Any]]):
        """Populates single-block style overrides without touching global config."""
        if not block:
            self.style_block_title.setText("当前气泡: 未选中任何气泡")
            self.block_style_override_cb.setEnabled(False)
            self.block_style_override_cb.setChecked(False)
            self.style_controls_box.setEnabled(False)
            return

        b_id = str(block.get("id", "Unknown"))[:6]
        self.style_block_title.setText(f"当前气泡: #{b_id}")
        self.block_style_override_cb.setEnabled(True)

        has_override = (
            block.get("font_family_override") is not None or
            block.get("font_size_override") is not None or
            block.get("font_bold_override") is not None or
            block.get("stroke_mode_override") is not None or
            block.get("text_color_override") is not None
        )

        self.block_style_override_cb.blockSignals(True)
        self.block_style_override_cb.setChecked(has_override)
        self.block_style_override_cb.blockSignals(False)
        self.style_controls_box.setEnabled(has_override)

        # Font family
        cur_font = block.get("font_family_override") or getattr(self.config.style, "font_family", "霞鹜文楷")
        selected_idx = 0
        for i, (label, real_font) in enumerate(FONT_CHOICES):
            if real_font.lower() in cur_font.lower() or cur_font.lower() in label.lower():
                selected_idx = i
                break
        self.block_font_combo.blockSignals(True)
        self.block_font_combo.setCurrentIndex(selected_idx)
        self.block_font_combo.blockSignals(False)

        # Bold
        is_bold = block.get("font_bold_override")
        if is_bold is None:
            is_bold = getattr(self.config.style, "font_bold", True)
        self.block_bold_cb.blockSignals(True)
        self.block_bold_cb.setChecked(bool(is_bold))
        self.block_bold_cb.blockSignals(False)

        # Stroke mode
        sm = block.get("stroke_mode_override") or getattr(self.config.style, "stroke_mode", "auto")
        s_idx = 0
        if sm == StrokeMode.MANUAL.value:
            s_idx = 1
        elif sm == StrokeMode.OFF.value:
            s_idx = 2
        self.block_stroke_mode_combo.blockSignals(True)
        self.block_stroke_mode_combo.setCurrentIndex(s_idx)
        self.block_stroke_mode_combo.blockSignals(False)

        # Stroke width
        sw = block.get("stroke_width_override")
        if sw is None:
            sw = getattr(self.config.style, "stroke_width", 2.0)
        self.block_stroke_w_slider.blockSignals(True)
        self.block_stroke_w_slider.setValue(int(sw * 10))
        self.block_stroke_w_lbl.setText(f"{sw:.1f}px")
        self.block_stroke_w_slider.blockSignals(False)

        # Text color
        tc = block.get("text_color_override") or block.get("text_color") or "#000000"
        self._block_custom_color = tc
        self.block_color_btn.setStyleSheet(f"background-color: {tc}; color: #FFFFFF; font-size: 11px;")

        # Rotation angle
        ang = float(block.get("angle_override", block.get("angle", 0.0)) or 0.0)
        self.block_angle_slider.blockSignals(True)
        self.block_angle_slider.setValue(int(round(ang)))
        self.block_angle_lbl.setText(f"{ang:+.1f}°" if ang != 0 else "0.0°")
        self.block_angle_slider.blockSignals(False)

    def _on_bubble_list_clicked(self, item: QListWidgetItem):
        block = item.data(Qt.ItemDataRole.UserRole)
        if block:
            self._populate_detail(block)
            self.sig_block_selected.emit(str(block.get("id", "")))

    def _on_apply_and_next_clicked(self):
        """Saves current text, triggers re-render, and advances to next bubble in list."""
        if self.selected_block:
            self.selected_block["original_text"] = self.orig_text_edit.toPlainText()
            self.selected_block["translated_text"] = self.trans_text_edit.toPlainText()
            self.sig_block_updated.emit(self.selected_block)

            # Update list item display text
            curr_item = self.bubble_list.currentItem()
            if curr_item:
                b_id = str(self.selected_block.get("id", ""))[:6]
                trans = self.selected_block.get("translated_text", "").replace("\n", " ")[:14]
                orig = self.selected_block.get("original_text", "").replace("\n", " ")[:14]
                display_txt = trans if trans else orig
                curr_item.setText(f"#{b_id} [{self.selected_block.get('type', 'bubble')}]: {display_txt}")

        self.sig_re_render_requested.emit()

        # Advance to next bubble if available
        curr_row = self.bubble_list.currentRow()
        if curr_row >= 0 and curr_row + 1 < self.bubble_list.count():
            next_row = curr_row + 1
            self.bubble_list.setCurrentRow(next_row)
            item = self.bubble_list.item(next_row)
            block = item.data(Qt.ItemDataRole.UserRole)
            if block:
                self._populate_detail(block)
                self.trans_text_edit.setFocus()
                self.trans_text_edit.selectAll()

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
            self.sig_block_deleted.emit(str(b_id))
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

    def _on_bubble_list_context_menu(self, pos):
        item = self.bubble_list.itemAt(pos)
        if not item:
            return
        row = self.bubble_list.row(item)
        if row < 0 or row >= len(self.current_blocks):
            return
        self._on_bubble_list_clicked(item)

        menu = QMenu(self)
        b = self.current_blocks[row]
        bid = str(b.get("id", ""))[:6]
        header = menu.addAction(f"气泡 #{bid}")
        header.setEnabled(False)
        menu.addSeparator()

        act_merge_prev = menu.addAction("🔗 与上一气泡合并 (修复切分)")
        act_merge_prev.setEnabled(row > 0)

        act_merge_next = menu.addAction("🔗 与下一气泡合并 (修复切分)")
        act_merge_next.setEnabled(row < len(self.current_blocks) - 1)

        menu.addSeparator()
        act_swap_prev = menu.addAction("⬆️ 与上一气泡互换翻译")
        act_swap_prev.setEnabled(row > 0)

        act_swap_next = menu.addAction("⬇️ 与下一气泡互换翻译")
        act_swap_next.setEnabled(row < len(self.current_blocks) - 1)

        menu.addSeparator()
        act_del = menu.addAction("🗑️ 删除气泡")

        action = menu.exec(self.bubble_list.mapToGlobal(pos))
        if action == act_merge_prev:
            self._merge_blocks(row, row - 1)
        elif action == act_merge_next:
            self._merge_blocks(row, row + 1)
        elif action == act_swap_prev:
            self._swap_blocks_translation(row, row - 1)
        elif action == act_swap_next:
            self._swap_blocks_translation(row, row + 1)
        elif action == act_del:
            self._on_delete_block()

    def _on_merge_prev_clicked(self):
        if not self.selected_block:
            return
        curr_id = self.selected_block.get("id")
        idx = self._find_block_index_by_id(curr_id)
        if idx > 0:
            self._merge_blocks(idx, idx - 1)

    def _on_merge_next_clicked(self):
        if not self.selected_block:
            return
        curr_id = self.selected_block.get("id")
        idx = self._find_block_index_by_id(curr_id)
        if 0 <= idx < len(self.current_blocks) - 1:
            self._merge_blocks(idx, idx + 1)

    def _merge_blocks(self, idx1: int, idx2: int):
        if not (0 <= idx1 < len(self.current_blocks)) or not (0 <= idx2 < len(self.current_blocks)) or idx1 == idx2:
            return
        b1 = self.current_blocks[idx1]
        b2 = self.current_blocks[idx2]

        y1 = min(b1.get("ymin", 0), b1.get("ymax", 0))
        y2 = min(b2.get("ymin", 0), b2.get("ymax", 0))

        has_cjk = any(_is_cjk_char(c) for c in str(b1.get("original_text", ""))) or any(_is_cjk_char(c) for c in str(b2.get("original_text", "")))
        h1 = abs(b1.get("ymax", 0) - b1.get("ymin", 0))
        w1 = abs(b1.get("xmax", 0) - b1.get("xmin", 0))
        h2 = abs(b2.get("ymax", 0) - b2.get("ymin", 0))
        w2 = abs(b2.get("xmax", 0) - b2.get("xmin", 0))
        is_vert = has_cjk and (h1 >= w1 * 1.2 or h2 >= w2 * 1.2)

        if is_vert:
            first, second = (b1, b2) if max(b1.get("xmax", 0), b1.get("xmin", 0)) >= max(b2.get("xmax", 0), b2.get("xmin", 0)) else (b2, b1)
        else:
            first, second = (b1, b2) if y1 <= y2 else (b2, b1)

        merged_xmin = min(b1.get("xmin", 0), b2.get("xmin", 0))
        merged_ymin = min(b1.get("ymin", 0), b2.get("ymin", 0))
        merged_xmax = max(b1.get("xmax", 0), b2.get("xmax", 0))
        merged_ymax = max(b1.get("ymax", 0), b2.get("ymax", 0))

        ot1 = str(first.get("original_text", "")).strip()
        ot2 = str(second.get("original_text", "")).strip()
        merged_orig = f"{ot1}\n{ot2}" if ot1 and ot2 else (ot1 or ot2)

        tt1 = str(first.get("translated_text", "")).strip()
        tt2 = str(second.get("translated_text", "")).strip()
        if tt1 and tt2:
            if not has_cjk and not any(_is_cjk_char(c) for c in tt1 + tt2):
                merged_trans = f"{tt1} {tt2}"
            else:
                merged_trans = f"{tt1}{tt2}" if not tt1.endswith(("\n", "。", "！", "？", "~", " ")) else f"{tt1} {tt2}"
        else:
            merged_trans = tt1 or tt2

        # Update first block
        first["xmin"] = merged_xmin
        first["ymin"] = merged_ymin
        first["xmax"] = merged_xmax
        first["ymax"] = merged_ymax
        first["original_text"] = merged_orig
        first["translated_text"] = merged_trans

        # Delete second block
        del_idx = idx2 if first is b1 else idx1
        self.current_blocks.pop(del_idx)

        # Notify & re-render
        self.set_blocks(self.current_blocks)
        self.select_block_by_id(first.get("id"))
        self.sig_blocks_reordered.emit(self.current_blocks)
        self.sig_re_render_requested.emit()

    # -------------------------------------------------------------------------
    # Single-Block Style Overrides (Decoupled from Global Settings)
    # -------------------------------------------------------------------------
    def _on_block_style_override_toggled(self, checked: bool):
        self.style_controls_box.setEnabled(checked)
        if not self.selected_block:
            return
        if not checked:
            # Clear overrides
            self.selected_block["font_family_override"] = None
            self.selected_block["font_bold_override"] = None
            self.selected_block["font_size_override"] = None
            self.selected_block["stroke_mode_override"] = None
            self.selected_block["stroke_width_override"] = None
            self.selected_block["text_color_override"] = None
            self.selected_block["angle_override"] = None
        else:
            # Apply current style tab selections to block
            _, real_font = FONT_CHOICES[self.block_font_combo.currentIndex()]
            self.selected_block["font_family_override"] = real_font
            self.selected_block["font_bold_override"] = self.block_bold_cb.isChecked()
            self.selected_block["stroke_mode_override"] = self.block_stroke_mode_combo.currentData()
            self.selected_block["stroke_width_override"] = self.block_stroke_w_slider.value() / 10.0
            self.selected_block["text_color_override"] = self._block_custom_color
            self.selected_block["angle_override"] = float(self.block_angle_slider.value())

        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _on_block_font_family_changed(self, idx: int):
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        if 0 <= idx < len(FONT_CHOICES):
            _, real_font = FONT_CHOICES[idx]
            self.selected_block["font_family_override"] = real_font
            self.sig_block_updated.emit(self.selected_block)
            self.sig_re_render_requested.emit()

    def _on_block_font_scale_changed(self, val: int):
        scale = val / 10.0
        self.block_size_val_label.setText(f"{scale:.1f}x")
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        base_size = self.selected_block.get("font_size", 16.0) or 16.0
        self.selected_block["font_size_override"] = round(base_size * scale, 1)
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _on_block_bold_toggled(self, checked: bool):
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        self.selected_block["font_bold_override"] = checked
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _on_block_italic_toggled(self, checked: bool):
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        self.selected_block["font_italic_override"] = checked
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _on_block_stroke_mode_changed(self, idx: int):
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        self.selected_block["stroke_mode_override"] = self.block_stroke_mode_combo.currentData()
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _on_block_stroke_w_changed(self, val: int):
        sw = val / 10.0
        self.block_stroke_w_lbl.setText(f"{sw:.1f}px")
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        self.selected_block["stroke_width_override"] = sw
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _set_quick_text_color(self, hex_color: str):
        if not self.selected_block:
            return
        if not self.block_style_override_cb.isChecked():
            self.block_style_override_cb.setChecked(True)
        self._block_custom_color = hex_color
        text_fg = "#000000" if hex_color.upper() in ("#FFFFFF", "#FFF") else "#FFFFFF"
        self.block_color_btn.setStyleSheet(f"background-color: {hex_color}; color: {text_fg}; font-size: 11px;")
        self.selected_block["text_color_override"] = hex_color
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _pick_block_color(self):
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        color = QColorDialog.getColor(QColor(self._block_custom_color), self, "选择气泡文字颜色")
        if color.isValid():
            self._block_custom_color = color.name()
            self.block_color_btn.setStyleSheet(f"background-color: {self._block_custom_color}; color: #FFFFFF; font-size: 11px;")
            self.selected_block["text_color_override"] = self._block_custom_color
            self.sig_block_updated.emit(self.selected_block)
            self.sig_re_render_requested.emit()

    def _on_block_angle_changed(self, val: int):
        self.block_angle_lbl.setText(f"{val:+.1f}°" if val != 0 else "0.0°")
        if not self.selected_block or not self.block_style_override_cb.isChecked():
            return
        self.selected_block["angle_override"] = float(val)
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _set_quick_angle(self, angle: float):
        if not self.selected_block:
            return
        if not self.block_style_override_cb.isChecked():
            self.block_style_override_cb.setChecked(True)
        self.block_angle_slider.blockSignals(True)
        self.block_angle_slider.setValue(int(angle))
        self.block_angle_lbl.setText(f"{angle:+.1f}°" if angle != 0 else "0.0°")
        self.block_angle_slider.blockSignals(False)
        self.selected_block["angle_override"] = float(angle)
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()

    def _on_reset_block_style_clicked(self):
        if not self.selected_block:
            return
        self.block_style_override_cb.setChecked(False)
        self.selected_block["font_family_override"] = None
        self.selected_block["font_bold_override"] = None
        self.selected_block["font_size_override"] = None
        self.selected_block["stroke_mode_override"] = None
        self.selected_block["stroke_width_override"] = None
        self.selected_block["text_color_override"] = None
        self.selected_block["angle_override"] = None
        orig_ang = float(self.selected_block.get("angle", 0.0) or 0.0)
        self.block_angle_slider.blockSignals(True)
        self.block_angle_slider.setValue(int(round(orig_ang)))
        self.block_angle_lbl.setText(f"{orig_ang:+.1f}°" if orig_ang != 0 else "0.0°")
        self.block_angle_slider.blockSignals(False)
        self.sig_block_updated.emit(self.selected_block)
        self.sig_re_render_requested.emit()
