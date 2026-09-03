from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QSlider, QCheckBox, QColorDialog,
    QFrame, QTabWidget, QSpinBox, QDoubleSpinBox, QScrollArea, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from ..core.config_manager import ConfigManager

class InspectorPanel(QFrame):
    sig_re_render_requested = pyqtSignal()
    sig_translate_page_requested = pyqtSignal()
    sig_erase_page_requested = pyqtSignal()
    sig_export_page_requested = pyqtSignal()
    sig_block_updated = pyqtSignal(dict)
    sig_block_deleted = pyqtSignal(str)

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setObjectName("inspectorCard")
        self.current_blocks = []
        self.selected_block = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

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
        layout.setContentsMargins(4, 8, 4, 8)
        layout.setSpacing(8)

        # Bubble selector list
        self.bubble_list = QListWidget(self)
        self.bubble_list.setMaximumHeight(120)
        self.bubble_list.itemClicked.connect(self._on_bubble_list_clicked)
        layout.addWidget(QLabel("已识别对话气泡列表:"))
        layout.addWidget(self.bubble_list)

        # Selected Bubble Details
        self.detail_frame = QFrame(self)
        self.detail_frame.setObjectName("cardFrame")
        detail_layout = QVBoxLayout(self.detail_frame)
        detail_layout.setContentsMargins(10, 10, 10, 10)
        detail_layout.setSpacing(8)

        self.block_title = QLabel("选择一个气泡进行编辑", self)
        self.block_title.setStyleSheet("font-weight: 700; color: #0A84FF; font-size: 13px;")
        detail_layout.addWidget(self.block_title)

        detail_layout.addWidget(QLabel("原始日文/原文:"))
        self.orig_text_edit = QTextEdit(self)
        self.orig_text_edit.setMaximumHeight(65)
        detail_layout.addWidget(self.orig_text_edit)

        detail_layout.addWidget(QLabel("翻译后中文/译文:"))
        self.trans_text_edit = QTextEdit(self)
        self.trans_text_edit.setMaximumHeight(75)
        self.trans_text_edit.textChanged.connect(self._on_trans_text_changed)
        detail_layout.addWidget(self.trans_text_edit)

        btn_row = QHBoxLayout()
        self.del_block_btn = QPushButton("🗑️ 删除气泡", self)
        self.del_block_btn.setObjectName("dangerBtn")
        self.del_block_btn.clicked.connect(self._on_delete_block)
        btn_row.addWidget(self.del_block_btn)

        self.apply_block_btn = QPushButton("✓ 更新重绘", self)
        self.apply_block_btn.setObjectName("primaryBtn")
        self.apply_block_btn.clicked.connect(self.sig_re_render_requested.emit)
        btn_row.addWidget(self.apply_block_btn)
        detail_layout.addLayout(btn_row)

        layout.addWidget(self.detail_frame)
        layout.addStretch()
        return widget

    def _create_style_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(12)

        cfg = self.config_manager.data

        # Font Family
        layout.addWidget(QLabel("字体选择:"))
        self.font_combo = QComboBox(self)
        self.font_combo.addItems(["霞鹜文楷", "得意黑", "幼圆", "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "Arial", "MS Gothic"])
        self.font_combo.setCurrentText(cfg.get("font_family", "霞鹜文楷"))
        self.font_combo.currentTextChanged.connect(lambda v: self._update_cfg("font_family", v))
        layout.addWidget(self.font_combo)

        # Font Size Scale
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("字号缩放:"))
        self.size_val_label = QLabel(f"{cfg.get('font_size_scale', 1.0):.1f}x", self)
        size_row.addWidget(self.size_val_label)
        layout.addLayout(size_row)

        self.size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.size_slider.setRange(5, 25) # 0.5 to 2.5
        self.size_slider.setValue(int(cfg.get("font_size_scale", 1.0) * 10))
        self.size_slider.valueChanged.connect(self._on_size_slider_changed)
        layout.addWidget(self.size_slider)

        # Auto Fit & Bold
        toggles_row = QHBoxLayout()
        self.auto_fit_cb = QCheckBox("自适应气泡字号", self)
        self.auto_fit_cb.setChecked(cfg.get("auto_fit_font_size", True))
        self.auto_fit_cb.toggled.connect(lambda v: self._update_cfg("auto_fit_font_size", v))
        toggles_row.addWidget(self.auto_fit_cb)

        self.bold_cb = QCheckBox("粗体", self)
        self.bold_cb.setChecked(cfg.get("font_bold", False))
        self.bold_cb.toggled.connect(lambda v: self._update_cfg("font_bold", v))
        toggles_row.addWidget(self.bold_cb)
        layout.addLayout(toggles_row)

        # Text Color
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("文字颜色:"))
        self.text_color_btn = QPushButton(cfg.get("text_color", "#000000"), self)
        self.text_color_btn.setStyleSheet(f"background-color: {cfg.get('text_color', '#000000')}; color: #FFFFFF; font-weight: bold;")
        self.text_color_btn.clicked.connect(self._choose_text_color)
        color_row.addWidget(self.text_color_btn)
        layout.addLayout(color_row)

        # Text Stroke / Outline
        layout.addWidget(QLabel("描边轮廓 (Stroke):"))
        self.stroke_mode_combo = QComboBox(self)
        self.stroke_mode_combo.addItems(["auto - 智能黑白反差", "manual - 自定义描边", "off - 无描边"])
        mode = cfg.get("stroke_mode", "auto")
        if mode == "manual": self.stroke_mode_combo.setCurrentIndex(1)
        elif mode == "off": self.stroke_mode_combo.setCurrentIndex(2)
        else: self.stroke_mode_combo.setCurrentIndex(0)
        self.stroke_mode_combo.currentIndexChanged.connect(self._on_stroke_mode_changed)
        layout.addWidget(self.stroke_mode_combo)

        # Background Bubble Fill Opacity
        bg_op_row = QHBoxLayout()
        bg_op_row.addWidget(QLabel("气泡背景填充不透明度:"))
        self.bg_op_label = QLabel(f"{int(cfg.get('bg_opacity', 0.95) * 100)}%", self)
        bg_op_row.addWidget(self.bg_op_label)
        layout.addLayout(bg_op_row)

        self.bg_op_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.bg_op_slider.setRange(0, 100)
        self.bg_op_slider.setValue(int(cfg.get("bg_opacity", 0.95) * 100))
        self.bg_op_slider.valueChanged.connect(self._on_bg_op_changed)
        layout.addWidget(self.bg_op_slider)

        # Re-render Button
        apply_btn = QPushButton("🎨 实时应用排版", self)
        apply_btn.setObjectName("primaryBtn")
        apply_btn.clicked.connect(self.sig_re_render_requested.emit)
        layout.addWidget(apply_btn)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _create_action_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(12)

        cfg = self.config_manager.data

        # Target Language
        layout.addWidget(QLabel("目标翻译语言:"))
        self.target_lang_combo = QComboBox(self)
        self.target_lang_combo.addItems(["简体中文", "繁體中文", "English", "日本語", "한국어"])
        self.target_lang_combo.setCurrentText(cfg.get("target_lang", "简体中文"))
        self.target_lang_combo.currentTextChanged.connect(lambda v: self._update_cfg("target_lang", v))
        layout.addWidget(self.target_lang_combo)

        # Provider / Model
        layout.addWidget(QLabel("翻译引擎与模型:"))
        self.provider_combo = QComboBox(self)
        self.provider_combo.addItems(["DeepSeek (deepseek-chat)", "OpenAI (gpt-4o-mini)", "Gemini (gemini-1.5-flash)", "自定义 API"])
        prov = cfg.get("provider", "deepseek")
        if prov == "openai": self.provider_combo.setCurrentIndex(1)
        elif prov == "gemini": self.provider_combo.setCurrentIndex(2)
        elif prov == "custom": self.provider_combo.setCurrentIndex(3)
        else: self.provider_combo.setCurrentIndex(0)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        layout.addWidget(self.provider_combo)

        # Action Buttons
        self.translate_btn = QPushButton("🚀 一键翻译当前页面", self)
        self.translate_btn.setObjectName("primaryBtn")
        self.translate_btn.clicked.connect(self.sig_translate_page_requested.emit)
        layout.addWidget(self.translate_btn)

        self.erase_btn = QPushButton("🧹 仅清除原文字背景", self)
        self.erase_btn.clicked.connect(self.sig_erase_page_requested.emit)
        layout.addWidget(self.erase_btn)

        self.export_btn = QPushButton("💾 导出高保真译图", self)
        self.export_btn.clicked.connect(self.sig_export_page_requested.emit)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        return widget

    def set_blocks(self, blocks: list):
        self.current_blocks = blocks or []
        self.bubble_list.clear()
        for idx, b in enumerate(self.current_blocks):
            preview = b.get("original_text", "").replace("\n", " ")[:15]
            item = QListWidgetItem(f"#{idx+1} [{b.get('type','bubble')}] {preview}")
            item.setData(Qt.ItemDataRole.UserRole, b.get("id"))
            self.bubble_list.addItem(item)

        if self.current_blocks:
            self.select_block_by_id(self.current_blocks[0].get("id"))
        else:
            self.block_title.setText("未检测到气泡")
            self.orig_text_edit.clear()
            self.trans_text_edit.clear()

    def select_block_by_id(self, block_id: str):
        for idx, b in enumerate(self.current_blocks):
            if str(b.get("id")) == str(block_id):
                self.selected_block = b
                self.bubble_list.setCurrentRow(idx)
                self.block_title.setText(f"气泡 #{idx+1} (ID: {block_id})")
                self.orig_text_edit.setPlainText(b.get("original_text", ""))
                self.trans_text_edit.setPlainText(b.get("translated_text", ""))
                self.tab_widget.setCurrentIndex(0)
                break

    def _on_bubble_list_clicked(self, item: QListWidgetItem):
        b_id = item.data(Qt.ItemDataRole.UserRole)
        self.select_block_by_id(b_id)

    def _on_trans_text_changed(self):
        if self.selected_block:
            self.selected_block["translated_text"] = self.trans_text_edit.toPlainText()
            self.sig_block_updated.emit(self.selected_block)

    def _on_delete_block(self):
        if self.selected_block:
            b_id = self.selected_block.get("id")
            self.current_blocks = [b for b in self.current_blocks if b.get("id") != b_id]
            self.sig_block_deleted.emit(b_id)
            self.set_blocks(self.current_blocks)

    def _choose_text_color(self):
        color = QColorDialog.getColor(QColor(self.config_manager.get("text_color", "#000000")), self)
        if color.isValid():
            hex_c = color.name().upper()
            self._update_cfg("text_color", hex_c)
            self.text_color_btn.setText(hex_c)
            self.text_color_btn.setStyleSheet(f"background-color: {hex_c}; color: #FFFFFF; font-weight: bold;")

    def _on_size_slider_changed(self, val: int):
        scale = val / 10.0
        self.size_val_label.setText(f"{scale:.1f}x")
        self._update_cfg("font_size_scale", scale)

    def _on_stroke_mode_changed(self, idx: int):
        modes = ["auto", "manual", "off"]
        self._update_cfg("stroke_mode", modes[idx])

    def _on_bg_op_changed(self, val: int):
        self.bg_op_label.setText(f"{val}%")
        self._update_cfg("bg_opacity", val / 100.0)

    def _on_provider_changed(self, idx: int):
        providers = ["deepseek", "openai", "gemini", "custom"]
        models = ["deepseek-chat", "gpt-4o-mini", "gemini-1.5-flash", "custom-model"]
        self._update_cfg("provider", providers[idx])
        self._update_cfg("model", models[idx])

    def _update_cfg(self, key: str, val):
        self.config_manager.set(key, val)
