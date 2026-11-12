"""
app/ui/settings/page_style_dialog.py
Apple macOS Sheet-style Independent Page Typography Configuration Dialog.
Allows customizing font family, bolding, scaling, and strokes for an individual manga page.
Clicking 'Apply' re-renders only the current page without altering global settings.
"""
import os
from typing import Optional, Dict, Any
from copy import deepcopy

from PyQt6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QCheckBox, QGroupBox, QFrame, QColorDialog, QScrollArea, QListView
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QPixmap, QIcon

from app.core.models import StyleConfig, TextColorMode, BgColorMode, StrokeMode, OnomatopoeiaMode
from app.ui.theme.icons import get_icon
from app.ui.theme.tokens import get_tokens, build_stylesheet


FONT_CHOICES = [
    ("霞鹜文楷 (日漫可爱 - 推荐)", "霞鹜文楷"),
    ("幼圆 (圆润可爱)", "幼圆"),
    ("得意黑 (潮流漫画)", "得意黑"),
    ("Comic Sans MS (卡通英文)", "Comic Sans MS"),
    ("Ink Free (随性手绘)", "Ink Free"),
    ("Segoe Print (手写涂鸦)", "Segoe Print"),
    ("楷体 (清秀书法)", "楷体"),
    ("微软雅黑 (系统字体)", "Microsoft YaHei"),
    ("黑体 (标准黑体)", "SimHei"),
    ("Arial", "Arial"),
]


def _configure_combo_popup(combo: QComboBox) -> None:
    view = QListView(combo)
    view.setUniformItemSizes(True)
    combo.setView(view)


def _contrast_text_color(hex_color: str) -> str:
    c = QColor(hex_color or "#000000")
    if not c.isValid():
        return "#FFFFFF"
    lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#000000" if lum > 165 else "#FFFFFF"


class PageStyleDialog(QDialog):
    """
    Modal dialog to customize typography parameters for a single manga page.
    """
    sig_applied = pyqtSignal(object)  # Emits StyleConfig or None

    def __init__(
        self,
        item_data: Dict[str, Any],
        global_style: StyleConfig,
        current_page_style: Optional[StyleConfig] = None,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.item_data = item_data
        self.global_style = global_style
        self.page_style = deepcopy(current_page_style) if current_page_style else deepcopy(global_style)
        self.has_custom_override = current_page_style is not None
        self.applied_style: Optional[StyleConfig] = None

        app_inst = QApplication.instance()
        if app_inst and not app_inst.styleSheet():
            self.setStyleSheet(build_stylesheet(get_tokens("dark")))

        self.setWindowTitle(f"单页排版样式 — {os.path.basename(item_data.get('path', '当前页'))}")
        self.resize(540, 660)
        self.setMinimumSize(480, 520)

        self._init_ui()
        self._load_values()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        # Header Info Card (macOS Inset Banner)
        header = QFrame(self)
        header.setObjectName("pageStyleHeader")
        header.setStyleSheet("""
            QFrame#pageStyleHeader {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(128, 128, 128, 0.18);
                border-radius: 10px;
                padding: 8px 12px;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(6, 4, 6, 4)
        h_layout.setSpacing(12)

        self.thumb_lbl = QLabel(header)
        self.thumb_lbl.setFixedSize(36, 48)
        self.thumb_lbl.setStyleSheet("background: rgba(0, 0, 0, 0.25); border-radius: 5px; border: 1px solid rgba(128, 128, 128, 0.2);")
        self.thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        path = self.item_data.get("path", "")
        if path and os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                self.thumb_lbl.setPixmap(pix.scaled(36, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        h_layout.addWidget(self.thumb_lbl)

        info_col = QVBoxLayout()
        info_col.setSpacing(3)
        name_lbl = QLabel(os.path.basename(path), header)
        name_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
        info_col.addWidget(name_lbl)

        tip_lbl = QLabel("为当前单页定制独立字体与排版样式，仅对本页生效。", header)
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet("font-size: 11px; color: #98989D;")
        info_col.addWidget(tip_lbl)
        h_layout.addLayout(info_col, 1)

        main_layout.addWidget(header)

        # Enable Independent Override Checkbox
        self.override_cb = QCheckBox("启用本页独立排版 (覆盖全局设置)", self)
        self.override_cb.setChecked(self.has_custom_override)
        self.override_cb.setStyleSheet("font-weight: 600; font-size: 12px; color: #0A84FF;")
        self.override_cb.toggled.connect(self._on_override_toggled)
        main_layout.addWidget(self.override_cb)

        # Scrollable form body to prevent vertical clipping on compact screens or high DPI
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        scroll_content = QWidget()
        scroll_content.setObjectName("settingsScrollContent")
        scroll_content.setStyleSheet("QWidget#settingsScrollContent { background-color: transparent; }")
        form_layout = QVBoxLayout(scroll_content)
        form_layout.setContentsMargins(2, 2, 6, 4)
        form_layout.setSpacing(12)

        # Group 1: Typography & Font
        box_font = QGroupBox("字体与字号", scroll_content)
        box_font_layout = QVBoxLayout(box_font)
        box_font_layout.setContentsMargins(12, 10, 12, 12)
        box_font_layout.setSpacing(10)

        box_font_layout.addWidget(QLabel("字体:"))
        self.font_combo = QComboBox(box_font)
        _configure_combo_popup(self.font_combo)
        for label, _ in FONT_CHOICES:
            self.font_combo.addItem(label)
        box_font_layout.addWidget(self.font_combo)

        row_checks = QHBoxLayout()
        row_checks.setSpacing(16)
        self.bold_cb = QCheckBox("粗体", box_font)
        self.bold_cb.setStyleSheet("font-weight: 500;")
        row_checks.addWidget(self.bold_cb)

        self.italic_cb = QCheckBox("斜体", box_font)
        row_checks.addWidget(self.italic_cb)

        self.auto_fit_cb = QCheckBox("自动适应气泡大小", box_font)
        row_checks.addWidget(self.auto_fit_cb)
        row_checks.addStretch()
        box_font_layout.addLayout(row_checks)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("字号缩放:"))
        self.scale_val_lbl = QLabel("1.0x", box_font)
        self.scale_val_lbl.setStyleSheet("font-weight: 600; color: #0A84FF;")
        scale_row.addWidget(self.scale_val_lbl)
        scale_row.addStretch()
        box_font_layout.addLayout(scale_row)

        self.scale_slider = QSlider(Qt.Orientation.Horizontal, box_font)
        self.scale_slider.setRange(5, 30)
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        box_font_layout.addWidget(self.scale_slider)

        form_layout.addWidget(box_font)

        # Group 2: Color & Stroke
        box_color = QGroupBox("颜色、描边与拟声词", scroll_content)
        box_color_layout = QVBoxLayout(box_color)
        box_color_layout.setContentsMargins(12, 10, 12, 12)
        box_color_layout.setSpacing(10)

        row_txt_color = QHBoxLayout()
        row_txt_color.setSpacing(8)
        row_txt_color.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_tc = QLabel("文字颜色:", box_color)
        lbl_tc.setFixedWidth(76)
        row_txt_color.addWidget(lbl_tc)
        self.txt_color_mode_combo = QComboBox(box_color)
        self.txt_color_mode_combo.setFixedHeight(30)
        _configure_combo_popup(self.txt_color_mode_combo)
        self.txt_color_mode_combo.addItem("原文提取颜色", TextColorMode.ORIGINAL.value)
        self.txt_color_mode_combo.addItem("自定义纯色", TextColorMode.CUSTOM.value)
        self.txt_color_mode_combo.currentIndexChanged.connect(self._on_txt_color_mode_changed)
        row_txt_color.addWidget(self.txt_color_mode_combo, 1)

        self.txt_color_btn = QPushButton("选择颜色", box_color)
        self.txt_color_btn.setFixedSize(88, 30)
        self.txt_color_btn.clicked.connect(self._pick_text_color)
        row_txt_color.addWidget(self.txt_color_btn)
        box_color_layout.addLayout(row_txt_color)

        row_stroke = QHBoxLayout()
        row_stroke.setSpacing(8)
        row_stroke.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_sm = QLabel("文字描边:", box_color)
        lbl_sm.setFixedWidth(76)
        row_stroke.addWidget(lbl_sm)
        self.stroke_mode_combo = QComboBox(box_color)
        self.stroke_mode_combo.setFixedHeight(30)
        _configure_combo_popup(self.stroke_mode_combo)
        self.stroke_mode_combo.addItem("智能对比度 (推荐)", StrokeMode.AUTO.value)
        self.stroke_mode_combo.addItem("自定义描边", StrokeMode.MANUAL.value)
        self.stroke_mode_combo.addItem("关闭描边", StrokeMode.OFF.value)
        self.stroke_mode_combo.currentIndexChanged.connect(self._on_stroke_mode_changed)
        row_stroke.addWidget(self.stroke_mode_combo, 1)
        box_color_layout.addLayout(row_stroke)

        row_stroke_w = QHBoxLayout()
        row_stroke_w.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        row_stroke_w.addWidget(QLabel("描边粗细:"))
        self.stroke_w_lbl = QLabel("3.5px", box_color)
        self.stroke_w_lbl.setStyleSheet("font-weight: 600; color: #0A84FF;")
        row_stroke_w.addWidget(self.stroke_w_lbl)
        row_stroke_w.addStretch()
        self.auto_stroke_w_cb = QCheckBox("自动识别描边粗细", box_color)
        row_stroke_w.addWidget(self.auto_stroke_w_cb)
        box_color_layout.addLayout(row_stroke_w)

        self.stroke_w_slider = QSlider(Qt.Orientation.Horizontal, box_color)
        self.stroke_w_slider.setRange(5, 50)
        self.stroke_w_slider.valueChanged.connect(self._on_stroke_w_changed)
        box_color_layout.addWidget(self.stroke_w_slider)

        row_bg = QHBoxLayout()
        row_bg.setSpacing(8)
        row_bg.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_bg = QLabel("背景覆盖:", box_color)
        lbl_bg.setFixedWidth(76)
        row_bg.addWidget(lbl_bg)
        self.bg_mode_combo = QComboBox(box_color)
        self.bg_mode_combo.setFixedHeight(30)
        _configure_combo_popup(self.bg_mode_combo)
        self.bg_mode_combo.addItem("原图周边环境自适应", BgColorMode.ORIGINAL.value)
        self.bg_mode_combo.addItem("纯白覆盖", BgColorMode.CUSTOM.value)
        self.bg_mode_combo.addItem("透明无背景仅文字", BgColorMode.NONE.value)
        row_bg.addWidget(self.bg_mode_combo, 1)
        box_color_layout.addLayout(row_bg)

        row_onoma = QHBoxLayout()
        row_onoma.setSpacing(8)
        row_onoma.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_onoma = QLabel("拟声词处理:", box_color)
        lbl_onoma.setFixedWidth(76)
        row_onoma.addWidget(lbl_onoma)
        self.onoma_mode_combo = QComboBox(box_color)
        self.onoma_mode_combo.setFixedHeight(30)
        _configure_combo_popup(self.onoma_mode_combo)
        self.onoma_mode_combo.addItem("正常消除并翻译 (推荐)", OnomatopoeiaMode.NORMAL.value)
        self.onoma_mode_combo.addItem("半透明艺术保留", OnomatopoeiaMode.TRANSPARENT.value)
        self.onoma_mode_combo.addItem("跳过不处理 (保持原图)", OnomatopoeiaMode.IGNORE.value)
        row_onoma.addWidget(self.onoma_mode_combo, 1)
        box_color_layout.addLayout(row_onoma)

        form_layout.addWidget(box_color)
        form_layout.addStretch()

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll, 1)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.reset_btn = QPushButton("重置为全局默认", self)
        self.reset_btn.setMinimumHeight(32)
        self.reset_btn.setIcon(get_icon("refresh", color="#98989D", size=12))
        self.reset_btn.setToolTip("将本页文字设置恢复与全局设置一致")
        self.reset_btn.clicked.connect(self._reset_to_global)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch()

        self.cancel_btn = QPushButton("取消", self)
        self.cancel_btn.setMinimumHeight(32)
        self.cancel_btn.setMinimumWidth(72)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("应用至当前页", self)
        self.apply_btn.setProperty("class", "primaryBtn")
        self.apply_btn.setObjectName("primaryBtn")
        self.apply_btn.setMinimumHeight(32)
        self.apply_btn.setIcon(get_icon("check", color="#FFFFFF", size=13))
        self.apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.apply_btn)

        main_layout.addLayout(btn_layout)

    def _load_values(self):
        cfg = self.page_style
        target_font = cfg.font_family.lower()
        selected_idx = 0
        for idx, (label, real_font) in enumerate(FONT_CHOICES):
            if real_font.lower() in target_font or target_font in label.lower():
                selected_idx = idx
                break
        self.font_combo.setCurrentIndex(selected_idx)

        self.bold_cb.setChecked(cfg.font_bold)
        self.italic_cb.setChecked(cfg.font_italic)
        self.auto_fit_cb.setChecked(cfg.auto_fit_font_size)

        scale_val = int(round(cfg.font_size_scale * 10))
        self.scale_slider.setValue(max(5, min(30, scale_val)))
        self.scale_val_lbl.setText(f"{cfg.font_size_scale:.1f}x")

        self._custom_text_color = cfg.custom_text_color or "#000000"
        self._update_txt_color_btn()
        mode_idx = 1 if cfg.text_color_mode == TextColorMode.CUSTOM.value else 0
        self.txt_color_mode_combo.setCurrentIndex(mode_idx)

        stroke_idx = 0
        if cfg.stroke_mode == StrokeMode.MANUAL.value:
            stroke_idx = 1
        elif cfg.stroke_mode == StrokeMode.OFF.value:
            stroke_idx = 2
        self.stroke_mode_combo.setCurrentIndex(stroke_idx)

        sw_val = int(round(cfg.stroke_width * 10))
        self.stroke_w_slider.setValue(max(5, min(50, sw_val)))
        self.stroke_w_lbl.setText(f"{cfg.stroke_width:.1f}px")
        self.auto_stroke_w_cb.setChecked(getattr(cfg, "auto_stroke_width", True))

        bg_idx = 0
        if cfg.bg_color_mode == BgColorMode.CUSTOM.value:
            bg_idx = 1
        elif cfg.bg_color_mode == BgColorMode.NONE.value:
            bg_idx = 2
        self.bg_mode_combo.setCurrentIndex(bg_idx)

        onoma_val = getattr(cfg, "onomatopoeia_mode", OnomatopoeiaMode.NORMAL.value)
        o_idx = 0
        if onoma_val == OnomatopoeiaMode.TRANSPARENT.value:
            o_idx = 1
        elif onoma_val == OnomatopoeiaMode.IGNORE.value:
            o_idx = 2
        self.onoma_mode_combo.setCurrentIndex(o_idx)

        self._on_override_toggled(self.override_cb.isChecked())

    def _on_override_toggled(self, checked: bool):
        self.font_combo.setEnabled(checked)
        self.bold_cb.setEnabled(checked)
        self.italic_cb.setEnabled(checked)
        self.auto_fit_cb.setEnabled(checked)
        self.scale_slider.setEnabled(checked)
        self.txt_color_mode_combo.setEnabled(checked)
        self.txt_color_btn.setEnabled(checked and self.txt_color_mode_combo.currentIndex() == 1)
        self._update_txt_color_btn()
        self.stroke_mode_combo.setEnabled(checked)
        self.stroke_w_slider.setEnabled(checked and self.stroke_mode_combo.currentIndex() != 2)
        self.auto_stroke_w_cb.setEnabled(checked and self.stroke_mode_combo.currentIndex() != 2)
        self.bg_mode_combo.setEnabled(checked)
        self.onoma_mode_combo.setEnabled(checked)

    def _on_scale_changed(self, val: int):
        scale = val / 10.0
        self.scale_val_lbl.setText(f"{scale:.1f}x")

    def _on_stroke_w_changed(self, val: int):
        sw = val / 10.0
        self.stroke_w_lbl.setText(f"{sw:.1f}px")

    def _on_txt_color_mode_changed(self, idx: int):
        self.txt_color_btn.setEnabled(self.override_cb.isChecked() and idx == 1)
        self._update_txt_color_btn()

    def _on_stroke_mode_changed(self, idx: int):
        self.stroke_w_slider.setEnabled(self.override_cb.isChecked() and idx != 2)
        self.auto_stroke_w_cb.setEnabled(self.override_cb.isChecked() and idx != 2)

    def _update_txt_color_btn(self):
        text_fg = _contrast_text_color(self._custom_text_color)
        self.txt_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._custom_text_color};
                color: {text_fg};
                border: 1px solid rgba(128, 128, 128, 0.45);
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 8px;
            }}
            QPushButton:disabled {{
                background-color: transparent;
                color: #636366;
                border: 1px solid #2C2C30;
            }}
        """)

    def _pick_text_color(self):
        color = QColorDialog.getColor(QColor(self._custom_text_color), self, "选择文字颜色")
        if color.isValid():
            self._custom_text_color = color.name()
            self._update_txt_color_btn()

    def _reset_to_global(self):
        self.page_style = deepcopy(self.global_style)
        self.override_cb.setChecked(False)
        self._load_values()

    def _on_apply(self):
        if not self.override_cb.isChecked():
            self.applied_style = None
        else:
            _, real_font = FONT_CHOICES[self.font_combo.currentIndex()]
            self.applied_style = StyleConfig(
                font_family=real_font,
                font_bold=self.bold_cb.isChecked(),
                font_italic=self.italic_cb.isChecked(),
                auto_fit_font_size=self.auto_fit_cb.isChecked(),
                font_size_scale=self.scale_slider.value() / 10.0,
                text_color_mode=self.txt_color_mode_combo.currentData(),
                custom_text_color=self._custom_text_color,
                stroke_mode=self.stroke_mode_combo.currentData(),
                stroke_width=self.stroke_w_slider.value() / 10.0,
                auto_stroke_width=self.auto_stroke_w_cb.isChecked(),
                stroke_color=getattr(self.page_style, 'stroke_color', '#FFFFFF'),
                bg_color_mode=self.bg_mode_combo.currentData(),
                line_spacing=getattr(self.page_style, 'line_spacing', 1.30),
                auto_line_spacing=getattr(self.page_style, 'auto_line_spacing', True),
                onomatopoeia_mode=self.onoma_mode_combo.currentData() or OnomatopoeiaMode.NORMAL.value,
                reading_direction=getattr(self.page_style, 'reading_direction', 'manga_rtl'),
            )
        self.sig_applied.emit(self.applied_style)
        self.accept()
