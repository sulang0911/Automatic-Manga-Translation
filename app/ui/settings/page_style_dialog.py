"""
app/ui/settings/page_style_dialog.py
Independent Page-level Typography Configuration Dialog.
Allows customizing font family, bolding, scaling, and strokes for an individual manga page.
Clicking 'Apply' re-renders only the current page without altering global settings.
"""
import os
from typing import Optional, Dict, Any
from copy import deepcopy

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QCheckBox, QGroupBox, QFrame, QColorDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QPixmap, QIcon

from app.core.models import StyleConfig, TextColorMode, BgColorMode, StrokeMode, OnomatopoeiaMode
from app.ui.theme.icons import get_icon


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
        # If page already has independent style, use a copy of it; otherwise use copy of global
        self.page_style = deepcopy(current_page_style) if current_page_style else deepcopy(global_style)
        self.has_custom_override = current_page_style is not None
        self.applied_style: Optional[StyleConfig] = None

        self.setWindowTitle(f"🎨 单页文字设置 — {os.path.basename(item_data.get('path', '当前页'))}")
        self.resize(540, 620)
        self.setMinimumSize(480, 520)

        self._init_ui()
        self._load_values()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(12)

        # Header Info Card
        header = QFrame(self)
        header.setObjectName("pageStyleHeader")
        header.setStyleSheet("""
            #pageStyleHeader {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 8px 12px;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(12)

        # Thumbnail
        self.thumb_lbl = QLabel(header)
        self.thumb_lbl.setFixedSize(40, 52)
        self.thumb_lbl.setStyleSheet("background: rgba(0,0,0,0.2); border-radius: 4px;")
        self.thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        path = self.item_data.get("path", "")
        if path and os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                self.thumb_lbl.setPixmap(pix.scaled(40, 52, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        h_layout.addWidget(self.thumb_lbl)

        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        name_lbl = QLabel(os.path.basename(path), header)
        name_lbl.setStyleSheet("font-weight: 600; font-size: 13px; color: #FFFFFF;")
        info_col.addWidget(name_lbl)

        tip_lbl = QLabel("为本张漫画单页定制独立文字排版，点击【应用】仅对此页生效，不影响其他页面。", header)
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet("font-size: 11px; opacity: 0.7;")
        info_col.addWidget(tip_lbl)
        h_layout.addLayout(info_col, 1)

        main_layout.addWidget(header)

        # Enable Independent Override Checkbox
        self.override_cb = QCheckBox("启用本页独立排版 (覆盖全局设置)", self)
        self.override_cb.setChecked(self.has_custom_override)
        self.override_cb.setStyleSheet("font-weight: 600; font-size: 12px; color: #3B82F6;")
        self.override_cb.toggled.connect(self._on_override_toggled)
        main_layout.addWidget(self.override_cb)

        # Group 1: Typography & Font
        box_font = QGroupBox("字体与排版设置", self)
        box_font_layout = QVBoxLayout(box_font)
        box_font_layout.setSpacing(8)

        box_font_layout.addWidget(QLabel("选择字体 (默认可爱字体):"))
        self.font_combo = QComboBox(box_font)
        for label, _ in FONT_CHOICES:
            self.font_combo.addItem(label)
        box_font_layout.addWidget(self.font_combo)

        # Bold & Italic & Auto-fit
        row_checks = QHBoxLayout()
        self.bold_cb = QCheckBox("粗体 (Bold)", box_font)
        self.bold_cb.setStyleSheet("font-weight: 600;")
        row_checks.addWidget(self.bold_cb)

        self.italic_cb = QCheckBox("斜体 (Italic)", box_font)
        row_checks.addWidget(self.italic_cb)

        self.auto_fit_cb = QCheckBox("自动适应气泡大小", box_font)
        row_checks.addWidget(self.auto_fit_cb)
        box_font_layout.addLayout(row_checks)

        # Scale slider
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("字号缩放比例:"))
        self.scale_val_lbl = QLabel("1.0x", box_font)
        self.scale_val_lbl.setStyleSheet("font-weight: 600; color: #3B82F6;")
        scale_row.addWidget(self.scale_val_lbl)
        scale_row.addStretch()
        box_font_layout.addLayout(scale_row)

        self.scale_slider = QSlider(Qt.Orientation.Horizontal, box_font)
        self.scale_slider.setRange(5, 30)  # 0.5x to 3.0x
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        box_font_layout.addWidget(self.scale_slider)

        main_layout.addWidget(box_font)

        # Group 2: Color & Stroke
        box_color = QGroupBox("文字颜色与描边", self)
        box_color_layout = QVBoxLayout(box_color)
        box_color_layout.setSpacing(8)

        # Text Color Row
        row_txt_color = QHBoxLayout()
        row_txt_color.addWidget(QLabel("文字颜色:"))
        self.txt_color_mode_combo = QComboBox(box_color)
        self.txt_color_mode_combo.addItem("原文提取颜色", TextColorMode.ORIGINAL.value)
        self.txt_color_mode_combo.addItem("自定义纯色", TextColorMode.CUSTOM.value)
        self.txt_color_mode_combo.currentIndexChanged.connect(self._on_txt_color_mode_changed)
        row_txt_color.addWidget(self.txt_color_mode_combo, 1)

        self.txt_color_btn = QPushButton("选择颜色", box_color)
        self.txt_color_btn.setFixedWidth(80)
        self.txt_color_btn.clicked.connect(self._pick_text_color)
        row_txt_color.addWidget(self.txt_color_btn)
        box_color_layout.addLayout(row_txt_color)

        # Stroke Row
        row_stroke = QHBoxLayout()
        row_stroke.addWidget(QLabel("文字描边:"))
        self.stroke_mode_combo = QComboBox(box_color)
        self.stroke_mode_combo.addItem("智能对比度 (推荐)", StrokeMode.AUTO.value)
        self.stroke_mode_combo.addItem("自定义白色/指定色描边", StrokeMode.MANUAL.value)
        self.stroke_mode_combo.addItem("关闭描边", StrokeMode.OFF.value)
        self.stroke_mode_combo.currentIndexChanged.connect(self._on_stroke_mode_changed)
        row_stroke.addWidget(self.stroke_mode_combo, 1)
        box_color_layout.addLayout(row_stroke)

        # Stroke Width Row
        row_stroke_w = QHBoxLayout()
        row_stroke_w.addWidget(QLabel("描边粗细 (px):"))
        self.stroke_w_lbl = QLabel("2.0px", box_color)
        self.stroke_w_lbl.setStyleSheet("font-weight: 500;")
        row_stroke_w.addWidget(self.stroke_w_lbl)
        row_stroke_w.addStretch()
        box_color_layout.addLayout(row_stroke_w)

        self.stroke_w_slider = QSlider(Qt.Orientation.Horizontal, box_color)
        self.stroke_w_slider.setRange(5, 50)  # 0.5px to 5.0px
        self.stroke_w_slider.valueChanged.connect(self._on_stroke_w_changed)
        box_color_layout.addWidget(self.stroke_w_slider)

        # Background Fill
        row_bg = QHBoxLayout()
        row_bg.addWidget(QLabel("背景气泡覆盖:"))
        self.bg_mode_combo = QComboBox(box_color)
        self.bg_mode_combo.addItem("原图周边环境自适应", BgColorMode.ORIGINAL.value)
        self.bg_mode_combo.addItem("纯白覆盖", BgColorMode.CUSTOM.value)
        self.bg_mode_combo.addItem("透明无背景仅文字", BgColorMode.NONE.value)
        row_bg.addWidget(self.bg_mode_combo, 1)
        box_color_layout.addLayout(row_bg)

        # Onomatopoeia strategy
        row_onoma = QHBoxLayout()
        row_onoma.addWidget(QLabel("语气词与拟声词:"))
        self.onoma_mode_combo = QComboBox(box_color)
        self.onoma_mode_combo.addItem("正常消除并翻译 (默认推荐)", OnomatopoeiaMode.NORMAL.value)
        self.onoma_mode_combo.addItem("半透明艺术保留", OnomatopoeiaMode.TRANSPARENT.value)
        self.onoma_mode_combo.addItem("跳过不处理 (保持原图)", OnomatopoeiaMode.IGNORE.value)
        row_onoma.addWidget(self.onoma_mode_combo, 1)
        box_color_layout.addLayout(row_onoma)

        main_layout.addWidget(box_color)

        main_layout.addStretch()

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("🔄 重置为全局设置", self)
        self.reset_btn.setToolTip("将本页文字设置恢复与全局设置一致")
        self.reset_btn.clicked.connect(self._reset_to_global)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch()

        self.cancel_btn = QPushButton("取消", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("✨ 应用 (仅修改当前页)", self)
        self.apply_btn.setProperty("class", "primaryBtn")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: #FFFFFF;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
        """)
        self.apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.apply_btn)

        main_layout.addLayout(btn_layout)

    def _load_values(self):
        # Font family
        cfg = self.page_style
        target_font = cfg.font_family.lower()
        selected_idx = 0
        for idx, (label, real_font) in enumerate(FONT_CHOICES):
            if real_font.lower() in target_font or target_font in label.lower():
                selected_idx = idx
                break
        self.font_combo.setCurrentIndex(selected_idx)

        # Bold & Italic & Auto fit
        self.bold_cb.setChecked(cfg.font_bold)
        self.italic_cb.setChecked(cfg.font_italic)
        self.auto_fit_cb.setChecked(cfg.auto_fit_font_size)

        # Scale
        scale_val = int(round(cfg.font_size_scale * 10))
        self.scale_slider.setValue(max(5, min(30, scale_val)))
        self.scale_val_lbl.setText(f"{cfg.font_size_scale:.1f}x")

        # Text color
        self._custom_text_color = cfg.custom_text_color or "#000000"
        self._update_txt_color_btn()
        mode_idx = 1 if cfg.text_color_mode == TextColorMode.CUSTOM.value else 0
        self.txt_color_mode_combo.setCurrentIndex(mode_idx)

        # Stroke mode
        stroke_idx = 0
        if cfg.stroke_mode == StrokeMode.MANUAL.value:
            stroke_idx = 1
        elif cfg.stroke_mode == StrokeMode.OFF.value:
            stroke_idx = 2
        self.stroke_mode_combo.setCurrentIndex(stroke_idx)

        # Stroke width
        sw_val = int(round(cfg.stroke_width * 10))
        self.stroke_w_slider.setValue(max(5, min(50, sw_val)))
        self.stroke_w_lbl.setText(f"{cfg.stroke_width:.1f}px")

        # Bg mode
        bg_idx = 0
        if cfg.bg_color_mode == BgColorMode.CUSTOM.value:
            bg_idx = 1
        elif cfg.bg_color_mode == BgColorMode.NONE.value:
            bg_idx = 2
        self.bg_mode_combo.setCurrentIndex(bg_idx)

        # Onomatopoeia mode
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
        self.stroke_mode_combo.setEnabled(checked)
        self.stroke_w_slider.setEnabled(checked and self.stroke_mode_combo.currentIndex() != 2)
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

    def _on_stroke_mode_changed(self, idx: int):
        self.stroke_w_slider.setEnabled(self.override_cb.isChecked() and idx != 2)

    def _update_txt_color_btn(self):
        self.txt_color_btn.setStyleSheet(f"background-color: {self._custom_text_color}; color: #FFFFFF; font-size: 11px;")

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
            # User chose to discard override and follow global
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
                stroke_color=getattr(self.page_style, 'stroke_color', '#FFFFFF'),
                bg_color_mode=self.bg_mode_combo.currentData(),
                line_spacing=getattr(self.page_style, 'line_spacing', 1.15),
                onomatopoeia_mode=self.onoma_mode_combo.currentData() or OnomatopoeiaMode.NORMAL.value,
                reading_direction=getattr(self.page_style, 'reading_direction', 'manga_rtl'),
            )
        self.sig_applied.emit(self.applied_style)
        self.accept()
