"""
app/ui/settings/settings_dialog.py
Apple HIG System Preferences Modal Dialog.
Supports configuring LLM Providers, OCR Engines, Inpainting Models, and Typography styles.
"""
from typing import Optional, Tuple
from PyQt6.QtWidgets import (
    QApplication, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QSlider, QCheckBox, QListView,
    QStackedWidget, QListWidget, QListWidgetItem, QGroupBox, QFileDialog, QScrollArea, QFrame,
    QColorDialog, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from app.core.config import AppConfig, DEFAULT_SYSTEM_PROMPT
from app.core.models import StyleConfig, TextColorMode, BgColorMode, StrokeMode, OnomatopoeiaMode
from app.ui.theme.tokens import get_tokens, build_stylesheet
from desktop.core.translation_engine import TranslationEngine
from app.ui.settings.page_style_dialog import FONT_CHOICES


def _configure_combo_popup(combo: QComboBox) -> None:
    """Ensures QComboBox uses a QListView popup so QSS item padding and solid backgrounds render cleanly on Windows."""
    view = QListView(combo)
    view.setUniformItemSizes(True)
    combo.setView(view)


def _build_data_combo(items, current_value: str) -> QComboBox:
    """Creates a themed QComboBox populated with (label, value) pairs and selects current_value."""
    combo = QComboBox()
    _configure_combo_popup(combo)
    for label, val in items:
        combo.addItem(label, val)
    idx = combo.findData(current_value)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    return combo


def _contrast_text_color(hex_color: str) -> str:
    """Returns #000000 or #FFFFFF for readable contrast against hex_color."""
    c = QColor(hex_color or "#000000")
    if not c.isValid():
        return "#FFFFFF"
    lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return "#000000" if lum > 165 else "#FFFFFF"


class AppModelTestWorker(QThread):
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, provider: str, api_key: str, model: str, endpoint: str,
                 system_prompt: str, test_text: str, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.test_text = test_text
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if self._is_cancelled:
            return
        try:
            engine = TranslationEngine(
                provider=self.provider,
                api_key=self.api_key,
                model=self.model,
                custom_endpoint=self.endpoint,
                system_prompt=self.system_prompt,
                target_lang="简体中文"
            )
            if self._is_cancelled:
                return
            result = engine.test_connection(self.test_text)
            if not self._is_cancelled:
                self.finished_signal.emit(True, result)
        except Exception as e:
            if not self._is_cancelled:
                self.finished_signal.emit(False, str(e))


class SettingsDialog(QDialog):
    """
    Apple HIG System Settings Dialog with categorized navigation list on the left
    and tabbed preference panels on the right.
    """
    sig_apply_all_pages = pyqtSignal()

    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.re_render_all_requested = False
        self._test_worker: Optional[AppModelTestWorker] = None
        self.setWindowTitle("系统偏好设置")
        self.resize(780, 680)
        self.setMinimumSize(720, 600)

        theme_name = getattr(self.config, "theme", "dark") or "dark"
        self._tokens = get_tokens(theme_name)
        app_inst = QApplication.instance()
        if app_inst and not app_inst.styleSheet():
            self.setStyleSheet(build_stylesheet(self._tokens))

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left Category Navigation List (macOS System Settings Sidebar)
        self.nav_list = QListWidget(self)
        self.nav_list.setObjectName("settingsNavList")
        self.nav_list.setFixedWidth(188)
        self.nav_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav_list.setSpacing(2)

        categories = [
            ("🤖 AI 翻译大模型", 0),
            ("🔍 OCR 文字识别", 1),
            ("🧹 图像背景修复", 2),
            ("📝 译文文字设置", 3),
            ("💾 导出与缓存", 4),
            ("📖 专有名词词典", 5),
        ]
        for name, _ in categories:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(168, 38))
            self.nav_list.addItem(item)

        main_layout.addWidget(self.nav_list)

        # Right Stacked Pages
        right_container = QVBoxLayout()
        right_container.setContentsMargins(0, 0, 0, 0)
        right_container.setSpacing(12)
        self.stack = QStackedWidget(self)
        right_container.addWidget(self.stack, 1)

        self.stack.addWidget(self._create_llm_page())
        self.stack.addWidget(self._create_ocr_page())
        self.stack.addWidget(self._create_inpaint_page())
        self.stack.addWidget(self._create_typography_page())
        self.stack.addWidget(self._create_export_page())
        self.stack.addWidget(self._create_glossary_page())

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(8, 4, 12, 0)
        btn_layout.setSpacing(10)
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("取消", self)
        self.cancel_btn.setMinimumHeight(32)
        self.cancel_btn.setMinimumWidth(76)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.save_btn = QPushButton("仅保存配置", self)
        self.save_btn.setMinimumHeight(32)
        self.save_btn.setMinimumWidth(96)
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)

        self.apply_all_btn = QPushButton("✨ 保存并应用到全部页面", self)
        self.apply_all_btn.setProperty("class", "primaryBtn")
        self.apply_all_btn.setObjectName("primaryBtn")
        self.apply_all_btn.setMinimumHeight(32)
        self.apply_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: #FFFFFF;
                border: 1px solid transparent;
                border-radius: 7px;
                padding: 5px 16px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        self.apply_all_btn.clicked.connect(self._on_apply_all)
        btn_layout.addWidget(self.apply_all_btn)

        right_container.addLayout(btn_layout)
        main_layout.addLayout(right_container, 1)

    def _create_scroll_page(self) -> Tuple[QScrollArea, QWidget, QVBoxLayout]:
        """Creates a properly scoped transparent scroll container so child widgets retain their theme backgrounds."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        widget = QWidget()
        widget.setObjectName("settingsScrollContent")
        widget.setStyleSheet("QWidget#settingsScrollContent { background-color: transparent; }")

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 12, 8)
        layout.setSpacing(14)
        return scroll, widget, layout

    def _create_llm_page(self) -> QWidget:
        scroll, widget, layout = self._create_scroll_page()

        box = QGroupBox("LLM 大模型提供商配置")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(12, 10, 12, 12)
        box_layout.setSpacing(8)

        box_layout.addWidget(QLabel("服务提供商 (Provider):"))
        self.provider_combo = QComboBox()
        _configure_combo_popup(self.provider_combo)
        self.provider_combo.addItem("deepseek", "deepseek")
        self.provider_combo.addItem("openai", "openai")
        self.provider_combo.addItem("gemini", "gemini")
        self.provider_combo.addItem("antigravity (Gemini 反代兼容)", "antigravity")
        self.provider_combo.addItem("custom", "custom")
        cur_provider = str(getattr(self.config.llm, "provider", "deepseek") or "deepseek").strip().lower()
        if "antigravity" in cur_provider:
            cur_provider = "antigravity"
        prov_idx = self.provider_combo.findData(cur_provider)
        if prov_idx >= 0:
            self.provider_combo.setCurrentIndex(prov_idx)
        else:
            self.provider_combo.setCurrentIndex(0)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        box_layout.addWidget(self.provider_combo)

        box_layout.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setText(getattr(self.config.llm, "api_key", ""))
        self.api_key_edit.setPlaceholderText("sk-...")
        box_layout.addWidget(self.api_key_edit)

        box_layout.addWidget(QLabel("模型名称 (Model):"))
        self.model_edit = QLineEdit()
        self.model_edit.setText(getattr(self.config.llm, "model", "deepseek-chat"))
        box_layout.addWidget(self.model_edit)

        box_layout.addWidget(QLabel("自定义端点 URL (可选):"))
        self.endpoint_edit = QLineEdit()
        self.endpoint_edit.setText(getattr(self.config.llm, "endpoint", ""))
        self.endpoint_edit.setPlaceholderText("https://api.example.com/v1")
        box_layout.addWidget(self.endpoint_edit)

        # System Prompt
        prompt_header = QHBoxLayout()
        prompt_header.setContentsMargins(0, 4, 0, 0)
        prompt_header.addWidget(QLabel("模型系统提示词 (System Prompt):"))
        prompt_header.addStretch()

        reset_prompt_btn = QPushButton("🔄 恢复默认提示词")
        reset_prompt_btn.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                padding: 3px 10px;
                border-radius: 5px;
                min-height: 18px;
            }
        """)
        reset_prompt_btn.clicked.connect(lambda: self.sys_prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT))
        prompt_header.addWidget(reset_prompt_btn)
        box_layout.addLayout(prompt_header)

        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setPlaceholderText("输入大模型的系统提示词...")
        self.sys_prompt_edit.setMinimumHeight(96)
        self.sys_prompt_edit.setMaximumHeight(140)
        self.sys_prompt_edit.setPlainText(getattr(self.config.llm, "system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT)
        box_layout.addWidget(self.sys_prompt_edit)

        layout.addWidget(box)

        # Test Connectivity Group
        test_group = QGroupBox("🔗 模型连通性测试")
        test_layout = QVBoxLayout(test_group)
        test_layout.setContentsMargins(12, 10, 12, 12)
        test_layout.setSpacing(8)

        test_layout.addWidget(QLabel("测试文本 (向模型发送以检验连通与翻译):"))
        test_input_row = QHBoxLayout()
        test_input_row.setSpacing(8)
        test_input_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.test_text_input = QLineEdit("Hello! This is a translation connectivity test.")
        self.test_text_input.setFixedHeight(30)
        self.test_text_input.setPlaceholderText("输入测试语句...")
        test_input_row.addWidget(self.test_text_input, 1)

        self.test_conn_btn = QPushButton("🚀 测试连通性")
        self.test_conn_btn.setFixedHeight(30)
        self.test_conn_btn.setMinimumWidth(118)
        self.test_conn_btn.clicked.connect(self._run_connection_test)
        test_input_row.addWidget(self.test_conn_btn)
        test_layout.addLayout(test_input_row)

        self.test_result_label = QLabel("点击“测试连通性”验证当前配置是否可以正常调用并翻译。")
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setStyleSheet("color: #8E8E93; padding: 4px 2px; font-size: 12px;")
        test_layout.addWidget(self.test_result_label)

        layout.addWidget(test_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_ocr_page(self) -> QWidget:
        scroll, widget, layout = self._create_scroll_page()

        box = QGroupBox("OCR 文字识别设置")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(12, 10, 12, 12)
        box_layout.setSpacing(10)

        box_layout.addWidget(QLabel("OCR 引擎后端:"))
        self.ocr_engine_combo = QComboBox()
        _configure_combo_popup(self.ocr_engine_combo)
        self.ocr_engine_combo.addItem("Comic-Text-Detector + 分语种智能识别 (纯 PyTorch 漫画专属，强烈推荐)", "ctd")
        self.ocr_engine_combo.addItem("PaddleOCR 3.x (通用中/日/英，国漫推荐)", "paddle")
        self.ocr_engine_combo.addItem("PaddleOCR + Manga-OCR (解耦模式)", "paddle_manga")
        self.ocr_engine_combo.addItem("EasyOCR (英文/欧美漫推荐)", "easyocr")

        current_eng = getattr(self.config.ocr, "engine", "ctd")
        found_idx = self.ocr_engine_combo.findData(current_eng)
        if found_idx >= 0:
            self.ocr_engine_combo.setCurrentIndex(found_idx)
        elif current_eng == "manga_ocr":
            self.ocr_engine_combo.setCurrentIndex(0)
        else:
            self.ocr_engine_combo.setCurrentText(current_eng)
        box_layout.addWidget(self.ocr_engine_combo)

        hint_label = QLabel("💡 提示：纯 PyTorch 模式结合了 CTD 漫画检测器与分语种高精度识别（日文 Manga-OCR / 英文 PP-OCRv6 / 中文 PP-OCRv5 Server / 韩文 PP-OCRv5），源语言选“自动识别”时将按页面文字自动分流，全 CUDA 加速且无需额外 Paddle GPU 环境。")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #8E8E93; font-size: 11px; padding: 2px 0px;")
        box_layout.addWidget(hint_label)

        self.gpu_cb = QCheckBox("启用 GPU 硬件加速 (CUDA / MPS)")
        self.gpu_cb.setChecked(not getattr(self.config.ocr, "force_cpu", False))
        box_layout.addWidget(self.gpu_cb)

        box_layout.addWidget(QLabel("源语言 (Source Language):"))
        self.source_lang_combo = QComboBox()
        _configure_combo_popup(self.source_lang_combo)
        self.source_lang_combo.addItems(["自动识别", "日语", "韩语", "英语", "繁体中文", "简体中文"])
        current_source = getattr(self.config, "source_lang", "自动识别")
        self.source_lang_combo.setCurrentText(current_source)
        box_layout.addWidget(self.source_lang_combo)

        self.recog_onoma_cb = QCheckBox("识别并翻译背景语气词与拟声词 (Onomatopoeia)")
        self.recog_onoma_cb.setChecked(getattr(self.config.ocr, "recognize_onomatopoeia", False))
        box_layout.addWidget(self.recog_onoma_cb)

        layout.addWidget(box)

        # Dual-Model Ensemble & Fusion Settings (sibling QGroupBox to avoid nested box-in-box clipping)
        ens_group = QGroupBox("双模型增强模式 (Ensemble & Fusion)")
        ens_layout = QVBoxLayout(ens_group)
        ens_layout.setContentsMargins(12, 10, 12, 12)
        ens_layout.setSpacing(8)

        self.ensemble_det_cb = QCheckBox("启用双模型联合找框 (CTD + 通用检测器互补，大幅减少漏识)")
        self.ensemble_det_cb.setChecked(getattr(self.config.ocr, "ensemble_detection", False))
        ens_layout.addWidget(self.ensemble_det_cb)
        det_tip = QLabel("说明：结合气泡检测模型与通用文本检测器，使用 IoU 覆盖度去重，可大幅召回矩形独白框、背景招牌等文字，降低漏翻率。")
        det_tip.setStyleSheet("color: #8E8E93; font-size: 11px; padding-left: 24px; padding-bottom: 4px;")
        det_tip.setWordWrap(True)
        ens_layout.addWidget(det_tip)

        self.ensemble_rec_cb = QCheckBox("启用双模型交叉验证识字 (Manga-OCR + 通用识别器融合)")
        self.ensemble_rec_cb.setChecked(getattr(self.config.ocr, "ensemble_recognition", False))
        ens_layout.addWidget(self.ensemble_rec_cb)
        rec_tip = QLabel("说明：利用通用识别器辅助纠正日文模型在纯英文人名、年龄数字（如 Friend B (20)、Chris）上的假名幻觉，强强互补。")
        rec_tip.setStyleSheet("color: #8E8E93; font-size: 11px; padding-left: 24px;")
        rec_tip.setWordWrap(True)
        ens_layout.addWidget(rec_tip)

        layout.addWidget(ens_group)

        # Per-Language SOTA Recognizers & VLM Fallback
        rec_group = QGroupBox("分语种精准识别与 VLM 兜底 (Per-Language Recognizers)")
        rec_layout = QVBoxLayout(rec_group)
        rec_layout.setContentsMargins(12, 10, 12, 12)
        rec_layout.setSpacing(8)

        rec_layout.addWidget(QLabel("英文识别器 (English Recognizer):"))
        self.en_rec_combo = _build_data_combo([
            ("PP-OCRv6 Medium (PyTorch CUDA，英文对话/CG推荐)", "ppocr_v6"),
            ("OpenOCR SVTRv2 Server (PyTorch CUDA，备选切换)", "svtrv2"),
            ("PP-OCRv5 Server (PyTorch CUDA)", "ppocr_v5_server"),
            ("EasyOCR (旧版兼容)", "easyocr"),
        ], getattr(self.config.ocr, "en_recognizer", "ppocr_v6"))
        rec_layout.addWidget(self.en_rec_combo)

        rec_layout.addWidget(QLabel("中文识别器 (Chinese Recognizer):"))
        self.zh_rec_combo = _build_data_combo([
            ("PP-OCRv5 Server (PyTorch CUDA，简繁中推荐)", "ppocr_v5_server"),
            ("PP-OCRv6 Medium (PyTorch CUDA，中英混排佳)", "ppocr_v6"),
            ("EasyOCR (旧版兼容)", "easyocr"),
        ], getattr(self.config.ocr, "zh_recognizer", "ppocr_v5_server"))
        rec_layout.addWidget(self.zh_rec_combo)

        rec_layout.addWidget(QLabel("日文识别器 (Japanese Recognizer):"))
        self.ja_rec_combo = _build_data_combo([
            ("Manga-OCR (日漫竖排/注音/艺术字强烈推荐)", "manga_ocr"),
            ("PP-OCRv5 Server (PyTorch CUDA)", "ppocr_v5_server"),
            ("EasyOCR (旧版兼容)", "easyocr"),
        ], getattr(self.config.ocr, "ja_recognizer", "manga_ocr"))
        rec_layout.addWidget(self.ja_rec_combo)

        rec_layout.addWidget(QLabel("PaddleOCR 后端版本 (仅选用 Paddle 引擎时生效):"))
        self.paddle_ver_combo = _build_data_combo([
            ("PP-OCRv5 (高精度，不兼容语种自动回退 v3)", "PP-OCRv5"),
            ("PP-OCRv3 (经典轻量 Mobile)", "PP-OCRv3"),
        ], getattr(self.config.ocr, "paddle_ocr_version", "PP-OCRv5"))
        rec_layout.addWidget(self.paddle_ver_combo)

        vlm_row = QHBoxLayout()
        self.vlm_fallback_cb = QCheckBox("启用 VLM 多模态低置信度二次兜底识字 (使用上方已配视觉模型)")
        self.vlm_fallback_cb.setChecked(getattr(self.config.ocr, "vlm_fallback", False))
        vlm_row.addWidget(self.vlm_fallback_cb, 1)
        vlm_row.addWidget(QLabel("触发阈值:"))
        self.vlm_thresh_spin = QDoubleSpinBox()
        self.vlm_thresh_spin.setRange(0.30, 0.95)
        self.vlm_thresh_spin.setSingleStep(0.05)
        self.vlm_thresh_spin.setDecimals(2)
        self.vlm_thresh_spin.setValue(float(getattr(self.config.ocr, "vlm_conf_threshold", 0.72)))
        vlm_row.addWidget(self.vlm_thresh_spin)
        rec_layout.addLayout(vlm_row)

        layout.addWidget(rec_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_inpaint_page(self) -> QWidget:
        scroll, widget, layout = self._create_scroll_page()

        box = QGroupBox("背景文字抹除与修复")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(12, 10, 12, 12)
        box_layout.setSpacing(10)

        box_layout.addWidget(QLabel("抹除引擎 (Inpaint Engine):"))
        self.inpaint_combo = QComboBox()
        _configure_combo_popup(self.inpaint_combo)
        self.inpaint_combo.addItems(["lama", "opencv"])
        self.inpaint_combo.setCurrentText(getattr(self.config.inpaint, "engine", "lama"))
        box_layout.addWidget(self.inpaint_combo)

        box_layout.addWidget(QLabel("OpenCV 备用算法:"))
        self.inpaint_method_combo = QComboBox()
        _configure_combo_popup(self.inpaint_method_combo)
        self.inpaint_method_combo.addItems(["telea", "ns"])
        self.inpaint_method_combo.setCurrentText(getattr(self.config.inpaint, "opencv_method", "telea"))
        box_layout.addWidget(self.inpaint_method_combo)

        self.vram_safe_cb = QCheckBox("显存安全模式 (超大图自适应下采样)")
        self.vram_safe_cb.setChecked(getattr(self.config.inpaint, "vram_safe_downscale", True))
        box_layout.addWidget(self.vram_safe_cb)

        layout.addWidget(box)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_typography_page(self) -> QWidget:
        scroll, widget, layout = self._create_scroll_page()

        # 1. 默认字体与排版
        box_font = QGroupBox("全局默认字体与排版风格")
        box_font_layout = QVBoxLayout(box_font)
        box_font_layout.setContentsMargins(12, 10, 12, 12)
        box_font_layout.setSpacing(10)

        box_font_layout.addWidget(QLabel("默认字体 (默认可爱字体 - 霞鹜文楷):"))
        self.typo_font_combo = QComboBox()
        _configure_combo_popup(self.typo_font_combo)
        for label, _ in FONT_CHOICES:
            self.typo_font_combo.addItem(label)

        current_font = getattr(self.config.style, "font_family", "霞鹜文楷")
        selected_idx = 0
        for i, (label, real_font) in enumerate(FONT_CHOICES):
            if real_font.lower() in current_font.lower() or current_font.lower() in label.lower():
                selected_idx = i
                break
        self.typo_font_combo.setCurrentIndex(selected_idx)
        box_font_layout.addWidget(self.typo_font_combo)

        row_checks = QHBoxLayout()
        row_checks.setSpacing(16)
        self.typo_bold_cb = QCheckBox("粗体 (Bold)")
        self.typo_bold_cb.setChecked(getattr(self.config.style, "font_bold", False))
        self.typo_bold_cb.setStyleSheet("font-weight: 600;")
        row_checks.addWidget(self.typo_bold_cb)

        self.typo_italic_cb = QCheckBox("斜体 (Italic)")
        self.typo_italic_cb.setChecked(getattr(self.config.style, "font_italic", False))
        row_checks.addWidget(self.typo_italic_cb)

        self.typo_auto_fit_cb = QCheckBox("自动适应气泡大小 (二分寻优)")
        self.typo_auto_fit_cb.setChecked(getattr(self.config.style, "auto_fit_font_size", True))
        row_checks.addWidget(self.typo_auto_fit_cb)
        row_checks.addStretch()
        box_font_layout.addLayout(row_checks)

        # Font scale
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("字号缩放比例:"))
        self.typo_scale_lbl = QLabel(f"{getattr(self.config.style, 'font_size_scale', 1.0):.1f}x")
        self.typo_scale_lbl.setStyleSheet("font-weight: 600; color: #3B82F6;")
        scale_row.addWidget(self.typo_scale_lbl)
        scale_row.addStretch()
        box_font_layout.addLayout(scale_row)

        self.typo_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.typo_scale_slider.setRange(5, 30)
        self.typo_scale_slider.setValue(int(round(getattr(self.config.style, "font_size_scale", 1.0) * 10)))
        self.typo_scale_slider.valueChanged.connect(lambda v: self.typo_scale_lbl.setText(f"{v/10.0:.1f}x"))
        box_font_layout.addWidget(self.typo_scale_slider)

        # Line spacing
        spacing_row = QHBoxLayout()
        spacing_row.addWidget(QLabel("行间距比例:"))
        self.typo_spacing_lbl = QLabel(f"{getattr(self.config.style, 'line_spacing', 1.30):.2f}x")
        self.typo_spacing_lbl.setStyleSheet("font-weight: 600; color: #3B82F6;")
        spacing_row.addWidget(self.typo_spacing_lbl)
        spacing_row.addStretch()
        self.typo_auto_spacing_cb = QCheckBox("行间距智能识别 (默认开启)")
        self.typo_auto_spacing_cb.setChecked(getattr(self.config.style, "auto_line_spacing", True))
        spacing_row.addWidget(self.typo_auto_spacing_cb)
        box_font_layout.addLayout(spacing_row)

        self.typo_spacing_slider = QSlider(Qt.Orientation.Horizontal)
        self.typo_spacing_slider.setRange(10, 20)
        self.typo_spacing_slider.setValue(int(round(getattr(self.config.style, "line_spacing", 1.30) * 10)))
        self.typo_spacing_slider.valueChanged.connect(lambda v: self.typo_spacing_lbl.setText(f"{v/10.0:.2f}x"))
        box_font_layout.addWidget(self.typo_spacing_slider)

        layout.addWidget(box_font)

        # 2. 颜色与描边
        box_color = QGroupBox("文字颜色与外边缘描边")
        box_color_layout = QVBoxLayout(box_color)
        box_color_layout.setContentsMargins(12, 10, 12, 12)
        box_color_layout.setSpacing(10)

        # Text Color
        row_txt_color = QHBoxLayout()
        row_txt_color.setSpacing(8)
        row_txt_color.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_tc = QLabel("文字颜色:")
        lbl_tc.setFixedWidth(84)
        row_txt_color.addWidget(lbl_tc)
        self.typo_txt_color_mode = QComboBox()
        self.typo_txt_color_mode.setFixedHeight(30)
        _configure_combo_popup(self.typo_txt_color_mode)
        self.typo_txt_color_mode.addItem("原文提取颜色", TextColorMode.ORIGINAL.value)
        self.typo_txt_color_mode.addItem("自定义纯色", TextColorMode.CUSTOM.value)
        mode_idx = 1 if getattr(self.config.style, "text_color_mode", "original") == TextColorMode.CUSTOM.value else 0
        self.typo_txt_color_mode.setCurrentIndex(mode_idx)
        self.typo_txt_color_mode.currentIndexChanged.connect(self._on_txt_color_mode_changed)
        row_txt_color.addWidget(self.typo_txt_color_mode, 1)

        self._global_custom_text_color = getattr(self.config.style, "custom_text_color", "#000000") or "#000000"
        self.typo_txt_color_btn = QPushButton("选择颜色")
        self.typo_txt_color_btn.setFixedSize(88, 30)
        self.typo_txt_color_btn.setEnabled(mode_idx == 1)
        self._update_global_txt_color_btn()
        self.typo_txt_color_btn.clicked.connect(self._pick_global_text_color)
        row_txt_color.addWidget(self.typo_txt_color_btn)
        box_color_layout.addLayout(row_txt_color)

        # Stroke Mode
        row_stroke = QHBoxLayout()
        row_stroke.setSpacing(8)
        row_stroke.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_sm = QLabel("文字描边:")
        lbl_sm.setFixedWidth(84)
        row_stroke.addWidget(lbl_sm)
        self.typo_stroke_mode = QComboBox()
        self.typo_stroke_mode.setFixedHeight(30)
        _configure_combo_popup(self.typo_stroke_mode)
        self.typo_stroke_mode.addItem("智能对比度 (推荐)", StrokeMode.AUTO.value)
        self.typo_stroke_mode.addItem("自定义描边", StrokeMode.MANUAL.value)
        self.typo_stroke_mode.addItem("关闭描边", StrokeMode.OFF.value)
        cur_stroke = getattr(self.config.style, "stroke_mode", "auto")
        s_idx = 0
        if cur_stroke == StrokeMode.MANUAL.value:
            s_idx = 1
        elif cur_stroke == StrokeMode.OFF.value:
            s_idx = 2
        self.typo_stroke_mode.setCurrentIndex(s_idx)
        self.typo_stroke_mode.currentIndexChanged.connect(self._on_stroke_mode_changed)
        row_stroke.addWidget(self.typo_stroke_mode, 1)
        box_color_layout.addLayout(row_stroke)

        # Stroke Width
        row_sw = QHBoxLayout()
        row_sw.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        row_sw.addWidget(QLabel("描边粗细 (px):"))
        self.typo_stroke_w_lbl = QLabel(f"{getattr(self.config.style, 'stroke_width', 3.5):.1f}px")
        self.typo_stroke_w_lbl.setStyleSheet("font-weight: 600; color: #3B82F6;")
        row_sw.addWidget(self.typo_stroke_w_lbl)
        row_sw.addStretch()
        self.typo_auto_stroke_w_cb = QCheckBox("描边粗细自动识别 (默认开启)")
        self.typo_auto_stroke_w_cb.setChecked(getattr(self.config.style, "auto_stroke_width", True))
        self.typo_auto_stroke_w_cb.setEnabled(s_idx != 2)
        row_sw.addWidget(self.typo_auto_stroke_w_cb)
        box_color_layout.addLayout(row_sw)

        self.typo_stroke_w_slider = QSlider(Qt.Orientation.Horizontal)
        self.typo_stroke_w_slider.setRange(5, 50)
        self.typo_stroke_w_slider.setValue(int(round(getattr(self.config.style, "stroke_width", 3.5) * 10)))
        self.typo_stroke_w_slider.setEnabled(s_idx != 2)
        self.typo_stroke_w_slider.valueChanged.connect(lambda v: self.typo_stroke_w_lbl.setText(f"{v/10.0:.1f}px"))
        box_color_layout.addWidget(self.typo_stroke_w_slider)

        # Bg mode
        row_bg = QHBoxLayout()
        row_bg.setSpacing(8)
        row_bg.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_bg = QLabel("背景气泡覆盖:")
        lbl_bg.setFixedWidth(84)
        row_bg.addWidget(lbl_bg)
        self.typo_bg_mode = QComboBox()
        self.typo_bg_mode.setFixedHeight(30)
        _configure_combo_popup(self.typo_bg_mode)
        self.typo_bg_mode.addItem("原图周边环境自适应", BgColorMode.ORIGINAL.value)
        self.typo_bg_mode.addItem("纯白覆盖", BgColorMode.CUSTOM.value)
        self.typo_bg_mode.addItem("透明无背景仅文字", BgColorMode.NONE.value)
        cur_bg = getattr(self.config.style, "bg_color_mode", "original")
        bg_idx = 0
        if cur_bg == BgColorMode.CUSTOM.value:
            bg_idx = 1
        elif cur_bg == BgColorMode.NONE.value:
            bg_idx = 2
        self.typo_bg_mode.setCurrentIndex(bg_idx)
        row_bg.addWidget(self.typo_bg_mode, 1)
        box_color_layout.addLayout(row_bg)

        # Onomatopoeia strategy
        row_onoma = QHBoxLayout()
        row_onoma.setSpacing(8)
        row_onoma.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_onoma = QLabel("拟声词处理:")
        lbl_onoma.setFixedWidth(84)
        lbl_onoma.setToolTip("语气词与拟声词 (Onomatopoeia) 处理策略")
        row_onoma.addWidget(lbl_onoma)
        self.typo_onoma_combo = QComboBox()
        self.typo_onoma_combo.setFixedHeight(30)
        _configure_combo_popup(self.typo_onoma_combo)
        self.typo_onoma_combo.addItem("正常消除并翻译显示 (默认推荐)", OnomatopoeiaMode.NORMAL.value)
        self.typo_onoma_combo.addItem("半透明艺术保留", OnomatopoeiaMode.TRANSPARENT.value)
        self.typo_onoma_combo.addItem("跳过不处理 (保持原图)", OnomatopoeiaMode.IGNORE.value)
        cur_onoma = getattr(self.config.style, "onomatopoeia_mode", OnomatopoeiaMode.NORMAL.value)
        onoma_idx = 0
        if cur_onoma == OnomatopoeiaMode.TRANSPARENT.value:
            onoma_idx = 1
        elif cur_onoma == OnomatopoeiaMode.IGNORE.value:
            onoma_idx = 2
        self.typo_onoma_combo.setCurrentIndex(onoma_idx)
        row_onoma.addWidget(self.typo_onoma_combo, 1)
        box_color_layout.addLayout(row_onoma)

        layout.addWidget(box_color)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_export_page(self) -> QWidget:
        scroll, widget, layout = self._create_scroll_page()

        box = QGroupBox("导出与磁盘缓存")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(12, 10, 12, 12)
        box_layout.setSpacing(12)

        self.compress_cb = QCheckBox("导出时启用视觉无损压缩 (固定导出为 WebP，大幅缩减输出体积)")
        self.compress_cb.setToolTip(
            "勾选后，批量导出与整话 ZIP 导出将统一固定为 WebP 格式（单页保存弹窗也将默认选中 WebP）；\n"
            "若同时将缓存图像格式设为 WebP，已渲染页面导出可直接复制缓存，无需重复编解码。"
        )
        self.compress_cb.setChecked(getattr(self.config.style, "export_compressed", False))
        box_layout.addWidget(self.compress_cb)

        self.auto_cache_cb = QCheckBox("自动保存中间结果到本地磁盘缓存 (.amt_cache)")
        self.auto_cache_cb.setChecked(getattr(self.config, "auto_save_cache", True))
        box_layout.addWidget(self.auto_cache_cb)

        row_cache_fmt = QHBoxLayout()
        row_cache_fmt.setSpacing(8)
        row_cache_fmt.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        lbl_cache_fmt = QLabel("缓存图像格式:")
        lbl_cache_fmt.setFixedWidth(96)
        row_cache_fmt.addWidget(lbl_cache_fmt)
        self.cache_format_combo = QComboBox()
        self.cache_format_combo.setFixedHeight(30)
        _configure_combo_popup(self.cache_format_combo)
        self.cache_format_combo.addItem("极速无损 BMP (推荐，读写快约 40 倍，体积较大)", "bmp")
        self.cache_format_combo.addItem("高压缩 WebP (极省硬盘空间，CPU 编码较慢)", "webp")
        cur_fmt = str(getattr(self.config, "cache_format", "bmp") or "bmp").lower()
        self.cache_format_combo.setCurrentIndex(1 if cur_fmt == "webp" else 0)
        row_cache_fmt.addWidget(self.cache_format_combo, 1)
        box_layout.addLayout(row_cache_fmt)

        layout.addWidget(box)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_glossary_page(self) -> QWidget:
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QToolBar
        from PyQt6.QtGui import QAction

        scroll, content, layout = self._create_scroll_page()

        group = QGroupBox("专有名词词典 (Glossary)")
        glayout = QVBoxLayout(group)
        glayout.setContentsMargins(12, 10, 12, 12)
        glayout.setSpacing(10)

        toolbar = QToolBar()
        toolbar.setMovable(False)
        add_btn = QAction("➕ 添加词条", self)
        del_btn = QAction("🗑️ 删除选定", self)
        toolbar.addAction(add_btn)
        toolbar.addAction(del_btn)
        glayout.addWidget(toolbar)

        self.glossary_table = QTableWidget(0, 2)
        self.glossary_table.setMinimumHeight(340)
        self.glossary_table.setAlternatingRowColors(True)
        self.glossary_table.verticalHeader().setVisible(False)
        self.glossary_table.verticalHeader().setDefaultSectionSize(32)
        self.glossary_table.setHorizontalHeaderLabels(["原文 (Source)", "译文 (Target)"])
        self.glossary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.glossary_table.horizontalHeader().setMinimumHeight(32)
        glayout.addWidget(self.glossary_table, 1)

        glossary = getattr(self.config, "glossary", {})
        for src, dst in glossary.items():
            row = self.glossary_table.rowCount()
            self.glossary_table.insertRow(row)
            self.glossary_table.setItem(row, 0, QTableWidgetItem(src))
            self.glossary_table.setItem(row, 1, QTableWidgetItem(dst))

        def _add_row():
            row = self.glossary_table.rowCount()
            self.glossary_table.insertRow(row)
            self.glossary_table.setItem(row, 0, QTableWidgetItem(""))
            self.glossary_table.setItem(row, 1, QTableWidgetItem(""))
            self.glossary_table.editItem(self.glossary_table.item(row, 0))

        def _del_row():
            rows = set(item.row() for item in self.glossary_table.selectedItems())
            for row in sorted(rows, reverse=True):
                self.glossary_table.removeRow(row)

        add_btn.triggered.connect(_add_row)
        del_btn.triggered.connect(_del_row)

        layout.addWidget(group, 1)
        scroll.setWidget(content)
        return scroll

    def _on_txt_color_mode_changed(self, idx: int):
        self.typo_txt_color_btn.setEnabled(idx == 1)
        self._update_global_txt_color_btn()

    def _on_stroke_mode_changed(self, idx: int):
        self.typo_stroke_w_slider.setEnabled(idx != 2)
        self.typo_auto_stroke_w_cb.setEnabled(idx != 2)

    def _update_global_txt_color_btn(self):
        fg = _contrast_text_color(self._global_custom_text_color)
        self.typo_txt_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._global_custom_text_color};
                color: {fg};
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

    def _pick_global_text_color(self):
        color = QColorDialog.getColor(QColor(self._global_custom_text_color), self, "选择全局文字颜色")
        if color.isValid():
            self._global_custom_text_color = color.name()
            self._update_global_txt_color_btn()

    def _get_selected_provider(self) -> str:
        data = self.provider_combo.currentData()
        if data:
            return str(data).strip().lower()
        txt = self.provider_combo.currentText().strip().lower()
        if "antigravity" in txt:
            return "antigravity"
        return txt

    def _on_provider_changed(self, _provider_text: str = ""):
        provider = self._get_selected_provider()
        defaults = {
            "deepseek": "deepseek-chat",
            "openai": "gpt-4o",
            "gemini": "gemini-2.5-flash",
            "antigravity": "gemini-2.5-flash",
            "custom": "custom-model"
        }
        self.model_edit.setText(defaults.get(provider, "deepseek-chat"))
        if hasattr(self, "endpoint_edit"):
            if provider == "antigravity":
                self.endpoint_edit.setPlaceholderText("https://your-proxy.com 或 https://your-proxy.com/antigravity/v1beta")
            else:
                self.endpoint_edit.setPlaceholderText("https://api.example.com/v1")

    def _run_connection_test(self):
        provider = self._get_selected_provider()
        api_key = self.api_key_edit.text().strip()
        endpoint = self.endpoint_edit.text().strip()
        default_m = "deepseek-chat" if provider == "deepseek" else ("gemini-2.5-flash" if provider in ("gemini", "antigravity") else "gpt-4o-mini")
        model = self.model_edit.text().strip() or default_m
        system_prompt = self.sys_prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT
        test_text = self.test_text_input.text().strip() or "Hello! This is a translation connectivity test."

        self._retire_test_worker(wait_ms=100)

        self.test_conn_btn.setEnabled(False)
        self.test_conn_btn.setText("⏳ 正在测试...")
        self.test_result_label.setText("正在连接大模型并发送测试文本，请稍候...")
        self.test_result_label.setStyleSheet("color: #0A84FF; padding: 4px; font-size: 12px;")

        self._test_worker = AppModelTestWorker(
            provider=provider,
            api_key=api_key,
            model=model,
            endpoint=endpoint,
            system_prompt=system_prompt,
            test_text=test_text,
            parent=self
        )
        self._test_worker.finished_signal.connect(self._on_test_finished)
        self._test_worker.start()

    def _retire_test_worker(self, wait_ms: int = 100):
        worker = getattr(self, "_test_worker", None)
        if worker is None:
            return
        self._test_worker = None
        from app.core.hardware import retire_qthread
        signals = (worker.finished_signal,) if hasattr(worker, "finished_signal") else ()
        retire_qthread(worker, wait_ms=wait_ms, signals=signals)

    def _on_test_finished(self, success: bool, message: str):
        sender = self.sender()
        if sender is not None and sender is not self._test_worker:
            return
        self.test_conn_btn.setEnabled(True)
        self.test_conn_btn.setText("🚀 测试连通性")
        if success:
            self.test_result_label.setText(f"✅ 连通成功！模型翻译结果：\n“{message}”")
            self.test_result_label.setStyleSheet(
                "color: #34C759; padding: 8px; background-color: rgba(52, 199, 89, 0.12); "
                "border: 1px solid rgba(52, 199, 89, 0.3); border-radius: 6px; font-weight: 500; font-size: 12px;"
            )
        else:
            self.test_result_label.setText(f"❌ 连通失败：\n{message}")
            self.test_result_label.setStyleSheet(
                "color: #FF453A; padding: 8px; background-color: rgba(255, 69, 58, 0.12); "
                "border: 1px solid rgba(255, 69, 58, 0.3); border-radius: 6px; font-size: 12px;"
            )

    def _save_to_config(self):
        # LLM
        self.config.llm.provider = self._get_selected_provider()
        self.config.llm.api_key = self.api_key_edit.text().strip()
        self.config.llm.model = self.model_edit.text().strip()
        self.config.llm.endpoint = self.endpoint_edit.text().strip()
        self.config.llm.system_prompt = self.sys_prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT

        # OCR
        eng_data = self.ocr_engine_combo.currentData()
        self.config.ocr.engine = eng_data if eng_data is not None else self.ocr_engine_combo.currentText()
        self.config.ocr.force_cpu = not self.gpu_cb.isChecked()
        self.config.ocr.ensemble_detection = self.ensemble_det_cb.isChecked()
        self.config.ocr.ensemble_recognition = self.ensemble_rec_cb.isChecked()
        self.config.ocr.recognize_onomatopoeia = self.recog_onoma_cb.isChecked()
        self.config.ocr.en_recognizer = self.en_rec_combo.currentData() or "ppocr_v6"
        self.config.ocr.zh_recognizer = self.zh_rec_combo.currentData() or "ppocr_v5_server"
        self.config.ocr.ja_recognizer = self.ja_rec_combo.currentData() or "manga_ocr"
        self.config.ocr.paddle_ocr_version = self.paddle_ver_combo.currentData() or "PP-OCRv5"
        self.config.ocr.vlm_fallback = self.vlm_fallback_cb.isChecked()
        self.config.ocr.vlm_conf_threshold = round(float(self.vlm_thresh_spin.value()), 2)
        self.config.source_lang = self.source_lang_combo.currentText()
        from app.core.pipeline.utils import source_lang_to_ocr_lang
        self.config.ocr.lang = source_lang_to_ocr_lang(self.config.source_lang)

        # Inpaint
        self.config.inpaint.engine = self.inpaint_combo.currentText()
        self.config.inpaint.opencv_method = self.inpaint_method_combo.currentText()
        self.config.inpaint.vram_safe_downscale = self.vram_safe_cb.isChecked()

        # Typography Style
        _, real_font = FONT_CHOICES[self.typo_font_combo.currentIndex()]
        self.config.style.font_family = real_font
        self.config.style.font_bold = self.typo_bold_cb.isChecked()
        self.config.style.font_italic = self.typo_italic_cb.isChecked()
        self.config.style.auto_fit_font_size = self.typo_auto_fit_cb.isChecked()
        self.config.style.font_size_scale = self.typo_scale_slider.value() / 10.0
        self.config.style.line_spacing = self.typo_spacing_slider.value() / 10.0
        self.config.style.auto_line_spacing = self.typo_auto_spacing_cb.isChecked()
        self.config.style.text_color_mode = self.typo_txt_color_mode.currentData()
        self.config.style.custom_text_color = self._global_custom_text_color
        self.config.style.stroke_mode = self.typo_stroke_mode.currentData()
        self.config.style.stroke_width = self.typo_stroke_w_slider.value() / 10.0
        self.config.style.auto_stroke_width = self.typo_auto_stroke_w_cb.isChecked()
        self.config.style.bg_color_mode = self.typo_bg_mode.currentData()
        self.config.style.onomatopoeia_mode = self.typo_onoma_combo.currentData()

        # Export & cache
        self.config.style.export_compressed = self.compress_cb.isChecked()
        self.config.auto_save_cache = self.auto_cache_cb.isChecked()
        if hasattr(self, "cache_format_combo"):
            self.config.cache_format = self.cache_format_combo.currentData() or "bmp"
            from app.core.cache.cache_manager import get_cache_manager
            get_cache_manager().set_cache_format(self.config.cache_format)

        # Save glossary
        if hasattr(self, 'glossary_table'):
            from app.core.translation import TranslationManager
            glossary = {}
            for row in range(self.glossary_table.rowCount()):
                src_item = self.glossary_table.item(row, 0)
                dst_item = self.glossary_table.item(row, 1)
                if src_item and dst_item:
                    src = src_item.text().strip()
                    dst = dst_item.text().strip()
                    if src and dst:
                        glossary[src] = dst
            self.config.glossary = glossary
            TranslationManager.get_instance().set_glossary(glossary)

        self.config.save("desktop_config.json")

    def _on_save(self):
        self._save_to_config()
        self.re_render_all_requested = False
        self.accept()

    def _on_apply_all(self):
        self._save_to_config()
        self.re_render_all_requested = True
        self.sig_apply_all_pages.emit()
        self.accept()

    def done(self, result: int):
        self._retire_test_worker(wait_ms=150)
        super().done(result)

    def closeEvent(self, event):
        self._retire_test_worker(wait_ms=150)
        super().closeEvent(event)
