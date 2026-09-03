"""
app/ui/settings/settings_dialog.py
Apple HIG System Preferences Modal Dialog.
Supports configuring LLM Providers, OCR Engines, Inpainting Models, and Typography styles.
"""
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QSlider, QCheckBox,
    QStackedWidget, QListWidget, QListWidgetItem, QGroupBox, QFileDialog, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal

from app.core.config import AppConfig, DEFAULT_SYSTEM_PROMPT
from desktop.core.translation_engine import TranslationEngine


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

    def run(self):
        try:
            engine = TranslationEngine(
                provider=self.provider,
                api_key=self.api_key,
                model=self.model,
                custom_endpoint=self.endpoint,
                system_prompt=self.system_prompt,
                target_lang="简体中文"
            )
            result = engine.test_connection(self.test_text)
            self.finished_signal.emit(True, result)
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class SettingsDialog(QDialog):
    """
    Apple HIG System Settings Dialog with categorized navigation list on the left
    and tabbed preference panels on the right.
    """

    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.setWindowTitle("系统偏好设置")
        self.resize(720, 640)
        self.setMinimumSize(680, 560)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left Category Navigation List
        self.nav_list = QListWidget(self)
        self.nav_list.setObjectName("settingsNavList")
        self.nav_list.setFixedWidth(160)

        categories = [
            ("🤖 AI 翻译大模型", 0),
            ("🔍 OCR 文字识别", 1),
            ("🧹 图像背景修复", 2),
            ("🎨 排版与导出", 3),
        ]
        for name, _ in categories:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(140, 38))
            self.nav_list.addItem(item)

        main_layout.addWidget(self.nav_list)

        # Right Stacked Pages
        right_container = QVBoxLayout()
        self.stack = QStackedWidget(self)
        right_container.addWidget(self.stack)

        self.stack.addWidget(self._create_llm_page())
        self.stack.addWidget(self._create_ocr_page())
        self.stack.addWidget(self._create_inpaint_page())
        self.stack.addWidget(self._create_export_page())

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("取消", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.save_btn = QPushButton("保存配置", self)
        self.save_btn.setProperty("class", "primaryBtn")
        self.save_btn.clicked.connect(self._on_save)
        btn_layout.addWidget(self.save_btn)

        right_container.addLayout(btn_layout)
        main_layout.addLayout(right_container)

    def _create_llm_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(12)

        box = QGroupBox("LLM 大模型提供商配置")
        box_layout = QVBoxLayout(box)

        box_layout.addWidget(QLabel("服务提供商 (Provider):"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["deepseek", "openai", "gemini", "custom"])
        self.provider_combo.setCurrentText(getattr(self.config.llm, "provider", "deepseek"))
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
        prompt_header.addWidget(QLabel("模型系统提示词 (System Prompt):"))
        prompt_header.addStretch()

        reset_prompt_btn = QPushButton("🔄 恢复默认提示词")
        reset_prompt_btn.setStyleSheet("font-size: 11px; padding: 2px 8px; border-radius: 4px;")
        reset_prompt_btn.clicked.connect(lambda: self.sys_prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT))
        prompt_header.addWidget(reset_prompt_btn)
        box_layout.addLayout(prompt_header)

        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setPlaceholderText("输入大模型的系统提示词...")
        self.sys_prompt_edit.setMinimumHeight(85)
        self.sys_prompt_edit.setMaximumHeight(130)
        self.sys_prompt_edit.setPlainText(getattr(self.config.llm, "system_prompt", DEFAULT_SYSTEM_PROMPT) or DEFAULT_SYSTEM_PROMPT)
        box_layout.addWidget(self.sys_prompt_edit)

        layout.addWidget(box)

        # Test Connectivity Group
        test_group = QGroupBox("🔗 模型连通性测试")
        test_layout = QVBoxLayout(test_group)
        test_layout.setSpacing(8)

        test_layout.addWidget(QLabel("测试文本 (向模型发送以检验连通与翻译):"))
        test_input_row = QHBoxLayout()
        self.test_text_input = QLineEdit("Hello! This is a translation connectivity test.")
        self.test_text_input.setPlaceholderText("输入测试语句...")
        test_input_row.addWidget(self.test_text_input)

        self.test_conn_btn = QPushButton("🚀 测试连通性")
        self.test_conn_btn.setFixedWidth(110)
        self.test_conn_btn.clicked.connect(self._run_connection_test)
        test_input_row.addWidget(self.test_conn_btn)
        test_layout.addLayout(test_input_row)

        self.test_result_label = QLabel("点击“测试连通性”验证当前配置是否可以正常调用并翻译。")
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setStyleSheet("color: rgba(255,255,255,0.6); padding: 4px; font-size: 12px;")
        test_layout.addWidget(self.test_result_label)

        layout.addWidget(test_group)
        layout.addStretch()

        scroll.setWidget(widget)
        return scroll

    def _create_ocr_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        box = QGroupBox("OCR 文字识别设置")
        box_layout = QVBoxLayout(box)

        box_layout.addWidget(QLabel("OCR 引擎后端:"))
        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItems(["easyocr", "paddle"])
        self.ocr_engine_combo.setCurrentText(getattr(self.config.ocr, "engine", "easyocr"))
        box_layout.addWidget(self.ocr_engine_combo)

        self.gpu_cb = QCheckBox("启用 GPU 硬件加速 (CUDA / MPS)")
        self.gpu_cb.setChecked(not getattr(self.config.ocr, "force_cpu", False))
        box_layout.addWidget(self.gpu_cb)

        box_layout.addWidget(QLabel("源语言 (Source Language):"))
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["自动识别", "日语", "韩语", "英语", "繁体中文", "简体中文"])
        current_source = getattr(self.config, "source_lang", "自动识别")
        self.source_lang_combo.setCurrentText(current_source)
        box_layout.addWidget(self.source_lang_combo)

        layout.addWidget(box)
        layout.addStretch()
        return widget

    def _create_inpaint_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        box = QGroupBox("背景文字抹除与修复")
        box_layout = QVBoxLayout(box)

        box_layout.addWidget(QLabel("抹除引擎 (Inpaint Engine):"))
        self.inpaint_combo = QComboBox()
        self.inpaint_combo.addItems(["lama", "opencv"])
        self.inpaint_combo.setCurrentText(getattr(self.config.inpaint, "engine", "lama"))
        box_layout.addWidget(self.inpaint_combo)

        box_layout.addWidget(QLabel("OpenCV 备用算法:"))
        self.inpaint_method_combo = QComboBox()
        self.inpaint_method_combo.addItems(["telea", "ns"])
        self.inpaint_method_combo.setCurrentText(getattr(self.config.inpaint, "opencv_method", "telea"))
        box_layout.addWidget(self.inpaint_method_combo)

        self.vram_safe_cb = QCheckBox("显存安全模式 (超大图自适应下采样)")
        self.vram_safe_cb.setChecked(getattr(self.config.inpaint, "vram_safe_downscale", True))
        box_layout.addWidget(self.vram_safe_cb)

        layout.addWidget(box)
        layout.addStretch()
        return widget

    def _create_export_page(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)

        # Typography Font Selection
        box_font = QGroupBox("漫画排版默认字体")
        box_font_layout = QVBoxLayout(box_font)
        box_font_layout.addWidget(QLabel("全局默认漫画字体 (可在右侧检查器单块覆盖):"))
        self.default_font_combo = QComboBox()
        self.default_font_combo.addItems([
            "霞鹜文楷 (日漫萌系)",
            "幼圆 (圆润可爱)",
            "得意黑 (潮流漫画)",
            "Comic Sans MS (卡通英文)",
            "Ink Free (随性手绘)",
            "Segoe Print (手写涂鸦)",
            "楷体 (清秀书法)",
            "Microsoft YaHei",
            "SimHei",
        ])
        current_default_font = getattr(self.config.style, "font_family", "霞鹜文楷")
        for i in range(self.default_font_combo.count()):
            if current_default_font.lower() in self.default_font_combo.itemText(i).lower():
                self.default_font_combo.setCurrentIndex(i)
                break
        box_font_layout.addWidget(self.default_font_combo)
        layout.addWidget(box_font)

        box = QGroupBox("导出与缓存选项")
        box_layout = QVBoxLayout(box)

        self.compress_cb = QCheckBox("导出时启用视觉无损压缩 (大幅缩减体积)")
        self.compress_cb.setChecked(getattr(self.config.style, "export_compressed", False))
        box_layout.addWidget(self.compress_cb)

        self.auto_cache_cb = QCheckBox("自动保存中间结果到本地磁盘缓存")
        self.auto_cache_cb.setChecked(getattr(self.config, "auto_save_cache", True))
        box_layout.addWidget(self.auto_cache_cb)

        layout.addWidget(box)
        layout.addStretch()
        return widget

    def _on_provider_changed(self, provider: str):
        defaults = {
            "deepseek": "deepseek-chat",
            "openai": "gpt-4o",
            "gemini": "gemini-2.5-flash",
            "custom": "custom-model"
        }
        self.model_edit.setText(defaults.get(provider, "deepseek-chat"))

    def _run_connection_test(self):
        provider = self.provider_combo.currentText().strip()
        api_key = self.api_key_edit.text().strip()
        endpoint = self.endpoint_edit.text().strip()
        model = self.model_edit.text().strip() or ("deepseek-chat" if provider == "deepseek" else "gpt-4o-mini")
        system_prompt = self.sys_prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT
        test_text = self.test_text_input.text().strip() or "Hello! This is a translation connectivity test."

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

    def _on_test_finished(self, success: bool, message: str):
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

    def _on_save(self):
        # Update config attributes
        self.config.llm.provider = self.provider_combo.currentText()
        self.config.llm.api_key = self.api_key_edit.text().strip()
        self.config.llm.model = self.model_edit.text().strip()
        self.config.llm.endpoint = self.endpoint_edit.text().strip()
        self.config.llm.system_prompt = self.sys_prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT

        self.config.ocr.engine = self.ocr_engine_combo.currentText()
        self.config.ocr.force_cpu = not self.gpu_cb.isChecked()
        self.config.source_lang = self.source_lang_combo.currentText()

        self.config.inpaint.engine = self.inpaint_combo.currentText()
        self.config.inpaint.opencv_method = self.inpaint_method_combo.currentText()
        self.config.inpaint.vram_safe_downscale = self.vram_safe_cb.isChecked()

        clean_font = self.default_font_combo.currentText().split("(")[0].split("（")[0].strip()
        self.config.style.font_family = clean_font
        self.config.style.export_compressed = self.compress_cb.isChecked()
        self.config.auto_save_cache = self.auto_cache_cb.isChecked()

        self.config.save("desktop_config.json")
        self.accept()
