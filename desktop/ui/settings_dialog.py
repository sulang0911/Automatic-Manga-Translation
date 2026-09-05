from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QComboBox, QSlider, QCheckBox,
    QFrame, QStackedWidget, QListWidget, QListWidgetItem,
    QSpinBox, QFileDialog, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from ..core.config_manager import ConfigManager, DEFAULT_SYSTEM_PROMPT
from ..core.translation_engine import TranslationEngine


class ModelTestWorker(QThread):
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, provider: str, api_key: str, model: str, custom_endpoint: str,
                 system_prompt: str, test_text: str, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.custom_endpoint = custom_endpoint
        self.system_prompt = system_prompt
        self.test_text = test_text

    def run(self):
        try:
            engine = TranslationEngine(
                provider=self.provider,
                api_key=self.api_key,
                model=self.model,
                custom_endpoint=self.custom_endpoint,
                system_prompt=self.system_prompt,
                target_lang="简体中文"
            )
            result = engine.test_connection(self.test_text)
            self.finished_signal.emit(True, result)
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class SettingsDialog(QDialog):
    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setWindowTitle("系统偏好设置")
        self.resize(720, 640)
        self.setMinimumSize(680, 560)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Left Category Navigation List (Apple Settings style)
        self.nav_list = QListWidget(self)
        self.nav_list.setFixedWidth(160)
        self.nav_list.setStyleSheet("""
            QListWidget {
                background-color: #1A1A1C;
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 10px;
                padding: 4px;
            }
            QListWidget::item {
                border-radius: 8px;
                padding: 10px 12px;
                font-weight: 500;
            }
            QListWidget::item:selected {
                background-color: #0A84FF;
                color: #FFFFFF;
            }
        """)

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
        self.stack.addWidget(self._create_typo_page())

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        # Bottom Action Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()

        cancel_btn = QPushButton("取消", self)
        cancel_btn.clicked.connect(self.reject)
        bottom_bar.addWidget(cancel_btn)

        save_btn = QPushButton("保存配置", self)
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save_and_close)
        bottom_bar.addWidget(save_btn)

        right_container.addLayout(bottom_bar)
        main_layout.addLayout(right_container)

        self._load_values()

    def _create_llm_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(10)

        title = QLabel("AI 翻译服务与端点配置")
        title.setObjectName("headingLabel")
        layout.addWidget(title)

        # Provider
        layout.addWidget(QLabel("服务提供商:"))
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(["DeepSeek", "OpenAI", "Google Gemini", "自定义 API (OpenAI 兼容)"])
        layout.addWidget(self.llm_provider_combo)

        # API Key
        layout.addWidget(QLabel("API Key (令牌):"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        layout.addWidget(self.api_key_input)

        # Custom Endpoint / Base URL
        layout.addWidget(QLabel("API 基础端点 (Base URL):"))
        self.endpoint_input = QLineEdit()
        self.endpoint_input.setPlaceholderText("https://api.deepseek.com/v1")
        layout.addWidget(self.endpoint_input)

        # Model Name
        layout.addWidget(QLabel("模型名称 (Model):"))
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("deepseek-chat / gpt-4o-mini")
        layout.addWidget(self.model_input)

        # System Prompt Header with Reset Button
        prompt_header = QHBoxLayout()
        prompt_label = QLabel("模型系统提示词 (System Prompt):")
        prompt_header.addWidget(prompt_label)
        prompt_header.addStretch()

        reset_prompt_btn = QPushButton("🔄 恢复默认提示词")
        reset_prompt_btn.setStyleSheet("font-size: 11px; padding: 2px 8px; border-radius: 4px;")
        reset_prompt_btn.clicked.connect(lambda: self.sys_prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT))
        prompt_header.addWidget(reset_prompt_btn)
        layout.addLayout(prompt_header)

        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setPlaceholderText("输入大模型的系统提示词...")
        self.sys_prompt_edit.setMinimumHeight(85)
        self.sys_prompt_edit.setMaximumHeight(130)
        layout.addWidget(self.sys_prompt_edit)

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

        scroll.setWidget(page)
        return scroll

    def _create_ocr_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        title = QLabel("本地 OCR 图像文字识别引擎")
        title.setObjectName("headingLabel")
        layout.addWidget(title)

        # Engine Choice
        layout.addWidget(QLabel("OCR 引擎核心:"))
        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItems([
            "Comic-Text-Detector + Manga-OCR (纯 PyTorch 漫画专属，强烈推荐)",
            "PaddleOCR 3.x (通用中/日/英，国漫推荐)",
            "PaddleOCR + Manga-OCR (解耦模式)",
            "EasyOCR (PyTorch 通用 GPU 兼容)",
            "PaddleOCR 强制 CPU 模式"
        ])
        layout.addWidget(self.ocr_engine_combo)

        # GPU Acceleration
        self.gpu_cb = QCheckBox("启用 NVIDIA GPU 硬件加速 (CUDA)")
        layout.addWidget(self.gpu_cb)

        # Source Language
        layout.addWidget(QLabel("识别源语言排版:"))
        self.ocr_lang_combo = QComboBox()
        self.ocr_lang_combo.addItems(["日语 (japan - 竖排/横排深度对齐)", "简体中文 (ch)", "英语 (en)", "韩语 (korean)"])
        layout.addWidget(self.ocr_lang_combo)

        # Dual-Model Ensemble & Fusion Settings
        ens_group = QGroupBox("双模型增强模式 (Ensemble & Fusion)")
        ens_layout = QVBoxLayout(ens_group)

        self.ensemble_det_cb = QCheckBox("启用双模型联合找框 (CTD + 通用检测器互补，大幅减少漏识)")
        ens_layout.addWidget(self.ensemble_det_cb)
        det_tip = QLabel("说明：结合气泡检测模型与通用文本检测器，使用 IoU 覆盖度去重，可大幅召回矩形独白框、背景招牌等文字，降低漏翻率。")
        det_tip.setStyleSheet("color: #8E8E93; font-size: 11px; margin-left: 20px; margin-bottom: 6px;")
        det_tip.setWordWrap(True)
        ens_layout.addWidget(det_tip)

        self.ensemble_rec_cb = QCheckBox("启用双模型交叉验证识字 (Manga-OCR + 通用识别器融合)")
        ens_layout.addWidget(self.ensemble_rec_cb)
        rec_tip = QLabel("说明：利用通用识别器辅助纠正日文模型在纯英文人名、年龄数字（如 Friend B (20)、Chris）上的假名幻觉，强强互补。")
        rec_tip.setStyleSheet("color: #8E8E93; font-size: 11px; margin-left: 20px;")
        rec_tip.setWordWrap(True)
        ens_layout.addWidget(rec_tip)

        layout.addWidget(ens_group)

        layout.addStretch()
        return page

    def _create_inpaint_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        title = QLabel("图像背景消除与高保真修复")
        title.setObjectName("headingLabel")
        layout.addWidget(title)

        # Engine Choice
        layout.addWidget(QLabel("修复算法引擎:"))
        self.inpaint_engine_combo = QComboBox()
        self.inpaint_engine_combo.addItems([
            "自动选择 (优先 LaMa 深度学习，回退 OpenCV)",
            "OpenCV 智能羽化算法 (Telea - 极速)",
            "OpenCV Navier-Stokes 纹理过渡"
        ])
        layout.addWidget(self.inpaint_engine_combo)

        # Dilation parameters
        dil_box = QGroupBox("智能掩码膨胀与羽化参数")
        dil_layout = QVBoxLayout(dil_box)

        b_row = QHBoxLayout()
        b_row.addWidget(QLabel("对话气泡膨胀像素 (Bubble Dilation):"))
        self.bubble_dil_spin = QSpinBox()
        self.bubble_dil_spin.setRange(1, 20)
        b_row.addWidget(self.bubble_dil_spin)
        dil_layout.addLayout(b_row)

        o_row = QHBoxLayout()
        o_row.addWidget(QLabel("拟声词膨胀像素 (Onomatopoeia Dilation):"))
        self.onoma_dil_spin = QSpinBox()
        self.onoma_dil_spin.setRange(1, 30)
        o_row.addWidget(self.onoma_dil_spin)
        dil_layout.addLayout(o_row)

        f_row = QHBoxLayout()
        f_row.addWidget(QLabel("羽化融合半径 (Feather Radius):"))
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(1, 20)
        f_row.addWidget(self.feather_spin)
        dil_layout.addLayout(f_row)

        layout.addWidget(dil_box)
        layout.addStretch()
        return page

    def _create_typo_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        title = QLabel("排版与导出设置")
        title.setObjectName("headingLabel")
        layout.addWidget(title)

        # Onomatopoeia strategy
        layout.addWidget(QLabel("拟声词处理策略:"))
        self.onoma_mode_combo = QComboBox()
        self.onoma_mode_combo.addItems(["正常消除并翻译 (Normal)", "半透明艺术保留 (Transparent)", "跳过不处理 (Ignore)"])
        layout.addWidget(self.onoma_mode_combo)

        # Default export directory
        layout.addWidget(QLabel("默认导出文件夹:"))
        export_row = QHBoxLayout()
        self.export_dir_input = QLineEdit()
        self.export_dir_input.setPlaceholderText("留空则保存至原图所在目录")
        export_row.addWidget(self.export_dir_input)

        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_export_dir)
        export_row.addWidget(browse_btn)
        layout.addLayout(export_row)

        layout.addStretch()
        return page

    def _browse_export_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认导出目录")
        if dir_path:
            self.export_dir_input.setText(dir_path)

    def _load_values(self):
        cfg = self.config_manager.data
        # LLM
        prov = cfg.get("provider", "deepseek")
        if prov == "openai": self.llm_provider_combo.setCurrentIndex(1)
        elif prov == "gemini": self.llm_provider_combo.setCurrentIndex(2)
        elif prov == "custom": self.llm_provider_combo.setCurrentIndex(3)
        else: self.llm_provider_combo.setCurrentIndex(0)

        self.api_key_input.setText(cfg.get("api_key", ""))
        self.endpoint_input.setText(cfg.get("custom_endpoint", ""))
        self.model_input.setText(cfg.get("model", "deepseek-chat"))
        self.sys_prompt_edit.setPlainText(cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)

        # OCR
        ocr_eng = cfg.get("ocr_engine", "ctd")
        if ocr_eng in ("ctd", "manga_ocr", "pure_pytorch"): self.ocr_engine_combo.setCurrentIndex(0)
        elif ocr_eng == "paddle": self.ocr_engine_combo.setCurrentIndex(1)
        elif ocr_eng == "paddle_manga": self.ocr_engine_combo.setCurrentIndex(2)
        elif ocr_eng == "easyocr": self.ocr_engine_combo.setCurrentIndex(3)
        elif ocr_eng == "cpu_paddle": self.ocr_engine_combo.setCurrentIndex(4)
        else: self.ocr_engine_combo.setCurrentIndex(0)

        self.gpu_cb.setChecked(cfg.get("use_gpu", True))
        lang_map = ["japan", "ch", "en", "korean"]
        cur_lang = cfg.get("ocr_lang", "japan")
        if cur_lang in lang_map:
            self.ocr_lang_combo.setCurrentIndex(lang_map.index(cur_lang))
        self.ensemble_det_cb.setChecked(bool(cfg.get("ocr_ensemble_detection", False)))
        self.ensemble_rec_cb.setChecked(bool(cfg.get("ocr_ensemble_recognition", False)))
        
        # Inpaint
        inpaint_eng = cfg.get("inpaint_engine", "auto")
        if inpaint_eng == "opencv_telea": self.inpaint_engine_combo.setCurrentIndex(1)
        elif inpaint_eng == "opencv_ns": self.inpaint_engine_combo.setCurrentIndex(2)
        else: self.inpaint_engine_combo.setCurrentIndex(0)

        self.bubble_dil_spin.setValue(cfg.get("bubble_dilation", 3))
        self.onoma_dil_spin.setValue(cfg.get("onomatopoeia_dilation", 6))
        self.feather_spin.setValue(cfg.get("feather_radius", 4))

        # Typography & Onomatopoeia
        onoma_mode = cfg.get("onomatopoeia_mode", "normal")
        if onoma_mode == "transparent":
            self.onoma_mode_combo.setCurrentIndex(1)
        elif onoma_mode == "ignore":
            self.onoma_mode_combo.setCurrentIndex(2)
        else:
            self.onoma_mode_combo.setCurrentIndex(0)
        self.export_dir_input.setText(cfg.get("recent_export_dir", ""))

    def _run_connection_test(self):
        prov_map = ["deepseek", "openai", "gemini", "custom"]
        provider = prov_map[self.llm_provider_combo.currentIndex()]
        api_key = self.api_key_input.text().strip()
        custom_endpoint = self.endpoint_input.text().strip()
        model = self.model_input.text().strip() or ("deepseek-chat" if provider == "deepseek" else "gpt-4o-mini")
        system_prompt = self.sys_prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT
        test_text = self.test_text_input.text().strip() or "Hello! This is a translation connectivity test."

        self.test_conn_btn.setEnabled(False)
        self.test_conn_btn.setText("⏳ 正在测试...")
        self.test_result_label.setText("正在连接大模型并发送测试文本，请稍候...")
        self.test_result_label.setStyleSheet("color: #0A84FF; padding: 4px; font-size: 12px;")

        self._test_worker = ModelTestWorker(
            provider=provider,
            api_key=api_key,
            model=model,
            custom_endpoint=custom_endpoint,
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

    def _save_and_close(self):
        prov_map = ["deepseek", "openai", "gemini", "custom"]
        ocr_map = ["ctd", "paddle", "paddle_manga", "easyocr", "cpu_paddle"]
        inpaint_map = ["auto", "opencv_telea", "opencv_ns"]
        onoma_map = ["normal", "transparent", "ignore"]
        lang_map = ["japan", "ch", "en", "korean"]
        ocr_lang_val = lang_map[self.ocr_lang_combo.currentIndex()] if self.ocr_lang_combo.currentIndex() < len(lang_map) else "japan"

        updates = {
            "provider": prov_map[self.llm_provider_combo.currentIndex()],
            "api_key": self.api_key_input.text().strip(),
            "custom_endpoint": self.endpoint_input.text().strip(),
            "model": self.model_input.text().strip() or "deepseek-chat",
            "system_prompt": self.sys_prompt_edit.toPlainText().strip() or DEFAULT_SYSTEM_PROMPT,
            "ocr_engine": ocr_map[self.ocr_engine_combo.currentIndex()],
            "ocr_lang": ocr_lang_val,
            "ocr_ensemble_detection": self.ensemble_det_cb.isChecked(),
            "ocr_ensemble_recognition": self.ensemble_rec_cb.isChecked(),
            "use_gpu": self.gpu_cb.isChecked(),
            "inpaint_engine": inpaint_map[self.inpaint_engine_combo.currentIndex()],
            "bubble_dilation": self.bubble_dil_spin.value(),
            "onomatopoeia_dilation": self.onoma_dil_spin.value(),
            "feather_radius": self.feather_spin.value(),
            "onomatopoeia_mode": onoma_map[self.onoma_mode_combo.currentIndex()],
            "recent_export_dir": self.export_dir_input.text().strip()
        }
        self.config_manager.update(updates)
        self.accept()
