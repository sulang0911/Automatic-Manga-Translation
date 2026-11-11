"""
app/ui/widgets/in_place_editor.py
Apple macOS popover-inspired floating bubble editor and contextual typography micro-bar.
"""
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QFrame, QToolButton, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QKeyEvent

from app.ui.theme.icons import get_icon


class ContextualTypographyBar(QFrame):
    """
    Floating macOS micro-pill toolbar positioned above a selected speech bubble on canvas.
    Provides 1-click vertical/horizontal direction toggle, font size stepper, and font selection.
    """
    sig_orientation_toggled = pyqtSignal(dict, str)
    sig_font_size_changed = pyqtSignal(dict, int)
    sig_font_family_changed = pyqtSignal(dict, str)
    sig_rerender_requested = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("contextualTypographyBar")
        self.setFixedHeight(34)
        self.block_data: Optional[Dict[str, Any]] = None
        self.hide()

        self.setStyleSheet("""
            #contextualTypographyBar {
                background-color: rgba(28, 28, 32, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 10px;
                padding: 2px 6px;
            }
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 6px;
                color: #F5F5F7;
                padding: 3px 5px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.12);
            }
            QLabel {
                color: #F5F5F7;
                font-size: 11px;
                font-weight: 600;
            }
            QComboBox {
                background-color: rgba(255, 255, 255, 0.08);
                color: #F5F5F7;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                font-size: 11px;
                padding: 1px 6px;
                max-width: 92px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(4)

        # 1. Orientation toggle
        self.btn_orient = QToolButton(self)
        self.btn_orient.setIcon(get_icon("direction_vertical", color="#0A84FF", size=14))
        self.btn_orient.setToolTip("切换横排 / 竖排排版 (快捷键: V)")
        self.btn_orient.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_orient.clicked.connect(self._on_toggle_orientation)
        layout.addWidget(self.btn_orient)

        sep1 = QFrame(self)
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setStyleSheet("color: rgba(255, 255, 255, 0.14);")
        layout.addWidget(sep1)

        # 2. Font size stepper
        self.btn_size_down = QToolButton(self)
        self.btn_size_down.setText("－")
        self.btn_size_down.setToolTip("缩小字号 (快捷键: [ )")
        self.btn_size_down.clicked.connect(lambda: self._step_font_size(-1))
        layout.addWidget(self.btn_size_down)

        self.lbl_size = QLabel("14", self)
        self.lbl_size.setFixedWidth(22)
        self.lbl_size.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_size)

        self.btn_size_up = QToolButton(self)
        self.btn_size_up.setText("＋")
        self.btn_size_up.setToolTip("放大字号 (快捷键: ] )")
        self.btn_size_up.clicked.connect(lambda: self._step_font_size(1))
        layout.addWidget(self.btn_size_up)

        sep2 = QFrame(self)
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: rgba(255, 255, 255, 0.14);")
        layout.addWidget(sep2)

        # 3. Quick font dropdown
        self.font_combo = QComboBox(self)
        self.font_combo.addItems(["默认字体", "思源黑体", "思源宋体", "漫画体", "圆体"])
        self.font_combo.currentTextChanged.connect(self._on_font_changed)
        layout.addWidget(self.font_combo)

        # 4. Rerender action button
        self.btn_rerender = QToolButton(self)
        self.btn_rerender.setIcon(get_icon("sparkles", color="#0A84FF", size=13))
        self.btn_rerender.setToolTip("即时重新渲染此气泡")
        self.btn_rerender.clicked.connect(self._on_rerender)
        layout.addWidget(self.btn_rerender)

    def attach_block(self, block_data: Dict[str, Any]):
        """Attaches to a selected block and initializes its parameters."""
        self.block_data = block_data
        style = block_data.get("style", {}) or {}
        direction = style.get("direction", "vertical")
        icon_name = "direction_vertical" if direction == "vertical" else "direction_horizontal"
        self.btn_orient.setIcon(get_icon(icon_name, color="#0A84FF", size=14))

        f_size = style.get("font_size") or 14
        self.lbl_size.setText(str(int(f_size)))

    def _on_toggle_orientation(self):
        if not self.block_data:
            return
        style = self.block_data.setdefault("style", {})
        curr = style.get("direction", "vertical")
        new_dir = "horizontal" if curr == "vertical" else "vertical"
        style["direction"] = new_dir
        icon_name = "direction_vertical" if new_dir == "vertical" else "direction_horizontal"
        self.btn_orient.setIcon(get_icon(icon_name, color="#0A84FF", size=14))
        self.sig_orientation_toggled.emit(self.block_data, new_dir)

    def _step_font_size(self, delta: int):
        if not self.block_data:
            return
        style = self.block_data.setdefault("style", {})
        curr = int(style.get("font_size", 14) or 14)
        new_size = max(8, min(72, curr + delta))
        style["font_size"] = new_size
        self.lbl_size.setText(str(new_size))
        self.sig_font_size_changed.emit(self.block_data, new_size)

    def _on_font_changed(self, font_name: str):
        if not self.block_data:
            return
        style = self.block_data.setdefault("style", {})
        style["font_family"] = font_name
        self.sig_font_family_changed.emit(self.block_data, font_name)

    def _on_rerender(self):
        if self.block_data:
            self.sig_rerender_requested.emit(self.block_data)


class InPlaceBubbleEditor(QFrame):
    """
    Floating macOS popover card near double-clicked bubble on canvas.
    Supports in-place editing, Ctrl+Enter commit and next bubble jump, and Esc cancel.
    """
    sig_commit_and_next = pyqtSignal(dict, str)
    sig_commit = pyqtSignal(dict, str)
    sig_closed = pyqtSignal()
    sig_retranslate_block = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("inPlaceBubbleEditor")
        self.setFixedWidth(320)
        self.block_data: Optional[Dict[str, Any]] = None
        self.hide()

        self.setStyleSheet("""
            #inPlaceBubbleEditor {
                background-color: rgba(28, 28, 32, 0.96);
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 12px;
                padding: 10px 12px;
            }
            QLabel {
                color: #98989D;
                font-size: 11px;
            }
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.06);
                color: #F5F5F7;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                font-size: 13px;
                padding: 6px 8px;
            }
            QTextEdit:focus {
                border-color: #0A84FF;
                background-color: rgba(255, 255, 255, 0.09);
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # Header Row
        header = QHBoxLayout()
        self.title_lbl = QLabel("气泡编辑", self)
        self.title_lbl.setStyleSheet("color: #F5F5F7; font-weight: 600; font-size: 12px;")
        header.addWidget(self.title_lbl)

        header.addStretch()

        self.btn_close = QToolButton(self)
        self.btn_close.setIcon(get_icon("close", color="#98989D", size=11))
        self.btn_close.setStyleSheet("background: transparent; border: none; border-radius: 5px; padding: 3px;")
        self.btn_close.clicked.connect(self.close_editor)
        header.addWidget(self.btn_close)
        layout.addLayout(header)

        # OCR Original text (readonly preview)
        self.ocr_lbl = QLabel("原文: (无)", self)
        self.ocr_lbl.setStyleSheet("color: #8E8E93; font-size: 11px;")
        self.ocr_lbl.setWordWrap(True)
        self.ocr_lbl.setMaximumHeight(38)
        layout.addWidget(self.ocr_lbl)

        # Translation Text Edit
        self.text_edit = QTextEdit(self)
        self.text_edit.setFixedHeight(68)
        self.text_edit.setPlaceholderText("输入译文内容...")
        layout.addWidget(self.text_edit)

        # Action Buttons Row
        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        hint_lbl = QLabel("Ctrl+Enter 下一条", self)
        hint_lbl.setStyleSheet("color: #636366; font-size: 11px;")
        action_row.addWidget(hint_lbl, 1)

        self.btn_commit = QPushButton("应用修改", self)
        self.btn_commit.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_commit.setStyleSheet("""
            QPushButton {
                background-color: #0A84FF;
                color: #FFFFFF;
                border: none;
                border-radius: 7px;
                padding: 5px 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0071E3;
            }
        """)
        self.btn_commit.clicked.connect(self._on_commit_and_next)
        action_row.addWidget(self.btn_commit)

        layout.addLayout(action_row)

    def attach_block(self, block_data: Dict[str, Any], block_idx: int = 1):
        """Loads bubble block data into editor and focuses input."""
        self.block_data = block_data
        self.title_lbl.setText(f"气泡 #{block_idx:02d}")
        ocr_text = block_data.get("original_text") or block_data.get("raw_text") or block_data.get("text") or "(无)"
        self.ocr_lbl.setText(f"原文: {ocr_text}")

        trans_text = block_data.get("translated_text") or block_data.get("translation") or ""
        self.text_edit.setPlainText(trans_text)
        self.text_edit.selectAll()
        self.text_edit.setFocus()

    def _on_commit_and_next(self):
        if not self.block_data:
            return
        new_text = self.text_edit.toPlainText().strip()
        self.block_data["translation"] = new_text
        self.block_data["translated_text"] = new_text
        self.sig_commit_and_next.emit(self.block_data, new_text)

    def close_editor(self):
        self.hide()
        self.sig_closed.emit()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier or not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self._on_commit_and_next()
                return
        elif event.key() == Qt.Key.Key_Escape:
            self.close_editor()
            return
        super().keyPressEvent(event)
