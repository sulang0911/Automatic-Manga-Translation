"""
app/ui/widgets/batch_progress_pill.py
Non-blocking Apple HIG floating batch progress pill.
Allows user to inspect/read completed pages while subsequent pages process in the background.
"""
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor

from app.ui.theme.icons import get_icon


class BatchProgressPill(QFrame):
    """
    Floating horizontal status capsule displayed during batch translation.
    """
    sig_cancel_requested = pyqtSignal()
    sig_open_export_dir = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("batchProgressPill")
        self.setFixedHeight(40)
        self.hide()

        self.setStyleSheet("""
            #batchProgressPill {
                background-color: rgba(24, 24, 28, 0.94);
                border: 1px solid #0A84FF;
                border-radius: 20px;
                padding: 4px 14px;
            }
            QLabel {
                color: #ECECEF;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            QProgressBar {
                background-color: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 4px;
                text-align: center;
                height: 6px;
                min-width: 100px;
                max-width: 120px;
            }
            QProgressBar::chunk {
                background-color: #0A84FF;
                border-radius: 4px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Pulse / Activity indicator icon
        self.icon_lbl = QLabel(self)
        self.icon_lbl.setPixmap(get_icon("sparkles", color="#0A84FF", size=14).pixmap(14, 14))
        layout.addWidget(self.icon_lbl)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Step Text (e.g. "第 3/10 页 · 正在翻译")
        self.status_lbl = QLabel("正在准备批处理...", self)
        self.status_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #FFFFFF;")
        layout.addWidget(self.status_lbl)

        # ETA Text
        self.eta_lbl = QLabel("", self)
        self.eta_lbl.setStyleSheet("font-size: 11px; color: #A1A1AA;")
        layout.addWidget(self.eta_lbl)

        # Cancel button
        self.btn_cancel = QPushButton("取消", self)
        self.btn_cancel.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 0.15);
                color: #EF4444;
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 12px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(239, 68, 68, 0.25);
            }
        """)
        self.btn_cancel.clicked.connect(self.sig_cancel_requested.emit)
        layout.addWidget(self.btn_cancel)

    def update_progress(self, current: int, total: int, step_text: str = "", eta_seconds: Optional[int] = None):
        """Updates progress metrics smoothly."""
        total = max(1, total)
        pct = int((current / total) * 100)
        self.progress_bar.setValue(pct)

        step_str = f" · {step_text}" if step_text else ""
        self.status_lbl.setText(f"批处理 ({current}/{total}){step_str}")

        if eta_seconds is not None and eta_seconds > 0:
            m = eta_seconds // 60
            s = eta_seconds % 60
            eta_str = f"剩余 ~{m}分{s}秒" if m > 0 else f"剩余 ~{s}秒"
            self.eta_lbl.setText(eta_str)
        else:
            self.eta_lbl.setText("")

    def show_finished(self, success_count: int, failed_count: int = 0):
        """Displays completion state."""
        self.progress_bar.setValue(100)
        fail_msg = f", {failed_count} 失败" if failed_count > 0 else ""
        self.status_lbl.setText(f"批处理完成 ({success_count} 成功{fail_msg})")
        self.eta_lbl.setText("")
        self.btn_cancel.setText("关闭")
