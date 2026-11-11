"""
app/ui/widgets/empty_state.py
Apple macOS Sonoma-inspired empty-state welcome placeholder.
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor

from app.ui.theme.icons import render_svg_pixmap, get_icon


class EmptyStateWidget(QWidget):
    """
    Centered macOS welcome placeholder for the central canvas when no pages are loaded.
    """
    sig_open_folder_clicked = pyqtSignal()
    sig_open_files_clicked = pyqtSignal()
    sig_open_shortcuts_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("emptyStateWidget")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)

        card = QFrame(self)
        card.setObjectName("emptyStateCard")
        card.setStyleSheet("""
            #emptyStateCard {
                background-color: rgba(32, 32, 36, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 36px 44px;
                max-width: 480px;
            }
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(12)

        icon_lbl = QLabel(card)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = render_svg_pixmap("sparkles", color="#0A84FF", size=44)
        icon_lbl.setPixmap(pix)
        card_layout.addWidget(icon_lbl)

        title_lbl = QLabel("AetherLens 漫画翻译工作台", card)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet("color: #F5F5F7; font-size: 18px; font-weight: 600; letter-spacing: 0.3px;")
        card_layout.addWidget(title_lbl)

        sub_lbl = QLabel("将漫画图片、章节文件夹或 ZIP 压缩包拖入窗口\n即可自动检测气泡、擦除背景并完成排版翻译", card)
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_lbl.setStyleSheet("color: #98989D; font-size: 13px; line-height: 1.5;")
        card_layout.addWidget(sub_lbl)

        card_layout.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_folder = QPushButton("打开章节文件夹", card)
        self.btn_folder.setIcon(get_icon("folder_open", color="#FFFFFF", size=14))
        self.btn_folder.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_folder.setStyleSheet("""
            QPushButton {
                background-color: #0A84FF;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 7px 16px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0071E3;
            }
        """)
        self.btn_folder.clicked.connect(self.sig_open_folder_clicked.emit)
        btn_row.addWidget(self.btn_folder)

        self.btn_files = QPushButton("导入漫画图片", card)
        self.btn_files.setIcon(get_icon("plus", color="#F5F5F7", size=13))
        self.btn_files.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_files.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                color: #F5F5F7;
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 8px;
                padding: 7px 16px;
                font-size: 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.14);
            }
        """)
        self.btn_files.clicked.connect(self.sig_open_files_clicked.emit)
        btn_row.addWidget(self.btn_files)

        card_layout.addLayout(btn_row)
        card_layout.addSpacing(6)

        tips_lbl = QLabel("按 ? 查看快捷键  ·  双击画布气泡可直接改字", card)
        tips_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tips_lbl.setStyleSheet("color: #636366; font-size: 11px;")
        card_layout.addWidget(tips_lbl)

        layout.addWidget(card)
