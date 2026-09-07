"""
app/ui/widgets/empty_state.py
Elegant empty-state canvas placeholder conforming to Apple HIG & Linear design.
Guides the user to import files, open folders, and understand keyboard shortcuts.
"""
from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QCursor

from app.ui.theme.icons import render_svg_pixmap, get_icon


class EmptyStateWidget(QWidget):
    """
    Centered welcome placeholder for the central canvas when no pages are loaded.
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

        # Centered Card
        card = QFrame(self)
        card.setObjectName("emptyStateCard")
        card.setStyleSheet("""
            #emptyStateCard {
                background-color: rgba(28, 28, 32, 0.75);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 40px 50px;
                max-width: 520px;
            }
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(14)

        # Brand Icon
        icon_lbl = QLabel(card)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = render_svg_pixmap("sparkles", color="#0A84FF", size=48)
        icon_lbl.setPixmap(pix)
        card_layout.addWidget(icon_lbl)

        # Title
        title_lbl = QLabel("开启漫画智能翻译工作区", card)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet("color: #ECECEF; font-size: 18px; font-weight: 700; letter-spacing: 0.5px;")
        card_layout.addWidget(title_lbl)

        # Subtitle
        sub_lbl = QLabel("直接拖入漫画图片、章节目录或 ZIP 压缩包\n或通过下方按钮载入文件开启智能翻译与精修", card)
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_lbl.setStyleSheet("color: #8E8E93; font-size: 13px; line-height: 1.5;")
        card_layout.addWidget(sub_lbl)

        card_layout.addSpacing(10)

        # Action Buttons Row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_folder = QPushButton("打开章节文件夹", card)
        self.btn_folder.setIcon(get_icon("folder_open", color="#FFFFFF", size=14))
        self.btn_folder.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_folder.setStyleSheet("""
            QPushButton {
                background-color: #0A84FF;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
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
        self.btn_files.setIcon(get_icon("plus", color="#ECECEF", size=13))
        self.btn_files.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.btn_files.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                color: #ECECEF;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                padding: 8px 16px;
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

        card_layout.addSpacing(10)

        # Quick tips row
        tips_lbl = QLabel("💡 提示：按 [ ? ] 键随时唤起键盘快捷键速查表 · 双击气泡即可就地改字", card)
        tips_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tips_lbl.setStyleSheet("color: #636366; font-size: 11px;")
        card_layout.addWidget(tips_lbl)

        layout.addWidget(card)
