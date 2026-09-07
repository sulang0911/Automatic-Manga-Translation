"""
app/ui/widgets/shortcuts_dialog.py
Apple / Linear style keyboard shortcuts cheat sheet modal.
"""
from typing import Optional, List, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor

from app.ui.theme.icons import get_icon


class ShortcutsDialog(QDialog):
    """
    Modal cheat sheet displaying all available keyboard shortcuts categorized clearly.
    """

    CATEGORIES = [
        ("🖥️ 视图与布局", [
            ("1 ~ 5", "切换视图模式 (译图 / 对比 / 双联 / 原图 / 抹字)"),
            ("Ctrl + B", "展开 / 折叠左侧页面列表"),
            ("Ctrl + + / -", "放大 / 缩小画布"),
            ("Ctrl + 0", "缩放适应窗口 (Fit in View)"),
            ("Ctrl + 1", "1:1 原图真实像素比例"),
        ]),
        ("📖 漫画阅读与翻页", [
            ("A / PageUp", "快速翻至上一页"),
            ("D / PageDown", "快速翻至下一页"),
            ("HUD < / >", "底部浮动控制条翻页"),
            ("滚轮越界", "滑动至页面边缘顺畅切页"),
        ]),
        ("✏️ 气泡校对与快速精修", [
            ("双击气泡", "就地弹出浮动修改卡片"),
            ("Tab / Shift+Tab", "在当前页面各气泡之间顺次跳切高亮"),
            ("Ctrl + Enter", "提交译文修改并直接跳转下一气泡"),
            ("Esc", "取消或关闭浮动编辑卡片"),
            ("R", "启用/退出手动框选新建气泡"),
            ("O", "启用/退出手动框选并即时 OCR 翻译"),
        ]),
        ("🔤 字体排版与历史控制", [
            ("[  /  ]", "微调减小 / 增大选中气泡字号 (即调即见)"),
            ("V", "一键切换选中气泡横排 ↔ 竖排排版"),
            ("Ctrl + Z", "撤销上一步操作 (Undo)"),
            ("Ctrl + Y", "重做撤销操作 (Redo)"),
            ("?  /  F1", "打开本快捷键速查面板"),
        ]),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("键盘快捷键速查 (Shortcuts)")
        self.setFixedWidth(680)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        self.setStyleSheet("""
            QDialog {
                background-color: #18181B;
                border: 1px solid #2A2A2E;
                border-radius: 12px;
            }
            QLabel {
                color: #ECECEF;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
            }
            .kbd-badge {
                font-family: "JetBrains Mono", "SF Mono", "Cascadia Code", monospace;
                font-size: 11px;
                font-weight: 600;
                color: #0A84FF;
                background-color: rgba(10, 132, 255, 0.12);
                border: 1px solid rgba(10, 132, 255, 0.25);
                border-radius: 4px;
                padding: 2px 7px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Header row
        header = QHBoxLayout()
        title_lbl = QLabel("⌨️ 键盘快捷键速查", self)
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 700; color: #FFFFFF;")
        header.addWidget(title_lbl)

        header.addStretch()

        close_btn = QPushButton("完成 (Esc)", self)
        close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #26262B;
                color: #ECECEF;
                border: 1px solid #3A3A40;
                border-radius: 6px;
                padding: 4px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #323238;
            }
        """)
        close_btn.clicked.connect(self.accept)
        header.addWidget(close_btn)

        layout.addLayout(header)

        # Grid of categories
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(16)

        for cat_idx, (cat_title, items) in enumerate(self.CATEGORIES):
            col = cat_idx % 2
            row = cat_idx // 2

            group_box = QFrame(self)
            group_box.setStyleSheet("""
                background-color: #202024;
                border: 1px solid #2A2A2E;
                border-radius: 8px;
                padding: 8px 12px;
            """)
            g_layout = QVBoxLayout(group_box)
            g_layout.setContentsMargins(10, 10, 10, 10)
            g_layout.setSpacing(8)

            cat_lbl = QLabel(cat_title, group_box)
            cat_lbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #FFFFFF; background: transparent;")
            g_layout.addWidget(cat_lbl)

            for key_str, desc in items:
                row_widget = QWidget(group_box)
                row_widget.setStyleSheet("background: transparent;")
                r_layout = QHBoxLayout(row_widget)
                r_layout.setContentsMargins(0, 2, 0, 2)
                r_layout.setSpacing(8)

                desc_lbl = QLabel(desc, row_widget)
                desc_lbl.setStyleSheet("color: #A1A1AA; font-size: 11px;")
                r_layout.addWidget(desc_lbl, 1)

                kbd_lbl = QLabel(key_str, row_widget)
                kbd_lbl.setProperty("class", "kbd-badge")
                kbd_lbl.setStyleSheet("""
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 11px;
                    font-weight: 600;
                    color: #0A84FF;
                    background-color: rgba(10, 132, 255, 0.12);
                    border: 1px solid rgba(10, 132, 255, 0.28);
                    border-radius: 4px;
                    padding: 2px 6px;
                """)
                r_layout.addWidget(kbd_lbl)

                g_layout.addWidget(row_widget)

            grid.addWidget(group_box, row, col)

        layout.addLayout(grid)
