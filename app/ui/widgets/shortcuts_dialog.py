"""
app/ui/widgets/shortcuts_dialog.py
Apple macOS-inspired keyboard shortcuts cheat sheet modal.
"""
from typing import Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor


class ShortcutsDialog(QDialog):
    """
    Modal cheat sheet displaying all available keyboard shortcuts categorized clearly.
    """

    CATEGORIES = [
        ("视图与布局", [
            ("1 ~ 5", "切换视图模式 (译图 / 对比 / 双联 / 原图 / 抹字)"),
            ("Ctrl + B", "展开 / 折叠左侧页面栏"),
            ("Ctrl + + / -", "放大 / 缩小画布"),
            ("Ctrl + 0", "缩放适应窗口 (Fit in View)"),
            ("Ctrl + 1", "1:1 原图真实像素比例"),
        ]),
        ("漫画阅读与翻页", [
            ("A / PageUp", "快速翻至上一页"),
            ("D / PageDown", "快速翻至下一页"),
            ("HUD < / >", "底部浮动控制条翻页"),
            ("滚轮越界", "滑动至页面边缘顺畅切页"),
        ]),
        ("气泡校对与快速精修", [
            ("双击气泡", "就地弹出浮动修改卡片"),
            ("Tab / Shift+Tab", "在当前页面各气泡之间顺次切换"),
            ("Ctrl + Enter", "提交译文修改并直接跳转下一气泡"),
            ("Esc", "取消或关闭浮动编辑卡片"),
            ("R", "启用/退出手动框选新建气泡"),
            ("O", "启用/退出手动框选并即时 OCR 翻译"),
        ]),
        ("字体排版与历史控制", [
            ("[  /  ]", "微调减小 / 增大选中气泡字号"),
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
                background-color: #1C1C1F;
                border: 1px solid #2C2C30;
                border-radius: 14px;
            }
            QLabel {
                color: #F5F5F7;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 22)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title_lbl = QLabel("键盘快捷键", self)
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 600; color: #F5F5F7;")
        header.addWidget(title_lbl)

        header.addStretch()

        close_btn = QPushButton("完成", self)
        close_btn.setProperty("class", "primaryBtn")
        close_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        close_btn.setFixedHeight(28)
        close_btn.clicked.connect(self.accept)
        header.addWidget(close_btn)

        layout.addLayout(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)

        for cat_idx, (cat_title, items) in enumerate(self.CATEGORIES):
            col = cat_idx % 2
            row = cat_idx // 2

            group_box = QFrame(self)
            group_box.setStyleSheet("""
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 10px;
                padding: 8px 12px;
            """)
            g_layout = QVBoxLayout(group_box)
            g_layout.setContentsMargins(10, 10, 10, 10)
            g_layout.setSpacing(7)

            cat_lbl = QLabel(cat_title, group_box)
            cat_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #0A84FF; background: transparent;")
            g_layout.addWidget(cat_lbl)

            for key_str, desc in items:
                row_widget = QWidget(group_box)
                row_widget.setStyleSheet("background: transparent;")
                r_layout = QHBoxLayout(row_widget)
                r_layout.setContentsMargins(0, 1, 0, 1)
                r_layout.setSpacing(8)

                desc_lbl = QLabel(desc, row_widget)
                desc_lbl.setStyleSheet("color: #98989D; font-size: 11px;")
                r_layout.addWidget(desc_lbl, 1)

                kbd_lbl = QLabel(key_str, row_widget)
                kbd_lbl.setProperty("class", "kbd-badge")
                kbd_lbl.setStyleSheet("""
                    font-size: 11px;
                    font-weight: 600;
                    color: #F5F5F7;
                    background-color: rgba(255, 255, 255, 0.09);
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 5px;
                    padding: 2px 7px;
                """)
                r_layout.addWidget(kbd_lbl)

                g_layout.addWidget(row_widget)

            grid.addWidget(group_box, row, col)

        layout.addLayout(grid)
