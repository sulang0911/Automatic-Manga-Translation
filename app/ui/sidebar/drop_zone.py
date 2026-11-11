"""
app/ui/sidebar/drop_zone.py
Apple macOS Finder-inspired drag-and-drop file and chapter folder import zone.
Automatically collapses into a slim 28px action strip once pages are loaded.
"""
from typing import Optional, List
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QWidget, QPushButton, QMenu
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

from app.ui.theme.icons import get_icon


class DropZoneWidget(QFrame):
    """
    Drag-and-Drop file and directory intake zone.
    Accepts single/batch manga images or full chapter directories.
    """
    sig_paths_dropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._is_drag_over = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon
        self.icon_label = QLabel(self)
        self.icon_label.setPixmap(get_icon("folder_open", color="#0A84FF", size=22).pixmap(22, 22))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        # Primary Text
        self.title_label = QLabel("拖入图片或漫画文件夹", self)
        self.title_label.setObjectName("dropZoneTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        # Subtitle
        self.sub_label = QLabel("支持 JPG / PNG / WebP / ZIP", self)
        self.sub_label.setObjectName("dropZoneSub")
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sub_label)

        # Quick Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)
        btn_layout.setContentsMargins(0, 4, 0, 0)

        self.btn_import_files = QPushButton("选择图片", self)
        self.btn_import_files.setIcon(get_icon("plus", color="#98989D", size=12))
        self.btn_import_files.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import_files.setFixedHeight(26)
        self.btn_import_files.clicked.connect(self._open_file_dialog)
        btn_layout.addWidget(self.btn_import_files)

        self.btn_import_folder = QPushButton("选择文件夹", self)
        self.btn_import_folder.setIcon(get_icon("folder_open", color="#0A84FF", size=12))
        self.btn_import_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_import_folder.setFixedHeight(26)
        self.btn_import_folder.clicked.connect(self._open_folder_dialog)
        btn_layout.addWidget(self.btn_import_folder)

        layout.addLayout(btn_layout)

    def set_compact(self, compact: bool):
        """Switches between expanded landing card and slim 28px bar when pages exist."""
        if compact:
            self.icon_label.hide()
            self.sub_label.hide()
            self.btn_import_files.hide()
            self.btn_import_folder.hide()
            self.title_label.setText("＋ 拖入或点击继续添加页面")
            self.title_label.setStyleSheet("font-size: 11px; font-weight: 500; color: #98989D;")
            self.layout().setContentsMargins(8, 5, 8, 5)
            self.layout().setSpacing(0)
            self.setMaximumHeight(30)
        else:
            self.setMaximumHeight(16777215)
            self.icon_label.show()
            self.sub_label.show()
            self.btn_import_files.show()
            self.btn_import_folder.show()
            self.title_label.setText("拖入图片或漫画文件夹")
            self.title_label.setStyleSheet("")
            self.icon_label.setPixmap(get_icon("folder_open", color="#0A84FF", size=22).pixmap(22, 22))
            self.layout().setContentsMargins(10, 12, 10, 12)
            self.layout().setSpacing(5)
            self.btn_import_files.setFixedHeight(26)
            self.btn_import_folder.setFixedHeight(26)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            menu = QMenu(self)
            action_files = menu.addAction(get_icon("plus", color="#0A84FF", size=14), "选择漫画图片文件...")
            action_folder = menu.addAction(get_icon("folder_open", color="#0A84FF", size=14), "选择漫画文件夹 (自动扫描页面)...")

            chosen = menu.exec(self.mapToGlobal(event.pos()))
            if chosen == action_files:
                self._open_file_dialog()
            elif chosen == action_folder:
                self._open_folder_dialog()
        super().mousePressEvent(event)

    def _open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择漫画图片",
            "",
            "漫画图片 (*.png *.jpg *.jpeg *.webp *.bmp);;所有文件 (*.*)"
        )
        if files:
            self.sig_paths_dropped.emit(files)

    def _open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择漫画文件夹 (自动递归扫描所有页面)",
            ""
        )
        if folder:
            self.sig_paths_dropped.emit([folder])

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            self._is_drag_over = True
            self.setProperty("dragOver", True)
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._is_drag_over = False
        self.setProperty("dragOver", False)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self._is_drag_over = False
        self.setProperty("dragOver", False)
        self.style().polish(self)

        urls = event.mimeData().urls()
        paths: List[str] = []
        for url in urls:
            if url.isLocalFile():
                paths.append(url.toLocalFile())

        if paths:
            self.sig_paths_dropped.emit(paths)
            event.acceptProposedAction()
