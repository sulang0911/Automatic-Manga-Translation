"""
app/ui/sidebar/page_list.py
Chapter page thumbnail list with natural alphanumeric sorting and status dot indicators.
"""
import os
import re
import uuid
from typing import List, Dict, Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QToolButton, QFrame, QMenu, QFileDialog, QPushButton
)
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtGui import QPixmap, QIcon, QColor, QPainter, QBrush, QDragEnterEvent, QDropEvent, QCursor

from app.ui.widgets.progress_pill import StatusDot
from app.ui.theme.icons import get_icon
from app.core.cache.cache_manager import get_cache_manager


def natural_sort_key(s: str) -> list:
    """Natural alphanumeric sort key, sorting 'page2.png' before 'page10.png'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', os.path.basename(s))]


def is_ignored_cache_or_export(file_path: str) -> bool:
    """Returns True if the path or file is an internal cache folder, hidden file, or intermediate export."""
    norm = os.path.normpath(os.path.abspath(file_path))
    parts = norm.split(os.sep)
    for part in parts[:-1]:
        if part.startswith(".") or part in ("translation_cache", "__pycache__", ".amt_cache"):
            return True
    fname = parts[-1].lower()
    if fname.startswith("."):
        return True
    ignored_suffixes = (
        ".erased.webp", ".erased.png",
        ".rendered.webp", ".rendered.png",
        "_erased.png", "_translated.png",
        ".blocks.json"
    )
    return any(fname.endswith(sfx) for sfx in ignored_suffixes)


class PageItemWidget(QWidget):
    """Custom item renderer for each manga page in the queue."""
    sig_remove = pyqtSignal(str)

    def __init__(self, item_data: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.item_data = item_data
        self.item_id = item_data["id"]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Status Dot
        self.status_dot = StatusDot(color="#71717A", size=8, parent=self)
        layout.addWidget(self.status_dot)

        # Thumbnail
        self.thumb_label = QLabel(self)
        self.thumb_label.setObjectName("thumbLabel")
        self.thumb_label.setFixedSize(36, 48)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._load_thumbnail(item_data["path"])
        layout.addWidget(self.thumb_label)

        # Text column: filename and status caption
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        self.name_label = QLabel(os.path.basename(item_data["path"]), self)
        self.name_label.setStyleSheet("font-weight: 500; font-size: 12px;")
        text_col.addWidget(self.name_label)

        self.status_label = QLabel(item_data.get("status_text", "等待中"), self)
        self.status_label.setStyleSheet("font-size: 11px; opacity: 0.65;")
        text_col.addWidget(self.status_label)

        layout.addLayout(text_col, 1)

        # Remove button
        self.remove_btn = QToolButton(self)
        self.remove_btn.setIcon(get_icon("trash", color="#71717A", size=14))
        self.remove_btn.setToolTip("从队列中移除")
        self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_btn.setStyleSheet("border: none; background: transparent; padding: 2px;")
        self.remove_btn.clicked.connect(lambda: self.sig_remove.emit(self.item_id))
        layout.addWidget(self.remove_btn)

    def _load_thumbnail(self, path: str):
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(
                    36, 48,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.thumb_label.setPixmap(scaled)

    def update_status(self, status: str, message: str = ""):
        if not message:
            status_map = {
                "done": "已完成",
                "completed": "已完成",
                "processing": "处理中",
                "failed": "失败",
                "error": "失败",
                "queued": "等待中",
                "pending": "等待中",
            }
            message = status_map.get(status.lower(), status)
        self.item_data["status"] = status
        self.item_data["status_text"] = message
        self.status_label.setText(message)

        dot_colors = {
            "queued": "#71717A",
            "pending": "#71717A",
            "processing": "#3B82F6",
            "completed": "#10B981",
            "done": "#10B981",
            "failed": "#EF4444",
            "error": "#EF4444",
        }
        self.status_dot.set_color(dot_colors.get(status.lower(), "#71717A"))


class PageListWidget(QWidget):
    """
    Queue and page list component supporting chapter import, natural alphanumeric sorting,
    and progress status tracking.
    """
    sig_page_selected = pyqtSignal(dict)
    sig_page_removed = pyqtSignal(str)
    sig_clear_requested = pyqtSignal()
    sig_start_batch = pyqtSignal()
    sig_translate_page = pyqtSignal(dict)
    sig_export_page = pyqtSignal(dict)
    sig_count_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.items_data: List[Dict[str, Any]] = []
        self._item_widgets: Dict[str, PageItemWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header Bar
        header = QWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 6, 8, 4)
        header_layout.setSpacing(6)

        title = QLabel("章节页面", header)
        title.setStyleSheet("font-weight: 600; font-size: 13px;")
        header_layout.addWidget(title)

        self.count_badge = QLabel("0 页", header)
        self.count_badge.setObjectName("countBadge")
        header_layout.addWidget(self.count_badge)
        header_layout.addStretch()

        # Add Files / Folder Button
        self.add_btn = QToolButton(header)
        self.add_btn.setIcon(get_icon("folder_open", color="#3B82F6", size=14))
        self.add_btn.setToolTip("添加图片文件或漫画文件夹")
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.setStyleSheet("border: none; background: transparent;")
        self.add_btn.clicked.connect(self._on_add_clicked)
        header_layout.addWidget(self.add_btn)

        self.clear_btn = QToolButton(header)
        self.clear_btn.setIcon(get_icon("trash", color="#A1A1AA", size=14))
        self.clear_btn.setToolTip("清空页面列表")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setStyleSheet("border: none; background: transparent;")
        self.clear_btn.clicked.connect(self.clear_all)
        header_layout.addWidget(self.clear_btn)

        layout.addWidget(header)

        # List Widget
        self.list_widget = QListWidget(self)
        self.list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.list_widget, 1)

        # Batch Translate Button
        self.batch_btn = QPushButton("🚀 批量翻译全部", self)
        self.batch_btn.setToolTip("批量翻译页面队列中的全部漫画 (Batch Translate All)")
        self.batch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.batch_btn.setEnabled(False)
        self.batch_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(59, 130, 246, 0.15);
                color: #3B82F6;
                border: 1px solid rgba(59, 130, 246, 0.35);
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3B82F6;
                color: #FFFFFF;
            }
            QPushButton:disabled {
                opacity: 0.35;
                color: #71717A;
                border-color: rgba(255, 255, 255, 0.08);
            }
        """)
        self.batch_btn.clicked.connect(self.sig_start_batch.emit)
        layout.addWidget(self.batch_btn)

    def add_paths(self, paths: List[str]):
        """
        Adds image paths or scans directories recursively, sorting them naturally
        and deduplicating against existing entries.
        """
        valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        discovered_files: List[str] = []

        for p in paths:
            if not p:
                continue
            if os.path.isfile(p):
                if is_ignored_cache_or_export(p):
                    continue
                ext = os.path.splitext(p)[1].lower()
                if ext in valid_extensions:
                    discovered_files.append(os.path.abspath(p))
            elif os.path.isdir(p):
                for root, dirs, files in os.walk(p):
                    # Prune hidden directories and internal cache directories in-place
                    dirs[:] = [
                        d for d in dirs
                        if not d.startswith(".") and d not in ("translation_cache", "__pycache__", ".amt_cache")
                    ]
                    for f in files:
                        full_f_path = os.path.join(root, f)
                        if is_ignored_cache_or_export(full_f_path):
                            continue
                        ext = os.path.splitext(f)[1].lower()
                        if ext in valid_extensions:
                            discovered_files.append(os.path.abspath(full_f_path))

        # Natural alphanumeric sorting
        discovered_files.sort(key=natural_sort_key)

        existing_paths = {item["path"] for item in self.items_data}
        new_items = []
        cache_mgr = get_cache_manager()
        for path in discovered_files:
            if path not in existing_paths:
                existing_paths.add(path)
                if cache_mgr.is_fully_translated(path):
                    status = "completed"
                    status_text = "已完成(缓存)"
                elif cache_mgr.has_cache(path)["erased"]:
                    status = "processing"
                    status_text = "已擦除(缓存)"
                else:
                    status = "queued"
                    status_text = "等待中"

                item_data = {
                    "id": str(uuid.uuid4()),
                    "path": path,
                    "status": status,
                    "status_text": status_text,
                }
                new_items.append(item_data)
                self.items_data.append(item_data)

        # Populate GUI items
        for data in new_items:
            list_item = QListWidgetItem(self.list_widget)
            list_item.setSizeHint(QSize(200, 56))
            widget = PageItemWidget(data, self.list_widget)
            widget.sig_remove.connect(self.remove_item)
            self.list_widget.addItem(list_item)
            self.list_widget.setItemWidget(list_item, widget)
            self._item_widgets[data["id"]] = widget

        self._update_count()

        # Select first item if newly added
        if self.list_widget.count() > 0 and self.list_widget.currentRow() < 0:
            self.list_widget.setCurrentRow(0)

    def remove_item(self, item_id: str):
        """Removes a page item by its unique ID."""
        for row in range(len(self.items_data)):
            if self.items_data[row]["id"] == item_id:
                self.items_data.pop(row)
                self.list_widget.takeItem(row)
                self._item_widgets.pop(item_id, None)
                self._update_count()
                self.sig_page_removed.emit(item_id)
                break

    def clear_all(self):
        """Clears all pages from queue."""
        self.items_data.clear()
        self._item_widgets.clear()
        self.list_widget.clear()
        self._update_count()
        self.sig_clear_requested.emit()

    def update_item_status(self, item_id: str, status: str, message: str = ""):
        """Updates status indicator and text for a specific page."""
        if not message:
            status_map = {
                "done": "已完成",
                "completed": "已完成",
                "processing": "处理中",
                "failed": "失败",
                "error": "失败",
                "queued": "等待中",
                "pending": "等待中",
            }
            message = status_map.get(status.lower(), status)
        widget = self._item_widgets.get(item_id)
        if widget:
            widget.update_status(status, message)
        else:
            for item in self.items_data:
                if item["id"] == item_id:
                    item["status"] = status
                    item["status_text"] = message
                    break

    def _update_count(self):
        count = len(self.items_data)
        self.count_badge.setText(f"{count} 页")
        if hasattr(self, "batch_btn"):
            self.batch_btn.setEnabled(count > 0)
        self.sig_count_changed.emit(count)

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        row = self.list_widget.row(item)
        if not (0 <= row < len(self.items_data)):
            return
        data = self.items_data[row]
        path = data.get("path", "")

        menu = QMenu(self)
        act_translate = menu.addAction(get_icon("play", color="#3B82F6", size=14), "翻译此页面")
        act_export = menu.addAction(get_icon("download", color="#10B981", size=14), "导出此页面...")
        menu.addSeparator()
        act_locate = menu.addAction(get_icon("folder_open", color="#A1A1AA", size=14), "在文件资源管理器中定位")
        menu.addSeparator()
        act_clear_cache = menu.addAction(get_icon("trash", color="#F59E0B", size=14), "清除本页缓存 (.amt_cache)")
        act_remove = menu.addAction(get_icon("trash", color="#EF4444", size=14), "从列表中移除")

        chosen = menu.exec(self.list_widget.mapToGlobal(pos))
        if chosen == act_translate:
            self.sig_translate_page.emit(data)
        elif chosen == act_export:
            self.sig_export_page.emit(data)
        elif chosen == act_locate:
            if path and os.path.exists(path):
                import subprocess
                subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')
        elif chosen == act_clear_cache:
            if path:
                get_cache_manager().clear_cache(path)
                self.update_item_status(data["id"], "queued", "等待中")
        elif chosen == act_remove:
            self.remove_item(data["id"])

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self.items_data):
            self.sig_page_selected.emit(self.items_data[row])

    def _on_add_clicked(self):
        """Displays menu to import images or full chapter folder."""
        menu = QMenu(self)
        act_files = menu.addAction(get_icon("eye", color="#3B82F6", size=14), "添加图片文件...")
        act_folder = menu.addAction(get_icon("folder_open", color="#3B82F6", size=14), "添加漫画文件夹 (自动扫描所有页面)...")
        chosen = menu.exec(QCursor.pos())
        if chosen == act_files:
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择漫画图片", "", "漫画图片 (*.png *.jpg *.jpeg *.webp *.bmp);;所有文件 (*.*)"
            )
            if files:
                self.add_paths(files)
        elif chosen == act_folder:
            folder = QFileDialog.getExistingDirectory(self, "选择漫画文件夹", "")
            if folder:
                self.add_paths([folder])

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        paths = [url.toLocalFile() for url in urls if url.isLocalFile()]
        if paths:
            self.add_paths(paths)
            event.acceptProposedAction()
